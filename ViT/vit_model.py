"""A from-scratch, ~250-line Vision Transformer (ViT) you can read in one sitting.

Paper: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
       Dosovitskiy et al., ICLR 2021            arXiv:2010.11929
       (referred to below as "the paper"; Eq. numbers follow the paper)

The whole forward pass is Eq. (1)-(4) + Eq. (9) of the paper:

    Eq.1   x_p = linear projection of each P x P patch        -> PatchEmbed
    Eq.2   A = softmax(q k^T / sqrt(d_h))                     -> MultiHeadSelfAttention
    Eq.3   MSA(x) = concat(head_1, ..., head_h) * W_O         -> MultiHeadSelfAttention
    Eq.4   MLP(x) = GELU(x W_1 + b_1) * W_2 + b_2             -> MLP
    Eq.9   z_L = LN(z_0) preceded by L pre-norm blocks        -> TransformerBlock / ViT

Deliberately omitted (see README "what is missing"): drop-path, multi-GPU/FSDP,
Flash-Attention, hybrid conv stem, and pretraining on JFT-300M.

Reference implementations worth reading afterwards:
    https://github.com/google-research/vision_transformer   (official, JAX + PyTorch)
    https://github.com/rwightman/pytorch-image-models       (timm, the de-facto standard)

Run this file directly to print the parameter table of the paper's models:
    python vit_model.py
"""

import torch
import torch.nn as nn
import torch.nn.init as init


# ======================================================================================
# 1. Patchify: (B, C, H, W) -> (B, N, D)
# ======================================================================================
def patchify(images, patch_size):
    """Split images into non-overlapping patches, no linear layer applied yet.

    (B, C, H, W) -> (B, N, P*P*C) with N = (H/P) * (W/P).
    This is the "im2col" version -- it is the mathematical definition of the split,
    written out explicitly so you can see that a ViT really just eats a sequence.
    `PatchEmbed` below does the exact same thing with one Conv2d.
    """
    B, C, H, W = images.shape
    assert H % patch_size == 0 and W % patch_size == 0, "image size must be divisible by patch size"
    # B, C, H//P, P, W//P, P  -- the two new axes are the patch grid and the patch content
    x = images.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)                # B, grid_h, grid_w, C, P, P
    return x.reshape(B, -1, C * patch_size * patch_size)  # B, N, P*P*C


class PatchEmbed(nn.Module):
    """Eq.1: flatten each patch + trainable linear projection E (P*P*C -> D).

    A non-overlapping, stride-P convolution IS exactly "flatten patch, matmul with E":
    output pixel (i,j) of the conv sees only the (i,j)-th patch, and the kernel is
    a (D, C, P, P) tensor == E^T reshaped.  So we use the conv: one fused kernel
    instead of unfold + matmul (same numbers).
    """

    def __init__(self, img_size=28, patch_size=4, in_chans=1, embed_dim=96):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)          # B, D, H//P, W//P
        x = x.flatten(2)          # B, D, N
        return x.transpose(1, 2)  # B, N, D


# ======================================================================================
# 2. Multi-head self-attention, Eq.(2) and Eq.(3)
# ======================================================================================
class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product attention, h heads in parallel.

    For every token we produce q, k, v of size d_h = D / h by three different linear maps,
    but implementations (including the official one) concatenate them into a single
    (D -> 3D) `qkv` matrix and split afterwards -- same math, one matmul.

        attn = softmax( q k^T / sqrt(d_h) )      Eq.2
        head = attn v                            Eq.2
        MSA  = concat(head_1..head_h) W_O        Eq.3

    Why / sqrt(d_h): q.k is a sum of d_h products; if q,k entries are ~N(0,1) the dot
    product has variance d_h, so softmax saturates and gradients die. Dividing by
    sqrt(d_h) keeps the logits at unit variance.
    """

    def __init__(self, dim, num_heads=4, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)          # W_O of Eq.3
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attn=False):
        B, N, D = x.shape

        # (B,N,D) -> (B,N,3,D*h) -> (3,B,h,N,D/h)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                  # each (B, h, N, d_h)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B,h,N,N)  Eq.2
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v)                          # (B,h,N,d_h)
        out = out.transpose(1, 2).reshape(B, N, D)    # concat heads -> (B,N,D)
        out = self.proj_drop(self.proj(out))          # W_O
        return (out, attn) if return_attn else out


# ======================================================================================
# 3. MLP (Eq.4) and the Transformer block
# ======================================================================================
class MLP(nn.Module):
    """Eq.4: Linear(D -> mlp_ratio*D) -> GELU -> Linear(mlp_ratio*D -> D).

    Note it is *per token*: tokens never mix here, all mixing happens in attention.
    The 4x widening is inherited from NLP Transformers and is where most of the
    parameters (and FLOPs) of a ViT live -- see the parameter breakdown in the README.
    """

    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class TransformerBlock(nn.Module):
    """One encoder layer. We use the *pre-norm* residual layout, which is what the
    paper trained (Fig.1 right) even though Fig.1 left draws post-norm:

        x = x + MSA(LN(x))
        x = x + MLP(LN(x))

    Post-norm (x = LN(x + f(x))) needs a warmup and diverges at depth ~12 without it;
    pre-norm keeps a clean residual path so gradients flow straight to the input.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop=drop)

    def forward(self, x, return_attn=False):
        if return_attn:
            attn_out, attn = self.attn(self.norm1(x), return_attn=True)
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
            return x, attn
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ======================================================================================
# 4. The whole model
# ======================================================================================
class ViT(nn.Module):
    """ViT = PatchEmbed -> [CLS] + learned position embeddings -> L x Block -> LN -> head.

    Args:
        img_size:    square input resolution (can be changed later, see interpolate_pos_embed)
        patch_size:  P. The paper's headline is P=16 with 224x224 -> 196 tokens.
        pool:        "cls" (paper) or "mean" (average the token outputs; often slightly
                     better when there is no large-scale pretraining).
    """

    def __init__(self, img_size=28, patch_size=4, in_chans=1, num_classes=10,
                 embed_dim=96, depth=4, num_heads=4, mlp_ratio=4.0,
                 drop_rate=0.0, attn_drop_rate=0.0, pool="cls", use_pos_embed=True):
        super().__init__()
        assert pool in ("cls", "mean")
        self.pool = pool
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.use_pos_embed = use_pos_embed  # ablate it: how much does position matter?

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        # The extra token the paper prepends: a learned vector that attends to all patches
        # and becomes the image representation (Eq.9 uses z_L^0). It carries no pixels.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Learned 1D position embeddings, one per token (including CLS).
        # The paper found 1D learned embeddings ~= 2D-aware ones, i.e. the model does
        # *not* need to be told the 2D grid -- spatial structure is re-learned by attention.
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop_rate, attn_drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        # Paper's init: trunc_normal(std=0.02) for the two learned "tokens"; the rest of
        # the weights follow ViT's default truncated normal too.
        init.trunc_normal_(self.cls_token, std=0.02)
        init.trunc_normal_(self.pos_embed, std=0.02)
        if not use_pos_embed:
            self.pos_embed.requires_grad_(False)  # only keep it alive for checkpoint compat

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_features(self, x, return_attn=False):
        """Everything up to (and including) the final LayerNorm: returns (B, 1+N, D)."""
        x = self.patch_embed(x)                       # B, N, D
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)  # B, 1+N, D
        if self.use_pos_embed:
            x = x + self.pos_embed                 # position info injected by addition
        x = self.pos_drop(x)

        attns = []
        for blk in self.blocks:
            if return_attn:
                x, a = blk(x, return_attn=True)
                attns.append(a)
            else:
                x = blk(x)
        x = self.norm(x)
        return (x, attns) if return_attn else x

    def forward(self, x, return_attn=False):
        feats, attns = self.forward_features(x, return_attn=True) if return_attn \
            else (self.forward_features(x), None)
        out = feats[:, 0] if self.pool == "cls" else feats.mean(dim=1)
        logits = self.head(out)
        if return_attn:
            return logits, attns
        return logits


# ======================================================================================
# 5. Utilities that make the paper's claims checkable
# ======================================================================================
def interpolate_pos_embed(model, new_img_size, mode="bicubic"):
    """Resize the learned position embeddings to a new input resolution in place.

    The paper fine-tunes a 224px model at 384/512px (Sec. 3.2): the patch grid grows
    (224/16 -> 14x14, 384/16 -> 24x24), so the pos-embed "image" is resampled as a
    bilinear/bicubic 2D grid while keeping the CLS row untouched.
    """
    old_h, old_w = model.patch_embed.grid_size
    patch_size = model.patch_embed.patch_size
    if new_img_size % patch_size != 0:
        raise ValueError(f"new_img_size ({new_img_size}) must be divisible by patch_size ({patch_size})")
    new_grid = new_img_size // patch_size
    pos = model.pos_embed.detach()  # (1, 1+H*W, D)
    cls_pos, grid_pos = pos[:, :1], pos[:, 1:]
    grid_pos = grid_pos.reshape(1, old_h, old_w, -1).permute(0, 3, 1, 2)      # 1,D,h,w
    grid_pos = nn.functional.interpolate(grid_pos, size=(new_grid, new_grid),
                                         mode=mode, align_corners=False)
    grid_pos = grid_pos.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, -1)
    model.pos_embed = nn.Parameter(torch.cat([cls_pos, grid_pos], dim=1), requires_grad=model.pos_embed.requires_grad)
    model.patch_embed.img_size = (new_img_size, new_img_size)
    model.patch_embed.grid_size = (new_grid, new_grid)
    model.patch_embed.num_patches = new_grid * new_grid
    model.num_patches = new_grid * new_grid
    return model


def count_parameters(model, trainable_only=True):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def parameter_breakdown(model):
    """Parameters per component -- shows that the MLP holds ~2/3 of a ViT's weights."""
    groups = {}
    for name, p in model.named_parameters():
        key = name.split(".")[0]
        if key == "blocks" or key.startswith("blocks"):
            key = "transformer blocks"
        groups[key] = groups.get(key, 0) + p.numel()
    return dict(sorted(groups.items(), key=lambda kv: -kv[1]))


# ======================================================================================
# 6. The four sizes from the paper's Table 1, so you can reproduce their numbers
# ======================================================================================
def vit_tiny_patch16_224(num_classes=1000, **kw):
    return ViT(img_size=224, patch_size=16, in_chans=3, num_classes=num_classes,
               embed_dim=192, depth=12, num_heads=3, **kw)


def vit_base_patch16_224(num_classes=1000, **kw):
    """ViT-B/16: 86M params -- matches Table 1 of the paper."""
    return ViT(img_size=224, patch_size=16, in_chans=3, num_classes=num_classes,
               embed_dim=768, depth=12, num_heads=12, **kw)


def vit_large_patch16_224(num_classes=1000, **kw):
    """ViT-L/16: 307M params."""
    return ViT(img_size=224, patch_size=16, in_chans=3, num_classes=num_classes,
               embed_dim=1024, depth=24, num_heads=16, **kw)


def vit_huge_patch14_224(num_classes=1000, **kw):
    """ViT-H/14: 632M params."""
    return ViT(img_size=224, patch_size=14, in_chans=3, num_classes=num_classes,
               embed_dim=1280, depth=32, num_heads=16, **kw)


if __name__ == "__main__":
    print(f"{'model':<24}{'params':>10}  {'tokens':>7}")
    for name, factory in [("vit_tiny_patch16_224", vit_tiny_patch16_224),
                          ("vit_base_patch16_224", vit_base_patch16_224),
                          ("vit_large_patch16_224", vit_large_patch16_224),
                          ("vit_huge_patch14_224", vit_huge_patch14_224)]:
        m = factory(num_classes=1000)
        print(f"{name:<24}{count_parameters(m)/1e6:>9.1f}M  {m.num_patches:>7}")
