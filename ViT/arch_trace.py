"""Walk a ViT forward pass and print the shape of every tensor, plus where the
parameters and FLOPs go.  This is the "see the architecture" script -- no training.

    python arch_trace.py                 # ViT-B/16 at 224x224 (the paper's setting)
    python arch_trace.py --tiny          # the 0.46M toy model used in the notebook (4x28x28)
    python arch_trace.py --img 384       # fine-tuning at 384px: watch the token count grow
"""

import argparse
import torch

from vit_model import (ViT, count_parameters, parameter_breakdown, vit_base_patch16_224,
                       vit_huge_patch14_224, vit_large_patch16_224, vit_tiny_patch16_224)


# ======================================================================================
# helpers
# ======================================================================================
def gmacs(vit):
    """Analytic multiply-accumulate count for one image.  Per block:
         qkv   N*D*3D          proj  N*D*D          mlp  N*D*4D + N*4D*D
         attn  q k^T: N*N*D,   attn v: N*N*D      (summed over heads => D, not d_h)
    """
    N = vit.num_patches + 1
    D = vit.embed_dim
    per_block = 3 * N * D * D + 2 * N * N * D + N * D * D + 8 * N * D * D
    patch_embed = N * (vit.patch_embed.patch_size ** 2) * _in_chans(vit) * D
    head = D * vit.head.out_features  # classification reads only the pooled CLS/mean vector
    return vit.depth * per_block + patch_embed + head


def _in_chans(vit):
    return vit.patch_embed.proj.in_channels


def print_pipeline(model, x):
    print("=" * 96)
    print(f"{model.__class__.__name__}: input {tuple(x.shape)}  |  "
          f"{model.num_patches} patches + 1 CLS = {model.num_patches + 1} tokens  |  "
          f"D = {model.embed_dim}")
    print("=" * 96)
    print(f"{'tensor':<44}{'shape':<26}")
    print("-" * 96)
    with torch.no_grad():
        print(f"{'input image':<44}{str(tuple(x.shape)):<26}")

        t = model.patch_embed(x)
        print(f"{'patch_embed: conv PxP/stride P':<44}{str(tuple(t.shape)):<26}  "
              f"# (B, N, D) -- the image is now a sentence")
        cls = model.cls_token.expand(t.shape[0], -1, -1)
        print(f"{'cls_token (learned vector)':<44}{str(tuple(cls.shape)):<26}")
        t = torch.cat([cls, t], dim=1)
        print(f"{'concat [CLS; patches]':<44}{str(tuple(t.shape)):<26}")
        if model.use_pos_embed:
            t = t + model.pos_embed
            print(f"{'+ pos_embed (learned, 1 per token)':<44}{str(tuple(t.shape)):<26}")
        else:
            print("position embedding disabled")

        # ---- inside one block (all blocks have identical shapes) ----
        print("-" * 96)
        print("inside blocks[0]  (shown once; each block has the same tensor shapes)")
        print("-" * 96)
        if not model.blocks:
            raise ValueError("architecture trace needs at least one Transformer block")
        blk = model.blocks[0]
        h = blk.norm1(t)
        print(f"{'  LayerNorm1':<44}{str(tuple(h.shape)):<26}")
        qkv = blk.attn.qkv(h)
        print(f"{'  qkv = Linear(D -> 3D)':<44}{str(tuple(qkv.shape)):<26}")
        q, k, v = qkv.reshape(h.shape[0], h.shape[1], 3, blk.attn.num_heads,
                              blk.attn.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        print(f"{'  split into q, k, v  (B, h, N, d_h)':<44}{str(tuple(q.shape)):<26}")
        attn = (q @ k.transpose(-2, -1)) * blk.attn.scale
        print(f"{'  attn = q k^T / sqrt(d_h)':<44}{str(tuple(attn.shape)):<26}  "
              f"# every token x every token")
        attn = attn.softmax(-1)
        print(f"{'  softmax over the last axis':<44}{str(tuple(attn.shape)):<26}  "
              f"# each row sums to 1")
        o = (attn @ v).transpose(1, 2).reshape(h.shape)
        print(f"{'  @ v, concat heads':<44}{str(tuple(o.shape)):<26}")
        o = blk.attn.proj(o)
        print(f"{'  W_O (Linear D -> D)':<44}{str(tuple(o.shape)):<26}")
        t = t + o
        print(f"{'  + residual':<44}{str(tuple(t.shape)):<26}")
        h = blk.norm2(t)
        h = blk.mlp.fc1(h)
        print(f"{'  MLP fc1 (D -> 4D) + GELU':<44}{str(tuple(h.shape)):<26}")
        h = blk.mlp.fc2(h)
        print(f"{'  MLP fc2 (4D -> D)':<44}{str(tuple(h.shape)):<26}")
        t = t + h
        print(f"{'  + residual':<44}{str(tuple(t.shape)):<26}")

        # The first block was expanded above for visibility; start at block 1 here.
        for blk in model.blocks[1:]:
            t = blk(t)
        print("-" * 96)
        print(f"{'after all blocks':<44}{str(tuple(t.shape)):<26}")
        t = model.norm(t)
        print(f"{'final LayerNorm':<44}{str(tuple(t.shape)):<26}")
        pooled = t[:, 0] if model.pool == "cls" else t.mean(dim=1)
        print(f"{f'pool ({model.pool})':<44}{str(tuple(pooled.shape)):<26}  "
              f"# 197 token vectors -> 1 image vector")
        logits = model.head(pooled)
        print(f"{'head (Linear D -> num_classes)':<44}{str(tuple(logits.shape)):<26}")
    print("=" * 96)


def print_params(model, name=""):
    total = count_parameters(model)
    print(f"\nparameter breakdown  ({total/1e6:.1f}M total)")
    print("-" * 60)
    for k, v in parameter_breakdown(model).items():
        print(f"  {k:<24}{v:>12,}  {100*v/total:5.1f}%")
    blk = model.blocks[0]
    attn_p = count_parameters(blk.attn)
    mlp_p = count_parameters(blk.mlp)
    print(f"\n  one block = {count_parameters(blk):,}  "
          f"(attention {100*attn_p/count_parameters(blk):.0f}% / MLP {100*mlp_p/count_parameters(blk):.0f}%)")
    n = model.num_patches + 1
    print(f"  tokens = {n}, D = {model.embed_dim}, depth = {model.depth}\n")


def print_compute(model, label):
    macs = gmacs(model)
    print(f"  {label:<20}{count_parameters(model)/1e6:>7.1f}M params"
          f"{model.num_patches:>7} patches{macs/1e9:>10.1f} GMAC  ~{2*macs/1e9:>8.1f} GFLOPs")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tiny", action="store_true", help="4x28x28 toy model instead of ViT-B/16")
    p.add_argument("--img", type=int, default=None, help="input resolution, e.g. 384")
    args = p.parse_args()

    if args.tiny:
        model = ViT(img_size=28, patch_size=4, in_chans=1, num_classes=10)
        x = torch.randn(1, 1, 28, 28)
        label = "tiny ViT (28x28, P=4)"
    else:
        img = args.img or 224
        model = ViT(img_size=img, patch_size=16, in_chans=3, num_classes=1000,
                    embed_dim=768, depth=12, num_heads=12)
        x = torch.randn(1, 3, img, img)
        label = f"ViT-B/16 @ {img}px"

    print(f"\n>>> {label}\n")
    print_pipeline(model, x)
    print_params(model)
    print("the model just traced:")
    print_compute(model, label)
    print("\nthe four sizes from the paper's Table 1 (all at 224px):")
    print_compute(vit_tiny_patch16_224(num_classes=1000), "ViT-Ti/16")
    print_compute(vit_base_patch16_224(num_classes=1000), "ViT-B/16")
    print_compute(vit_large_patch16_224(num_classes=1000), "ViT-L/16")
    print_compute(vit_huge_patch14_224(num_classes=1000), "ViT-H/14")
    print()
