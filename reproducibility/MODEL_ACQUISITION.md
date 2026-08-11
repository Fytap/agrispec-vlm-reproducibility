# DINOv2 model acquisition and verification

The modern representation control uses the public `timm` identifier `vit_small_patch14_dinov2.lvd142m`.

- Architecture: ViT-S/14.
- Pretraining: DINOv2 self-supervised LVD-142M.
- Input size: 224 x 224 RGB.
- `timm` version in recorded runs: 1.0.28.
- Hugging Face identifier: `timm/vit_small_patch14_dinov2.lvd142m`.
- Upstream weight URL: `https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth`.
- Loaded state-dictionary SHA-256: `8eb9c9b3a65e7889cf9744bf093f8980d0a91e9c0697ba3a5ae048798617fbf9`.

Frozen unmasked inputs are tight candidate bounding boxes without padding. The component-masked control retains proposal-component RGB and replaces outside-component pixels with the per-image RGB median before resize. Fine-tuning unfreezes only the final transformer block, final normalization, and binary head.

The archived T070_A032 unmasked embedding SHA-256 is `e470069e816644a3eefb86deb5b4e22082d042d96de3314ce8f1f4d31e9873d0`; the component-masked embedding SHA-256 is `892c77fae76f8e763308f5af9be42a5142bbb7fcb63278b8b30b90e9295f2957`.

The extraction scripts download through `timm` when the weights are not cached. They do not read target truth. Check upstream licensing and redistribution conditions before publishing weights; this package does not redistribute them.
