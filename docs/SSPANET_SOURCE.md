# SSPANet source provenance

- Author's project page: https://hellojahid.github.io/paper/sspanet/sspanet.html
- The project's Code link resolves to https://github.com/HelloJahid/SSPANet
- Pinned commit: `ed15ffbeabba15d3f3f3caba53bea6c34537c810`
- Original file: https://github.com/HelloJahid/SSPANet/blob/ed15ffbeabba15d3f3f3caba53bea6c34537c810/sspanet_demo.py
- Vendored verbatim at `training/detectors/modules/sspanet.py`.
- SHA256 of checked-out bytes: `772019425071767dae5a535f2dda4ab38e10ebd7c588ee55b289d6f5ef831a5a`.
- Upstream header identifies Md Jahid Hasan as code owner and states MIT License; the header is retained. Upstream has no separate LICENSE file at this commit.
- Retrieved 2026-09-12. The file calls itself DEMO CODE. This is the author's public implementation, not a verified reproduction of the full MRI training experiment.

## Code is authoritative for this integration

The published prose and equations are not fully consistent with the public code. We preserve the CODE, as requested:

1. `ZPool` reduces over channels to two spatial maps (max, mean). Despite its class name `ChannelAttention`, it creates a spatial gate using a 7x7 convolution and BN.
2. `SpatialAttention` computes `sqrt(mean(x**2) + epsilon)` along each spatial axis. This is RMS, not centered standard deviation. Negative CLIP features are valid because they are squared before square root.
3. The strip convolutions are dense 3x1 and 1x3 convolutions followed by BN; then a dense 1x1 convolution and sigmoid.
4. `ATTN_Block` is parallel: `x + x * sigmoid(channel(x) + spatial(x))`. It does not implement the sequential channel/spatial equation in the PDF.
5. The residual multiplier is in [1,2]; this block does not directly suppress absolute feature magnitudes. No claim that its gate is a calibrated forgery probability is justified.

At C=1024, ATTN_Block has 7,345,252 trainable parameters. Dense convolutions mean that calling this a negligible-cost adapter is misleading. It retains the original BatchNorm running statistics, eps and momentum settings. Do not silently replace BN with LN or insert a channel bottleneck and still claim exact reproduction.

## Outside the original block

The detector reshapes CLIP patch tokens, invokes ATTN_Block at 1024 channels, averages refined patches, normalizes CLS and pooled patch vectors separately, and fuses them with a learned scalar initialized to 0.1. A shared 1x1 patch classifier produces 256 logits, and the mean of the top 16 logits receives BCE supervision. The fused 1024-vector is L2-normalized and receives two-class CE. These integration choices are OUR ADAPTATION, not part of SSPANet or the cited contrastive MIL paper.

Only LayerNorm gamma/beta inside CLIP are trainable. All SSPANet parameters, its BN affine parameters, fusion scalar, classification head and MIL head are trainable. The backbone's other biases are frozen. Default CLIP-L/14 gives about 7.45M trainable parameters altogether; the exact count is recorded at runtime.
