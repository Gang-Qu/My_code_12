# Official Depth Anything V2 metric model dependency

Source: https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth/depth_anything_v2

This small source package and its original LICENSE were copied unchanged from the official repository archive already present in the user's workspace. The original archive did not retain a Git commit identifier; per-file SHA256 is recorded in each DA run's `global_prior_metadata.json`.

Weights: `depth-anything/Depth-Anything-V2-Metric-Hypersim-Small`, Hugging Face revision `3bc65d4e14a6786a61acec16453c50e12bf5f338`.

Official file page: https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/blob/main/depth_anything_v2_metric_hypersim_vits.pth

Verified official SHA256: `b782898d8a3e8be1f639de33837ed85e9b4b73e40f8f5e5cd99067588d722545`.

The server could not connect directly to Hugging Face, so the identical checkpoint was fetched through hf-mirror.com and verified against the official hash before loading. Weights are not included in this source tree and were not downloaded to the local Mac.


## My_code_10 local implementation change (2026-11-20)

`depth_anything_v2/dinov2_layers/attention.py` now has an optional PyTorch native
scaled-dot-product-attention path, selected by `model.configure_attention`.
It preserves all parameter names, shapes, original license headers, scaling and
dropout semantics. `attention_backend='original'` retains the previous path.
The upstream-copy statement above describes the initial import; this file is
now locally modified. No pretrained weights were downloaded for this change.

My_code_12 keeps the My_code_10 attention implementation. The optional
coord_tokens keyword now passes a zero-initialized external pixel-coordinate
projection through dpt.py into the patch tokens in dinov2.py, before the
transformer blocks. Official DA state_dict keys and tensor shapes are preserved.

For relative checkpoints, dpt.py selects the upstream nonnegative ReLU output and omits the metric depth multiplier. The trainable conversion to metres is separate from the vendored DA network, in model.py.
