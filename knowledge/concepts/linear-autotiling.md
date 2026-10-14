# Generated Linear auto-tiling

RhinoForge Linear execution uses one generated shape planner across supported
FP16, W8A16, and packed W4A16 paths. The caller supplies the checked tensor and
quantization contract; the shared implementation resolves a valid tile for the
actual `M`, `K`, `N`, dtype, partition, and available execution envelope.

## Contract

- The public path has no auto-tiling enable switch and no model-specific tile
  override.
- `M = 1` follows the same generated planner instead of a separate public GEMV
  route.
- W8A16 requires the matching signed-weight and scale metadata expected by its
  checked checkpoint profile.
- W4A16 uses the packed group representation admitted by the runtime. For
  input width `K` and group size 32, a one-dimensional per-output-channel
  scale is repeated across `K / 32` groups before the controller-striped
  layout is produced; callers must not invent another packing convention.
- Failure to find a valid tile is a profile-admission failure, not permission
  to reuse a tile from a nearby model.

## Porting consequence

Keep quantization and packing in the existing converter/helpers, then call the
ordinary Linear path. Remove retired per-shape tile and auto-tiling environment
values from configs. Validate the exact projection geometry, dequantized
reference, same-semantic output, and model task gate.

## Shared MLP and weight reuse

The standard eight-core FP16 causal decoder can fuse SiLU with the following
multiply when it uses sequential traversal, batch one, a compatible local MLP
width and ACC16. Selection happens during profile installation; it has no
separate environment switch. Other activations, quantization, accumulation,
core layouts and composite owners keep their admitted execution paths.

The optional W8A16 row-weight reuse path is narrower: row-partitioned eight-core
Linear with `M=512`, `N=1024`, `K=2048`, ACC16, no bias and disjoint SPM input and
output. The matching expansion asset must contain the required operator.
Capability and geometry are checked before selecting the route. Other shapes
or absent capability use ordinary auto-tiling; a selected route with invalid
operands is rejected. Shape equality alone does not let another model adopt a
model-owned route.

These selections become part of the physical plan and Graph identity. They do
not extend a weight or activation lifetime across chunks, denoise steps or
component boundaries. Keep existing owner and allocation checks when adopting
shared operations.

## Sources

- [Quantization](../../docs/quantization.md)
- [Runtime configuration](../../docs/runtime_config.md)
- [Linear implementation](../../src/ops/rpu_linear.cpp)
- [Linear planner](../../src/ops/rpu_linear_tiling.h)
- [Packed INT4 helper](../../python/rpu_backend/quant/int4_pgrp_pack.py)
- [Weight swizzle](weight-swizzle.md)
