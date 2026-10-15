# d3d12_tessellator on M1: repro, dumps and pure-Metal replays

Material for the issue "d3d12_tessellator (layers): triangles go missing on M1 after a larger draw". Measured on an M1 Pro (Apple7), macOS 27.0.1, Mullion `b6f4faf`.

## Patches

They were made on `b6f4faf` with our M1 fixes applied (the changes of PRs #7-#25). `exp-tess-order` and `exp-metal-log` also apply on plain `b6f4faf`, while `exp-mesh-dump` needs the fixes under it. Without PR #7, no mesh pipeline builds on Apple7.

| Patch | Switches | What it does |
|---|---|---|
| `exp-tess-order.patch` | `MULLION_PATCH_ORDER=0,13`, `MULLION_ONLY_CASE`, `MULLION_LAYERS_ONLY`, `MULLION_DIRECT_ONLY`, `MULLION_SLEEP_MS`, `MULLION_CLEAR_APART`, `MULLION_COPY_APART`, `MULLION_DUMP` | Test only: draws the layered patches in a chosen order |
| `exp-mesh-dump.patch` | `EXP_MESH_DUMP=<unix dir>` | The tessellation GS records every emitted vertex (all output registers) in a buffer at mesh index 10; `dump-<n>.bin` per draw |
| `exp-metal-log.patch` | `EXP_METAL_LOG=1` | `winemetal_unix.c` prints every commit, encoder and render, compute and blit command to stderr |

Repro (about 20 s):

```
MULLION_LAYERS_ONLY=1 MULLION_ONLY_CASE="tri, integer, layers" MULLION_DIRECT_ONLY=1 MULLION_PATCH_ORDER=0,13 <run d3d12_tessellator dxil>
```

Expected: `failed: 1272 wrong`. With `MULLION_PATCH_ORDER=13`, `13,13` or `2,13` the test passes.

## Data

- `dumps/`: `exp-mesh-dump` output, read with `tools/mesh_dump_diff.py A B`. It holds patch 13 alone, patch 0 drawn first, and patch 13 after patch 0. Patch 13's two dumps are identical: 603 records of 16 words each.
- `air/`: Mullion's compiled functions for "tri, integer, layers" (`4_vshs_main`, `3_ds_main` and the pixel shader), saved from the failing run.
- `metal-log/`: `exp-metal-log` output for orders `13,2` and `0,13,2`. Patch 13's commands are identical in both; only addresses differ.
- `results/`: every run's numbers, as JSON.

## Pure-Metal replays (macOS, no Wine)

```
swiftc -O tools/metal_mullion_replay.swift -o replay
./replay air/4_vshs_main.metallib air/3_ds_main.metallib air/0_ps_main6c402e5c.metallib 64,64,64,64,64,33 2,16.5,2,2,12,33 mullionenv
```

`metal_mullion_replay.swift` runs Mullion's own three functions with the same factors, root signature, draw arguments and grid. It draws B alone, then A, then B again. Modes:
- `private`: the render target in private memory;
- `mullionenv`: a 1 GiB untracked arena at Mullion's offsets, fences, a residency set and explicit states;
- `nocull`, `same`.

`metal_mesh_replay.swift` replays the dumped triangles through a hand-written mesh shader. `metal_mesh_tess_shape.swift` is a synthetic mesh with the same shape.

None of the replays loses a pixel on the M1 Pro, natively or as x86_64 under Rosetta.
