# Mullion 0.4 on an M1 Pro: screenshots

Apple M1 Pro (Apple7), macOS 27.0.1, Wine 11.0. "Earlier build" is `b6f4faf` plus the M1 pull requests (15 Oct). "0.4" is the tag `7bcdcb4`, run with `--family native`.

| File | What it is |
|---|---|
| `bgfx-14-shadowvolumes-metal.png` | bgfx `14-shadowvolumes`, native Metal, frame 59 |
| `bgfx-14-shadowvolumes-earlier-build.png` | the same demo on D3D12, earlier build, frame 60 (0.01 % of the pixels differ from Metal) |
| `bgfx-14-shadowvolumes-0.4.png` | the same demo on D3D12, 0.4, frame 60 (20.83 % differ) |
| `godot-lights-and-shadows-metal.png` | Godot 4.7.2 `2d/lights_and_shadows`, native Metal, frame 60 |
| `godot-lights-and-shadows-earlier-build.png` | the same demo on D3D12, earlier build, frame 60 (0 % differ) |
| `godot-lights-and-shadows-0.4.png` | the same demo on D3D12, 0.4, frame 60 (76.29 % differ) |

bgfx's D3D12 renderer saves the back buffer it presented one frame before the one Metal saves, so its frame 60 is compared with Metal's frame 59.
