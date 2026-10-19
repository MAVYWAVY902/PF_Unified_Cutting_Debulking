# Trajectories

Tool-pose recordings (`pfsm-trajectory v1`) for `tool_bench --replay` and the viewer `--replay`.
Scripted ones (`script_*`) were written by `tool_bench --write-traj` and are stored as files
because regenerating them is not bit-identical.

| file | config | tool | purpose |
|---|---|---|---|
| `try_mouse.csv` | cube_test | sphere | first real mouse recording (0.5b) |
| `script_cube_sweep.csv` | cube_test | sphere | scripted half-embedded sweep |
| `script_tumor_sweep.csv` | tumor_debulk | sphere | scripted sweep on the tumour |
| `script_dissector_tipdown.csv` | tumor_debulk_dissector | dissector, tip down | scripted mesh-tool sweep; made with the old touch_at but checked (`--pass-through`): nothing inside / through the tool at switch-on, first contact at frame 65 vs 64 expected (< one 0.05 mm approach step) |
| `script_tumorcut_scalpel_surfacetouch.csv` | tumor_cut | scalpel | **cutting benchmark**: blade switches on outside the tissue (touch = exact tool-mesh vs tissue-surface contact), edge-first descent 2 mm, then sweep along x |
| `script_tumorcut_scalpel.csv` | tumor_cut | scalpel | **stress test, kept on purpose**: written before the touch_at fix — the blade switches on ~2 mm inside the tissue (104 cluster edges already through the blade at switch-on) |

Both scalpel trajectories sweep along the blade's edge direction (x) after the descent; the blunt
front face then plows uncut tissue (≈177 intact edges through the blade per frame on the
surfacetouch trajectory, `tool_bench --pass-through`). That is a simulator limitation, not a
trajectory error.

Generate the surfacetouch one with
`tool_bench ../../config/tumor_cut.yaml OUT --tool mesh --depth 0.002 --write-traj F` (run from `build/`).
