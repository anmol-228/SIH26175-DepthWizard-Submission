# DW-S1-08 Day-3 Viewer: FPS / Load Measurement Report

**Author:** Aishee Mukherjee (aisheeem7)
**Scope:** PR #36 (fly-camera bounds, mesh/LOD handling, query-param loading, unit/branch banner)
**Related:** Issue #8 (DW-S1-05, Sprint-1 FPS/load baseline, closed) established the original native/LOD numbers on the pre-PR-#29 stress viewer (`viewer.html`). This report covers the **new offline `viewer/` scaffold + Day-3 additions**, on the placeholder mesh, on this contributor's own machine — a distinct measurement, not a replacement for #8's baseline.

## Machine / Environment

| | |
|---|---|
| CPU | Intel(R) Core(TM) i3-1005G1 @ 1.20GHz |
| GPU | Intel(R) UHD Graphics (integrated, no dedicated GPU) |
| OS | Windows 10 Home Single Language, x64 |
| Browser | Chrome 152.0.7977.76 (64-bit, Stable) |
| Server | `python -m http.server 8793`, served locally from `viewer/` |

## Mesh Tested

| | |
|---|---|
| File | `placeholder.glb` (synthetic heightfield, real `heightfield_to_mesh()` output) |
| Triangles | 7,938 |
| Note | Only mesh variant available locally — `K3_mesh_pipeline.py` (native/medium/small/LOD `.glb` generation) hasn't been run on this machine, so native-vs-LOD comparison from issue #8 could not be repeated here. This report covers only the placeholder tier. |

## Method

Real-time FPS counter added to the viewer HUD (rolling 500ms window, driven by `requestAnimationFrame` timestamps — not estimated). Measured during active navigation across multiple recorded test sessions in this conversation, not idle. Load time read directly from the HUD's own load-time measurement (`performance.now()` delta around the GLTFLoader call).

## Results

| Scenario | FPS observed | Notes |
|---|---|---|
| Orbit mode, idle/light drag | 59.2 – 60.5 | Multiple sessions |
| Orbit mode, aggressive scroll-zoom-in (to `minDistance` bound) | 59.7 – 60.3 | ~58s continuous test, no drops even at closest zoom |
| Fly mode, WASD movement | 59.5 – 60.8 | Multiple sessions, including sustained aggressive movement |
| Fly mode, repeated close-approach to steep ridge geometry | 59.7 – 60.1 | ~45s continuous stress test |

**FPS held steady at ~60fps (the likely vsync/display cap) across every navigation mode and stress scenario tested, with no observed drops, stutters, or degradation over sustained sessions (45–58+ seconds continuous).**

**Load time:** ranged 25.0ms – 741.0ms across different test sessions. This range reflects real variation (browser cache state, whether other tabs/processes were active, OS-level scheduling on a shared low-spec machine) rather than a single reliable number — reported as a range rather than picking one favorable figure, per this team's evidence-discipline standard.

## Navigation responsiveness

No perceptible input lag in either mode across all sessions. Orbit drag/scroll and fly-mode WASD + pointer-look both felt immediate. No visible rendering glitches during navigation (see PR #36 for the two real bugs found and fixed during this same testing process: fly-camera bounds and the terrain-height-clamp axis bug).

## Console / errors

Zero console errors observed across all test sessions on this mesh/scaffold combination.

## Claim boundaries

- This is a **capability + performance measurement**, not a claim about how the *real* pipeline output (once DW-S1-07/#24 lands) will perform — the placeholder is a small (7,938-triangle) synthetic mesh; real native-resolution meshes are far larger (per issue #8: ~10.6M faces native) and were not available to test here.
- No native-vs-LOD comparison was possible in this report (see Mesh Tested note above) — that comparison remains issue #8's, on the old stress viewer; it has not been repeated on this new scaffold.
- ~60fps here reflects this specific low-spec, integrated-graphics, 7,938-triangle scenario only.

## Evidence

Real-time HUD FPS counter, `requestAnimationFrame`-timed, screen-recorded across multiple sessions during this development/testing process (see PR #36 commit history and review thread for the underlying recordings referenced during bug-fixing).
