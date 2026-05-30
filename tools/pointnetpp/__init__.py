"""PointNet++-inspired hierarchical parameter representation for diff-PBD calibration.

Phase 0-1 POC. Uses PointNet++ architectural ideas (FPS + multi-scale grouping +
distance-weighted feature propagation) WITHOUT any trained neural network — pure
geometric algorithm as inductive bias on stiff_pp / tau_pp fields.

Modules:
    fps       — pure-PyTorch Farthest Point Sampling (no torch_cluster dependency)
"""
