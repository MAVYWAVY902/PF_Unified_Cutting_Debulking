#!/usr/bin/env python3
"""Generate Differentiable Simulation Survey PDF (Chinese)"""
from fpdf import FPDF


class SurveyPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, 'Differentiable Simulation for Robotics - Literature Survey', align='R')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'{self.page_no()} / {{nb}}', align='C')

    def stitle(self, t):
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(25, 60, 120)
        self.cell(0, 10, t, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(25, 60, 120)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def stitle2(self, t):
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(50, 50, 50)
        self.cell(0, 8, t, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def txt(self, t):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 5.5, t)
        self.ln(2)

    def bold_txt(self, label, t):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(0, 0, 0)
        self.write(5.5, label)
        self.set_font('Helvetica', '', 10)
        self.write(5.5, t)
        self.ln(7)

    def code(self, c):
        self.set_font('Courier', '', 8)
        self.set_fill_color(240, 242, 248)
        self.set_text_color(30, 30, 30)
        x, w = self.l_margin, self.w - self.l_margin - self.r_margin
        lines = c.strip().split('\n')
        h = 4.0
        th = len(lines) * h + 6
        if self.get_y() + th > self.h - 20:
            self.add_page()
        y0 = self.get_y()
        self.rect(x, y0, w, th, 'F')
        self.set_xy(x + 3, y0 + 3)
        for line in lines:
            self.cell(w - 6, h, line, new_x="LMARGIN", new_y="NEXT")
            self.set_x(x + 3)
        self.set_y(y0 + th + 3)

    def trow(self, cells, bold=False, fill=False, widths=None):
        self.set_font('Helvetica', 'B' if bold else '', 8)
        if fill:
            self.set_fill_color(225, 232, 245)
        if widths is None:
            widths = [int((self.w - self.l_margin - self.r_margin) / len(cells))] * len(cells)
        for c, w in zip(cells, widths):
            self.cell(w, 7, c, border=1, fill=fill)
        self.ln()

    def num_item(self, n, t):
        self.set_font('Helvetica', 'B', 10)
        self.cell(8, 5.5, f'{n}.')
        self.set_font('Helvetica', '', 10)
        self.multi_cell(self.w - self.l_margin - self.r_margin - 8, 5.5, t)
        self.ln(1)

    def blue_txt(self, t):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(25, 60, 120)
        self.multi_cell(0, 5.5, t)
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def paper_entry(self, title, venue, desc):
        self.set_font('Helvetica', 'B', 9.5)
        self.set_text_color(25, 60, 120)
        self.multi_cell(0, 5, title)
        self.set_font('Helvetica', 'I', 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, venue, new_x="LMARGIN", new_y="NEXT")
        self.set_font('Helvetica', '', 9.5)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 5, desc)
        self.ln(3)


pdf = SurveyPDF()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()

# Title
pdf.set_font('Helvetica', 'B', 20)
pdf.set_text_color(20, 50, 100)
pdf.cell(0, 12, 'Differentiable Simulation', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 12, 'for Robotics', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(3)
pdf.set_font('Helvetica', '', 12)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 8, 'Literature Survey (2020-2026)', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(3)
pdf.set_font('Helvetica', 'I', 10)
pdf.set_text_color(120, 120, 120)
pdf.cell(0, 8, 'P(F) Shape Matching Debulking Project Reference', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(8)

# ============================================================
# Section 1
# ============================================================
pdf.stitle('1. Differentiable Simulation Overview')
pdf.txt(
    'Differentiable simulation means computing exact gradients of simulation output '
    'w.r.t. input parameters (tool trajectory, material properties, etc.) via automatic '
    'differentiation (autodiff). This enables gradient-based optimization of robot control, '
    'system identification, and design through physics.'
)

pdf.stitle2('Three Methods for Gradient Computation')
w3 = [35, 55, 55]
pdf.trow(['Method', 'How It Works', 'Pros / Cons'], bold=True, fill=True, widths=w3)
pdf.trow(['Finite Difference', 'Perturb each param, re-run sim', 'Simple / O(N) sims per grad'], widths=w3)
pdf.trow(['Tape-based AD', 'Record ops, auto backward', 'Exact / Memory-heavy for long rollout'], widths=w3)
pdf.trow(['Analytical Gradient', 'Hand-derive adjoint equations', 'Fast & exact / Hard to implement'], widths=w3)
pdf.trow(['Implicit Diff (IFT)', 'Differentiate equilibrium eqn', 'Any horizon / Only for steady-state'], widths=w3)

# ============================================================
# Section 2: Core Frameworks
# ============================================================
pdf.stitle('2. Core Differentiable Simulation Frameworks')

pdf.paper_entry(
    'DiffTaichi: Differentiable Programming for Physical Simulation',
    'ICLR 2020 | MIT',
    'Taichi-based tape AD. 188x faster than TensorFlow. Supports elastic, cloth, rigid/soft coupling. '
    'Light-weight tape records program structure. Foundation for PlasticineLab.'
)
pdf.paper_entry(
    'Brax: A Differentiable Physics Engine for Large Scale Rigid Body Simulation',
    'NeurIPS 2021 Benchmarks | Google',
    'JAX-based massively parallel rigid body engine. 4 pipelines (MJX, Generalized, PBD, Spring). '
    'Includes PPO/SAC/ES. Differentiable through entire sim + learning loop.'
)
pdf.paper_entry(
    'NVIDIA Warp',
    '2022-present | NVIDIA',
    'Python -> CUDA JIT. wp.Tape for reverse-mode AD. wp.svd3 differentiable. '
    'Supports PyTorch/JAX interop. Gradient checkpointing for memory. '
    'Production-grade performance. Used in Isaac Sim ecosystem.'
)
pdf.paper_entry(
    'Dojo: A Differentiable Physics Engine for Robotics',
    '2022 | Stanford',
    'Hard contact + friction via SOCP. Implicit differentiation (IFT). '
    'Julia with Python bindings. Applications: traj opt, policy learning, system ID.'
)
pdf.paper_entry(
    'Genesis: A Universal Physics Engine',
    '2024 | Multi-lab (20+ labs)',
    'Universal engine: MPM, FEM, PBD, SPH, Rigid. Currently MPM differentiable. '
    'Photo-realistic ray-tracing. Positioning as next-gen platform.'
)

# ============================================================
# Section 3: Differentiable PBD
# ============================================================
pdf.stitle('3. Differentiable Position-Based Dynamics (PBD)')

pdf.paper_entry(
    'DiffXPBD: Differentiable Position-Based Simulation of Compliant Constraint Dynamics',
    'ACM SIGGRAPH 2023',
    'XPBD with analytical gradient formulation. Handles 26M+ DOF meshes + collisions. '
    'No relaxation needed. Most directly relevant to our Shape Matching PBD framework. '
    'Shows PBD CAN be made differentiable with proper gradient derivation.'
)
pdf.paper_entry(
    'DiffPD: Differentiable Projective Dynamics',
    'ACM TOG',
    'Implicit time integration + Cholesky prefactorization. '
    'Exploits precomputed factorization to speed backward pass. Soft-body focused.'
)

# ============================================================
# Section 4: Soft Body
# ============================================================
pdf.stitle('4. Differentiable Soft Body Simulation')

pdf.paper_entry(
    'PlasticineLab: A Soft-Body Manipulation Benchmark',
    'ICLR 2021',
    'First soft-body benchmark with differentiable physics (DiffTaichi/MPM backend). '
    'Tasks: shape plasticine to target. 100-200 step rollout backprop works. '
    'Analytical gradients in soft-body benchmark.'
)
pdf.paper_entry(
    'DiffTactile: A Physics-based Differentiable Tactile Simulator',
    'ICLR 2024',
    'FEM-based elastomer simulation. Multi-material (elastic, elastoplastic, cables). '
    'Penalty contact model. End-to-end differentiable.'
)
pdf.paper_entry(
    'SoftMAC: Differentiable Soft Body Simulation with Forecast-based Contact',
    '2023',
    'MPM soft bodies. Forecast-based contact reduces penetration. '
    'Two-way coupling: rigid + cloth + soft. Differentiable.'
)
pdf.paper_entry(
    'Fast and Reliable Gradients for Deformables Across Frictional Contact',
    '2025',
    'Analytical gradients validated vs FD. 33x speedup over finite difference. '
    'Handles soft AND stiff materials, small/large deformations.'
)

# ============================================================
# Section 5: Long-Horizon Problem
# ============================================================
pdf.stitle('5. Long-Rollout Gradient Stability (Core Challenge)')

pdf.txt(
    'Backpropagating through 100+ simulation steps causes: (1) GPU memory explosion '
    '(tape stores all intermediate states), (2) gradient vanishing/exploding '
    '(500 matrix multiplications), (3) gradient discontinuity at contact events. '
    'This is the #1 open problem in differentiable simulation.'
)

pdf.stitle2('Mitigation Strategies')
w4 = [30, 30, 30, 55]
pdf.trow(['Strategy', 'Rollout Steps', 'Memory', 'Who Uses It'], bold=True, fill=True, widths=w4)
pdf.trow(['Truncated BPTT', '10-50', 'Low', 'SAPO (2024), Brax'], widths=w4)
pdf.trow(['Checkpointing', '100-500', 'Medium', 'Warp, DiffTaichi'], widths=w4)
pdf.trow(['Implicit Diff (IFT)', 'Any', 'Low', 'Dojo, contact-implicit MPC'], widths=w4)
pdf.trow(['RL + short grad', 'RL:inf, grad:10-50', 'Low', 'SAPO (2024) [BEST]'], widths=w4)

# ============================================================
# Section 6: RL + Differentiable Sim
# ============================================================
pdf.stitle('6. RL + Short-Horizon Differentiable Sim (Most Relevant)')

pdf.paper_entry(
    'SAPO: Stabilizing RL in Differentiable Multiphysics Simulation',
    '2024 | Rewarped Platform',
    'KEY PAPER for our approach. Short-horizon actor-critic: backprop only 10-50 steps, '
    'use RL for long-horizon. Handles rigid + articulated + deformables. '
    'Addresses exploding/vanishing gradients. Solutions: truncate stiff contacts, '
    'reweight/smooth gradients. Maximum entropy formulation (like SAC).'
)
pdf.paper_entry(
    'Accelerated Policy Learning with Parallel Differentiable Simulation',
    'ICLR 2022',
    'Short-horizon actor-critic with truncated gradient backprop. '
    'Parallel environment simulation. Foundation for SAPO approach.'
)
pdf.paper_entry(
    'Learning Deployable Locomotion via Differentiable Simulation',
    'ICRA 2024',
    'First successful sim-to-real transfer of legged locomotion learned entirely '
    'in differentiable simulator. Proves differentiable sim -> real robot works.'
)

pdf.blue_txt('SAPO (2024) validates our approach: short-horizon autodiff (5-50 steps) + '
             'long-horizon RL (SAC). This is exactly what our P(F) framework enables.')

# ============================================================
# Section 7: Surgical / Medical
# ============================================================
pdf.stitle('7. Differentiable Surgical Simulation')

pdf.paper_entry(
    'VR-augmented Differentiable Simulations for Digital Twin',
    'Nature Scientific Reports 2025',
    'Digital twin for surgical planning. Trajectory generation in segmented medical images. '
    'VR overlay. Differentiable simulation for planning optimization.'
)
pdf.paper_entry(
    'Real-time Capable Learning-based Visual Tool Pose Correction via Differentiable Sim',
    '2025',
    'dVRK-compatible differentiable simulator. Supervised learning: tool segmentation -> '
    'pose corrections. Parallel computing for efficient gradients.'
)
pdf.paper_entry(
    'Warp-SurgicalSim (Our Reference Project)',
    'Unpublished | XPBD + Warp + SAC',
    'XPBD surgical sim with adhesion bonds. Claims "differentiable" but does NOT use '
    'wp.Tape or backprop through physics. Pure RL (SAC) training. '
    'Our P(F) framework has actual short-horizon differentiability -- a real advantage.'
)

# ============================================================
# Section 8: System Identification
# ============================================================
pdf.stitle('8. System Identification via Differentiable Sim')

pdf.paper_entry(
    'gradSim: Differentiable Simulation for System ID and Visuomotor Control',
    'ICLR 2021',
    'Differentiable sim + rendering. Recovers physical properties from video. '
    'Parameters: friction, elasticity, material params. Backprop from pixels -> physics.'
)
pdf.paper_entry(
    'DPSI: Differentiable Physics-based System ID for Elastoplastic Materials',
    '2024',
    'Robots infer physics from manipulation + 3D point clouds. '
    'Directly relevant to our Task 2 (inverse tissue parameter estimation).'
)

# ============================================================
# Section 9: Our Position
# ============================================================
pdf.stitle('9. Our P(F) Framework Position in the Landscape')

pdf.stitle2('What We Have (Verified)')
pdf.num_item(1, 'P(F) sigmoid damage model -- smooth, differentiable geometry sensing')
pdf.num_item(2, 'Shape matching PBD in Warp with wp.svd3 -- differentiable SVD')
pdf.num_item(3, 'Soft removal via sigmoid mask -- differentiable particle removal')
pdf.num_item(4, 'Short-horizon gradient flow verified (5 steps, Phase 4)')
pdf.num_item(5, 'Finite difference gradients correct direction (Phase 4)')

pdf.stitle2('Our Unique Contribution vs Existing Work')
w2 = [72, 72]
pdf.trow(['Existing Work', 'Our P(F) Framework'], bold=True, fill=True, widths=w2)
pdf.trow(['DiffXPBD: generic soft body', 'Surgical debulking with tissue sensing'], widths=w2)
pdf.trow(['PlasticineLab: shape deformation', 'Volume removal (debulking)'], widths=w2)
pdf.trow(['SAPO: generic RL + diff sim', 'P(F) geometry-aware reward'], widths=w2)
pdf.trow(['Warp-SurgicalSim: claims diff, not real', 'Actual short-horizon autodiff'], widths=w2)
pdf.trow(['Beihang SM: no differentiability', 'P(F) sigmoid is differentiable'], widths=w2)

pdf.stitle2('Recommended Paper Framing')
pdf.code(
""""Short-horizon differentiable surgical simulation with
P(F) geometry sensing, enabling analytic policy gradients
for sample-efficient RL training of debulking strategies."

Key claims:
1. P(F) provides differentiable, geometry-aware damage sensing
   (vs hard thresholds in Beihang/CRESSim)
2. Short-horizon autodiff through shape matching + P(F) sigmoid
   (verified, 5-step gradient flow)
3. Combined with SAC for long-horizon planning
   (following SAPO 2024 paradigm)
4. First differentiable surgical debulking simulation
   (Warp-SurgicalSim is not actually differentiable)""")

# ============================================================
# Section 10: Key References
# ============================================================
pdf.stitle('10. Key References')

refs = [
    ('DiffXPBD', 'ACM SIGGRAPH 2023', 'arxiv.org/abs/2301.01396'),
    ('SAPO / Rewarped', '2024', 'arxiv.org/abs/2412.12089'),
    ('PlasticineLab', 'ICLR 2021', 'arxiv.org/abs/2104.03311'),
    ('DiffTaichi', 'ICLR 2020', 'arxiv.org/abs/1910.00935'),
    ('Brax', 'NeurIPS 2021', 'arxiv.org/abs/2106.13281'),
    ('NVIDIA Warp', '2022-present', 'nvidia.github.io/warp/'),
    ('Dojo', '2022', 'arxiv.org/abs/2203.00806'),
    ('gradSim', 'ICLR 2021', 'arxiv.org/abs/2104.02646'),
    ('DiffTactile', 'ICLR 2024', 'openreview.net/forum?id=eJHnSg783t'),
    ('Genesis', '2024', 'github.com/Genesis-Embodied-AI/Genesis'),
    ('Fast Deformable Gradients', '2025', 'arxiv.org/abs/2603.16478'),
    ('Deployable Locomotion', 'ICRA 2024', 'arxiv.org/abs/2404.02887'),
    ('Accelerated Policy Learning', 'ICLR 2022', 'arxiv.org/abs/2204.07137'),
    ('Beihang SM GPU', 'CMPB 2024', 'ScienceDirect'),
    ('VR Differentiable Digital Twin', 'Sci Reports 2025', 'nature.com/articles/s41598-025-99608-2'),
]

w_ref = [45, 35, 65]
pdf.trow(['Paper', 'Venue', 'Link'], bold=True, fill=True, widths=w_ref)
for name, venue, link in refs:
    pdf.trow([name, venue, link], widths=w_ref)

out = '/home/yunxin/pf-shape-matching/DiffSim_Survey.pdf'
pdf.output(out)
print(f'PDF generated: {out}')
