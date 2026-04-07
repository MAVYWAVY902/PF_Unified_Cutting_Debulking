#!/usr/bin/env python3
"""Generate Differentiable Sim + Robot Arm + Learning Survey PDF (Chinese-friendly, black text)"""
from fpdf import FPDF


class SurveyPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, 'Differentiable Sim + Robot Arm + Policy Learning Survey', align='R')
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'{self.page_no()} / {{nb}}', align='C')

    def stitle(self, t):
        self.set_font('Helvetica', 'B', 13)
        self.set_text_color(0, 0, 0)
        self.cell(0, 10, t, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(0, 0, 0)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def stitle2(self, t):
        self.set_font('Helvetica', 'B', 10.5)
        self.set_text_color(0, 0, 0)
        self.cell(0, 7, t, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def txt(self, t):
        self.set_font('Helvetica', '', 9.5)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 5, t)
        self.ln(2)

    def bold_txt(self, label, t):
        self.set_font('Helvetica', 'B', 9.5)
        self.set_text_color(0, 0, 0)
        self.write(5, label)
        self.set_font('Helvetica', '', 9.5)
        self.write(5, t)
        self.ln(6)

    def code(self, c):
        self.set_font('Courier', '', 7.5)
        self.set_fill_color(242, 242, 242)
        self.set_text_color(0, 0, 0)
        x = self.l_margin
        w = self.w - self.l_margin - self.r_margin
        lines = c.strip().split('\n')
        h = 3.8
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
        self.set_font('Helvetica', 'B' if bold else '', 7.5)
        self.set_text_color(0, 0, 0)
        if fill:
            self.set_fill_color(230, 230, 230)
        if widths is None:
            w_total = self.w - self.l_margin - self.r_margin
            widths = [int(w_total / len(cells))] * len(cells)
        for c, w in zip(cells, widths):
            self.cell(w, 6, c, border=1, fill=fill)
        self.ln()

    def paper(self, title, venue, desc):
        if self.get_y() > self.h - 30:
            self.add_page()
        self.set_font('Helvetica', 'B', 9)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 4.5, title)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(80, 80, 80)
        self.cell(0, 4.5, venue, new_x="LMARGIN", new_y="NEXT")
        self.set_font('Helvetica', '', 9)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 4.5, desc)
        self.ln(2)

    def num_item(self, n, t):
        self.set_font('Helvetica', 'B', 9.5)
        self.set_text_color(0, 0, 0)
        self.cell(8, 5, f'{n}.')
        self.set_font('Helvetica', '', 9.5)
        self.multi_cell(self.w - self.l_margin - self.r_margin - 8, 5, t)
        self.ln(1)


pdf = SurveyPDF()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.set_left_margin(15)
pdf.set_right_margin(15)
pdf.add_page()

# Title
pdf.set_font('Helvetica', 'B', 18)
pdf.set_text_color(0, 0, 0)
pdf.cell(0, 12, 'Differentiable Simulation', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 12, '+ Robot Arm + Policy Learning', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(3)
pdf.set_font('Helvetica', '', 11)
pdf.set_text_color(60, 60, 60)
pdf.cell(0, 8, 'CUSA Debulking: From Differentiable Sim to Real Robot', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(8)

# ============================================================
# 1. SAPO/SHAC
# ============================================================
pdf.stitle('1. SAPO / SHAC: Short-Horizon Differentiable RL')

pdf.txt(
    'SAPO (Soft Analytic Policy Optimization, ICLR 2025) and SHAC (Short-Horizon Actor-Critic, '
    'ICLR 2022) are the state-of-the-art methods for combining differentiable simulation with RL. '
    'They solve the long-rollout gradient explosion problem by only backpropagating through '
    'short windows (10-50 steps), using a learned critic to estimate long-term value.'
)

pdf.stitle2('SAPO Algorithm Details')
pdf.code(
"""SAPO (ICLR 2025, Rewarped Platform):

  Core idea: short-horizon analytic gradient + actor-critic RL

  For each training iteration:
    1. Sample parallel trajectories (N envs on GPU)
    2. Divide trajectory into short windows (10-50 steps)
    3. For each window:
       - Forward: run differentiable sim (Warp)
       - Backward: backprop gradient through sim
       - Policy gradient = analytic gradient from diff sim
       - Value target = multi-step reward + critic(last_state)
    4. Update actor with analytic policy gradient
    5. Update critic with TD learning (standard, no diff sim needed)

  Key differences from PPO/SAC:
    - Actor gradient comes from DIFF SIM (not reward sampling)
    - Critic is separate (trained via TD, not diff sim)
    - Maximum entropy formulation for exploration
    - 17x+ faster than PPO in wall-clock time""")

pdf.stitle2('SHAC vs SAPO vs PPO vs SAC')
w5 = [25, 30, 30, 30, 30]
pdf.trow(['Aspect', 'SAPO (2025)', 'SHAC (2022)', 'PPO', 'SAC'], bold=True, fill=True, widths=w5)
pdf.trow(['Gradient source', 'Diff sim (analytic)', 'Diff sim (analytic)', 'Reward sampling', 'Reward sampling'], widths=w5)
pdf.trow(['Backprop steps', '10-50', '10-50', '0 (no backprop)', '0 (no backprop)'], widths=w5)
pdf.trow(['Need diff sim?', 'Yes', 'Yes', 'No', 'No'], widths=w5)
pdf.trow(['Sample efficiency', 'Very high', 'Very high', 'Medium', 'High'], widths=w5)
pdf.trow(['Wall-clock speed', '17x vs PPO', '17x vs PPO', '1x (baseline)', '~2x vs PPO'], widths=w5)
pdf.trow(['Deformable?', 'Yes (Rewarped)', 'Mainly rigid', 'Any sim', 'Any sim'], widths=w5)
pdf.trow(['Stability', 'High (entropy)', 'High', 'Very high', 'Medium'], widths=w5)
pdf.trow(['On/off policy', 'On-policy', 'On-policy', 'On-policy', 'Off-policy'], widths=w5)

# ============================================================
# 2. Gradient Stability
# ============================================================
pdf.stitle('2. Long-Rollout Gradient Problem & Solutions')

pdf.txt(
    'Backpropagating through 100+ simulation steps causes: (1) GPU memory explosion, '
    '(2) gradient vanishing/exploding (chain of matrix multiplications), '
    '(3) gradient discontinuity at contact events. This is the #1 open problem.'
)

pdf.stitle2('Mitigation Strategies')
w4 = [32, 28, 25, 60]
pdf.trow(['Strategy', 'Backprop Steps', 'Memory', 'Who Uses It'], bold=True, fill=True, widths=w4)
pdf.trow(['Truncated BPTT', '10-50', 'Low', 'SAPO, SHAC (BEST for us)'], widths=w4)
pdf.trow(['Checkpointing', '100-500', 'Medium', 'Warp, DiffTaichi'], widths=w4)
pdf.trow(['Implicit Diff (IFT)', 'Any', 'Low', 'Dojo, contact-implicit MPC'], widths=w4)
pdf.trow(['Adaptive Horizon', 'Variable', 'Medium', 'AHAC (auto-adjust)'], widths=w4)
pdf.trow(['Gradient Smoothing', 'Any', 'Low', 'Soft contact approx'], widths=w4)

pdf.txt(
    'For our P(F) debulking sim: truncated BPTT (SAPO-style) is the best choice. '
    'Phase 4 verified 5-step gradient flow. SAPO uses 10-50 step windows. '
    'Our P(F) sigmoid is smooth, which helps gradient stability.'
)

# ============================================================
# 3. Method Comparison
# ============================================================
pdf.stitle('3. RL vs IL vs Diffusion: Which Method for Our Task?')

w5b = [28, 25, 25, 33, 34]
pdf.trow(['Method', 'Need Demo?', 'Need Diff Sim?', 'Sample Efficiency', 'Best For'], bold=True, fill=True, widths=w5b)
pdf.trow(['SAPO (diff+RL)', 'No', 'Yes', 'Very high (17x PPO)', 'OUR BEST CHOICE'], widths=w5b)
pdf.trow(['PPO (pure RL)', 'No', 'No', 'Low', 'Backup if diff fails'], widths=w5b)
pdf.trow(['SAC (pure RL)', 'No', 'No', 'Medium', 'Off-policy backup'], widths=w5b)
pdf.trow(['Diffusion Policy', 'Yes (50+ demos)', 'No', 'High (with demos)', 'Needs teleop setup'], widths=w5b)
pdf.trow(['ACT', 'Yes', 'No', 'Medium', 'Real-time deploy'], widths=w5b)
pdf.trow(['BC + diff refine', 'Few demos', 'Yes', 'High', 'Hybrid approach'], widths=w5b)

pdf.txt(
    'SAPO is our best choice because: (1) we have differentiable sim (P(F) + Warp), '
    '(2) no demonstration data needed, (3) 17x sample efficiency vs PPO, '
    '(4) the short-horizon approach matches our verified 5-step gradient capability.'
)

# ============================================================
# 4. Key Papers
# ============================================================
pdf.stitle('4. Key Papers')

pdf.paper('SAPO: Stabilizing RL in Differentiable Multiphysics Simulation',
          'ICLR 2025 | Rewarped Platform (NVIDIA Warp)',
          'Short-horizon analytic gradient + actor-critic. Handles rigid + articulated + '
          'deformable (elastic, plastic, fluid). Maximum entropy formulation. '
          '17x faster than PPO. Our primary reference for training approach.')

pdf.paper('SHAC: Short-Horizon Actor-Critic',
          'ICLR 2022',
          'Foundation for SAPO. Truncated differentiable gradient + actor-critic updates. '
          'Parallel environment sampling. Gradient smoothing for non-smooth contact. '
          'Tested on locomotion + manipulation tasks.')

pdf.paper('Stress-Guided RL for Gentle Manipulation of Deformable Objects',
          '2025 | UFactory xArm 7 + Tofu',
          'Zero-shot sim-to-real for deformable manipulation. RL with stress-based reward. '
          'Real robot successfully manipulates tofu without damage. '
          'Domain randomization on material properties.')

pdf.paper('HIL-SERL: Human-in-the-Loop Sample-Efficient RL',
          'Science Robotics 2024',
          'Real robot RL with human corrections. Precise dexterous manipulation. '
          'Few-shot fine-tuning on real hardware. Practical sim-to-real bridge.')

pdf.paper('SRT-H: Autonomous Surgical Task via Language-Conditioned IL',
          'Science Robotics 2025',
          'Autonomous gallbladder removal on pig cadaver. Language-conditioned IL. '
          '100% accuracy on standardized surgical tasks. da Vinci Research Kit.')

pdf.paper('PlasticineLab: Soft-Body Manipulation Benchmark',
          'ICLR 2021',
          'DiffTaichi backend. 100-200 step backprop works for plasticine. '
          'Trajectory optimization through differentiable MPM physics.')

# ============================================================
# 5. Sim-to-Real
# ============================================================
pdf.stitle('5. Sim-to-Real for Deformable Object Manipulation')

pdf.stitle2('Three Approaches (Ranked by Maturity)')

pdf.num_item(1, 'Domain Randomization (most mature): randomize tissue stiffness, damping, '
    'density, tool friction in sim. Train with randomized params. Deploy zero-shot. '
    'Success: tofu manipulation (xArm 7), tissue retraction (surgical robot).')

pdf.num_item(2, 'Residual Sim-to-Real (emerging): train base policy in sim, learn residual '
    'correction from few real episodes. HIL-SERL (Science Robotics 2024): human-in-the-loop '
    'RL on real robot. Practical and sample-efficient.')

pdf.num_item(3, 'Vision Domain Adaptation (surgical specific): GAN-based image translation '
    'from sim to real endoscopic images. Train vision policy on adapted images. '
    '50% sim-to-real success for tissue retraction.')

pdf.stitle2('Our Sim-to-Real Strategy')
pdf.code(
"""Step 1: Domain Randomization in Warp Sim
  - Tissue: stiffness [0.3, 1.0], damping [0.90, 0.99], density [800, 1200]
  - Tool: radius [0.8x, 1.2x], friction [0.1, 0.5]
  - Geometry: tumor shape variation (multiple .msh files)

Step 2: Train SAPO Policy
  - 16-32 parallel envs on single GPU
  - Short-horizon (10-50 steps) analytic gradient
  - P(F) geometry-aware reward: +volume - lambda*nerve_damage

Step 3: Zero-Shot Transfer to Real Robot
  - Franka Panda + CUSA-like tool on end-effector
  - Silicone 3-layer phantom (capsule + tumor + nerve)
  - RealSense depth camera for observation
  - If zero-shot fails: HIL fine-tuning (10-50 real episodes)""")

# ============================================================
# 6. Diff Sim Frameworks
# ============================================================
pdf.stitle('6. Differentiable Simulation Frameworks')

w4f = [30, 25, 40, 50]
pdf.trow(['Framework', 'Backend', 'Deformable Support', 'Notes'], bold=True, fill=True, widths=w4f)
pdf.trow(['Rewarped', 'Warp (CUDA)', 'Elastic, plastic, fluid', 'SAPO paper. Best for us'], widths=w4f)
pdf.trow(['Genesis', 'DiffTaichi', 'MPM, FEM, PBD', '10-80x faster than others'], widths=w4f)
pdf.trow(['Brax', 'JAX', 'Limited (rigid focus)', 'Massively parallel'], widths=w4f)
pdf.trow(['Dojo', 'Julia', 'Limited', 'Hard contact + friction'], widths=w4f)
pdf.trow(['DiffTaichi', 'Taichi', 'Elastic, cloth, fluid', '188x faster than TF'], widths=w4f)
pdf.trow(['Newton', 'Warp+MuJoCo', 'Planned', 'NVIDIA 2025. Isaac Lab'], widths=w4f)

# ============================================================
# 7. Our Plan
# ============================================================
pdf.stitle('7. Our Complete Plan: P(F) Debulking + SAPO + Franka')

pdf.code(
"""=== System Architecture ===

  Isaac Lab (Franka Panda arm dynamics)
       |
       | end-effector position
       v
  Warp (P(F) Shape Matching tissue physics)
       |
       | damage, alpha, positions
       v
  SAPO Training (short-horizon diff grad + critic)
       |
       | trained policy
       v
  Real Franka + Silicone Phantom (sim-to-real transfer)


=== Training Details ===

  Observation (7D):
    - end-effector position (3D)
    - local P(F) mean around tool (1D)  <-- geometry-aware!
    - volume removed so far (1D)
    - nerve damage so far (1D)
    - distance to nearest nerve (1D)

  Action (3D):
    - end-effector velocity (dx, dy, dz)

  Reward:
    r = +delta_volume_removed
        - lambda * delta_nerve_damage
        - mu * ||action||^2  (energy penalty)
        + bonus * (volume > target)

  P(F) advantage:
    - nerve_damage = sum of damage on nerve-layer particles
    - P(F) automatically gives high value to nerve (sparse)
    - Gradient naturally guides tool AWAY from nerve


=== Timeline ===

  Week 1-2: Gym env wrapper + SAPO implementation
  Week 3-4: Training + hyperparameter tuning
  Week 5: Ablation (SAPO vs PPO vs SAC)
  Week 6: Visualization + paper figures
  Week 7: Sim-to-real (Franka + phantom)
  Week 8: Paper writing""")

# ============================================================
# 8. Paper Framing
# ============================================================
pdf.stitle('8. Paper Framing & Venue Targets')

pdf.stitle2('Title Options')
pdf.txt(
    '(1) "Short-Horizon Differentiable Surgical Simulation with P(F) Geometry Sensing '
    'for Sample-Efficient Debulking Policy Learning"\n\n'
    '(2) "P(F)-Guided Analytic Policy Gradient for Autonomous Vestibular Schwannoma Debulking"\n\n'
    '(3) "Differentiable Geometry-Aware Reward Design for Surgical Debulking via SAPO"'
)

pdf.stitle2('Key Claims')
pdf.num_item(1, 'P(F) provides differentiable, geometry-aware damage sensing that naturally '
    'protects nerve tissue (vs hard thresholds in prior work)')
pdf.num_item(2, 'Short-horizon autodiff through shape matching + P(F) sigmoid enables '
    'analytic policy gradients (SAPO-compatible)')
pdf.num_item(3, 'P(F)-based reward gives 10x+ sample efficiency vs standard RL reward design')
pdf.num_item(4, 'First differentiable surgical debulking simulation with policy learning')

pdf.stitle2('Target Venues')
w3v = [30, 30, 85]
pdf.trow(['Venue', 'Deadline', 'Fit'], bold=True, fill=True, widths=w3v)
pdf.trow(['CoRL 2026', '~Jun 2026', 'Best fit: robot learning + diff sim'], widths=w3v)
pdf.trow(['ICRA 2027', '~Sep 2026', 'Needs real robot experiment'], widths=w3v)
pdf.trow(['RSS 2026', '~Feb 2026', 'Methods focus, tight deadline'], widths=w3v)
pdf.trow(['RA-L', 'Rolling', 'Good for incremental contribution'], widths=w3v)

# ============================================================
# 9. References
# ============================================================
pdf.stitle('9. Key References')
w2r = [75, 70]
pdf.trow(['Paper', 'Link / Venue'], bold=True, fill=True, widths=w2r)
pdf.trow(['SAPO / Rewarped (ICLR 2025)', 'arxiv.org/abs/2412.12089'], widths=w2r)
pdf.trow(['SHAC (ICLR 2022)', 'short-horizon-actor-critic.github.io'], widths=w2r)
pdf.trow(['Stress-Guided RL (xArm + Tofu)', 'arxiv.org/abs/2510.25405'], widths=w2r)
pdf.trow(['HIL-SERL (Science Robotics 2024)', 'hil-serl.github.io'], widths=w2r)
pdf.trow(['SRT-H Autonomous Surgery (Sci Rob)', 'doi:10.1126/scirobotics.adt5254'], widths=w2r)
pdf.trow(['PlasticineLab (ICLR 2021)', 'arxiv.org/abs/2104.03311'], widths=w2r)
pdf.trow(['DiffTaichi (ICLR 2020)', 'arxiv.org/abs/1910.00935'], widths=w2r)
pdf.trow(['Brax (NeurIPS 2021)', 'arxiv.org/abs/2106.13281'], widths=w2r)
pdf.trow(['Genesis (2024)', 'github.com/Genesis-Embodied-AI/Genesis'], widths=w2r)
pdf.trow(['Newton (NVIDIA 2025)', 'developer.nvidia.com/blog/newton'], widths=w2r)
pdf.trow(['Contact-Implicit MPC (TRO 2024)', 'msl.stanford.edu/papers/le_cleach_fast_2024'], widths=w2r)
pdf.trow(['DiffTactile (ICLR 2024)', 'difftactile.github.io'], widths=w2r)
pdf.trow(['Grasp D (Diff Contact Sim)', 'graspd.github.io'], widths=w2r)
pdf.trow(['PODS (Policy Opt via Diff Sim)', 'ICML 2021'], widths=w2r)

out = '/home/yunxin/pf-shape-matching/DiffSim_Learning_Survey.pdf'
pdf.output(out)
print(f'PDF generated: {out}')
