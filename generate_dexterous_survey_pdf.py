#!/usr/bin/env python3
"""Generate Dexterous Hand Manipulation Survey PDF"""
from fpdf import FPDF


class SurveyPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, 'Dexterous Hand Manipulation - Learning Methods Survey (2023-2026)', align='R')
        self.ln(10)
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'{self.page_no()} / {{nb}}', align='C')
    def stitle(self, t):
        self.set_font('Helvetica', 'B', 13)
        self.set_text_color(140, 30, 30)
        self.cell(0, 10, t, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(140, 30, 30)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)
    def stitle2(self, t):
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(50, 50, 50)
        self.cell(0, 8, t, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)
    def txt(self, t):
        self.set_font('Helvetica', '', 9.5)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 5.2, t)
        self.ln(2)
    def code(self, c):
        self.set_font('Courier', '', 7.5)
        self.set_fill_color(245, 240, 240)
        self.set_text_color(30, 30, 30)
        x, w = self.l_margin, self.w - self.l_margin - self.r_margin
        lines = c.strip().split('\n')
        h = 3.8
        th = len(lines) * h + 6
        if self.get_y() + th > self.h - 20: self.add_page()
        y0 = self.get_y()
        self.rect(x, y0, w, th, 'F')
        self.set_xy(x + 3, y0 + 3)
        for line in lines:
            self.cell(w - 6, h, line, new_x="LMARGIN", new_y="NEXT")
            self.set_x(x + 3)
        self.set_y(y0 + th + 3)
    def trow(self, cells, bold=False, fill=False, widths=None):
        self.set_font('Helvetica', 'B' if bold else '', 7.5)
        if fill: self.set_fill_color(240, 230, 230)
        if widths is None:
            widths = [int((self.w - self.l_margin - self.r_margin) / len(cells))] * len(cells)
        for c, w in zip(cells, widths):
            self.cell(w, 6.5, c, border=1, fill=fill)
        self.ln()
    def paper(self, title, venue, desc):
        self.set_font('Helvetica', 'B', 9)
        self.set_text_color(140, 30, 30)
        self.multi_cell(0, 4.5, title)
        self.set_font('Helvetica', 'I', 8.5)
        self.set_text_color(100, 100, 100)
        self.cell(0, 4.5, venue, new_x="LMARGIN", new_y="NEXT")
        self.set_font('Helvetica', '', 9)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 4.5, desc)
        self.ln(2)
    def red_txt(self, t):
        self.set_font('Helvetica', 'B', 9.5)
        self.set_text_color(140, 30, 30)
        self.multi_cell(0, 5.2, t)
        self.set_text_color(0, 0, 0)
        self.ln(1)

pdf = SurveyPDF()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()

# Title
pdf.set_font('Helvetica', 'B', 20)
pdf.set_text_color(120, 20, 20)
pdf.cell(0, 12, 'Dexterous Hand Manipulation', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.set_font('Helvetica', '', 13)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 10, 'Learning Methods Survey (2023-2026)', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(8)

# ============================================================
# Section 1: Training Methods
# ============================================================
pdf.stitle('1. Training Methods Comparison')

w6 = [22, 28, 28, 22, 22, 23]
pdf.trow(['Method', 'Representative', 'Venue', 'Pros', 'Cons', 'Best For'], bold=True, fill=True, widths=w6)
pdf.trow(['RL (PPO)', 'OpenAI Dexterity, BiDexHands', 'IJRR19, ICLR25', 'Contact-rich', 'Sample hungry', 'In-hand rotation'], widths=w6)
pdf.trow(['RL (SAC)', 'DexGraspRL', '2025', 'Sample efficient', 'Still millions', 'Grasping'], widths=w6)
pdf.trow(['Imitation', 'DexMV, DexUMI', 'ECCV22, CoRL25', 'Natural demos', 'Needs data', 'Tool use'], widths=w6)
pdf.trow(['Diffusion', 'ALOHA Unleashed, UniDexFPM', 'DeepMind24', 'Multi-modal', 'Slow (~10Hz)', 'Complex seq'], widths=w6)
pdf.trow(['ACT', 'FTACT, SERNF', '2024-25', 'Fast inference', 'Less expressive', 'Real-time'], widths=w6)
pdf.trow(['Teacher-Student', 'DextrAH-RGB', '2024', 'Sim->real', 'Two-stage', 'Vision policy'], widths=w6)
pdf.trow(['World Model', 'DexWM', '2025', 'Cross-embodiment', 'Model error', 'Video->robot'], widths=w6)
pdf.trow(['VLA+low-level', 'DexVLG, OpenVLA', 'ICCV25', 'Language+prec', 'Complex pipe', 'General manip'], widths=w6)
pdf.trow(['LLM Reward', 'Eureka, DrEureka', 'ICLR24, RSS24', 'Auto reward', 'LLM cost', 'RL reward design'], widths=w6)

# ============================================================
# Section 2: Dominant Pipeline
# ============================================================
pdf.stitle('2. 2024-2025 Dominant Training Pipeline')
pdf.code(
"""Current best practice (most papers follow this):

  Step 1: Teacher Policy (RL)
    - PPO in IsaacGym/Isaac Lab with privileged state info
    - Observations: joint angles, object pose, contact forces (sim-only)
    - Domain randomization: physics, actuation, object shape

  Step 2: Student Policy (Distillation)
    - Behavior cloning / DAgger from teacher
    - Input: RGB / depth / point cloud (real-sensor compatible)
    - Output: joint velocities or positions

  Step 3: Sim-to-Real Transfer
    - Zero-shot (domain randomization) OR
    - Few-shot fine-tuning (SERNF/DEFT, <100 real episodes)

  Alternative Pipeline (rising):
    - Teleoperation (DexUMI/Bunny-VisionPro) -> collect 50-200 demos
    - Train Diffusion Policy / ACT directly on demos
    - Deploy on real robot""")

# ============================================================
# Section 3: Key Projects
# ============================================================
pdf.stitle('3. Key Projects: Teleoperation & Data Collection')

pdf.paper('DexUMI: Human Hand as Universal Manipulation Interface',
          'CoRL 2025 | Stanford, Columbia, NVIDIA',
          'Human hand with exoskeleton + robot hand inpainting. 86% success across multiple platforms. '
          'Key insight: bridge embodiment gap via hardware adapter + visual inpainting.')

pdf.paper('Bunny-VisionPro: Real-Time Bimanual Dexterous Teleoperation',
          'ICRA 2024',
          'Apple Vision Pro for bimanual teleoperation. Dual 24-DoF (xArm-7 + Ability hands). '
          'Outperforms prior teleop systems. Enables high-quality demo collection.')

pdf.paper('UMI: Universal Manipulation Interface',
          'ICRA 2024',
          'Hand-held gripper for portable data collection. Zero-shot generalization. '
          'Variants: MV-UMI, ActiveUMI, DexUMI, exUMI.')

pdf.paper('DexCap: Scalable Portable Mocap Data Collection',
          'RSS 2024',
          'EMF tracking gloves + RGB-D LiDAR. 3x faster than teleoperation. '
          '72% success from 30-minute mocap sessions.')

pdf.paper('AnyTeleop: General Vision-Based Dexterous Teleoperation',
          'ICRA 2023',
          'Hardware-agnostic teleoperation with only RGB/RGB-D camera. '
          'Higher success than specialized systems in 8/10 tasks.')

pdf.stitle('4. Key Projects: RL & Policy Learning')

pdf.paper('BiDexHands: Human-Level Bimanual Dexterous Manipulation',
          'ICLR 2025 | PKU-MARL',
          'Dual Shadow Hands benchmark. 50+ bimanual tasks. IPPO teacher -> vision student. '
          'Isaac Gym. Widely used benchmark for dexterous research.')

pdf.paper('Eureka: Human-Level Reward Design via LLM',
          'ICLR 2024',
          'GPT-4 auto-generates reward functions for RL. Outperforms human rewards on 83% of 29 tasks. '
          '52% average improvement. Shadow Hand pen spinning tricks.')

pdf.paper('DrEureka: LLM-Guided Sim-to-Real Transfer',
          'RSS 2024',
          'LLM automates domain randomization configuration. Eliminates manual DR tuning. '
          'Validated on quadruped + dexterous manipulation.')

pdf.paper('DextrAH-RGB: Visuomotor Policies to Grasp Anything',
          '2024',
          'Teacher (state RL, IPPO) -> Student (vision, DAgger). Isaac Sim. '
          'Zero-shot sim-to-real transfer for diverse object grasping.')

pdf.stitle('5. Key Projects: Diffusion & Imitation')

pdf.paper('ALOHA Unleashed: Simple Recipe for Robot Dexterity',
          'Google DeepMind 2024',
          'Diffusion Policy on 26k+ demos. Tasks: shoelace tying, shirt hanging, gear insertion. '
          'Key insight: large-scale data + diffusion = complex dexterity. $32K hardware.')

pdf.paper('DexMV: Learning from Human Videos',
          'ECCV 2022 | Google',
          'YouTube videos -> hand pose -> robot demos. 83% success on real Allegro Hand. '
          'Eliminates need for VR/teleoperation for data collection.')

pdf.paper('UniDexFPM: Dexterous Functional Pre-Grasp Manipulation',
          '2024',
          'Mixture-of-experts -> diffusion policy. 72.6% across 30+ object categories. '
          'Isaac Gym. Pre-grasp reorientation before grasping.')

pdf.paper('Tilde: Diffusion + DAgger for Dexterous Manipulation',
          'RSS 2024',
          'Diffusion policy + human DAgger corrections. DeltaHand (4-finger, 12-DoF). '
          '90% average success across 7 tasks.')

pdf.stitle('6. Key Projects: World Models & Foundation Models')

pdf.paper('DexWM: Dexterous Interaction World Model',
          '2025',
          '900+ hours human and robot data. Cross-embodiment transfer. '
          'Beats Diffusion Policy by 50%+ on Franka + Allegro. Zero-shot transfer.')

pdf.paper('EgoDex: Learning from Large-Scale Egocentric Video',
          'Apple 2025',
          '829 hours egocentric video. 194 household tasks. Apple Vision Pro capture. '
          'Largest dexterous manipulation dataset. Open-source.')

pdf.paper('HumanPlus: Humanoid Shadowing and Imitation',
          'CoRL 2024',
          '33-DoF humanoid with dual 6-DoF hands. Learns shoe wearing, warehouse unloading, typing. '
          '60-100% success from ~40 demos. Real-time human shadowing.')

pdf.paper('OpenVLA: Open-Source Vision-Language-Action Model',
          '2024',
          '7B parameter VLA. 970k real-world demos. 50%+ success on diverse manipulation. '
          'Fully open-source and fine-tunable.')

# ============================================================
# Section 7: Hardware
# ============================================================
pdf.stitle('7. Hardware Platforms')
w4 = [30, 20, 25, 70]
pdf.trow(['Hand', 'Cost', 'DOF', 'Notes'], bold=True, fill=True, widths=w4)
pdf.trow(['Shadow Hand', '$100K+', '24', 'Industry standard for RL research. UK ARIA $57M funding'], widths=w4)
pdf.trow(['Allegro Hand', '$16K', '16', 'Good middle ground. DexMV, DexPoint, DexCap'], widths=w4)
pdf.trow(['LEAP Hand', '$2K', '16', 'RSS 2024. 3D-printed. Outperforms Allegro in tests'], widths=w4)
pdf.trow(['RUKA v2', '<$2K', '15', 'NYU 2025. Tendon-driven. Open-source. Fishing line tendons'], widths=w4)
pdf.trow(['ORCA Hand', '$1.5-6K', '17', 'ETH 2025. Tendon-driven. 10K+ cycles. Optional tactile'], widths=w4)
pdf.trow(['Ability Hand', '$20K+', '10', 'Used in Bunny-VisionPro. Commercial grade'], widths=w4)
pdf.trow(['DeltaHand', 'Research', '12', 'CMU. Non-anthropomorphic delta robot design'], widths=w4)

# ============================================================
# Section 8: Surgical
# ============================================================
pdf.stitle('8. Surgical Dexterous Manipulation')

pdf.paper('Dexterity-Enhancing Handheld Robot for Endoscopic Neurosurgery',
          'npj Robotics 2025 | First in-vivo study',
          'Panda Surgical handheld robot. Interchangeable articulated end-effectors '
          '(forceps, curettes, dissectors). 2-3 DoF per tool. First clinical validation '
          'for dexterous neurosurgery.')

pdf.paper('SutureBot: Autonomous End-to-End Surgical Suturing',
          '2025 | arXiv:2510.20965',
          'da Vinci Research Kit. 1,890 demos for needle pickup, tissue insertion, knot tying. '
          'Imitation learning. First comprehensive autonomous suturing framework.')

pdf.paper('Dex Surgical: Commercial Surgical Robotics',
          'Commercial | PrecisionDex 3D Technology',
          '7-DoF dexterous laparoscopic tools. Commercial solution for minimally invasive surgery.')

pdf.red_txt('Gap: No existing work combines dexterous hand + CUSA tool + differentiable debulking simulation. '
            'This is our unique opportunity.')

# ============================================================
# Section 9: Our Application
# ============================================================
pdf.stitle('9. Application to Our Project: Dexterous CUSA Debulking')

pdf.code(
"""Proposed System: Dexterous Hand + CUSA Tool + P(F) Differentiable Sim

Hardware:
  - LEAP Hand ($2K) or Allegro Hand ($16K)
  - CUSA-like tool mounted on fingertips
  - RealSense depth camera for observation

Training Pipeline (following dominant paradigm):
  Step 1: Teacher Policy (RL, PPO/SAC)
    - Sim: Isaac Lab (hand dynamics) + Warp (P(F) tissue physics)
    - obs: hand joints + CUSA tip pos + local P(F) distribution
    - action: hand joint velocities
    - reward: +volume_removed - lambda*nerve_damage(P(F))
    - P(F) provides geometry-aware dense reward (our advantage)

  Step 2: Student Policy (Vision-based)
    - Teacher (state) -> Student (depth image) via DAgger
    - OR: Diffusion Policy on teleop demos (ALOHA-style)

  Step 3: Sim-to-Real
    - Domain randomization on tissue properties
    - Short-horizon autodiff for local trajectory refinement
    - Transfer to real LEAP Hand + silicone phantom

Paper Framing:
  "Differentiable Geometry-Aware Debulking with Dexterous Manipulation:
   P(F)-guided RL for Vestibular Schwannoma Surgery"

Venues: CoRL, ICRA, RSS""")

pdf.stitle('10. Recommended Reading Priority')
w3 = [55, 25, 65]
pdf.trow(['Paper', 'Venue', 'Why Read'], bold=True, fill=True, widths=w3)
pdf.trow(['SAPO (Stabilizing RL + DiffSim)', '2024', 'Our exact approach: short diff + long RL'], widths=w3)
pdf.trow(['DexUMI', 'CoRL 2025', 'Best teleop for dexterous data collection'], widths=w3)
pdf.trow(['BiDexHands', 'ICLR 2025', 'PPO benchmark for dexterous RL'], widths=w3)
pdf.trow(['ALOHA Unleashed', 'DeepMind 2024', 'Diffusion policy for complex dexterity'], widths=w3)
pdf.trow(['Eureka / DrEureka', 'ICLR24/RSS24', 'LLM auto reward + domain randomization'], widths=w3)
pdf.trow(['DextrAH-RGB', '2024', 'Teacher-student sim-to-real pipeline'], widths=w3)
pdf.trow(['LEAP Hand', 'RSS 2024', 'Low-cost hardware ($2K)'], widths=w3)
pdf.trow(['DexWM (World Model)', '2025', 'Cross-embodiment, video learning'], widths=w3)
pdf.trow(['Neurosurgery Handheld Robot', 'npj 2025', 'First clinical dexterous neurosurgery'], widths=w3)
pdf.trow(['DexDeform', 'ICLR 2023', 'Dexterous + deformable + diff physics'], widths=w3)

out = '/home/yunxin/pf-shape-matching/Dexterous_Hand_Survey.pdf'
pdf.output(out)
print(f'PDF generated: {out}')
