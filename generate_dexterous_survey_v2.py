#!/usr/bin/env python3
"""Generate Dexterous Hand Manipulation Survey PDF - Chinese, black text, proper margins"""
from fpdf import FPDF


class SurveyPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, 'Dexterous Hand Manipulation Survey (2023-2026)', align='R')
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
        if self.get_y() > self.h - 35:
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

    def bold_line(self, t):
        self.set_font('Helvetica', 'B', 9.5)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 5, t)
        self.ln(1)


pdf = SurveyPDF()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.set_left_margin(15)
pdf.set_right_margin(15)
pdf.add_page()

# Title
pdf.set_font('Helvetica', 'B', 20)
pdf.set_text_color(0, 0, 0)
pdf.cell(0, 12, 'Dexterous Hand Manipulation', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.set_font('Helvetica', '', 12)
pdf.set_text_color(60, 60, 60)
pdf.cell(0, 8, 'Learning Methods Survey (2023-2026)', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(8)

# ============================================================
# 1
# ============================================================
pdf.stitle('1. Training Methods Comparison')
pdf.txt('The following table summarizes the main training paradigms for dexterous hand manipulation, with representative works, advantages, and limitations.')

w6 = [22, 28, 24, 24, 24, 24]
pdf.trow(['Method', 'Representative', 'Venue', 'Pros', 'Cons', 'Best For'], bold=True, fill=True, widths=w6)
pdf.trow(['RL (PPO)', 'OpenAI, BiDexHands', 'IJRR19,ICLR25', 'Contact-rich', 'Sample hungry', 'In-hand rot'], widths=w6)
pdf.trow(['RL (SAC)', 'DexGraspRL', '2025', 'More efficient', 'Still millions', 'Grasping'], widths=w6)
pdf.trow(['Imitation', 'DexMV, DexUMI', 'ECCV22,CoRL25', 'Natural demos', 'Needs data', 'Tool use'], widths=w6)
pdf.trow(['Diffusion', 'ALOHA, UniDexFPM', 'DeepMind24', 'Multi-modal', 'Slow ~10Hz', 'Complex seq'], widths=w6)
pdf.trow(['ACT', 'FTACT, SERNF', '2024-25', 'Fast inference', 'Less express', 'Real-time'], widths=w6)
pdf.trow(['Teacher-Stu', 'DextrAH-RGB', '2024', 'Sim->real', 'Two-stage', 'Vision policy'], widths=w6)
pdf.trow(['World Model', 'DexWM', '2025', 'Cross-embod', 'Model error', 'Video->robot'], widths=w6)
pdf.trow(['VLA+low', 'DexVLG, OpenVLA', 'ICCV25', 'Lang+prec', 'Complex pipe', 'General'], widths=w6)
pdf.trow(['LLM Reward', 'Eureka, DrEureka', 'ICLR24,RSS24', 'Auto reward', 'LLM cost', 'Reward design'], widths=w6)

# ============================================================
# 2
# ============================================================
pdf.stitle('2. 2024-2025 Dominant Training Pipeline')
pdf.code(
"""Current best practice:

  Step 1: Teacher Policy (RL, PPO/SAC in IsaacGym/Isaac Lab)
    - obs: joint angles + object pose + contact forces (privileged)
    - Domain randomization: physics, actuation, object shape

  Step 2: Student Policy (Distillation via BC / DAgger)
    - Input: RGB / depth / point cloud (real-sensor compatible)
    - Output: joint velocities or positions

  Step 3: Sim-to-Real Transfer
    - Zero-shot (domain randomization)
    - OR few-shot fine-tuning (SERNF/DEFT, <100 real episodes)

  Alternative (rising):
    - Teleoperation (DexUMI / Bunny-VisionPro) -> 50-200 demos
    - Diffusion Policy / ACT on demos -> deploy directly""")

# ============================================================
# 3
# ============================================================
pdf.stitle('3. Teleoperation & Data Collection')

pdf.paper('DexUMI: Human Hand as Universal Manipulation Interface',
          'CoRL 2025 | Stanford, Columbia, NVIDIA',
          'Human hand + exoskeleton + robot hand inpainting. 86% success across '
          'multiple platforms. Bridges embodiment gap via hardware adapter + '
          'visual inpainting. Key teleoperation system for dexterous demos.')

pdf.paper('Bunny-VisionPro: Real-Time Bimanual Dexterous Teleoperation',
          'ICRA 2024',
          'Apple Vision Pro for bimanual teleoperation. Dual 24-DoF system '
          '(xArm-7 + Ability hands). Outperforms prior teleop systems.')

pdf.paper('UMI: Universal Manipulation Interface',
          'ICRA 2024',
          'Hand-held gripper for portable data collection. Zero-shot generalization '
          'to novel objects/environments. Variants: MV-UMI, DexUMI, exUMI.')

pdf.paper('DexCap: Scalable Portable Mocap Data Collection',
          'RSS 2024',
          'EMF tracking gloves + RGB-D LiDAR. 3x faster than teleoperation. '
          '72% success from 30-minute mocap sessions. Backpack-mounted.')

pdf.paper('AnyTeleop: General Vision-Based Dexterous Teleoperation',
          'ICRA 2023',
          'Hardware-agnostic. Only needs RGB/RGB-D camera. Higher success than '
          'specialized systems in 8/10 tasks. Works over internet.')

# ============================================================
# 4
# ============================================================
pdf.stitle('4. RL & Policy Learning')

pdf.paper('BiDexHands: Human-Level Bimanual Dexterous Manipulation',
          'ICLR 2025 | PKU-MARL',
          'Dual Shadow Hands. 50+ bimanual tasks. IPPO teacher -> vision student. '
          'Isaac Gym. Most widely used dexterous RL benchmark.')

pdf.paper('Eureka: Human-Level Reward Design via LLM',
          'ICLR 2024',
          'GPT-4 auto-generates RL reward functions. Outperforms human-designed '
          'rewards on 83% of 29 tasks. 52% average improvement. Shadow Hand '
          'pen spinning tricks. Eliminates manual reward engineering.')

pdf.paper('DrEureka: LLM-Guided Sim-to-Real Transfer',
          'RSS 2024',
          'LLM automates domain randomization. Eliminates manual DR tuning. '
          'Validated on quadruped + dexterous manipulation. Builds on Eureka.')

pdf.paper('DextrAH-RGB: Visuomotor Policies to Grasp Anything',
          '2024',
          'Teacher (state RL, IPPO) -> Student (vision, DAgger). Isaac Sim. '
          'Zero-shot sim-to-real. Successful diverse object grasping.')

pdf.paper('SAPO: Stabilizing RL in Differentiable Multiphysics Sim',
          '2024',
          'Short-horizon autodiff (10-50 steps) + long-horizon RL (SAC). '
          'Addresses gradient exploding/vanishing. Most relevant to our '
          'P(F) differentiable sim approach.')

# ============================================================
# 5
# ============================================================
pdf.stitle('5. Diffusion & Imitation Learning')

pdf.paper('ALOHA Unleashed: Simple Recipe for Robot Dexterity',
          'Google DeepMind 2024',
          'Diffusion Policy on 26k+ demos. Shoelace tying, shirt hanging, gear '
          'insertion. Key insight: large-scale data + diffusion = complex dexterity. '
          '$32K bimanual hardware. Open-source.')

pdf.paper('DexMV: Learning from Human Videos',
          'ECCV 2022 | Google',
          'YouTube videos -> hand pose extraction -> robot demos. 83% success on '
          'real Allegro Hand. Eliminates need for VR/teleoperation.')

pdf.paper('UniDexFPM: Dexterous Functional Pre-Grasp Manipulation',
          '2024',
          'Mixture-of-experts -> diffusion policy. 72.6% across 30+ object '
          'categories. Isaac Gym. Pre-grasp reorientation before grasping.')

pdf.paper('Tilde: Diffusion + DAgger for Dexterous Manipulation',
          'RSS 2024',
          'Diffusion policy + human DAgger corrections. DeltaHand (12-DoF). '
          '90% success across 7 tasks. Real-time expert corrections.')

pdf.paper('DexDeform: Dexterous Deformable Object Manipulation',
          'ICLR 2023',
          'Human demos + differentiable physics optimization. 6 deformable tasks '
          '(Play-Doh). Most relevant for deformable + dexterous combination.')

# ============================================================
# 6
# ============================================================
pdf.stitle('6. World Models & Foundation Models')

pdf.paper('DexWM: Dexterous Interaction World Model',
          '2025',
          '900+ hours human and robot data. Cross-embodiment transfer. Beats '
          'Diffusion Policy by 50%+ on Franka + Allegro. Zero-shot transfer.')

pdf.paper('EgoDex: Learning from Large-Scale Egocentric Video',
          'Apple 2025',
          '829 hours egocentric video. 194 household tasks. Apple Vision Pro '
          'capture. Largest dexterous manipulation dataset. Open-source.')

pdf.paper('HumanPlus: Humanoid Shadowing and Imitation',
          'CoRL 2024',
          '33-DoF humanoid with dual 6-DoF hands. Shoe wearing, warehouse '
          'unloading, typing. 60-100% success from ~40 demos.')

pdf.paper('OpenVLA: Open-Source Vision-Language-Action Model',
          '2024',
          '7B VLA. 970k real-world demos. 50%+ success. Fully open-source.')

pdf.paper('RoboPianist: Dexterous Piano Playing',
          'CoRL 2023',
          'Learns 150 piano pieces. Deep RL with bimanual anthropomorphic hands. '
          'Demonstrates extreme coordination. RP1M: 1M expert trajectories.')

# ============================================================
# 7
# ============================================================
pdf.stitle('7. Hardware Platforms')
w4 = [28, 18, 18, 82]
pdf.trow(['Hand', 'Cost', 'DOF', 'Notes'], bold=True, fill=True, widths=w4)
pdf.trow(['Shadow Hand', '$100K+', '24', 'RL research standard. UK ARIA $57M. OpenAI, DeepMind'], widths=w4)
pdf.trow(['Allegro Hand', '$16K', '16', 'DexMV, DexPoint, DexCap. Good middle ground'], widths=w4)
pdf.trow(['LEAP Hand', '$2K', '16', 'RSS 2024. 3D-printed. Outperforms Allegro. Open-source'], widths=w4)
pdf.trow(['RUKA v2', '<$2K', '15', 'NYU 2025. Tendon-driven. Fishing line. Open-source'], widths=w4)
pdf.trow(['ORCA Hand', '$1.5-6K', '17', 'ETH 2025. 10K+ cycles. Optional tactile. MIT license'], widths=w4)
pdf.trow(['Ability Hand', '$20K+', '10', 'Bunny-VisionPro. Commercial grade'], widths=w4)
pdf.trow(['DeltaHand', 'Research', '12', 'CMU. Non-anthropomorphic delta robot design'], widths=w4)
pdf.trow(['DexRobot 021', 'Commercial', '19', "CES 2026. 23 sensors. Rubik's cube solving"], widths=w4)

# ============================================================
# 8
# ============================================================
pdf.stitle('8. Surgical Dexterous Manipulation')

pdf.paper('Dexterity-Enhancing Handheld Robot for Endoscopic Neurosurgery',
          'npj Robotics 2025 (first in-vivo study)',
          'Panda Surgical handheld robot. Interchangeable end-effectors (forceps, '
          'curettes, dissectors). 2-3 DoF per tool. First clinical validation '
          'for dexterous neurosurgery. Bimanual without brain compression.')

pdf.paper('SutureBot: Autonomous End-to-End Surgical Suturing',
          '2025 (arXiv:2510.20965)',
          'da Vinci Research Kit. 1,890 demos. Needle pickup, tissue insertion, '
          'knot tying. Imitation learning. First comprehensive suturing framework.')

pdf.paper('Dex Surgical: Commercial Surgical Robotics',
          'Commercial',
          'PrecisionDex 3D technology. 7-DoF dexterous laparoscopic tools. '
          'Minimally invasive surgery. Commercial market solution.')

pdf.bold_line('Gap: No existing work combines dexterous hand + CUSA tool + '
              'differentiable debulking simulation. This is our unique opportunity.')

# ============================================================
# 9
# ============================================================
pdf.stitle('9. Application: Dexterous CUSA Debulking')

pdf.code(
"""Proposed System: Dexterous Hand + CUSA Tool + P(F) Differentiable Sim

Hardware:
  - LEAP Hand ($2K) or Allegro Hand ($16K)
  - CUSA-like tool mounted on fingertips
  - RealSense depth camera for observation

Training Pipeline:
  Step 1: Teacher Policy (RL, PPO/SAC)
    - Sim: Isaac Lab (hand) + Warp (P(F) tissue)
    - obs: hand joints + CUSA tip + local P(F) distribution
    - action: hand joint velocities
    - reward: +volume_removed - lambda*nerve_damage(P(F))

  Step 2: Student Policy (vision-based)
    - Teacher -> Student via DAgger
    - OR Diffusion Policy on teleop demos

  Step 3: Sim-to-Real
    - Domain randomization on tissue properties
    - Short-horizon autodiff for local refinement
    - Transfer to LEAP Hand + silicone phantom

Paper Framing:
  "Differentiable Geometry-Aware Debulking with Dexterous Manipulation:
   P(F)-guided RL for Vestibular Schwannoma Surgery"

Target Venues: CoRL, ICRA, RSS""")

# ============================================================
# 10
# ============================================================
pdf.stitle('10. Recommended Reading Priority')
w3r = [50, 25, 70]
pdf.trow(['Paper', 'Venue', 'Why Read'], bold=True, fill=True, widths=w3r)
pdf.trow(['SAPO (RL + DiffSim)', '2024', 'Our exact approach: short diff + long RL'], widths=w3r)
pdf.trow(['DexUMI', 'CoRL 2025', 'Best teleop for dexterous data collection'], widths=w3r)
pdf.trow(['BiDexHands', 'ICLR 2025', 'PPO benchmark for dexterous RL'], widths=w3r)
pdf.trow(['ALOHA Unleashed', 'DeepMind 2024', 'Diffusion policy for complex dexterity'], widths=w3r)
pdf.trow(['Eureka / DrEureka', 'ICLR24/RSS24', 'LLM auto reward + domain randomization'], widths=w3r)
pdf.trow(['DextrAH-RGB', '2024', 'Teacher-student sim-to-real pipeline'], widths=w3r)
pdf.trow(['LEAP Hand', 'RSS 2024', 'Low-cost hardware ($2K). Open-source'], widths=w3r)
pdf.trow(['DexWM', '2025', 'Cross-embodiment world model from video'], widths=w3r)
pdf.trow(['DexDeform', 'ICLR 2023', 'Dexterous + deformable + diff physics'], widths=w3r)
pdf.trow(['Neurosurgery Robot', 'npj 2025', 'First clinical dexterous neurosurgery'], widths=w3r)
pdf.trow(['DexMV', 'ECCV 2022', 'YouTube video -> robot policy'], widths=w3r)
pdf.trow(['EgoDex', 'Apple 2025', '829hr dataset. Largest dexterous dataset'], widths=w3r)

out = '/home/yunxin/pf-shape-matching/Dexterous_Hand_Survey.pdf'
pdf.output(out)
print(f'PDF generated: {out}')
