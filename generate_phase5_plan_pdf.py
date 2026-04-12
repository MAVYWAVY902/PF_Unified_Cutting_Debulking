"""Generate Phase 5 Plan PDF (Chinese) for P(F) SAPO Training."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

# Try to register CJK font
try:
    pdfmetrics.registerFont(TTFont('SimSun', '/usr/share/fonts/truetype/arphic/uming.ttc'))
    CJK_FONT = 'SimSun'
except:
    try:
        pdfmetrics.registerFont(TTFont('NotoSansCJK', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'))
        CJK_FONT = 'NotoSansCJK'
    except:
        CJK_FONT = 'Helvetica'

OUTPUT = "Phase5_SAPO_Training_Plan.pdf"


def build_pdf():
    doc = SimpleDocTemplate(OUTPUT, pagesize=A4,
                            topMargin=20*mm, bottomMargin=15*mm,
                            leftMargin=18*mm, rightMargin=18*mm)

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle('CTitle', parent=styles['Title'],
                                  fontName=CJK_FONT, fontSize=20, leading=26,
                                  textColor=colors.HexColor('#1a5276'))
    h1 = ParagraphStyle('CH1', parent=styles['Heading1'],
                         fontName=CJK_FONT, fontSize=14, leading=20,
                         textColor=colors.HexColor('#1a5276'),
                         spaceBefore=14, spaceAfter=6)
    h2 = ParagraphStyle('CH2', parent=styles['Heading2'],
                         fontName=CJK_FONT, fontSize=12, leading=16,
                         textColor=colors.HexColor('#2e86c1'),
                         spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle('CBody', parent=styles['Normal'],
                           fontName=CJK_FONT, fontSize=10, leading=15,
                           spaceBefore=3, spaceAfter=3)
    code = ParagraphStyle('Code', parent=styles['Code'],
                           fontName='Courier', fontSize=8.5, leading=12,
                           leftIndent=10, backColor=colors.HexColor('#f4f4f4'))
    small = ParagraphStyle('Small', parent=body, fontSize=8.5, leading=12,
                            textColor=colors.HexColor('#555555'))

    def tbl(data, col_widths=None):
        t = Table(data, colWidths=col_widths, hAlign='LEFT')
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2e86c1')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), CJK_FONT),
            ('FONTNAME', (0, 1), (-1, -1), CJK_FONT),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f8ff')]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        return t

    story = []

    # ---- Cover ----
    story.append(Spacer(1, 30*mm))
    story.append(Paragraph("P(F) Shape Matching", title_style))
    story.append(Paragraph("Phase 5: SAPO Training Plan", title_style))
    story.append(Spacer(1, 8*mm))
    story.append(Paragraph("Differentiable Surgical Debulking Simulation + Policy Learning", body))
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph("CUSA Debulking: From Differentiable Sim to Real Robot", small))
    story.append(Spacer(1, 15*mm))
    story.append(HRFlowable(width="80%", color=colors.HexColor('#2e86c1')))
    story.append(Spacer(1, 5*mm))
    story.append(Paragraph("Target: CoRL 2026 (~Jun 2026)", body))
    story.append(Paragraph("Method: SAPO (Soft Analytic Policy Optimization, ICLR 2025)", body))
    story.append(Paragraph("Platform: NVIDIA Warp + PyTorch", body))
    story.append(PageBreak())

    # ---- 1. Project Status ----
    story.append(Paragraph("1. Project Status", h1))
    story.append(Paragraph("Phase 1-4 completed. Warp-based P(F) solver is functional and differentiable.", body))
    story.append(tbl([
        ["Phase", "Content", "Status"],
        ["Phase 1", "Mesh loading + Warp array init", "DONE"],
        ["Phase 2", "Core Shape Matching (SVD)", "DONE"],
        ["Phase 3", "P(F) Damage + Soft Removal", "DONE"],
        ["Phase 4", "Autodiff Verification", "DONE"],
        ["Phase 5", "SAPO Training Env + Policy", "IN PROGRESS"],
        ["Phase 6", "Paper Experiments", "TODO"],
    ], col_widths=[60, 200, 80]))
    story.append(Spacer(1, 5*mm))
    story.append(Paragraph("Phase 4 verified: 5-step gradient flows through knife_pos -> edge_damage -> particle_damage -> alpha -> loss. Finite difference matches analytic direction.", body))

    # ---- 2. Training Method ----
    story.append(Paragraph("2. Training Method: SAPO", h1))
    story.append(Paragraph("SAPO (ICLR 2025) = short-horizon analytic gradient + actor-critic RL + max entropy.", body))
    story.append(Paragraph("2.1 SAPO vs Alternatives", h2))
    story.append(tbl([
        ["Dimension", "SAPO", "SHAC", "PPO"],
        ["Gradient source", "Diff sim (analytic)", "Diff sim (analytic)", "Reward sampling"],
        ["Backprop steps", "10-50", "10-50", "0"],
        ["Sample efficiency", "17x vs PPO", "17x vs PPO", "1x (baseline)"],
        ["Deformable support", "Yes (Rewarped)", "Rigid only", "Any sim"],
        ["Stability", "High (entropy)", "Medium", "Very high"],
        ["Exploration", "Max-entropy auto", "None (collapse risk)", "Clip ratio"],
    ], col_widths=[90, 110, 100, 90]))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph("2.2 SAPO Algorithm", h2))
    story.append(Paragraph(
        "For each epoch:<br/>"
        "1. obs = env.initialize_trajectory() -- reset, cut gradient<br/>"
        "2. Record H=32 steps inside wp.Tape():<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;action ~ pi_theta(obs) via reparameterization<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;obs, rew, done = env.step(action) -- differentiable physics<br/>"
        "3. tape.backward() -- analytic gradient through sim<br/>"
        "4. Update actor with policy gradient (analytic, not REINFORCE)<br/>"
        "5. Update critic via TD(lambda) MSE<br/>"
        "6. Update entropy temperature alpha", body))

    # ---- 3. System Architecture ----
    story.append(Paragraph("3. System Architecture", h1))
    story.append(Paragraph(
        "Action (3D velocity) -> apply_action_kernel -> knife_pos_new<br/>"
        "&nbsp;&nbsp;-> solver._substep() x sim_substeps<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;-> predict_kernel (gravity + damping)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;-> collision_kernel (floor + knife sphere SDF)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;-> cluster_solve_kernel (SVD shape matching)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;-> apply_kernel (stiffness correction)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;-> update_edge_damage_kernel (P(F) sigmoid)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;-> update_particle_damage_kernel (atomic_max)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;-> update_soft_removal_kernel (sigmoid alpha)<br/>"
        "&nbsp;&nbsp;-> compute_obs kernels -> obs_buf (7D)<br/>"
        "&nbsp;&nbsp;-> compute_reward_kernel -> rew_buf", code))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph("All operations inside wp.Tape(). Gradient flows: reward -> alpha -> damage -> knife_pos -> action -> policy parameters.", body))

    # ---- 4. Observation / Action / Reward ----
    story.append(Paragraph("4. Observation / Action / Reward", h1))
    story.append(Paragraph("4.1 Observation (7D)", h2))
    story.append(tbl([
        ["Dim", "Name", "Source", "Differentiable"],
        ["0-2", "ee_pos (x,y,z)", "knife_pos array", "Yes"],
        ["3", "local_pf_mean", "Mean damage near knife (3x radius)", "Yes"],
        ["4", "volume_removed", "Sum(1-alpha) for non-fixed particles", "Yes"],
        ["5", "nerve_damage", "Sum(damage) on nerve-labeled particles", "Yes"],
        ["6", "dist_to_nerve", "Min distance to nerve particle", "No (info only)"],
    ], col_widths=[30, 100, 170, 75]))
    story.append(Paragraph("4.2 Action (3D)", h2))
    story.append(Paragraph("End-effector velocity (dx, dy, dz). Applied via: knife_pos_new = knife_pos + action * action_scale. Bounded by tanh in actor network.", body))
    story.append(Paragraph("4.3 Reward", h2))
    story.append(Paragraph(
        "r = +delta_volume_removed - lambda * delta_nerve_damage - mu * ||action||^2 + bonus * sigmoid(V - target)<br/>"
        "Default: lambda=5.0, mu=0.01, bonus=10.0, target=50 particles", body))

    # ---- 5. Hyperparameters ----
    story.append(PageBreak())
    story.append(Paragraph("5. SAPO Hyperparameters", h1))
    story.append(tbl([
        ["Parameter", "Value", "Note"],
        ["horizon_len", "32", "SAPO/SHAC default; Phase 4 verified 5 steps"],
        ["sim_substeps", "4", "Physics substeps per env.step()"],
        ["actor_lr", "2e-3", "AdamW optimizer"],
        ["critic_lr", "5e-4", "AdamW optimizer"],
        ["beta1", "0.7", "Lower than default 0.9 for analytic gradients"],
        ["gamma", "0.99", "Discount factor"],
        ["lambda (TD)", "0.95", "TD(lambda) for critic targets"],
        ["max_grad_norm", "0.5", "Gradient clipping (SAPO uses 0.5)"],
        ["critic_iterations", "16", "Critic updates per epoch"],
        ["no_target_critic", "True", "SAPO key: no target network"],
        ["init_alpha", "1.0", "Entropy temperature initial value"],
        ["target_entropy", "-1.5", "-dim(A)/2 = -3/2"],
        ["hidden_dim", "64", "Actor/critic MLP hidden size"],
        ["activation", "SiLU", "Smooth activation (not ReLU/ELU)"],
        ["normalization", "LayerNorm", "On each hidden layer"],
    ], col_widths=[100, 60, 220]))

    # ---- 6. Key Implementation Details ----
    story.append(Paragraph("6. Key Implementation Details", h1))
    story.append(Paragraph("6.1 knife_pos Differentiability (Critical Fix)", h2))
    story.append(Paragraph(
        "knife_pos changed from scalar wp.vec3 to wp.array(dtype=wp.vec3, shape=(1,), requires_grad=True). "
        "This enables gradient flow through collision_kernel and update_edge_damage_kernel. "
        "All kernel signatures updated to read knife_pos[0] from array.", body))
    story.append(Paragraph("6.2 Particle Damage Atomic Max", h2))
    story.append(Paragraph(
        "update_particle_damage_kernel uses int-scaled wp.atomic_max (3-pass: float->int, atomic_max, int->float). "
        "This is NOT differentiable, but gradient still flows through edge_damage -> soft_removal sigmoid.", body))
    story.append(Paragraph("6.3 PyTorch-Warp Gradient Bridge", h2))
    story.append(Paragraph(
        "tape.backward() computes d(reward)/d(action) for each timestep. "
        "These analytic gradients are transferred to PyTorch via numpy and used as "
        "the policy gradient signal (replacing REINFORCE's score function).", body))

    # ---- 7. File Structure ----
    story.append(Paragraph("7. File Structure", h1))
    story.append(tbl([
        ["File", "Description"],
        ["warp_sim/envs/debulk_env.py", "SAPO-compatible Gym environment"],
        ["warp_sim/kernels/obs_reward.py", "Observation + reward Warp kernels"],
        ["warp_sim/train_sapo.py", "SAPO training entry point"],
        ["warp_sim/config/debulk_sapo.yaml", "Training hyperparameters"],
        ["warp_sim/test_env.py", "Environment smoke test"],
        ["warp_sim/test_gradient_32step.py", "32-step gradient stability test"],
        ["pf_shape_matching_warp/sim/solver.py", "P(F) solver (knife_pos refactored)"],
        ["pf_shape_matching_warp/sim/kernels.py", "PBD kernels (knife_pos array)"],
        ["pf_shape_matching_warp/sim/pf_damage.py", "Damage kernels (knife_pos array)"],
    ], col_widths=[220, 200]))

    # ---- 8. Timeline ----
    story.append(Paragraph("8. Timeline", h1))
    story.append(tbl([
        ["Week", "Task", "Deliverable"],
        ["1-2", "DebulkEnv + gradient verification", "test_env + test_gradient_32step pass"],
        ["3-4", "SAPO training + hyperparameter tuning", "Policy learns volume up + nerve down"],
        ["5", "Ablation (SAPO vs PPO vs SAC)", "Sample efficiency comparison"],
        ["6", "Visualization + paper figures", "Polyscope trajectory + Pareto curves"],
        ["7-8", "Paper writing -> CoRL 2026", "Complete draft"],
    ], col_widths=[40, 200, 180]))

    # ---- 9. Risks ----
    story.append(Paragraph("9. Risks and Mitigations", h1))
    story.append(tbl([
        ["Risk", "Impact", "Mitigation"],
        ["32-step gradient explosion", "Training fails", "Reduce to 16 steps / add checkpointing"],
        ["atomic_max breaks gradient", "Partial gradient", "Edge damage is differentiable; alpha sigmoid preserves flow"],
        ["PyTorch-Warp bridge", "Complex interop", "Use action.grad from Warp tape directly"],
        ["GPU memory (32 steps x 20k)", "OOM", "Reduce particles (5k) / reduce horizon"],
        ["Cutting interval too sparse", "No gradient signal", "Set cutting_interval=1"],
    ], col_widths=[120, 80, 210]))

    # ---- 10. Success Criteria ----
    story.append(Paragraph("10. Success Criteria", h1))
    story.append(Paragraph(
        "1. test_env.py passes (obs shape, step loop, reset)<br/>"
        "2. test_gradient_32step.py passes (grad_norm &lt; 1e4, no NaN/Inf)<br/>"
        "3. Training on cube_dense.msh: reward increases over 100 epochs<br/>"
        "4. Policy achieves volume_removed > 30, nerve_damage &lt; 5<br/>"
        "5. SAPO achieves 10x+ sample efficiency vs PPO baseline", body))

    doc.build(story)
    print(f"Generated: {OUTPUT}")


if __name__ == "__main__":
    build_pdf()
