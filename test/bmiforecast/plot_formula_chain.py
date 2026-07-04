"""Generate a formula chain visualization for the BMI forecast rolling pipeline.

Produces: results_bmiforecast/bmiforecast_formula_chain.png
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Formula chain (DeepPySR Interpretable, BMI Forecast source, best R²) ────
AGES = [8, 10, 14, 17, 20, 23, 27]

# Each entry: (age_label, r2, prior_bmi_var, formula_display, covariate_terms)
# formula_display uses LaTeX-like shorthand for readability
STEPS = [
    {
        'age': 'Age 8',
        'r2': None,
        'prior': None,
        'formula_lines': ['y₈ BMI  (observed / predicted)'],
        'terms': {
            'prior_bmi': [],
            'genetics': [],
            'covariates': [],
        },
        'is_input': True,
    },
    {
        'age': 'Age 10',
        'r2': 0.810,
        'prior': 'y₈ BMI',
        'formula_lines': [
            'BMI₁₀ = 1.17 × y₈BMI',
            '      + PGS002313',
            '      + f(prepreg_cig, weight_who_ga,',
            '           smk_cig_p, childcareprof, preg_alc)',
        ],
        'terms': {
            'prior_bmi': ['y₈BMI'],
            'genetics': ['PGS002313'],
            'covariates': ['prepreg_cig', 'weight_who_ga', 'smk_cig_p', 'childcareprof', 'preg_alc'],
        },
    },
    {
        'age': 'Age 14',
        'r2': 0.765,
        'prior': 'y₁₀ BMI',
        'formula_lines': [
            'BMI₁₄ = y₁₀BMI',
            '      + (occup_f + smk_exp₂)',
            '        / (mht_wt + PGS002161 × b_midarm)',
            '      + 2.05',
        ],
        'terms': {
            'prior_bmi': ['y₁₀BMI'],
            'genetics': ['PGS002161'],
            'covariates': ['occup_f', 'smk_exp2', 'mht_wt', 'b_midarm', 'breastfed_excl'],
        },
    },
    {
        'age': 'Age 17',
        'r2': 0.770,
        'prior': 'y₁₄ BMI',
        'formula_lines': [
            'BMI₁₇ = 0.90 × y₁₄BMI',
            '      + 0.90 × cond(y₅_bmiz − 1.09,',
            '           edu_f + sin(prepreg_wt)',
            '           × exp(preg_cig))',
            '      + 3.67',
        ],
        'terms': {
            'prior_bmi': ['y₁₄BMI'],
            'genetics': [],
            'covariates': ['y5_bmiz', 'edu_f', 'prepreg_weight', 'smk_exp2', 'preg_cig', 'agebirth_m'],
        },
    },
    {
        'age': 'Age 20',
        'r2': 0.742,
        'prior': 'y₁₇ BMI',
        'formula_lines': [
            'BMI₂₀ = y₁₇BMI',
            '      − 0.4 × y₅_a10 × smk_t3',
            '        × sin(fam_splitup + f(birth_year,',
            '             breastfed_any, childcarerel))',
            '      + 0.4 × miggen_child',
        ],
        'terms': {
            'prior_bmi': ['y₁₇BMI'],
            'genetics': [],
            'covariates': ['y5_a10', 'smk_t3', 'fam_splitup', 'birth_year', 'breastfed_any',
                           'childcarerel', 'miggen_child'],
        },
    },
    {
        'age': 'Age 23',
        'r2': 0.864,
        'prior': 'y₂₀ BMI',
        'formula_lines': [
            'BMI₂₃ = PGS002853',
            '      + (0.94 + 0.18×sin(dogs×preg_cig))',
            '        × (y₂₀BMI − y1_a10/2 − fam_splitup)',
            '      + 4.41',
        ],
        'terms': {
            'prior_bmi': ['y₂₀BMI'],
            'genetics': ['PGS002853'],
            'covariates': ['dogs_quant_preg', 'preg_cig', 'y1_a10', 'fam_splitup1'],
        },
    },
    {
        'age': 'Age 27',
        'r2': 0.760,
        'prior': 'y₂₃ BMI',
        'formula_lines': [
            'BMI₂₇ = 0.90 × y₂₃BMI',
            '      + y₅_a9×sin(y₅_bmiz)',
            '        / (46.0 − y₂₃BMI)',
            '      + PGS002313 + log(m_wt1)',
        ],
        'terms': {
            'prior_bmi': ['y₂₃BMI'],
            'genetics': ['PGS002313'],
            'covariates': ['y5_a9', 'y5_bmiz', 'preg_alc_unit', 'parity_m', 'm_wt1'],
        },
    },
]

# ── Colour palette ───────────────────────────────────────────────────────────
C_INPUT   = '#d0e8f7'   # light blue — input node
C_PRIOR   = '#2196F3'   # blue      — prior BMI arrow / highlight
C_GEN     = '#4CAF50'   # green     — PRS / genetics
C_COV     = '#FF9800'   # orange    — covariates
C_BOX     = '#F5F5F5'   # light grey — formula boxes
C_EDGE    = '#9E9E9E'   # grey      — box border
C_ARROW   = '#546E7A'   # dark grey — arrows
C_AGE     = '#37474F'   # dark      — age label
C_R2      = '#6D4C41'   # brown     — R² annotation


def make_legend():
    return [
        mpatches.Patch(facecolor=C_PRIOR,  label='Prior predicted BMI (rolling input)'),
        mpatches.Patch(facecolor=C_GEN,    label='Polygenic risk score (PRS)'),
        mpatches.Patch(facecolor=C_COV,    label='Clinical / environmental covariates'),
    ]


def draw_formula_chain(out_path):
    n = len(STEPS)
    legend_h  = 0.90   # space reserved below last box for legend
    fig_h = 1.90 + n * 1.55 + legend_h
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, fig_h)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    box_x0, box_w = 1.0, 8.0
    box_h_input = 0.58
    box_h_step  = 1.45
    y_gap       = 0.14
    y_arrow     = 0.22

    y_cursor = fig_h - 0.75   # start from top (below title+subtitle)

    for i, step in enumerate(STEPS):
        is_input = step.get('is_input', False)
        bh = box_h_input if is_input else box_h_step

        box_y = y_cursor - bh

        # Draw box
        fc = C_INPUT if is_input else C_BOX
        rect = FancyBboxPatch(
            (box_x0, box_y), box_w, bh,
            boxstyle='round,pad=0.05',
            linewidth=1.4,
            edgecolor=C_PRIOR if is_input else C_EDGE,
            facecolor=fc,
            zorder=2,
        )
        ax.add_patch(rect)

        # Age label (left, bold)
        ax.text(
            box_x0 + 0.15, box_y + bh / 2,
            step['age'],
            ha='left', va='center',
            fontsize=11, fontweight='bold', color=C_AGE,
            zorder=3,
        )

        if is_input:
            ax.text(
                box_x0 + box_w / 2, box_y + bh / 2,
                step['formula_lines'][0],
                ha='center', va='center',
                fontsize=11, color='#1565C0', style='italic',
                zorder=3,
            )
        else:
            # R² annotation (right side)
            if step['r2'] is not None:
                ax.text(
                    box_x0 + box_w - 0.12,
                    box_y + bh - 0.10,
                    f"R² = {step['r2']:.3f}",
                    ha='right', va='top',
                    fontsize=9.5, color=C_R2,
                    zorder=3,
                )

            # Formula lines
            formula_lines = step['formula_lines']
            n_lines = len(formula_lines)
            line_h = (bh - 0.20) / n_lines
            text_x = box_x0 + 1.10
            for li, line in enumerate(formula_lines):
                ly = box_y + bh - 0.16 - li * line_h - line_h * 0.5
                ax.text(
                    text_x, ly, line,
                    ha='left', va='center',
                    fontsize=9.5,
                    color='#1565C0' if (li == 0 and 'BMI' in line) else '#333333',
                    fontweight='semibold' if (li == 0) else 'normal',
                    zorder=3,
                    family='monospace',
                )

            # Coloured term dots (bottom right area)
            dot_y = box_y + 0.13
            dot_x = box_x0 + box_w - 0.20
            terms = step['terms']
            dot_items = []
            if terms['prior_bmi']:
                dot_items.append((C_PRIOR, 'Prior BMI'))
            if terms['genetics']:
                dot_items.append((C_GEN, 'PRS'))
            if terms['covariates']:
                dot_items.append((C_COV, 'Covariates'))
            for di, (dc, _dl) in enumerate(dot_items):
                ax.add_patch(plt.Circle(
                    (dot_x - di * 0.30, dot_y + 0.06),
                    0.08, color=dc, zorder=4,
                ))

        y_cursor = box_y

        # Arrow: from bottom of current box to top of next box
        if i < n - 1:
            next_bh = box_h_input if STEPS[i + 1].get('is_input', False) else box_h_step
            next_box_top = box_y - y_arrow - y_gap
            ax.annotate(
                '',
                xy=(box_x0 + box_w / 2, next_box_top),
                xytext=(box_x0 + box_w / 2, box_y),
                arrowprops=dict(
                    arrowstyle='->', color=C_ARROW,
                    lw=1.8,
                    connectionstyle='arc3,rad=0.0',
                ),
                zorder=3,
            )
            y_cursor -= (y_arrow + y_gap)

    # Legend placed just below the last box
    last_box_bottom = y_cursor - 0.12
    legend_y = last_box_bottom - 0.12   # data coords, top of legend area

    legend_handles = make_legend()
    legend_labels  = [h.get_label() for h in legend_handles]
    swatch_w, swatch_h = 0.32, 0.16
    row_h   = 0.25
    lx = box_x0 + 0.15   # left-align with box edge

    for ci, (handle, label) in enumerate(zip(legend_handles, legend_labels)):
        y = legend_y - ci * row_h
        ax.add_patch(plt.Rectangle(
            (lx, y - swatch_h / 2), swatch_w, swatch_h,
            color=handle.get_facecolor(), zorder=5,
        ))
        ax.text(lx + swatch_w + 0.14, y, label,
                va='center', ha='left', fontsize=9.5, color='#333333', zorder=5)

    # Title
    ax.text(
        5.0, fig_h - 0.12,
        'DeepPySR Rolling BMI Forecast - Formula Chain (Interpretable Model)',
        ha='center', va='top',
        fontsize=12, fontweight='bold', color='#263238',
    )

    # Subtitle
    ax.text(
        5.0, fig_h - 0.42,
        "Each step takes the prior age's predicted BMI as input, "
        'combined with PRS and covariates fixed at baseline.',
        ha='center', va='top',
        fontsize=9, color='#546E7A', style='italic',
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, 'results_bmiforecast')
    os.makedirs(out_dir, exist_ok=True)
    draw_formula_chain(os.path.join(out_dir, 'bmiforecast_formula_chain.png'))
