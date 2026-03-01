import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

def plot_inspiration():
    # --- Setup and Global Styling ---
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.titlepad'] = 0
    bg_color = 'white'

    # fig = plt.figure(figsize=(12, 7), facecolor=bg_color)
    # gs = GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.1)

    # --- PANEL 1. DECOMPOSE: Sparse KAN (4-6-6-1) ---
    fig1 = plt.figure(figsize=(6, 5), facecolor=bg_color)
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.set_facecolor(bg_color)
    ax1.axis('off')

    # Define Architecture: 4 input, 6 hidden, 6 hidden, 1 output
    layers = [4, 6, 1]
    pos = {}
    for i, nodes in enumerate(layers):
        for j in range(nodes):
            # Calculate y-offset to center the layers
            pos[(i, j)] = (i, j - nodes/2.0 + 0.5)

    # Highlight specific "Active" paths for the visual
    # Key: (layer_idx, from_node, to_node)
    active_edges = {
        (0, 0, 5): 'steelblue', (0,3,5): 'steelblue', (1,5,0): 'steelblue',
        (0, 0, 3): 'forestgreen', (0, 1, 3): 'forestgreen', (0, 3, 3): 'forestgreen', # Path 2
        (1,3,0): 'forestgreen',
        (0, 0, 0): 'darkorange',  (0,1,0): 'darkorange',  (0,3,0): 'darkorange',   # Path 3
        (1,0,0):'darkorange',
    }

    # 1. Draw ALL edges (Fully Connected Layer)
    for i in range(len(layers) - 1):
        for j in range(layers[i]):
            for k in range(layers[i+1]):
                key = (i, j, k)
                if key in active_edges:
                    # Highlighted active paths
                    color = active_edges[key]
                    ax1.plot([pos[(i, j)][0], pos[(i+1, k)][0]],
                             [pos[(i, j)][1], pos[(i+1, k)][1]],
                             color=color, alpha=0.7, linestyle='-', linewidth=2.5, zorder=2)
                else:
                    # Background full connectivity
                    ax1.plot([pos[(i, j)][0], pos[(i+1, k)][0]],
                             [pos[(i, j)][1], pos[(i+1, k)][1]],
                             color='grey', alpha=0.8, linestyle='--', linewidth=1.8, zorder=1)

    # 2. Draw Nodes
    for node, p in pos.items():
        layer_idx, node_idx = node
        # Default node style
        face_color = 'white'
        edge_color = '#b0b0b0'

        # Color specific input/hidden nodes to match the active paths
        if layer_idx == 0:
            if node_idx == 0: face_color = 'steelblue'
            elif node_idx == 1: face_color = 'forestgreen'
            elif node_idx == 3: face_color = 'steelblue'
        elif layer_idx == 1:
            if node_idx == 0: face_color = 'darkorange'
            elif node_idx == 3: face_color = 'forestgreen'
            elif node_idx == 5: face_color = 'steelblue'
        elif layer_idx == 2:
            if node_idx == 3: face_color = 'forestgreen'
        elif layer_idx == 3: # Final Output Node
            face_color = 'gray'

        if face_color != 'white': edge_color = 'black'

        ax1.scatter(p[0], p[1], s=140, c=face_color, edgecolors=edge_color, zorder=3, linewidth=1)

    # dummy_ax1 = fig.add_subplot(gs[0, 0], frameon=False, xticks=[], yticks=[])
    # add_panel_title(dummy_ax1, "1. Decompose: Sparse KAN")
    # add_caption(dummy_ax1, "High-dimensional problem broken into independent\n1D univariate functions.")
    fig1.savefig('./panel_1.png', dpi=300, bbox_inches='tight')

    # --- PANEL 2. EXTRACT: MD Activations ---
    fig2 = plt.figure(figsize=(9, 3), facecolor=bg_color)
    inner_gs = fig2.add_gridspec(1, 3, wspace=0.3)
    ax2_1 = fig2.add_subplot(inner_gs[0, 0], projection='3d')
    ax2_2 = fig2.add_subplot(inner_gs[0, 1], projection='3d')
    ax2_3 = fig2.add_subplot(inner_gs[0, 2], projection='3d')

    # Function 1: 2 variables, one high weight, one low weight (z = sin(3.14*x1) + 0.01*exp(x2))
    x1, x2 = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
    z1 = np.sin(3.14 * x1) + 0.01 * np.exp(x2)
    ax2_1.plot_surface(x1, x2, z1, cmap='viridis', edgecolor='none', alpha=0.8)

    # Function 2: 3 variables (x, y, z as inputs, color as output)
    n_pts = 100
    x, y, z = np.random.uniform(-1, 1, (3, n_pts))
    w2 = np.sin(np.pi * x) + np.cos(np.pi * y) + z**2
    ax2_2.scatter(x, y, z, c=w2, cmap='plasma', s=20, alpha=0.6)

    # Function 3: 3 variables (x, y, z as inputs, color as output)
    w3 = np.exp(x) * np.sin(np.pi * y) + np.log(z + 2)
    ax2_3.scatter(x, y, z, c=w3, cmap='inferno', s=20, alpha=0.6)

    for ax in [ax2_1, ax2_2, ax2_3]:
        ax.set_facecolor(bg_color)
        # ax.axis('off') # Keep axis as requested
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.tick_params(labelsize=8)

    # Empty helper for the main panel title/caption
    # dummy_ax2 = fig.add_subplot(gs[0, 1], frameon=False, xticks=[], yticks=[])
    # add_panel_title(dummy_ax2, "2. Extract: 1D Activations")
    # add_caption(dummy_ax2, "Learned splines are extracted as individual datasets.")
    fig2.savefig('./panel_2.png', dpi=300, bbox_inches='tight')

    # --- PANEL 3. DISTILL: PySR ---
    fig3 = plt.figure(figsize=(6, 5), facecolor=bg_color)
    ax3 = fig3.add_subplot(1, 1, 1)
    ax3.set_facecolor(bg_color)
    x_p = np.linspace(-1, 1, 100)
    y_p = np.sin(3.14 * x_p)
    ax3.plot(x_p, y_p, color='red', linewidth=2.5)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # dummy_ax3 = fig.add_subplot(gs[1, 0], frameon=False, xticks=[], yticks=[])
    # add_panel_title(dummy_ax3, "3. Distill: PySR")
    # add_caption(dummy_ax3, "PySR finds the exact mathematical form for each spline.")
    fig3.savefig('./panel_3.png', dpi=300, bbox_inches='tight')

    # --- PANEL 4. REASSEMBLE: Hybrid Model ---
    fig4 = plt.figure(figsize=(18, 10), facecolor=bg_color)
    ax4 = fig4.add_subplot(1, 1, 1, projection='3d')
    ax4.set_facecolor(bg_color)
    ax4.view_init(elev=0, azim=0) # Horizontal and vertical view
    ax4.set_box_aspect((2, 1, 0.7), zoom=1.1)
    X, Y = np.meshgrid(np.linspace(-2, 2, 50), np.linspace(-2, 2, 50))
    # Complicated function: Z = sin(3.14*X) * cos(Y) + exp(-(X**2 + Y**2))
    Z = np.sin(3.14 * X) * np.cos(Y) + np.exp(-(X**2 + Y**2))
    ax4.plot_surface(X, Y, Z, cmap='magma', edgecolor='black', linewidth=0.1, alpha=0.8)
    # ax4.set_title(r"$f(x) = [\sin(3.14*x_1) + 0.1] + [e^{2*x_2}] + [0.5*x_3]$"+"\n", fontsize=14, pad=-10)

    # Clean up 3D background
    ax4.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax4.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax4.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Make ticks more compact
    ax4.tick_params(axis='x', pad=-5)
    ax4.tick_params(axis='y', pad=-5)
    ax4.tick_params(axis='z', pad=-5)

    # dummy_ax4 = fig.add_subplot(gs[1, 1], frameon=False, xticks=[], yticks=[])
    # add_panel_title(dummy_ax4, "4. Reassemble: Hybrid Model")
    # add_caption(dummy_ax4, "Final explainable, high-performance model.")
    fig4.savefig('./panel_4.png', dpi=300, bbox_inches='tight')

    # plt.tight_layout()
    # fig.subplots_adjust(bottom=0.1, top=0.95, left=0.03, right=0.98)
    # plt.savefig('./test.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_kan_pysr():
    years = [8,10,14, 16,20,23,27]
    kan = [0.8032456418098101,0.6462173829279889,0.5532327497737577,
           0.427730423213436,
           0.3816877828131088,
           0.3808706325058055,
           0.3440053603833246,
           ]
    pysr = [0.18,0.37,0.15,0.15,0,0.01,0.01]

    plt.figure(figsize=(6, 4))
    plt.plot(years, kan, marker='o', label='KAN')
    plt.plot(years, pysr, marker='s', label='DeepPySR')
    plt.xlabel('Age in years')
    plt.ylabel('R2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('KAN vs DeepPySR R2 Performance using Real World Data')
    plt.savefig('./kan_pysr_r2.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_kan_pysr()
    plot_inspiration()