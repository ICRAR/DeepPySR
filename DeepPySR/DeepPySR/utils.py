import os
import re
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import networkx as nx
from pycirclize import Circos

def ensure_output_dir(path: str = "outputs") -> str:
    os.makedirs(path, exist_ok=True)
    return path

def mse_and_r2(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    mse = float(np.mean((y_true - y_pred) ** 2))
    var = float(np.var(y_true))
    r2 = 1.0 - (mse / var) if var > 0 else float("nan")
    return mse, r2

def detect_basic_function_for_var(sym_expr: sp.Expr, var_symbol: sp.Symbol) -> str:
    s = str(sym_expr)
    v = str(var_symbol)
    tags = []
    for fn in ["sin", "cos", "exp", "log", "sqrt", "tanh", "square", "asin", "acos", "atanh"]:
        if re.search(rf"\b{fn}\({re.escape(v)}\)", s):
            tags.append(fn)
    if not tags:
        if re.search(rf"\b{re.escape(v)}\b", s):
            if re.search(rf"{re.escape(v)}\*{re.escape(v)}|{re.escape(v)}\^2|{re.escape(v)}\*\*2", s):
                return "nonlinear"
            return "linear/mixed"
        return "unused"
    return ",".join(tags)

def is_redundant(
        target_name: str,
        sym_expr: sp.Expr,
        all_relationships: list[dict],
        threshold: float = 1e-4
) -> bool:
    """
    Checks redundancy by evaluating the new expression against existing
    relationships over a vectorized numerical grid.
    """
    if not all_relationships:
        return False

    # 1. Expand the expression by substituting previous relationships
    # We do this numerically to avoid SymPy's symbolic bottleneck
    target_sym = sp.Symbol(target_name)
    combined_expr = target_sym - sym_expr

    # Sort relationships by layer to ensure we substitute 'deep' vars first
    sorted_rels = sorted(all_relationships, key=lambda r: r["layer"], reverse=True)

    # 2. Create a substitution map for all intermediate variables
    sub_map = {rel["target_symbol"]: rel["sympy"] for rel in sorted_rels}

    # 3. Identify truly independent variables (root inputs like x0, x1...)
    # Instead of full symbolic expansion, we lambdify the expression
    # with all discovered variables as potential inputs.
    all_involved = set(str(s) for s in combined_expr.free_symbols)
    for r in all_relationships:
        all_involved.update(r["involved"])

    # Remove the target itself and other intermediate symbols
    root_vars = sorted([v for v in all_involved if v.startswith("x") and v not in sub_map])
    root_symbols = [sp.Symbol(v) for v in root_vars]

    try:
        # Create a fast NumPy function
        # We use 'lambdify' on the final expanded form
        final_expr = combined_expr
        for _ in range(len(all_relationships)): # Iterative expansion
            new_expr = final_expr.subs(sub_map)
            if new_expr == final_expr: break
            final_expr = new_expr

        f = sp.lambdify(root_symbols, final_expr, modules="numpy")

        # 4. Generate high-volume test data (Vectorized)
        n_points = 5000
        test_data = [np.random.uniform(0.1, 1.0, n_points) for _ in root_symbols]

        # Evaluate
        y_pred = f(*test_data)

        # If the difference is near zero, the new relationship is already implied.
        if np.allclose(y_pred, 0, atol=threshold):
            return True

        # Also check if the expression itself expands to a constant (original behavior).
        # We check the standard deviation of its expansion.
        final_rhs = sym_expr
        for _ in range(len(all_relationships)):
            new_expr = final_rhs.subs(sub_map)
            if new_expr == final_rhs: break
            final_rhs = new_expr
        
        rhs_symbols = sorted([str(s) for s in final_rhs.free_symbols])
        if not rhs_symbols: # It's a symbolic constant
            return True
        
        f_rhs = sp.lambdify([sp.Symbol(s) for s in rhs_symbols], final_rhs, modules="numpy")
        rhs_test_data = [np.random.uniform(0.1, 1.0, n_points) for _ in rhs_symbols]
        if np.std(f_rhs(*rhs_test_data)) < threshold:
            return True

    except Exception:
        # Fallback to False if math errors occur (e.g. log of negative)
        return False

    return False

def plot_n_layer_graph(
    relationships: list[dict],
    save_path: str,
    feature_names: list[str] = None
):
    total_vars = len(feature_names) if feature_names else 0
    G = nx.DiGraph()
    node_data = {} 
    edges = [] 
    
    rel_map = {r["target"]: r for r in relationships}

    def add_nodes_recursive(target_name, parent_id, current_layer, max_layers):
        node_id = target_name
        
        if node_id not in node_data:
            rel = rel_map.get(target_name)
            formula = rel["formula"] if rel else ""
            node_data[node_id] = {
                "label": target_name,
                "layer": current_layer,
                "formula": formula
            }
        else:
            if current_layer > node_data[node_id]["layer"]:
                node_data[node_id]["layer"] = current_layer

        if parent_id:
            parent_rel = rel_map.get(node_data[parent_id]["label"])
            tag = ""
            if parent_rel:
                tag = detect_basic_function_for_var(parent_rel["sympy"], sp.Symbol(target_name))
            
            edge = (node_id, parent_id, tag)
            if edge not in edges:
                edges.append(edge)

        if current_layer < max_layers:
            rel = rel_map.get(target_name)
            if rel:
                for vname in rel["involved"]:
                    if vname != target_name:
                        add_nodes_recursive(vname, node_id, current_layer + 1, max_layers)

    max_rel_layer = max((r["layer"] for r in relationships), default=0)
    add_nodes_recursive("y", None, 0, max_rel_layer)

    layers = {}
    for nid, data in node_data.items():
        l = data["layer"]
        if l not in layers:
            layers[l] = []
        layers[l].append(nid)

    pos = {}
    max_l = max(layers.keys()) if layers else 0
    pos["y"] = np.array([0.0, 0.0])
    
    layout_tree_children = {} 
    visited = {"y"}
    queue = ["y"]
    while queue:
        parent = queue.pop(0)
        layout_tree_children[parent] = []
        children = [src for src, dst, tag in edges if dst == parent]
        for child in children:
            if child not in visited:
                visited.add(child)
                layout_tree_children[parent].append(child)
                queue.append(child)

    def assign_radial_pos(parent, start_angle, end_angle, radius_step=1.0):
        children = layout_tree_children.get(parent, [])
        if not children:
            return
        
        n_children = len(children)
        parent_l = node_data[parent]["layer"]
        child_radius = (parent_l + 1) * radius_step
        sector_width = end_angle - start_angle
        
        for i, child in enumerate(children):
            angle = start_angle + (i + 0.5) * (sector_width / n_children)
            x = child_radius * np.cos(angle)
            y = child_radius * np.sin(angle)
            pos[child] = np.array([x, y])
            
            child_start = start_angle + i * (sector_width / n_children)
            child_end = start_angle + (i + 1) * (sector_width / n_children)
            assign_radial_pos(child, child_start, child_end, radius_step)

    assign_radial_pos("y", 0, 2 * np.pi)

    for nid in node_data:
        if nid not in pos:
            l = node_data[nid]["layer"]
            pos[nid] = np.array([l, 0.0]) 

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
    ax_graph = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])

    colors = [
        "#FFD700", "#87CEEB", "#90EE90", "#FFB6C1", "#DDA0DD", 
        "#FFA07A", "#20B2AA", "#778899", "#B0C4DE", "#FFFFE0",
        "#F0E68C", "#E6E6FA", "#FFF0F5", "#FFE4E1", "#F0FFF0"
    ]
    for l in range(max_l + 1):
        nodelist = [nid for nid in node_data if node_data[nid]["layer"] == l]
        if not nodelist: continue
        color = colors[l % len(colors)]
        size = 1500 / (l + 1)**0.5
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color=color, 
                               node_size=size, alpha=0.9, ax=ax_graph, edgecolors="black", linewidths=1)
    
    labels = {nid: d["label"] for nid, d in node_data.items()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=11, font_weight="bold", ax=ax_graph)

    for src, dst, tag in edges:
        G.add_edge(src, dst, label=tag)
        
    for u, v in G.edges():
        rel_v = rel_map.get(v)
        score = rel_v["score"] if rel_v else 0.1
        lw = 0.5 + 2.0 * min(score, 5.0) / 5.0 
        
        ax_graph.annotate("",
                    xy=pos[v], xycoords='data',
                    xytext=pos[u], textcoords='data',
                    arrowprops=dict(arrowstyle="->", color="gray",
                                    shrinkA=15, shrinkB=15,
                                    patchA=None, patchB=None,
                                    connectionstyle="arc3,rad=0.2",
                                    mutation_scale=15,
                                    alpha=0.6,
                                    linewidth=lw),
                    )

    edge_labels = nx.get_edge_attributes(G, "label")
    edge_labels = {k: v for k, v in edge_labels.items() if v and v != "linear/mixed" and v != "unused"}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax_graph, label_pos=0.5, alpha=0.8)

    ax_graph.set_title(f"Symbolic Relationship Hierarchy (depth={max_l})", fontsize=14)
    ax_graph.axis("off")

    ax_table.axis("off")
    table_data = []
    
    def sort_key(v):
        if v == "y":
            return (0, 0, "")
        if v.startswith("x") and v[1:].isdigit():
            return (1, int(v[1:]), v)
        return (2, 0, v)

    unique_targets = sorted(list(set(d["label"] for d in node_data.values())), key=sort_key)
    
    for t in unique_targets:
        rel = rel_map.get(t)
        if rel:
            f_str = rel["formula"]
            if len(f_str) > 30:
                f_str = f_str[:27] + "..."
            table_data.append([t, rel["layer"], f"{rel['score']:.3g}", f_str])

    if table_data:
        table = ax_table.table(cellText=table_data, colLabels=["Target", "Layer", "Score", "Symbolic Formula"], 
                               loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.2)
        table.auto_set_column_width([0, 1, 2, 3])
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.get_text().set_weight('bold')
                cell.set_facecolor('#f2f2f2')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

def plot_circlize(
        relationships: list[dict],
        save_path: str,
        feature_names: list[str] = None
):
    # 1. Collect and sort all unique variables
    all_vars = {"y"}
    for rel in relationships:
        all_vars.add(rel["target"])
        for v in rel["involved"]:
            all_vars.add(v)

    # Add missing variables if feature_names is provided
    if feature_names:
        for name in feature_names:
            all_vars.add(name)

    # Sort: 'y' first, then x0, x1, etc.
    def sort_key(v):
        if v == "y":
            return (0, 0, "")
        if v.startswith("x") and v[1:].isdigit():
            return (1, int(v[1:]), v)
        return (2, 0, v)

    sorted_vars = sorted(list(all_vars), key=sort_key)

    # 2. Identify variables directly influencing 'y'
    y_rel = next((r for r in relationships if r["target"] == "y"), None)
    direct_to_y = set(y_rel["involved"]) if y_rel else set()
    
    # 3. Define layer colors
    max_layer = max((r["layer"] for r in relationships), default=1)
    cmap = plt.get_cmap("viridis", max_layer)
    layer_colors = {i+1: cmap(i) for i in range(max_layer)}

    # 4. Initialize pyCirclize Circos
    # We assign each variable a sector of equal size (10 units)
    sectors = {v: 10 for v in sorted_vars}
    circos = Circos(sectors, space=2) # Reduced space from 3 to 2 for compactness

    for sector in circos.sectors:
        # Create a track for the variable "dots" and labels
        # Track from radius 95 to 100
        node_track = sector.add_track((95, 100))

        v_name = sector.name
        # Determine styling
        if v_name == "y":
            color = "#D32F2F" # Professional Dark Red
            size = 120 # Reduced from 150
            zorder = 5
        elif v_name in direct_to_y:
            color = "#1976D2" # Professional Dark Blue
            size = 80 # Reduced from 100
            zorder = 4
        else:
            color = "#CFD8DC" # Subtle Gray-Blue
            size = 40 # Reduced from 50
            zorder = 3

        # Place the dot in the middle of the sector
        node_track.scatter([5], [97.5], color=color, s=size, edgecolors="black", lw=0.5, zorder=zorder)

        # Add labels outside the track
        # Reduced radius from 115 to 108 and font size from 10 to 9
        node_track.text(v_name, x=5, r=108, size=9, weight="bold")

    # 5. Draw Relationship Links
    for rel in relationships:
        target = rel["target"]
        score = rel["score"]
        layer = rel["layer"]
        color = layer_colors.get(layer, "black")

        # Scale line thickness by score (clamped for visual stability)
        # Publication standard: usually between 0.5 and 5.0
        lw = 0.5 + 4.0 * (min(score, 10.0) / 10.0)

        # Color intensity can also reflect score
        alpha = 0.3 + 0.6 * (min(score, 10.0) / 10.0)

        for source in rel["involved"]:
            if source == target: continue

            # Draw a curved link between the centers of the two sectors
            circos.link(
                (target, 4, 6),   # (SectorName, StartPos, EndPos)
                (source, 4, 6),
                color=color,
                alpha=alpha,
                lw=lw,
                direction=1 # Arrow pointing towards the target
            )

    # 6. Final Rendering and Export
    # Use GridSpec to place the legend at the bottom of the plot
    fig = plt.figure(figsize=(6, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[5, 1])
    
    ax_circos = fig.add_subplot(gs[0], projection="polar")
    circos.plotfig(ax=ax_circos)
    ax_circos.set_title("Variable Influence Hierarchy", fontsize=14, fontweight="bold", pad=15)

    ax_legend = fig.add_subplot(gs[1])

    # 7. Legend Logic
    ax_legend.axis("off")
    
    # Thickness legend
    legend_elements = []
    for s in [1, 5, 10]:
        lw = 0.5 + 4.0 * (min(s, 10.0) / 10.0)
        legend_elements.append(plt.Line2D([0], [0], color='black', lw=lw, label=f'Score: {s}'))
    
    # Color legend
    for layer, col in layer_colors.items():
        legend_elements.append(plt.Line2D([0], [0], color=col, lw=2, label=f'Layer {layer}'))
    
    ax_legend.legend(handles=legend_elements, loc='center', title="Mappings", 
                     title_fontsize='9', fontsize='8', ncol=3) 

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()