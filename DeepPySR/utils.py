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
        threshold: float = 1e-4,
        layer: int = 1
) -> bool:
    """
    Checks redundancy by evaluating the new expression against existing
    relationships over a vectorized numerical grid.
    """
    if not all_relationships:
        return False
        
    # We never consider root targets redundant (layer 1) based on identity check
    # as we want to capture all root targets even if they are constant or same as others.
    if layer == 1:
        return False

    # 1. Identify all involved symbols across all relationships
    all_involved = set(str(s) for s in sym_expr.free_symbols)
    for r in all_relationships:
        all_involved.update(r["involved"])
    
    # 2. Identify root symbols (those that are NOT targets of any relationship)
    intermediate_targets = {r["target"] for r in all_relationships}
    root_vars = sorted([v for v in all_involved if v not in intermediate_targets])
    root_symbols = [sp.Symbol(v) for v in root_vars]

    try:
        # 3. Generate high-volume test data for root symbols
        n_points = 2000 # Reduced for speed, still enough for redundancy check
        test_data = {v: np.random.uniform(0.1, 1.0, n_points) for v in root_vars}

        # 4. Evaluate relationships layer by layer numerically
        # Sort by layer to ensure dependencies are computed first
        sorted_rels = sorted(all_relationships, key=lambda r: r["layer"])
        
        values = test_data.copy()
        
        for rel in sorted_rels:
            involved = rel["involved"]
            # Check if we have all needed inputs for this relationship
            if all(v in values for v in involved):
                f_rel = sp.lambdify([sp.Symbol(v) for v in involved], rel["sympy"], modules="numpy")
                args = [values[v] for v in involved]
                values[rel["target"]] = f_rel(*args)

        # 5. Evaluate the new expression
        involved_new = sorted([str(s) for s in sym_expr.free_symbols])
        if not all(v in values for v in involved_new):
            # Cannot fully expand numerically, might happen if some inputs are missing
            return False
            
        f_new = sp.lambdify([sp.Symbol(v) for v in involved_new], sym_expr, modules="numpy")
        new_val = f_new(*[values[v] for v in involved_new])

        # 6. Check if new_val is close to target_val (if target is in values)
        if target_name in values:
            target_val = values[target_name]
            if np.allclose(new_val, target_val, atol=threshold):
                return True

        # 7. Check if it's a constant
        if np.std(new_val) < threshold:
            return True

    except Exception:
        # Fallback to False if math errors occur
        return False

    return False

def plot_n_layer_graph(
    relationships: list[dict],
    save_path: str,
    feature_names: list[str] = None,
    target_variable: str = "Synthetic Data"
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
    
    # Handle multiple target variables
    if isinstance(target_variable, list):
        for tv in target_variable:
            add_nodes_recursive(tv, None, 0, max_rel_layer)
    else:
        add_nodes_recursive(target_variable, None, 0, max_rel_layer)

    layers = {}
    for nid, data in node_data.items():
        l = data["layer"]
        if l not in layers:
            layers[l] = []
        layers[l].append(nid)

    pos = {}
    layout_tree_children = {}
    max_l = max(layers.keys()) if layers else 0

    # Sort roots by target_variable order if it's a list
    roots = []
    if isinstance(target_variable, list):
        roots = [tv for tv in target_variable if tv in node_data]
    elif target_variable in node_data:
        roots = [target_variable]
    
    if not roots:
        # Fallback to layer 0 nodes if no matches
        roots = [nid for nid, data in node_data.items() if data["layer"] == 0]

    visited = set(roots)
    queue = list(roots)
    
    # Position roots
    n_roots = len(roots)
    for i, root in enumerate(roots):
        if n_roots > 1:
            # Spread roots if there are multiple
            angle = (i / n_roots) * 2 * np.pi
            pos[root] = np.array([0.5 * np.cos(angle), 0.5 * np.sin(angle)])
        else:
            pos[root] = np.array([0.0, 0.0])

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

    for i, root in enumerate(roots):
        if n_roots > 1:
            root_start = (i / n_roots) * 2 * np.pi
            root_end = ((i + 1) / n_roots) * 2 * np.pi
        else:
            root_start = 0
            root_end = 2 * np.pi
        assign_radial_pos(root, root_start, root_end, radius_step=1.0)

    for nid in node_data:
        if nid not in pos:
            l = node_data[nid]["layer"]
            pos[nid] = np.array([l, 0.0]) 

    fig, ax_graph = plt.subplots(figsize=(6, 4))

    colors = [
        "#FFD700", "#87CEEB", "#90EE90", "#FFB6C1", "#DDA0DD", 
        "#FFA07A", "#20B2AA", "#778899", "#B0C4DE", "#FFFFE0",
        "#F0E68C", "#E6E6FA", "#FFF0F5", "#FFE4E1", "#F0FFF0"
    ]
    for l in range(max_l + 1):
        nodelist = [nid for nid in node_data if node_data[nid]["layer"] == l]
        if not nodelist: continue
        color = colors[l % len(colors)]
        size = 800 / (l + 1)**0.5
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color=color, 
                               node_size=size, alpha=0.9, ax=ax_graph, edgecolors="black", linewidths=1)
    
    labels = {nid: d["label"] for nid, d in node_data.items()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight="bold", ax=ax_graph)

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
    
    # In NetworkX 3.5, draw_networkx_edge_labels can fail with "too many values to unpack"
    # if it doesn't correctly handle curved edges from our custom annotate() call.
    # Since we manually annotate edges with arrows and curvature, we only use
    # draw_networkx_edge_labels for the text labels.
    try:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax_graph, label_pos=0.5, alpha=0.8)
    except Exception as e:
        print(f"Warning: Could not draw edge labels: {e}")

    # ax_graph.set_title(f"Symbolic Relationship Hierarchy (depth={max_l})", fontsize=14)
    ax_graph.set_title(f"DeepPySR Interpretation for {target_variable}", fontsize=14)
    ax_graph.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

def plot_circlize(
        relationships: list[dict],
        save_path: str,
        feature_names: list[str] = None,
        target_variable: str = "Synthetic Data"
):
    # 1. Collect and sort all unique variables
    all_vars = set()
    if isinstance(target_variable, list):
        for tv in target_variable:
            all_vars.add(tv)
    else:
        all_vars.add(target_variable)
        
    for rel in relationships:
        all_vars.add(rel["target"])
        for v in rel["involved"]:
            all_vars.add(v)

    # Add missing variables if feature_names is provided
    if feature_names:
        for name in feature_names:
            all_vars.add(name)

    # 1.5 Create a mapping for display if needed
    # (Though relationships should already be mapped)
    display_names = {v: v for v in all_vars}
    if feature_names:
        # If there are still x0, x1... in all_vars, map them
        for i, name in enumerate(feature_names):
            if f"x{i}" in all_vars:
                display_names[f"x{i}"] = name

    # Sort: root targets first, then x0, x1, etc.
    roots_set = set(target_variable) if isinstance(target_variable, list) else {target_variable}
    
    def sort_key(v):
        if v in roots_set:
            # If multiple roots, maintain their relative order
            if isinstance(target_variable, list):
                return (0, target_variable.index(v), v)
            return (0, 0, v)
        if v.startswith("x") and v[1:].isdigit():
            return (1, int(v[1:]), v)
        return (2, 0, v)

    sorted_vars = sorted(list(all_vars), key=sort_key)

    # 2. Identify variables directly influencing roots
    direct_to_roots = set()
    for tv in roots_set:
        tv_rel = next((r for r in relationships if r["target"] == tv), None)
        if tv_rel:
            direct_to_roots.update(tv_rel["involved"])
    
    # 3. Define layer colors
    max_layer = max((r["layer"] for r in relationships), default=1)
    cmap = plt.get_cmap("viridis", max_layer)
    layer_colors = {i+1: cmap(i) for i in range(max_layer)}

    # 4. Initialize pyCirclize Circos
    # We assign each variable a sector of equal size (10 units)
    sectors = {v: 10 for v in sorted_vars}
    
    # Adaptive sector spacing based on number of variables
    # More variables -> less space between sectors
    # Fewer variables -> more space between sectors
    num_vars = len(sorted_vars)
    space = max(2, 10 - num_vars * 0.5) 
    circos = Circos(sectors, space=space)

    # 5. Draw sectors and nodes
    for sector in circos.sectors:
        # Create a track for the variable "dots" and labels
        # Track from radius 95 to 100
        node_track = sector.add_track((95, 100))

        v_name = sector.name
        # Determine styling
        if v_name in roots_set:
            color = "#D32F2F" # Professional Dark Red
            size = 120 # Reduced from 150
            zorder = 5
        elif v_name in direct_to_roots:
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
        # Use target_variable instead of "y"
        label = display_names.get(v_name, v_name)
        if v_name == "y":
            label = target_variable

        # Adaptive radius and smaller font size
        label_radius = 105
        # Determine orientation: vertical for long labels, horizontal for very short ones?
        # User requested they all appear outside for the dots, from start till end.
        # "Vertical" in pycirclize means it points outwards radially.
        orientation = "vertical"

        # Smaller font size as requested (was 10, now 8)
        # We can also adjust the figure size based on the longest label
        node_track.text(label, x=5, r=label_radius, size=8, weight="bold", orientation=orientation)

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
    # Adaptive figure size based on variable name length
    max_label_len = max([len(display_names.get(v, v)) for v in sorted_vars] + [len(target_variable)])
    # A base size of 3-4 inches, plus something proportional to the longest label
    # Max label length of 30-40 characters might need 6-8 inches.
    base_size = 4
    # The plot is a circle, labels extend radially.
    # Long labels can significantly increase the required figure size.
    # 10 characters might be ~1 inch.
    added_size_per_char = 0.08
    added_size = max(0, (max_label_len - 5) * added_size_per_char) 
    fig_size = base_size + added_size
    
    fig = plt.figure(figsize=(fig_size, fig_size + 1))
    gs = fig.add_gridspec(2, 1, height_ratios=[5, 1])
    
    ax_circos = fig.add_subplot(gs[0], projection="polar")
    circos.plotfig(ax=ax_circos)
    ax_circos.set_title(f"DeepPySR: {target_variable}", fontsize=12, fontweight="bold", pad=20 + (added_size * 5))

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