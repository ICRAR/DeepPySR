import os
import re
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import networkx as nx

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
    threshold: float = 1e-3
) -> bool:
    if not all_relationships:
        return False

    target_sym = sp.Symbol(target_name)
    expr_to_check = target_sym - sym_expr
    
    sorted_rels = sorted(all_relationships, key=lambda r: r["layer"])
    sub_map = []
    for rel in sorted_rels:
        t_sym = rel.get("target_symbol") or sp.Symbol(rel["target"])
        sub_map.append((t_sym, rel["sympy"]))
    
    expr_to_check = expr_to_check.subs(sub_map)

    try:
        free_vars = sorted(list(expr_to_check.free_symbols), key=lambda s: str(s))
        if not free_vars:
            return abs(float(expr_to_check)) < threshold
            
        f = sp.lambdify(free_vars, expr_to_check, modules=[
            "numpy", 
            {
                "sin": np.sin, "cos": np.cos, "exp": np.exp, "log": np.log, 
                "sqrt": np.sqrt, "tanh": np.tanh, "square": np.square,
                "asin": np.arcsin, "acos": np.arccos, "atanh": np.arctanh
            }
        ])
        
        n_points = 100
        test_data = [np.random.uniform(0.1, 1.0, n_points) for _ in free_vars]
        values = f(*test_data)
        
        if np.isscalar(values):
            return abs(float(values)) < threshold
            
        if np.allclose(values, 0, atol=threshold, rtol=threshold):
            return True
            
        if np.std(values) < threshold and np.abs(np.mean(values)) < threshold:
            return True
                
    except Exception:
        pass

    return False

def plot_n_layer_graph(
    relationships: list[dict],
    save_path: str,
    total_vars: int = 0
):
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
    unique_targets = sorted(list(set(d["label"] for d in node_data.values())), 
                            key=lambda x: (0 if x=="y" else 1, int(x[1:]) if x.startswith("x") and x[1:].isdigit() else x))
    
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
