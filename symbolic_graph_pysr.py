import os
import re
import sys
import json
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import networkx as nx

from pysr import PySRRegressor

sys.setrecursionlimit(5000)


def ensure_output_dir(path: str = "outputs") -> str:
    os.makedirs(path, exist_ok=True)
    return path


def generate_data(n_samples: int = 1000, n_vars: int = 10, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n_samples, n_vars))

    # Construct some hidden symbolic relationships among variables
    # Layer 3 (can be enabled if n_layers >= 3 is used)
    X[:,4] = np.cos(X[:,8]+X[:,7])
    X[:,5] = X[:,8] + X[:,9]
    
    # Layer 2
    X[:,1] = X[:,4] + np.sin(X[:,5]/(X[:,6]+1))
    X[:,2] = X[:,5] * np.exp(X[:,7]-1)

    # Layer 1
    y = X[:,0] * np.exp(X[:,1]-1) + X[:,2]*np.sin(X[:,3])
    return X, y


def fit_symbolic_regression(
    X: np.ndarray,
    y: np.ndarray,
    maxsize: int = 20,
    niterations: int = 100,
    seed: int = 42,
    population_size: int | None = None,
    ncycles_per_iteration: int = 50,
    parsimony: float = 1e-5,
):
    # Strengthened search configuration for better fit on complex, noisy targets
    if population_size is None:
        population_size = 500
    procs = max(1, (os.cpu_count() or 2) - 1)

    model = PySRRegressor(
        niterations=niterations,
        maxsize=maxsize,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp", "log", "sqrt", "tanh", "square", "asin", "acos", "atanh"],
        model_selection="best",
        parsimony=parsimony,
        random_state=seed,
        progress=False,
        population_size=population_size,
        ncycles_per_iteration=ncycles_per_iteration,
        procs=procs,
    )
    model.fit(X, y)
    return model


def get_best_equation_info(model: PySRRegressor, eq_index: int = 0):
    eqs = model.equations_
    best = eqs.iloc[eq_index]
    formula = str(best["equation"]) if "equation" in best else str(best.get("sympy_format", ""))
    complexity = int(best.get("complexity", -1))
    loss = float(best.get("loss", np.nan))
    score = float(best.get("score", np.nan)) if "score" in best else None
    # Fallback score: inverse loss (higher is better)
    inv_loss_score = None
    if not np.isfinite(loss) or loss <= 0:
        inv_loss_score = None
    else:
        inv_loss_score = 1.0 / loss
    if score is None or not np.isfinite(score):
        score = inv_loss_score
    
    try:
        sym_expr = model.sympy(eq_index)
    except Exception:
        # Fallback to simple parse
        sym_expr = sp.sympify(formula)

    involved = sorted({str(s) for s in sym_expr.free_symbols})

    return {
        "formula": formula,
        "sympy": sym_expr,
        "complexity": complexity,
        "loss": loss,
        "score": score,
        "involved_vars": involved,
    }


def eval_equation_on(
    sym_expr: sp.Expr, X: np.ndarray
) -> np.ndarray:
    # Build lambdified function f(x0, x1, ..., xN)
    vars_sorted = sorted(list(sym_expr.free_symbols), key=lambda s: int(str(s)[1:]) if str(s).startswith("x") and str(s)[1:].isdigit() else 1e9)
    # Ensure we always provide a full argument list in order x0..x{d-1}
    d = X.shape[1]
    arg_symbols = [sp.Symbol(f"x{i}") for i in range(d)]
    f = sp.lambdify(arg_symbols, sym_expr, modules={"sin": np.sin, "cos": np.cos, "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "tanh": np.tanh, "square": np.square, "asin": np.arcsin, "acos": np.arccos, "atanh": np.arctanh})
    return f(*[X[:, i] for i in range(d)])


def select_best_idx(model: PySRRegressor) -> int:
    """Return index of the equation with the highest score."""
    eqs = model.equations_
    if "score" in eqs.columns:
        return int(eqs["score"].idxmax())
    # Fallback to the last one if score is not available (PySR usually sorts by complexity/loss)
    return len(eqs) - 1


def mse_and_r2(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    mse = float(np.mean((y_true - y_pred) ** 2))
    var = float(np.var(y_true))
    r2 = 1.0 - (mse / var) if var > 0 else float("nan")
    return mse, r2


def involved_indices(involved_vars):
    idxs = []
    for v in involved_vars:
        m = re.fullmatch(r"x(\d+)", v)
        if m:
            idxs.append(int(m.group(1)))
    return sorted(idxs)


def detect_basic_function_for_var(sym_expr: sp.Expr, var_symbol: sp.Symbol) -> str:
    # Inspect string patterns for common functions applied to the variable
    s = str(sym_expr)
    v = str(var_symbol)
    tags = []
    for fn in ["sin", "cos", "exp", "log", "sqrt", "tanh", "square", "asin", "acos", "atanh"]:
        if re.search(rf"\b{fn}\({re.escape(v)}\)", s):
            tags.append(fn)
    if not tags:
        # If variable appears linearly? quick heuristic: check occurrences not inside functions
        # Very rough: if appears and not with power >1
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
    """
    Check if the relationship target_name = sym_expr is already implied by existing relationships.
    Uses both symbolic simplification and numerical verification on random data.
    """
    if not all_relationships:
        return False

    target_sym = sp.Symbol(target_name)
    expr_to_check = target_sym - sym_expr
    
    # Symbolic substitution using a pre-built map for speed.
    # We sort by layer to ensure that substitutions are performed from top-level down.
    # Note: all_relationships is typically already sorted by layer due to BFS discovery.
    sorted_rels = sorted(all_relationships, key=lambda r: r["layer"])
    sub_map = []
    for rel in sorted_rels:
        # Use cached symbol if available, otherwise create it.
        t_sym = rel.get("target_symbol") or sp.Symbol(rel["target"])
        sub_map.append((t_sym, rel["sympy"]))
    
    # Batch substitution is much faster than iterative subs calls.
    # One pass with an ordered list of tuples handles nested dependencies correctly.
    expr_to_check = expr_to_check.subs(sub_map)
        
        # Use nsimplify to handle floating point constants that are close to rationals
        # # But skip if it's too complex or just rely on numerical check if it fails/takes time
        # try:
        #     expr_to_check = sp.simplify(sp.nsimplify(expr_to_check, tolerance=1e-4, rational=True))
        #     if expr_to_check == 0:
        #         return True
        # except Exception:
        #     pass

    # Numerical verification on RANDOM data (to avoid dependencies in the project's data X)
    try:
        free_vars = sorted(list(expr_to_check.free_symbols), key=lambda s: str(s))
        if not free_vars:
            # It's a constant
            return abs(float(expr_to_check)) < threshold
            
        # Evaluate expr_to_check on random values for its free symbols
        f = sp.lambdify(free_vars, expr_to_check, modules=[
            "numpy", 
            {
                "sin": np.sin, "cos": np.cos, "exp": np.exp, "log": np.log, 
                "sqrt": np.sqrt, "tanh": np.tanh, "square": np.square,
                "asin": np.arcsin, "acos": np.arccos, "atanh": np.arctanh
            }
        ])
        
        # Test on 100 random points
        n_points = 100
        test_data = [np.random.uniform(0.1, 1.0, n_points) for _ in free_vars]
        values = f(*test_data)
        
        # Handle cases where values might be scalars (if expr became constant)
        if np.isscalar(values):
            return abs(float(values)) < threshold
            
        # If all values are very close to zero, or it's a constant very close to zero
        if np.allclose(values, 0, atol=threshold, rtol=threshold):
            return True
            
        if np.std(values) < threshold and np.abs(np.mean(values)) < threshold:
            return True
                
    except Exception:
        pass

    return False


def discover_n_layers(
    X: np.ndarray,
    y: np.ndarray,
    max_layers: int,
    maxsize: int = 20,
    seed: int = 42,
    score_threshold: float = 0.1,
    niterations: int = 100,
    population_size: int = 1000,
    ncycles_per_iteration: int = 50,
    parsimony: float = 1e-5,
):
    """
    Recursively discover symbolic relationships up to max_layers.
    Layer 1: y = f(x_i, x_j, ...)
    Layer 2: x_i = f(x_k, x_l, ...) for each x involved in Layer 1
    ...
    """
    all_relationships = []
    max_reached_layer = 0
    
    # We use a queue-based approach to explore layers
    # Each item in queue: (target_name, target_data, current_layer, parent_name)
    queue = [("y", y, 1, None)]
    processed_targets = set()

    while queue:
        target_name, target_y, layer, parent_name = queue.pop(0)
        if layer > max_layers:
            continue
        
        max_reached_layer = max(max_reached_layer, layer)
        
        # Avoid processing the same variable multiple times at the same or higher layer?
        # Actually, the requirement says "The second layer can contain repeated x if the same x is involved in multiple other x."
        # This suggests we might want to allow it, but for discovery (PySR), we only need to find the formula once.
        # However, for the tree structure, we might want to know where it came from.
        
        # To avoid infinite loops or redundant PySR runs:
        if target_name in processed_targets and target_name != "y":
             continue
        processed_targets.add(target_name)

        print(f"\n--- Discovering relationships for {target_name} at layer {layer} ---")
        
        # Determine features for this target
        if target_name == "y":
            X_input = X
            cols = list(range(X.shape[1]))
        else:
            # If target is x{i}, we use all other x{j} as features, except for the x{k} in the previous layer.
            idx = int(target_name[1:])
            parent_idx = None
            if parent_name and parent_name.startswith("x"):
                try:
                    parent_idx = int(parent_name[1:])
                except ValueError:
                    pass
            
            cols = [j for j in range(X.shape[1]) if j != idx and j != parent_idx]
            X_input = X[:, cols]

        try:
            model = fit_symbolic_regression(
                X_input,
                target_y,
                maxsize=maxsize,
                niterations=niterations,
                seed=seed + layer + len(all_relationships),
                population_size=population_size,
                ncycles_per_iteration=ncycles_per_iteration,
                parsimony=parsimony,
            )
            best_idx = select_best_idx(model)
            info = get_best_equation_info(model, eq_index=best_idx)
            
            # Map back variable names if needed
            sym_expr = info["sympy"]
            
            # Round coefficients to 2 decimal points
            for a in sp.preorder_traversal(sym_expr):
                if isinstance(a, sp.Float):
                    sym_expr = sym_expr.subs(a, round(a, 2))

            if target_name != "y":
                mapping = {sp.Symbol(f"x{k}"): sp.Symbol(f"x{cols[k]}") for k in range(len(cols))}
                # Using xreplace instead of simplify to avoid recursion issues if expr is complex
                sym_expr = sym_expr.xreplace(mapping)
            
            involved = sorted({str(s) for s in sym_expr.free_symbols})
            
            # Check for redundancy
            if is_redundant(target_name, sym_expr, all_relationships):
                print(f"Relationship for {target_name} is redundant. Skipping.")
                continue

            relationship = {
                "target": target_name,
                "target_symbol": sp.Symbol(target_name),
                "layer": layer,
                "sympy": sym_expr,
                "formula": str(sym_expr),
                "involved": involved,
                "score": info["score"],
                "complexity": info["complexity"]
            }
            
            # Add to relationships if the relationship is strong enough
            if info["score"] > score_threshold:
                all_relationships.append(relationship)
            else:
                print(f"Relationship for {target_name} is weak (score={info['score']:.4g} < threshold={score_threshold}). Treating as leaf node.")
            
            # Add involved variables to queue for the next layer
            # Only continue if the relationship is reasonably strong
            if layer < max_layers and info["score"] > score_threshold:
                for vname in involved:
                    # vname is like 'x2', 'x9'
                    try:
                        v_idx = int(vname[1:])
                        queue.append((vname, X[:, v_idx], layer + 1, target_name))
                    except (ValueError, IndexError):
                        continue
                    
        except Exception as e:
            print(f"Warning: modeling {target_name} failed: {e}")
            
    if max_reached_layer < max_layers:
        print(f"\nEarly stop: the deepest layer for an explored relationship is {max_reached_layer}, which is shallower than the given {max_layers} layers.")

    return all_relationships


def plot_n_layer_graph(
    relationships: list[dict],
    save_path: str,
    total_vars: int = 0
):
    G = nx.DiGraph()
    
    node_data = {} # node_id -> {label, layer, formula}
    edges = [] # (src_id, dst_id, tag, style)
    
    # Helper to find relationship for a variable
    rel_map = {r["target"]: r for r in relationships}

    # Use a recursive approach to build the graph nodes and edges
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
            # If node exists, update its layer to be the maximum (furthest from output 'y')
            # to ensure inputs are at the bottom
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

        # Explore children regardless of whether node existed, 
        # but avoid infinite loops and respect max_layers
        if current_layer < max_layers:
            rel = rel_map.get(target_name)
            if rel:
                for vname in rel["involved"]:
                    # Avoid direct self-loops
                    if vname != target_name:
                        add_nodes_recursive(vname, node_id, current_layer + 1, max_layers)

    max_rel_layer = max((r["layer"] for r in relationships), default=0)
    add_nodes_recursive("y", None, 0, max_rel_layer)

    # Hierarchical Layout
    # Group nodes by layer
    layers = {}
    for nid, data in node_data.items():
        l = data["layer"]
        if l not in layers:
            layers[l] = []
        layers[l].append(nid)

    # Radial Layout Calculation
    pos = {}
    max_l = max(layers.keys()) if layers else 0
    
    # We'll use a radial tree layout approach.
    # 1. Assign 'y' to the center
    pos["y"] = np.array([0.0, 0.0])
    
    # 2. To handle DAG, we'll build a spanning tree for layout purposes
    # or just use the first parent for each node.
    layout_tree_children = {} # parent -> [children]
    visited = {"y"}
    queue = ["y"]
    while queue:
        parent = queue.pop(0)
        layout_tree_children[parent] = []
        # Find all children of this parent in the graph
        # Children are nodes that have an edge pointing TO this parent
        children = [src for src, dst, tag in edges if dst == parent]
        for child in children:
            if child not in visited:
                visited.add(child)
                layout_tree_children[parent].append(child)
                queue.append(child)

    # 3. Recursively assign angular sectors
    def assign_radial_pos(parent, start_angle, end_angle, radius_step=1.0):
        children = layout_tree_children.get(parent, [])
        if not children:
            return
        
        n_children = len(children)
        parent_l = node_data[parent]["layer"]
        child_radius = (parent_l + 1) * radius_step
        
        sector_width = end_angle - start_angle
        # If it's the root 'y', we use the full circle.
        # Otherwise, we might want to leave some gap between groups.
        
        for i, child in enumerate(children):
            # Calculate angle for this child
            angle = start_angle + (i + 0.5) * (sector_width / n_children)
            
            x = child_radius * np.cos(angle)
            y = child_radius * np.sin(angle)
            pos[child] = np.array([x, y])
            
            # Sub-sector for child's own children
            child_start = start_angle + i * (sector_width / n_children)
            child_end = start_angle + (i + 1) * (sector_width / n_children)
            assign_radial_pos(child, child_start, child_end, radius_step)

    assign_radial_pos("y", 0, 2 * np.pi)

    # For any nodes not in our spanning tree (shouldn't happen with 'y' as root), 
    # give them a default position.
    for nid in node_data:
        if nid not in pos:
            l = node_data[nid]["layer"]
            pos[nid] = np.array([l, 0.0]) # fallback

    # Create figure with two subplots: graph and formula table
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
    ax_graph = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])

    # Draw nodes
    # Expanded color palette to support more layers
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
    
    # Draw labels
    labels = {nid: d["label"] for nid, d in node_data.items()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=11, font_weight="bold", ax=ax_graph)

    # Draw edges with curves
    for src, dst, tag in edges:
        G.add_edge(src, dst, label=tag)
        
    # Draw curved edges
    for u, v in G.edges():
        # Score-based thickness
        # The edge is from child to parent (u -> v). We want the score of the relationship where v is the target.
        rel_v = rel_map.get(v)
        score = rel_v["score"] if rel_v else 0.1
        lw = 0.5 + 2.0 * min(score, 5.0) / 5.0 # Scale thickness
        
        # Curvature depends on distance and direction
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

    # Edge labels (tags)
    edge_labels = nx.get_edge_attributes(G, "label")
    # Only draw non-empty tags
    edge_labels = {k: v for k, v in edge_labels.items() if v and v != "linear/mixed" and v != "unused"}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax_graph, label_pos=0.5, alpha=0.8)

    ax_graph.set_title(f"Symbolic Relationship Hierarchy (depth={max_l})", fontsize=14)
    ax_graph.axis("off")

    # Table of formulas
    ax_table.axis("off")
    table_data = []
    # Only show unique targets in table
    unique_targets = sorted(list(set(d["label"] for d in node_data.values())), 
                            key=lambda x: (0 if x=="y" else 1, int(x[1:]) if x.startswith("x") and x[1:].isdigit() else x))
    
    for t in unique_targets:
        rel = rel_map.get(t)
        if rel:
            # Shorten formula if too long
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
        # Adjust column widths
        table.auto_set_column_width([0, 1, 2, 3])
        # Bold the headers
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.get_text().set_weight('bold')
                cell.set_facecolor('#f2f2f2')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="run1", help="Name of this run, used for output directory")
    parser.add_argument("--complexity", type=int, default=20, help="Max complexity (maxsize) for PySR")
    parser.add_argument("--layers", type=int, default=4, help="Number of layers to discover")
    parser.add_argument("--threshold", type=float, default=5, help="Score threshold for symbolic regression")
    parser.add_argument("--niterations", type=int, default=50, help="Number of iterations for PySR")
    parser.add_argument("--population_size", type=int, default=1000, help="Population size for PySR")
    parser.add_argument("--ncycles_per_iteration", type=int, default=50, help="Number of cycles per iteration for PySR")
    parser.add_argument("--parsimony", type=float, default=1e-5, help="Parsimony for PySR")
    args, unknown = parser.parse_known_args()


    maxsize = args.complexity
    max_layers = args.layers
    score_threshold = args.threshold
    niterations = args.niterations
    population_size = args.population_size
    ncycles_per_iteration = args.ncycles_per_iteration
    parsimony = args.parsimony

    print(f"Using max complexity: {maxsize}, max layers: {max_layers}, threshold: {score_threshold}")
    print(f"PySR Params: niterations={niterations}, population_size={population_size}, ncycles_per_iteration={ncycles_per_iteration}, parsimony={parsimony}")

    out_dir = ensure_output_dir(os.path.join("outputs", args.run_name))
    rel_json_path = os.path.join(out_dir, "relationships.json")
    # 1) Create data
    X, y = generate_data(n_samples=1000, n_vars=10, seed=123)
    # 2) Discover relationships
    if os.path.exists(rel_json_path):
        print(f"Warning: relationships file {rel_json_path} already exists. Skipping discovery.")
        with open(rel_json_path, "r") as f:
            all_relationships = json.load(f)
        
        # Reconstruct SymPy objects from formulas
        for r in all_relationships:
            if "formula" in r:
                expr = sp.sympify(r["formula"])
                # Round coefficients just in case
                for a in sp.preorder_traversal(expr):
                    if isinstance(a, sp.Float):
                        expr = expr.subs(a, round(a, 2))
                r["sympy"] = expr
                r["formula"] = str(expr)
            if "target" in r:
                r["target_symbol"] = sp.Symbol(r["target"])
        
        print("\n=== Loaded Relationships from JSON ===")
        for r in all_relationships:
            print(f"Layer {r['layer']} | {r['target']} = {r['formula']} (score={r['score']:.4g})")
    else:
        all_relationships = discover_n_layers(
            X, y,
            max_layers=max_layers,
            maxsize=maxsize,
            seed=123,
            score_threshold=score_threshold,
            niterations=niterations,
            population_size=population_size,
            ncycles_per_iteration=ncycles_per_iteration,
            parsimony=parsimony,
        )

        # Print discovered equations
        print("\n=== All Discovered Relationships ===")
        for r in all_relationships:
            print(f"Layer {r['layer']} | {r['target']} = {r['formula']} (complexity={r['complexity']}, score={r['score']:.4g})")
            print(f"  involved: {r['involved']}")

        # Save relationships to JSON
        # Remove non-serializable SymPy objects
        serializable_relationships = []
        for r in all_relationships:
            r_copy = r.copy()
            r_copy.pop("sympy", None)
            r_copy.pop("target_symbol", None)
            serializable_relationships.append(r_copy)


        with open(rel_json_path, "w") as f:
            json.dump(serializable_relationships, f, indent=4)
        print(f"Saved all relationships to: {rel_json_path}")

    # 3) Graph plots
    if all_relationships:
        hierarchical_fig = os.path.join(out_dir, f"hard_graph_n_layers_{max_layers}.png")
        plot_n_layer_graph(all_relationships, hierarchical_fig, total_vars=X.shape[1])
        print(f"Saved n-layer hierarchical graph to: {hierarchical_fig}")
    else:
        print("No relationships discovered.")


if __name__ == "__main__":
    main()
