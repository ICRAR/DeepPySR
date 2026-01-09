import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import networkx as nx
try:
    from pysr import PySRRegressor
    _PYSR_AVAILABLE = True
except Exception:
    _PYSR_AVAILABLE = False

def _safe_log_torch(x):
    return torch.log(torch.clamp(x, min=1e-6))

def _safe_inv_torch(x):
    eps = 1e-6
    x_clamped = torch.where(x >= 0, torch.clamp(x, min=eps), torch.clamp(x, max=-eps))
    return 1.0 / x_clamped

def _safe_sqrt_torch(x):
    return torch.sqrt(torch.clamp(x, min=0.0))

def _safe_exp_torch(x):
    return torch.exp(torch.clamp(x, min=-50.0, max=50.0))

FUNCTION_LIBRARY = {
    'sin': {'np': np.sin, 'torch': torch.sin, 'sympy': sp.sin},
    'cos': {'np': np.cos, 'torch': torch.cos, 'sympy': sp.cos},
    'tan': {'np': np.tan, 'torch': torch.tan, 'sympy': sp.tan},
    'exp': {'np': np.exp, 'torch': _safe_exp_torch, 'sympy': sp.exp},
    'log': {'np': np.log, 'torch': _safe_log_torch, 'sympy': sp.log},
    'sqrt': {'np': np.sqrt, 'torch': _safe_sqrt_torch, 'sympy': sp.sqrt},
    'tanh': {'np': np.tanh, 'torch': torch.tanh, 'sympy': sp.tanh},
    'sigmoid': {'np': lambda x: 1/(1+np.exp(-x)), 'torch': torch.sigmoid, 'sympy': lambda v: 1/(1 + sp.exp(-v))},
    'identity': {'np': lambda x: x, 'torch': lambda x: x, 'sympy': lambda v: v},
    'square': {'np': lambda x: x**2, 'torch': lambda x: x**2, 'sympy': lambda v: v**2},
    'cube': {'np': lambda x: x**3, 'torch': lambda x: x**3, 'sympy': lambda v: v**3},
    'inv': {'np': lambda x: 1/x, 'torch': _safe_inv_torch, 'sympy': lambda v: 1/v},
    'gauss': {'np': lambda x: np.exp(-x**2), 'torch': lambda x: torch.exp(-(x**2)), 'sympy': lambda v: sp.exp(-(v**2))},
    # Add more functions if needed for more complex symbolic fitting
}

def B_batch(x, grid, k=0):
    x = x.unsqueeze(dim=2)
    grid = grid.unsqueeze(dim=0)
    if k == 0:
        value = (x >= grid[:, :, :-1]) * (x < grid[:, :, 1:]).float()
    else:
        B_km1 = B_batch(x[:, :, 0], grid=grid[0], k=k - 1)
        denom1 = (grid[:, :, k:-1] - grid[:, :, :-(k + 1)]).clamp(min=1e-6)
        denom2 = (grid[:, :, k + 1:] - grid[:, :, 1:(-k)]).clamp(min=1e-6)
        term1 = (x - grid[:, :, :-(k + 1)]) / denom1 * B_km1[:, :, :-1]
        term2 = (grid[:, :, k + 1:] - x) / denom2 * B_km1[:, :, 1:]
        value = term1 + term2
    value = torch.nan_to_num(value)
    return value

def coef2curve(x_eval, grid, coef, k):
    b_splines = B_batch(x_eval, grid, k=k)
    y_eval = torch.einsum('ijk,jlk->ijl', b_splines, coef)
    return y_eval

def curve2coef(x_eval, y_eval, grid, k):
    batch = x_eval.shape[0]
    in_dim = x_eval.shape[1]
    out_dim = y_eval.shape[2]
    n_coef = grid.shape[1] - k - 1
    mat = B_batch(x_eval, grid, k)
    mat = mat.permute(1, 0, 2)[:, None, :, :].expand(in_dim, out_dim, batch, n_coef)
    y_eval = y_eval.permute(1, 2, 0).unsqueeze(dim=3)
    coef = torch.linalg.lstsq(mat, y_eval).solution.squeeze(-1)
    return coef

def extend_grid(grid, k_extend=0):
    h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)
    for i in range(k_extend):
        grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
        grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)
    return grid

class KANLayer(nn.Module):
    def __init__(self, in_dim=1, out_dim=1, num=15, k=3, noise_scale=0.1, scale_base=1.0, scale_sp=0.1, base_fun=nn.Identity(), grid_eps=0.02, grid_range=[-3, 3], sp_trainable=True, sb_trainable=True):
        super(KANLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num = num
        self.k = k
        grid = torch.linspace(grid_range[0], grid_range[1], steps=num + 1)[None, :].expand(in_dim, num + 1)
        self.grid = nn.Parameter(extend_grid(grid, k_extend=k), requires_grad=False)
        noises = (torch.rand(self.num + 1, self.in_dim, self.out_dim) - 0.5) * noise_scale / num
        self.coef = nn.Parameter(curve2coef(self.grid[:, self.k:-self.k].permute(1, 0), noises, self.grid, self.k))
        self.scale_base = nn.Parameter(torch.ones(in_dim, out_dim) * scale_base, requires_grad=sb_trainable)
        self.scale_sp = nn.Parameter(torch.ones(in_dim, out_dim) * scale_sp, requires_grad=sp_trainable)
        self.base_fun = base_fun
        self.grid_eps = grid_eps
        self.mask = nn.Parameter(torch.ones(in_dim, out_dim), requires_grad=False)
        self.grid_range = grid_range
        self.symbolic_enabled = False
        self.sym_name = None
        self.c = None
        self.d = None
        self.fun = None
        self.sym_fun = None
        self.best_r2 = -1

    def forward(self, x):
        batch = x.shape[0]
        # Only use symbolic path if parameters properly initialized
        if self.symbolic_enabled and (self.c is not None) and (self.d is not None) and (self.fun is not None):
            y_base = torch.tensor(self.c, device=x.device) * self.fun(x) + torch.tensor(self.d, device=x.device)
            y_base = torch.nan_to_num(y_base, nan=0.0, posinf=1e6, neginf=-1e6)
            y = y_base[:, :, None]
            y = self.mask[None, :, :] * y
            y = y.sum(dim=1)
            postacts = y.clone()
            postspline = torch.zeros_like(y)
        else:
            base = self.base_fun(x)
            spline = coef2curve(x, self.grid, self.coef, self.k)
            y = self.scale_base[None, :, :] * base[:, :, None] + self.scale_sp[None, :, :] * spline
            y = self.mask[None, :, :] * y
            y = y.sum(dim=1)
            postacts = y.clone()
            postspline = spline.clone()
        return y, x, postacts, postspline

    def update_grid_from_samples(self, x):
        batch = x.shape[0]
        x_pos = torch.sort(x, dim=0)[0]
        y_eval = coef2curve(x_pos, self.grid, self.coef, self.k)
        num_interval = self.grid.shape[1] - 1 - 2 * self.k
        ids = [int(batch / num_interval * i) for i in range(num_interval)] + [batch - 1]
        grid_adaptive = x_pos[ids, :].permute(1, 0)
        h = (grid_adaptive[:, [-1]] - grid_adaptive[:, [0]]) / num_interval
        grid_uniform = grid_adaptive[:, [0]] + h * torch.arange(num_interval + 1)[None, :]
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        self.grid.data = extend_grid(grid, k_extend=self.k)
        self.coef.data = curve2coef(x_pos, y_eval, self.grid, self.k)

    def prune(self, threshold=5e-3):
        norm = self.scale_base.abs() + self.scale_sp.abs() * self.coef.abs().mean()
        if norm.item() < threshold:
            self.mask.data = torch.zeros_like(self.mask)

    def auto_symbolic(self, lib=None, threshold=0.95):
        """
        Fit c * f(x) without per-edge bias. In GraphKAN, node j aggregates
        sums over incoming edges. An intercept per edge would accumulate and
        distort the node value. To align with MultiKAN-style auto symbolic,
        we estimate only a scale c and fix d = 0.
        """
        if self.mask.item() == 0:
            return
        print("Fitting symbolic for this layer (no per-edge bias)")
        x_np = np.linspace(self.grid_range[0], self.grid_range[1], 1000)
        x_t = torch.from_numpy(x_np).float().unsqueeze(1)
        with torch.no_grad():
            y_t, _, _, _ = self(x_t)
        y_np = y_t.squeeze().numpy()
        candidate_names = list(FUNCTION_LIBRARY.keys()) if lib is None else [n for n in lib if n in FUNCTION_LIBRARY]
        candidate_names = [n for n in candidate_names if callable(FUNCTION_LIBRARY[n].get('np', None))]
        fun_dict_torch = {n: FUNCTION_LIBRARY[n]['torch'] for n in candidate_names if 'torch' in FUNCTION_LIBRARY[n]}
        sym_dict = {n: FUNCTION_LIBRARY[n]['sympy'] for n in candidate_names if 'sympy' in FUNCTION_LIBRARY[n]}
        best_r2 = -1
        best_name = None
        best_c = None
        for name in candidate_names:
            base_np_fun = FUNCTION_LIBRARY[name]['np']
            with np.errstate(all='ignore'):
                fvals = base_np_fun(x_np)
                valid_mask = np.isfinite(fvals) & np.isfinite(y_np)
                if valid_mask.sum() < 50:
                    continue
                f_fit = fvals[valid_mask]
                y_fit = y_np[valid_mask]
                denom = np.sum(f_fit * f_fit)
                if denom <= 1e-12:
                    continue
                c_hat = np.sum(f_fit * y_fit) / denom
                pred = c_hat * f_fit
                if not np.all(np.isfinite(pred)):
                    continue
                ss_res = np.sum((y_fit - pred)**2)
                ss_tot = np.sum((y_fit - y_fit.mean())**2)
                r2 = 1 - ss_res / (ss_tot + 1e-12)
                if r2 > best_r2:
                    best_r2 = r2
                    best_name = name
                    best_c = float(c_hat)
        if best_r2 > threshold and best_name is not None:
            self.sym_name = best_name
            self.c = best_c
            self.d = 0.0
            self.fun = fun_dict_torch.get(best_name, lambda x: x)
            self.sym_fun = sym_dict.get(best_name, lambda v: v)
            self.symbolic_enabled = True
            self.best_r2 = best_r2
            print(f"Best fit: {best_name} with R2 {best_r2:.4f}, c={self.c:.4f}, d fixed to 0")
        else:
            print("No good symbolic fit found")

    def symbolic_formula(self, var='x'):
        if not self.symbolic_enabled:
            return None
        v = sp.symbols(var)
        formula = self.c * self.sym_fun(v) + self.d
        return sp.simplify(formula)

# Advanced GraphKAN: Integrates KAN layers as edge functions in a GCN-like structure.
# Connectivity rules:
# - Inputs: may connect to inputs and hidden nodes; also to output y.
# - Hidden: may connect to hidden nodes and to output y; NOT to inputs (no h->x edges).
# - Output y: receives from inputs/hidden only (no outgoing edges).
# Node states (including inputs) are updated via message passing to learn dependencies (e.g., x3 = f(x1, x2)).
# Reconstruction loss on inputs forces the model to learn how to reconstruct each input from others, discovering dependencies.
# Hidden nodes are free to learn intermediate representations for more complex relationships.
# Multiple layers allow deeper propagation of information.
class GraphKAN(nn.Module):
    def __init__(self, num_inputs=3, num_hidden=2, num_intervals=15, spline_order=3, grid_range=[-3, 3], num_layers=2, use_layernorm=False):
        super(GraphKAN, self).__init__()
        # Store topology
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_nodes = num_inputs + num_hidden
        self.num_layers = num_layers
        self.input_indices = list(range(num_inputs))
        self.hidden_indices = list(range(num_inputs, num_inputs + num_hidden))
        self.phis = nn.ModuleDict()
        # Build edges according to connectivity rules (no hidden -> input edges)
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j:
                    continue
                else:
                    self.phis[f'{i}_{j}'] = KANLayer(
                        num=num_intervals, k=spline_order, grid_range=grid_range, base_fun=nn.SiLU()
                    )
        # No LayerNorm (parameter kept only for backward-compatibility)
        self.ln = None

        # Storage for PySR-derived symbolic edge formulas and R² per edge
        # Keys are tuples (i, j) where i -> j is an edge index.
        self.edge_formulas = {}
        self.edge_r2 = {}

        # Storage for node-level equations discovered by PySR (multi-input)
        # self.node_equations[j] = {
        #   'expr': SymPy expression f(parents) with global symbols,
        #   'target': sympy symbol of target node,
        #   'vars': set of global parent indices used in expr,
        #   'r2': float R^2 score for node-level fit
        # }
        self.node_equations = {}
        # Deduplicated set of equations reserved after resolving symmetric/inverse duplicates
        # Keyed by frozenset of variable indices involved (parents ∪ {j})
        self.reserved_equations = {}


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Message-passing forward pass.

        Args:
            X: Tensor of shape (batch, num_nodes). Columns correspond to
               [inputs | hidden].

        Returns:
            Tensor of shape (batch, num_nodes) with the updated states after
            `num_layers` rounds of aggregation.
        """
        # Sanity: expect correct number of features
        assert X.dim() == 2 and X.size(1) == self.num_nodes, (
            f"Expected input with {self.num_nodes} features, got {X.size(1)}"
        )

        states = X
        if self.num_layers <= 0:
            return states

        for _ in range(self.num_layers):
            next_states = torch.zeros_like(states)
            # For each target node j, aggregate messages from active parents i
            for j in range(self.num_nodes):
                parents = self._active_in_neighbors(j)
                if len(parents) == 0:
                    # If no active parents, keep current state (residual)
                    next_states[:, j] = states[:, j]
                    continue
                acc = None
                for i in parents:
                    # Each edge module maps scalar input x_i -> contribution to x_j
                    y_ij, _, _, _ = self.phis[f"{i}_{j}"](states[:, i:i+1])
                    acc = y_ij if acc is None else (acc + y_ij)
                # acc is (batch, 1)
                next_states[:, j:j+1] = acc

            states = next_states

        return states


    def _active_in_neighbors(self, j: int):
        """Return list of source indices i with active mask on edge i->j."""
        nbrs = []
        for i in range(self.num_nodes):
            key = f"{i}_{j}"
            if key in self.phis and self.phis[key].mask.item() != 0:
                nbrs.append(i)
        return nbrs

    @torch.no_grad()
    def symbolicize_with_pysr(self, X: torch.Tensor, niterations: int = 60,
                               binary_operators=None, unary_operators=None,
                               parsimony: float = 1e-4, maxsize: int = 20):
        """
        For each node j, fit a multi-input symbolic model y_j = f(x_parents)
        using only 1-hop active parents of j. Store, for each target node, the
        highest R^2, the involved variables (global indices), the target node,
        and the discovered equation. Edge thickness in plotting is based on the
        stored node-level R^2.

        Notes:
        - Requires PySR; if unavailable, returns gracefully.
        - Does not alter forward computation; only records formulas and marks
          discovered edges as symbolic for visualization.
        """
        if not _PYSR_AVAILABLE:
            print("PySR not available; skipping symbolicize_with_pysr.")
            return

        if binary_operators is None:
            binary_operators = ["+", "-", "*", "/"]
        if unary_operators is None:
            unary_operators = ["sin", "cos", "exp", "log", "sqrt", "tanh", "square", "asin", "acos", "atanh"]

        X_np = X.detach().cpu().numpy()

        for j in range(self.num_nodes):
            parents = self._active_in_neighbors(j)
            if len(parents) == 0:
                continue
            Xj = X_np[:, parents]
            yj = X_np[:, j]

            try:
                # Build PySR model with best-effort variable naming across versions.
                # Prefer `variable_names`; fall back to no names if unsupported.
                model = None
                last_error = None
                try:
                    model = PySRRegressor(
                        niterations=niterations,
                        binary_operators=binary_operators,
                        unary_operators=unary_operators,
                        model_selection="best",
                        parsimony=parsimony,
                        maxsize=maxsize,
                        variable_names=[f"x{pi}" for pi in parents],
                    )
                except Exception as e_var:
                    last_error = e_var
                    try:
                        # Some older releases used `feature_names`; try it just in case.
                        model = PySRRegressor(
                            niterations=niterations,
                            binary_operators=binary_operators,
                            unary_operators=unary_operators,
                            model_selection="best",
                            parsimony=parsimony,
                            maxsize=maxsize,
                            feature_names=[f"x{pi}" for pi in parents],
                        )
                    except Exception as e_feat:
                        last_error = e_feat
                        # Final fallback: no names, will remap symbols post-hoc.
                        model = PySRRegressor(
                            niterations=niterations,
                            binary_operators=binary_operators,
                            unary_operators=unary_operators,
                            model_selection="best",
                            parsimony=parsimony,
                            maxsize=maxsize,
                        )
                model.fit(Xj, yj)
            except Exception as e:
                print(f"PySR failed for node {j}: {e}")
                continue

            # Fetch best sympy expression for this node
            try:
                expr = model.sympy(0)
            except Exception:
                expr = None

            # R^2 of node-level fit (for info)
            try:
                pred = model.predict(Xj)
                ss_res = float(((pred - yj) ** 2).sum())
                mu = float(yj.mean())
                ss_tot = float(((yj - mu) ** 2).sum()) + 1e-12
                r2_node = 1.0 - ss_res / ss_tot
            except Exception:
                r2_node = 0.0

            # If the expression uses local names x0..x{k-1}, remap them to global x{parents[l]}
            if expr is not None:
                try:
                    sym_idxs = []
                    for s in expr.free_symbols:
                        name = str(s)
                        if name.startswith("x") and name[1:].isdigit():
                            sym_idxs.append(int(name[1:]))
                    if len(sym_idxs) > 0 and all(0 <= idx < len(parents) for idx in sym_idxs):
                        sub_map = {sp.Symbol(f"x{li}"): sp.Symbol(f"x{parents[li]}") for li in range(len(parents))}
                        expr = sp.simplify(expr.xreplace(sub_map))
                except Exception:
                    pass

            # Extract involved variables as GLOBAL indices from expr symbols
            involved_vars = set()
            if expr is not None:
                try:
                    for s in expr.free_symbols:
                        name = str(s)
                        if name.startswith("x") and name[1:].isdigit():
                            involved_vars.add(int(name[1:]))
                except Exception:
                    involved_vars = set(parents)
            else:
                involved_vars = set(parents)

            # Store node-level equation info
            try:
                target_symbol = sp.Symbol(f"x{j}")
            except Exception:
                target_symbol = None

            # Also store string representation for easy display
            expr_str = None
            if expr is not None:
                try:
                    expr_str = str(expr)
                except Exception:
                    expr_str = None

            self.node_equations[j] = {
                'expr': expr,
                'expr_str': expr_str,
                'target': target_symbol,
                'vars': involved_vars,
                'r2': float(r2_node),
            }



    def plot_with_formula(self):
        """Plot graph using node-level symbolic equations.

        - Draw an (undirected) edge i—>j for each parent i involved in
          node j's equation stored in self.node_equations[j]['vars'].
        - Edge thickness for all edges pointing to the same target j is
          proportional to that node's R^2.
        - Display the equation of each target node near the node itself.
        - No per-edge function labels are shown.
        """
        import matplotlib.pyplot as plt
        import networkx as nx

        node_eqs = getattr(self, 'node_equations', {})

        # Graph init
        G_sym = nx.Graph()  # undirected style (no arrows)
        for n in range(self.num_nodes):
            G_sym.add_node(n)

        # Collect edges and widths from node-level info
        sym_edges = []
        edge_widths = []
        target_r2 = {}
        for j in range(self.num_nodes):
            info = node_eqs.get(j)
            if not info:
                continue
            r2 = float(info.get('r2', 0.0))
            vars_set = info.get('vars', set()) or set()
            if r2 <= 0.0 or len(vars_set) == 0:
                continue
            target_r2[j] = r2
            for i in vars_set:
                if i == j:
                    continue
                if i < 0 or i >= self.num_nodes:
                    continue
                G_sym.add_edge(i, j)
                sym_edges.append((i, j))
                edge_widths.append(1.5 + 6.0 * max(0.0, r2))

        # Labels for nodes
        node_labels = {i: f'x{i+1}' for i in range(self.num_inputs)}
        node_labels.update({self.num_inputs + i: f'h{i+1}' for i in range(self.num_hidden)})

        base_graph = G_sym
        pos = nx.spring_layout(base_graph, seed=42)

        # Draw
        plt.figure(figsize=(11, 8))
        node_colors = ['lightblue' if n < self.num_inputs else 'lightgreen' for n in range(self.num_nodes)]
        nx.draw_networkx_nodes(base_graph, pos, node_color=node_colors, node_size=1500, edgecolors='k')
        nx.draw_networkx_labels(base_graph, pos, labels=node_labels, font_size=12, font_weight='bold')

        if len(sym_edges) > 0:
            nx.draw_networkx_edges(base_graph, pos, edgelist=sym_edges, edge_color='red', width=edge_widths, arrows=False)

        # Annotate nodes with their equation strings
        for j, info in node_eqs.items():
            expr_str = info.get('expr_str') if info else None
            r2 = float(info.get('r2', 0.0)) if info else 0.0
            vars_set = info.get('vars', set()) if info else set()
            if expr_str and r2 > 0.0 and len(vars_set) > 0 and j in pos:
                x, y = pos[j]
                # Slight offset so it doesn't overlap the node label
                plt.text(x + 0.02, y - 0.06, f"f(x) = {expr_str}\nR²={r2:.3f}",
                         fontsize=9, color='darkred', ha='left', va='top',
                         bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='darkred', alpha=0.7))

        plt.title("GraphKAN: Node equations (edge width ∝ node R²)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Toy data
    num_samples = 5000
    x1 = torch.rand(num_samples) * 2 - 1
    x2 = torch.rand(num_samples) * 2 - 1
    x3 = x1 + torch.cos(x2)
    x4 = x2 + torch.sin(x1)

    num_hidden = 0
    # Assemble features dynamically: [inputs | hidden placeholders]
    num_inputs = 4
    features = torch.cat([
        x1.unsqueeze(1),
        x2.unsqueeze(1),
        x3.unsqueeze(1),
        x4.unsqueeze(1),
        torch.zeros(num_samples, num_hidden),
    ], dim=1)

    train_features = features[:800]
    test_features = features[800:]

    # Model (aligned with rawlvl0 defaults for fair comparison)
    model = GraphKAN(num_inputs=num_inputs, num_hidden=num_hidden,
                     num_intervals=15, spline_order=3, grid_range=[-1, 1], num_layers=1,
                     use_layernorm=False)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    def train(epochs=1000, weight=1):
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            out = model(train_features)

            loss = F.mse_loss(out[:, :num_inputs], train_features[:, :num_inputs])
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                model.eval()
                with torch.no_grad():
                    test_out = model(test_features)
                    test_mse = F.mse_loss(test_out[:, :num_inputs], test_features[:, :num_inputs])
                print(f"Epoch {epoch} | Test MSE: {test_mse:.6f} | Total Loss: {loss.item():.4f}")
            if epoch % 20 == 0 and epoch > 0:
                for name, phi in model.phis.items():
                    i = int(name.split('_')[0])
                    phi.update_grid_from_samples(train_features[:, i:i+1])

    # Prune & symbolic
    # train(weight=1)
    # model.prune(threshold=1e-5)
    # model.eval()
    # with torch.no_grad():
    #     test_out = model(test_features)
    #     test_mse = F.mse_loss(test_out[:, :num_inputs], test_features[:, :num_inputs])
    # print(f"Pruned MSE: {test_mse:.6f}")

    # Try PySR multi-input symbolicization if available; else fallback to per-edge auto_symbolic

    print("Running symbolicize_with_pysr (multi-input → one output)...")
    model.symbolicize_with_pysr(train_features, niterations=40)


    model.eval()
    with torch.no_grad():
        test_out = model(test_features)
        test_mse = F.mse_loss(test_out[:, :num_inputs], test_features[:, :num_inputs])
    print(f"Symbolised MSE: {test_mse:.6f}")

    model.plot_with_formula()