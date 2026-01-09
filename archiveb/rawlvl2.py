import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from scipy.optimize import curve_fit
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import networkx as nx

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
        if self.symbolic_enabled:
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
    def __init__(self, num_inputs=3, num_hidden=2, num_intervals=15, spline_order=3, grid_range=[-3, 3], num_layers=2,
                 use_layernorm=True):
        super(GraphKAN, self).__init__()
        # Store topology
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_nodes = num_inputs + num_hidden
        self.num_layers = num_layers
        self.use_layernorm = use_layernorm
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
        if self.use_layernorm:
            self.ln = nn.LayerNorm(self.num_nodes)
        else:
            self.ln = None

    def update_step(self, x):
        batch = x.shape[0]
        new_x = torch.zeros_like(x)
        for j in range(self.num_nodes):
            msgs = 0.0
            for i in range(self.num_nodes):
                if f'{i}_{j}' in self.phis:
                    phi = self.phis[f'{i}_{j}']
                    msg, _, _, _ = phi(x[:, i:i+1])
                    msgs = msgs + msg.squeeze(1)
            if j < self.num_inputs:           # input nodes: predict from others
                new_x[:, j] = msgs            # no residual path
            else:                             # hidden nodes: allow residual
                new_x[:, j] = x[:, j] + msgs

        # Optional layer normalization
        if self.use_layernorm and self.ln is not None:
            new_x = self.ln(new_x)
        return new_x

    def forward(self, x):
        x = x.clone()
        for _ in range(self.num_layers):
            x = self.update_step(x)
        return x

    def prune(self, threshold=5e-3):
        for phi in self.phis.values():
            phi.prune(threshold)

    def auto_symbolic(self):
        for name, phi in self.phis.items():
            if phi.mask.item() > 0:
                print(name)
                phi.auto_symbolic(threshold=0.85)

    def resolve_bidirectional(self):
        for i in range(self.num_nodes):
            for j in range(i+1, self.num_nodes):
                key_ij = f'{i}_{j}'
                key_ji = f'{j}_{i}'
                if key_ij in self.phis and key_ji in self.phis:
                    phi_ij = self.phis[key_ij]
                    phi_ji = self.phis[key_ji]
                    if phi_ij.symbolic_enabled and phi_ji.symbolic_enabled:
                        if phi_ij.best_r2 > phi_ji.best_r2:
                            phi_ji.symbolic_enabled = False
                            phi_ji.mask.data = torch.zeros_like(phi_ji.mask)
                        else:
                            phi_ij.symbolic_enabled = False
                            phi_ij.mask.data = torch.zeros_like(phi_ij.mask)



    def plot_with_formula(self, layout='spring', seed=42, k=None, scale=2.0, jitter=0.0,
                           show_nonsymbolic=False,
                           r2_influence=True, r2_power=1.0,
                           spring_weight_scale=5.0,
                           kamada_min_dist=0.3, kamada_max_dist=3.0):
        """
        Plot the discovered graph.

        Args:
            layout: 'spring' (scatter), 'kamada', or 'layered'. Default 'spring'.
            seed: random seed for spring layout.
            k: optimal distance between nodes for spring layout (passed to networkx.spring_layout).
            scale: scale factor for spring layout.
            jitter: add small random jitter to positions in layered layout to avoid exact alignment.
            show_nonsymbolic: if True, also draw faint gray edges for active but non-symbolic connections.
        """
        G_sym = nx.DiGraph()
        G_all = nx.DiGraph()
        edge_labels = {}
        edge_r2 = {}

        # Add all nodes first so layouts place isolated nodes too
        for n in range(self.num_nodes):
            G_sym.add_node(n)
            G_all.add_node(n)

        # Collect edges
        sym_edges = []
        nonsym_edges = []
        for name, phi in self.phis.items():
            i, j = map(int, name.split('_'))
            if phi.mask.item() != 0:
                G_all.add_edge(i, j)
                if phi.symbolic_enabled:
                    G_sym.add_edge(i, j)
                    sym_edges.append((i, j))
                    edge_labels[(i, j)] = f"{phi.c:+.2f}*{phi.sym_name}"
                    # Cache R2 for layout/width scaling
                    try:
                        edge_r2[(i, j)] = float(getattr(phi, 'best_r2', 0.0))
                    except Exception:
                        edge_r2[(i, j)] = 0.0
                elif show_nonsymbolic:
                    nonsym_edges.append((i, j))

        # Node labels
        node_labels = {i: f'x{i+1}' for i in range(self.num_inputs)}
        node_labels.update({self.num_inputs + i: f'h{i+1}' for i in range(self.num_hidden)})

        # Choose layout
        if layout == 'spring':
            base_graph = G_sym if len(G_sym.edges) > 0 else G_all
            # Apply R² as spring weights: higher R² -> stronger attraction (closer)
            if r2_influence and len(base_graph.edges) > 0:
                for (u, v) in base_graph.edges:
                    r2 = edge_r2.get((u, v), 0.0)
                    w = 1.0 + spring_weight_scale * max(0.0, r2) ** r2_power
                    base_graph[u][v]['weight'] = w
            pos = nx.spring_layout(base_graph, seed=seed, k=k, scale=scale, weight='weight')
            # Ensure positions for any nodes missing due to isolated in base_graph
            for n in range(self.num_nodes):
                if n not in pos:
                    pos[n] = np.random.default_rng(seed + n).random(2) * scale
        elif layout == 'kamada':
            base_graph = G_sym if len(G_sym.edges) > 0 else G_all
            # Kamada-Kawai uses edge 'weight' as distances in shortest paths.
            # Map higher R² to shorter desired distances.
            if r2_influence and len(base_graph.edges) > 0:
                dmin, dmax = kamada_min_dist, kamada_max_dist
                for (u, v) in base_graph.edges:
                    r2 = edge_r2.get((u, v), 0.0)
                    # distance decreases with r2
                    dist = dmax - (dmax - dmin) * max(0.0, r2) ** r2_power
                    base_graph[u][v]['weight'] = max(1e-6, float(dist))
            pos = nx.kamada_kawai_layout(base_graph, scale=scale, weight='weight')
            for n in range(self.num_nodes):
                if n not in pos:
                    pos[n] = np.random.default_rng(seed + n).random(2) * scale
        else:  # layered
            pos = {}
            for i in range(self.num_inputs):
                y = 0.0 + (np.random.rand() - 0.5) * jitter if jitter > 0 else 0.0
                pos[i] = (i - (self.num_inputs - 1) / 2, y)
            for i in range(self.num_hidden):
                y = 1.5 + (np.random.rand() - 0.5) * jitter if jitter > 0 else 1.5
                pos[self.num_inputs + i] = (i * 1.5 - (self.num_hidden - 1) * 0.75, y)

        # Draw
        plt.figure(figsize=(11, 8))

        # Color-code inputs vs hidden
        node_colors = []
        for n in range(self.num_nodes):
            node_colors.append('lightblue' if n < self.num_inputs else 'lightgreen')

        nx.draw_networkx_nodes(G_all, pos, node_color=node_colors, node_size=1800, edgecolors='k')
        nx.draw_networkx_labels(G_all, pos, labels=node_labels, font_size=14, font_weight='bold')

        # Optional non-symbolic edges (faint gray)
        if show_nonsymbolic and len(nonsym_edges) > 0:
            nx.draw_networkx_edges(
                G_all, pos, edgelist=nonsym_edges, edge_color='lightgray', width=1.5, alpha=0.6,
                arrows=True, arrowstyle='-|>', arrowsize=18, connectionstyle='arc3,rad=0.1'
            )

        # Symbolic edges (highlighted red); width scaled by R² if available
        if len(sym_edges) > 0:
            if r2_influence:
                widths = [2.0 + 4.0 * max(0.0, edge_r2.get(e, 0.0)) ** r2_power for e in sym_edges]
            else:
                widths = 3.0
            nx.draw_networkx_edges(
                G_sym, pos, edgelist=sym_edges, edge_color='red', width=widths, alpha=0.95,
                arrows=True, arrowstyle='-|>', arrowsize=22, connectionstyle='arc3,rad=0.15'
            )
            nx.draw_networkx_edge_labels(G_sym, pos, edge_labels=edge_labels, font_size=11, font_color='darkred')

        plt.title("GraphKAN: Discovered symbolic links", fontsize=18, pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

# Toy data
num_samples = 5000
x1 = torch.rand(num_samples) * 2 - 1
x2 = torch.rand(num_samples) * 2 - 1
x3 = x1+torch.cos(x2)
x4 = x2+torch.sin(x1)

num_hidden = 0
# Assemble features dynamically: [inputs | hidden placeholders | output placeholder]
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
train(weight=1)
model.prune(threshold=1e-5)
model.eval()
with torch.no_grad():
    test_out = model(test_features)
    test_mse = F.mse_loss(test_out[:, :num_inputs], test_features[:, :num_inputs])
print(f"Pruned MSE: {test_mse:.6f}")
model.auto_symbolic()
model.eval()
with torch.no_grad():
    test_out = model(test_features)
    test_mse = F.mse_loss(test_out[:, :num_inputs], test_features[:, :num_inputs])
print(f"Symbolised MSE: {test_mse:.6f}")
model.resolve_bidirectional()

model.eval()
with torch.no_grad():
    test_out = model(test_features)
    test_mse = F.mse_loss(test_out[:, :num_inputs], test_features[:, :num_inputs])
print(f"Directional MSE: {test_mse:.6f}")

# --- Print identified formulas similar to rawlvl0 ---

# Helper: round all floating constants in a SymPy expression to a fixed number of decimal places
def _round_sympy_expr(expr, decimals=2):
    if expr is None:
        return None
    floats = list(expr.atoms(sp.Float))
    if not floats:
        return expr
    repl = {}
    for a in floats:
        try:
            repl[a] = sp.Float(f"{float(a):.{decimals}f}")
        except Exception:
            repl[a] = a
    return expr.xreplace(repl)

# Build node names dynamically
node_names = [f'x{i+1}' for i in range(num_inputs)] + [f'h{i+1}' for i in range(num_hidden)] + ['y']

print("\nDiscovered symbolic edges:")
for name, phi in model.phis.items():
    i, j = map(int, name.split('_'))
    if phi.mask.item() == 0 or not phi.symbolic_enabled:
        continue
    expr = phi.symbolic_formula(var=node_names[i])
    expr = _round_sympy_expr(expr, decimals=2)
    print(f"{node_names[i]} → {node_names[j]} : {expr}")


model.plot_with_formula()