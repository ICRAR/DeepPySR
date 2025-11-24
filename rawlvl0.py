import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import sympy as sp
from scipy.optimize import curve_fit

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

""" Centralized function library registry - Provides NumPy, Torch, and SymPy callables for each named function """
FUNCTION_LIBRARY = {
    # Identity and powers
    'x': {
        'np': (lambda x: x),
        'torch': (lambda x: x),
        'sympy': (lambda v: v),
        'inverse': 'x',
    },
    'x^2': {
        'np': (lambda x: x**2),
        'torch': (lambda x: x**2),
        'sympy': (lambda v: v**2),
        'inverse': 'sqrt',
    },
    'x^3': {
        'np': (lambda x: x**3),
        'torch': (lambda x: x**3),
        'sympy': (lambda v: v**3),
        'inverse': 'cbrt',
    },
    # Exponentials and logarithms
    'exp': {
        'np': np.exp,
        'torch': torch.exp,
        'sympy': sp.exp,
    },
    'ln': {
        'np': np.log,
        'torch': torch.log,
        'sympy': sp.log,
        'inverse': 'exp',
    },
    # Roots
    'sqrt': {
        'np': (lambda x: np.sqrt(np.abs(x))),
        'torch': (lambda x: torch.sqrt(torch.abs(x))),
        'sympy': (lambda v: sp.sqrt(sp.Abs(v))),
        'inverse': 'x^2',
    },
    'cbrt': {
        'np': np.cbrt,
        'torch': (lambda x: torch.sign(x) * torch.pow(torch.abs(x) + 1e-12, 1.0 / 3.0)),
        'sympy': (lambda v: sp.real_root(v, 3)),
        'inverse': 'x^3',
    },
    # Trigonometric
    'sin': {
        'np': np.sin,
        'torch': torch.sin,
        'sympy': sp.sin,
        'inverse': 'arcsin',
    },
    'arcsin': {
        'np': np.arcsin,
        'torch': torch.asin,
        'sympy': sp.asin,
        'inverse': 'sin',
    },
    'tan': {
        'np': np.tan,
        'torch': torch.tan,
        'sympy': sp.tan,
        'inverse': 'arctan',
    },
    'arctan': {
        'np': np.arctan,
        'torch': torch.atan,
        'sympy': sp.atan,
        'inverse': 'tan',
    },
    # Hyperbolic
    'tanh': {
        'np': np.tanh,
        'torch': torch.tanh,
        'sympy': sp.tanh,
        'inverse': 'atanh',
    },
    'atanh': {
        'np': np.arctanh,
        'torch': torch.atanh,
        'sympy': sp.atanh,
        'inverse': 'tanh',
    },
    # Others
    'abs': {
        'np': np.abs,
        'torch': torch.abs,
        'sympy': sp.Abs,
    },
}

# B-spline basis function (Cox-de Boor recursion) in pure PyTorch
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

# Evaluate spline curve: sum c_i * B_i(x)
def coef2curve(x_eval, grid, coef, k):
    b_splines = B_batch(x_eval, grid, k=k)
    y_eval = torch.einsum('ijk,jlk->ijl', b_splines, coef)
    # Always return (batch, in_dim, out_dim); callers can squeeze if needed
    return y_eval

# Fit coefficients to data via least squares
def curve2coef(x_eval, y_eval, grid, k):
    batch = x_eval.shape[0]
    in_dim = x_eval.shape[1]  # 1
    out_dim = y_eval.shape[2]  # 1
    n_coef = grid.shape[1] - k - 1
    mat = B_batch(x_eval, grid, k)
    mat = mat.permute(1, 0, 2)[:, None, :, :].expand(in_dim, out_dim, batch, n_coef)
    # Targets shaped as (in_dim, out_dim, batch, 1)
    y_eval = y_eval.permute(1, 2, 0).unsqueeze(dim=3)
    # Batched least squares: solution has shape (in_dim, out_dim, n_coef, 1)
    coef = torch.linalg.lstsq(mat, y_eval).solution.squeeze(-1)
    return coef

# Extend grid for boundary knots
def extend_grid(grid, k_extend=0):
    h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)
    for i in range(k_extend):
        grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
        grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)
    return grid

# KAN Layer with your robust symbolic fitting
class KANLayer(nn.Module):
    def __init__(self, in_dim=1, out_dim=1, num=15, k=3, noise_scale=0.1, scale_base=1.0, scale_sp=1.0, base_fun=nn.SiLU(), grid_eps=0.02, grid_range=[-3, 3], sp_trainable=True, sb_trainable=True):
        super(KANLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num = num
        self.k = k
        grid = torch.linspace(grid_range[0], grid_range[1], steps=num + 1)[None, :].expand(in_dim, num + 1)
        self.grid = nn.Parameter(extend_grid(grid, k_extend=k), requires_grad=False)
        noises = (torch.rand(self.num + 1, self.in_dim, self.out_dim) - 0.5) * noise_scale / num
        # noises already has shape (batch=num+1, in_dim, out_dim); pass directly
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
            y_base = torch.tensor(self.c, device=x.device) * self.fun(x) + torch.tensor(self.d, device=x.device)  # (batch, in_dim)
            y = y_base[:, :, None]  # (batch, in_dim, 1)
            y = self.mask[None, :, :] * y
            y = y.sum(dim=1)  # (batch, 1)
            postacts = y.clone()
            postspline = torch.zeros_like(y)
        else:
            base = self.base_fun(x)  # (batch, in_dim)
            spline = coef2curve(x, self.grid, self.coef, self.k)  # (batch, in_dim, out_dim)
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
        # y_eval has shape (batch, in_dim, out_dim); pass directly
        self.coef.data = curve2coef(x_pos, y_eval, self.grid, self.k)

    def prune(self, threshold=5e-3):
        norm = self.scale_base.abs() + self.scale_sp.abs() * self.coef.abs().mean()
        if norm.item() < threshold:
            self.mask.data = torch.zeros_like(self.mask)

    def auto_symbolic(self, lib=None, threshold=0.95):
        if self.mask.item() == 0:
            return
        print("Fitting symbolic for this layer")
        x_np = np.linspace(self.grid_range[0], self.grid_range[1], 1000)
        x_t = torch.from_numpy(x_np).float().unsqueeze(1)
        with torch.no_grad():
            y_t, _, _, _ = self(x_t)
        y_np = y_t.squeeze().numpy()
        candidate_names = list(FUNCTION_LIBRARY.keys()) if lib is None else [n for n in lib if n in FUNCTION_LIBRARY]
        candidate_names = [n for n in candidate_names if callable(FUNCTION_LIBRARY[n].get('np', None))]
        fun_dict_np = {n: (lambda fn: (lambda x, c, d: c * fn(x) + d))(FUNCTION_LIBRARY[n]['np']) for n in candidate_names}
        fun_dict_torch = {n: FUNCTION_LIBRARY[n]['torch'] for n in candidate_names if 'torch' in FUNCTION_LIBRARY[n]}
        sym_dict = {n: FUNCTION_LIBRARY[n]['sympy'] for n in candidate_names if 'sympy' in FUNCTION_LIBRARY[n]}
        best_r2 = -1
        best_name = None
        best_popt = None
        for name in candidate_names:
            if name not in fun_dict_np:
                continue
            fit_fun = fun_dict_np[name]
            base_np_fun = FUNCTION_LIBRARY[name]['np']
            with np.errstate(all='ignore'):
                fvals = base_np_fun(x_np)
                valid_mask = np.isfinite(fvals) & np.isfinite(y_np)
                if valid_mask.sum() < 50:
                    continue
                x_fit = x_np[valid_mask]
                y_fit = y_np[valid_mask]
                try:
                    popt, _ = curve_fit(fit_fun, x_fit, y_fit, p0=[1.0, 0.0], maxfev=10000)
                    pred = fit_fun(x_fit, *popt)
                    if not np.all(np.isfinite(pred)):
                        continue
                    ss_res = np.sum((y_fit - pred)**2)
                    ss_tot = np.sum((y_fit - y_fit.mean())**2)
                    r2 = 1 - ss_res / (ss_tot + 1e-6)
                    if r2 > best_r2:
                        best_r2 = r2
                        best_name = name
                        best_popt = popt
                except:
                    continue
        if best_r2 > threshold:
            self.sym_name = best_name
            self.c, self.d = best_popt
            self.fun = fun_dict_torch.get(best_name, lambda x: x)
            self.sym_fun = sym_dict.get(best_name, lambda v: v)
            self.symbolic_enabled = True
            self.best_r2 = best_r2
            print(f"Best fit: {best_name} with R2 {best_r2:.4f}")
        else:
            print("No good symbolic fit found")

    def symbolic_formula(self, var='x'):
        if not self.symbolic_enabled:
            return None
        v = sp.symbols(var)
        formula = self.c * self.sym_fun(v) + self.d
        return sp.simplify(formula)

# Graph KAN with hidden (empty) nodes and multi-layer message passing
class GraphKAN(nn.Module):
    def __init__(self, num_inputs=3, num_hidden=3, num_outputs=1, num_intervals=15, spline_order=3, grid_range=[-3, 3], num_layers=2):
        super(GraphKAN, self).__init__()
        self.num_nodes = num_inputs + num_hidden + num_outputs
        self.num_layers = num_layers
        self.y_index = self.num_nodes - 1
        self.input_indices = list(range(num_inputs))
        self.hidden_indices = list(range(num_inputs, num_inputs + num_hidden))
        self.phis = nn.ModuleDict()
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j:
                    continue
                # All x to y: one-way from x to y
                if i in self.input_indices and j == self.y_index:
                    self.phis[f'{i}_{j}'] = KANLayer(in_dim=1, out_dim=1, num=num_intervals, k=spline_order, grid_range=grid_range, base_fun=nn.SiLU())
                    continue
                # All h to y: one-way from h to y
                if i in self.hidden_indices and j == self.y_index:
                    self.phis[f'{i}_{j}'] = KANLayer(in_dim=1, out_dim=1, num=num_intervals, k=spline_order, grid_range=grid_range, base_fun=nn.SiLU())
                    continue
                # Bi-directional between x and x, x and h, h and h
                if (i in self.input_indices and j in self.input_indices) or \
                        (i in self.input_indices and j in self.hidden_indices) or \
                        (i in self.hidden_indices and j in self.hidden_indices):
                    self.phis[f'{i}_{j}'] = KANLayer(in_dim=1, out_dim=1, num=num_intervals, k=spline_order, grid_range=grid_range, base_fun=nn.SiLU())

    def update_step(self, x):
        batch_size = x.shape[0]
        new_x = torch.zeros_like(x)
        for j in range(self.num_nodes):
            msgs = 0
            for i in range(self.num_nodes):
                if i != j:
                    key = f'{i}_{j}'
                    if key in self.phis:
                        phi = self.phis[key]
                        msg, _, _, _ = phi(x[:, i].unsqueeze(1))
                        msgs += msg.squeeze(1)
            new_x[:, j] = msgs
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
                phi.auto_symbolic(threshold=0.95)

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

# Toy data
num_samples = 1000
x1 = torch.rand(num_samples) * 2 - 1
x2 = torch.rand(num_samples) * 2 - 1
x3 = x1 + torch.sin(x2)
y_real = x3 + torch.exp(x1)
num_hidden = 0
features = torch.cat([x1.unsqueeze(1), x2.unsqueeze(1), x3.unsqueeze(1), torch.zeros(num_samples, num_hidden), torch.zeros(num_samples, 1)], dim=1)
num_train = int(num_samples * 0.8)
train_features = features[:num_train]
train_y = y_real[:num_train]
test_features = features[num_train:]
test_y = y_real[num_train:]

# Model
model = GraphKAN(num_inputs=3, num_hidden=num_hidden, num_outputs=1, num_intervals=15, spline_order=3, grid_range=[-1, 1], num_layers=1)
optimizer = optim.Adam(model.parameters(), lr=0.005)


def train():
    for epoch in range(600):
        optimizer.zero_grad()
        output = model(train_features)
        pred_y = output[:, model.y_index]
        main_loss = F.mse_loss(pred_y, train_y)
        rec_loss = sum(F.mse_loss(output[:, j], train_features[:, j]) for j in model.input_indices)
        loss = main_loss + rec_loss
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            test_pred = model(test_features)[:, model.y_index]
            test_loss = F.mse_loss(test_pred, test_y)
            print(f"Epoch {epoch}: Train {loss.item():.4f}, Test {test_loss.item():.4f}")
        if epoch % 50 == 0 and epoch > 0:
            for name, phi in model.phis.items():
                i = int(name.split('_')[0])
                phi.update_grid_from_samples(train_features[:, i].unsqueeze(1))

# Prune & symbolic
train()
model.prune()
model.auto_symbolic()
model.resolve_bidirectional()

# Print discovered edges
node_names = ['x1', 'x2', 'x3'] + [f'h{i+1}' for i in range(num_hidden)] + ['y']
print("\nDiscovered symbolic edges:")
for name, phi in model.phis.items():
    i, j = map(int, name.split('_'))
    if phi.mask.item() == 0 or not phi.symbolic_enabled:
        continue
    formula = phi.symbolic_formula(var=node_names[i])
    print(f"{node_names[i]} → {node_names[j]} : {formula}")

# Compose full symbolic y
print("\nComposed symbolic y:")
x1_sym, x2_sym, x3_sym = sp.symbols('x1 x2 x3')
current = [x1_sym, x2_sym, x3_sym] + [0] * num_hidden + [0]
non_symbolic_edges = False
for layer in range(model.num_layers):
    new_state = [0] * model.num_nodes
    for j in range(model.num_nodes):
        msgs = 0
        has_non_sym = False
        for i in range(model.num_nodes):
            if i == j:
                continue
            key = f'{i}_{j}'
            if key not in model.phis:
                continue
            phi = model.phis[key]
            if phi.mask.item() == 0:
                continue
            if phi.symbolic_enabled:
                f = phi.symbolic_formula(var='tmp')
                msgs += f.subs('tmp', current[i])
            else:
                has_non_sym = True
        if has_non_sym:
            non_symbolic_edges = True
        new_state[j] = sp.simplify(msgs)
    current = new_state
if non_symbolic_edges:
    print("Partial (some edges non-symbolic)")
print(sp.simplify(current[-1]))

import networkx as nx
import matplotlib.pyplot as plt
# After model.auto_symbolic()
# Helper: round all floating constants in a SymPy expression to a fixed number of decimal places
def _round_sympy_expr(expr, decimals=1):
    if expr is None:
        return None
    floats = list(expr.atoms(sp.Float))
    if not floats:
        return expr
    repl = {}
    for a in floats:
        try:
            repl[a] = sp.Float(f"{float(a):.{decimals}f}")
        except Exception:  # Fallback: keep the original if conversion fails
            repl[a] = a
    return expr.xreplace(repl)
# Create a directed graph
G = nx.DiGraph()
# Add nodes
for idx, name in enumerate(node_names):
    G.add_node(name)
# Add symbolic edges with labels
for name, phi in model.phis.items():
    i, j = map(int, name.split('_'))
    if phi.mask.item() == 0 or not phi.symbolic_enabled:
        continue
    expr = phi.symbolic_formula(var=node_names[i])
    expr = _round_sympy_expr(expr, decimals=1)
    formula_str = str(expr)
    G.add_edge(node_names[i], node_names[j], label=formula_str)
# Position nodes (layered layout for clarity: inputs left, hidden middle, output right)
pos = {}
input_nodes = node_names[:3]
hidden_nodes = node_names[3:-1]
output_node = node_names[-1]
for idx, node in enumerate(input_nodes):
    pos[node] = (0, idx - len(input_nodes)/2)
for idx, node in enumerate(hidden_nodes):
    pos[node] = (1, idx - len(hidden_nodes)/2)
pos[output_node] = (2, 0)
# Draw the graph
plt.figure(figsize=(10, 8))
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=12, arrowsize=20)
nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'label'), font_size=12)
plt.title('GraphKAN Model Structure with Symbolic Edges')
plt.axis('off')
plt.show()