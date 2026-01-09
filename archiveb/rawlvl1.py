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

FUNCTION_LIBRARY = {
    'sin': {'np': np.sin, 'torch': torch.sin, 'sympy': sp.sin},
    'cos': {'np': np.cos, 'torch': torch.cos, 'sympy': sp.cos},
    'exp': {'np': np.exp, 'torch': torch.exp, 'sympy': sp.exp},
    'log': {'np': np.log, 'torch': torch.log, 'sympy': sp.log},
    'sqrt': {'np': np.sqrt, 'torch': torch.sqrt, 'sympy': sp.sqrt},
    'tanh': {'np': np.tanh, 'torch': torch.tanh, 'sympy': sp.tanh},
    'sigmoid': {'np': lambda x: 1/(1+np.exp(-x)), 'torch': torch.sigmoid, 'sympy': lambda v: 1/(1 + sp.exp(-v))},
    'identity': {'np': lambda x: x, 'torch': lambda x: x, 'sympy': lambda v: v},
    'square': {'np': lambda x: x**2, 'torch': lambda x: x**2, 'sympy': lambda v: v**2},
    'cube': {'np': lambda x: x**3, 'torch': lambda x: x**3, 'sympy': lambda v: v**3},
    'inv': {'np': lambda x: 1/x, 'torch': lambda x: 1/x, 'sympy': lambda v: 1/v},
    'gauss': {'np': lambda x: np.exp(-x**2), 'torch': lambda x: torch.exp(-x**2), 'sympy': lambda v: sp.exp(-v**2)},
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
    def __init__(self, num_inputs=3, num_hidden=2, num_outputs=1, num_intervals=15, spline_order=3, grid_range=[-3, 3], num_layers=2,
                 use_layernorm=True):
        super(GraphKAN, self).__init__()
        # Store topology
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.num_nodes = num_inputs + num_hidden + num_outputs
        self.num_layers = num_layers
        self.use_layernorm = use_layernorm
        # Index of the single output node (assumes num_outputs == 1)
        self.y_index = self.num_nodes - 1
        self.input_indices = list(range(num_inputs))
        self.hidden_indices = list(range(num_inputs, num_inputs + num_hidden))
        self.phis = nn.ModuleDict()
        # Build edges according to connectivity rules (no hidden -> input edges)
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j:
                    continue
                # Any input or hidden can point to y (output receiver only)
                if i in self.input_indices and j == self.y_index:
                    self.phis[f'{i}_{j}'] = KANLayer(
                        num=num_intervals, k=spline_order, grid_range=grid_range, base_fun=nn.SiLU()
                    )
                # Input source: allow to inputs and hidden
                if i in self.hidden_indices and j == self.y_index:
                    self.phis[f'{i}_{j}'] = KANLayer(
                        num=num_intervals, k=spline_order, grid_range=grid_range, base_fun=nn.SiLU()
                    )
                # Hidden source: allow only to hidden (no hidden -> input)
                if (i in self.input_indices and j in self.input_indices) or \
                    (i in self.input_indices and j in self.hidden_indices) or \
                    (i in self.hidden_indices and j in self.hidden_indices):
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
                if i != j and f'{i}_{j}' in self.phis:
                    phi = self.phis[f'{i}_{j}']
                    msg, _, _, _ = phi(x[:, i:i+1])
                    msgs = msgs + msg.squeeze(1)
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

    def get_full_symbolic_formula(self):
        """Compose y purely in terms of original inputs by back-substituting hidden nodes.

        Returns a LaTeX string like 'y = ...' with all hidden variables eliminated.
        If some contributing edges are non-symbolic, it will still compose using the
        available symbolic edges (effectively ignoring the non-symbolic ones).
        """
        expr, _partial = self.compose_symbolic_y(decimals=2)
        if expr is None:
            return "y = (no symbolic edges found)"
        return f"y = {sp.latex(expr)}"

    def compose_symbolic_y(self, decimals=2):
        """Returns (expr, partial) where expr is SymPy expression for y in terms of inputs only.

        - Uses only edges with identified symbolic functions (phi.symbolic_enabled and mask>0).
        - Propagates symbols across self.num_layers to back-substitute hidden nodes.
        - Rounds floating constants inside the final expression to the given decimals.
        - partial=True if some encountered edges into a node were non-symbolic (and thus ignored).
        """
        # Create SymPy symbols for inputs
        input_syms = sp.symbols(' '.join([f'x{i+1}' for i in range(self.num_inputs)]))
        hidden_syms = sp.symbols(' '.join([f'h{i+1}' for i in range(self.num_hidden)])) if self.num_hidden > 0 else ()
        output_sym = sp.symbols('y')
        if self.num_inputs == 1:
            input_syms = (input_syms,)  # ensure tuple-like access
        if self.num_hidden == 1:
            hidden_syms = (hidden_syms,)
        if self.num_outputs == 1:
            output_sym = (output_sym,)
        # Initialize state: inputs are symbols; hidden and y start at 0
        current = list(input_syms) + list(hidden_syms) + list(output_sym)
        partial = False
        new_state = [0] * self.num_nodes
        for j in range(self.num_nodes):
            msgs = 0
            has_non_sym = False
            for i in range(self.num_nodes):
                if i == j:
                    continue
                key = f'{i}_{j}'
                if key not in self.phis:
                    continue
                phi = self.phis[key]
                if phi.mask.item() == 0:
                    continue
                if phi.symbolic_enabled:
                    f_expr = phi.symbolic_formula(var='tmp')
                    msgs += f_expr.subs('tmp', current[i])
                else:
                    has_non_sym = True
            if has_non_sym:
                partial = True
            new_state[j] = msgs
        
        # If hidden symbols appear in y, substitute them with their composed expressions
        # until no hidden symbols remain or no further change occurs.
        y_expr = new_state[self.y_index]
        if self.num_hidden > 0:
            # Build substitution map: h_k -> new_state[num_inputs + k]
            subs_map = {}
            for k in range(self.num_hidden):
                h_sym = sp.symbols(f'h{k+1}')
                subs_map[h_sym] = sp.simplify(new_state[self.num_inputs + k])

            # Iteratively substitute to resolve nested hidden dependencies
            max_iters = max(1, self.num_hidden * 3)
            for _ in range(max_iters):
                # Stop if there are no hidden symbols left in y_expr
                if not any(sp.symbols(f'h{k+1}') in y_expr.free_symbols for k in range(self.num_hidden)):
                    break
                new_y = y_expr.xreplace(subs_map)
                if new_y == y_expr:
                    break
                y_expr = sp.simplify(new_y)
            
        
        # Helper to round all Float constants inside a SymPy expression
        def _round_expr(e, n):
            if e is None:
                return None
            floats = list(e.atoms(sp.Float))
            if not floats:
                return e
            repl = {}
            for a in floats:
                try:
                    repl[a] = sp.Float(f"{float(a):.{n}f}")
                except Exception:
                    repl[a] = a
            return e.xreplace(repl)

        # Final simplify after substitutions above
        y_expr = sp.simplify(y_expr)
        if y_expr == 0:
            # No symbolic path reached y
            return None, partial
        y_expr = _round_expr(y_expr, decimals)
        return y_expr, partial

    def plot_with_formula(self):
        G = nx.DiGraph()
        pos = {}
        edge_labels = {}
        edge_colors = []

        # Layout
        for i in range(self.num_inputs):
            pos[i] = (i - (self.num_inputs - 1) / 2, 0)
        for i in range(self.num_hidden):
            pos[self.num_inputs + i] = (i * 1.5 - (self.num_hidden - 1) * 0.75, 1.5)
        pos[self.y_index] = (0.5, 3)

        node_labels = {i: f'x{i+1}' for i in range(self.num_inputs)}
        node_labels.update({self.num_inputs + i: f'h{i+1}' for i in range(self.num_hidden)})
        node_labels[self.y_index] = 'y'

        for name, phi in self.phis.items():
            if phi.mask.item() != 0 and phi.symbolic_enabled:
                i, j = map(int, name.split('_'))
                G.add_edge(i, j)
                # Annotate symbolic edges with a red arrow and a label
                label = f"{phi.c:+.2f}*{phi.sym_name}"
                edge_labels[(i, j)] = label
                edge_colors.append('red')

        plt.figure(figsize=(11, 8))
        nx.draw(G, pos, with_labels=True, labels=node_labels, node_color='lightcyan',
                node_size=3000, font_size=16, font_weight='bold', arrows=True,
                arrowstyle='->', arrowsize=25, edge_color=edge_colors, width=2.5)

        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=3, alpha=0.9)
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=11, font_color='darkblue')

        formula = self.get_full_symbolic_formula()
        plt.text(0.02, 0.02, formula, transform=plt.gca().transAxes,
                 fontsize=15, verticalalignment='bottom', bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9))

        plt.title("GraphKAN: Full Symbolic Discovery (including hidden nodes)", fontsize=18, pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

# Toy data
num_samples = 1000
x1 = torch.rand(num_samples) * 2 - 1
x2 = torch.rand(num_samples) * 2 - 1
x3 = torch.rand(num_samples) * 2 - 1
y_real = x3 +torch.sin(x2) + torch.exp(x1)

num_hidden = 0
# Assemble features dynamically: [inputs | hidden placeholders | output placeholder]
num_inputs = 3
num_outputs = 1
features = torch.cat([
    x1.unsqueeze(1),
    x2.unsqueeze(1),
    x3.unsqueeze(1),
    torch.zeros(num_samples, num_hidden),
    torch.zeros(num_samples, num_outputs)
], dim=1)

train_features = features[:800]
train_y = y_real[:800]
test_features = features[800:]
test_y = y_real[800:]

# Model (aligned with rawlvl0 defaults for fair comparison)
model = GraphKAN(num_inputs=num_inputs, num_hidden=num_hidden, num_outputs=num_outputs,
                 num_intervals=15, spline_order=3, grid_range=[-1, 1], num_layers=1,
                 use_layernorm=False)
optimizer = optim.Adam(model.parameters(), lr=5e-3)


def train(epochs=1000):

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(train_features)
        pred_y = out[:, model.y_index]
        loss_y = F.mse_loss(pred_y, train_y)


        loss_rec = sum(F.mse_loss(out[:, i], train_features[:, i]) for i in range(model.num_inputs))
        loss = loss_y + loss_rec
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                test_out = model(test_features)[:, -1]
                test_mse = F.mse_loss(test_out, test_y).item()
            print(f"Epoch {epoch} | Test MSE: {test_mse:.6f} | Total Loss: {loss.item():.4f}")
        if epoch % 20 == 0 and epoch > 0:
            for name, phi in model.phis.items():
                i = int(name.split('_')[0])
                if i < model.num_inputs:
                    phi.update_grid_from_samples(train_features[:, i:i+1])

# Prune & symbolic
train()
model.prune(threshold=1e-3)
model.auto_symbolic()
model.resolve_bidirectional()

model.eval()
with torch.no_grad():
    test_out = model(test_features)[:, -1]
    test_mse = F.mse_loss(test_out, test_y).item()
print(f"Test MSE: {test_mse:.6f}")

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

print("\nComposed symbolic y:")
formula_str = model.get_full_symbolic_formula()
print(formula_str)

model.plot_with_formula()