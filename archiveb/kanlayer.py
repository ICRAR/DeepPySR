import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import curve_fit
import sympy as sp

def _safe_log_torch(x):
    return torch.log(torch.clamp(x, min=1e-6))

def _safe_inv_torch(x):
    # clamp away from zero while preserving sign
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