import torch
import numpy as np
import pandas as pd
import sympy as sp
import os
import csv
from kan import KAN
from pysr import PySRRegressor
from .utils import ensure_output_dir, plot_n_layer_graph, plot_circlize

class KANPySRRegressor:
    """
    KANPySRRegressor combines Kolmogorov-Arnold Networks (KAN) with PySR for symbolic distillation.
    
    1. KAN Training: Shallow KAN architecture using B-splines.
    2. Univariate Extraction: Extract learned splines as 1D datasets.
    3. Symbolic Distillation via PySR: Fit extracted splines with exact mathematical expressions.
    """
    def __init__(
        self,
        kan_width=[2, 5, 1],
        kan_grid=5,
        kan_k=3,
        kan_steps=500,
        kan_lamb=0.1,
        kan_lamb_entropy=2.0,
        output_dir="outputs/kan_pysr",
        **pysr_kwargs
    ):
        self.kan_width = kan_width
        self.kan_grid = kan_grid
        self.kan_k = kan_k
        self.kan_steps = kan_steps
        self.kan_lamb = kan_lamb
        self.kan_lamb_entropy = kan_lamb_entropy
        self.output_dir = output_dir
        self.pysr_kwargs = pysr_kwargs
        self.model = None
        self.symbolic_expressions = {}
        self.relationships_ = []

    def get_params(self, deep=True):
        return {
            "kan_width": self.kan_width,
            "kan_grid": self.kan_grid,
            "kan_k": self.kan_k,
            "kan_steps": self.kan_steps,
            "kan_lamb": self.kan_lamb,
            "kan_lamb_entropy": self.kan_lamb_entropy,
            "output_dir": self.output_dir,
            **self.pysr_kwargs
        }

    def set_params(self, **params):
        for key, value in params.items():
            if key in ["kan_width", "kan_grid", "kan_k", "kan_steps", "kan_lamb", "kan_lamb_entropy", "output_dir"]:
                setattr(self, key, value)
            else:
                self.pysr_kwargs[key] = value
        return self

    def fit(self, X, y):
        ensure_output_dir(self.output_dir)
        
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = [str(c) for c in X.columns.tolist()]
            X_data = torch.from_numpy(X.values).float()
        else:
            self.feature_names_in_ = [f"x{i}" for i in range(X.shape[1])]
            X_data = torch.from_numpy(X).float()

        if isinstance(y, pd.DataFrame):
            self.names_out = [str(c) for c in y.columns.tolist()]
            y_data = torch.from_numpy(y.values).float()
        elif isinstance(y, pd.Series):
            self.names_out = [str(y.name) if y.name else "y0"]
            y_data = torch.from_numpy(y.values).float().reshape(-1, 1)
        else:
            y_data = torch.from_numpy(y).float()
            if len(y_data.shape) == 1:
                y_data = y_data.reshape(-1, 1)
            self.names_out = [f"y{i}" for i in range(y_data.shape[1])]
        
        # 1. KAN Training
        self.model = KAN(width=self.kan_width, grid=self.kan_grid, k=self.kan_k)
        # Update self.kan_width because KAN might have normalized it
        self.kan_width = self.model.width
        
        dataset = {
            'train_input': X_data,
            'train_label': y_data,
            'test_input': X_data,
            'test_label': y_data
        }
        
        print("--- Training KAN ---")
        self.model.fit(dataset, opt="LBFGS", steps=self.kan_steps, lamb=self.kan_lamb, lamb_entropy=self.kan_lamb_entropy)
        
        print("--- Pruning KAN ---")
        self.model.prune()
        
        # 2. Univariate Extraction and 3. Symbolic Distillation
        print("--- Extracting Univariate Splines and Distilling with PySR ---")
        self._distill_splines(X_data)
        
        return self

    def _distill_splines(self, X_data):
        self.model(X_data)  # forward pass to populate model.acts
        self.symbolic_expressions = {} # (layer, node_index) -> sympy_expression
        self.relationships_ = []

        for l in range(len(self.kan_width) - 1):
            width_l = self.kan_width[l]
            width_next = self.kan_width[l+1]
            if isinstance(width_l, list): width_l = width_l[0]
            if isinstance(width_next, list): width_next = width_next[0]

            for j in range(width_next):
                node_target_name = f"L{l+1}_N{j}" if l < len(self.kan_width) - 2 else "y"
                
                # Identify involved inputs for this node j in layer l
                # self.model.act_fun[l].mask is [in_dim, out_dim]
                mask = self.model.act_fun[l].mask[:, j].detach().cpu().numpy()
                involved_indices = np.where(mask > 0)[0]
                
                if len(involved_indices) == 0:
                    print(f"Node L{l+1}_N{j} has no involved inputs. Skipping.")
                    self.symbolic_expressions[(l, j)] = sp.Float(0.0)
                    continue

                involved_names = [f"L{l}_N{i}" if l > 0 else f"x{i}" for i in involved_indices]
                
                # Extract inputs to this node
                X_node = self.model.acts[l][:, involved_indices].detach().cpu().numpy()
                
                # Extract target for this node
                with torch.no_grad():
                    postacts = self.model.act_fun[l](self.model.acts[l])[0]
                    y_node = postacts[:, j].detach().cpu().numpy()

                print(f"Distilling node {node_target_name} with {len(involved_indices)} inputs: {involved_names}")

                pysr_model = PySRRegressor(**self.pysr_kwargs)
                pysr_model.fit(X_node, y_node, variable_names=involved_names)

                # Get the best formula
                best_expr = pysr_model.sympy()
                
                # Check if PySR pruned some variables that KAN thought were important
                pysr_involved = [str(s) for s in best_expr.free_symbols]
                final_involved = involved_names

                # But we can make sure 'involved' matches what's in the formula.
                if any(isinstance(best_expr, (sp.Symbol, sp.Expr)) for _ in [0]): # Just to have a block
                    final_involved = [name for name in involved_names if sp.Symbol(name) in best_expr.free_symbols]
                    if not final_involved and not best_expr.is_constant:
                        # Fallback if somehow free_symbols doesn't match involved_names
                        final_involved = involved_names

                self.symbolic_expressions[(l, j)] = best_expr

                self.relationships_.append({
                    "target": node_target_name,
                    "target_symbol": sp.Symbol(node_target_name),
                    "layer": l + 1,
                    "sympy": best_expr,
                    "formula": str(best_expr),
                    "involved": final_involved,
                    "score": 1.0,  # Placeholder
                    "loss": 0.0,   # Placeholder
                    "complexity": 1  # Placeholder
                })

        self.save_relationships()

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X_input = X.values
        else:
            X_input = X
            
        current_values = X_input.T # shape (n_features, n_samples)
        
        for l in range(len(self.kan_width) - 1):
            width_l = self.kan_width[l]
            width_next = self.kan_width[l+1]
            if isinstance(width_l, list): width_l = width_l[0]
            if isinstance(width_next, list): width_next = width_next[0]
            
            next_values = np.zeros((width_next, X_input.shape[0]))
            for j in range(width_next):
                # Identify involved inputs for this node
                expr = self.symbolic_expressions[(l, j)]
                involved_names = [str(s) for s in expr.free_symbols]
                
                # Filter involved_names to only include those from this layer
                # internal names are like L{l}_N{i} or x{i}
                valid_prefix = f"L{l}_N" if l > 0 else "x"
                involved_names = [n for n in involved_names if n.startswith(valid_prefix)]
                
                if not involved_names:
                    if expr.is_constant:
                        next_values[j] = float(expr)
                    else:
                        next_values[j] = 0
                    continue
                
                # Get indices for involved names
                involved_indices = []
                for name in involved_names:
                    try:
                        idx = int(name[len(valid_prefix):])
                        involved_indices.append(idx)
                    except ValueError:
                        pass

                # Wrap the expression for evaluation
                func = sp.lambdify([sp.Symbol(name) for name in involved_names], expr, modules=['numpy'])
                
                # Prepare arguments for the lambdified function
                args = [current_values[idx] for idx in involved_indices]
                next_values[j] = func(*args)
                
            current_values = next_values
            
        return current_values[0] # Assuming single output for now

    def sympy(self):
        """Returns the full symbolic expression by composing distilled splines."""
        
        # We start from input symbols
        current_exprs = [sp.Symbol(name) for name in self.feature_names_in_]
        
        for l in range(len(self.kan_width) - 1):
            width_l = self.kan_width[l]
            width_next = self.kan_width[l+1]
            if isinstance(width_l, list): width_l = width_l[0]
            if isinstance(width_next, list): width_next = width_next[0]
            
            next_exprs = []
            for j in range(width_next):
                expr = self.symbolic_expressions[(l, j)]
                
                # Identify involved inputs for this node from the expression
                involved_names = [str(s) for s in expr.free_symbols]
                valid_prefix = f"L{l}_N" if l > 0 else "x"
                involved_names = [n for n in involved_names if n.startswith(valid_prefix)]
                
                if not involved_names:
                    if expr.is_constant:
                        next_exprs.append(sp.Float(float(expr)))
                    else:
                        next_exprs.append(sp.Float(0.0))
                    continue
                
                # Substitute each involved input symbol with its expression from previous layer
                sub_map = {}
                for name in involved_names:
                    try:
                        idx_in_layer = int(name[len(valid_prefix):])
                        sub_map[sp.Symbol(name)] = current_exprs[idx_in_layer]
                    except ValueError:
                        pass
                
                node_expr = expr.subs(sub_map)
                next_exprs.append(node_expr)
                
            current_exprs = next_exprs
            
        return current_exprs[0] # Assuming single output

    def _get_mapped_relationships(self):
        """Returns a copy of relationships with original variable names."""
        if not hasattr(self, "feature_names_in_"):
            return self.relationships_
            
        mapping = {f"x{i}": name for i, name in enumerate(self.feature_names_in_)}
        mapped_rels = []
        for rel in self.relationships_:
            new_rel = rel.copy()
            # Map target if it's xi
            if new_rel["target"].startswith("x"):
                try:
                    idx = int(new_rel["target"][1:])
                    new_rel["target"] = mapping.get(new_rel["target"], new_rel["target"])
                except ValueError:
                    pass
            
            # Map sympy formula
            # Note: For KAN, involved are x{i} at layer 0, and L{l}_N{i} at other layers.
            # We only want to map x{i} to original feature names.
            sym_mapping = {sp.Symbol(f"x{i}"): sp.Symbol(name) for i, name in enumerate(self.feature_names_in_)}
            new_rel["sympy"] = new_rel["sympy"].subs(sym_mapping)
            new_rel["formula"] = str(new_rel["sympy"])
            
            # Map involved names in the list too
            new_involved = []
            for inv in new_rel["involved"]:
                if inv.startswith("x"):
                    try:
                        idx = int(inv[1:])
                        new_involved.append(mapping.get(inv, inv))
                    except ValueError:
                        new_involved.append(inv)
                else:
                    new_involved.append(inv)
            new_rel["involved"] = new_involved
            
            mapped_rels.append(new_rel)
        return mapped_rels

    def save_relationships(self, filename="relationships.csv"):
        path = os.path.join(self.output_dir, filename)
        mapped_rels = self._get_mapped_relationships()
        columns = ["layer", "target", "formula", "involved", "score", "loss", "complexity"]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for r in mapped_rels:
                row = {col: r.get(col) for col in columns}
                # Format involved as a string if it's a list
                if isinstance(row["involved"], list):
                    row["involved"] = ", ".join(row["involved"])
                writer.writerow(row)

    def plot(self, filename="kan_pysr_hierarchy.png"):
        feature_names = getattr(self, "feature_names_in_", None)
        mapped_rels = self._get_mapped_relationships()
        if not mapped_rels and (not feature_names):
            print("No relationships or variables to plot.")
            return
        path = os.path.join(self.output_dir, filename)
        plot_n_layer_graph(mapped_rels, path, feature_names=feature_names)
        print(f"Plot saved to {path}")

    def plot_circle(self, filename="kan_pysr_circle.png"):
        feature_names = getattr(self, "feature_names_in_", None)
        mapped_rels = self._get_mapped_relationships()
        if not mapped_rels and (not feature_names):
            print("No relationships or variables to plot.")
            return
        path = os.path.join(self.output_dir, filename)
        plot_circlize(mapped_rels, path, feature_names=feature_names)
        print(f"Plot saved to {path}")
