# atha/monte_carlo/results.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from atha.monte_carlo.statistics import MCStatistics


@dataclass
class MonteCarloResult:
    param_names: List[str]
    param_samples: np.ndarray   # shape (N, k)
    Y_samples: np.ndarray       # shape (N,)
    converged: np.ndarray       # shape (N,), bool
    stats: Optional[MCStatistics]
    sobol: Optional[dict] = None

    def print_summary(self) -> None:
        n_total = len(self.Y_samples)
        n_conv = int(np.sum(self.converged))
        print(f"Monte Carlo Results: N={n_total}, converged={n_conv} ({100*n_conv/n_total:.1f}%)")
        if self.stats:
            s = self.stats
            print(f"  Mean:    {s.mean:.4g}")
            print(f"  Std:     {s.std:.4g}")
            print(f"  CV:      {s.cv_pct:.2f}%")
            print(f"  95% CI:  [{s.p5:.4g}, {s.p95:.4g}]")
        if self.sobol:
            print("\nSobol Sensitivity Indices:")
            print(f"  {'Parameter':<20} {'S1':>8} {'ST':>8}")
            for name, s1, st in zip(self.param_names, self.sobol["S1"], self.sobol["ST"]):
                print(f"  {name:<20} {s1:>8.3f} {st:>8.3f}")

    @property
    def n_failed(self) -> int:
        return int(np.sum(~self.converged))

    def plot_histogram(
        self,
        xlabel: str = "Output",
        title: str = "",
        bins: int = 40,
        show: bool = True,
        **hist_kw,
    ):
        import matplotlib.pyplot as plt
        Y_valid = self.Y_samples[self.converged]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(Y_valid, bins=bins, density=True, edgecolor="white", alpha=0.8, **hist_kw)
        if self.stats:
            ax.axvline(self.stats.mean, color="red", linestyle="--",
                       label=f"Mean = {self.stats.mean:.3g}")
            ax.axvline(self.stats.p5,  color="orange", linestyle=":",
                       label=f"5th pct = {self.stats.p5:.3g}")
            ax.axvline(self.stats.p95, color="orange", linestyle=":",
                       label=f"95th pct = {self.stats.p95:.3g}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Probability Density")
        ax.set_title(title or "Monte Carlo Output Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def plot_sobol_indices(self, show: bool = True):
        if self.sobol is None:
            raise RuntimeError("No Sobol indices — run sensitivity analysis first")
        import matplotlib.pyplot as plt
        names = self.param_names
        S1 = self.sobol["S1"]
        ST = self.sobol["ST"]
        S1_conf = self.sobol.get("S1_conf", np.zeros_like(S1))
        ST_conf = self.sobol.get("ST_conf", np.zeros_like(ST))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, max(4, len(names) * 0.5)))
        y = np.arange(len(names))
        ax1.barh(y, S1, xerr=S1_conf, capsize=4, alpha=0.8)
        ax1.set_yticks(y); ax1.set_yticklabels(names)
        ax1.set_xlabel("First-Order Index S_i"); ax1.set_title("Main Effects")
        ax2.barh(y, ST, xerr=ST_conf, capsize=4, alpha=0.8, color="orange")
        ax2.set_yticks(y); ax2.set_yticklabels(names)
        ax2.set_xlabel("Total-Order Index S_Ti"); ax2.set_title("Total Effects")
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def save(self, filename: str) -> None:
        import h5py
        with h5py.File(filename, "w") as f:
            f.create_dataset("param_samples", data=self.param_samples)
            f.create_dataset("Y_samples", data=self.Y_samples)
            f.create_dataset("converged", data=self.converged.astype(np.uint8))
            f.attrs["param_names"] = self.param_names
            if self.stats:
                sg = f.create_group("statistics")
                for attr in ("N_samples", "mean", "std", "cv_pct", "min", "max",
                             "median", "p1", "p5", "p95", "p99", "mean_ci_95"):
                    sg.attrs[attr] = getattr(self.stats, attr)
            if self.sobol:
                sg = f.create_group("sobol")
                for key in ("S1", "ST", "S1_conf", "ST_conf"):
                    if key in self.sobol:
                        sg.create_dataset(key, data=self.sobol[key])

    @classmethod
    def load(cls, filename: str) -> "MonteCarloResult":
        import h5py
        from atha.monte_carlo.statistics import MCStatistics
        with h5py.File(filename, "r") as f:
            param_names = list(f.attrs["param_names"])
            param_samples = f["param_samples"][:]
            Y_samples = f["Y_samples"][:]
            converged = f["converged"][:].astype(bool)
            stats = None
            if "statistics" in f:
                sg = f["statistics"]
                stats = MCStatistics(**{k: sg.attrs[k] for k in sg.attrs})
            sobol = None
            if "sobol" in f:
                sobol = {k: f["sobol"][k][:] for k in f["sobol"]}
        return cls(param_names=param_names, param_samples=param_samples,
                   Y_samples=Y_samples, converged=converged, stats=stats, sobol=sobol)
