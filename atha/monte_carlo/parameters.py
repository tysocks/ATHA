# atha/monte_carlo/parameters.py
from __future__ import annotations
from enum import Enum
from typing import Optional
import numpy as np
from scipy.stats import norm


class ParameterType(Enum):
    NORMAL = "normal"
    UNIFORM = "uniform"
    LOGNORMAL = "lognormal"


class UncertainParameter:
    """
    Uncertain input for Monte Carlo sampling.

    Supported construction styles::

        UncertainParameter("Pc", 20.6e6, ParameterType.NORMAL, sigma=0.4e6)
        UncertainParameter("Pc", 20.6e6, ParameterType.NORMAL, sigma_pct=2.0)
        UncertainParameter("T", ParameterType.NORMAL, mean=91.0, std=0.5)
        UncertainParameter("x", nominal=2.0, dist_type=ParameterType.NORMAL, sigma=0.1)
    """

    def __init__(self, name: str, *args, **kwargs) -> None:
        self.name = name
        self.component_attr: Optional[str] = kwargs.pop("component_attr", None)

        sigma = kwargs.pop("sigma", None)
        std = kwargs.pop("std", None)
        if std is not None:
            sigma = std

        nominal = kwargs.pop("nominal", None)
        mean = kwargs.pop("mean", None)
        if mean is not None:
            nominal = mean

        dist_type: Optional[ParameterType] = kwargs.pop("dist_type", None)
        self.sigma_pct: Optional[float] = kwargs.pop("sigma_pct", None)
        self.sigma_log: Optional[float] = kwargs.pop("sigma_log", None)
        self.lower: Optional[float] = kwargs.pop("lower", None)
        self.upper: Optional[float] = kwargs.pop("upper", None)

        if len(args) == 2 and isinstance(args[1], ParameterType):
            nominal = float(args[0])
            dist_type = args[1]
        elif len(args) == 1 and isinstance(args[0], ParameterType):
            dist_type = args[0]
        elif len(args) == 1 and not isinstance(args[0], ParameterType):
            raise TypeError(
                "Single positional arg after name must be ParameterType "
                "(use mean=/nominal= for value)."
            )
        elif len(args) > 2:
            raise TypeError(f"Too many positional arguments: {args}")

        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")

        if dist_type is None:
            raise TypeError(f"UncertainParameter '{name}' requires dist_type")
        if nominal is None:
            raise TypeError(f"UncertainParameter '{name}' requires nominal or mean")

        self.nominal = float(nominal)
        self.dist_type: ParameterType = dist_type
        self.sigma: Optional[float] = sigma

        self._validate()

    def _validate(self) -> None:
        if self.dist_type == ParameterType.NORMAL:
            if self.sigma is None and self.sigma_pct is not None:
                self.sigma = self.nominal * self.sigma_pct / 100.0
            if self.sigma is None:
                raise ValueError(f"NORMAL parameter '{self.name}' requires sigma, std, or sigma_pct")
        elif self.dist_type == ParameterType.UNIFORM:
            if self.lower is None or self.upper is None:
                raise ValueError(f"UNIFORM parameter '{self.name}' requires lower and upper")
        elif self.dist_type == ParameterType.LOGNORMAL:
            if self.sigma_log is None:
                raise ValueError(f"LOGNORMAL parameter '{self.name}' requires sigma_log")

    def sample(self, rng) -> float:
        if self.dist_type == ParameterType.NORMAL:
            return float(rng.normal(self.nominal, self.sigma))
        elif self.dist_type == ParameterType.UNIFORM:
            return float(rng.uniform(self.lower, self.upper))
        elif self.dist_type == ParameterType.LOGNORMAL:
            return float(np.exp(rng.normal(np.log(self.nominal), self.sigma_log)))
        raise ValueError(f"Unknown ParameterType: {self.dist_type}")

    def transform_unit(self, u: float) -> float:
        if self.dist_type == ParameterType.NORMAL:
            return norm.ppf(u) * self.sigma + self.nominal
        elif self.dist_type == ParameterType.UNIFORM:
            return u * (self.upper - self.lower) + self.lower
        elif self.dist_type == ParameterType.LOGNORMAL:
            return float(np.exp(norm.ppf(u) * self.sigma_log + np.log(self.nominal)))
        raise ValueError(f"Unknown ParameterType: {self.dist_type}")
