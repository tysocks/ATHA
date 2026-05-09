from __future__ import annotations

from atha.network import NetworkProblem, NetworkResidual, NetworkSolution, NetworkVariable

AlgebraicVariable = NetworkVariable
AlgebraicResidual = NetworkResidual
AlgebraicSolution = NetworkSolution
AlgebraicNetworkProblem = NetworkProblem

__all__ = [
    "AlgebraicNetworkProblem",
    "AlgebraicResidual",
    "AlgebraicSolution",
    "AlgebraicVariable",
]
