"""Solver package.

``AlgebraicNetworkProblem`` aliases remain supported. The older
``SteadyStateSolver`` / ``TransientSolver`` classes under this package operate
on ``EngineLayout`` and are legacy relative to the generic-port DAE runner.
"""

from atha.solver.algebraic import AlgebraicNetworkProblem, AlgebraicResidual, AlgebraicSolution, AlgebraicVariable

__all__ = [
    "AlgebraicNetworkProblem",
    "AlgebraicResidual",
    "AlgebraicSolution",
    "AlgebraicVariable",
]
