"""HDF5 I/O for TestProfileResult objects."""
from __future__ import annotations
import numpy as np
import h5py
from atha.profiles.result import PhaseResult, TestProfileResult


def save_profile_result(result: TestProfileResult, filename: str) -> None:
    """Save a TestProfileResult to an HDF5 file.

    Parameters
    ----------
    result : TestProfileResult
        The profile result to save.
    filename : str
        Path to the HDF5 file to write.
    """
    with h5py.File(filename, "w") as f:
        f.attrs["profile_name"] = result.profile_name
        f.attrs["state_names"] = result.state_names
        f.attrs["success"] = result.success
        if result.abort_reason is not None:
            f.attrs["abort_reason"] = result.abort_reason
        if result.abort_time is not None:
            f.attrs["abort_time"] = result.abort_time

        for i, phase in enumerate(result.phases):
            grp = f.create_group(f"phase_{i:03d}")
            grp.attrs["name"] = phase.name
            grp.attrs["abort_triggered"] = phase.abort_triggered
            grp.attrs["state_names"] = phase.state_names
            grp.create_dataset("t", data=phase.t)
            grp.create_dataset("X", data=phase.X)
            grp.create_dataset("X_final", data=phase.X_final)


def load_profile_result(filename: str) -> TestProfileResult:
    """Load a TestProfileResult from an HDF5 file.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file to read.

    Returns
    -------
    TestProfileResult
        The loaded profile result.
    """
    with h5py.File(filename, "r") as f:
        profile_name = str(f.attrs["profile_name"])
        state_names = list(f.attrs["state_names"])
        abort_reason = str(f.attrs["abort_reason"]) if "abort_reason" in f.attrs else None
        abort_time = float(f.attrs["abort_time"]) if "abort_time" in f.attrs else None

        phases = []
        for key in sorted(f.keys()):
            grp = f[key]
            pr = PhaseResult(
                name=str(grp.attrs["name"]),
                t=grp["t"][:],
                X=grp["X"][:],
                state_names=list(grp.attrs["state_names"]),
                X_final=grp["X_final"][:],
                abort_triggered=bool(grp.attrs["abort_triggered"]),
            )
            phases.append(pr)

    return TestProfileResult(
        profile_name=profile_name,
        phases=phases,
        state_names=state_names,
        abort_reason=abort_reason,
        abort_time=abort_time,
    )
