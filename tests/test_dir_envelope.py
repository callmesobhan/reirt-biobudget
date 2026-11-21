"""Tests for 3D diffeomorphic SVF integration, AAPM TG-132 QA, and DIR uncertainty envelopes."""

from __future__ import annotations

import numpy as np

from reirt_biobudget.dir_envelope import (
    audit_dvf_quality,
    compute_dir_dose_envelope,
    compute_jacobian_determinant,
    integrate_svf_to_dvf,
    warp_array,
)
from reirt_biobudget.ir import Volume3D


def test_rigid_shift_analytical_equivalence_and_unit_jacobian() -> None:
    shape = (24, 24, 24)
    spacing = (2.0, 2.0, 2.0)
    zz, yy, xx = np.meshgrid(
        np.arange(shape[0]) * spacing[0],
        np.arange(shape[1]) * spacing[1],
        np.arange(shape[2]) * spacing[2],
        indexing="ij",
    )
    dose = 60.0 * np.exp(-(((zz - 24.0) ** 2 + (yy - 24.0) ** 2 + (xx - 24.0) ** 2) / (2 * 6.0**2)))

    # Uniform pull-back shift u = (+2, -4, +2) mm -> warped peak moves to (22, 28, 22) mm
    dvf = np.zeros((3, *shape), dtype=np.float64)
    dvf[0] = 2.0
    dvf[1] = -4.0
    dvf[2] = 2.0

    jac = compute_jacobian_determinant(dvf, spacing)
    assert np.allclose(jac, 1.0, atol=1e-12)

    warped = warp_array(dose, dvf, spacing, order=1, cval=0.0, clamp_nonnegative=True)
    # Positivity & non-overshooting bounds
    assert float(np.min(warped)) >= 0.0
    assert float(np.max(warped)) <= float(np.max(dose)) + 1e-12

    peak_idx = np.unravel_index(int(np.argmax(warped)), shape)
    peak_mm = tuple(peak_idx[i] * spacing[i] for i in range(3))
    assert np.allclose(peak_mm, (22.0, 28.0, 22.0), atol=1e-6)


def test_svf_diffeomorphism_positive_jacobian_and_low_ice() -> None:
    shape = (24, 24, 24)
    spacing = (2.5, 2.5, 2.5)
    zz, yy, xx = np.meshgrid(
        np.linspace(-1, 1, shape[0]),
        np.linspace(-1, 1, shape[1]),
        np.linspace(-1, 1, shape[2]),
        indexing="ij",
    )
    env = np.exp(-(zz**2 + yy**2 + xx**2) * 2.5)
    vel = np.zeros((3, *shape), dtype=np.float64)
    vel[0] = 3.0 * np.sin(zz) * env
    vel[1] = 2.5 * yy * env
    vel[2] = -2.5 * xx * env

    dvf_fwd = integrate_svf_to_dvf(vel, spacing, steps=6)
    dvf_inv = integrate_svf_to_dvf(-vel, spacing, steps=6)

    quality, _jac, _ice = audit_dvf_quality(dvf_fwd, spacing, dvf_inv_mm=dvf_inv)
    assert quality.folded_voxel_count == 0
    assert quality.jacobian_min > 0.5
    assert quality.ice_p95_mm < 0.2
    assert quality.tg132_pass is True


def test_dir_envelope_monotonicity_and_tre_scaling() -> None:
    shape = (20, 20, 20)
    spacing = (2.5, 2.5, 2.5)
    zz, yy, xx = np.meshgrid(
        np.arange(shape[0]) * spacing[0],
        np.arange(shape[1]) * spacing[1],
        np.arange(shape[2]) * spacing[2],
        indexing="ij",
    )
    dose_arr = 70.0 * np.exp(-(((zz - 25.0) ** 2 + (yy - 25.0) ** 2 + (xx - 25.0) ** 2) / (2 * 8.0**2)))
    vol = Volume3D(
        data=dose_arr,
        spacing_mm=spacing,
        frame_of_reference_uid="1.2.826.0.1.3680043.8.498.TEST.ENV",
    )
    zero_dvf = np.zeros((3, *shape), dtype=np.float64)

    env_small = compute_dir_dose_envelope(vol, zero_dvf, tre_sigma_mm=1.0, n_ensemble=12, seed=7)
    env_large = compute_dir_dose_envelope(vol, zero_dvf, tre_sigma_mm=3.5, n_ensemble=12, seed=7)

    # Exact voxel-wise quantile ordering
    assert np.all(env_large.dose_p05.data <= env_large.dose_p50.data + 1e-6)
    assert np.all(env_large.dose_p50.data <= env_large.dose_p95.data + 1e-6)
    assert np.all(env_large.dose_p95.data <= env_large.dose_worst.data + 1e-6)

    # Larger TRE must increase mean dose uncertainty spread
    assert float(np.mean(env_large.dose_spread_90.data)) > float(np.mean(env_small.dose_spread_90.data))
