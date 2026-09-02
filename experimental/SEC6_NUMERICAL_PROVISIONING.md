# Section VI numerical provisioning cache

**Updated:** 2026-09-03  
**Permanent exchange convention:** `J_over_2_ladder_v1`  
**Historical production base:** `data/evidence_jobs/spin1_production_20260806T074051Z/`  
**Historical sparse-convergence addendum:** `data/evidence_jobs/spin1_production_20260810T082123Z/`  
**Historical Sec. VI provisioning addendum:** `data/evidence_jobs/spin1_sec6_provisioning_20260820T052954Z/`  
**Historical common-window integration addendum:** `data/evidence_jobs/spin1_sec6_integration_20260825T073925Z/`

## Status and migration rule

The August timestamped evidence folders are **immutable historical inputs**. They were
produced through manuscript-facing helpers that used a ladder prefactor of one. The
low-level qlinks Spin-1 exchange operator itself was already conventional, so the old
helpers effectively supplied twice the intended exchange coefficient.

All new qlinks calculations and all manuscript-facing derived evidence use

\[
H_{XY}=J\sum_r(S_r^xS_{r+1}^x+S_r^yS_{r+1}^y)
=\frac{J}{2}\sum_r(S_r^+S_{r+1}^-+S_r^-S_{r+1}^+).
\]

For the completed `h=D=0` Sec. VI evidence the migration is exact:

\[
H_{\rm new}=\frac12H_{\rm legacy},\qquad
E_{\rm new}=\frac12E_{\rm legacy},
\]

with unchanged eigenvectors. Therefore normalized witness expectation values,
trace-distance diagnostics, dark ranks, state counts, and covariance widths are
unchanged when the energy window is also divided by two. Energy-dimensional residuals,
gaps, tolerances, and spectral coverage are divided by two; matched `beta J` values are
doubled.

Do **not** edit or overwrite the historical folders. Use
`spin1_exchange_convention_migrate_evidence.py` or the dedicated Docker runner to
construct a convention-stamped derived layer.

P0 is scientifically and numerically closed. This convention migration is a
normalization/provenance repair, not a request for a new `L=14` production solve.

## Locked Hamiltonian and representative point

Define

\[
K_d(t_d)=\frac12\sum_r\left[
 t_dS_r^+S_{r+d}^-+t_d^*S_r^-S_{r+d}^+
\right],
\]

and

\[
H_\kappa=K_1(J)+K_3(0.1J)+K_2(i\kappa),
\qquad \kappa_\star/J=0.1.
\]

Use even `L>=8`, PBC, `M=-2`, the tower momentum sector, and `h=D=0` for the
representative thermodynamic evidence. `L=6` is only a pre-asymptotic/geometry control
because range three coincides with its reverse on that ring.

The exact tower-compatibility rule remains

\[
t_d^*+(-1)^dt_d=0.
\]

The factor `1/2` multiplies the complete kinetic operator and therefore does not alter
this zero condition.

The representative point continues to break ordinary inversion and the unitary
`C_A` anticommutation while retaining the antiunitary spectral reflection
`Theta=C_A K`. Exact-energy blocks remain basis independent.

The principal sampled positive grid remains

`kappa/J in {0.05, 0.10, 0.15, 0.20}`,

with `kappa=0` used only as a symmetry-enhanced endpoint control.

## Convention checks that must remain mechanical

For a nearest-neighbor exchange with parameter `J`, qlinks must satisfy

\[
\langle 00|H|+-\rangle
=\langle 00|H|-+\rangle=J.
\]

The migration validation job also checks:

- the two `L=5`, `M=-2` shell modes have kinetic energies `0` and `-2J` with zero
  boundary residual;
- the decorated `L=4,n=1` PBC counterexample has normalized residual
  `sqrt(2)|J|`;
- homogeneous and finite-`D` Hamiltonians obey the exact old-to-new factor-one-half
  mapping;
- optional dense `L=8`/`L=10` spot checks reuse the old eigenvectors with eigenvalues
  divided by two;
- the generic `L=8` cage/Jacobian quantity is remeasured rather than inferred from
  the old mixed-coordinate Jacobian.

These are cheap checks. They do not justify or require an `L=14` eigensolve.

## Completed representative-point evidence in current units

At `kappa_star/J=0.1`, the translated joint-dark projector has rank one through
`L=14`. The historical `L=14` calculation returned 8192 shift-invert eigenpairs in a
resolved sector of dimension 35925. In the permanent convention the same returned
vectors cover approximately

\[
|E|\lesssim 1.04192,
\]

rather than the historical displayed `2.08384`.

The permanent primary window is

\[
W_L:\qquad \Delta E=\frac{J}{2}L^{1/4},
\]

with protocol name `quarter_power_c0p5`. For `J=1` and `L=14`,

\[
\Delta E_{14}=0.966755<1.04192.
\]

It contains the same 7615 raw and 7614 retained states as the historical
`L^(1/4)` prefactor-one window. The defining raw witness values are unchanged:

\[
(\tau_A,\tau_Z,\tau_Y)_{L=14}^{\rm raw}
=(0.113204,0.220109,0.328044),
\]

with cleaned companion `(0.113212,0.220243,0.328219)`.

Across `L=8,10,12,14`, the primary raw state counts remain
`28,157,1083,7615`, so the previously recorded `log(N_win)/L` sequence is unchanged.
It remains a positive-entropy consistency trend rather than a proof of a limiting
entropy density.

The raw--raw and clean--clean matching sequences are likewise invariant under the
uniform energy rescaling:

\[
\Delta_L^{\rm rr}
=0.031654,\ 0.021878,\ 0.013029,\ 0.012411,
\]

\[
\Delta_L^{\rm cc}
=0.026664,\ 0.020872,\ 0.012796,\ 0.012246
\quad (L=8,10,12,14).
\]

The permanent fixed control is

\[
\Delta E=J/2,
\]

with protocol name `fixed_width_0p5`. At `J=1` it contains the same 4011 raw `L=14`
states as the historical `Delta E=1` control and gives the same normalized
`Delta_14^cc=0.012882`.

The historical 10000-eigenpair convergence calculation must not be repeated. In current
units its covered half-width is

\[
1.27698
\]

instead of `2.55396`. The preserved state counts for the current primary, fixed
`J/2`, and current `Delta E=0.375` controls are respectively `7615`, `4011`, and
`3063`. The previously established cross-budget changes of at most `2.6e-11` in the
normalized cleaned observables remain valid.

The exact fixed-`M`, `beta=0` limits remain `(1/9,2/9,1/3)`.

## Two-site concentration

The complete magnetization-preserving two-site Hermitian algebra has dimension 19.
The convention migration does not change the normalized covariance widths when the
window is mapped to the same eigenstate set.

For the permanent primary `quarter_power_c0p5` window the raw widths for
`L=8,10,12,14` remain

\[
0.1685908,\quad 0.0763339,\quad 0.0469195,\quad 0.0174573.
\]

For the permanent `fixed_width_0p5` control they remain

\[
0.1760308,\quad 0.0927004,\quad 0.0616927,\quad 0.0237316.
\]

The corresponding raw state counts remain `28,157,1083,7615` and
`24,101,609,4011`. No concentration exponent is fitted or required.

## Cache and provenance discipline

1. Historical August evidence is read-only.
2. A current product must explicitly declare
   `spin1_xy_exchange_convention = J_over_2_ladder_v1`.
3. Missing convention metadata means historical legacy only inside the explicit
   migration/rescaling path; active numerical jobs must not silently interpret it as
   current.
4. Legacy spectral arrays may be reused only by mapping eigenvalues and
   energy-dimensional metadata by `1/2`, keeping eigenvectors unchanged, and validating
   the mapped eigenpairs against the current Hamiltonian.
5. Current spectral checkpoints are schema-v2 and convention-stamped.
6. No implicit eigensolver fallback is allowed when an old/incompatible checkpoint is
   found.
7. Never repeat the historical `L=14` 10000-eigenpair convergence calculation merely to
   rebuild tables or figures.
8. Figure renderers consume already-mapped energy-density columns and must never apply a
   second factor of `1/2`.

## One-time migration runner

After the migration PR is merged, use one explicit timestamped run ID:

```bash
export QLINKS_EVIDENCE_RUN_ID=spin1_exchange_convention_migration_20260903T000000Z
export QLINKS_NUM_THREADS=16

scripts/docker/docker_run_spin1_exchange_convention_migration.sh --stage status
scripts/docker/docker_run_spin1_exchange_convention_migration.sh --stage migrate-p0
scripts/docker/docker_run_spin1_exchange_convention_migration.sh --stage migrate-p1
scripts/docker/docker_run_spin1_exchange_convention_migration.sh --stage validate
scripts/docker/docker_run_spin1_exchange_convention_migration.sh --stage jacobian-l8
scripts/docker/docker_run_spin1_exchange_convention_migration.sh --stage render-p0
```

Replace the example timestamp with the actual run timestamp. The default cheap dense
validation is `L=8`. An optional `L=10` spot check may be requested by setting

```bash
QLINKS_SPIN1_CONVENTION_DENSE_SIZES=8,10
```

for the `validate` stage. The migration runner contains no `L=14` eigensolve route.

## Claim boundary

The migration does not strengthen or weaken the approved Sec. VI scientific claim. It
puts the draft and qlinks on the conventional exchange normalization while preserving
the already-established finite-size evidence exactly where the mapping is uniform.
The manuscript claim remains strong finite-size evidence for deformation-stable
interference-caged many-body scars; a literal thermodynamic open-interval theorem is
not inferred from this normalization repair.
