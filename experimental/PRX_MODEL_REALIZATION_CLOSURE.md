# Spin-1 model-realization thermodynamic closure

**Scope:** analytic closure of the spin-1 XY realization in the even-\(L\), \(M=-2\),
tower-momentum sequence. This result deliberately uses a wider raw energy window than
the existing finite-size \(L^{1/4}\) diagnostic. It does not retroactively prove
thermodynamic statements for that narrower numerical window.

## 1. Locked model and window

Use
\[
H_L(\kappa)=K_1(J)+K_3(0.1J)+K_2(i\kappa),
\]
with
\[
K_d(t)=\frac12\sum_r
\left(t S_r^+S_{r+d}^-+t^*S_r^-S_{r+d}^+\right).
\]

Take even \(L\), periodic boundary conditions, fixed \(M=-2\), and the tower momentum
\[
q_L=
\begin{cases}
\pi,&L=0\pmod 4,\\
0,&L=2\pmod 4.
\end{cases}
\]

The compatible open family is
\[
U=\{\,\kappa:0.05<\kappa/J<0.20\,\},
\]
and all uniform estimates below are taken on its compact closure
\(K=[0.05J,0.20J]\). The representative point \(\kappa_\star/J=0.1\) lies in \(U\);
the symmetry-enhanced \(\kappa=0\) point is not needed.

For the thermodynamic definition use the raw half-width
\[
\Delta E_L=c|J|L^{3/4},\qquad c>0\ \text{fixed}.
\]
Thus the full raw window has width \(2c|J|L^{3/4}\). This is admissible because
\(\Delta E_L/L\to0\). It is the handoff choice \(\eta=1/4\) in
\(\Delta E_L\sim L^{1/2+\eta}\).

## 2. Exact spectral center in the selected resolved sector

Define the sublattice grading
\[
C_A=(-1)^{\sum_{r\in A}(S_r^z+1)}.
\]
In the fixed-\(M\) product basis, one-site translation exchanges the two sublattices.
The parity change is
\[
(-1)^{\sum_B(m_r+1)-\sum_A(m_r+1)}
=(-1)^{L+M}=+1
\]
for even \(L\) and \(M=-2\). Hence \(C_A\) commutes with one-site translation in this
fixed-\(M\) space.

Complex conjugation \(\mathcal K\) maps momentum \(q\) to \(-q\). The selected momenta
are \(0\) or \(\pi\), so they are self-conjugate. Therefore
\[
\Theta=C_A\mathcal K
\]
preserves the selected \((M,q_L)\) sector.

For an odd-range real exchange, \(\mathcal K\) leaves the coefficient unchanged and
\(C_A\) contributes a minus sign. For the even-range purely imaginary exchange,
\(\mathcal K\) sends \(i\kappa\) to \(-i\kappa\), while \(C_A\) contributes no
exchange-parity sign. Consequently
\[
\Theta H_L(\kappa)\Theta^{-1}=-H_L(\kappa)
\]
for every real \(\kappa\) in the compatible family. The resolved spectrum is exactly
symmetric about zero, so
\[
\operatorname{Tr}\!\left[\rho_{\beta=0}^{(M,q_L)}H_L(\kappa)\right]=0.
\]

The staggered-bimagnon compatibility rule
\(t_d^*+(-1)^d t_d=0\) also gives the selected tower state exact energy
\(E_{\psi,L}=0\). The cage energy is therefore the exact beta-zero spectral center.

## 3. Linear resolved-sector energy-variance bound

### 3.1 Fixed-\(M\) identity-character contribution

For one unordered range-\(d\) pair term, a spin-1 product configuration has at most two
nonzero final product configurations. In the permanent \(J/2\) ladder convention every
allowed exchange matrix element has magnitude at most \(|t_d|\). For all sufficiently
large even \(L\), distinct range-1, range-2, and range-3 unordered pairs change distinct
site sets, so their one-step final product configurations do not collide.

Therefore the diagonal row norm obeys
\[
(H_L^2)_{\sigma\sigma}
\le
2L\left(|J|^2+|0.1J|^2+|\kappa|^2\right).
\]
Averaging over the fixed-\(M\) sector preserves the bound. Uniformly on \(K\),
\[
\frac{\operatorname{Tr}_{M=-2}H_L(\kappa)^2}
     {\dim\mathcal H_{M=-2}}
\le 2.1\,J^2L,
\]
because \(2(1+0.1^2+0.2^2)=2.1\).

Small periodic sizes in which different geometric ranges alias are irrelevant to this
asymptotic statement and are separately covered by the finite-size numerical evidence.

### 3.2 Translation-character defect lemma

Let \(A_X\) be an operator supported on at most \(s\) sites, with norm bounded
independently of \(L\). For a nonidentity translation \(T^r\), a product configuration
contributing to
\[
\operatorname{Tr}_{M}(T^rA_X)
\]
must agree with its \(r\)-shift at every site outside the finite defect set \(X\).
Translation by \(r\) has \(g=\gcd(L,r)\) cycles. Removing at most \(s\) matching
conditions can increase the number of independently assignable local values by at most
\(s\). Hence
\[
\left|\operatorname{Tr}_{M}(T^rA_X)\right|
\le C_X\,3^{\gcd(L,r)+s},
\]
where \(C_X\) is independent of \(L\). The fixed-\(M\) condition can only reduce the
number of admissible configurations. For \(r\not\equiv0\pmod L\),
\(\gcd(L,r)\le L/2\).

The estimate depends on support cardinality, not geometric diameter. It therefore also
applies to products of two translated fixed-support operators even when the two supports
are far apart.

### 3.3 Applying the lemma to \(H_L^2\)

Expand \(H_L^2\) into products of local pair terms. There are \(O(L^2)\) such products,
each supported on at most four sites and with uniformly bounded coefficient on \(K\).
Thus every nonidentity translation-character contribution is exponentially smaller than
the identity contribution.

The already-proved selected momentum-sector dimension satisfies
\[
D_{L,q_L}
=
\frac{T_{L,-2}}{L}+O(3^{L/2})
=
\exp[(\log3+o(1))L].
\]
After division by \(D_{L,q_L}\), a convenient uniform bound on the character correction is
\[
O\!\left(J^2L^{7/2}3^{-L/2}\right).
\]
(The displayed polynomial power is deliberately nonoptimal; only exponential suppression
is used.)

Consequently
\[
\operatorname{Tr}\!\left[
\rho_{\beta=0}^{(M,q_L)}H_L(\kappa)^2
\right]
\le
(2.1+o(1))J^2L
\]
uniformly for \(\kappa\in K\). Since the mean energy is exactly zero, this is also the
variance. Equivalently, for every fixed \(\epsilon>0\), there is an \(L_0\), independent
of \(\kappa\in K\), such that
\[
\operatorname{Var}_{\beta=0,M,q_L}(H_L)
\le (2.1+\epsilon)J^2L
\qquad (L\ge L_0).
\]

No spectral fit or diagonalization is used.

## 4. Almost-full raw window and entropy density

Let \(P_W\) be the selected-sector spectral projector for
\[
|E|\le\Delta E_L=c|J|L^{3/4},
\]
and let \(f_{\rm out}\) be the fraction of the resolved sector outside that window.
Chebyshev gives, uniformly on \(K\),
\[
f_{\rm out}
\le
\frac{(2.1+\epsilon)J^2L}{c^2J^2L^{3/2}}
=
\frac{2.1+\epsilon}{c^2}L^{-1/2}
\longrightarrow0.
\]

Therefore
\[
\frac{N_{\rm win,L}}{D_{L,q_L}}\to1.
\]
Since the resolved sector has entropy density \(\log3\),
\[
\lim_{L\to\infty}\frac1L\log N_{\rm win,L}=\log3.
\]

This closes the positive raw-window entropy gate for the new \(L^{3/4}\)
thermodynamic window. It does not make an asymptotic claim about the existing
\((J/2)L^{1/4}\) numerical window.

## 5. Vanishing declared caged fraction

For this model-level realization the declared exceptional family is the exact staggered
bimagnon tower. At fixed \(M=-2\) there is exactly one tower member. Across all
magnetizations the full tower has only \(L+1\) members, so either count is subexponential.

In the defining resolved \(M=-2\) window,
\[
N_{\rm cage,L}=1,
\qquad
\frac{N_{\rm cage,L}}{N_{\rm win,L}}\to0
\]
exponentially. This does not claim a census of every accidental finite-size compact
eigenstate; it specifies the declared exact tower that enters the realization.

## 6. Positive thermodynamic local witness

Use
\[
Y_r=(S_r^z)^2-I.
\]
Every tower configuration has only \(S^z=\pm1\), so \(Y_r\) annihilates every tower
state. Its normalized positive witness is
\[
Q_r^Y=Y_r^\dagger Y_r,
\]
the projector onto local \(S^z=0\), and \(0\le Q_r^Y\le1\).

At fixed \(M=-2\),
\[
\operatorname{Tr}\!\left[\rho_{\beta=0}^{M}Q_r^Y\right]
=
\frac{T_{L-1,-2}}{T_{L,-2}}
\longrightarrow\frac13.
\]
For any fixed-support bounded local observable \(O_R\), the character-defect lemma gives
\[
\operatorname{Tr}\!\left[\rho_{\beta=0}^{(M,q_L)}O_R\right]
-
\operatorname{Tr}\!\left[\rho_{\beta=0}^{M}O_R\right]
=
O\!\left(\operatorname{poly}(L)3^{-L/2}\right).
\]
Hence the selected resolved beta-zero expectation of \(Q_r^Y\) also tends to \(1/3\).

If a fraction \(f\) of a normalized ensemble is removed and \(0\le Q\le1\), then
\[
\langle Q\rangle_W
\ge
\frac{\langle Q\rangle_{\rm full}-f}{1-f}.
\]
Taking \(f=f_{\rm out}\) gives
\[
\lim_{L\to\infty}
\operatorname{Tr}\!\left[\rho_{\rm mc,W}^{(M,q_L)}Q_r^Y\right]
=
\frac13.
\]
Thus, for example, the uniform lower bound \(1/6\) holds for all sufficiently large
\(L\). The beta-zero resolved reference is Hamiltonian-independent, and the outside
fraction is uniform on \(K\), so the lower bound is uniform on \(U\).

## 7. Arbitrary fixed-region background concentration

This is the all-bounded-region gate in Definition III.2.

### 7.1 Exact momentum compression

For a fixed bounded region \(R\) and bounded Hermitian \(O_R\), define
\[
\overline O_L=\frac1L\sum_x T^xO_RT^{-x}.
\]
Let \(P_q\) be the selected momentum projector. Since
\(P_qT^x=e^{iqx}P_q\) and \(T^{-x}P_q=e^{-iqx}P_q\),
\[
P_qO_RP_q=P_q\overline O_LP_q.
\]
Every energy projector \(P_{E,q}\) is a subprojector of \(P_q\), hence
\[
P_{E,q}O_RP_{E,q}
=
P_{E,q}\overline O_LP_{E,q}.
\]
This is exact and removes any dependence on basis choice inside degenerate energy blocks.

If \(O_R\) changes the conserved total magnetization, only its magnetization-preserving
compression contributes inside the fixed-\(M\) sector. It therefore suffices to prove the
bound for the charge-preserving local compression.

### 7.2 Fixed-\(M\) covariance bound

For a fixed product pattern \(a=(a_1,\ldots,a_s)\), with local magnetization \(m(a)\),
the exact fixed-\(M\) marginal is
\[
\Pr_M(a)
=
\frac{T_{L-s,M-m(a)}}{T_{L,M}}.
\]
For fixed \(M=-2\) and fixed \(s\), the same local lattice central-limit estimate used
in the resolved-sector entropy proof gives, uniformly over the finite set of local
patterns,
\[
\Pr_M(a)=3^{-s}[1+O(L^{-1})].
\]
For two disjoint fixed-size regions, the joint marginal has the same form with their
combined support. Therefore bounded diagonal local observables on separated translates
have covariance \(O(L^{-1})\).

For a general charge-preserving local operator, the trace only samples diagonal matrix
elements. On disjoint supports,
\[
\langle\sigma|O_0O_d|\sigma\rangle
=
\langle\sigma|O_0|\sigma\rangle
\langle\sigma|O_d|\sigma\rangle:
\]
if either local action changes its local product configuration, disjointness prevents the
other action from undoing that change. Thus the same \(O(L^{-1})\) covariance estimate
holds for arbitrary bounded charge-preserving \(O_R\) at nonoverlapping separations.
Only \(O(1)\) translation separations make the two supports overlap, and those
covariances are merely \(O(1)\).

Translation invariance then gives
\[
\operatorname{Var}_{M}(\overline O_L)
=
\frac1L\sum_d\operatorname{Cov}_{M}(O_0,O_d)
=
O(L^{-1}).
\]

### 7.3 Momentum resolution

Each product \(O_0O_d\) is supported on at most twice the fixed local support cardinality,
independently of \(d\). Applying the character-defect lemma term by term shows that
passing from fixed \(M\) to the selected momentum sector changes the mean and second
moment of \(\overline O_L\) only by an exponentially small amount. Hence
\[
\operatorname{Var}_{M,q_L}(\overline O_L)
\le
\frac{C_{R,O}}{L}
+
O\!\left(\operatorname{poly}(L)3^{-L/2}\right)
\]
for every fixed \(R\) and bounded \(O_R\).

### 7.4 Conditioning on the wide raw window

Let
\[
\mu_q=
\operatorname{Tr}\!\left[
\rho_{\beta=0}^{(M,q_L)}\overline O_L
\right].
\]
The positive operator
\[
B=(\overline O_L-\mu_q)^2
\]
satisfies
\[
\operatorname{Tr}(\rho_W B)
\le
\frac{
\operatorname{Tr}(\rho_{\beta=0}^{(M,q_L)}B)
}{1-f_{\rm out}}.
\]
Since \(f_{\rm out}\to0\),
\[
\operatorname{Var}_{W}(\overline O_L)=O(L^{-1}).
\]
The variance about the window's own mean is no larger than the second moment about
\(\mu_q\).

For every energy block in the window, momentum compression gives
\[
P_{E,q}(O_R-\mu_W)P_{E,q}
=
P_{E,q}(\overline O_L-\mu_W)P_{E,q}.
\]
For Hermitian \(A=\overline O_L-\mu_W\), compression obeys
\[
\operatorname{Tr}\!\left[(P_{E,q}AP_{E,q})^2\right]
\le
\operatorname{Tr}\!\left[P_{E,q}A^2P_{E,q}\right].
\]
Summing over energy blocks therefore bounds the basis-independent block Frobenius second
moment by the microcanonical second moment of \(\overline O_L\).

Markov's inequality then gives, for every fixed \(\varepsilon>0\),
\[
f_{L,\varepsilon}[O_R]
\le
\frac{C_{R,O}+o(1)}{\varepsilon^2L}
\longrightarrow0.
\]

The Hermitian operator space on every fixed support size \(r\) is finite dimensional.
Proving the estimate on a complete local basis extends it to every bounded Hermitian
\(O_R\) by norm equivalence, with \(r\) arbitrary but fixed before \(L\to\infty\).

This closes the arbitrary-fixed-bounded-region background-concentration gate.

## 8. Uniform compatible deformation family

The exact tower continuation and bounded local caging operators were already established
for the complex-Hermitian compatible exchange family. The new thermodynamic ingredients
are uniform on \(K\):

1. the energy-variance coefficient is at most \(2.1J^2L\), before an exponentially small
   momentum-character correction;
2. the wide-window outside fraction is uniformly \(O(L^{-1/2})\);
3. the beta-zero fixed-\((M,q)\) state and its local trace/covariance bounds are
   Hamiltonian-independent;
4. the \(Y\) witness is \(\kappa\)-independent and has limiting activity \(1/3\);
5. the background-concentration constants depend on the fixed local operator, not on
   \(\kappa\), except through the uniformly controlled window conditioning.

Consequently the literal thermodynamic realization holds uniformly on the
size-independent open interval \(U=(0.05J,0.20J)\). The symmetry-enhanced \(\kappa=0\)
point is intentionally outside this open-family statement.

## 9. Closure verdict

Under the new admissible wide raw window, the four requested verdicts are

- formal_framework_closed = true;
- model_instantiation_closed = true;
- literal_icqmbs_realization_closed = true, realized by the spin-1 tower sequence with
  \(\Delta E_L=c|J|L^{3/4}\);
- deformation_stable_icqmbs_closed = true on \(0.05<\kappa/J<0.20\).

The square QDM is not a premise of this theorem. Its exact constrained-Hilbert-space
constructions remain a complementary model instantiation, while its fixed-width
thermodynamic ICQMBS classification remains open.

## 10. Claim boundary retained for the old numerical protocol

The existing finite-size window
\[
\Delta E_L=(J/2)L^{1/4}
\]
remains a stricter numerical diagnostic. No local-limit theorem on that narrow energy
scale is established here. The manuscript must distinguish the analytic \(L^{3/4}\)
proof window from the narrower numerical \(L^{1/4}\) data rather than identifying them.
