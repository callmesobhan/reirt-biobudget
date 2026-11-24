# `reirt-biobudget` — Formal Mathematical & Engineering Specification

> **Research Software Notice:** `reirt-biobudget` is academic research software for verifiable
> medical physics modeling and is **not a medical device**.

---

## 1. System Architecture

`reirt-biobudget` separates **numerical physical ground truth** from **agentic clinical reasoning**:

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    FRONTIER AGENT HARNESS (MCP / CLI)                   │
│  • TG-263 Ontology Alignment & Heterogeneous Course Ingestion           │
│  • Closed-Loop Adversarial Assumption Sweeping & Tipping-Point Search   │
│  • Multi-Schedule / Multi-Modality Co-Design (6 Regimes)                │
│  • Independent Critic-Verifier Gate & ESTRO/EORTC Dossier Emitter       │
└───────────────────────────────▲─────────────────────────────────────────┘
                                │ Typed JSON-RPC 2.0 / Pydantic IR
┌───────────────────────────────▼─────────────────────────────────────────┐
│                    PROPRIETARY BIOPHYSICAL CORE ENGINE                  │
│  1. Multi-Regime Forward & Exact Analytical Inverse Radiobiology Kernel │
│  2. 3D SVF Diffeomorphic DIR-Uncertainty Envelope + AAPM TG-132 Audit   │
│  3. 3D Constraint-Safe Remaining Physical Dose Budget Optimizer         │
│  4. SHA-256 Provenance Ledger & Hard Physical Abstention Gate           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Multi-Regime Biophysical Forward & Exact Inverse Kernels

Let $D \ge 0$ be total physical dose (Gy) delivered in $n \ge 1$ fractions ($d = D/n$), and $\alpha/\beta > 0$ be the tissue-specific fractionation sensitivity (Gy).

### 2.1 Standard Linear-Quadratic (LQ) — Fowler (1989)
$$\mathrm{BED}_{\mathrm{LQ}}(D, n) = D\left(1 + \frac{d}{\alpha/\beta}\right), \qquad \mathrm{EQD2} = \frac{\mathrm{BED}}{1 + \frac{2}{\alpha/\beta}}$$

**Exact Inverse** (Citardauq form, numerically stable as $\mathrm{BED} \to 0$):
$$D = \frac{2\,\mathrm{BED}}{1 + \sqrt{1 + \frac{4\,\mathrm{BED}}{n(\alpha/\beta)}}}$$

### 2.2 $C^1$-Continuous Linear-Quadratic-Linear (LQL) — Astrahan (2008)
At high doses per fraction ($d > d_T$, e.g., SBRT/SRS), pure LQ overpredicts cell kill. LQL transitions with exact $C^0$ and $C^1$ continuity at $d = d_T$ to constant log-survival slope $\gamma/\alpha = 1 + 2 d_T / (\alpha/\beta)$:

$$\mathrm{BED}_{\mathrm{LQL}}(D, n) = \begin{cases} n d \left(1 + \dfrac{d}{\alpha/\beta}\right), & d \le d_T \\[8pt] n \left[ d_T \left(1 + \dfrac{d_T}{\alpha/\beta}\right) + (d - d_T)\left(1 + \dfrac{2 d_T}{\alpha/\beta}\right) \right], & d > d_T \end{cases}$$

**Exact Inverse:** Let $B_T = n d_T (1 + d_T/(\alpha/\beta))$. For $\mathrm{BED} \le B_T$ invert the LQ branch; for $\mathrm{BED} > B_T$:
$$D = n \left[ d_T + \frac{\mathrm{BED}/n - d_T(1 + d_T/(\alpha/\beta))}{1 + 2 d_T/(\alpha/\beta)} \right]$$

### 2.3 Lea–Catcheside Incomplete Repair — Dale (1985)
For mono-exponential sublethal damage repair rate $\mu = \ln(2)/T_{1/2}$, single-fraction duration $T$, and $m$ daily fractions separated by $\Delta t$:

$$G(\mu T) = \frac{2}{(\mu T)^2}\left(\mu T - 1 + e^{-\mu T}\right), \qquad h_m(\theta) = \frac{2}{m}\frac{\theta}{1-\theta}\left(m - \frac{1-\theta^m}{1-\theta}\right), \quad \theta = e^{-\mu \Delta t}$$

$$\mathrm{BED}_{\mathrm{IR}}(D, n) = D\left(1 + \frac{\big(G(\mu T) + h_m\big)\,d}{\alpha/\beta}\right)$$

For $\mu T < 10^{-3}$, $G(\mu T)$ uses the 4th-order Taylor expansion $1 - x/3 + x^2/12 - x^3/60$ to prevent catastrophic cancellation.

### 2.4 Phenomenological Variable Proton RBE — McNamara et al. (2015)
From voxel-wise dose-averaged $\mathrm{LET}_d$ ($\mathrm{keV}/\mu\mathrm{m}$):
$$\mathrm{RBE}_{\max} = p_0 + \frac{p_1}{\alpha/\beta}\,\mathrm{LET}_d, \qquad \mathrm{RBE}_{\min} = p_2 + p_3 \sqrt{\alpha/\beta}\,\mathrm{LET}_d$$
with $p_0 = 0.99064$, $p_1 = 0.35605$, $p_2 = 1.1012$, $p_3 = -0.0038703$.

Defining $d_{\mathrm{eff}} = \mathrm{RBE}(d_p) \cdot d_p = \frac{1}{2}\left(\sqrt{(\alpha/\beta)^2 + 4 d_p (\alpha/\beta)\mathrm{RBE}_{\max} + 4 d_p^2 \mathrm{RBE}_{\min}^2} - \alpha/\beta\right)$, algebraic squaring yields the **exact quadratic identity**:
$$\frac{\mathrm{BED}}{n} = d_{\mathrm{eff}}\left(1 + \frac{d_{\mathrm{eff}}}{\alpha/\beta}\right) = \mathrm{RBE}_{\max}\,d_p + \frac{\mathrm{RBE}_{\min}^2}{\alpha/\beta}\,d_p^2$$

Therefore, the **exact closed-form inverse** for physical proton dose $D_p = n d_p$ is:
$$D_p = n \cdot \frac{2(\mathrm{BED}/n)}{\mathrm{RBE}_{\max} + \sqrt{\mathrm{RBE}_{\max}^2 + 4\frac{\mathrm{RBE}_{\min}^2}{\alpha/\beta}(\mathrm{BED}/n)}}$$

---

## 3. 3D Diffeomorphic DIR-Uncertainty Envelope & AAPM TG-132 Audit

1. **Pull-back Warping:** Prior 3D physical dose $D_1$ is mapped onto the Course-2 reference grid via trilinear interpolation (`order=1`), guaranteeing strict positivity preservation and zero Gibbs overshoot.
2. **Jacobian Determinant Audit:**
   $$J(\mathbf{x}) = \det\big(\mathbf{I} + \nabla \mathbf{u}(\mathbf{x})\big)$$
   evaluated in physical millimeters. Any folded voxel ($J \le 0$) inside patient tissue triggers a **hard `AbstentionError("DIR_TOPOLOGY_FOLDING")`**.
3. **Inverse-Consistency Error (ICE):**
   $$\mathrm{ICE}(\mathbf{x}) = \left\| \mathbf{u}_f(\mathbf{x}) + \mathbf{u}_b\big(\mathbf{x} + \mathbf{u}_f(\mathbf{x})\big) \right\|_2$$
   audited against AAPM TG-132 (warning at $\mathrm{ICE}_{p95} > 2.0\ \mathrm{mm}$; hard abstention at $> 5.0\ \mathrm{mm}$).
4. **Diffeomorphic Uncertainty Ensemble:** $M$ zero-mean perturbations combining rigid residual alignment error and Gaussian-smoothed elastic Stationary Velocity Fields integrated via scaling-and-squaring ($\exp(\mathbf{v})$) calibrated to clinical 3D RMS Target Registration Error $\sigma_{\mathrm{TRE}}$ (mm), yielding voxel-wise $p_{05}$, nominal, $p_{50}$, $p_{95}$, and worst-case envelopes.

---

## 4. Formal Verification Invariants

Every run enforces and records five non-negotiable invariants in the SHA-256 `ProvenanceRecord`:
1. `COURSE_FRAME_OF_REFERENCE_UID_PRESENT` — Explicit DICOM UID on every course.
2. `DIR_POSITIVE_JACOBIAN_DIFFEOMORPHISM` — Zero folded voxels ($J > 0$ everywhere in tissue).
3. `DIR_INVERSE_CONSISTENCY_WITHIN_HARD_BOUND` — $\mathrm{ICE}_{p95} \le 5.0\ \mathrm{mm}$.
4. `DIR_ENVELOPE_QUANTILE_MONOTONICITY` — $p_{05} \le p_{95} \le \mathrm{worst}$ at every voxel.
5. `FORWARD_INVERSE_RESIDUAL_LE_1e-06_GY` — Independent forward evaluation of the inverted physical dose ceiling matches remaining EQD2 to $\le 10^{-6}\ \mathrm{Gy}$ (measured typically $< 10^{-12}\ \mathrm{Gy}$).

---

## 5. Missing-DICOM / Legacy-PDF Physics-Constrained 3D Dose Envelope Reconstruction (`v0.3.0`)

When prior DICOM-RT files are unavailable (only an unstructured outside-hospital PDF/text summary with prescription, fractions, and sparse OAR DVH metrics), `legacy_reconstructor.py` reconstructs provably ordered 3D dose bounds $0 \le D_{\mathrm{lower}}(\mathbf{x}) \le D_{\mathrm{nom}}(\mathbf{x}) \le D_{\mathrm{upper}}(\mathbf{x})$ directly on the current planning CT:

1. **Source-Span Provenance & Abstention:** Every extracted fact is bound to its verbatim `[start_char, end_char]` quote. Hard `AbstentionError` gates refuse missing fractionation (`LEGACY_MISSING_FRACTIONATION`), conflicting facts (`LEGACY_CONFLICTING_FACTS`), unphysical DVH constraints (`LEGACY_UNPHYSICAL_DVH_CONSTRAINT`), or missing spatial/DVH anchors (`LEGACY_NO_ANCHOR_CONSTRAINTS`).
2. **Lipschitz-Penumbra 3D Upper Bound:** For each OAR $j$ with reported near-max $m_j$ and inter-course anatomy shift cushion $U_j = m_j(1 + \epsilon_{\mathrm{rel}}) + \epsilon_{\mathrm{abs}}$, physical beam penumbra limits $|\nabla D| \le G_{\max}$ (e.g., $3.8\ \mathrm{Gy/mm}$ for 6 MV IMRT, $5.5\ \mathrm{Gy/mm}$ for SBRT, $6.5\ \mathrm{Gy/mm}$ for protons) bound exterior dose via 3D physical Euclidean distance transforms:
   $$D_{\mathrm{upper}}(\mathbf{x}) \le \min_j \Big( U_j + G_{\max}\,\mathrm{dist}(\mathbf{x}, \Omega_j) \Big)$$
3. **Projections Onto Convex Sets (POCS):** Alternating bounded water-filling (`mean(clip(d + \delta, L, U)) == D_{\mathrm{mean}}`) and monotone near-max projections satisfy every reported legacy DVH metric to $< 0.05\ \mathrm{Gy}$ (measured `0.000 Gy` across OpenKBP `pt_1`–`pt_3`).
4. **DICOM Recovery Value ($\Delta D_{\mathrm{headroom}}$):** Quantifies the exact additional Course-2 physical dose headroom unlocked if the outside hospital's original DICOM is recovered:
   $$\Delta D_{\mathrm{recovery}} = D_{\mathrm{rem,DICOM}} - D_{\mathrm{rem,Legacy}}$$
