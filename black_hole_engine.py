#!/usr/bin/env python3
"""
Black hole null-geodesic engine (G = c = 1, M = 1)

Pipeline per metric  ds^2 = -f dt^2 + dr^2/f + r^2 dphi^2  (equatorial plane):
  1. symbolic derivation (Lagrangian -> constants of motion -> orbit equation)
  2. horizons, photon sphere, critical impact parameter (grid scan + brentq,
     with an exact symbolic answer attempted when the polynomial is simple)
  3. vectorised RK4 ray tracing for a STATIC OBSERVER at finite radius r_obs
  4. plots: ray diagram, ray-traced shadow image, effective potential

Usage:
    python black_hole_engine.py                 # interactive menu
    python black_hole_engine.py --metric 2
    python black_hole_engine.py --metric all --no-show     # compare all 8
"""
import argparse
import os
from dataclasses import dataclass, field

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from scipy.optimize import brentq

# ----------------------------------------------------------------------------
# Symbols and metric registry
# ----------------------------------------------------------------------------
r = sp.Symbol("r", positive=True)
M = sp.Symbol("M", positive=True)
Q = sp.Symbol("Q", positive=True)      # Reissner-Nordstrom charge
Lam = sp.Symbol("Lambda")              # Kottler cosmological constant
g = sp.Symbol("g", positive=True)      # Bardeen monopole charge
l = sp.Symbol("l", positive=True)      # Hayward length scale
Ads = sp.Symbol("Ads", positive=True)  # AdS radius
a = sp.Symbol("a", positive=True)      # Kiselev normalisation
w = sp.Symbol("w")                     # Kiselev equation of state
gam = sp.Symbol("gamma", positive=True)  # Mannheim-Kazanas linear term
k = sp.Symbol("k", positive=True)        # Mannheim-Kazanas quadratic term

R_MAX = 1.0e3          # outer limit for root scans
R_OBS_DEFAULT = 20.0   # preferred observer radius (in M)


@dataclass
class Metric:
    name: str
    f: sp.Expr
    params: dict = field(default_factory=dict)  # EXACT sympy numbers only


def _m(name, f, **extra):
    params = {M: sp.Integer(1)}
    params.update(extra)
    return Metric(name, f, params)


METRICS = {
    "1": _m("Schwarzschild (Standard)", 1 - 2 * M / r),
    "2": _m("Reissner-Nordström (Bercas Q)", 1 - 2 * M / r + Q**2 / r**2),
    "3": _m("Kottler / S-deSitter (Pemalar Lambda)",
            1 - 2 * M / r - Lam * r**2 / 3),
    "4": _m("Bardeen (Regular / Tanpa Singularity)",
            1 - 2 * M * r**2 / (r**2 + g**2) ** sp.Rational(3, 2)),
    "5": _m("Schwarzschild-Anti-de Sitter (SAdS)",
            1 - 2 * M / r + r**2 / Ads**2),
    "6": _m("Hayward (Regular / Teras Quantum)",
            1 - 2 * M * r**2 / (r**3 + 2 * M * l**2)),
    "7": _m("Kiselev (Quintessence Fluid)",
            1 - 2 * M / r - a / r ** (3 * w + 1)),
    "8": _m("Mannheim-Kazanas (Conformal Gravity)",
            1 - 2 * M / r + gam * r - k * r**2),
}
# Parameter values (exact rationals -> exact symbolic output).
METRICS["2"].params = {M: sp.Integer(1), Q: sp.Rational(1, 2)}
METRICS["3"].params = {M: sp.Integer(1), Lam: sp.Rational(1, 100)}
METRICS["4"].params = {M: sp.Integer(1), g: sp.Rational(3, 5)}
METRICS["5"].params = {M: sp.Integer(1), Ads: sp.Integer(10)}
METRICS["6"].params = {M: sp.Integer(1), l: sp.Rational(1, 2)}
METRICS["7"].params = {M: sp.Integer(1), a: sp.Rational(1, 20),
                       w: sp.Rational(-2, 3)}
# NOTE (Mannheim-Kazanas): the physical values (gamma ~ 3e-28 /m,
# k ~ 1e-54..1e-50 /m^2) are utterly negligible when lengths are in units of GM/c^2
# for a stellar-mass hole. These are DIMENSIONLESS illustrative values so the
# conformal terms are actually visible. See physical_to_geometric() below.
METRICS["8"].params = {M: sp.Integer(1), gam: sp.Rational(1, 100),
                       k: sp.Rational(1, 10000)}


def physical_to_geometric(gamma_per_m, k_per_m2, mass_kg):
    """Convert SI MK parameters to the dimensionless values used with M = 1."""
    G, c = 6.67430e-11, 2.99792458e8
    length = G * mass_kg / c**2          # GM/c^2 in metres
    return gamma_per_m * length, k_per_m2 * length**2


# ----------------------------------------------------------------------------
# Display helpers (work in Jupyter and in a plain terminal)
# ----------------------------------------------------------------------------
try:
    get_ipython  # noqa: F821
    from IPython.display import display as _display
    IN_NOTEBOOK = True
except NameError:
    IN_NOTEBOOK = False


def show(title, lhs, rhs):
    eq = sp.Eq(lhs, rhs)
    print(f"--- {title} ---")
    if IN_NOTEBOOK:
        _display(eq)
    else:
        sp.pprint(eq, use_unicode=True)
    print("LaTeX:", sp.latex(eq), "\n")


# ----------------------------------------------------------------------------
# 1. Symbolic derivation (generic f(r), then the chosen metric)
# ----------------------------------------------------------------------------
def derive_geodesic_equations(metric):
    E, L, b = sp.symbols("E L b", positive=True)
    td, rd, pd = sp.symbols("t_dot r_dot phi_dot", real=True)
    rd2 = sp.Symbol("rd2")
    fr = sp.Function("f")(r)
    half = sp.Rational(1, 2)

    lag = half * (-fr * td**2 + rd**2 / fr + r**2 * pd**2)
    show("LAGRANGIAN", sp.Symbol("Lagrangian"), lag)

    td_sol = sp.solve(sp.Eq(sp.diff(lag, td), -E), td)[0]     # p_t   = -E
    pd_sol = sp.solve(sp.Eq(sp.diff(lag, pd), L), pd)[0]      # p_phi =  L
    show("CONSTANT OF MOTION: t_dot", td, td_sol)
    show("CONSTANT OF MOTION: phi_dot", pd, pd_sol)

    null = (2 * lag).subs({td: td_sol, pd: pd_sol}).subs(rd, sp.sqrt(rd2))
    rd2_sol = sp.simplify(sp.solve(sp.Eq(null, 0), rd2)[0])
    show("RADIAL EQUATION r_dot^2", sp.Symbol("r_dot^2"), rd2_sol)

    drdphi2 = sp.simplify((rd2_sol / pd_sol**2).subs(E, L / b))
    show("ORBIT EQUATION (dr/dphi)^2  [generic f]",
         sp.Symbol("(dr/dphi)^2"), drdphi2)
    specific = sp.simplify(drdphi2.subs(fr, metric.f))
    show("ORBIT EQUATION (dr/dphi)^2  [this metric]",
         sp.Symbol("(dr/dphi)^2"), specific)


# ----------------------------------------------------------------------------
# 2. Numerics helpers
# ----------------------------------------------------------------------------
def lam(expr, var):
    """lambdify that always returns a float ndarray shaped like the input."""
    fn = sp.lambdify(var, expr, modules="numpy")

    def wrapped(x):
        x = np.asarray(x, dtype=float)
        with np.errstate(all="ignore"):
            return np.asarray(fn(x), dtype=float) + np.zeros_like(x)

    return wrapped


def find_roots(func, lo, hi, n=40000):
    """All simple roots of func in [lo, hi]: [(root, +1 rising / -1 falling)]."""
    grid = np.geomspace(lo, hi, n)
    vals = func(grid)
    ok = np.isfinite(vals[:-1]) & np.isfinite(vals[1:])
    idx = np.where(ok & (np.sign(vals[:-1]) * np.sign(vals[1:]) < 0))[0]
    out = []
    for i in idx:
        root = brentq(lambda x: float(func(np.float64(x))),
                      grid[i], grid[i + 1], xtol=1e-14)
        out.append((root, 1 if vals[i + 1] > vals[i] else -1))
    return out


def exact_root_match(expr, numeric, max_deg=2):
    """Exact root of the numerator of `expr` closest to `numeric`, if the
    numerator is a low-degree polynomial in r; otherwise None."""
    try:
        poly = sp.Poly(sp.expand(sp.numer(sp.together(expr))), r)
        if not 1 <= poly.degree() <= max_deg:
            return None
        for cand in sp.roots(poly, r):
            val = complex(sp.N(cand))
            if abs(val.imag) < 1e-12 and abs(val.real - numeric) < 1e-7 * max(1, abs(numeric)):
                return sp.simplify(cand)
    except Exception:
        return None
    return None


# ----------------------------------------------------------------------------
# 3. Full analysis of one metric
# ----------------------------------------------------------------------------
@dataclass
class Analysis:
    key: str
    metric: Metric
    f_sym: sp.Expr
    f_fn: object
    V_fn: object
    H_fn: object
    ud2: object
    r_h: float
    r_c: float
    r_inner: list
    r_ph: float
    b_crit: float
    r_obs: float
    b_max: float          # largest impact parameter visible from r_obs (alpha = 90 deg)
    exact: dict


def analyze(key, metric):
    f_sym = metric.f.subs(metric.params)
    f_fn = lam(f_sym, r)

    # --- horizons --------------------------------------------------------
    roots = find_roots(f_fn, 1e-2, R_MAX)
    rising = [x for x, d in roots if d > 0]
    if not rising:
        raise RuntimeError("No event horizon found: f(r) never crosses zero "
                           "upwards (naked singularity for these parameters?).")
    r_h = max(rising)
    falling_out = [x for x, d in roots if d < 0 and x > r_h]
    r_c = min(falling_out) if falling_out else None
    r_inner = sorted(x for x, d in roots if d < 0 and x < r_h)

    # --- photon sphere: maximum of V = f / r^2 outside the horizon --------
    V_sym = f_sym / r**2
    dV_sym = sp.diff(V_sym, r)
    V_fn, dV_fn = lam(V_sym, r), lam(dV_sym, r)
    d2V_fn = lam(sp.diff(V_sym, r, 2), r)
    hi = r_c * (1 - 1e-6) if r_c else R_MAX
    ph_roots = find_roots(dV_fn, r_h * (1 + 1e-6), hi)
    maxima = [x for x, _ in ph_roots if float(d2V_fn(np.float64(x))) < 0]
    if not maxima:
        raise RuntimeError("No photon sphere (maximum of V) outside the horizon.")
    r_ph = max(maxima)
    b_crit = 1.0 / np.sqrt(float(V_fn(np.float64(r_ph))))

    # --- exact symbolic versions where possible ---------------------------
    exact = {}
    exact["r_h"] = exact_root_match(f_sym, r_h)
    exact["r_ph"] = exact_root_match(dV_sym, r_ph)
    if exact["r_ph"] is not None:
        try:
            bx = 1 / sp.sqrt(V_sym.subs(r, exact["r_ph"]))
            bx = sp.sqrtdenest(sp.simplify(sp.radsimp(bx)))
            exact["b_crit"] = bx if sp.count_ops(bx) <= 20 else None  # skip ugly forms
        except Exception:
            exact["b_crit"] = None
    else:
        exact["b_crit"] = None

    # --- observer: static, must sit where f > 0 ---------------------------
    r_obs = R_OBS_DEFAULT
    if r_c is not None and r_obs > 0.9 * r_c:
        r_obs = r_ph + 0.6 * (r_c - r_ph)
    f_obs = float(f_fn(np.float64(r_obs)))
    assert f_obs > 0, "observer not in a static region"
    b_max = r_obs / np.sqrt(f_obs)

    # --- geodesic ODE in u = 1/r:  u'' = -1/2 dH/du,  H = f(1/u) u^2 -------
    u = sp.Symbol("u", positive=True)
    H_sym = (f_sym.subs(r, 1 / u) * u**2)
    H_fn = lam(H_sym, u)
    ud2 = lam(-sp.Rational(1, 2) * sp.diff(H_sym, u), u)

    return Analysis(key, metric, f_sym, f_fn, V_fn, H_fn, ud2, r_h, r_c,
                    r_inner, r_ph, b_crit, r_obs, b_max, exact)


def report(an):
    print(f"\n=== {an.metric.name} ===")
    show("LAPSE FUNCTION f(r)", sp.Symbol("f"), an.metric.f)
    print("parameters:", {str(s): v for s, v in an.metric.params.items()})
    ex = an.exact
    print(f"event horizon       r_h  = {an.r_h:.8f}" +
          (f"   exact: {ex['r_h']}" if ex["r_h"] is not None else ""))
    if an.r_inner:
        print("inner horizon(s)         = " + ", ".join(f"{x:.6f}" for x in an.r_inner))
    if an.r_c:
        print(f"cosmological horizon r_c = {an.r_c:.8f}")
    print(f"photon sphere       r_ph = {an.r_ph:.8f}" +
          (f"   exact: {ex['r_ph']}" if ex["r_ph"] is not None else ""))
    print(f"critical impact par b_c  = {an.b_crit:.8f}" +
          (f"   exact: {ex['b_crit']}" if ex["b_crit"] is not None else ""))
    print(f"observer radius     r_obs= {an.r_obs:.3f}  (b_max visible = {an.b_max:.3f})\n")


# ----------------------------------------------------------------------------
# 4. Vectorised ray tracer (static observer at r_obs, rays traced inwards)
# ----------------------------------------------------------------------------
def _rk4(ud2, u, up, h):
    k1u, k1p = up, ud2(u)
    k2u = up + 0.5 * h * k1p
    k2p = ud2(u + 0.5 * h * k1u)
    k3u = up + 0.5 * h * k2p
    k3p = ud2(u + 0.5 * h * k2u)
    k4u = up + h * k3p
    k4p = ud2(u + h * k3u)
    return (u + h / 6 * (k1u + 2 * k2u + 2 * k3u + k4u),
            up + h / 6 * (k1p + 2 * k2p + 2 * k3p + k4p))


CAPTURED, ESCAPED, TRAPPED, INVALID = 1, 2, 0, -1


def trace_rays(an, b, dphi=0.01, max_phi=6 * np.pi, keep_history=False):
    """Integrate many photons at once.

    Each ray starts at the observer (r_obs, phi = 0) heading inward with impact
    parameter b, and ends when it (a) reaches the horizon -> CAPTURED, (b) comes
    back out to r_obs -> ESCAPED (phi_final is where it exits), or (c) runs out
    of angle (winding at the photon sphere) -> TRAPPED.
    """
    b = np.atleast_1d(np.asarray(b, dtype=float))
    n = b.size
    u0 = 1.0 / an.r_obs
    up0_sq = 1.0 / b**2 - float(an.H_fn(np.float64(u0)))   # (u')^2 = 1/b^2 - H(u)
    valid = up0_sq > 0
    u = np.full(n, u0)
    up = np.sqrt(np.where(valid, up0_sq, 0.0))
    status = np.where(valid, TRAPPED, INVALID)
    alive = valid.copy()
    phi_final = np.full(n, np.nan)
    u_cap = 1.0 / (an.r_h * 1.002)
    hist = [u.copy()] if keep_history else None

    for step in range(int(max_phi / dphi)):
        if not alive.any():
            break
        u_new, up_new = _rk4(an.ud2, u, up, dphi)
        phi_prev = step * dphi
        cap = alive & (u_new >= u_cap)
        esc = alive & ~cap & (u_new <= u0) & (up_new < 0)
        phi_final[cap] = phi_prev + dphi
        du = u[esc] - u_new[esc]
        frac = np.clip((u[esc] - u0) / np.where(du == 0, 1, du), 0, 1)
        phi_final[esc] = phi_prev + dphi * frac
        status[cap], status[esc] = CAPTURED, ESCAPED
        if keep_history:
            hist.append(np.where(alive, u_new, np.nan))
        u = np.where(alive, u_new, u)
        up = np.where(alive, up_new, up)
        alive &= ~(cap | esc)

    out = dict(b=b, status=status, phi_final=phi_final, dphi=dphi)
    if keep_history:
        out["u_hist"] = np.array(hist)
    return out


# ----------------------------------------------------------------------------
# 5. Plots
# ----------------------------------------------------------------------------
def _finish(fig, path, show_plot):
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=150)
    print("saved", path)
    if show_plot:
        plt.show()
    plt.close(fig)


def plot_rays(an, outdir, show_plot):
    b_vals = np.linspace(0.3, min(0.98 * an.b_max, 2.2 * an.b_crit), 90)
    tr = trace_rays(an, np.append(b_vals, an.b_crit * (1 + 1e-5)),
                    keep_history=True)
    hist, dphi = tr["u_hist"], tr["dphi"]
    phi = np.arange(hist.shape[0]) * dphi

    segs, cols, crit = [], [], None
    for j in range(len(b_vals) + 1):
        m = np.isfinite(hist[:, j])
        rr = 1.0 / hist[m, j]
        xy = np.c_[rr * np.cos(phi[m]), rr * np.sin(phi[m])]
        if j == len(b_vals):
            crit = xy
            continue
        color = "cyan" if tr["status"][j] == ESCAPED else "red"
        for sgn in (1, -1):                       # mirror: spherical symmetry
            segs.append(xy * [1, sgn])
            cols.append(color)

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.add_collection(LineCollection(segs, colors=cols, alpha=0.35, linewidths=0.8))
        ax.plot(*crit.T, color="yellow", lw=2, zorder=10, label="Critical geodesic")
        ax.add_patch(Circle((0, 0), an.r_h, color="black", zorder=5))
        ax.add_patch(Circle((0, 0), an.r_h, fill=False, ec="red", lw=2, zorder=6,
                            label=f"Event horizon ({an.r_h:.3f}M)"))
        ax.add_patch(Circle((0, 0), an.r_ph, fill=False, ec="white", ls="--", lw=1.4,
                            zorder=4, label=f"Photon sphere ({an.r_ph:.3f}M)"))
        ax.scatter([an.r_obs], [0], c="lime", s=40, zorder=11,
                   label=f"Observer (r={an.r_obs:.1f}M)")
        lim = 1.15 * an.r_obs
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal",
               xlabel="x / M", ylabel="y / M",
               title=f"Null geodesics from a static observer — {an.metric.name}")
        ax.legend(loc="upper left")
        _finish(fig, os.path.join(outdir, f"{an.key}_geodesics.png"), show_plot)


def render_shadow(an, outdir, show_plot, res=800):
    """Ray-traced image. Spherical symmetry => trace 1-D in b, map to the 2-D
    image by radius. Escaped rays are coloured with a checkerboard 'sky' sampled
    at the direction they came from, so lensing / Einstein rings are visible."""
    fov = 3.0 * an.b_crit
    x = np.linspace(-fov, fov, res)
    X, Y = np.meshgrid(x, x)
    rho, psi = np.hypot(X, Y), np.arctan2(Y, X)

    b_grid = np.linspace(0.01, min(np.hypot(fov, fov), 0.995 * an.b_max), 6000)
    tr = trace_rays(an, b_grid, dphi=0.01, max_phi=8 * np.pi)
    dark = np.isin(tr["status"], (CAPTURED, TRAPPED)).astype(float)
    phi_f = np.nan_to_num(tr["phi_final"])

    dark_img = np.interp(rho, b_grid, dark) > 0.5
    phi_img = np.interp(rho, b_grid, phi_f)

    theta_s = np.mod(phi_img, 2 * np.pi)           # polar angle on the sky sphere
    wrap = theta_s > np.pi
    theta_s = np.where(wrap, 2 * np.pi - theta_s, theta_s)
    psi_s = np.mod(psi + np.pi * wrap, 2 * np.pi)
    cell = (np.floor(theta_s / (np.pi / 12)) + np.floor(psi_s / (np.pi / 12))) % 2
    image = 0.25 + 0.65 * cell
    image[dark_img] = 0.0

    fig = plt.figure(figsize=(8, 8), facecolor="black")
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap="magma", vmin=0, vmax=1, origin="lower",
              extent=[-fov, fov, -fov, fov])
    ax.add_patch(Circle((0, 0), an.b_crit, fill=False, ec="cyan", ls="--", lw=1,
                        label=f"$b_{{crit}}$ = {an.b_crit:.3f}M"))
    ax.set_title(f"Ray-traced shadow — {an.metric.name}", color="white", fontsize=14)
    ax.set_xlabel("Impact parameter x (M)", color="white")
    ax.set_ylabel("Impact parameter y (M)", color="white")
    ax.tick_params(colors="white")
    ax.legend(loc="upper right")
    _finish(fig, os.path.join(outdir, f"{an.key}_shadow.png"), show_plot)


def plot_potential(an, outdir, show_plot):
    r_max = min(15.0, 0.98 * an.r_c) if an.r_c else 15.0
    r_arr = np.linspace(an.r_h * 1.001, max(r_max, 1.5 * an.r_ph), 2000)
    V = an.V_fn(r_arr)
    V_peak = float(an.V_fn(np.float64(an.r_ph)))

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(r_arr, V, color="#00e5ff", lw=2, label=r"$V_{eff}(r)=f(r)/r^2$")
        ax.scatter([an.r_ph], [V_peak], color="#ff4444", s=100, zorder=5)
        ax.axvline(an.r_ph, color="yellow", ls="--", alpha=0.7,
                   label=f"Photon sphere ($r={an.r_ph:.3f}M$)")
        ax.axhline(V_peak, color="gray", ls="--", alpha=0.5,
                   label=r"Peak $=1/b_{crit}^2$")
        ax.axvline(an.r_h, color="white", ls=":", alpha=0.5,
                   label=f"Event horizon ($r={an.r_h:.3f}M$)")
        ax.set(xlim=(an.r_h, r_arr[-1]),
               ylim=(min(0.0, 1.1 * V.min()), 1.25 * V_peak),
               xlabel="Radius ($r/M$)", ylabel=r"$V_{eff}$",
               title=f"Photon effective potential — {an.metric.name}")
        ax.grid(alpha=0.1)
        ax.legend(loc="upper right")
        _finish(fig, os.path.join(outdir, f"{an.key}_veff.png"), show_plot)


# ----------------------------------------------------------------------------
# 6. Driver
# ----------------------------------------------------------------------------
def run_one(key, outdir, show_plot, derive=True):
    metric = METRICS[key]
    if derive:
        derive_geodesic_equations(metric)
    an = analyze(key, metric)
    report(an)
    plot_rays(an, outdir, show_plot)
    render_shadow(an, outdir, show_plot)
    plot_potential(an, outdir, show_plot)
    return an


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", help="1-8 or 'all'")
    ap.add_argument("--outdir", default="bh_output")
    ap.add_argument("--no-show", action="store_true", help="save figures only")
    args = ap.parse_args()

    if args.no_show:
        plt.switch_backend("Agg")
    os.makedirs(args.outdir, exist_ok=True)

    choice = args.metric
    if choice is None:
        print("--- Pemilih Metrik Lohong Hitam (Berdasarkan Framework Tesis) ---")
        for key, m in METRICS.items():
            print(f"{key}: {m.name}")
        choice = input("Pilih nombor metrik (1-8, atau 'all'): ").strip()

    if choice == "all":
        rows = []
        for key in METRICS:
            an = run_one(key, args.outdir, show_plot=False, derive=False)
            rows.append((an.metric.name, an.r_h, an.r_ph, an.b_crit))
        print("\n{:<42}{:>10}{:>10}{:>10}".format("Metric", "r_h", "r_ph", "b_crit"))
        for name, rh, rph, bc in rows:
            print(f"{name:<42}{rh:>10.4f}{rph:>10.4f}{bc:>10.4f}")
    elif choice in METRICS:
        run_one(choice, args.outdir, show_plot=not args.no_show)
    else:
        print("Pilihan tidak sah.")


if __name__ == "__main__":
    main()
