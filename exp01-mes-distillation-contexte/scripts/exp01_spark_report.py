"""Experiment 01, serving cost report: figures + tables from the H100 measurements and the DGX Spark simulation.

    python exp01_spark_report.py      (in ~/exp/e1; reads results/exp01_sim_h100.json, exp01_sim_spark_closed.json;
                                       writes report/spark/ figures and tables.json)
Main regime: steady state, N people who each ask a new question as soon as they get their answer (N requests always
in progress), which shows the ceiling of the machine.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

E1 = Path.home() / "exp" / "e1"
OUT = E1 / "report" / "spark"
OUT.mkdir(parents=True, exist_ok=True)
COL = {"ft": "#2a78d6", "faq_pc": "#1baf7a", "faq": "#eb6834"}
MARK = {"ft": "o", "faq_pc": "s", "faq": "^"}
LAB = {"ft": "Modèle entraîné (question seule)", "faq_pc": "Modèle de départ + FAQ, cache de préfixe",
       "faq": "Modèle de départ + FAQ, sans cache"}
SC = ("ft", "faq_pc", "faq")
TXT, TXT2, GRID, SURF = "#0b0b0b", "#52514e", "#e2e0d9", "#fcfcfb"

closed = json.load(open(E1 / "results" / "exp01_sim_spark_closed.json"))["closed"]
h100 = json.load(open(E1 / "results" / "exp01_sim_h100.json"))
NS = sorted({r["n"] for r in closed["spark"]})


def get(name, scen, n):
    return next(r for r in closed[name] if r["scenario"] == scen and r["n"] == n)


def style(ax, title, ylabel, logy=False):
    ax.set_facecolor(SURF)
    ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    ax.set_xticks(NS)
    ax.set_xticklabels([str(n) for n in NS])
    ax.set_xlabel("Requêtes en cours en permanence (usagers actifs)", color=TXT2, fontsize=9)
    ax.set_ylabel(ylabel, color=TXT2, fontsize=9)
    ax.set_title(title, color=TXT, fontsize=10.5, loc="left")
    ax.grid(True, color=GRID, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TXT2, labelsize=8.5)


def fmt(v, unit=""):
    if v >= 1000:
        return f"{v:,.0f}".replace(",", " ") + unit
    return (f"{v:.1f}" if v < 10 else f"{v:.0f}").replace(".", ",") + unit


def fig(key, fname, title, ylabel, logy, unit):
    f, ax = plt.subplots(figsize=(7.4, 4.0), dpi=170)
    f.patch.set_facecolor(SURF)
    for s in SC:
        y = [get("spark", s, n)[key] for n in NS]
        lo = [get("spark_opt", s, n)[key] for n in NS]
        hi = [get("spark_pess", s, n)[key] for n in NS]
        ax.fill_between(NS, [min(a, b) for a, b in zip(lo, hi)], [max(a, b) for a, b in zip(lo, hi)],
                        color=COL[s], alpha=0.10, linewidth=0)
        ax.plot(NS, y, color=COL[s], linewidth=2, marker=MARK[s], markersize=4.5, label=LAB[s])
        ax.annotate(fmt(y[-1], unit), (NS[-1], y[-1]), xytext=(6, 0), textcoords="offset points", fontsize=8,
                    color=TXT2, va="center")
    style(ax, title, ylabel, logy)
    ax.legend(frameon=False, fontsize=8, labelcolor=TXT, loc="upper left")
    f.tight_layout()
    f.savefig(OUT / fname, facecolor=SURF)
    plt.close(f)


fig("answers_per_hour", "fig_capacite.png", "Réponses servies par heure par un DGX Spark : le plafond (estimation)",
    "réponses par heure (échelle logarithmique)", True, "")
fig("mean_answer_s", "fig_reponse.png", "Temps moyen pour obtenir la réponse complète (DGX Spark, estimation)",
    "secondes (échelle logarithmique)", True, " s")
fig("mean_ttft_s", "fig_premier_mot.png", "Temps moyen avant le premier mot de la réponse (DGX Spark, estimation)",
    "secondes (échelle logarithmique)", True, " s")

# H100 validation scatter: fitted points (n <= 128) and held-out points (256, 512)
val = h100["h100_validation"]
f, ax = plt.subplots(figsize=(5.4, 4.6), dpi=170)
f.patch.set_facecolor(SURF)
for s in SC:
    vs = [v for v in val if v["scenario"] == s]
    fit_pts = [v for v in vs if v["n"] <= 128]
    held = [v for v in vs if v["n"] > 128]
    ax.scatter([v["meas_total_med"] for v in fit_pts], [v["sim_total_med"] for v in fit_pts], color=COL[s],
               marker=MARK[s], s=26, label=LAB[s], zorder=3, edgecolors=SURF, linewidths=1)
    if held:
        ax.scatter([v["meas_total_med"] for v in held], [v["sim_total_med"] for v in held], facecolors="none",
                   edgecolors=COL[s], marker=MARK[s], s=60, linewidths=1.6, zorder=3,
                   label=LAB[s].split(" (")[0] + " : contrôle, 256 et 512 requêtes")
lim = [1, 400]
ax.plot(lim, lim, color=TXT2, linewidth=0.8, linestyle="--")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_facecolor(SURF); ax.grid(True, color=GRID, linewidth=0.6)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.set_xlabel("mesuré sur 2 H100 (s, médiane de la réponse complète)", color=TXT2, fontsize=8.5)
ax.set_ylabel("simulé (s)", color=TXT2, fontsize=8.5)
ax.set_title("Validation du simulateur sur les mesures H100", color=TXT, fontsize=10.5, loc="left")
ax.tick_params(colors=TXT2, labelsize=8)
ax.legend(frameon=False, fontsize=6.5, labelcolor=TXT, loc="upper left")
f.tight_layout()
f.savefig(OUT / "fig_validation.png", facecolor=SURF)
plt.close(f)

rows = []
for n in NS:
    a, b, c = (get("spark", s, n) for s in SC)
    rows.append(f"| {n} | {fmt(a['answers_per_hour'])} | {fmt(b['answers_per_hour'])} | {fmt(c['answers_per_hour'])} | "
                f"{fmt(a['mean_ttft_s'], ' s')} | {fmt(b['mean_ttft_s'], ' s')} | {fmt(c['mean_ttft_s'], ' s')} | "
                f"{fmt(a['mean_answer_s'], ' s')} | {fmt(b['mean_answer_s'], ' s')} | {fmt(c['mean_answer_s'], ' s')} |")


def plateau(name, s):
    return max(get(name, s, n)["answers_per_hour"] for n in NS)


plat = []
for name, lab in (("spark_opt", "optimiste"), ("spark", "central"), ("spark_pess", "pessimiste"),
                  ("spark_tuned", "central, vLLM réglé pour le débit")):
    p = [plateau(name, s) for s in SC]
    plat.append(f"| {lab} | {fmt(p[0])} | {fmt(p[1])} | {fmt(p[2])} | {p[0] / p[1]:.1f} | {p[0] / p[2]:.0f} |".replace(".", ","))

vrows = []
for v in val:
    vrows.append(f"| {LAB[v['scenario']]} | {v['n']}{' (contrôle)' if v['n'] > 128 else ''} | {v['meas_ttft_med']:.2f} | "
                 f"{v['sim_ttft_med']:.2f} | {v['meas_total_med']:.2f} | {v['sim_total_med']:.2f} |".replace(".", ","))
err = lambda vs: sorted(abs(v["sim_total_med"] / v["meas_total_med"] - 1) * 100 for v in vs)
e_fit, e_held = err([v for v in val if v["n"] <= 128]), err([v for v in val if v["n"] > 128])
tables = {"steady": "\n".join(rows), "plateau": "\n".join(plat), "validation": "\n".join(vrows),
          "err_fit_med": e_fit[len(e_fit) // 2], "err_fit_max": e_fit[-1], "err_held": e_held, "fit": h100["h100_fit"],
          "n_val": len(val)}
json.dump(tables, open(OUT / "tables.json", "w"), indent=1, ensure_ascii=False)
print(tables["steady"]); print(tables["plateau"])
print("fit error med/max", tables["err_fit_med"], tables["err_fit_max"], "held-out", e_held, "points", len(val))
