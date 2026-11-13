"""Experiment 01, serving cost report: figures + Markdown from the H100 measurements and the DGX Spark simulation.

    python exp01_spark_report.py      (in ~/exp/e1; reads results/exp01_bench_h100.jsonl, exp01_sim_h100.json,
                                       exp01_sim_spark.json; writes report/spark/)
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

sim = json.load(open(E1 / "results" / "exp01_sim_spark.json"))
h100 = json.load(open(E1 / "results" / "exp01_sim_h100.json"))
spark = sim["spark"]


def get(name, scen, n):
    return next(r for r in sim["spark"][name] if r["scenario"] == scen and r["n"] == n)


NS = sorted({r["n"] for r in spark["spark"]})


def style(ax, title, ylabel, logy=False):
    ax.set_facecolor(SURF)
    ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    ax.set_xticks(NS)
    ax.set_xticklabels([str(n) for n in NS])
    ax.set_xlabel("Requêtes simultanées (une question par usager)", color=TXT2, fontsize=9)
    ax.set_ylabel(ylabel, color=TXT2, fontsize=9)
    ax.set_title(title, color=TXT, fontsize=10.5, loc="left")
    ax.grid(True, color=GRID, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TXT2, labelsize=8.5)


def line_fig(metric, key, fname, title, ylabel, logy):
    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=170)
    fig.patch.set_facecolor(SURF)
    for s in SC:
        y = [get("spark", s, n)[metric][key] for n in NS]
        lo = [get("spark_opt", s, n)[metric][key] for n in NS]
        hi = [get("spark_pess", s, n)[metric][key] for n in NS]
        ax.fill_between(NS, lo, hi, color=COL[s], alpha=0.10, linewidth=0)
        ax.plot(NS, y, color=COL[s], linewidth=2, marker=MARK[s], markersize=5, label=LAB[s])
        ax.annotate(f"{y[-1]:.0f} s" if y[-1] >= 10 else f"{y[-1]:.1f} s", (NS[-1], y[-1]), xytext=(6, 0),
                    textcoords="offset points", fontsize=8, color=TXT2, va="center")
    style(ax, title, ylabel, logy)
    ax.legend(frameon=False, fontsize=8, labelcolor=TXT, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / fname, facecolor=SURF)
    plt.close(fig)


line_fig("total", "mean", "fig_reponse.png", "Temps moyen pour obtenir la réponse complète (DGX Spark, estimation)",
         "secondes (échelle logarithmique)", True)
line_fig("ttft", "mean", "fig_premier_mot.png", "Temps moyen avant le premier mot de la réponse (DGX Spark, estimation)",
         "secondes (échelle logarithmique)", True)

# capacity (Little's law on the simulated burst: N users / mean answer time)
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=170)
fig.patch.set_facecolor(SURF)
for s in SC:
    y = [n * 3600 / get("spark", s, n)["total"]["mean"] for n in NS]
    ax.plot(NS, y, color=COL[s], linewidth=2, marker=MARK[s], markersize=5, label=LAB[s])
    ax.annotate(f"{y[-1]:,.0f}".replace(",", " "), (NS[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                fontsize=8, color=TXT2, va="center")
style(ax, "Réponses par heure qu'un DGX Spark peut servir (estimation)", "réponses par heure")
ax.legend(frameon=False, fontsize=8, labelcolor=TXT, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "fig_capacite.png", facecolor=SURF)
plt.close(fig)

# H100 validation scatter
val = h100["h100_validation"]
fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=170)
fig.patch.set_facecolor(SURF)
for s in SC:
    vs = [v for v in val if v["scenario"] == s]
    ax.scatter([v["meas_total_med"] for v in vs], [v["sim_total_med"] for v in vs], color=COL[s], marker=MARK[s], s=26,
               label=LAB[s] + " (réponse)", zorder=3, edgecolors=SURF, linewidths=1)
    ax.scatter([v["meas_ttft_med"] for v in vs], [v["sim_ttft_med"] for v in vs], facecolors="none", edgecolors=COL[s],
               marker=MARK[s], s=26, label=LAB[s] + " (premier mot)", zorder=3)
lim = [0.02, 300]
ax.plot(lim, lim, color=TXT2, linewidth=0.8, linestyle="--")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_facecolor(SURF); ax.grid(True, color=GRID, linewidth=0.6)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.set_xlabel("mesuré sur 2 H100 (s, médiane)", color=TXT2, fontsize=9)
ax.set_ylabel("simulé (s, médiane)", color=TXT2, fontsize=9)
ax.set_title("Validation du simulateur sur 21 mesures H100", color=TXT, fontsize=10.5, loc="left")
ax.tick_params(colors=TXT2, labelsize=8)
ax.legend(frameon=False, fontsize=6.5, labelcolor=TXT, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "fig_validation.png", facecolor=SURF)
plt.close(fig)


def f(x):
    return f"{x:.1f}" if x < 10 else f"{x:.0f}"


rows = []
for n in NS:
    a, b, c = (get("spark", s, n) for s in SC)
    rows.append(f"| {n} | {f(a['ttft']['mean'])} s | {f(a['total']['mean'])} s | {f(b['ttft']['mean'])} s | "
                f"{f(b['total']['mean'])} s | {f(c['ttft']['mean'])} s | {f(c['total']['mean'])} s | "
                f"{b['total']['mean'] / a['total']['mean']:.1f} | {c['total']['mean'] / a['total']['mean']:.1f} |")
SPARK_TABLE = "\n".join(rows)

cap = []
for n in (1, 16, 64, 128):
    v = [n * 3600 / get("spark", s, n)["total"]["mean"] for s in SC]
    cap.append(f"| {n} | {v[0]:,.0f} | {v[1]:,.0f} | {v[2]:,.0f} |".replace(",", " "))
CAP_TABLE = "\n".join(cap)

sens = []
for name, lab in (("spark_opt", "optimiste"), ("spark", "central"), ("spark_pess", "pessimiste")):
    for n in (1, 16, 64):
        a, b, c = (get(name, s, n)["total"]["mean"] for s in SC)
        sens.append(f"| {lab} | {n} | {f(a)} s | {f(b)} s | {f(c)} s |")
SENS_TABLE = "\n".join(sens)

meas_rows = []
for v in val:
    meas_rows.append(f"| {LAB[v['scenario']]} | {v['n']} | {v['meas_ttft_med']:.2f} | {v['sim_ttft_med']:.2f} | "
                     f"{v['meas_total_med']:.2f} | {v['sim_total_med']:.2f} | {v['meas_wall']:.1f} | {v['sim_wall']:.1f} |")
MEAS_TABLE = "\n".join(meas_rows)
errs = [abs(v["sim_total_med"] / v["meas_total_med"] - 1) for v in val]
ERR_MED = sorted(errs)[len(errs) // 2] * 100
ERR_MAX = max(errs) * 100
fit = h100["h100_fit"]

json.dump({"spark_table": SPARK_TABLE, "cap": CAP_TABLE, "sens": SENS_TABLE, "meas": MEAS_TABLE, "err_med": ERR_MED,
           "err_max": ERR_MAX, "fit": fit}, open(OUT / "tables.json", "w"), indent=1, ensure_ascii=False)
print(SPARK_TABLE); print(CAP_TABLE); print(SENS_TABLE); print(f"validation error on answer time: median {ERR_MED:.1f} %, max {ERR_MAX:.1f} %")
