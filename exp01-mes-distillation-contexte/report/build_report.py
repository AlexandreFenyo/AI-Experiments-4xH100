"""Build RAPPORT.html and RAPPORT.pdf from RAPPORT.md (Markdown -> HTML, then Edge headless print-to-PDF).

    python build_report.py
"""

import subprocess
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

# Figure: completeness and faithfulness at each step of the recipe (two series, one axis, direct labels).
STEPS = [
    ("Ancien Qwen3-14B", 30.7, 54.7),
    ("Gemma 31B, anciennes données (548 ex.)", 29.8, 58.5),
    ("Qwen3-14B, nouvelle méthode (15 496 ex.)", 68.7, 85.7),
    ("Gemma 31B, nouvelle méthode (15 496 ex.)", 74.6, 95.0),
    ("Gemma 31B, 42 770 exemples", 77.3, 98.7),
    ("Gemma 31B, recette combinée (48 203 ex.)", 75.7, 97.6),
    ("Mistral 128B, recette combinée", 76.9, 99.0),
    ("Gemma 31B, réponses officielles + refus (R1)", 88.6, 99.7),
    ("Gemma 31B avec la FAQ en contexte (plafond)", 79.0, 99.6),
]
C1, C2 = "#2a78d6", "#1baf7a"


def figure():
    w, left, bar_h, gap, row = 760, 300, 11, 2, 40
    scale = (w - left - 60) / 100
    h = 34 + row * len(STEPS) + 26
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Complétude et fidélité à chaque étape" '
           f'style="width:100%;height:auto;font-family:Segoe UI,system-ui,sans-serif">']
    out.append(f'<rect x="{left}" y="6" width="10" height="10" rx="2" fill="{C1}"/><text x="{left + 15}" y="15" font-size="11" fill="#52514e">Complétude</text>')
    out.append(f'<rect x="{left + 100}" y="6" width="10" height="10" rx="2" fill="{C2}"/><text x="{left + 115}" y="15" font-size="11" fill="#52514e">Fidélité à la FAQ</text>')
    for g in range(0, 101, 25):
        x = left + g * scale
        out.append(f'<line x1="{x}" x2="{x}" y1="26" y2="{h - 22}" stroke="#e9e7e1"/>'
                   f'<text x="{x}" y="{h - 8}" font-size="10" fill="#86847d" text-anchor="middle">{g} %</text>')
    for i, (name, c, f) in enumerate(STEPS):
        y = 34 + i * row
        out.append(f'<text x="{left - 10}" y="{y + 16}" font-size="11.5" fill="#0b0b0b" text-anchor="end">{name}</text>')
        for k, (v, col) in enumerate(((c, C1), (f, C2))):
            yy = y + 3 + k * (bar_h + gap)
            out.append(f'<rect x="{left}" y="{yy}" width="{v * scale:.1f}" height="{bar_h}" rx="3" fill="{col}"/>'
                       f'<text x="{left + v * scale + 5:.1f}" y="{yy + 9}" font-size="10.5" fill="#52514e">{str(v).replace(".", ",")}</text>')
    out.append("</svg>")
    return "".join(out)


CSS = """
@page { size: A4; margin: 16mm 15mm; }
body { font: 10.5pt/1.5 "Segoe UI", system-ui, sans-serif; color: #0b0b0b; max-width: 180mm; margin: 0 auto; }
h1 { font-size: 21pt; margin: 0 0 4pt; letter-spacing: -.01em; }
h2 { font-size: 13.5pt; margin: 20pt 0 6pt; padding-top: 6pt; border-top: 1px solid #e2e0d9; }
h3 { font-size: 11.5pt; margin: 14pt 0 4pt; }
p, li { orphans: 3; widows: 3; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0 10pt; font-size: 9pt; page-break-inside: avoid; }
th, td { border-bottom: 1px solid #e2e0d9; padding: 4pt 6pt; text-align: left; vertical-align: top; }
th { color: #52514e; font-weight: 600; border-bottom: 1.5px solid #c9c6bd; }
td:not(:first-child), th:not(:first-child) { font-variant-numeric: tabular-nums; }
code { font: 9pt Consolas, monospace; background: #f3f2ee; padding: 0 3px; border-radius: 3px; }
a { color: #1c5cab; text-decoration: none; word-break: break-all; }
figure { margin: 10pt 0 4pt; page-break-inside: avoid; }
figcaption { font-size: 9pt; color: #52514e; margin-top: 2pt; }
strong { font-weight: 650; }
"""


def main():
    md = (HERE / "RAPPORT.md").read_text(encoding="utf-8")
    assert "{{" not in md, "unfilled placeholder in RAPPORT.md"
    html = markdown.markdown(md, extensions=["tables", "sane_lists"])
    fig = (f"<figure>{figure()}<figcaption>Complétude et fidélité sur les 704 questions de test, modèle interrogé sans "
           "la FAQ (sauf la dernière ligne).</figcaption></figure>")
    html = html.replace("<h2>2. Contexte</h2>", fig + "<h2>2. Contexte</h2>", 1)
    page = f'<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>Faire entrer une FAQ dans les poids d\'un modèle</title><style>{CSS}</style></head><body>{html}</body></html>'
    (HERE / "RAPPORT.html").write_text(page, encoding="utf-8")
    subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={HERE / 'RAPPORT.pdf'}", (HERE / "RAPPORT.html").as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("built", HERE / "RAPPORT.pdf", (HERE / "RAPPORT.pdf").stat().st_size, "bytes")


if __name__ == "__main__":
    main()
