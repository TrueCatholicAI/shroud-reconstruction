"""Standardize navigation across all HTML files in docs/."""
import re
from pathlib import Path

DOCS = Path(__file__).resolve().parent / "docs"

# Canonical nav — every existing page, in display order
NAV_LINKS = [
    ("index.html", "Home"),
    ("methodology.html", "Methodology"),
    ("findings.html", "Findings"),
    ("reconstruction.html", "Reconstruction"),
    ("study2.html", "Study 2"),
    ("neural-depth.html", "Neural Depth"),
    ("full-body.html", "Full Body"),
    ("formation-analysis.html", "Formation"),
    ("distance-function.html", "Distance"),
    ("scourge-analysis.html", "Scourge"),
    ("neave-comparison.html", "Neave"),
    ("wound-mapping.html", "Wound Mapping"),
    ("wavelet-analysis.html", "Wavelet"),
    ("bilateral-analysis.html", "Bilateral"),
    ("seed-sweep.html", "Seed Sweep"),
    ("curvature-analysis.html", "Curvature"),
    ("sudarium.html", "Sudarium"),
    ("art-historical.html", "Art History"),
    ("viewer.html", "3D Viewer"),
    ("summary.html", "Summary"),
    ("about.html", "About"),
]

def build_nav(active_href):
    """Build the nav HTML with the active page marked."""
    links = []
    for href, label in NAV_LINKS:
        cls = ' class="active"' if href == active_href else ''
        links.append(f'      <li><a href="{href}"{cls}>{label}</a></li>')
    link_block = "\n".join(links)
    return f'''<nav>
  <div class="nav-inner">
    <a class="nav-brand" href="index.html">Shroud Reconstruction</a>
    <ul class="nav-links">
{link_block}
    </ul>
  </div>
</nav>'''

# Match the entire <nav>...</nav> block
NAV_PATTERN = re.compile(r'<nav>.*?</nav>', re.DOTALL)

updated = 0
for html_file in sorted(DOCS.glob("*.html")):
    content = html_file.read_text(encoding="utf-8")
    match = NAV_PATTERN.search(content)
    if not match:
        print(f"  SKIP {html_file.name}: no <nav> found")
        continue

    old_nav = match.group(0)
    new_nav = build_nav(html_file.name)

    if old_nav == new_nav:
        print(f"  OK   {html_file.name}: nav already correct")
        continue

    content = content[:match.start()] + new_nav + content[match.end():]
    html_file.write_text(content, encoding="utf-8")
    updated += 1
    print(f"  FIX  {html_file.name}")

print(f"\nUpdated {updated} files")
