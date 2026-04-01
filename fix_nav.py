"""
Check and fix navigation consistency across all HTML files.
"""
import os
import re

# Standard nav order based on site structure
STANDARD_NAV = '''      <ul class="nav-links">
        <li><a href="index.html">Home</a></li>
        <li><a href="methodology.html">Methodology</a></li>
        <li><a href="findings.html">Findings</a></li>
        <li><a href="reconstruction.html">Reconstruction</a></li>
        <li><a href="study2.html">Study 2</a></li>
        <li><a href="neural-depth.html">Neural Depth</a></li>
        <li><a href="full-body.html">Full Body</a></li>
        <li><a href="formation-analysis.html">Formation</a></li>
        <li><a href="distance-function.html">Distance</a></li>
        <li><a href="scourge-analysis.html">Scourge</a></li>
        <li><a href="neave-comparison.html">Neave</a></li>
        <li><a href="wound-mapping.html">Wound Mapping</a></li>
        <li><a href="wavelet-analysis.html">Wavelet</a></li>
        <li><a href="bilateral-analysis.html">Bilateral</a></li>
        <li><a href="seed-sweep.html">Seed Sweep</a></li>
        <li><a href="curvature-analysis.html">Curvature</a></li>
        <li><a href="sudarium.html">Sudarium</a></li>
        <li><a href="art-historical.html">Art History</a></li>
        <li><a href="highres-miller.html">HighRes Miller</a></li>
        <li><a href="dorsal-analysis.html">Dorsal</a></li>
        <li><a href="ratio-analysis.html">Ratio</a></li>
        <li><a href="viewer.html">3D Viewer</a></li>
        <li><a href="summary.html">Summary</a></li>
        <li><a href="about.html">About</a></li>
      </ul>'''

# All 24 HTML pages
ALL_PAGES = [
    'index.html', 'methodology.html', 'findings.html', 'reconstruction.html',
    'study2.html', 'neural-depth.html', 'full-body.html', 'formation-analysis.html',
    'distance-function.html', 'scourge-analysis.html', 'neave-comparison.html',
    'wound-mapping.html', 'wavelet-analysis.html', 'bilateral-analysis.html',
    'seed-sweep.html', 'curvature-analysis.html', 'sudarium.html', 'art-historical.html',
    'highres-miller.html', 'dorsal-analysis.html', 'ratio-analysis.html', 'viewer.html',
    'summary.html', 'about.html'
]

def extract_nav(html_content):
    """Extract nav section from HTML."""
    match = re.search(r'<nav>.*?</nav>', html_content, re.DOTALL)
    if match:
        return match.group(0)
    return None

def count_nav_links(nav_content):
    """Count nav links."""
    return len(re.findall(r'<li><a href=', nav_content))

def check_nav_consistency():
    """Check all HTML files for nav consistency."""
    docs_dir = 'docs'
    issues = []
    
    for html_file in ALL_PAGES:
        filepath = os.path.join(docs_dir, html_file)
        if not os.path.exists(filepath):
            print(f"MISSING: {html_file}")
            continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        nav = extract_nav(content)
        if nav:
            link_count = count_nav_links(nav)
            print(f"{html_file}: {link_count} nav links")
        else:
            print(f"NO NAV: {html_file}")
            issues.append(html_file)
    
    return issues

if __name__ == '__main__':
    print("Checking navigation consistency...")
    issues = check_nav_consistency()
    print(f"\nFiles with issues: {issues}")