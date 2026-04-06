"""Methods and references dialog for LunaRad-PEEK."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QTabWidget,
    QTextEdit,
    QPushButton,
    QWidget,
)
from PySide6.QtCore import Qt


METHODS_TEXT = """
<h2>LunaRad-PEEK: Methods and Scientific Basis</h2>

<h3>1. Overview</h3>
<p>LunaRad-PEEK is a conceptual radiation shielding analysis tool that employs an
OLTARIS-inspired workflow for estimating radiation exposure within lunar habitats.
The tool performs directional ray casting from target dosimetry points through habitat
geometry, computing material-specific path lengths and converting these to areal densities.
Transmitted radiation responses are estimated using empirically-parameterized dose-depth
functions derived from published transport calculations.</p>

<p><b>IMPORTANT:</b> This is NOT a Monte Carlo particle transport solver. Results are
conceptual estimates suitable for comparative design analysis.</p>

<h3>2. Radiation Environments</h3>

<h4>2.1 Galactic Cosmic Rays (GCR)</h4>
<p>GCR flux is parameterized using a Badhwar-O'Neill-inspired model with solar modulation
parameter &phi; (MV). Higher &phi; corresponds to greater solar activity and reduced GCR flux.
The unshielded free-space GCR dose rate is estimated as:</p>
<p><code>D(&phi;) = A &times; exp(-&phi;/B) + C</code></p>
<p>where A=580, B=650, C=180, calibrated to yield ~660 mSv/yr at solar minimum (&phi;=400 MV)
and ~250 mSv/yr at solar maximum (&phi;=1200 MV).</p>
<p>Lunar surface dose is approximately 50% of free-space (2&pi; solid-angle shielding from
the lunar body below).</p>

<h4>2.2 Solar Particle Events (SPE)</h4>
<p>SPE fluence spectra are parameterized using Band-function fits:</p>
<p><code>dF/dE = J&sub;0 &times; E<sup>-&gamma;</sup> &times; exp(-E/E&sub;0)</code></p>
<p>Parameters are fit to historical events from literature (Tylka et al., 2006).</p>

<h4>2.3 Solar Wind</h4>
<p>Solar wind (~1 keV/nucleon protons at ~400 km/s) is treated as a surface interaction
only. It is completely stopped by any solid shielding material and is NOT included in
habitat interior biological dose calculations.</p>

<h3>3. Shielding Analysis</h3>
<p>The core shielding analysis uses directional ray casting:</p>
<ol>
<li>From each dosimetry target point, rays are cast in N approximately-uniform directions
(default 162, using Fibonacci spiral distribution).</li>
<li>Each ray is intersected with all geometry surfaces using the M&ouml;ller-Trumbore algorithm.</li>
<li>Path lengths through each material region are computed and converted to areal density:
<code>&rho;<sub>areal</sub> = &rho;<sub>material</sub> &times; path_length</code></li>
<li>Transmitted dose is estimated using exponential attenuation with material-specific
parameters:
<code>D(x) = D&sub;0 &times; exp(-x/&lambda;) &times; B(x)</code></li>
</ol>

<p>where x = areal density (g/cm&sup2;), &lambda; = effective attenuation length (~25 g/cm&sup2;
for GCR in regolith), and B(x) is an approximate neutron buildup factor.</p>

<h3>4. Quality Factors</h3>
<p>Dose equivalent (Sv) = Dose (Gy) &times; Q(LET), following ICRP 103 recommendations.
Quality factor Q decreases with shielding as heavy ions fragment, approximately:</p>
<p><code>Q(x) = 3.5 &times; exp(-x/200) + 2.0 &times; (1 - exp(-x/200))</code></p>

<h3>5. Assumptions and Limitations</h3>
<ul>
<li>1D transport approximation along each ray direction</li>
<li>Exponential attenuation model with empirically-calibrated parameters</li>
<li>No electromagnetic cascade simulation</li>
<li>No detailed nuclear fragmentation transport</li>
<li>Secondary neutron buildup is approximate</li>
<li>Applicable range: wall thicknesses 0-100 g/cm&sup2;</li>
<li>Expected agreement with full transport: within 20-30% for GCR dose equivalent</li>
</ul>

<h3>6. Units</h3>
<table border="1" cellpadding="4">
<tr><th>Quantity</th><th>Unit</th></tr>
<tr><td>Geometry</td><td>meters (m)</td></tr>
<tr><td>Areal density</td><td>g/cm&sup2;</td></tr>
<tr><td>Absorbed dose rate</td><td>mGy/yr or mGy/event</td></tr>
<tr><td>Dose equivalent rate</td><td>mSv/yr or mSv/event</td></tr>
<tr><td>Particle flux</td><td>particles/cm&sup2;/s or /event</td></tr>
<tr><td>Material density</td><td>g/cm&sup3;</td></tr>
</table>
"""

REFERENCES_TEXT = """
<h2>References</h2>

<ol>
<li><b>Badhwar, G.D. & O'Neill, P.M.</b> (1996). "Galactic cosmic radiation model and
its applications." <i>Advances in Space Research</i>, 17(2), 7-17.</li>

<li><b>O'Neill, P.M.</b> (2010). "Badhwar-O'Neill 2010 Galactic Cosmic Ray Flux Model."
<i>IEEE Trans. Nuclear Science</i>, 57(6), 3148-3153.</li>

<li><b>Tylka, A.J. et al.</b> (2006). "CREME96: A Revision of the Cosmic Ray Effects
on Micro-Electronics Code." <i>IEEE Trans. Nuclear Science</i>, 44(6), 2150-2160.</li>

<li><b>NCRP Report No. 132</b> (2000). "Radiation Protection Guidance for Activities
in Low-Earth Orbit."</li>

<li><b>NCRP Report No. 153</b> (2006). "Information Needed to Make Radiation Protection
Recommendations for Space Missions Beyond Low-Earth Orbit."</li>

<li><b>ICRP Publication 60</b> (1991). "1990 Recommendations of the International
Commission on Radiological Protection."</li>

<li><b>ICRP Publication 103</b> (2007). "The 2007 Recommendations of the International
Commission on Radiological Protection."</li>

<li><b>Cucinotta, F.A. et al.</b> (2006). "Space radiation cancer risk projections and
uncertainties." <i>Radiation Research</i>, 166(5), 811-823.</li>

<li><b>Wilson, J.W. et al.</b> (1995). "HZETRN: Description of a Free-Space Ion and
Nucleon Transport and Shielding Computer Program." NASA TP-3495.</li>

<li><b>Wilson, J.W. et al.</b> (1997). "Issues in space radiation protection."
<i>Health Physics</i>, 68(1), 50-58.</li>

<li><b>Singleterry, R.C. et al.</b> (2011). "OLTARIS: On-Line Tool for the Assessment
of Radiation in Space." <i>Acta Astronautica</i>, 68, 1086-1097.</li>

<li><b>Heiken, G.H., Vaniman, D.T. & French, B.M.</b> (1991). <i>Lunar Sourcebook:
A User's Guide to the Moon</i>. Cambridge University Press.</li>

<li><b>Miller, J. et al.</b> (2009). "Lunar soil as shielding against space radiation."
<i>Radiation Measurements</i>, 44(2), 163-167.</li>

<li><b>Kim, M.Y. et al.</b> (2009). "Recommendations for NASA's Space Radiation Risk
Assessment: Design Limits and Acceptable Risk Levels." NASA/TP-2009-214788.</li>

<li><b>Kiefer, W.S. et al.</b> (2012). "The density and porosity of lunar rocks."
<i>Geophysical Research Letters</i>, 39, L07201.</li>

<li><b>Sato, T. et al.</b> (2011). "Dose estimation for astronauts using dose conversion
coefficients." <i>Radiation Research</i>, 175, 235-245.</li>

<li><b>King, J.H.</b> (1974). "Solar Proton Fluences for 1977-1983 Space Missions."
<i>J. Spacecraft & Rockets</i>, 11(6), 401-408.</li>

<li><b>Mewaldt, R.A. et al.</b> (2005). "Solar-particle energy spectra during the large
events of October-November 2003." <i>Proc. 29th ICRC</i>.</li>

<li><b>Bruno, A. et al.</b> (2019). "Spectral Analysis of the September 2017 Solar
Energetic Particle Events." <i>Space Weather</i>, 17, 419-437.</li>

<li><b>Feldman, W.C. et al.</b> (2001). "Evidence for water ice near the lunar poles."
<i>J. Geophysical Research</i>, 106(E10), 23231-23252.</li>
</ol>
"""


class MethodsDialog(QDialog):
    """Dialog showing methods documentation and references."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("LunaRad-PEEK — Methods & References")
        self.setMinimumSize(700, 600)

        layout = QVBoxLayout(self)

        tabs = QTabWidget()

        # Methods tab
        methods_text = QTextEdit()
        methods_text.setReadOnly(True)
        methods_text.setHtml(METHODS_TEXT)
        tabs.addTab(methods_text, "Methods")

        # References tab
        refs_text = QTextEdit()
        refs_text.setReadOnly(True)
        refs_text.setHtml(REFERENCES_TEXT)
        tabs.addTab(refs_text, "References")

        layout.addWidget(tabs)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
