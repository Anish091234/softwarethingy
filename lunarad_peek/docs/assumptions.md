# LunaRad: Assumptions and Limitations

## Tool Classification

LunaRad is a **conceptual design and early-stage research tool** for estimating
radiation shielding effectiveness of lunar habitat wall configurations.

**It is NOT:**
- A Monte Carlo particle transport solver (GEANT4, PHITS, MCNP)
- A reproduction of NASA OLTARIS
- A clinical dosimetry tool
- Suitable for final engineering design without higher-fidelity validation

**It IS:**
- OLTARIS-inspired in workflow
- Geometry-aware and areal-density-based
- Useful for conceptual comparison of shielding configurations
- Extensible to future high-fidelity transport coupling

## Key Simplifications

### 1. Transport Model
- **1D transport approximation** along each ray direction
- Exponential attenuation model with empirically-calibrated parameters
- No full Boltzmann transport equation solution
- No electromagnetic cascade simulation

### 2. Radiation Interactions
- Primary particle attenuation only (plus approximate secondary correction)
- No detailed nuclear fragmentation transport
- Neutron buildup estimated via empirical correction factor B(x)
- Target fragmentation products not individually tracked

### 3. Quality Factors
- Based on ICRP 60/103 recommendations
- Applied as averaged Q(LET) values, not particle-by-particle
- Q assumed to decrease with shielding due to ion fragmentation

### 4. Geometry
- Triangle mesh representation with Möller-Trumbore ray-triangle intersection
- Multi-layer walls treated as concentric shells
- No detailed internal structure (fixtures, equipment) unless modeled

### 5. Environment Models
- GCR: Badhwar-O'Neill-inspired parameterization (not full spectral model)
- SPE: Band-function fits to historical events
- Solar wind: surface interaction only, correctly excluded from interior dose
- No trapped radiation (relevant for LEO, not lunar surface)

## Applicability Range

- Wall thicknesses: 0–100 g/cm² (areal density)
- Energies: relevant to GCR (100 MeV–10 GeV) and SPE (10 MeV–1 GeV)
- Materials: low-to-medium Z (regolith, PEEK, composites, aluminum, water)

## Expected Accuracy

- GCR dose/dose equivalent: within **20–30%** of full transport codes for simple geometries
- SPE dose: within **factor of 2** for well-characterized events
- Directional trends and comparative rankings: **reliable** for design comparison
- Absolute values: **approximate** — use for relative comparison, not certification

## When Results May Diverge

- Very thick shields (>100 g/cm²): secondary particle equilibrium not fully captured
- High-Z materials: nuclear interaction details become important
- Complex geometry with significant scattering: 1D ray approximation breaks down
- Mixed radiation fields where particle-by-particle transport matters

## Scientific Guardrails

All outputs are labeled with confidence level:
- **Conceptual Estimate**: based on parameterized models
- **Literature-Derived**: calibrated against published transport results
- **Validated Against Transport Code**: compared to HZETRN/PHITS/GEANT4 (future)

Units are always tracked and displayed. Extrapolation warnings are provided
when operating outside the validated parameter range.
