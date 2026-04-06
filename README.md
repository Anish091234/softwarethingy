# LunaRad

LunaRad is an open-source Python application for conceptual lunar habitat radiation
shielding analysis and visualization. It lets you build habitat geometry, assign
layered materials, define environmental conditions, run shielding analyses, and
generate figures for reports or project presentations.

## Open Source Status

This project is released as open-source software under the MIT License. You are
free to use, modify, share, and adapt it with attribution under the terms in
[LICENSE](LICENSE).

## Current Capabilities

- Build conceptual habitat geometries such as shell domes
- Define layered shielding materials and cover thicknesses
- Place astronaut targets inside the habitat
- Configure lunar radiation environment assumptions
- Run comparative shielding analyses
- Export visualization figures for project use

## Running LunaRad

On macOS, the easiest entrypoint is:

```bash
./Launch\ LunaRad.command
```

From a terminal, you can also start it with:

```bash
python3 run.py
```

If you need startup diagnostics:

```bash
./Diagnose\ LunaRad.command
```

## Basic Workflow

1. Build the habitat in the `Geometry` tab.
2. Add one or more astronaut targets.
3. Generate the geometry preview.
4. Configure the radiation environment.
5. Run the analysis.
6. Export figures from the `Visualization` tab.

## Installation

LunaRad targets Python 3.10+.

```bash
python3 -m pip install -e .
```

## Contributing

Contributions, bug reports, and improvement ideas are welcome. See
[CONTRIBUTING.md](CONTRIBUTING.md) for a lightweight starting point.

## Repository

- Homepage: <https://github.com/Anish091234/softwarethingy>
- Issues: <https://github.com/Anish091234/softwarethingy/issues>
