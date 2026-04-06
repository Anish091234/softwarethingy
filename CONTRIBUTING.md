# Contributing to LunaRad

Thanks for your interest in improving LunaRad.

## How To Contribute

- Open an issue for bugs, usability problems, or feature ideas
- Submit a pull request for focused improvements
- Keep changes small and well-described when possible
- Include screenshots or figures when UI or visualization behavior changes

## Local Development

```bash
python3 -m pip install -e .[dev]
```

Useful checks:

```bash
python3 -m pytest
python3 -m ruff check .
python3 -m mypy lunarad_peek
```

## Scope Notes

LunaRad is currently a conceptual and research-oriented tool. Please keep
scientific assumptions explicit in code, docs, and visual outputs so results are
easy to interpret and review.

## License

By contributing, you agree that your contributions will be licensed under the
MIT License used by this repository.
