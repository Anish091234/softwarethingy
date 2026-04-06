from __future__ import annotations

import pytest

import run


def test_find_supported_interpreter_returns_first_compatible(monkeypatch):
    monkeypatch.setattr(run.sys, "executable", "/current/python")
    monkeypatch.setattr(
        run,
        "_candidate_interpreters",
        lambda: ["/current/python", "/missing/python", "/good/python", "/later/python"],
    )
    monkeypatch.setattr(
        run,
        "_interpreter_supports",
        lambda candidate: candidate == "/good/python",
    )

    assert run._find_supported_interpreter() == "/good/python"


def test_ensure_supported_runtime_reexecs_with_compatible_interpreter(monkeypatch):
    monkeypatch.setattr(run, "_current_runtime_supported", lambda: False)
    monkeypatch.setattr(run, "_find_supported_interpreter", lambda: "/good/python")
    monkeypatch.delenv(run.REEXEC_GUARD_ENV, raising=False)

    captured: dict[str, object] = {}

    def fake_execv(path, argv):
        captured["path"] = path
        captured["argv"] = argv
        raise SystemExit(0)

    monkeypatch.setattr(run.os, "execv", fake_execv)

    with pytest.raises(SystemExit):
        run._ensure_supported_runtime()

    assert captured == {
        "path": "/good/python",
        "argv": ["/good/python", *run.sys.argv],
    }


def test_ensure_supported_runtime_prints_help_when_no_interpreter_found(
    monkeypatch, capsys
):
    monkeypatch.setattr(run, "_current_runtime_supported", lambda: False)
    monkeypatch.setattr(run, "_find_supported_interpreter", lambda: None)
    monkeypatch.setattr(run, "_missing_modules", lambda: ["PySide6"])
    monkeypatch.delenv(run.REEXEC_GUARD_ENV, raising=False)

    with pytest.raises(SystemExit):
        run._ensure_supported_runtime()

    output = capsys.readouterr().out
    assert "Missing or incompatible runtime" in output
    assert "PySide6" in output
    assert "python run.py" in output
