"""The package version, the development versions ``release.yml`` publishes,
and the version switcher that the monthly release writes."""

import json
from pathlib import Path
from runpy import run_path

from packaging.version import Version

import braindecode


def test_release_documentation_versions(tmp_path, monkeypatch, capsys):
    script = Path(__file__).resolve().parents[2] / "scripts" / "monthly_release.py"
    release = run_path(str(script))["main"]
    monkeypatch.chdir(tmp_path)

    def run(*args):
        assert release(list(args)) == 0
        return capsys.readouterr().out

    version = tmp_path / "braindecode" / "version.py"
    version.parent.mkdir()
    version.write_text('__version__ = "2.1.0dev0"\n')
    switcher = tmp_path / "docs" / "_static" / "versions.json"
    switcher.parent.mkdir(parents=True)
    (tmp_path / "docs" / "whats_new.rst").write_text(
        ".. _current:\n\nCurrent 2.1.0 (GitHub)\n======================\n"
    )
    url = "https://braindecode.org/"
    initial = [
        {"name": "2.1 (dev)", "version": "dev", "url": url + "dev/"},
        {"name": "2.0 (stable)", "version": "2.0", "url": url + "stable/"},
        {"name": "1.5", "version": "1.5", "url": url + "1.5/"},
    ]
    switcher.write_text(json.dumps(initial))
    for bump, dev, stable, archived in [
        ("patch", "2.1", "2.1", ["2.0", "1.5"]),
        ("major", "3.0", "2.1", ["2.0", "1.5"]),
        ("minor", "3.1", "3.0", ["2.1", "2.0", "1.5"]),
    ]:
        plan = dict(
            line.split("=", 1) for line in run("compute", "--bump", bump).splitlines()
        )
        run("finalize", "--next-dev-version", plan["next_dev_version"])
        run("next-dev", "--bump", bump)
        assert json.loads(switcher.read_text()) == [
            {"name": f"{dev} (dev)", "version": "dev", "url": url + "dev/"},
            {"name": f"{stable} (stable)", "version": stable, "url": url + "stable/"},
            *({"name": v, "version": v, "url": url + v + "/"} for v in archived),
        ]


def test_version_sorts_after_1_8_1():
    # Downstream code gates workarounds on the version: NeuralBench keeps a LoRA
    # pre-hook for LaBraM's qkv "until braindecode > 1.8.1" (braindecode#1194).
    # With master still reporting 1.8.1 the gate cannot tell the two apart, and
    # the hook adds the LoRA delta a second time on master. Released versions
    # (the tag commit of a release, and the sdist built from it) must pass too.
    assert Version(braindecode.__version__) > Version("1.8.1")


def test_dev_builds_follow_commit_order(tmp_path):
    # ``pip install --pre`` takes the highest version, so a build must outrank the
    # hash-numbered builds published earlier (below 2**32) and the builds of older
    # master commits, yet stay below the stable release.
    script = Path(__file__).resolve().parents[2] / "scripts" / "monthly_release.py"
    release = run_path(str(script))
    version_file = tmp_path / "version.py"
    version_file.write_text('__version__ = "2.1.0dev0"\n')

    def build(commits):
        args = ["dev-build", str(commits), "--init-file", str(version_file)]
        assert release["main"](args) == 0
        return Version(release["read_version"](version_file))

    older, newer = build(1367), build(1368)
    assert Version(f"2.1.0.dev{2**32 - 1}") < older < newer < Version("2.1.0")
