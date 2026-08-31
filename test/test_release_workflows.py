from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_stable_release_is_pr_driven():
    monthly = (ROOT / ".github/workflows/monthly-release.yml").read_text()
    publish = (ROOT / ".github/workflows/release.yml").read_text()

    assert "gh pr create" in monthly
    assert "gh-action-pypi-publish" not in monthly
    assert "is_devrelease" in publish
    assert "Open next-development pull request" in publish
