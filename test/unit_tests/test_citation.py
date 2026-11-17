"""Every copy of the Braindecode citation lists the authors of CITATION.cff."""

import re
from pathlib import Path

ROOT = Path(__file__).parents[2]
COPIES = ["README.rst", "docs/cite.rst", "braindecode/models/model.html.jinja"]


def _authors():
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    block = text.split("\nauthors:\n", 1)[1].split("\ntype:", 1)[0]
    for entry in re.split(r"\n  - ", "\n" + block)[1:]:
        given = re.search(r"given-names: (.+)", entry).group(1).strip('"\x27')
        family = re.search(r"family-names: (.+)", entry)
        yield given, family.group(1).strip('"\x27') if family else None


def test_bibtex_and_apa_authors_match_citation_cff():
    authors = list(_authors())
    bibtex = [f"{family}, {given}" if family else given for given, family in authors]
    apa = [
        f"{family}, " + " ".join(f"{name[0]}." for name in given.split())
        if family
        else given
        for given, family in authors
    ]
    apa = ", ".join(apa[:-1]) + ", & " + apa[-1]
    for path in COPIES:
        text = (ROOT / path).read_text(encoding="utf-8")
        block = re.search(
            r"@misc\{braindecode,\s*author = \{(.*?)\},\s*title", text, re.S
        )
        found = [
            a.replace("{", "").replace("}", "")
            for a in re.split(r"\s+and\s+", block.group(1).strip())
        ]
        assert found == bibtex, path
        if "APA" in text:
            assert apa in " ".join(text.split()), path
