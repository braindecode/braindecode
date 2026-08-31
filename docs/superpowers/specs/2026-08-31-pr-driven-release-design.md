# PR-driven stable releases

## Goal

Make each stable release reviewable as a pull request before any tag, PyPI
upload, or GitHub Release is created.

## Flow

1. The scheduled or manually dispatched monthly workflow computes the release
   from the current development version, finalizes the version and changelog on
   a `release/vX.Y.Z` branch, builds the distribution, runs `twine check`, and
   opens a release PR.  It never publishes.
2. Merging a release PR with commit subject `release: vX.Y.Z` triggers the
   stable-publish path on `master`.  That path verifies the tag is absent,
   builds and validates the exact merged source, pushes the annotated tag
   before uploading to PyPI, then creates the GitHub Release.
3. After a successful stable publish, that same workflow opens the existing
   next-development PR.  It runs `next-dev` only because the merged release PR
   already dated the changelog.
4. All other pushes to `master` retain the existing `.devN` PyPI publishing
   path.

## Scope

Only `.github/workflows/monthly-release.yml` and
`.github/workflows/release.yml` change.  `scripts/monthly_release.py` already
provides the required `compute`, `finalize`, and `next-dev` operations.

## Safety properties

- The release tag is pushed before the irreversible PyPI upload.
- A release PR merge is the only stable-publish trigger.
- The scheduled workflow has no publishing permissions or publish step.
- The existing `pypi-publish` concurrency group continues to serialize dev and
  stable publication.

## Verification

- Parse both workflow YAML files.
- Run the monthly workflow with `dry_run=true` to verify finalization, build,
  and strict package validation without creating a branch or PR.
- Review the generated release PR, merge it, and confirm the stable-publish
  workflow creates the tag, PyPI release, GitHub Release, and next-dev PR.
