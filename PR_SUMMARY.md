# PR #832: Draft PR for Version Increase to v1.4.0

## Overview
This PR prepares the braindecode repository for the v1.4.0 release scheduled for **November 22, 2025** (in 2 weeks).

## Changes Made

### 1. Version Update
- **File**: `braindecode/version.py`
- **Change**: Updated version from `1.3.0` to `1.4.0`

### 2. Documentation Updates
- **File**: `docs/whats_new.rst`
- **Change**: Added new section for v1.4.0 development, moved v1.3 to released versions section

### 3. Release Planning Document
- **File**: `RELEASE_NOTES_v1.4.0.md` (new)
- **Content**: Comprehensive analysis of:
  - All open Pull Requests with merge readiness assessment
  - All open Issues with priority ratings
  - Recommended timeline for release
  - Categorized PR/Issue lists by priority

## Key Findings

### Ready-to-Merge PRs (HIGH PRIORITY)
These PRs appear ready and should be prioritized for v1.4.0:

1. **PR #820** - HuggingFace Hub Integration
   - Status: 26 tests passing
   - Impact: Major feature addition
   - Author: @Kkuntal990
   
2. **PR #818** - SIENA and CHB_MIT Datasets
   - Status: Ready
   - Impact: New datasets
   - Author: @bruAristimunha (maintainer)
   
3. **PR #815** - LUNA Model Integration
   - Status: Ready
   - Impact: New model
   - Author: @bruAristimunha (maintainer)

### Medium Priority PRs
These may need additional review but could be included:
- PR #743: SimpleConv model (older, may need rebase)
- PR #687: set_description bug fix
- PR #614: RecordingSampler fix
- PR #612: SleepPhysionet tmax fix

### Critical Issues to Track
- Issue #828: Epilepsy Benchmarks coordination
- Issue #827: WindowsDataset deprecation planning
- Issue #793: HuggingFace mixin improvements

## Recommended Actions

### Immediate (Now - Nov 15)
1. Review and merge PR #820 (HuggingFace Hub) if CI passes
2. Merge PR #818 (datasets) and PR #815 (LUNA model)
3. Review medium-priority PRs for inclusion

### Mid-term (Nov 15-19)
1. Run comprehensive testing
2. Address any bugs found
3. Update changelog/what's new with actual changes

### Final (Nov 19-21)
1. Final documentation review
2. Verify all tests pass
3. Prepare release notes

### Release (Nov 22)
1. Create release tag v1.4.0
2. Publish to PyPI
3. Update documentation

## Files to Review

Please review the following files in this PR:
- `RELEASE_NOTES_v1.4.0.md` - Complete analysis
- `braindecode/version.py` - Version bump
- `docs/whats_new.rst` - Documentation structure

## Next Steps

This is a **DRAFT PR** to facilitate discussion. Please:
1. Review the priority assessment in `RELEASE_NOTES_v1.4.0.md`
2. Confirm which PRs should be included
3. Update the what's new section with actual changes as PRs are merged
4. Coordinate with PR authors for any needed updates

## Questions for Maintainers

1. Is the November 22 release date still realistic?
2. Are there any other PRs/issues that should be prioritized?
3. Should we include all three high-priority PRs or focus on a subset?
4. Are there any breaking changes we need to document?

---
**Created by:** Copilot Coding Agent  
**Date:** November 9, 2025  
**For Release:** v1.4.0 (scheduled November 22, 2025)
