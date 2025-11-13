# Release Notes for v1.4.0 (Target: November 22, 2025)

## Summary
This document lists all Pull Requests and Issues with potential to be merged for the upcoming v1.4.0 release.

---

## High Priority Pull Requests (Ready for Merge)

### PR #820 - Add Hugging Face Hub integration with Zarr format for dataset sharing
**Status:** Open (non-draft)  
**Author:** @Kkuntal990  
**Created:** 2025-11-05  
**Last Updated:** 2025-11-08  

**Summary:**
- Adds HuggingFace Hub integration to braindecode
- Enables easy sharing, versioning, and collaboration on EEG datasets
- Uses optimized Zarr format for dataset storage
- Supports three dataset types: WindowsDataset, EEGWindowsDataset, RawDataset
- **26 tests passing** ✅
- Live demo datasets available on HuggingFace Hub

**Merge Potential:** **HIGH** - Feature complete with passing tests and documentation

---

### PR #818 - Add SIENA and CHB_MIT datasets
**Status:** Open (non-draft)  
**Author:** @bruAristimunha (Maintainer)  
**Created:** 2025-11-04  
**Last Updated:** 2025-11-05  

**Summary:**
- Adds SIENA dataset support
- Adds CHB_MIT dataset support
- Updates relevant documentation

**Merge Potential:** **HIGH** - Authored by maintainer, important dataset additions

---

### PR #815 - LUNA model integration and utilities
**Status:** Open (non-draft)  
**Author:** @bruAristimunha (Maintainer)  
**Created:** 2025-11-04  
**Last Updated:** 2025-11-06  

**Summary:**
- Integrates LUNA model into framework
- Updates related utilities
- Improves channel location extraction functionality
- Various code fixes and enhancements

**Merge Potential:** **HIGH** - Model addition by maintainer

---

## Medium Priority Pull Requests

### PR #743 - Add SimpleConv model (DilatedConv from Meta Brain&AI Team)
**Status:** Open (non-draft)  
**Author:** @bruAristimunha (Maintainer)  
**Created:** 2025-04-29  
**Last Updated:** 2025-10-25  

**Summary:**
- Introduces SimpleConv model to enhance MEG model capabilities
- Code cleanup and improved naming conventions

**Merge Potential:** **MEDIUM** - Older PR, may need rebase and review

---

### PR #687 - Check if self._description is None before set_description
**Status:** Open (non-draft)  
**Author:** @Visvaria  
**Created:** 2025-01-02  
**Last Updated:** 2025-10-25  

**Summary:**
- Fixes bug where WindowsDataset created by _create_description has NoneType self._description
- Prevents errors when using set_description method

**Merge Potential:** **MEDIUM** - Bug fix, needs review and testing

---

### PR #614 - Add "recording" to keys in RecordingSampler; docs fix
**Status:** Open (non-draft)  
**Author:** @OverLordGoldDragon  
**Created:** 2024-05-17  
**Last Updated:** 2025-10-25  

**Summary:**
- Fixes RecordingSampler treating recordings as contiguous
- Documentation improvements
- Typo fixes

**Merge Potential:** **MEDIUM** - Important fix but older PR

---

### PR #612 - Fix tmax in SleepPhysionet
**Status:** Open (non-draft)  
**Author:** @OverLordGoldDragon  
**Created:** 2024-05-17  
**Last Updated:** 2025-10-25  

**Summary:**
- Fixes tmax calculation to be inclusive of time starts
- Removes redundant int conversion

**Merge Potential:** **MEDIUM** - Bug fix for dataset

---

### PR #580 - Create transfer learning tutorial
**Status:** Open (non-draft)  
**Author:** @javadbayazi  
**Created:** 2024-03-26  
**Last Updated:** 2025-10-25  

**Summary:**
- Adds transfer learning tutorial/example

**Merge Potential:** **LOW-MEDIUM** - Documentation improvement, older PR

---

## Critical Open Issues to Address

### Issue #828 - Epilepsy Benchmarks + Braindecode
**Priority:** HIGH  
**Created:** 2025-11-07  
**Author:** @bruAristimunha (Maintainer)  

**Summary:**
- Collaboration with EPFL team to port winning models from competition
- Student assignment for model adaptation needed
- Has associated Google Sheets for tracking

**Action Required:** Coordinate with students, track progress

---

### Issue #827 - Completely deprecate the Windows Dataset
**Priority:** HIGH  
**Created:** 2025-11-06  
**Author:** @bruAristimunha (Maintainer)  

**Summary:**
- Need to complete deprecation cycle for WindowsDataset
- Use only EEGWindowsDataset going forward
- Some functions still lacking equivalence

**Action Required:** Assess impact, plan deprecation

---

### Issue #823 - Temporal generalization Tutorial
**Priority:** MEDIUM  
**Created:** 2025-11-05  
**Author:** @bruAristimunha (Maintainer)  

**Summary:**
- Create tutorial for generalized time slice windows
- Demonstrate temporal decoding with braindecode

**Action Required:** Could be included in v1.4.0 if tutorial created quickly

---

### Issue #816 - Example at chance level
**Priority:** MEDIUM  
**Created:** 2025-11-04  
**Author:** @arnodelorme  

**Summary:**
- Example returns results at chance level (25%)
- Need to investigate and fix

**Action Required:** Investigate root cause

---

### Issue #809 - Improving the existent eegprep parameters
**Priority:** MEDIUM  
**Created:** 2025-11-01  
**Author:** @bruAristimunha (Maintainer)  

**Summary:**
- Classes shouldn't accept base class arguments not meant for users
- Need to add trivial constructors back

**Action Required:** Code cleanup

---

### Issue #807 - Dataset mixin
**Priority:** LOW-MEDIUM  
**Created:** 2025-11-01  
**Author:** @bruAristimunha (Maintainer)  

**Summary:**
- Need more details

**Action Required:** Get requirements clarification

---

### Issue #804 - Feature needed: distinguish RAW vs EPOCH format in Windows dataset
**Priority:** MEDIUM  
**Created:** 2025-10-31  
**Author:** @bruAristimunha (Maintainer)  

**Summary:**
- Need indication whether raw object is used as epoch
- Important for proper dataset handling

**Action Required:** Design and implement solution

---

### Issue #803 - Make models SincShallowNet, DeepSleepNet/AttnSleep compatible with Hugging Face
**Priority:** MEDIUM  
**Created:** 2025-10-31  
**Author:** @bruAristimunha (Maintainer)  

**Summary:**
- Improve HuggingFace compatibility for specific models

**Action Required:** Model updates needed

---

### Issue #800 - Save all model arguments when pushing to Hugging Face
**Priority:** MEDIUM  
**Created:** 2025-10-29  
**Author:** @PierreGtch  

**Summary:**
- For optimal reproducibility, save all model arguments
- Not just signal params

**Action Required:** Update HuggingFace integration

---

### Issue #798 - Improve the model card
**Priority:** LOW  
**Created:** 2025-10-29  
**Author:** @bruAristimunha (Maintainer)  

**Summary:**
- Future enhancement for model cards

**Action Required:** Documentation improvement

---

### Issue #793 - Improve our mixin to give Hugging Face hub capabilities
**Priority:** HIGH  
**Created:** 2025-10-28  
**Author:** @bruAristimunha (Maintainer)  

**Summary:**
- Enhance EEGMixin with HuggingFace capabilities
- Related to PR #531
- 9 comments showing active discussion

**Action Required:** May be addressed by PR #820

---

## Recommendations for v1.4.0 Release

### Must Have (Block Release)
1. **PR #820** - Hugging Face Hub integration (if tests pass in CI)
2. **PR #818** - SIENA and CHB_MIT datasets
3. **PR #815** - LUNA model integration

### Should Have (Include if Ready)
1. **PR #743** - SimpleConv model (needs rebase/review)
2. **PR #687** - Fix set_description bug
3. **PR #614** - RecordingSampler fix
4. **PR #612** - SleepPhysionet tmax fix

### Nice to Have (Future Release)
1. **PR #580** - Transfer learning tutorial
2. Various documentation improvements

### Critical Issues to Track
1. **Issue #828** - Epilepsy Benchmarks coordination
2. **Issue #827** - WindowsDataset deprecation plan
3. **Issue #793** - HuggingFace mixin improvements

---

## Version Update Plan

Current version: **1.3.0**  
Target version: **1.4.0**  
Release date: **November 22, 2025**

### Changes to Make:
1. Update `braindecode/version.py` from `1.3.0` to `1.4.0`
2. Create/update changelog
3. Update documentation version references if needed

---

## Timeline

- **Now - Nov 15:** Review and merge high-priority PRs
- **Nov 15-19:** Testing and bug fixes
- **Nov 19-21:** Final review and documentation
- **Nov 22:** Release v1.4.0

---

## Notes
- Total open PRs: 9
- Total open issues: 72
- PRs with high merge potential: 3-6
- Critical issues to track: 4-5

This release focuses on:
- Dataset sharing capabilities (HuggingFace Hub)
- New datasets (SIENA, CHB_MIT)
- New models (LUNA, potentially SimpleConv)
- Bug fixes and improvements
