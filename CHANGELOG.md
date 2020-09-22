# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).


## [Unreleased]

### Fixed
- amplitude gradients are correctly computed for layers with multiple filters
  (before, they were accidentally summed over all previous filters in the layer) [@robintibor]
- get_output_shape and compute_amplitude_gradients assume 3d, not 4d inputs [@robintibor]