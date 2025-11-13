# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.7](https://codeberg.org/gillesvink/inpaint/compare/v0.1.6...v0.1.7) - 2025-11-13

### Other

- Add feature flag for libm or std
- Add support for no-std
- Switch to core crate for most calls, add extra tests for rectangular inpainting and allow optionally all calculations in libm

## [0.1.6](https://codeberg.org/gillesvink/inpaint/compare/v0.1.5...v0.1.6) - 2025-11-13

### Other

- Fix issue that could cause out of bounds for pixel gradients

## [0.1.5](https://codeberg.org/gillesvink/inpaint/compare/v0.1.4...v0.1.5) - 2025-11-12

### Other

- Add check for neighbor being on edge

## [0.1.4](https://codeberg.org/gillesvink/inpaint/compare/v0.1.3...v0.1.4) - 2025-10-31

### Other

- Expose error struct as public

## [0.1.3](https://codeberg.org/gillesvink/inpaint/compare/v0.1.2...v0.1.3) - 2025-10-27

### Other

- Simply image traits by using image-ndarray crate and switch to views for processing

## [0.1.2](https://codeberg.org/gillesvink/inpaint/compare/v0.1.1...v0.1.2) - 2025-10-26

### Other

- Wrap error in InpaintError
- Set default to no features enabled and improve structure of prelude
- Add docs to telea inpaint and fix dimension mismatch when using rectangular images

## [0.1.1](https://codeberg.org/gillesvink/inpaint/compare/v0.1.0...v0.1.1) - 2025-10-25

### Other

- Lower dependency requirements for Python to support Python 3.8, fix image in readme  and add badges

## [0.1.0](https://codeberg.org/gillesvink/inpaint/releases/tag/v0.1.0) - 2025-10-25

### Other

- Initial release
