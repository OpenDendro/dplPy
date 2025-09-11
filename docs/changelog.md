# Changelog

For the complete changelog and release history, please see:

**[CHANGELOG.md](../CHANGELOG.md)**

This file contains detailed information about:

- Version releases and dates
- New features and enhancements  
- Bug fixes and improvements
- Breaking changes and migration notes
- Contributors and acknowledgments

## Recent Releases

### [0.1.6] - 2025-09-10

**Infrastructure Improvements:**
- Fixed broken URLs in dplpy.py from `https://opendendro.org/python/` to `https://opendendro.org/dplpy/`
- Standardized version numbers across all configuration files (0.1.6)
- Fixed inconsistent import usage in setup.py
- Resolved Python requirements inconsistency in setup.cfg
- Updated documentation URLs to working endpoints

### [0.1.5] - 2023-XX-XX

**Major Updates:**
- Updates to dplPy since January (mostly related to crossdating) (#37)
- Added seg and spag plots functionality
- Enhanced chron functionality  
- Improved robustness and input validation for detrend and autoreg methods
- Added support for repeated series names in .rwl reader
- Added support for negative year numbers in .rwl reader

**Documentation:**
- Updated README and usage instructions
- Improved documentation clarity with prompts
- Fixed typos and links in documentation

### [0.1.2] - 2022-XX-XX

**Core Features:**
- Implementation of residual and difference in detrend functionality
- Working with csaps smoothing spline implementation
- Summary statistics implementation including skew calculations
- Enhanced rwl.report() functionality similar to dplR
- Support for CSV and RWL file formats in readers
- Autoregressive modeling improvements

## Contributing to Changelog

When contributing to dplPy, please update the changelog by:

1. Adding entries to the [Unreleased] section of CHANGELOG.md
2. Following the [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format
3. Using these categories:
   - **Added** for new features
   - **Changed** for changes in existing functionality
   - **Deprecated** for soon-to-be removed features
   - **Removed** for now removed features
   - **Fixed** for any bug fixes
   - **Security** for vulnerability fixes

For detailed contribution guidelines, see [Contributing](contributing.md).