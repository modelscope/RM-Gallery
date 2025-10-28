# Changelog

All notable changes to RM-Gallery will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Documentation system overhaul with improved structure
- Interactive Jupyter Notebook examples (quickstart, custom-rm, evaluation)
- Comprehensive FAQ documentation
- End-to-end tutorial for building reward models from scratch
- Tutorial home page with learning paths
- README files in key directories (rm_gallery/core/reward/, rm_gallery/gallery/rm/, examples/)
- Search optimization with metadata (keywords, descriptions, tags)
- Sitemap for better navigation

### Changed
- Simplified navigation structure in mkdocs.yml (from 22 to 15 items)
- Reorganized documentation hierarchy for better user experience
- Enhanced README.md with clearer quick start section
- Improved mkdocs.yml with better search configuration and SEO

### Fixed
- Fixed Examples navigation path in mkdocs.yml
- Removed redundant "Evaluation Benchmarks" navigation category

## [0.1.4] - 2025-01-XX

### Added
- New reward model evaluation benchmarks
- Enhanced rubric generation capabilities
- Support for more LLM backends
- Improved batch evaluation performance
- Additional pre-built reward models for various scenarios

### Changed
- Updated documentation structure
- Improved API documentation with mkdocstrings
- Enhanced error messages and logging

### Fixed
- Bug fixes in reward model evaluation
- Performance improvements in batch processing
- Memory optimization for large-scale evaluations

## [0.1.3] - 2024-12-XX

### Added
- Initial support for custom reward models
- Basic evaluation framework
- Integration with VERL training framework
- Command-line interface for evaluation

### Changed
- Refactored core reward model architecture
- Improved data pipeline flexibility
- Updated dependencies to latest versions

### Fixed
- Issues with data loading from various sources
- Compatibility issues with different Python versions

## [0.1.2] - 2024-11-XX

### Added
- Support for pairwise and pointwise evaluation
- Basic rubric-based evaluation
- Data annotation tools
- Example scripts for common use cases

### Changed
- Improved reward model registry system
- Enhanced documentation with more examples
- Better error handling throughout the codebase

### Fixed
- Critical bug in reward calculation
- Issues with multi-GPU training
- Data preprocessing edge cases

## [0.1.1] - 2024-10-XX

### Added
- Initial documentation site
- Basic tutorial structure
- Support for HuggingFace datasets
- Example reward models for math and code

### Changed
- Restructured project layout
- Improved package organization
- Updated installation instructions

### Fixed
- Import errors in certain environments
- Configuration file parsing issues
- Minor bugs in reward model evaluation

## [0.1.0] - 2024-09-XX

### Added
- Initial release of RM-Gallery
- Core reward model architecture (BaseReward, BasePointWiseReward, BaseListWiseReward)
- Support for rule-based and LLM-based reward models
- Basic training pipeline integration
- Pre-built reward models for common scenarios (math, code, alignment)
- Data loading and processing utilities
- Registry system for reward models
- Basic evaluation tools
- Integration with OpenAI API
- Command-line tools (oaieval equivalent)
- Initial documentation and examples
- Support for Python 3.10+

### Technical Details
- Framework: Built on top of PyTorch and Transformers
- Architecture: Modular design with pluggable reward models
- Data Format: Standardized schema for consistency
- Evaluation: Support for multiple evaluation paradigms

---

## Release Notes

### Version Numbering

RM-Gallery follows [Semantic Versioning](https://semver.org/):
- **MAJOR** version: Incompatible API changes
- **MINOR** version: Backwards-compatible functionality additions
- **PATCH** version: Backwards-compatible bug fixes

### How to Upgrade

#### From 0.1.3 to 0.1.4
```bash
pip install --upgrade rm-gallery
```

**Breaking Changes**: None

**Deprecations**: None

**Migration Guide**: No migration needed, all APIs are backwards compatible.

#### From 0.1.2 to 0.1.3
```bash
pip install --upgrade rm-gallery
```

**Breaking Changes**:
- Reward model registry API slightly changed
- Data schema updated (automatic migration)

**Migration Guide**: See [Migration Guide v0.1.3](docs/migration/v0.1.3.md)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute to RM-Gallery.

When adding changes to this file:
1. Add new entries under `[Unreleased]`
2. Follow the existing format
3. Use categories: Added, Changed, Deprecated, Removed, Fixed, Security
4. Be specific and user-focused in descriptions
5. Link to relevant issues/PRs when applicable

## Questions?

- 📚 [Documentation](https://modelscope.github.io/RM-Gallery/)
- 💬 [GitHub Discussions](https://github.com/modelscope/RM-Gallery/discussions)
- 🐛 [Report Issues](https://github.com/modelscope/RM-Gallery/issues)

---

**Note**: Dates marked with XX are placeholders. Replace with actual release dates.

[Unreleased]: https://github.com/modelscope/RM-Gallery/compare/v0.1.4...HEAD
[0.1.4]: https://github.com/modelscope/RM-Gallery/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/modelscope/RM-Gallery/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/modelscope/RM-Gallery/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/modelscope/RM-Gallery/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/modelscope/RM-Gallery/releases/tag/v0.1.0

