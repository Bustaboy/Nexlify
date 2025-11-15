# Codecov Setup Guide

This guide explains how to set up Codecov integration for the Nexlify repository.

## Quick Setup

### 1. Sign up for Codecov

1. Go to [codecov.io](https://codecov.io/)
2. Sign in with your GitHub account
3. Authorize Codecov to access your repositories

### 2. Add Nexlify Repository

1. Once logged in, click "Add a repository"
2. Find and select `Bustaboy/Nexlify`
3. Codecov will generate a repository token

### 3. Add CODECOV_TOKEN to GitHub Secrets

1. Copy the repository token from Codecov
2. Go to your GitHub repository: `https://github.com/Bustaboy/Nexlify`
3. Navigate to **Settings** → **Secrets and variables** → **Actions**
4. Click **New repository secret**
5. Name: `CODECOV_TOKEN`
6. Value: Paste the token from Codecov
7. Click **Add secret**

### 4. Verify Setup

After the next workflow run:
1. Coverage reports will automatically upload to Codecov
2. PR comments will show coverage changes
3. Status checks will appear on pull requests
4. The coverage badge in README.md will update

## Configuration

The repository includes a `codecov.yml` configuration file with:

- **Coverage targets**: 75% overall, 85% for core modules
- **Patch coverage**: 70% for new code
- **PR comments**: Automatic coverage diff comments
- **Flags**: Separate tracking for quick-tests and full-tests
- **Ignore patterns**: Excludes tests/, scripts/, examples/

## Coverage Targets by Module

| Module | Target | Description |
|--------|--------|-------------|
| Core (core/, risk/, strategies/) | 85% | Critical trading logic |
| Financial (financial/, analytics/) | 75% | Financial operations |
| Utilities (utils/) | 90% | Helper functions |
| Overall Project | 75% | Entire codebase |

## Viewing Coverage Reports

### On Codecov.io
- Dashboard: `https://codecov.io/gh/Bustaboy/Nexlify`
- Detailed file-by-file coverage
- Coverage trends over time
- Sunburst charts and graphs

### Locally
After running tests with coverage:
```bash
pytest --cov=nexlify --cov-report=html
open htmlcov/index.html  # View HTML report
```

### In Pull Requests
- Codecov bot comments on PRs with coverage changes
- Status checks show if coverage meets thresholds
- Click "Details" on status checks to see full report

## Workflow Integration

Coverage is collected in two jobs:

1. **quick-tests** (Python 3.12 only)
   - Fast unit tests
   - Flag: `quick-tests,python-3.12`

2. **full-tests** (Python 3.11 & 3.12)
   - Complete test suite
   - Flags: `full-tests,python-3.11` and `full-tests,python-3.12`

Both jobs upload to Codecov, providing comprehensive coverage tracking.

## Troubleshooting

### Token not working
- Verify the token is correctly copied
- Check it's named exactly `CODECOV_TOKEN` (case-sensitive)
- Ensure it's a repository secret, not an environment secret

### Coverage not uploading
- Check GitHub Actions logs for upload errors
- Verify `codecov/codecov-action@v4` is used
- Ensure coverage.xml is generated before upload

### Badge not updating
- Wait 5-10 minutes after workflow completion
- Clear browser cache
- Check if the repository name in badge URL matches exactly

## Resources

- [Codecov Documentation](https://docs.codecov.com/)
- [codecov.yml Reference](https://docs.codecov.com/docs/codecov-yaml)
- [GitHub Actions Integration](https://docs.codecov.com/docs/github-actions-integration)
