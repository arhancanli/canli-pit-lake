# Contributing to canli-pit-lake

canli-pit-lake is the point-in-time data layer of [ALPHAC](https://github.com/arhancanli/alphac), published on its own.
Issues, corrections, reproducibility reports, documentation and tests are welcome.

## How changes land

The code under `src/` is extracted from ALPHAC and kept byte-identical to it;
`tools/check_parity.py` proves that. A change to library code therefore lands in
[ALPHAC](https://github.com/arhancanli/alphac) first and is re-extracted here. Open the issue
or pull request here anyway: it will be carried upstream with credit, and the parity check
keeps the two from drifting.

Documentation, examples, benchmarks and tests outside `src/` can be changed here directly.

## Evidence boundary

- Report a look-ahead, survivorship or cost defect with the smallest input that reproduces it.
- Do not commit licensed market data, credentials or account identifiers.
- Do not present simulated results as live or investable performance.

## Local checks

```sh
uv venv --python 3.12 && uv pip install -e ".[dev]"
uv run pytest
uv run python tools/check_parity.py
```

Pull requests should be narrow, include a test, and say which behaviour they change.
