# Integration tests

This directory holds dplPy's **integration tests**: end-to-end runs of the
public API on real datasets, exercising several modules together the way a user
would (read a file, detrend, build a chronology, crossdate, write it back out).
Every file is named `test_integ_<area>.py` and is collected by `pytest` along
with the unit tests.

## How these differ from the unit tests

| | `tests/unit/` | `tests/integs/` |
|---|---|---|
| Scope | one module (`test_<module>.py` mirrors `src/dplpy/<module>.py`) | several public functions chained together |
| Inputs | small or synthetic frames, often hand-built | the real datasets in `tests/data/` (ca533, co021, ca667, ...) |
| What is checked | exact numbers, edge cases, error messages, dplR/ARSTAN reference values | that the whole pipeline runs and returns results of the right shape/content, round-trips preserve data, every option combination completes |
| Speed | fast | slower (full collections, every detrend fit, plotting) |

A regression that pins a specific number belongs in a unit test; a test that
"reads this file, runs these three steps and makes sure nothing falls over"
belongs here.

## What is covered

- `test_integ_readers_and_writers.py` — read/write round-trips for `.csv` and
  Tucson `.rwl` (with and without headers, gap sentinels, long and irregular
  files), and duplicate-ID reporting on read.
- `test_integ_detrend.py` — every `detrend` curve with ratio and difference
  indices on a full collection.
- `test_integ_autoreg.py` — `ar_func` / `autoreg` on real series.
- `test_integ_chron.py` — `chron` with and without prewhitening and biweight.
- `test_integ_variance_stab.py` — `chron_stabilized` methods and options.
- `test_integ_xdate.py` — `xdate` across bin floors, window lengths,
  correlation types and prewhitening settings.
- `test_integ_statistics.py` — `stats`, `summary`, `report` on a collection.
- `test_integ_plots.py` — every `plot` type on a `.csv` and a headered `.rwl`
  dataset, headless.

## Running

From the repository root (`pytest` is configured in `pyproject.toml`, so no
`PYTHONPATH` is needed):

```
pytest tests/integs -rA          # all integration tests, with a per-test summary
pytest tests/integs/test_integ_xdate.py -q
pytest                           # unit + integration tests together
```

The integration layer runs in CI on every push for each supported Python
version (`.github/workflows/run_tests.yml`), after the unit tests.

## Writing one

- Name the file `test_integ_<area>.py`.
- Use the shared fixtures from `tests/conftest.py` (`ca533_rwl`, `ca533_rwi`,
  `co021_rwl`, and the `DATA` path for any other file under `tests/data/`), so
  the test does not depend on the working directory and does not re-read files.
- Pass `plot=False` / `show=False` (the suite runs on the `Agg` backend) and
  write any output to `tmp_path`.
- Describe new datasets in `tests/data/README.md`.
