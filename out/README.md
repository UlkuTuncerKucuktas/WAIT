# out/

Campaign output.

| | contents |
|---|---|
| `probes/` | measurements that fix a design parameter, one CSV each |
| `ledgers/` | append-only JSONL, one row per measured cell |
| `tables/` | derived summaries |
| `figures/` | plots generated from the ledgers |
| `logs/` | job stdout |

A ledger row carries the cell, the measurement, and the provenance to trust it:
`git_rev`, `env_hash`, `prepare_host`, `measure_host`, `repeat`, `arm`, and
`osc_cached_mb` before and after. A cell that failed carries an `error` field
instead of a measurement, so a missing cell and a failed one are distinct.

Working directories under `/arf/scratch` hold the file sets a run creates and
deletes; they are not tracked.
