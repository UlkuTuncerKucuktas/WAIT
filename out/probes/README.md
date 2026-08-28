# probes/

Measurements on TRUBA `/arf` — Lustre 2.15.3, two barbun nodes, unprivileged —
that fix a parameter the harness depends on.

| file | measures |
|---|---|
| `tier_benefit_staged` | staged open / fstat / read at 4 KiB on both tiers, with OST bulk RPCs per file |
| `inline_limit_by_extent` | which file sizes arrive inline, across DoM extents from 128 KiB to 1 MiB |
| `extents_granted` | the DoM extent the server grants for each requested value |
| `cliff_by_component_count` | read cost by file size on two- and three-component DoM layouts |
| `lazy_instantiation` | components instantiated and objects allocated as a file grows |
| `write_staged` | create, write and fsync separately, by tier and file size |
| `write_size_sweep` | durable-write margin at 4, 16 and 64 KiB, three repeats each |
| `s1_barrier_release` | barrier release per round, and sidecar cost cold against re-read |
| `s2_prefetch_threshold` | consumer stall against the compute-to-I/O ratio |
| `s4_ensemble` | write and generation margin against compute per task |
| `a4_drift` | MDT usage from `lfs df`, against background drift from other tenants |

The parameters these set are in `PLAN.md`; the reasoning is in `SCIENCE.md` §6.
