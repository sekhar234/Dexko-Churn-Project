# Phase 3–4 Implementation: Offline Snapshot + Certified Load

This build mirrors the current executable churn flow from the latest repository while remaining fully offline.

## Production-to-local mapping

| Production concept | Local implementation |
|---|---|
| Lakehouse Federation EDW reads | Synthetic EDW files under `data/source/edw/` |
| `jdbc_*_weekly` Delta tables | Partitioned local Parquet/CSV under `data/edw_cache/` |
| `snapshot_registry` Delta table | Local registry file with SUCCESS row per snapshot |
| `master_weekly` Delta table | Local partitioned master snapshot |
| `02_load_data` Spark notebook | `src/local_pipeline/load_data.py` |
| ZIP anomaly Delta table | `data/edw_cache/zipcode_anomaly_eda/processing_date=.../` |
| `dex_v2_base` | `data/certified/dex_v2_base.*` |
| `dex_v2_cust_attrs` | `data/certified/dex_v2_cust_attrs.*` |
| Databricks display/funnel | `data/reports/load_funnel.*` + QA Markdown/JSON |

## Snapshot behavior reproduced

1. Monday-aligned `snapshot_week`.
2. Five weekly source snapshots.
3. Single snapshot alignment across all five tables.
4. `master_weekly` left joins with dimension-key deduplication.
5. Join-loss guard at 5%.
6. `snapshot_registry` SUCCESS upsert for `master_weekly`.
7. Reruns replace the same local partition instead of appending duplicates.

## `02_load_data` behavior reproduced

1. Read latest SUCCESS snapshot.
2. Normalize/sanitize source columns.
3. Build one-row-per-key customer bridge.
4. Deduplicate/filter product dimension; exclude `UNASSIGNED`, `N/A`, `I/N`, `I/C` categories.
5. Deduplicate warehouse dimension and join BU levels.
6. Drop blank/zero/UNASSIGNED harmonized customer names.
7. Country-based ZIP normalization and anomaly routing.
8. Build Customer × ZIP attributes using DEX-only positive revenue.
9. Restrict attributes to OEM and Stocking Dealer.
10. Select winning group/branch/key by revenue and collect customer IDs.
11. Join attributes back at Customer × ZIP grain.
12. Apply invoice filters: date, intercompany, positive revenue/quantity, DEX entity.
13. Add `purchase_dt`, `salesorder_dt`, `is_positive_purchase`.
14. Persist certified `dex_v2_base` and `dex_v2_cust_attrs`.

## Deterministic small-run reference

Using seed 42, the current local reference run produced:

| Stage | Rows |
|---|---:|
| Raw invoice | 36,277 |
| After customer join | 36,277 |
| After product join | 36,277 |
| After warehouse join | 36,277 |
| After bad-name removal | 35,704 |
| After ZIP validation | 35,170 |
| OEM + Stocking Dealer | 29,615 |
| Invoice date non-null | 29,560 |
| Revenue > 0 | 29,391 |
| Quantity > 0 | 29,276 |
| DEX only | 28,250 |

Other outputs:

- `dex_v2_cust_attrs`: 175 Customer × ZIP rows
- ZIP anomaly audit: 534 rows
- `master_weekly` join loss: 0.0%
- Automated test suite: 4 passing tests

## Important implementation note

The local ingestion stages use pandas rather than Spark because the objective is an offline behavioral replica with no Databricks/Java dependency. Business rules, grains, filters and output contracts are aligned to the current code. Feature engineering is the next phase and will be implemented against these certified outputs.
