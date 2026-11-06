# Gurobi vs Aarav's CPLEX model: parameter alignment

**The CPLEX match is NOT verified.** The `cplex_obj` / `cplex_match` columns of
`old_datasets_summary_cplex.csv` stay empty until Aarav's objectives are loaded:

```bash
python -m rebalancing.run_old_datasets --param-set cplex --cplex-results <aarav.csv> --fill-only
```

(`<aarav.csv>` has the columns `city, seed, cplex_obj`. A match means exact equality.)

## What was run

- All 50 OldModelDatasets workbooks, `keep` variant only (zero-cost purchases kept), `--mip-gap 0`.
- **pow10**: FC1 100, FC2 1000, UD 10000, with the tight per-site M. These are the existing runs in `old_datasets_summary.csv`, unchanged.
- **cplex**: FC1 102, FC2 103, UD 104, with BIGM 10000 in (4)/(5). New runs:
  - summary: `old_datasets_summary_cplex.csv`
  - run directories: `results/rebalancing/old_datasets_cplex/`
  - MPS files for CPLEX (seeds 1001/2001/3001/4001/5001): `results/rebalancing/old_datasets_cplex/mps/`
- Every workbook has 3 suppliers and 1 part. The solver now refuses a workbook that doesn't.

## Checks (cplex, 50 instances)

| check | result |
|---|---|
| status OPTIMAL, gap 0 | 50/50 |
| `score_solution` PASS, which checks (2)–(5) with both the tight M and BIGM 10000 | 50/50 |
| ObjVal with BIGM 10000 = ObjVal with tight M (exact) | 50/50 |
| linking rows where BIGM 10000 is below the tight M | 0 (the largest per-site M is 149) |

The choice of big-M doesn't move the optimum. Any difference from CPLEX comes from the
cost parameters or the data, not from M.

## pow10 vs cplex, per city (means over 10 instances)

| city (hubs) | param set | objective | unmet | bought | shipped | Y_on | Z_on |
|---|---|---:|---:|---:|---:|---:|---:|
| Berlin (10) | pow10 | 20,659.5 | 0.0 | 171.5 | 149.7 | 2.6 | 4.4 |
| | cplex | 16,312.9 | 28.3 | 143.2 | 149.7 | 2.6 | 4.1 |
| Vienna (20) | pow10 | 64,893.6 | 0.0 | 292.3 | 608.9 | 7.9 | 10.4 |
| | cplex | 55,432.9 | 20.3 | 274.7 | 606.2 | 7.9 | 10.2 |
| London (50) | pow10 | 148,664.9 | 0.0 | 323.2 | 1,844.3 | 22.8 | 26.0 |
| | cplex | 125,388.3 | 2.4 | 320.8 | 1,844.3 | 22.8 | 26.0 |
| Paris (100) | pow10 | 263,535.3 | 0.0 | 266.2 | 3,770.5 | 43.5 | 48.2 |
| | cplex | 220,386.9 | 0.0 | 266.2 | 3,770.5 | 43.5 | 48.2 |
| Milan (200) | pow10 | 540,424.0 | 0.0 | 514.0 | 7,813.9 | 91.6 | 99.3 |
| | cplex | 451,535.1 | 0.0 | 514.0 | 7,813.9 | 91.5 | 99.3 |

Unmet, bought and shipped are in units. Y_on and Z_on count sites.

## Instances with unmet > 0 under cplex

6 of 50 instances. Under pow10, no instance leaves any need unmet.

| city | seed | units unmet | total need | share unmet |
|---|---:|---:|---:|---:|
| berlin | 1001 | 255 | 710 | 35.9% |
| berlin | 1008 | 28 | 311 | 9.0% |
| vienna | 2001 | 23 | 1,110 | 2.1% |
| vienna | 2006 | 17 | 901 | 1.9% |
| vienna | 2008 | 163 | 1,504 | 10.8% |
| london | 3008 | 24 | 2,470 | 1.0% |

Over all 50 instances, 510 of 157,545 units of need go unmet (0.3%).

With UD = 104, a unit stays unmet whenever covering it costs more than 104. Hub-to-hub
costs run 50–149 and purchase costs run 75–174, so this happens only for some units.
`old_datasets_summary_cplex.csv` has the per-instance cost breakdown.
