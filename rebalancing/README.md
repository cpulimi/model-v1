# Inventory rebalancing — base model, Gurobi

Base model only: constraints (2)–(5), nothing else. The extra constraints from
Dr. Fowler get added on top of this later.

## Model

Sets: sites I (any site can be a source *i* or a destination *j*), parts K, suppliers M.

| var | domain | meaning |
|---|---|---|
| `X[i,j,k]` | integer ≥ 0 | units of part k shipped from site i to site j |
| `T[m,j,k]` | integer ≥ 0 | units of part k bought from supplier m for site j |
| `U[j,k]` | integer ≥ 0 | unmet need of part k at site j |
| `Y[i]` | binary | 1 if site i ships anything out |
| `Z[j]` | binary | 1 if site j receives anything (shipped in or bought) |

```
min  Σ C·X + Σ B·T + UD·Σ U + FC1·Σ Y + FC2·Σ Z
(2)  Σ_j X[i,j,k]                              ≤ E[i,k]        ∀ i,k
(3)  Σ_i X[i,j,k] + Σ_m T[m,j,k] + U[j,k]      = N[j,k]        ∀ j,k with N > 0
(4)  Σ_{j,k} X[i,j,k]                          ≤ M_i · Y[i]    M_i = Σ_k E[i,k]
(5)  Σ_{i,k} X[i,j,k] + Σ_{m,k} T[m,j,k]       ≤ M_j · Z[j]    M_j = Σ_k N[j,k]
```

`--strong-linking` replaces (4)/(5) with per-arc rows:
`X[i,j,k] ≤ min(E[i,k],N[j,k])·Y[i]`, `X[i,j,k] ≤ min(E[i,k],N[j,k])·Z[j]`,
`T[m,j,k] ≤ N[j,k]·Z[j]`. The integer feasible set is the same (X ≤ E from (2);
X ≤ N and T ≤ N from (3)), but the LP relaxation is tighter. This is for comparing LP bounds.

### Sparsity

- `X` exists only on arcs with `E[i,k] > 0`, `N[j,k] > 0`, `i ≠ j`. These are built per part from that part's nonzero sources and sinks, never by looping over I×I×K.
- `T` and `U` exist only where `N[j,k] > 0`.
- `Y` exists only for sites with at least one outgoing arc, and `Z` only for sites with at least one incoming arc or purchase. Anywhere else, (4) or (5) would read `0 ≤ M·Y`.
- (2) gets a row only for `(i,k)` pairs that have an outgoing arc. Without one, the row would read `0 ≤ E`.
- Objective coefficients go in through `addVars(obj=...)`, so the model never builds a large objective expression.
- `C` can be keyed `(i,j)` (the same cost for every part) or `(i,j,k)`. The model checks which form it has from the key length.

## Files

| file | what |
|---|---|
| `data.py` | `Instance` dataclass, `generate_synthetic` (QVA paper, Table 1), `load_instance` (Aarav's old-model workbooks), `validate_instance`, CSV round trip |
| `solve_gurobi.py` | model build, solve, CLI, output writing |
| `score_solution.py` | independent recheck from CSVs + instance, no Gurobi |
| `collect_results.py` | gathers `summary.json` + `score.json` files into a markdown/CSV table |
| `run_old_datasets.py` | all 50 old datasets × {keep, block}, checked against the reference optima |
| `reference/old_datasets_reference_optima.csv` | reference optima (HiGHS, plus Gurobi up to 50 hubs) |
| `results/old_datasets_summary.csv` | one row per old-dataset run: ObjVal vs reference, scorer result, flows, per-instance data checks |
| `sbatch_rebalancing.sh` | Sol job script (written, not submitted) |
| `tests/` | pytest: hand-solved cases (a)–(e), property checks, data layer, CLI end to end |

## Run

From the repo root:

```bash
python -m pytest rebalancing/tests -q
python -m rebalancing.solve_gurobi --sites 10 --parts 1 --suppliers 5 --seed 0 --write-mps
python -m rebalancing.score_solution --run-dir results/rebalancing/syn_s10_p1_m5_seed0
python -m rebalancing.collect_results results/rebalancing
```

CLI: `--sites --parts --suppliers --seed --fc1 --fc2 --ud --time-limit --mip-gap
(1e-4) --threads --strong-linking --out-dir --write-mps`, plus `--ship-cost-by-part`,
`--run-name` and `--quiet`.

Each run directory contains:
- `summary.json`
- `X.csv`, `T.csv`, `U.csv`, `Y.csv`, `Z.csv` (nonzeros only, rounded to integers)
- `cost_breakdown.csv` (the five objective terms)
- `gurobi.log`
- `instance/` (the exact instance that was solved)
- `model.mps` (with `--write-mps`, for CPLEX)

`summary.json` also records:
- variable and constraint counts by type
- build and solve times
- peak memory: VmHWM on Linux. VmHWM isn't available on macOS, so there it's `null` and the `ru_maxrss` fallback is reported.
- Gurobi version, git commit, and all parameters

On Sol: `sbatch rebalancing/sbatch_rebalancing.sh --sites 100 --parts 100 --suppliers 5 --seed 0`
(see the header for a size ladder). The local pip `gurobipy` license caps models at
2000 vars / 2000 constraints.

## Old datasets (Aarav's Java generator)

Clone [cpulimi/OldModelDatasets](https://github.com/cpulimi/OldModelDatasets) next to this repo, then:

```bash
python -m rebalancing.solve_gurobi --instance ../OldModelDatasets/berlin_10_hubs/1001.xlsx --mip-gap 0
python -m rebalancing.run_old_datasets            # all 50 x {keep, block}
```

- **Workbook layout:** each `<city>_<n>_hubs/<seed>.xlsx` has the sheets `PARA_Eik`, `PARA_Njk`, `PARA_Bmjk` and `PARA_Cijk`. Row 1 is a header and every cell is text. There is 1 part, 3 suppliers, and the hubs act as both i and j.
- **What `load_instance` does:** converts every cell to int, drops zero E/N rows, drops the C diagonal, and keeps C keyed `(i,j,k)`. UD/FC1/FC2 come from the arguments. Ids are kept as the workbook's strings, so hub `1` and supplier `1` are different entities that share a label.
- **Run names:** runs are named `<city>_<seed>`, plus `_blockfree` with `--free-purchase block`.
- **Free purchases:** the Java generator sets `B[m,j] = 0` whenever m == j, so hubs 1–3 can buy from suppliers 1–3 for free. `validate_instance` warns about this. `--free-purchase block` removes the T variables whose B is 0, which gives the same optimum as the reference's "price at 175", because a 175 option is never the cheapest.
- **Results:** all 100 runs match the reference `highs_obj` exactly and pass `score_solution`. See `results/old_datasets_summary.csv`.

## Interpretation choices (check these)

1. **T index.** The source doc defines `T_ijk` but uses `T_mjk` everywhere else. The code uses `T[m,j,k]`.
2. **FC1 / FC2 / UD.** The paper prints "102, 103, 104". The defaults read these as 10², 10³, 10⁴ (100, 1000, 10000). All three are CLI arguments.
3. **"50% of sites have 0 excess."** This is applied per part: exactly `n_sites // 2` random sites have excess of that part, so with 5 sites, 2 have excess and 3 have need. A per-cell coin flip could leave a small instance with no excess at all.
4. **Synthetic ship cost** is drawn per site pair, `C[i,j]`, by default. `--ship-cost-by-part` draws `C[i,j,k]` instead. With 1 part (as in the paper) the two are the same thing.
5. **Every supplier sells every needed (j,k).** `T` exists for all m wherever N > 0, and a missing `B` is a validation error, as specified. If the real data has supplier catalogs, the `T` keys should come from B's keys instead.
6. **Z also counts purchases**, so a site that only buys still pays FC2. A site that leaves all of its need unmet pays nothing. FC2 is charged once per site, however many parts it receives.
7. **No supplier capacity.** Each (j,k) therefore buys its whole purchase from its single cheapest supplier. The code doesn't exploit this (no constraint or variable reduction was added), but it means |M| only adds dominated columns.
8. **Leftover excess** has no cost. A site can ship one part and receive another, paying both FC1 and FC2.
9. **Validation.** A (site, part) pair with both E > 0 and N > 0 gets a warning. It is allowed, because no self-shipment is possible. Negative values, missing costs on existing arcs, and **non-integer E or N** are errors. Fractional need would make (3) infeasible with integer X/T/U.
10. **MIP gap.** With the default `--mip-gap 1e-4`, Gurobi can return OPTIMAL while still slightly above the bound (e.g. 20-site seed 1: ObjVal 64246, bound 64241). Use `--mip-gap 0` to prove optimality exactly.
