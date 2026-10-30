"""Property checks on random synthetic instances, plus the CLI end to end."""

import json
from collections import defaultdict

import pytest

gp = pytest.importorskip("gurobipy")

from rebalancing import score_solution, solve_gurobi  # noqa: E402
from rebalancing.data import generate_synthetic, purchase_keys, ship_arcs  # noqa: E402
from rebalancing.score_solution import score  # noqa: E402
from rebalancing.solve_gurobi import build_model, solve  # noqa: E402

# 12 sites x 4 parts x 3 suppliers: 144 X + 72 T + 24 U + <=24 binaries, well
# under the pip license's 2000 var / 2000 constraint cap.
CASES = [
    dict(n_sites=12, n_parts=4, n_suppliers=3, seed=7),
    dict(n_sites=12, n_parts=4, n_suppliers=3, seed=8, ship_cost_by_part=True),
    # Cheap unmet cost so U, T and X all show up in one solution.
    dict(n_sites=12, n_parts=4, n_suppliers=3, seed=9, ud=120.0),
    # Expensive fixed costs so some sites are left out.
    dict(n_sites=12, n_parts=4, n_suppliers=3, seed=10, fc1=3000.0, fc2=5000.0),
]


@pytest.fixture(scope="module", params=CASES, ids=lambda c: f"seed{c['seed']}")
def solved(request):
    inst = generate_synthetic(**request.param)
    res = solve(inst, mip_gap=0.0, quiet=True)
    assert res["status"] == "OPTIMAL"
    return inst, res


def flows(sol):
    out_ik, in_jk, out_i, in_j = (defaultdict(int) for _ in range(4))
    for (i, j, k), v in sol["X"].items():
        out_ik[i, k] += v
        in_jk[j, k] += v
        out_i[i] += v
        in_j[j] += v
    buy_jk, buy_j = defaultdict(int), defaultdict(int)
    for (m, j, k), v in sol["T"].items():
        buy_jk[j, k] += v
        buy_j[j] += v
    return out_ik, in_jk, out_i, in_j, buy_jk, buy_j


def test_excess_and_need_constraints_hold(solved):
    inst, res = solved
    sol = res["solution"]
    out_ik, in_jk, _, _, buy_jk, _ = flows(sol)
    for (i, k), s in out_ik.items():  # (2)
        assert s <= inst.E[i, k]
    for (j, k), n in inst.N.items():  # (3)
        assert in_jk[j, k] + buy_jk[j, k] + sol["U"].get((j, k), 0) == n


def test_binaries_match_flows(solved):
    inst, res = solved
    sol = res["solution"]
    _, _, out_i, in_j, _, buy_j = flows(sol)
    for s in inst.sites:
        assert (sol["Y"].get(s, 0) == 1) == (out_i[s] > 0), f"Y[{s}]"
        assert (sol["Z"].get(s, 0) == 1) == (in_j[s] + buy_j[s] > 0), f"Z[{s}]"


def test_independent_score_matches(solved):
    inst, res = solved
    chk = score(inst, res["solution"])
    assert chk["violations"] == []
    assert chk["idle_binaries"] == []
    assert chk["objective"]["total"] == pytest.approx(res["ObjVal"], rel=1e-6)
    assert res["cost_breakdown"]["total"] == pytest.approx(res["ObjVal"], rel=1e-6)


def test_strong_linking_same_optimum(solved):
    inst, res = solved
    strong = solve(inst, mip_gap=0.0, strong_linking=True, quiet=True)
    assert strong["status"] == "OPTIMAL"
    assert strong["ObjVal"] == pytest.approx(res["ObjVal"], rel=1e-9)
    assert score(inst, strong["solution"], strong_linking=True)["violations"] == []


def test_variables_are_sparse():
    inst = generate_synthetic(12, 4, 3, seed=7)
    model, vars_, counts = build_model(inst, env=_quiet_env())
    for (i, j, k) in vars_["X"]:
        assert i != j and inst.E[i, k] > 0 and inst.N[j, k] > 0
    assert set(vars_["X"]) == set(ship_arcs(inst))
    assert set(vars_["T"]) == set(purchase_keys(inst))
    assert set(vars_["U"]) == set(inst.N)
    # per part: 6 sources x 6 sinks; 6 needs x 3 suppliers
    assert counts["vars"]["X"] == 4 * 6 * 6
    assert counts["vars"]["T"] == 4 * 6 * 3
    assert counts["vars"]["U"] == 4 * 6
    # base model: one row per (i,k) with arcs, (j,k) with need, Y site, Z site
    c = counts["constraints"]
    assert c["c2_excess"] == 24 and c["c3_need"] == 24
    assert c["c4_ship_link"] == counts["vars"]["Y"] and c["c5_recv_link"] == counts["vars"]["Z"]
    model.dispose()


def test_cli_end_to_end(tmp_path, capsys):
    rc = solve_gurobi.main([
        "--sites", "6", "--parts", "2", "--suppliers", "2", "--seed", "3",
        "--out-dir", str(tmp_path), "--write-mps", "--quiet",
    ])
    assert rc == 0
    run_dir = tmp_path / "syn_s6_p2_m2_seed3"
    for f in ("summary.json", "X.csv", "T.csv", "U.csv", "Y.csv", "Z.csv",
              "cost_breakdown.csv", "model.mps", "gurobi.log", "instance/meta.json"):
        assert (run_dir / f).exists(), f
    summary = json.loads((run_dir / "summary.json").read_text())
    for key in ("status", "ObjVal", "ObjBound", "MIPGap", "Runtime", "NodeCount", "num_vars",
                "num_constraints", "timing_seconds", "memory", "gurobi_version", "git", "params"):
        assert key in summary, key
    assert score_solution.main(["--run-dir", str(run_dir)]) == 0
    # The MPS round-trips to the same optimum (what Aarav will load into CPLEX).
    m = gp.read(str(run_dir / "model.mps"), env=_quiet_env())
    m.Params.MIPGap = 0.0
    m.optimize()
    assert m.ObjVal == pytest.approx(summary["ObjVal"], rel=1e-4)


def _quiet_env():
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    return env
