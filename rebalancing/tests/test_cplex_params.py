"""The cplex parameter set (FC1 102, FC2 103, UD 104, constant BIGM 10000).

Aarav's CPLEX model uses the literal 102/103/104 and BIGM = 10000; these tests
pin down what that changes (the costs) and what it must not change (the
optimum, whatever M the linking rows use).
"""

import json
import warnings
from pathlib import Path

import pytest

gp = pytest.importorskip("gurobipy")

from rebalancing import score_solution, solve_gurobi  # noqa: E402
from rebalancing.data import Instance, load_instance, validate_instance  # noqa: E402
from rebalancing.score_solution import score  # noqa: E402
from rebalancing.solve_gurobi import PARAM_SETS, check_old_dataset_shape, solve  # noqa: E402

CPLEX = PARAM_SETS["cplex"]
DATA_ROOT = Path(__file__).resolve().parents[3] / "OldModelDatasets"
SMALL_WORKBOOKS = sorted(DATA_ROOT.glob("berlin_10_hubs/*.xlsx")) + sorted(DATA_ROOT.glob("vienna_20_hubs/*.xlsx"))


def two_site_cplex(c, b, units=10):
    inst = Instance(
        sites=["A", "B"],
        parts=["p"],
        suppliers=["m"],
        E={("A", "p"): units},
        N={("B", "p"): units},
        C={("A", "B"): c},
        B={("m", "B", "p"): b},
        UD=CPLEX["ud"],
        FC1=CPLEX["fc1"],
        FC2=CPLEX["fc2"],
    )
    validate_instance(inst)
    return inst


@pytest.mark.parametrize("bigm", [CPLEX["bigm"], None], ids=["bigm10000", "tightM"])
def test_unmet_is_optimal_when_ud_is_below_every_option(bigm):
    # cplex params: FC1 102, FC2 103, UD 104. C = 105, B = 110, need 10.
    # Every unit costs more to ship (105) or buy (110) than to leave unmet (104),
    # before any fixed cost is added:
    #   unmet all : 10*104                   = 1040   (no Y, no Z)
    #   ship all  : 10*105 + FC1 102 + FC2 103 = 1255
    #   buy all   : 10*110          + FC2 103 = 1203
    #   any mix receiving r >= 1 units: 104*(10-r) + >=105*r + 103 > 1040
    # optimum 1040: U = 10, every other variable 0.
    inst = two_site_cplex(c=105, b=110)
    res = solve(inst, mip_gap=0.0, quiet=True, bigm=bigm)
    assert res["status"] == "OPTIMAL"
    sol = res["solution"]
    assert res["ObjVal"] == pytest.approx(1040)
    assert sol["U"] == {("B", "p"): 10}
    assert sol["X"] == {} and sol["T"] == {} and sol["Y"] == {} and sol["Z"] == {}
    assert res["cost_breakdown"] == {
        "ship": 0, "purchase": 0, "unmet": 1040.0,
        "fixed_ship_FC1": 0, "fixed_receive_FC2": 0, "total": 1040.0,
    }
    chk = score(inst, sol, bigm=bigm)
    assert chk["violations"] == [] and chk["objective"]["total"] == pytest.approx(1040)


def test_same_instance_buys_under_pow10():
    # Control: the same C/B under pow10 (FC1 100, FC2 1000, UD 10000):
    #   buy all   : 10*110         + 1000 = 2100
    #   ship all  : 10*105 + 100   + 1000 = 2150
    #   mixed x>=1: 105x + 110(10-x) + 1100 = 2200 - 5x >= 2150
    #   unmet all : 10*10000             = 100000
    # optimum 2100: buy 10, nothing unmet.
    inst = two_site_cplex(c=105, b=110)
    inst.UD, inst.FC1, inst.FC2 = 10000.0, 100.0, 1000.0
    res = solve(inst, mip_gap=0.0, quiet=True)
    assert res["ObjVal"] == pytest.approx(2100)
    assert res["solution"]["T"] == {("m", "B", "p"): 10} and res["solution"]["U"] == {}


def test_bigm_below_tight_m_warns_and_cuts_the_optimum():
    # Tight M for A and B is 10. With BIGM 4, A may ship at most 4 and B may
    # receive at most 4, so 6 units go unmet: the optimum moves. This is why
    # the runs record bigm_below_tight_M_rows (0 on every old dataset).
    inst = two_site_cplex(c=10, b=200)  # shipping all 10 is optimal with a valid M
    full = solve(inst, mip_gap=0.0, quiet=True, bigm=CPLEX["bigm"])
    assert full["ObjVal"] == pytest.approx(10 * 10 + 102 + 103)
    with pytest.warns(UserWarning, match="below the tight M"):
        cut = solve(inst, mip_gap=0.0, quiet=True, bigm=4.0)
    assert cut["counts"]["bigm_below_tight_M_rows"] == 2
    assert cut["ObjVal"] == pytest.approx(4 * 10 + 102 + 103 + 6 * 104)
    # The scorer flags the full solution against BIGM 4.
    assert any("BIGM = 4.0" in v for v in score(inst, full["solution"], bigm=4.0)["violations"])


def test_bigm_with_strong_linking_is_rejected():
    with pytest.raises(ValueError, match="strong_linking"):
        solve(two_site_cplex(c=10, b=20), quiet=True, strong_linking=True, bigm=10000.0)


@pytest.mark.skipif(not SMALL_WORKBOOKS, reason=f"no OldModelDatasets checkout at {DATA_ROOT}")
@pytest.mark.parametrize("xlsx", SMALL_WORKBOOKS, ids=lambda p: f"{p.parent.name.split('_')[0]}_{p.stem}")
def test_cplex_tight_m_and_bigm_give_the_same_optimum(xlsx):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*purchase costs on needed.*")
        inst = load_instance(xlsx, ud=CPLEX["ud"], fc1=CPLEX["fc1"], fc2=CPLEX["fc2"])
    tight = solve(inst, mip_gap=0.0, quiet=True, bigm=None)
    big = solve(inst, mip_gap=0.0, quiet=True, bigm=CPLEX["bigm"])
    assert tight["status"] == big["status"] == "OPTIMAL"
    assert big["counts"]["bigm_below_tight_M_rows"] == 0
    assert tight["ObjVal"] == big["ObjVal"]


def test_cli_param_set_resolution_and_run_names(tmp_path):
    args = solve_gurobi.parse_args(["--sites", "4", "--parts", "1", "--suppliers", "2"])
    assert (args.param_set, args.fc1, args.fc2, args.ud, args.bigm) == ("pow10", 100.0, 1000.0, 10000.0, None)
    assert solve_gurobi.run_name_suffix(args) == ""

    args = solve_gurobi.parse_args(["--sites", "4", "--parts", "1", "--suppliers", "2", "--param-set", "cplex"])
    assert (args.fc1, args.fc2, args.ud, args.bigm) == (102.0, 103.0, 104.0, 10000.0)
    assert solve_gurobi.run_name_suffix(args) == "_cplex"

    args = solve_gurobi.parse_args(["--sites", "4", "--parts", "1", "--suppliers", "2",
                                    "--param-set", "cplex", "--bigm", "tight", "--ud", "500"])
    assert (args.ud, args.bigm) == (500.0, None)
    assert solve_gurobi.run_name_suffix(args) == "_cplex_ud500_tightM"

    args = solve_gurobi.parse_args(["--sites", "4", "--parts", "1", "--suppliers", "2",
                                    "--free-purchase", "block", "--strong-linking"])
    assert solve_gurobi.run_name_suffix(args) == "_blockfree_strong"  # unchanged from before

    rc = solve_gurobi.main(["--sites", "6", "--parts", "1", "--suppliers", "2", "--seed", "3",
                            "--param-set", "cplex", "--mip-gap", "0", "--out-dir", str(tmp_path), "--quiet"])
    assert rc == 0
    run_dir = tmp_path / "syn_s6_p1_m2_seed3_cplex"
    s = json.loads((run_dir / "summary.json").read_text())
    assert s["params"]["param_set"] == "cplex" and s["params"]["bigm"] == 10000.0
    assert s["params"]["linking"] == "aggregated_constant_bigm"
    assert (s["instance"]["FC1"], s["instance"]["FC2"], s["instance"]["UD"]) == (102.0, 103.0, 104.0)
    assert score_solution.main(["--run-dir", str(run_dir)]) == 0


def test_old_dataset_shape_check_fails_loudly():
    inst = two_site_cplex(c=10, b=20)  # 1 supplier, 1 part
    with pytest.raises(SystemExit, match="expected 3 suppliers and 1 part, found 1 suppliers"):
        check_old_dataset_shape(inst, "fake.xlsx")
    inst.suppliers = ["1", "2", "3"]
    check_old_dataset_shape(inst, "fake.xlsx")  # passes
