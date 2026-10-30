"""Tiny instances whose optimum is worked out by hand in each test's comment.

Unless a test says otherwise: FC1 = 100, FC2 = 1000, UD = 10000.
"""

import pytest

gp = pytest.importorskip("gurobipy")

from rebalancing.data import Instance, validate_instance  # noqa: E402
from rebalancing.score_solution import score  # noqa: E402
from rebalancing.solve_gurobi import solve  # noqa: E402


def two_site(c, b, fc1=100.0, fc2=1000.0, ud=10000.0, units=10):
    """Site A has `units` excess of part p, site B needs `units`; one supplier m."""
    inst = Instance(
        sites=["A", "B"],
        parts=["p"],
        suppliers=["m"],
        E={("A", "p"): units},
        N={("B", "p"): units},
        C={("A", "B"): c},
        B={("m", "B", "p"): b},
        UD=ud,
        FC1=fc1,
        FC2=fc2,
    )
    validate_instance(inst)
    return inst


def run(inst):
    res = solve(inst, mip_gap=0.0, quiet=True)
    assert res["status"] == "OPTIMAL"
    chk = score(inst, res["solution"])
    assert chk["violations"] == []
    assert chk["objective"]["total"] == pytest.approx(res["ObjVal"], rel=1e-9)
    return res, res["solution"]


def test_a_ships_when_shipping_is_cheaper():
    # C = 10, B = 100, need 10.
    #   ship all : 10*10 + FC1 100 + FC2 1000 = 1200
    #   buy all  : 10*100          + FC2 1000 = 2000
    #   mixed x  : 10x + 100(10-x) + 1100, decreasing in x -> x = 10
    #   unmet    : 10*10000                   = 100000
    # optimum 1200: ship 10, Y[A] = Z[B] = 1.
    res, sol = run(two_site(c=10, b=100))
    assert res["ObjVal"] == pytest.approx(1200)
    assert sol["X"] == {("A", "B", "p"): 10}
    assert sol["T"] == {} and sol["U"] == {}
    assert sol["Y"] == {"A": 1} and sol["Z"] == {"B": 1}


def test_b_buys_when_buying_is_cheaper():
    # C = 80, B = 60, need 10.
    #   buy all  : 10*60          + FC2 1000 = 1600
    #   ship all : 10*80 + FC1 100 + FC2 1000 = 1900
    #   mixed x>=1: 80x + 60(10-x) + 1100 = 1700 + 20x > 1600
    # optimum 1600: buy 10 from m, Y[A] = 0, Z[B] = 1.
    res, sol = run(two_site(c=80, b=60))
    assert res["ObjVal"] == pytest.approx(1600)
    assert sol["X"] == {} and sol["U"] == {}
    assert sol["T"] == {("m", "B", "p"): 10}
    assert sol["Y"] == {} and sol["Z"] == {"B": 1}


def test_c_leaves_demand_unmet_when_ud_is_cheapest():
    # UD = 5, C = 10, B = 20, need 10.
    #   unmet    : 10*5                        = 50   (no Z: nothing received)
    #   ship all : 10*10 + FC1 100 + FC2 1000  = 1200
    #   buy all  : 10*20          + FC2 1000   = 1200
    # optimum 50: U = 10, every binary 0.
    res, sol = run(two_site(c=10, b=20, ud=5))
    assert res["ObjVal"] == pytest.approx(50)
    assert sol["U"] == {("B", "p"): 10}
    assert sol["X"] == {} and sol["T"] == {}
    assert sol["Y"] == {} and sol["Z"] == {}


@pytest.mark.parametrize(
    "fc1, expect_obj, expect_ship",
    [
        # FC1 = 500: ship all 10*10 + 500 + 1000 = 1600 vs buy all 10*20 + 1000 = 1200
        #            (mixed x>=1: 10x + 20(10-x) + 1500 >= 1610) -> buy, 1200.
        (500, 1200, False),
        # Control, FC1 = 50: ship all 100 + 50 + 1000 = 1150 < buy 1200 -> ship, 1150.
        (50, 1150, True),
    ],
)
def test_d_large_fc1_makes_one_shipment_not_worth_a_sender(fc1, expect_obj, expect_ship):
    res, sol = run(two_site(c=10, b=20, fc1=fc1))
    assert res["ObjVal"] == pytest.approx(expect_obj)
    if expect_ship:
        assert sol["X"] == {("A", "B", "p"): 10} and sol["Y"] == {"A": 1}
    else:
        assert sol["X"] == {} and sol["Y"] == {}
        assert sol["T"] == {("m", "B", "p"): 10}


@pytest.mark.parametrize(
    "free_purchase, expect_obj",
    [
        # B = 0 (the old Java generator's m == j artefact), C = 10, need 10.
        #   keep : buy 10 free + FC2 1000                  = 1000
        #          (ship all: 10*10 + FC1 100 + FC2 1000   = 1200)
        ("keep", 1000),
        # block: the zero-cost T is removed, so ship all   = 1200
        #          (unmet: 10*10000 = 100000)
        ("block", 1200),
    ],
)
def test_free_purchase_keep_vs_block(free_purchase, expect_obj):
    with pytest.warns(UserWarning, match="purchase costs on needed"):
        inst = two_site(c=10, b=0)
    res = solve(inst, mip_gap=0.0, quiet=True, free_purchase=free_purchase)
    sol = res["solution"]
    assert res["ObjVal"] == pytest.approx(expect_obj)
    assert res["counts"]["vars"]["T_free_blocked"] == (1 if free_purchase == "block" else 0)
    if free_purchase == "keep":
        assert sol["T"] == {("m", "B", "p"): 10} and sol["X"] == {}
    else:
        assert sol["T"] == {} and sol["X"] == {("A", "B", "p"): 10}
    assert score(inst, sol, free_purchase=free_purchase)["violations"] == []
    # The scorer flags a free purchase under block.
    keep_sol = {**sol, "T": {("m", "B", "p"): 10}, "X": {}, "Y": {}}
    assert any("free-purchase" in v for v in score(inst, keep_sol, free_purchase="block")["violations"])


def test_e_each_need_buys_only_from_its_cheapest_supplier():
    # No excess anywhere, so no X, no Y. Suppliers are uncapacitated, so every
    # (j,k) buys its whole need from its single cheapest supplier:
    #   (s0,p0) N=5  B = m0:10 m1:20 m2:30 -> m0   5*10 =  50
    #   (s0,p1) N=7  B = m0:25 m1:15 m2:35 -> m1   7*15 = 105
    #   (s1,p0) N=3  B = m0:40 m1:30 m2:12 -> m2   3*12 =  36
    #   (s1,p1) N=4  B = m0:18 m1:22 m2:19 -> m0   4*18 =  72
    #   purchases 263 + FC2 * 2 receiving sites 2000 = 2263
    prices = {
        ("s0", "p0"): (10, 20, 30),
        ("s0", "p1"): (25, 15, 35),
        ("s1", "p0"): (40, 30, 12),
        ("s1", "p1"): (18, 22, 19),
    }
    need = {("s0", "p0"): 5, ("s0", "p1"): 7, ("s1", "p0"): 3, ("s1", "p1"): 4}
    inst = Instance(
        sites=["s0", "s1"],
        parts=["p0", "p1"],
        suppliers=["m0", "m1", "m2"],
        E={},
        N=need,
        C={},
        B={(f"m{m}", j, k): p[m] for (j, k), p in prices.items() for m in range(3)},
        UD=10000.0,
        FC1=100.0,
        FC2=1000.0,
    )
    validate_instance(inst)
    res, sol = run(inst)
    assert res["ObjVal"] == pytest.approx(2263)
    assert sol["T"] == {
        ("m0", "s0", "p0"): 5,
        ("m1", "s0", "p1"): 7,
        ("m2", "s1", "p0"): 3,
        ("m0", "s1", "p1"): 4,
    }
    assert sol["X"] == {} and sol["U"] == {} and sol["Y"] == {}
    assert sol["Z"] == {"s0": 1, "s1": 1}
