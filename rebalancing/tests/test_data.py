"""Data layer: generator follows Table 1, validation, CSV round trip. No Gurobi needed."""

import pytest

from rebalancing.data import (
    Instance,
    generate_synthetic,
    load_instance,
    read_instance_csv,
    validate_instance,
    write_instance_csv,
)


def base_instance(**over):
    kw = dict(
        sites=["A", "B"],
        parts=["p"],
        suppliers=["m"],
        E={("A", "p"): 10},
        N={("B", "p"): 10},
        C={("A", "B"): 10},
        B={("m", "B", "p"): 20},
        UD=10000.0,
        FC1=100.0,
        FC2=1000.0,
    )
    kw.update(over)
    return Instance(**kw)


@pytest.mark.parametrize("n_sites", [5, 10, 20])
def test_generator_follows_table1(n_sites):
    inst = generate_synthetic(n_sites, 3, 5, seed=1)
    for k in inst.parts:
        exc = [s for s in inst.sites if (s, k) in inst.E]
        need = [s for s in inst.sites if (s, k) in inst.N]
        assert len(exc) == n_sites // 2
        assert sorted(exc + need) == sorted(inst.sites)  # need iff no excess
    assert all(20 <= v <= 120 for v in inst.E.values())
    assert all(50 <= v <= 150 for v in inst.N.values())
    assert all(50 <= v <= 150 for v in inst.C.values())
    assert all(75 <= v <= 175 for v in inst.B.values())
    assert len(inst.B) == len(inst.N) * 5
    assert (inst.FC1, inst.FC2, inst.UD) == (100.0, 1000.0, 10000.0)


def test_generator_is_reproducible():
    a, b = generate_synthetic(10, 2, 3, seed=4), generate_synthetic(10, 2, 3, seed=4)
    assert (a.E, a.N, a.C, a.B) == (b.E, b.N, b.C, b.B)
    c = generate_synthetic(10, 2, 3, seed=5)
    assert (a.E, a.N) != (c.E, c.N)


def test_ship_cost_by_part_keys():
    inst = generate_synthetic(6, 2, 2, seed=0, ship_cost_by_part=True)
    assert inst.ship_cost_by_part and all(len(key) == 3 for key in inst.C)


def test_both_ship_cost_formats():
    pair = base_instance(C={("A", "B"): 7})
    arc = base_instance(C={("A", "B", "p"): 9})
    assert pair.ship_cost("A", "B", "p") == 7
    assert arc.ship_cost("A", "B", "p") == 9


@pytest.mark.parametrize(
    "over, msg",
    [
        (dict(E={("A", "p"): -1}), "negative"),
        (dict(N={("B", "p"): 2.5}), "not an integer"),
        (dict(C={}), "no ship cost"),
        (dict(C={("B", "A"): 10}), "no ship cost"),
        (dict(B={}), "no purchase cost"),
        (dict(B={("m", "B", "p"): -3}), "negative"),
        (dict(UD=-1.0), "negative"),
        (dict(C={("A", "B"): 1, ("A", "B", "p"): 1}), "mixes"),
        (dict(E={("Q", "p"): 1}), "unknown site"),
    ],
)
def test_validation_errors(over, msg):
    with pytest.raises(ValueError, match=msg):
        validate_instance(base_instance(**over))


def test_validation_warns_on_excess_and_need_at_same_site():
    inst = base_instance(
        E={("A", "p"): 10, ("B", "p"): 3},
        C={("A", "B"): 10, ("B", "A"): 10},
        N={("B", "p"): 10, ("A", "p"): 4},
        B={("m", "B", "p"): 20, ("m", "A", "p"): 20},
    )
    with pytest.warns(UserWarning, match="both E > 0 and N > 0"):
        soft = validate_instance(inst)
    assert len(soft) == 1


def test_csv_round_trip(tmp_path):
    for by_part in (False, True):
        inst = generate_synthetic(8, 3, 2, seed=2, ship_cost_by_part=by_part)
        back = read_instance_csv(write_instance_csv(inst, tmp_path / str(by_part)))
        assert (back.E, back.N, back.C, back.B) == (inst.E, inst.N, inst.C, inst.B)
        assert (back.UD, back.FC1, back.FC2) == (inst.UD, inst.FC1, inst.FC2)


def test_load_instance_is_a_stub():
    with pytest.raises(NotImplementedError):
        load_instance("anything")
