import pytest

from Pyriod.combinations import evaluate_combination, validate_combination


def test_evaluate_combination():
    values = {"f0": 10.0, "f1": 2.0}

    assert evaluate_combination("f0+f1", values) == 12.0
    assert evaluate_combination("2*f0-f1", values) == 18.0
    assert evaluate_combination("f0/2", values) == 5.0


def test_rejects_function_calls():
    values = {"f0": 10.0}

    with pytest.raises(ValueError):
        evaluate_combination("__import__('os').system('echo nope')", values)


def test_validate_combination():
    assert validate_combination("f0+f1", {"f0", "f1"})
    assert not validate_combination("f0+unknown", {"f0", "f1"})