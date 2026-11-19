import re


def positive(val: float) -> None:
    assert val > 0, "Must be a positive number"

def resolution(val: tuple[float, float]) -> None:
    for v in val:
        assert v > 0 and (v % 2) == 0, "Must be positive even numbers"

def positive_integer(val: float) -> None:
    assert val > 0 and (val % 1) == 0, "Must be a positive whole number"

def non_negative(val: float) -> None:
    assert val >= 0, "Must be a non-negative number"

def positive_integer_range(val: tuple[float, float]) -> None:
    ok = True
    for v in val:
        ok = ok and v > 0 and (v % 1) == 0
    ok = ok and val[0] < val[1]
    assert ok, "Must be positive whole numbers, where a < b"

def negative(val: float) -> None:
    assert val < 0, "Must be a negative number"

# A diatonic, an optional accidental, and an octave.
note_pattern = re.compile("[A-Ga-g][#b]?[0-9]")

def tuning_frequency(val: str) -> None:
    if val.lstrip('-').replace('.', '', 1).isdigit():
        assert float(val) > 0, "Must be a positive number"
        return

    assert note_pattern.fullmatch(val), "Must be written in one of the following formats:\nD#2, Db2, D2, d#2, db2 or d2"

def both_positive(val: tuple[float, float]) -> None:
    for v in val:
        assert v > 0, "Must be positive numbers"

def between_0_1_inclusive(val: float) -> None:
    assert 0.0 <= val <= 1.0, "Must be in the range [0...1]"

def both_between_0_1_inclusive(val: tuple[float, float]) -> None:
    for v in val:
        0.0 <= v <= 1.0, "Both must be in the range [0...1]"
