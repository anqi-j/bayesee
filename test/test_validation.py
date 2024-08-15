import pytest
from bayesee.validation import *


def test_is_integer_typical_cases():
    assert is_integer(1) == True
    assert is_integer(0.5) == False


def test_all_integers_typical_cases():
    assert all_integers((1, 2, 3)) == True
    assert all_integers((1, 0.5, 3)) == False


def test_is_number_typical_cases():
    assert is_number(0.5) == True
    assert is_number("good") == False


def test_all_numbers_typical_cases():
    assert all_numbers((1, 0.5, 3)) == True
    assert all_numbers((1, 0.5, "good")) == False
