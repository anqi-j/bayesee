import pytest
from bayesee.validation import *

def test_all_integers_typical_cases():
    assert all_integers((1,2,3)) == True
    assert all_integers((1,0.5,3)) == False
