import numpy as np
import pytest

from datasets.taxonomy import load_taxonomy


def test_taxonomy_loads_expected_shape():
    tax = load_taxonomy()
    assert len(tax.harm_classes) == 9
    assert len(tax.confusable_classes) == 14
    assert tax.num_classes == 23
    # harm classes come first, confusables after
    assert tax.all_classes[:9] == tax.harm_classes
    assert tax.all_classes[9:] == tax.confusable_classes


def test_harm_ordering_is_sex_vio_gmb():
    tax = load_taxonomy()
    assert tax.harm_classes[0] == "sex_moan"
    assert tax.harm_categories == ("sex", "vio", "gmb")
    assert tax.category_of("vio_gunshot") == "vio"
    assert tax.category_of("gmb_table") == "gmb"
    assert tax.category_of("asmr") == "confusable"


def test_is_harm_and_index():
    tax = load_taxonomy()
    assert tax.is_harm("sex_moan") is True
    assert tax.is_harm("door") is False
    assert tax.index_of("sex_moan") == 0
    assert tax.index_of(tax.all_classes[-1]) == tax.num_classes - 1


def test_encode_multihot():
    tax = load_taxonomy()
    vec = tax.encode(["sex_moan", "asmr"])
    assert vec.shape == (23,)
    assert vec.sum() == 2.0
    assert vec[tax.index_of("sex_moan")] == 1.0
    assert vec[tax.index_of("asmr")] == 1.0


def test_encode_empty_is_all_zero():
    tax = load_taxonomy()
    assert np.count_nonzero(tax.encode([])) == 0


def test_unknown_class_raises():
    tax = load_taxonomy()
    with pytest.raises(KeyError):
        tax.index_of("nonexistent")
