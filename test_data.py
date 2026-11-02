"""Small checks that protect the two things this project is about:
no leakage in the honest feature set, and a time split that never peeks.

Run from the project folder:  pytest -q
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import common  # noqa: E402


def test_honest_and_leaky_features_do_not_overlap():
    assert set(common.HONEST_FEATURES).isdisjoint(common.LEAKY_FEATURES)


def test_honest_features_contain_no_payment_columns():
    for name in common.HONEST_FEATURES:
        lowered = name.lower()
        assert "paid" not in lowered and "payment" not in lowered, name


def test_label_columns_are_not_features():
    assert common.TARGET not in common.HONEST_FEATURES
    assert common.TARGET_ANY not in common.HONEST_FEATURES
    assert "yearOfLoss" not in common.HONEST_FEATURES  # year would let the model memorise events


def test_time_split_never_trains_on_the_future():
    years = list(range(2000, 2024)) * 5
    df = pd.DataFrame({"yearOfLoss": years})
    train, test = common.time_split(df)
    assert train["yearOfLoss"].max() <= common.TRAIN_UNTIL_YEAR
    assert test["yearOfLoss"].min() > common.TRAIN_UNTIL_YEAR
    assert len(train) + len(test) == len(df)


@pytest.mark.skipif(not common.MODEL_TABLE.exists(), reason="run steps 1 and 2 first")
class TestModelTable:
    def setup_method(self):
        self.df = pd.read_parquet(common.MODEL_TABLE)

    def test_label_is_between_zero_and_one(self):
        assert self.df[common.TARGET].between(0, 1).all()

    def test_no_missing_labels(self):
        assert self.df[common.TARGET].notna().all()
        assert self.df[common.TARGET_ANY].isin([0, 1]).all()

    def test_damage_and_coverage_are_positive(self):
        assert (self.df["buildingDamageAmount"] > 0).all()
        assert (self.df["totalBuildingInsuranceCoverage"] > 0).all()

    def test_all_declared_features_exist(self):
        for col in common.HONEST_FEATURES + common.LEAKY_FEATURES:
            assert col in self.df.columns, col

    def test_years_are_in_range(self):
        assert self.df["yearOfLoss"].min() >= common.MIN_YEAR
