"""Shared settings and helpers.

Every script imports from here, so paths, feature lists and split rules
live in exactly one place.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"
CLEAN_FILE = ROOT / "data" / "interim" / "claims_clean.parquet"
QUARANTINE_FILE = ROOT / "data" / "interim" / "claims_quarantine.parquet"
MODEL_TABLE = ROOT / "data" / "processed" / "model_table.parquet"
OUTPUT_DIR = ROOT / "outputs"

for _folder in (RAW_DIR, CLEAN_FILE.parent, MODEL_TABLE.parent, OUTPUT_DIR):
    _folder.mkdir(parents=True, exist_ok=True)  # so a fresh unzip just works

# ------------------------------------------------------------- settings
MIN_YEAR = 2000          # older records have many missing fields
TRAIN_UNTIL_YEAR = 2017  # time split: train on <= 2017, test on 2018 and later
SAMPLE_ROWS = None       # set to e.g. 300_000 for a quick trial run
RANDOM_STATE = 42

TARGET = "paid_ratio"    # amount paid on building / assessed building damage, 0..1
TARGET_ANY = "paid_any"  # 1 if anything was paid on the building claim

# --------------------------------------------- raw columns (FEMA OpenFEMA)
# Columns we need to exist. If one is missing the file is not what we expect.
REQUIRED_RAW = [
    "yearOfLoss", "state", "buildingDamageAmount",
    "amountPaidOnBuildingClaim", "totalBuildingInsuranceCoverage",
]
NUMERIC_RAW = [
    "yearOfLoss", "buildingDamageAmount", "amountPaidOnBuildingClaim",
    "totalBuildingInsuranceCoverage", "buildingPropertyValue",
    "buildingReplacementCost", "waterDepth", "floodWaterDuration",
    "elevationDifference", "baseFloodElevation", "policyCount",
    "latitude", "longitude",
    # payout-derived fields, only used by the deliberately leaky model:
    "netBuildingPaymentAmount", "amountPaidOnContentsClaim",
    "netContentsPaymentAmount", "amountPaidOnIncreasedCostOfComplianceClaim",
    "netIccPaymentAmount",
]
TEXT_RAW = [
    "state", "ratedFloodZone", "occupancyType", "causeOfDamage",
    "floodCharacteristicsIndicator", "basementEnclosureCrawlspaceType",
    "buildingDeductibleCode", "numberOfFloorsInTheInsuredBuilding",
    "condominiumCoverageTypeCode", "postFIRMConstructionIndicator",
    "elevatedBuildingIndicator", "primaryResidenceIndicator", "rateMethod",
    "nonPaymentReasonBuilding", "nonPaymentReasonContents",
]
DATE_RAW = ["dateOfLoss", "originalConstructionDate"]

# ----------------------------------------------------------- feature sets
# "Honest" = known once the adjuster's assessment is in, BEFORE payment is decided.
HONEST_NUMERIC = [
    "totalBuildingInsuranceCoverage", "buildingDamageAmount",
    "buildingPropertyValue", "buildingReplacementCost", "waterDepth",
    "floodWaterDuration", "elevationDifference", "baseFloodElevation",
    "policyCount", "latitude", "longitude",
    "deductible_amount", "building_age", "loss_month",
    "damage_to_coverage", "damage_to_value", "policy_math_ratio",
]
HONEST_CATEGORICAL = [
    "state", "ratedFloodZone", "occupancyType", "causeOfDamage",
    "floodCharacteristicsIndicator", "basementEnclosureCrawlspaceType",
    "buildingDeductibleCode", "numberOfFloorsInTheInsuredBuilding",
    "condominiumCoverageTypeCode", "postFIRMConstructionIndicator",
    "elevatedBuildingIndicator", "primaryResidenceIndicator", "rateMethod",
]
HONEST_FEATURES = HONEST_NUMERIC + HONEST_CATEGORICAL

# "Leaky" = only exist AFTER the payment decision has been made.
LEAKY_NUMERIC = [
    "netBuildingPaymentAmount", "amountPaidOnContentsClaim",
    "netContentsPaymentAmount", "amountPaidOnIncreasedCostOfComplianceClaim",
    "netIccPaymentAmount",
]
LEAKY_CATEGORICAL = ["nonPaymentReasonBuilding", "nonPaymentReasonContents"]
LEAKY_FEATURES = LEAKY_NUMERIC + LEAKY_CATEGORICAL

ALL_CATEGORICAL = HONEST_CATEGORICAL + LEAKY_CATEGORICAL


# ---------------------------------------------------------------- helpers
def find_raw_file():
    """Return the one .csv or .parquet file the user put in data/raw/."""
    files = sorted(list(RAW_DIR.glob("*.csv")) + list(RAW_DIR.glob("*.parquet")))
    if not files:
        sys.exit(f"No .csv or .parquet file found in {RAW_DIR}. See README step 2.")
    if len(files) > 1:
        print(f"Several files found, using the first: {files[0].name}")
    return files[0]


def load_model_table():
    """Read the modelling table and mark categorical columns for LightGBM."""
    if not MODEL_TABLE.exists():
        sys.exit("model_table.parquet not found. Run 1_ingest.py and 2_build_dataset.py first.")
    df = pd.read_parquet(MODEL_TABLE)
    if SAMPLE_ROWS and len(df) > SAMPLE_ROWS:
        df = df.sample(SAMPLE_ROWS, random_state=RANDOM_STATE).reset_index(drop=True)
    for col in ALL_CATEGORICAL:
        df[col] = df[col].astype("category")
    return df


def random_split(df):
    """80/20 split on shuffled rows. Easy, and too optimistic for claims data."""
    return train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)


def time_split(df):
    """Train on the past, test on the future. This is how the model is really used."""
    train = df[df["yearOfLoss"] <= TRAIN_UNTIL_YEAR]
    test = df[df["yearOfLoss"] > TRAIN_UNTIL_YEAR]
    if len(train) == 0 or len(test) == 0:
        sys.exit("Time split left train or test empty. Adjust TRAIN_UNTIL_YEAR in common.py.")
    return train, test


def make_regressor():
    return LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63,
                         random_state=RANDOM_STATE, verbose=-1)


def make_classifier():
    return LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=63,
                          random_state=RANDOM_STATE, verbose=-1)


def score(y_true, y_pred, damage):
    """Three numbers: error in ratio points, error in dollars, and R-squared."""
    y_true, y_pred, damage = np.asarray(y_true), np.asarray(y_pred), np.asarray(damage)
    return {
        "mae_ratio": float(mean_absolute_error(y_true, y_pred)),
        "mae_dollars": float(np.mean(np.abs(y_pred - y_true) * damage)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def top_importances(model, feature_names):
    """Feature importance by total gain, largest first."""
    gain = model.booster_.feature_importance(importance_type="gain")
    out = pd.DataFrame({"feature": feature_names, "gain": gain})
    out["share"] = out["gain"] / out["gain"].sum()
    return out.sort_values("gain", ascending=False).reset_index(drop=True)


def save_json(path, obj):
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
