"""Step 3: the NAIVE model. It looks great and it is wrong.

It makes two mistakes on purpose:
  1. It uses columns that only exist after the payment decision
     (other payment amounts, non-payment reasons).
  2. It tests on randomly chosen rows instead of on future claims.
"""
import sys

from common import (HONEST_FEATURES, LEAKY_FEATURES, OUTPUT_DIR, TARGET,
                    load_model_table, make_regressor, random_split, save_json,
                    score, top_importances)
import numpy as np


def main():
    df = load_model_table()

    # use only leaky columns that actually contain data in this file
    leaky = [c for c in LEAKY_FEATURES if df[c].notna().any()]
    print(f"Leaky columns with data: {leaky}")
    if not leaky:
        sys.exit("No leaky columns have data, so this demo cannot run on this file.")
    features = HONEST_FEATURES + leaky

    train, test = random_split(df)
    print(f"Train rows: {len(train):,}   Test rows: {len(test):,}   (random split)")

    model = make_regressor().fit(train[features], train[TARGET])
    pred = np.clip(model.predict(test[features]), 0, 1)

    metrics = score(test[TARGET], pred, test["buildingDamageAmount"])
    print("Naive model:", {k: round(v, 4) for k, v in metrics.items()})

    top_importances(model, features).to_csv(OUTPUT_DIR / "importance_naive.csv", index=False)
    save_json(OUTPUT_DIR / "naive_metrics.json", {
        "setup": "Naive: leaky features, random split",
        "features": "all + payout-derived",
        "split": "random rows",
        "n_train": len(train), "n_test": len(test), **metrics,
    })


if __name__ == "__main__":
    main()
