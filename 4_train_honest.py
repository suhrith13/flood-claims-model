"""Step 4: the HONEST models.

Same data, same algorithm, three fixes:
  1. Features limited to what is known before the payment decision.
  2. Evaluation on future years (time split).
  3. Compared against boring baselines, so we know the model adds something.

Also trains a small classifier for "is anything paid at all?".
"""
import numpy as np
from sklearn.metrics import roc_auc_score

from common import (HONEST_FEATURES, OUTPUT_DIR, TARGET, TARGET_ANY,
                    load_model_table, make_classifier, make_regressor,
                    random_split, save_json, score, time_split, top_importances)


def fit_and_score(train, test):
    model = make_regressor().fit(train[HONEST_FEATURES], train[TARGET])
    pred = np.clip(model.predict(test[HONEST_FEATURES]), 0, 1)
    return model, pred, score(test[TARGET], pred, test["buildingDamageAmount"])


def main():
    df = load_model_table()
    results = []

    # ---- baselines, evaluated on the same future test set as the honest model
    train_t, test_t = time_split(df)
    print(f"Time split -> train rows: {len(train_t):,} (<= year of loss in common.py)   test rows: {len(test_t):,}")

    median_pred = np.full(len(test_t), train_t[TARGET].median())
    results.append({"setup": "Baseline: always predict the median ratio",
                    "features": "none", "split": "time (future years)",
                    **score(test_t[TARGET], median_pred, test_t["buildingDamageAmount"])})

    rule_pred = test_t["policy_math_ratio"].fillna(train_t[TARGET].median()).clip(0, 1)
    results.append({"setup": "Baseline: policy math (damage - deductible, capped at coverage)",
                    "features": "3 columns", "split": "time (future years)",
                    **score(test_t[TARGET], rule_pred, test_t["buildingDamageAmount"])})

    # ---- honest features, but still a random split (isolates the effect of the split)
    train_r, test_r = random_split(df)
    _, _, m = fit_and_score(train_r, test_r)
    results.append({"setup": "Honest features, random split",
                    "features": "pre-decision only", "split": "random rows", **m})

    # ---- the real setup: honest features + time split
    model, pred, m = fit_and_score(train_t, test_t)
    results.append({"setup": "Honest features, time split (the real number)",
                    "features": "pre-decision only", "split": "time (future years)", **m})
    for r in results:
        print(f"{r['setup']}: MAE ratio {r['mae_ratio']:.4f}, MAE ${r['mae_dollars']:,.0f}, R2 {r['r2']:.3f}")

    top_importances(model, HONEST_FEATURES).to_csv(OUTPUT_DIR / "importance_honest.csv", index=False)

    # ---- save test predictions for the charts
    out = test_t[["state", "yearOfLoss", "buildingDamageAmount", TARGET]].copy()
    out["predicted"] = pred
    out.to_parquet(OUTPUT_DIR / "honest_test_predictions.parquet")

    # ---- stage 1: is anything paid at all?
    clf = make_classifier().fit(train_t[HONEST_FEATURES], train_t[TARGET_ANY])
    proba = clf.predict_proba(test_t[HONEST_FEATURES])[:, 1]
    auc = float(roc_auc_score(test_t[TARGET_ANY], proba))
    paid_share = float(test_t[TARGET_ANY].mean())
    print(f"'Paid at all' classifier: AUC {auc:.3f} (share of test claims paid: {paid_share:.1%})")

    save_json(OUTPUT_DIR / "honest_metrics.json", {
        "runs": results,
        "classifier": {"auc": auc, "share_paid_in_test": paid_share},
        "n_train_time_split": len(train_t), "n_test_time_split": len(test_t),
    })


if __name__ == "__main__":
    main()
