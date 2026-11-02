# What breaks when you train on negotiated payouts

A small, end-to-end ML pipeline on public flood-insurance claims. It shows why a
model trained naively on "what was finally paid" looks excellent and is wrong,
and what to do instead.

Everything runs locally from one downloaded file. No API calls, no cloud.

## The problem

In claims work, the label is rarely ground truth. The amount that gets paid is the
outcome of a negotiation, shaped by policy terms, who the adjuster is, which carrier
it is, and the paperwork that accumulates along the way. Train directly on it and
three things go wrong:

1. **The label encodes the process, not the truth.** The model learns how payouts
   usually come out, including any systematic under- or over-payment.
2. **Leakage.** Fields created during or after the decision (other payment amounts,
   denial reasons) explain the label almost perfectly. Offline metrics look great while
   the model is just reading the answer.
3. **Optimistic evaluation.** Randomly shuffled rows put near-identical claims
   (same event, same area, same period) on both sides of the split.

This repo builds the naive model on purpose, shows the problem, and fixes it.

## The data

FEMA's public **OpenFEMA "FIMA NFIP Redacted Claims v2"** dataset (about 2.7 million
flood-insurance claims). It is a proxy for a restoration-claims dataset, not the same
thing:

| In a restoration claim | In this project |
|---|---|
| Contractor's submitted estimate | `buildingDamageAmount` (adjuster-assessed damage) |
| Carrier-approved amount | `amountPaidOnBuildingClaim` |
| What I predict | `paid_ratio` = paid / assessed damage, clipped to 0..1 |

**Prediction moment:** the adjuster's assessment is in, the payment is not yet decided.
Only information available at that moment is allowed as a feature.

## Pipeline

```
data/raw/<file>.csv|parquet
   |  1_ingest.py        type every column, validate, quarantine bad rows
   v
data/interim/claims_clean.parquet  (+ claims_quarantine.parquet with a reason per row)
   |  2_build_dataset.py label + features, in SQL (DuckDB)
   v
data/processed/model_table.parquet
   |  3_train_naive.py   leaky features + random split   -> looks great, is wrong
   |  4_train_honest.py  pre-decision features + time split + baselines
   |  5_evaluate.py      results table + 3 charts
   v
outputs/charts
```

## How to run

```bash
# 1. install
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. get the data (one manual download, see "Data" above)
#    Download the CSV or Parquet file for "FIMA NFIP Redacted Claims v2" from
#    https://www.fema.gov/openfema-data-page/fima-nfip-redacted-claims-v2
#    and put that single file in data/raw/

# 3. run
python 1_ingest.py
python 2_build_dataset.py
python 3_train_naive.py
python 4_train_honest.py
python 5_evaluate.py

# 4. optional checks
pytest -q
```

For a quick trial run, set `SAMPLE_ROWS = 300_000` in `common.py`.

## Results

| Setup | Features | Split | MAE (ratio) | MAE ($ per claim) | R2 |
|---|---|---|---|---|---|
| Baseline: always predict the median ratio | none | time (future years) | 0.157 | 6,511 | -0.151 |
| Baseline: policy math (damage - deductible, capped at coverage) | 3 columns | time (future years) | 0.052 | 2,336 | 0.649 |
| Naive: leaky features, random split | all + payout-derived | random rows | 0.010 | 634 | 0.993 |
| Honest features, random split | pre-decision only | random rows | 0.055 | 2,563 | 0.808 |
| Honest features, time split (the real number) | pre-decision only | time (future years) | 0.079 | 4,105 | 0.646 |

'Paid at all' classifier (honest features, time split): AUC 0.955; 94.7% of test claims were paid something.

Charts are in `outputs/`: `importance_naive_vs_honest.png`, `calibration.png`,
`error_by_state.png`.

**How to read the table**

- The naive row is the trap. Its error is far lower than any honest setup because it
  can see payout-derived columns. That score would not survive contact with a live claim.
- "Honest, random split" vs "Honest, time split" isolates the effect of how you evaluate.
- The policy-math baseline (damage minus deductible, capped at coverage) is the bar
  the model has to beat. Much of any claim payout is plain arithmetic from the policy;
  the interesting part is what the arithmetic leaves unexplained.



## Design decisions

- **Gradient boosting (LightGBM).** Tabular data with mixed numeric and categorical
  columns and missing values. It handles all three natively and trains in minutes on a laptop.
- **Label as a ratio, not dollars.** Dollar payouts scale with property size, so the model
  would mostly learn "big house, big payout". The ratio asks the real question: what share
  of the assessed damage gets paid.
- **Time split.** Train on loss years up to `TRAIN_UNTIL_YEAR`, test on later years. In
  production a model only ever scores claims from after its training data.
- **`yearOfLoss` is not a feature.** With a time split the test years are never seen in
  training, and a year feature only invites the model to memorise event-specific effects.
- **Baselines first.** A constant prediction and a policy-math rule. If the model cannot
  beat the rule, it has not learned anything useful.
- **Bad rows are quarantined, not dropped silently.** Each gets a reason code and the
  counts are printed on every run.
- **Schema drift survives.** Required columns must exist; optional ones that go missing
  are filled with NULL and a warning is printed. Everything is read as text and converted
  with `TRY_CAST`, so one malformed value becomes NULL instead of crashing the load.
- **Simple on purpose.** Small scripts, shared settings in `common.py`, features defined in
  SQL. A new teammate can read the whole thing in an afternoon.

## Limitations (honest ones)

- **It is a proxy.** FEMA payouts are adjusted claims under a government program. There is
  no contractor-versus-carrier negotiation in this data, and no per-line-item detail.
- **Adjuster-assessed fields are treated as known at prediction time.** Damage amount,
  property value and water depth are assessed by an adjuster. That matches the "estimate
  submitted" moment, but it is a modelling choice I made, and it can be argued.
- **Denied claims are recognised through their denial reason.** Many denied claims have no
  payment value recorded but do carry a building non-payment reason. For those (with a
  damage amount present) the payment is set to 0 so they stay in the data. The denial reason
  is used only to define the label, never as a feature. Rows with a missing payment and no
  denial reason are still quarantined.
- **"Paid in full" means "paid at least the assessed damage".** About 16% of rows have a
  payment above the recorded damage, which is clipped to a ratio of 1. FEMA records damage
  as actual cash value, so I suspect replacement-cost settlements explain part of this, but I
  have not confirmed it.
- **Labels still reflect past practice.** Even the honest model learns how payouts have
  been decided, not what they should have been. Fixing that needs an audited set of claims
  reviewed by experts, which this public data does not have.
- **`waterDepth` has mixed units** (FEMA notes some values are in feet). It is left as is.
- **Not production code.** No serving layer, no monitoring, no experiment tracking.

## What I would do with real claims data

1. Model at line-item level, with submitted and approved values per item.
2. Add carrier, adjuster and franchise as features, and split by group so a carrier's
   claims never appear on both sides.
3. Build a small audited gold set of expert-reviewed claims and evaluate against it
   alongside the negotiated outcomes.
4. Serve the model behind an endpoint, track experiments, and monitor live drift weekly.
