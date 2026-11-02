"""Step 2: define the label and build features, all in SQL.

Input : data/interim/claims_clean.parquet
Output: data/processed/model_table.parquet

The label
---------
paid_ratio = amount paid on the building claim / assessed building damage,
clipped to [0, 1]. It is the proxy for "approved amount / submitted amount".
paid_any   = 1 if anything was paid.

The prediction moment
---------------------
We predict as if the adjuster's assessment is in but the payment is not yet
decided. That is the same moment as "contractor estimate submitted, carrier has
not answered yet". Only information available at that moment becomes a feature.
"""
import duckdb

from common import CLEAN_FILE, MIN_YEAR, MODEL_TABLE

SQL = f"""
CREATE TABLE model_table AS
WITH base AS (
    SELECT
        *,
        -- deductible code -> dollars (FEMA data dictionary)
        CASE UPPER(buildingDeductibleCode)
            WHEN '0' THEN 500   WHEN '1' THEN 1000  WHEN '2' THEN 2000
            WHEN '3' THEN 3000  WHEN '4' THEN 4000  WHEN '5' THEN 5000
            WHEN '9' THEN 750   WHEN 'A' THEN 10000 WHEN 'B' THEN 15000
            WHEN 'C' THEN 20000 WHEN 'D' THEN 25000 WHEN 'E' THEN 50000
            WHEN 'F' THEN 1250  WHEN 'G' THEN 1500  WHEN 'H' THEN 200
        END AS deductible_amount
    FROM read_parquet('{CLEAN_FILE}')
    WHERE yearOfLoss >= {MIN_YEAR}
      AND buildingDamageAmount > 0
      AND totalBuildingInsuranceCoverage > 0
)
SELECT
    -- ---------- label
    LEAST(amountPaidOnBuildingClaim / buildingDamageAmount, 1.0) AS paid_ratio,
    CAST(amountPaidOnBuildingClaim > 0 AS INTEGER)               AS paid_any,

    -- ---------- kept for splitting and evaluation, never used as features
    CAST(yearOfLoss AS INTEGER) AS yearOfLoss,

    -- ---------- raw numeric features (9999 means "not reported" in FEMA data)
    totalBuildingInsuranceCoverage, buildingDamageAmount, buildingPropertyValue,
    buildingReplacementCost, waterDepth, floodWaterDuration,
    CASE WHEN elevationDifference = 9999 THEN NULL ELSE elevationDifference END
        AS elevationDifference,
    baseFloodElevation, policyCount, latitude, longitude,

    -- ---------- derived numeric features
    deductible_amount,
    CASE WHEN yearOfLoss - YEAR(originalConstructionDate) BETWEEN 0 AND 200
         THEN yearOfLoss - YEAR(originalConstructionDate) END   AS building_age,
    MONTH(dateOfLoss)                                           AS loss_month,
    buildingDamageAmount / totalBuildingInsuranceCoverage       AS damage_to_coverage,
    buildingDamageAmount / NULLIF(buildingPropertyValue, 0)     AS damage_to_value,
    -- what the policy terms alone would pay: damage minus deductible, capped at coverage
    GREATEST(LEAST(buildingDamageAmount - deductible_amount,
                   totalBuildingInsuranceCoverage), 0) / buildingDamageAmount
                                                                AS policy_math_ratio,

    -- ---------- categorical features
    state, ratedFloodZone, occupancyType, causeOfDamage,
    floodCharacteristicsIndicator, basementEnclosureCrawlspaceType,
    buildingDeductibleCode, numberOfFloorsInTheInsuredBuilding,
    condominiumCoverageTypeCode, postFIRMConstructionIndicator,
    elevatedBuildingIndicator, primaryResidenceIndicator, rateMethod,

    -- ---------- leaky columns: exist only after the payment decision.
    -- Kept here ONLY so 3_train_naive.py can show what goes wrong.
    netBuildingPaymentAmount, amountPaidOnContentsClaim, netContentsPaymentAmount,
    amountPaidOnIncreasedCostOfComplianceClaim, netIccPaymentAmount,
    nonPaymentReasonBuilding, nonPaymentReasonContents
FROM base
"""


def main():
    con = duckdb.connect()
    n_in = con.execute(f"SELECT COUNT(*) FROM read_parquet('{CLEAN_FILE}')").fetchone()[0]
    con.execute(SQL)
    con.execute(f"COPY model_table TO '{MODEL_TABLE}' (FORMAT PARQUET)")

    n_out = con.execute("SELECT COUNT(*) FROM model_table").fetchone()[0]
    print(f"Clean rows in: {n_in:,}   modelling rows out: {n_out:,}")
    print(f"Dropped by scope filters (year >= {MIN_YEAR}, damage > 0, coverage > 0): {n_in - n_out:,}")

    clipped = con.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{CLEAN_FILE}')
        WHERE yearOfLoss >= {MIN_YEAR} AND buildingDamageAmount > 0
          AND totalBuildingInsuranceCoverage > 0
          AND amountPaidOnBuildingClaim > buildingDamageAmount""").fetchone()[0]
    print(f"Rows where paid > damage (ratio clipped to 1): {clipped:,}")

    zero, full, mean, med = con.execute("""
        SELECT AVG(CAST(paid_ratio = 0 AS INTEGER)),
               AVG(CAST(paid_ratio >= 0.99 AS INTEGER)),
               AVG(paid_ratio), MEDIAN(paid_ratio)
        FROM model_table""").fetchone()
    print(f"Label paid_ratio: mean {mean:.3f}, median {med:.3f}, "
          f"share paid nothing {zero:.1%}, share paid in full {full:.1%}")
    print("Years in table:", con.execute(
        "SELECT MIN(yearOfLoss), MAX(yearOfLoss) FROM model_table").fetchone())
    print(f"Saved {MODEL_TABLE}")


if __name__ == "__main__":
    main()
