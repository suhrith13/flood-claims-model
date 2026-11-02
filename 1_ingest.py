"""Step 1: read the raw FEMA claims file, type it, validate it, split good rows from bad.

Input : data/raw/<one .csv or .parquet file>   (downloaded by hand, see README)
Output: data/interim/claims_clean.parquet       rows that passed every check
        data/interim/claims_quarantine.parquet  rows that failed, with a reason

The file is read as text and converted with TRY_CAST, so a malformed value
becomes NULL instead of crashing the whole load.
"""
import duckdb

from common import (CLEAN_FILE, DATE_RAW, NUMERIC_RAW, QUARANTINE_FILE,
                    REQUIRED_RAW, TEXT_RAW, find_raw_file)


def reader_sql(path):
    """DuckDB expression that reads the file. CSV is read as all-text on purpose."""
    if path.suffix == ".parquet":
        return f"read_parquet('{path}')"
    return f"read_csv('{path}', header=true, all_varchar=true, ignore_errors=true)"


def column_expr(col, kind, available):
    """SQL for one output column. A column missing from the file becomes NULL."""
    if col not in available:
        null_type = {"num": "DOUBLE", "text": "VARCHAR", "date": "DATE"}[kind]
        return f"CAST(NULL AS {null_type}) AS {col}"
    src = f'CAST("{col}" AS VARCHAR)'
    if kind == "num":
        return f"TRY_CAST({src} AS DOUBLE) AS {col}"
    if kind == "text":
        return f"NULLIF(TRIM({src}), '') AS {col}"
    # dates arrive like 2017-08-29T00:00:00.000Z, so keep the first 10 characters
    return f"TRY_CAST(SUBSTR({src}, 1, 10) AS DATE) AS {col}"


def main():
    raw_file = find_raw_file()
    print(f"Reading {raw_file.name}")
    con = duckdb.connect()
    reader = reader_sql(raw_file)

    # --- schema check: required columns must exist, others may drift away
    available = {row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {reader}").fetchall()}
    missing_required = [c for c in REQUIRED_RAW if c not in available]
    if missing_required:
        raise SystemExit(f"Required columns missing from file: {missing_required}")
    wanted = set(NUMERIC_RAW + TEXT_RAW + DATE_RAW)
    missing_optional = sorted(wanted - available)
    if missing_optional:
        print(f"WARNING: {len(missing_optional)} optional columns not in file, filled with NULL: {missing_optional}")

    # --- type every column we use
    exprs = (
        [column_expr(c, "num", available) for c in dict.fromkeys(NUMERIC_RAW)]
        + [column_expr(c, "text", available) for c in dict.fromkeys(TEXT_RAW)]
        + [column_expr(c, "date", available) for c in DATE_RAW]
    )
    con.execute(f"CREATE TABLE typed AS SELECT {', '.join(exprs)} FROM {reader}")
    total = con.execute("SELECT COUNT(*) FROM typed").fetchone()[0]
    print(f"Rows read: {total:,}")

    # --- denied claims: no payment amount recorded, but a denial reason is.
    # Treat the missing payment as 0 so these claims reach the model.
    # (The denial reason only defines the label here, it is never a feature.)
    denied_filter = """amountPaidOnBuildingClaim IS NULL
          AND nonPaymentReasonBuilding IS NOT NULL
          AND buildingDamageAmount IS NOT NULL"""
    denied = con.execute(f"SELECT COUNT(*) FROM typed WHERE {denied_filter}").fetchone()[0]
    con.execute(f"UPDATE typed SET amountPaidOnBuildingClaim = 0 WHERE {denied_filter}")
    print(f"Denied claims with no payment value, set to paid = 0: {denied:,}")

    # --- validation: first failing rule wins, rows with no failure are clean
    con.execute("""
        CREATE TABLE flagged AS
        SELECT *,
          CASE
            WHEN yearOfLoss IS NULL THEN 'missing_year_of_loss'
            WHEN state IS NULL THEN 'missing_state'
            WHEN buildingDamageAmount IS NULL
              OR amountPaidOnBuildingClaim IS NULL
              OR totalBuildingInsuranceCoverage IS NULL THEN 'missing_required_amount'
            WHEN buildingDamageAmount < 0
              OR amountPaidOnBuildingClaim < 0 THEN 'negative_amount'
          END AS reason
        FROM typed
    """)
    con.execute(f"COPY (SELECT * EXCLUDE (reason) FROM flagged WHERE reason IS NULL) "
                f"TO '{CLEAN_FILE}' (FORMAT PARQUET)")
    con.execute(f"COPY (SELECT * FROM flagged WHERE reason IS NOT NULL) "
                f"TO '{QUARANTINE_FILE}' (FORMAT PARQUET)")

    clean = con.execute(f"SELECT COUNT(*) FROM read_parquet('{CLEAN_FILE}')").fetchone()[0]
    print(f"Clean rows kept: {clean:,}")
    print(f"Quarantined rows: {total - clean:,}")
    for reason, n in con.execute(
            "SELECT reason, COUNT(*) FROM flagged WHERE reason IS NOT NULL "
            "GROUP BY reason ORDER BY 2 DESC").fetchall():
        print(f"  {reason}: {n:,}")


if __name__ == "__main__":
    main()
