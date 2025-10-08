# utils/duck_backend.py
import duckdb
from pathlib import Path
import streamlit as st
import pandas as pd
from typing import List, Optional


def _qi(col: str) -> str:
    """Quote an identifier for SQL (handles spaces, caps, etc)."""
    return '"' + col.replace('"', '""') + '"'


@st.cache_resource
def get_con(persist: bool = True):
    """
    One DuckDB connection per Streamlit server process.
    If persist=True, keep a small on-disk DB to speed cold starts.
    """
    Path("Data").mkdir(parents=True, exist_ok=True)
    db_path = "Data/analogue_cache.duckdb" if persist else ":memory:"
    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads = 4")  # tune to your CPU
    return con


@st.cache_resource
def prepare_jpm_view(parquet_path: str) -> str:
    """
    Create logical VIEWS over the Parquet lazily.

    Normalizes:
      - year_month -> Time_Horizon ('YYYY-MM')
      - launch_date_strngth -> Launch Date ('YYYY-MM') and Launch Year ('YYYY')

    Creates friendly attribute/measure aliases and merged display columns:
      - ATC2        = atc2_code + ' - ' + atc2_name_en
      - ATC3        = atc3_code + ' - ' + atc3_name_en
      - Formulation 0/1/2/3 = formX_code + ' - ' + formX_name_en
      - Gx/Non-Gx   = generic_name_en
      - HCO Type    = ims_hco_type_name_en
      - Brand Name  = ims_product_1_name_en
      - Market      = jpm_market_name
      - Molecule Name = molecule_name_en
      - Package Description = package_name

    Measures renamed:
      jpm_bulk_monthly -> Bulk
      jpm_byen_monthly -> Yen Bn
      jpm_cash_monthly -> Cash
      jpm_cash_div_monthly -> Cash Div
      jpm_dot_monthly -> DOT
      jpm_dot_div_monthly -> DOT Div
      jpm_gram_monthly -> Gram
      jpm_gram_div_monthly -> Gram Div
      jpm_pat_day_monthly -> Patient Day
      jpm_unit_monthly -> Unit

    Returns the aggregated view name to query ("jpm_agg").
    """
    con = get_con()

    # Base normalization view
    con.execute(f"""
        CREATE OR REPLACE VIEW jpm_raw AS
        SELECT
            *,
            -- Normalize year_month -> 'YYYY-MM'
            CASE
                WHEN length(CAST(year_month AS VARCHAR)) = 6
                    THEN strftime(strptime(CAST(year_month AS VARCHAR), '%Y%m'), '%Y-%m')
                WHEN length(CAST(year_month AS VARCHAR)) = 7
                    THEN CAST(year_month AS VARCHAR)
                WHEN length(CAST(year_month AS VARCHAR)) = 10
                    THEN strftime(strptime(CAST(year_month AS VARCHAR), '%Y-%m-%d'), '%Y-%m')
                WHEN length(CAST(year_month AS VARCHAR)) = 19
                    THEN strftime(strptime(CAST(year_month AS VARCHAR), '%Y-%m-%d %H:%M:%S'), '%Y-%m')
                ELSE NULL
            END AS year_month_str,

            -- Normalize launch_date_strngth -> 'YYYY-MM'
            CASE
                WHEN length(CAST(launch_date_strngth AS VARCHAR)) = 6
                    THEN strftime(strptime(CAST(launch_date_strngth AS VARCHAR), '%Y%m'), '%Y-%m')
                WHEN length(CAST(launch_date_strngth AS VARCHAR)) = 7
                    THEN CAST(launch_date_strngth AS VARCHAR)
                WHEN length(CAST(launch_date_strngth AS VARCHAR)) = 10
                    THEN strftime(strptime(CAST(launch_date_strngth AS VARCHAR), '%Y-%m-%d'), '%Y-%m')
                WHEN length(CAST(launch_date_strngth AS VARCHAR)) = 19
                    THEN strftime(strptime(CAST(launch_date_strngth AS VARCHAR), '%Y-%m-%d %H:%M:%S'), '%Y-%m')
                ELSE NULL
            END AS launch_date_str
        FROM parquet_scan('{parquet_path}')
    """)

    # Aggregated view with friendly names + merged columns; exclude ims_product_1_code & package_code
    con.execute("""
        CREATE OR REPLACE VIEW jpm_agg AS
        WITH base AS (
            SELECT
                -- Friendly attribute aliases
                year_month_str                                        AS "Time_Horizon",
                ims_hco_type_name_en                                  AS "HCO Type",

                -- ATC merges ("code - name")
                atc2_code                                             AS atc2_code,
                atc2_name_en                                          AS atc2_name_en,
                CASE
                    WHEN atc2_code IS NOT NULL AND atc2_name_en IS NOT NULL THEN CAST(atc2_code AS VARCHAR) || ' - ' || CAST(atc2_name_en AS VARCHAR)
                    WHEN atc2_code IS NOT NULL THEN CAST(atc2_code AS VARCHAR)
                    WHEN atc2_name_en IS NOT NULL THEN CAST(atc2_name_en AS VARCHAR)
                    ELSE NULL
                END                                                   AS "ATC2",

                atc3_code                                             AS atc3_code,
                atc3_name_en                                          AS atc3_name_en,
                CASE
                    WHEN atc3_code IS NOT NULL AND atc3_name_en IS NOT NULL THEN CAST(atc3_code AS VARCHAR) || ' - ' || CAST(atc3_name_en AS VARCHAR)
                    WHEN atc3_code IS NOT NULL THEN CAST(atc3_code AS VARCHAR)
                    WHEN atc3_name_en IS NOT NULL THEN CAST(atc3_name_en AS VARCHAR)
                    ELSE NULL
                END                                                   AS "ATC3",

                company_name_en                                       AS "Company",

                -- Formulation merges
                form0_code                                            AS form0_code,
                form0_name_en                                         AS form0_name_en,
                CASE
                    WHEN form0_code IS NOT NULL AND form0_name_en IS NOT NULL THEN CAST(form0_code AS VARCHAR) || ' - ' || CAST(form0_name_en AS VARCHAR)
                    WHEN form0_code IS NOT NULL THEN CAST(form0_code AS VARCHAR)
                    WHEN form0_name_en IS NOT NULL THEN CAST(form0_name_en AS VARCHAR)
                    ELSE NULL
                END                                                   AS "Formulation 0",

                form1_code                                            AS form1_code,
                form1_name_en                                         AS form1_name_en,
                CASE
                    WHEN form1_code IS NOT NULL AND form1_name_en IS NOT NULL THEN CAST(form1_code AS VARCHAR) || ' - ' || CAST(form1_name_en AS VARCHAR)
                    WHEN form1_code IS NOT NULL THEN CAST(form1_code AS VARCHAR)
                    WHEN form1_name_en IS NOT NULL THEN CAST(form1_name_en AS VARCHAR)
                    ELSE NULL
                END                                                   AS "Formulation 1",

                form2_code                                            AS form2_code,
                form2_name_en                                         AS form2_name_en,
                CASE
                    WHEN form2_code IS NOT NULL AND form2_name_en IS NOT NULL THEN CAST(form2_code AS VARCHAR) || ' - ' || CAST(form2_name_en AS VARCHAR)
                    WHEN form2_code IS NOT NULL THEN CAST(form2_code AS VARCHAR)
                    WHEN form2_name_en IS NOT NULL THEN CAST(form2_name_en AS VARCHAR)
                    ELSE NULL
                END                                                   AS "Formulation 2",

                form3_code                                            AS form3_code,
                form3_name_en                                         AS form3_name_en,
                CASE
                    WHEN form3_code IS NOT NULL AND form3_name_en IS NOT NULL THEN CAST(form3_code AS VARCHAR) || ' - ' || CAST(form3_name_en AS VARCHAR)
                    WHEN form3_code IS NOT NULL THEN CAST(form3_code AS VARCHAR)
                    WHEN form3_name_en IS NOT NULL THEN CAST(form3_name_en AS VARCHAR)
                    ELSE NULL
                END                                                   AS "Formulation 3",

                generic_name_en                                       AS "Gx/Non-Gx",
                
                ims_product_1_name_en                                 AS "Brand Name",
                jpm_market_name                                       AS "Market",
                molecule_name_en                                      AS "Molecule Name",
                package_name                                          AS "Package Description",
                launch_date_str                                       AS "Launch Date",
                SUBSTR(launch_date_str, 1, 4)                         AS "Launch Year",

                -- Raw measures (we'll SUM and alias below)
                jpm_bulk_monthly, jpm_byen_monthly, jpm_cash_monthly, jpm_cash_div_monthly,
                jpm_dot_monthly, jpm_dot_div_monthly, jpm_gram_monthly, jpm_gram_div_monthly,
                jpm_pat_day_monthly, jpm_unit_monthly
            FROM jpm_raw
        )
        SELECT
            "Time_Horizon","HCO Type","ATC2","ATC3","Company",
            "Formulation 0","Formulation 1","Formulation 2","Formulation 3",
            "Gx/Non-Gx","Brand Name","Market","Molecule Name","Package Description",
            "Launch Year","Launch Date",

            -- Friendly measure names
            SUM(jpm_bulk_monthly)       AS "Bulk",
            SUM(jpm_byen_monthly)       AS "Yen Bn",
            SUM(jpm_cash_div_monthly)   AS "Cash Div",
            SUM(jpm_cash_monthly)       AS "Cash",
            SUM(jpm_dot_div_monthly)    AS "DOT Div",
            SUM(jpm_dot_monthly)        AS "DOT",
            SUM(jpm_gram_div_monthly)   AS "Gram Div",
            SUM(jpm_gram_monthly)       AS "Gram",
            SUM(jpm_pat_day_monthly)    AS "Patient Day",
            SUM(jpm_unit_monthly)       AS "Unit",

            -- Keep raw ATC & form codes/names (optional; helpful downstream)
            atc2_code, atc2_name_en, atc3_code, atc3_name_en,
            form0_code, form0_name_en, form1_code, form1_name_en,
            form2_code, form2_name_en, form3_code, form3_name_en
        FROM base
        GROUP BY ALL
    """)

    return "jpm_agg"


# ----------------------- Convenience full fetch -----------------------

def get_full_filtered_df(view_name: str, where_sql: str, cols: List[str], limit: Optional[int] = None) -> pd.DataFrame:
    """Run the saved WHERE without preview limit for exports/analytics."""
    con = get_con()
    # Quote all column identifiers (spaces supported)
    if cols:
        sel = ", ".join('"' + c.replace('"','""') + '"' for c in cols)
    else:
        sel = "*"
    sql = f"SELECT {sel} FROM {view_name} {where_sql}"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return con.execute(sql).df()
