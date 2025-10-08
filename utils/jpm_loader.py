# utils/jpm_loader.py
import os
import pandas as pd
from typing import Tuple
from utils.duck_backend import prepare_jpm_view

DATA_FOLDER = os.path.join("Data", "jpm_data")
DEFAULT_PARQUET = "raw_jpm_3.parquet"

# New display attributes
ATTR_COLS = [
    "Time_Horizon","HCO Type","ATC2","ATC3","Company",
    "Formulation 0","Formulation 1","Formulation 2","Formulation 3",
    "Gx/Non-Gx","Brand Name","Market","Molecule Name","Package Description",
    "Launch Year","Launch Date"
]

# New display measure names
MEASURE_COLS = [
    "Bulk","Yen Bn","Cash Div","Cash","DOT Div","DOT","Gram Div","Gram","Patient Day","Unit"
]


def load_jpm_data(parquet_file: str = DEFAULT_PARQUET) -> pd.DataFrame:
    """Pandas loader (mirrors DuckDB view naming)."""
    path = os.path.join(DATA_FOLDER, parquet_file)
    df = pd.read_parquet(path)
    # Replace 'OTHER' with 'NON-GENERIC' in 'generic_name_en'
    df['generic_name_en'].replace('OTHER','NON-GENERIC',inplace=True)
    #df['generic_name_en']=df['generic_name_en'].replace({'OTHER':'NON-GENERIC'})

    # Normalize times
    df['year_month'] = pd.to_datetime(df['year_month'], errors='coerce')
    df['Time_Horizon'] = df['year_month'].dt.strftime('%Y-%m')
    

    df['launch_date_strngth'] = pd.to_datetime(df.get('launch_date_strngth'), errors='coerce')
    df['Launch Date'] = df['launch_date_strngth'].dt.strftime('%Y-%m')
    df['Launch Year'] = df['launch_date_strngth'].apply(lambda x: str(int(x.year)) if pd.notnull(x) else None)

    # Friendly attributes
    df['HCO Type']       = df.get('ims_hco_type_name_en')
    df['Company']        = df.get('company_name_en')
    df['Gx/Non-Gx']      = df.get('generic_name_en')
    df['Brand Name']     = df.get('ims_product_1_name_en')
    df['Market']         = df.get('jpm_market_name')
    df['Molecule Name']  = df.get('molecule_name_en')
    df['Package Description'] = df.get('package_name')

    def _merge_pair(c, n):
        c = "" if pd.isna(c) else str(c).strip()
        n = "" if pd.isna(n) else str(n).strip()
        if c and n: return f"{c} - {n}"
        return c or n or None

    # ATC merges
    df['ATC2'] = [ _merge_pair(c, n) for c, n in zip(df.get('atc2_code'), df.get('atc2_name_en')) ]
    df['ATC3'] = [ _merge_pair(c, n) for c, n in zip(df.get('atc3_code'), df.get('atc3_name_en')) ]

    # Formulation merges
    df['Formulation 0'] = [ _merge_pair(c, n) for c, n in zip(df.get('form0_code'), df.get('form0_name_en')) ]
    df['Formulation 1'] = [ _merge_pair(c, n) for c, n in zip(df.get('form1_code'), df.get('form1_name_en')) ]
    df['Formulation 2'] = [ _merge_pair(c, n) for c, n in zip(df.get('form2_code'), df.get('form2_name_en')) ]
    df['Formulation 3'] = [ _merge_pair(c, n) for c, n in zip(df.get('form3_code'), df.get('form3_name_en')) ]

    # Rename measures to display names
    rename_map = {
        'jpm_bulk_monthly': 'Bulk',
        'jpm_byen_monthly': 'Yen Bn',
        'jpm_cash_div_monthly': 'Cash Div',
        'jpm_cash_monthly': 'Cash',
        'jpm_dot_div_monthly': 'DOT Div',
        'jpm_dot_monthly': 'DOT',
        'jpm_gram_div_monthly': 'Gram Div',
        'jpm_gram_monthly': 'Gram',
        'jpm_pat_day_monthly': 'Patient Day',
        'jpm_unit_monthly': 'Unit',
    }
    df = df.rename(columns=rename_map)

    # Aggregate
    agg_dict = {col: 'sum' for col in MEASURE_COLS if col in df.columns}
    df_agg = df.groupby(ATTR_COLS, dropna=False).agg(agg_dict).reset_index()

    # Optionally keep raw ATC/form columns
    for raw in ['atc2_code','atc2_name_en','atc3_code','atc3_name_en',
                'form0_code','form0_name_en','form1_code','form1_name_en',
                'form2_code','form2_name_en','form3_code','form3_name_en']:
        if raw in df.columns:
            df_agg[raw] = df.groupby(ATTR_COLS, dropna=False)[raw].first().values

    return df_agg


def load_jpm_duck(parquet_file: str = DEFAULT_PARQUET) -> str:
    path = os.path.join(DATA_FOLDER, parquet_file)
    view = prepare_jpm_view(path)
    return view