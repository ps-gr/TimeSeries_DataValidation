
"""
Dependencies:
pip install statsmodels scipy pandas numpy matplotlib pmdarima

"""

import pandas as pd
import numpy as np
#import warnings
#warnings.filterwarnings('ignore')

from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA

try:
    from pmdarima import auto_arima
    AUTO_ARIMA = True
except:
    AUTO_ARIMA = False

print("Libraries loaded successfully")

#Output file path
output_dir = r'C:\Projects_P\VS Workspace\DataFilter\Data\OutputCSV'

# LOAD DATA
fundfile =  r'C:\Projects_P\VS Workspace\DataFilter\Data\HistoricalFund\HF Data 450 funds x 100 Months.csv'

df = pd.read_csv(f'{fundfile}', index_col=0)


header_row_idx = None
for idx, row in df.iterrows():
    if 'Fund_Name' in str(row.values):
        header_row_idx = idx
        break

if header_row_idx is not None:
    df.columns = df.loc[header_row_idx].values
    df = df.drop(header_row_idx)

df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

for col in df.columns:
    if col not in ['Fund_Name', 'Fund_Type']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

fund_names = df['Fund_Name'].values
fund_types = df['Fund_Type'].values

numeric_columns = [c for c in df.columns if c not in ['Fund_Name', 'Fund_Type']]
fund_data = df[numeric_columns].values

print("Funds loaded:", len(fund_names))

# STEP 1 — DATA VOLUME FILTER
funds_with_data = []
for i, fund in enumerate(fund_names):
    data_points = np.sum(~np.isnan(fund_data[i]))
    if data_points >= 50:
        funds_with_data.append({
            "index": i,
            "name": fund,
            "type": fund_types[i],
            "data_points": data_points
        })

print("Funds passing data requirement:", len(funds_with_data))

# STEP 2 — ADF TEST
adf_results = []
funds_after_adf = []

for fund_info in funds_with_data:

    idx = fund_info["index"]
    name = fund_info["name"]

    ts = fund_data[idx]
    ts = ts[~np.isnan(ts)]

    if len(ts) < 10:
        continue

    try:
        # Run ADF Stationary Tests and Extract test statistic and p-value 
        adf = adfuller(ts, autolag='AIC')
        stat = adf[0]
        pvalue = adf[1]

        is_stationary = pvalue <= 0.05
        d = 0 if is_stationary else 1

        adf_results.append({
            "fund_name": name,
            "adf_stat": stat,
            "pvalue": pvalue,
            "stationary": is_stationary,
            "differencing": d
        })

        funds_after_adf.append({
            **fund_info,
            "d": d
        })

    except:
        pass

print("ADF tested funds:", len(adf_results))

# Save ADF results
adf_df = pd.DataFrame(adf_results)
adf_df.to_csv(f'{output_dir}/step2_adf_test_results.csv', index=False)

# STEP 3 — ARIMA MODELING

print("Step 3: ARIMA Modelling started...")
model_results = []

for fund in funds_after_adf:

    idx = fund["index"]
    name = fund["name"]
    d = fund["d"]

    ts = fund_data[idx]
    ts = ts[~np.isnan(ts)]

    try:

        if AUTO_ARIMA:

            model = auto_arima(
                ts,
                d=d,
                start_p=0,
                start_q=0,
                max_p=4,
                max_q=4,
                seasonal=False,
                information_criterion='aic'
            )

            order = model.order
            aic = model.aic()
            residuals = model.resid()

        else:

            model = ARIMA(ts, order=(1, d, 1)).fit()
            order = (1, d, 1)
            aic = model.aic
            residuals = model.resid

        model_results.append({
            "fund_name": name,
            "type": fund["type"],
            "order": order,
            "aic": aic,
            "residuals": residuals
        })

    except:
        pass

model_results = sorted(model_results, key=lambda x: x["aic"])

print("Models successfully fitted:", len(model_results))
aic_df = pd.DataFrame(model_results)
aic_df.to_csv(f'{output_dir}/step3_aic_ranking.csv', index=False)


# STEP 4 — RESIDUAL DIAGNOSTICS
diagnostics = []

for fund in model_results:

    name = fund["fund_name"]
    residuals = np.array(fund["residuals"])
    residuals = residuals[~np.isnan(residuals)]

    if len(residuals) < 20:
        continue

    try:

        sh_stat, sh_p = stats.shapiro(residuals)
        sh_pass = sh_p > 0.05

        lb = acorr_ljungbox(residuals, lags=[10], return_df=True)
        lb_p = lb["lb_pvalue"].iloc[0]
        lb_pass = lb_p > 0.05

        signs = np.sign(residuals - np.median(residuals))
        signs = signs[signs != 0]

        runs = np.sum(np.diff(signs) != 0) + 1

        n1 = np.sum(signs > 0)
        n2 = np.sum(signs < 0)

        expected = (2*n1*n2)/(n1+n2) + 1
        var = (2*n1*n2*(2*n1*n2-n1-n2))/(((n1+n2)**2)*(n1+n2-1))

        z = (runs - expected)/np.sqrt(var)
        runs_p = 2*(1-stats.norm.cdf(abs(z)))

        runs_pass = runs_p > 0.05

        all_pass = sh_pass and lb_pass and runs_pass

        diagnostics.append({
            "fund_name": name,
            "aic": fund["aic"],
            "shapiro_p": sh_p,
            "ljungbox_p": lb_p,
            "runs_p": runs_p,
            "pass_all": all_pass
        })

    except:
        pass

diag_df = pd.DataFrame(diagnostics)

passing = diag_df[diag_df["pass_all"]]
final = passing.sort_values("aic").head(10)

print("\nTop Funds Selected:", len(final))
print(final[["fund_name","aic"]])

#diag_df.to_csv("diagnostic_results.csv", index=False)
#final.to_csv("FINAL_TOP10_FUNDS.csv", index=False)

diag_df.to_csv(f'{output_dir}/diagnostic_results.csv', index=False)
final.to_csv(f'{output_dir}/FINAL_TOP10_FUNDS.csv', index=False)



print("\nFiles saved: diagnostic_results.csv, FINAL_TOP10_FUNDS.csv")


