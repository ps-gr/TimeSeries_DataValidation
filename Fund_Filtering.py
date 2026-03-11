"""
Dependencies:
pip install statsmodels scipy pandas numpy matplotlib pmdarima --break-system-packages -q
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.stats import levene, boxcox, boxcox_normmax
#from statsmodels.tsa.stattools import adfuller, ljungbox
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA

try:
    from pmdarima import auto_arima
    AUTO_ARIMA = True
    print("pmdarima available - using auto_arima")
except:
    AUTO_ARIMA = False
    print("pmdarima not available - using fixed ARIMA(1,d,1)")

print("Libraries loaded successfully\n")


# CONFIGURATION

output_dir = r'C:\Projects_P\VS Workspace\DataFilter\Data\OutputCSV'
fundfile = r'C:\Projects_P\VS Workspace\DataFilter\Data\HistoricalFund\HF Data 450 funds x 100 Months.csv'


# LOAD DATA
print("LOADING DATA")

df = pd.read_csv(fundfile, index_col=0)

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

print(f"Funds loaded: {len(fund_names)}\n")


#HETEROSCEDASTICITY CHECK 
#Levene's test for equal variance,if variance is unstable, apply Box-Cox transformation for variance stabilization 

def check_heteroscedasticity(ts, alpha=0.05):
       
    if len(ts) < 2:
        return ts, True, np.nan, None
    
    try:
        # Split into two halves
        mid = len(ts) // 2
        first_half = ts[:mid]
        second_half = ts[mid:]
        
        # Remove NaN values
        first_half = first_half[~np.isnan(first_half)]
        second_half = second_half[~np.isnan(second_half)]
        
        if len(first_half) < 2 or len(second_half) < 2:
            return ts, True, np.nan, None
        
        # LEVENE'S TEST: Tests if variances are equal
        # H0: Variances are equal (homoscedastic)
        # HA: Variances are different (heteroscedastic)
        levene_stat, levene_pvalue = levene(first_half, second_half)
        
        # If p-value > alpha, we fail to reject H0 → homoscedastic (stable)
        is_homoscedastic = levene_pvalue > alpha
        
        if is_homoscedastic:
            # Variance is stable, return original series
            return ts, True, levene_pvalue, None
        
        else:
            # Variance is unstable, apply Box-Cox transformation
            try:
                # Shift data if needed (Box-Cox requires positive values)
                ts_shifted = ts - np.min(ts) + 1e-10

                ts_transformed, lambda_param = boxcox(ts_shifted)
                
                return ts_transformed, False, levene_pvalue, lambda_param
            
            except:
                return ts, False, levene_pvalue, None
    
    except:
        return ts, True, np.nan, None



# STEP 1: DATA VOLUME FILTER


print("STEP 1: DATA VOLUME FILTERING")

funds_with_data = []
heteroscedasticity_results = []

for i, fund in enumerate(fund_names):
    data_points = np.sum(~np.isnan(fund_data[i]))
    if data_points >= 50:
        funds_with_data.append({
            "index": i,
            "name": fund,
            "type": fund_types[i],
            "data_points": data_points
        })

print(f"Funds passing data requirement: {len(funds_with_data)}\n")

# ============================================================================
# APPLY HETEROSCEDASTICITY CHECK & TRANSFORMATION
# ============================================================================

print("HETEROSCEDASTICITY CHECK & BOX-COX TRANSFORMATION")

funds_after_hetero = []

for fund_info in funds_with_data:
    
    idx = fund_info["index"]
    name = fund_info["name"]
    
    ts = fund_data[idx]
    ts = ts[~np.isnan(ts)]
    
    if len(ts) < 10:
        continue
    
    # Check heteroscedasticity
    ts_transformed, is_homoscedastic, levene_pvalue, lambda_param = check_heteroscedasticity(ts)
    
    heteroscedasticity_results.append({
        "fund_name": name,
        "levene_pvalue": levene_pvalue,
        "is_homoscedastic": is_homoscedastic,
        "box_cox_lambda": lambda_param,
        "transformation_applied": not is_homoscedastic
    })
    
    # Store transformed data
    funds_after_hetero.append({
        **fund_info,
        "ts_original": ts,
        "ts_transformed": ts_transformed,
        "levene_pvalue": levene_pvalue,
        "is_homoscedastic": is_homoscedastic
    })

# Save heteroscedasticity results
hetero_df = pd.DataFrame(heteroscedasticity_results)
hetero_df.to_csv(f'{output_dir}/heteroscedasticity_check.csv', index=False)

print(f"Heteroscedasticity checked: {len(heteroscedasticity_results)} funds")
print(f"  - Homoscedastic (stable variance): {hetero_df['is_homoscedastic'].sum()}")
print(f"  - Heteroscedastic (unstable variance): {(~hetero_df['is_homoscedastic']).sum()}")
print(f"  - Box-Cox transformations applied: {hetero_df['transformation_applied'].sum()}\n")
print(f"Saved: heteroscedasticity_check.csv\n")


# STEP 2: ADF Test

print("STEP 2: ADF")

def find_differencing_order(ts, max_d=2):
    """
    Find minimum differencing order needed for stationarity.
    Tests d=0, d=1, d=2 iteratively
    Stops when stationarity achieved
    Rejects funds needing d>2
    """
    
    current_ts = ts.copy()
    
    for d in range(max_d + 1):
        
        if len(current_ts) < 5:
            return -1, False, np.nan, np.nan
        
        try:
            adf = adfuller(current_ts, autolag='AIC')
            adf_stat = adf[0]
            adf_pvalue = adf[1]
            
            # If stationary, return d
            if adf_pvalue <= 0.05:
                return d, True, adf_pvalue, adf_stat
            
            # Otherwise, difference and continue
            current_ts = np.diff(current_ts)
        
        except:
            return -1, False, np.nan, np.nan
    
    # If reached max_d and still not stationary, reject
    return -1, False, np.nan, np.nan


adf_results = []
funds_after_adf = []

for fund_info in funds_after_hetero:
    
    name = fund_info["name"]
    
    # Use TRANSFORMED data for ADF test
    ts = fund_info["ts_transformed"]
    
    if len(ts) < 10:
        continue
    
    # Iterative differencing test
    d, is_stationary, pvalue, adf_stat = find_differencing_order(ts, max_d=2)
    
    # Discard if d > 2
    if d == -1 or not is_stationary:
        continue
    
    adf_results.append({
        "fund_name": name,
        "adf_stat": adf_stat,
        "pvalue": pvalue,
        "stationary": is_stationary,
        "differencing_order": d
    })
    
    funds_after_adf.append({
        **fund_info,
        "d": d,
        "adf_pvalue": pvalue
    })

print(f"ADF tested funds: {len(adf_results)}")
print(f"Funds with d ≤ 2: {len(funds_after_adf)}\n")

# Save ADF results
adf_df = pd.DataFrame(adf_results)
adf_df.to_csv(f'{output_dir}/step1_adf_test_results.csv', index=False)
print(f"Saved: step2_adf_test_results.csv\n")


# STEP 3: ARIMA MODELING

print("STEP 3: ARIMA MODELING AND AIC RANKING")


model_results = []

for i, fund in enumerate(funds_after_adf):
    
    if (i + 1) % 50 == 0:
        print(f"  Processing fund {i+1}/{len(funds_after_adf)}...")
    
    name = fund["name"]
    d = fund["d"]
    
  
    ts = fund["ts_transformed"]
    
    if len(ts) < 5:
        continue
    
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
                information_criterion='aic',
                suppress_warnings=True
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
            "order": str(order),
            "aic": aic,
            "residuals": residuals
        })
    
    except Exception as e:
        pass

print(f"\nModels successfully fitted: {len(model_results)}\n")

aic_df = pd.DataFrame(model_results)
aic_df.to_csv(f'{output_dir}/step2_aic_ranking.csv', index=False)
print(f"Saved: step3_aic_ranking.csv\n")



# STEP 4: RESIDUAL DIAGNOSTICS

print("STEP 4: RESIDUAL DIAGNOSTICS (STANDARDIZED)")


diagnostics = []

for fund in model_results:
    
    name = fund["fund_name"]
    residuals = np.array(fund["residuals"])
    residuals = residuals[~np.isnan(residuals)]
    
    if len(residuals) < 20:
        continue
    
    try:
        
        residuals_mean = np.mean(residuals)
        residuals_std = np.std(residuals)
        
        if residuals_std == 0:
            continue
        
        std_res = (residuals - residuals_mean) / residuals_std
        
        # SHAPIRO-WILK 
        sh_stat, sh_p = stats.shapiro(std_res)
        sh_pass = sh_p > 0.05
        
        #LJUNG-BOX
        try:
            lb_test = acorr_ljungbox(std_res, lags=10, return_df=True)
            lb_p = lb_test['lb_pvalue'].iloc[-1]
        except:
            lb_test = acorr_ljungbox(std_res, lags=10)
            lb_p = lb_test['lb_pvalue'].values[-1]
        
        lb_pass = lb_p > 0.05
        
        # RUNS TEST 
        signs = np.sign(std_res - np.median(std_res))
        signs = signs[signs != 0]
        
        if len(signs) < 2:
            continue
        
        runs = np.sum(np.diff(signs) != 0) + 1
        
        n1 = np.sum(signs > 0)
        n2 = np.sum(signs < 0)
        
        if n1 == 0 or n2 == 0:
            continue
        
        expected = (2 * n1 * n2) / (n1 + n2) + 1
        var = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / (((n1 + n2) ** 2) * (n1 + n2 - 1))
        
        if var <= 0:
            continue
        
        z = (runs - expected) / np.sqrt(var)
        runs_p = 2 * (1 - stats.norm.cdf(abs(z)))
        runs_pass = runs_p > 0.05
        
        # All tests pass?
        all_pass = sh_pass and lb_pass and runs_pass
        
        diagnostics.append({
            "fund_name": name,
            "aic": fund["aic"],
            "shapiro_p": sh_p,
            "ljungbox_p": lb_p,
            "runs_p": runs_p,
            "pass_all": all_pass
        })
    
    except Exception as e:
        pass

diag_df = pd.DataFrame(diagnostics)

print(f"Diagnostics tests completed: {len(diag_df)} funds\n")


# SELECTION WITH COMPOSITE SCORING

print("FINAL SELECTION - COMPOSITE SCORING")


# Filter to only funds that passed ALL tests
passing = diag_df[diag_df["pass_all"]].copy()

print(f"Funds passing ALL diagnostic tests: {len(passing)}\n")

if len(passing) > 0:
    
    # COMPUTE COMPOSITE SCORE
    passing["composite_score"] = (passing["shapiro_p"] + 
                                  passing["runs_p"] + 
                                  passing["ljungbox_p"]) / 3
    
    # Sort by composite score (higher is better)
    passing = passing.sort_values("composite_score", ascending=False)
    
    # Select top 10
    final = passing.head(10)
    
    print("Top 10 Funds (sorted by composite p-score):")
    print("-" * 80)
    print(final[["fund_name", "composite_score", "shapiro_p", "ljungbox_p", "runs_p"]].to_string(index=False))
    print("-" * 80)
    print(f"\nTop 10 funds selected\n")
else:
    print(" No funds passed all diagnostic tests!")
    print("Selecting top 10 by individual test performance\n")
    final = diag_df.nlargest(10, lambda x: (
        (x['shapiro_p'] > 0.05).astype(int) + 
        (x['ljungbox_p'] > 0.05).astype(int) + 
        (x['runs_p'] > 0.05).astype(int)
    ))


# SAVE RESULTS
print("SAVING RESULTS")

diag_df.to_csv(f'{output_dir}/step3_diagnostic_results.csv', index=False)
print(f"Saved: step4_diagnostic_results.csv")

final.to_csv(f'{output_dir}/FINAL_TOP10_FUNDS.csv', index=False)
print(f"Saved: FINAL_TOP10_FUNDS.csv")


print("Fund Filtering Complete")
