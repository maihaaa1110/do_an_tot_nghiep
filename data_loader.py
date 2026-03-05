"""
data_loader.py
──────────────
Load toàn bộ dữ liệu và model từ file thật.
Tất cả số đã kiểm tra khớp 100% với output notebook 09.

════════════════════════════════════════════════════════════════
CẤU TRÚC THƯ MỤC  (đặt cạnh app.py)
════════════════════════════════════════════════════════════════
  data/
  ├── panel/
  │   ├── data_clean_b7_full.csv          ← PRIMARY: 2256 obs (2014–2025)
  │   │                                      45 cols: raw + ratios + flags +
  │   │                                      FEATURES_ML + has_debt + target
  │   └── data_with_gap_and_target.csv    ← SUPPLEMENT: 2600 obs (2013–2025)
  │                                          114 cols: thêm ROE, DER, CR,
  │                                          CFO_TA, median_* (37 cols), _gap
  ├── predictions/
  │   ├── predictions_2025.csv            ← 188 obs Q1–Q4/2025 per firm
  │   └── firm_predictions_2025.csv       ← 47 firms × Q1%/Q2%/Q3%/Q4%/Mean%
  ├── evaluation/
  │   ├── cv_results.csv                  ← 48 rows (6 models × 8 folds)
  │   ├── bootstrap_ci.csv                ← 4 metrics, CI 95%
  │   ├── subgroup_analysis.csv           ← 8 subgroups
  │   ├── delong_test.csv                 ← 5 pairs RF vs others
  │   ├── nemenyi_test.csv                ← 15 pairs
  │   ├── threshold_analysis.csv          ← 3 strategies
  │   └── feature_importance.csv          ← 13 features, SHAP mean|φ|
  ├── reference/
  │   └── Kiem_tra_quy_mo_50_DN.xlsx      ← 50 firms, Lớn/Nhỏ classification
  └── archive/                            ← không dùng runtime
      ├── cleaning_log_b7.csv
      ├── firm_zero_analysis.csv
      ├── vif_report_b7.csv
      ├── best_params_per_fold.csv
      └── feature_ablation.csv

  models/
  ├── best_model.pkl      ← dict (xem PKL STRUCTURE bên dưới)
  └── all_models.pkl      ← dict (xem PKL STRUCTURE bên dưới)

════════════════════════════════════════════════════════════════
PKL STRUCTURE  (từ notebook 09, Cell 7)
════════════════════════════════════════════════════════════════
  best_model.pkl = {
    'pipeline'      : sklearn Pipeline(StandardScaler → RF),
    'model_name'    : 'RF',
    'features'      : ['ROA','NPM','TATO','ITO','DAR','QR','FCF_TA',
                       'ICR','has_debt','SIZE','ROA_lag1','ROA_lag4',
                       'REV_GROWTH_YOY'],
    'winsor_bounds' : {'ICR':(lo,hi), 'ITO':(lo,hi), ...},  ← tính trên train
    'fill_zero_vars': ['ICR','ITO','NPM','REV_GROWTH_YOY'],
    'cv_auc_mean'   : 0.8114,
    'cv_auc_std'    : 0.0480,
    'final_params'  : {'clf__max_depth':3, 'clf__max_features':0.5,
                       'clf__min_samples_leaf':9, 'clf__n_estimators':406},
    'train_period'  : '2014-2024',
    'saved_at'      : '2025-...',
  }

  all_models.pkl = {
    'models'        : {'RF':Pipeline, 'GBM':Pipeline, 'XGB':Pipeline,
                       'LR_L1':Pipeline, 'LR_L2':Pipeline, 'SVM':Pipeline},
    'cv_summary'    : pd.DataFrame,
    'cv_df'         : pd.DataFrame,
    'features'      : list[str],
    'winsor_bounds' : dict,
  }

════════════════════════════════════════════════════════════════
SỐ THAM CHIẾU  (từ notebook 09 outputs — dùng để kiểm tra)
════════════════════════════════════════════════════════════════
  Train: 2068 obs (2014–2024) | Target pos: 52.2%
  Holdout: 188 obs (2025)     | Target pos: 39.4%
  Best model: RF | CV AUC: 0.8114 ± 0.0480
  Holdout AUC: 0.6923 | PR-AUC: 0.5806 | F1: 0.5594 | Acc: 0.6649
  Youden threshold: 0.419
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))

# Panel
PANEL_CLEAN_CSV = os.path.join(ROOT, "data", "panel", "data_clean_b7_full.csv")
PANEL_FULL_CSV  = os.path.join(ROOT, "data", "panel", "data_with_gap_and_target.csv")

# Predictions
PRED_OBS_CSV    = os.path.join(ROOT, "data", "predictions", "predictions_2025.csv")
PRED_FIRM_CSV   = os.path.join(ROOT, "data", "predictions", "firm_predictions_2025.csv")

# Evaluation
CV_CSV          = os.path.join(ROOT, "data", "evaluation", "cv_results.csv")
BOOTSTRAP_CSV   = os.path.join(ROOT, "data", "evaluation", "bootstrap_ci.csv")
SUBGROUP_CSV    = os.path.join(ROOT, "data", "evaluation", "subgroup_analysis.csv")
DELONG_CSV      = os.path.join(ROOT, "data", "evaluation", "delong_test.csv")
NEMENYI_CSV     = os.path.join(ROOT, "data", "evaluation", "nemenyi_test.csv")
THRESHOLD_CSV   = os.path.join(ROOT, "data", "evaluation", "threshold_analysis.csv")
FEAT_IMP_CSV    = os.path.join(ROOT, "data", "evaluation", "feature_importance.csv")

# Reference
FIRM_SIZE_XLSX  = os.path.join(ROOT, "data", "reference", "Kiem_tra_quy_mo_50_DN.xlsx")

# Models
BEST_MODEL_PKL  = os.path.join(ROOT, "models", "best_model.pkl")
ALL_MODELS_PKL  = os.path.join(ROOT, "models", "all_models.pkl")


# ── Constants (từ notebook 09) ─────────────────────────────────────────────────

TICKERS_50 = [
    'API','C21','CCL','D11','D2D','DIG','DRH','DTA','DXG','FDC',
    'HAR','HDC','HDG','HLD','HQC','IDJ','IDV','IJC','ITA','ITC',
    'KBC','KDH','LGL','LHG','NBB','NDN','NLG','NTL','NVT','PDR',
    'PTL','PV2','PVL','PVR','PXL','QCG','RCL','SCR','SJS','SZC',
    'SZL','TDH','TIG','TIP','TIX','V21','VCR','VIC','VPH','VRC',
]
TICKERS_CLEAN = [t for t in TICKERS_50 if t not in ['PVR', 'VCR', 'PV2']]  # 47 firms

# FINAL_FEATURES từ notebook 09 Cell 2 — thứ tự phải khớp pipeline
FEATURES_ML = [
    'ROA', 'NPM', 'TATO', 'ITO', 'DAR', 'QR', 'FCF_TA', 'ICR',
    'has_debt', 'SIZE', 'ROA_lag1', 'ROA_lag4', 'REV_GROWTH_YOY',
]

# FILL_ZERO từ notebook 09 Cell 2
FILL_ZERO_VARS = ['ICR', 'ITO', 'NPM', 'REV_GROWTH_YOY']

# WINSOR_VARS từ notebook 09 Cell 2
WINSOR_VARS = ['ICR', 'ITO', 'NPM', 'QR', 'FCF_TA', 'ROE', 'ROA',
               'TATO', 'DER', 'REV_GROWTH_YOY']

# Signal bins từ notebook 09 Cell 8
# pd.cut(prob, bins=[0,.3,.45,.55,.7,1], labels=[...])
SIGNAL_BINS   = [0, 0.30, 0.45, 0.55, 0.70, 1.0]
SIGNAL_LABELS = ['Strong_Sell', 'Sell', 'Neutral', 'Buy', 'Strong_Buy']

# Ratio cols cho industry stats
# (data_clean có đủ tất cả trừ ROE/DER/CR/CFO_TA — lấy từ panel_full)
RATIO_COLS = [
    'ROA', 'ROE', 'NPM', 'TATO', 'ITO', 'DAR', 'DER', 'ICR', 'CR', 'QR',
    'CFO_TA', 'FCF_TA', 'FCF', 'SIZE', 'REV_GROWTH_YOY', 'REV_GROWTH_QOQ',
]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _require(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không tìm thấy: {path}\n"
            "Xem cấu trúc thư mục trong docstring data_loader.py"
        )
    return path


def _csv(path: str) -> pd.DataFrame:
    return pd.read_csv(_require(path), encoding='utf-8-sig')


# Cache
_cache: dict = {}


def clear_cache() -> None:
    """Xóa cache, buộc reload file lần sau."""
    _cache.clear()


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING — đúng pipeline notebook 09 Cell 2
# Chỉ dùng cho data mới đưa vào predict, KHÔNG cần cho data đã clean
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_for_inference(df_new: pd.DataFrame,
                             winsor_bounds: dict,
                             fill_zero_vars: list | None = None,
                             features: list | None = None) -> pd.DataFrame:
    """
    Áp dụng đúng 6 bước preprocessing từ notebook 09 Cell 2 cho data mới.

    Bước 1: abs(cogs), abs(interest_expense)
    Bước 2: ITO = cogs / avg_inventory  (nan nếu avg_inv ≤ 0)
    Bước 3: ICR = operating_profit / interest_expense  (nan nếu ie ≤ 0)
    Bước 4: has_debt = (interest_expense > 0).astype(int)
    Bước 5: Clip theo winsor_bounds từ train  ← tránh leakage
            ROA_lag1, ROA_lag4 dùng bounds['ROA']
    Bước 6: Fill NaN — FILL_ZERO_VARS → 0, còn lại → median cột

    Parameters
    ----------
    df_new        : DataFrame raw (cần có các cột tài chính cơ bản)
    winsor_bounds : dict từ best_model.pkl['winsor_bounds']
    fill_zero_vars: list, default FILL_ZERO_VARS
    features      : list, default FEATURES_ML

    Returns
    -------
    pd.DataFrame  chỉ chứa `features` cols, không NaN
    """
    if fill_zero_vars is None:
        fill_zero_vars = FILL_ZERO_VARS
    if features is None:
        features = FEATURES_ML

    d = df_new.copy()

    # Bước 1
    if 'cogs' in d.columns:
        d['cogs'] = d['cogs'].abs()
    if 'interest_expense' in d.columns:
        d['interest_expense'] = d['interest_expense'].abs()

    # Bước 2
    if {'cogs', 'avg_inventory'}.issubset(d.columns):
        d['ITO'] = np.where(d['avg_inventory'] > 0,
                            d['cogs'] / d['avg_inventory'], np.nan)

    # Bước 3
    if {'operating_profit', 'interest_expense'}.issubset(d.columns):
        d['ICR'] = np.where(d['interest_expense'] > 0,
                            d['operating_profit'] / d['interest_expense'], np.nan)

    # Bước 4
    if 'interest_expense' in d.columns:
        d['has_debt'] = (d['interest_expense'] > 0).astype(int)

    # Bước 5: Winsorize theo bounds từ train
    for v, (lo, hi) in winsor_bounds.items():
        if v in d.columns:
            d[v] = d[v].clip(lo, hi)
    if 'ROA' in winsor_bounds:
        lo_r, hi_r = winsor_bounds['ROA']
        for lv in ['ROA_lag1', 'ROA_lag4']:
            if lv in d.columns:
                d[lv] = d[lv].clip(lo_r, hi_r)

    # Bước 6: Impute
    for v in fill_zero_vars:
        if v in d.columns and v in features:
            d[v] = d[v].fillna(0)
    for col in features:
        if col in d.columns and d[col].isna().any():
            d[col] = d[col].fillna(d[col].median())

    return d[features]


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def get_best_model(use_cache: bool = True) -> dict:
    """
    Load models/best_model.pkl → dict.

    Keys được dùng trong app:
      model_obj['pipeline']       → .predict_proba(X)[:, 1]
      model_obj['features']       → list 13 features đúng thứ tự
      model_obj['winsor_bounds']  → dùng cho preprocess_for_inference()
      model_obj['fill_zero_vars'] → dùng cho preprocess_for_inference()
      model_obj['model_name']     → 'RF'
      model_obj['cv_auc_mean']    → 0.8114
      model_obj['cv_auc_std']     → 0.0480
      model_obj['final_params']   → hyperparams tốt nhất
    """
    if use_cache and 'best_model' in _cache:
        return _cache['best_model']

    with open(_require(BEST_MODEL_PKL), 'rb') as f:
        obj = pickle.load(f)

    if use_cache:
        _cache['best_model'] = obj
    return obj


def get_all_models(use_cache: bool = True) -> dict:
    """
    Load models/all_models.pkl → dict.

    Keys:
      obj['models']        → {'RF': Pipeline, 'GBM': Pipeline, ...}
      obj['cv_summary']    → DataFrame với AUC_mean, AUC_std, ...
      obj['cv_df']         → raw cv_results DataFrame
      obj['features']      → list 13 features
      obj['winsor_bounds'] → dict
    """
    if use_cache and 'all_models' in _cache:
        return _cache['all_models']

    with open(_require(ALL_MODELS_PKL), 'rb') as f:
        obj = pickle.load(f)

    if use_cache:
        _cache['all_models'] = obj
    return obj


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS — PANEL
# ══════════════════════════════════════════════════════════════════════════════

def get_panel(use_cache: bool = True) -> pd.DataFrame:
    """
    Load data_clean_b7_full.csv — file PRIMARY.

    Nội dung: 2256 obs × 45 cols (2014–2025, 47 firms)
      - 2068 obs train (2014–2024) | target pos: 52.2%
      - 188 obs holdout (2025)     | target pos: 39.4%

    Có sẵn: tất cả FEATURES_ML, has_debt, target, ROA_next
    Không có: ROE, DER, CR, CFO_TA  ← dùng get_panel_enriched() nếu cần

    Dùng cho: hiển thị lịch sử firm, filter train/holdout, predict.
    """
    if use_cache and 'panel' in _cache:
        return _cache['panel']

    df = _csv(PANEL_CLEAN_CSV)

    if use_cache:
        _cache['panel'] = df
    return df


def get_panel_enriched(use_cache: bool = True) -> pd.DataFrame:
    """
    data_clean_b7_full  MERGE  data_with_gap_and_target
    → 2256 obs × ~82 cols

    Thêm từ data_with_gap:
      - ROE, DER, CR, CFO_TA       ← đủ RATIO_COLS cho industry stats
      - 37 cột median_*            ← industry benchmark có sẵn
      - _gap cols                  ← firm vs industry gap

    Filter data_with_gap trước khi merge:
      - Bỏ năm 2013
      - Bỏ firm PVR, VCR, PV2
    """
    if use_cache and 'panel_enriched' in _cache:
        return _cache['panel_enriched']

    clean = get_panel(use_cache=use_cache)

    full = _csv(PANEL_FULL_CSV)
    full = full[full['year'] != 2013].copy()
    full = full[~full['firm'].isin(['PVR', 'VCR', 'PV2'])].copy()

    # Chỉ lấy cols bổ sung — tránh duplicate
    extra_cols = ['firm', 'year', 'quarter',
                  'ROE', 'DER', 'CR', 'CFO_TA'] + \
                 [c for c in full.columns if c.startswith('median_')] + \
                 [c for c in full.columns if c.endswith('_gap')]

    extra = full[[c for c in extra_cols if c in full.columns]].copy()

    df = clean.merge(extra, on=['firm', 'year', 'quarter'], how='left')

    if use_cache:
        _cache['panel_enriched'] = df
    return df


def get_train_data(use_cache: bool = True) -> pd.DataFrame:
    """
    Panel 2014–2024 từ data_clean — đúng tập train của model.
    2068 obs | target pos: 52.2%
    """
    if use_cache and 'train' in _cache:
        return _cache['train']

    df = get_panel(use_cache=use_cache)
    df = df[df['year'] <= 2024].copy().reset_index(drop=True)

    if use_cache:
        _cache['train'] = df
    return df


def get_holdout_data(use_cache: bool = True) -> pd.DataFrame:
    """
    Panel 2025 từ data_clean — holdout set.
    188 obs | target pos: 39.4%
    Không được dùng trong bất kỳ bước training nào.
    """
    if use_cache and 'holdout' in _cache:
        return _cache['holdout']

    df = get_panel(use_cache=use_cache)
    df = df[df['year'] == 2025].copy().reset_index(drop=True)

    if use_cache:
        _cache['holdout'] = df
    return df


def get_firm_data(ticker: str, enriched: bool = False) -> pd.DataFrame:
    """
    Lọc data của một firm cụ thể, sort theo (year, quarter).

    Parameters
    ----------
    ticker   : mã firm, e.g. 'VIC'
    enriched : True → dùng panel_enriched (có ROE, DER, median_*)
               False → dùng panel_clean (nhanh hơn, đủ cho FEATURES_ML)
    """
    df = get_panel_enriched() if enriched else get_panel()
    return (
        df[df['firm'] == ticker]
        .sort_values(['year', 'quarter'])
        .reset_index(drop=True)
    )


def get_industry_stats(use_cache: bool = True) -> pd.DataFrame:
    """
    Median / mean / std của RATIO_COLS theo (year, quarter).
    Dùng panel_enriched để có đủ ROE, DER, CR, CFO_TA.
    Chỉ tính trên train period (2014–2024).
    """
    if use_cache and 'industry_stats' in _cache:
        return _cache['industry_stats']

    df = get_panel_enriched(use_cache=use_cache)
    df = df[df['year'] <= 2024]
    cols = [c for c in RATIO_COLS if c in df.columns]
    stats = (
        df.groupby(['year', 'quarter'])[cols]
        .agg(['median', 'mean', 'std'])
        .reset_index()
    )

    if use_cache:
        _cache['industry_stats'] = stats
    return stats


def get_industry_medians(use_cache: bool = True) -> pd.DataFrame:
    """
    Trả về các cột median_* có sẵn từ data_with_gap_and_target.
    37 cột benchmark ngành, tính sẵn — không cần tính lại.

    Columns: firm, year, quarter + median_ROA, median_ROE, ...
    Dùng cho: so sánh firm vs industry trên từng quý.
    """
    if use_cache and 'industry_medians' in _cache:
        return _cache['industry_medians']

    df = get_panel_enriched(use_cache=use_cache)
    median_cols = [c for c in df.columns if c.startswith('median_')]
    result = df[['firm', 'year', 'quarter'] + median_cols].copy()

    if use_cache:
        _cache['industry_medians'] = result
    return result


def get_firm_size_map(use_cache: bool = True) -> pd.DataFrame:
    """
    Load Kiem_tra_quy_mo_50_DN.xlsx.

    Columns: firm | total_assets_bn | log_TA | Quy_Mo_Nghien_Cuu
    Quy_Mo_Nghien_Cuu: 'Lớn' (11 firms) | 'Nhỏ' (39 firms)
    """
    if use_cache and 'firm_size' in _cache:
        return _cache['firm_size']

    df = pd.read_excel(_require(FIRM_SIZE_XLSX))
    df = df.rename(columns={'symbol': 'firm'})

    if use_cache:
        _cache['firm_size'] = df
    return df


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS — PREDICTIONS 2025
# ══════════════════════════════════════════════════════════════════════════════

def get_predictions_obs(use_cache: bool = True) -> pd.DataFrame:
    """
    predictions_2025.csv — 188 obs × 10 cols.
    Columns: firm, year, quarter, target, prob_roa_cao, prob_pct,
             pred_50, pred_J, correct_50, signal
    Acc_50 = 0.6649 (khớp notebook: Acc=0.6649)
    """
    if use_cache and 'pred_obs' in _cache:
        return _cache['pred_obs']

    df = _csv(PRED_OBS_CSV)

    if use_cache:
        _cache['pred_obs'] = df
    return df


def get_predictions_firm(use_cache: bool = True) -> pd.DataFrame:
    """
    firm_predictions_2025.csv — 47 firms.
    Columns: firm, Q1%, Q2%, Q3%, Q4%, Mean%, Actual_rate, Acc_50
    Sorted by Mean% descending (LHG=86.1% đứng đầu).
    """
    if use_cache and 'pred_firm' in _cache:
        return _cache['pred_firm']

    df = _csv(PRED_FIRM_CSV)

    if use_cache:
        _cache['pred_firm'] = df
    return df


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS — MODEL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def get_cv_results(use_cache: bool = True) -> pd.DataFrame:
    """
    cv_results.csv — 48 rows (6 models × 8 folds).
    Columns: model, fold, val_year, n_train, n_val, AUC, PR_AUC, F1, Accuracy
    RF mean AUC = 0.8114 (khớp notebook).
    """
    if use_cache and 'cv' in _cache:
        return _cache['cv']
    df = _csv(CV_CSV)
    if use_cache:
        _cache['cv'] = df
    return df


def get_bootstrap_ci(use_cache: bool = True) -> pd.DataFrame:
    """
    bootstrap_ci.csv — 4 metrics, CI 95% (B=1000).
    Columns: metric, mean, ci_low, ci_high
    AUC: 0.6909 [0.6135, 0.7631] (khớp notebook).
    """
    if use_cache and 'bootstrap' in _cache:
        return _cache['bootstrap']
    df = _csv(BOOTSTRAP_CSV)
    if use_cache:
        _cache['bootstrap'] = df
    return df


def get_subgroup_analysis(use_cache: bool = True) -> pd.DataFrame:
    """
    subgroup_analysis.csv — 8 subgroups.
    Columns: subgroup, n, pos_rate, AUC, F1, Acc
    Gồm: Large/Mid/Small cap, Pre/COVID/Post-COVID, has_debt.
    """
    if use_cache and 'subgroup' in _cache:
        return _cache['subgroup']
    df = _csv(SUBGROUP_CSV)
    if use_cache:
        _cache['subgroup'] = df
    return df


def get_delong_test(use_cache: bool = True) -> pd.DataFrame:
    """
    delong_test.csv — 5 pairs (RF vs mỗi model còn lại).
    Columns: model_a, model_b, auc_a, auc_b, diff, z, p, sig
    Kết quả: 0/5 pair có p < 0.05 (khớp notebook).
    """
    if use_cache and 'delong' in _cache:
        return _cache['delong']
    df = _csv(DELONG_CSV)
    if use_cache:
        _cache['delong'] = df
    return df


def get_nemenyi_test(use_cache: bool = True) -> pd.DataFrame:
    """
    nemenyi_test.csv — 15 pairs.
    Columns: model_a, model_b, mean_rank_a, mean_rank_b, rank_diff, CD, significant, better
    Friedman p=0.0045 | RF rank=2.25 tốt nhất | 1/15 pair significant (khớp notebook).
    """
    if use_cache and 'nemenyi' in _cache:
        return _cache['nemenyi']
    df = _csv(NEMENYI_CSV)
    if use_cache:
        _cache['nemenyi'] = df
    return df


def get_threshold_analysis(use_cache: bool = True) -> pd.DataFrame:
    """
    threshold_analysis.csv — 3 strategies.
    Columns: Strategy, Threshold, Precision, Recall, F1, Acc, N_positive
    Youden's J = 0.419, F1=0.605, Acc=0.681 (khớp notebook).
    """
    if use_cache and 'threshold' in _cache:
        return _cache['threshold']
    df = _csv(THRESHOLD_CSV)
    if use_cache:
        _cache['threshold'] = df
    return df


def get_feature_importance(use_cache: bool = True) -> pd.DataFrame:
    """
    feature_importance.csv — 13 features.
    Columns: feature, importance, type  (type = 'SHAP mean|phi|')
    ROA_lag1=0.1083, ROA=0.0887, ROA_lag4=0.0587 (khớp notebook).
    Sorted by importance descending.
    """
    if use_cache and 'feat_imp' in _cache:
        return _cache['feat_imp']
    df = _csv(FEAT_IMP_CSV)
    df = df.sort_values('importance', ascending=False).reset_index(drop=True)
    if use_cache:
        _cache['feat_imp'] = df
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def predict_proba_batch(df: pd.DataFrame,
                        model_obj: dict | None = None,
                        already_preprocessed: bool = True) -> np.ndarray:
    """
    Dự báo P(ROA_cao) cho DataFrame.

    Parameters
    ----------
    df                    : DataFrame có đủ FEATURES_ML
    model_obj             : dict từ get_best_model() — None → tự load
    already_preprocessed  : True  → df đã sạch (từ data_clean), dùng trực tiếp
                            False → chạy preprocess_for_inference() trước

    Returns
    -------
    np.ndarray shape (n,) — P(ROA_cao) ∈ [0, 1]
    """
    if model_obj is None:
        model_obj = get_best_model()

    pipeline = model_obj['pipeline']
    features = model_obj['features']

    if already_preprocessed:
        missing = [c for c in features if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame thiếu features: {missing}")
        X = df[features]
    else:
        X = preprocess_for_inference(
            df,
            winsor_bounds=model_obj['winsor_bounds'],
            fill_zero_vars=model_obj.get('fill_zero_vars', FILL_ZERO_VARS),
            features=features,
        )

    return pipeline.predict_proba(X)[:, 1]


def predict_proba_single(row: pd.Series,
                         model_obj: dict | None = None) -> float:
    """
    Dự báo P(ROA_cao) cho 1 observation đã preprocessed.

    Parameters
    ----------
    row       : pd.Series chứa đủ FEATURES_ML (từ data_clean)
    model_obj : dict từ get_best_model()

    Returns
    -------
    float P(ROA_cao)
    """
    if model_obj is None:
        model_obj = get_best_model()
    X = pd.DataFrame([row[model_obj['features']]])
    return float(model_obj['pipeline'].predict_proba(X)[0, 1])


def prob_to_signal(prob: float) -> str:
    """
    Chuyển xác suất → signal label.
    Bins từ notebook 09 Cell 8:
      ≥0.70 → Strong_Buy | ≥0.55 → Buy | ≥0.45 → Neutral
      ≥0.30 → Sell       | <0.30 → Strong_Sell
    """
    if   prob >= 0.70: return 'Strong_Buy'
    elif prob >= 0.55: return 'Buy'
    elif prob >= 0.45: return 'Neutral'
    elif prob >= 0.30: return 'Sell'
    else:              return 'Strong_Sell'


def probs_to_signals(probs: np.ndarray) -> pd.Categorical:
    """Vectorised prob_to_signal dùng pd.cut — đúng notebook 09."""
    return pd.cut(probs, bins=SIGNAL_BINS, labels=SIGNAL_LABELS)


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE — shortcuts cho Streamlit pages
# ══════════════════════════════════════════════════════════════════════════════

def get_model_summary() -> pd.DataFrame:
    """
    AUC / F1 / Accuracy mean theo model từ cv_results.csv.
    Sorted by AUC desc. RF đứng đầu AUC=0.8114.
    """
    cv = get_cv_results()
    return (
        cv.groupby('model')[['AUC', 'F1', 'Accuracy']]
        .mean()
        .round(4)
        .sort_values('AUC', ascending=False)
        .reset_index()
    )


def get_youden_threshold() -> float:
    """
    Ngưỡng Youden's J từ threshold_analysis.csv.
    Trả về 0.419 (khớp notebook).
    """
    ta = get_threshold_analysis()
    row = ta[ta['Strategy'].str.contains("Youden", case=False, na=False)]
    return float(row.iloc[0]['Threshold']) if not row.empty else 0.419


def get_best_model_info() -> dict:
    """
    Trả về thông tin nhanh về best model mà không cần load pipeline.
    Keys: model_name, cv_auc_mean, cv_auc_std, final_params, train_period
    """
    obj = get_best_model()
    return {
        'model_name':   obj.get('model_name', 'RF'),
        'cv_auc_mean':  obj.get('cv_auc_mean', 0.8114),
        'cv_auc_std':   obj.get('cv_auc_std',  0.0480),
        'final_params': obj.get('final_params', {}),
        'train_period': obj.get('train_period', '2014-2024'),
    }


def data_source_badge() -> str:
    """Badge HTML trạng thái 3 file chính."""
    checks = {
        "Panel CSV":    PANEL_CLEAN_CSV,
        "Model PKL":    BEST_MODEL_PKL,
        "Predictions":  PRED_OBS_CSV,
    }
    parts = []
    all_ok = True
    for label, path in checks.items():
        ok = os.path.exists(path)
        all_ok = all_ok and ok
        parts.append(("" if ok else "❌ ") + label)
    color = "badge-green" if all_ok else "badge-yellow"
    return f"<span class='{color}'>" + " &nbsp;|&nbsp; ".join(parts) + "</span>"


def check_all_files() -> dict[str, bool]:
    """
    Kiểm tra sự tồn tại tất cả file.
    Dùng để debug: status = check_all_files()
    """
    registry = {
        "panel_clean":       PANEL_CLEAN_CSV,
        "panel_full":        PANEL_FULL_CSV,
        "predictions_obs":   PRED_OBS_CSV,
        "predictions_firm":  PRED_FIRM_CSV,
        "cv_results":        CV_CSV,
        "bootstrap_ci":      BOOTSTRAP_CSV,
        "subgroup":          SUBGROUP_CSV,
        "delong":            DELONG_CSV,
        "nemenyi":           NEMENYI_CSV,
        "threshold":         THRESHOLD_CSV,
        "feature_imp":       FEAT_IMP_CSV,
        "firm_size":         FIRM_SIZE_XLSX,
        "best_model":        BEST_MODEL_PKL,
        "all_models":        ALL_MODELS_PKL,
    }
    status = {name: os.path.exists(path) for name, path in registry.items()}
    missing = [k for k, v in status.items() if not v]
    if missing:
        warnings.warn(f"[data_loader] Thiếu {len(missing)} file: {missing}", stacklevel=2)
    return status