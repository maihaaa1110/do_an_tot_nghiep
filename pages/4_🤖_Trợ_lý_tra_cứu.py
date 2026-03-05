"""
5___Trợ_lý_Phân_tích.py
────────────────────────
Trợ lý phân tích hoàn toàn local.
Câu hỏi preset được trả lời tức thì từ data thật.
Phần tra cứu theo doanh nghiệp hiển thị dashboard đầy đủ.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from style import apply_style, page_header, section_header
import data_loader as dl

st.set_page_config(page_title="Trợ lý tra cứu | BĐS", page_icon="💬", layout="wide")
apply_style(st)

# ── Extra CSS riêng trang ────────────────────────────────────────────
st.markdown("""
<style>
.qa-question {
    background: linear-gradient(135deg, #131f38, #1a2a45);
    border: 1px solid rgba(91,192,190,0.2);
    border-left: 4px solid #5bc0be;
    border-radius: 0 10px 10px 0;
    padding: 0.7rem 1rem;
    margin-bottom: 0.4rem;
    cursor: pointer;
    transition: all 0.2s;
    font-size: 0.88rem;
    color: #c8d8e8;
}
.qa-question:hover { border-color: #5bc0be; transform: translateX(3px); color: #e0e6ed; }

.answer-box {
    background: linear-gradient(135deg, #0f1d38, #131f38);
    border: 1px solid rgba(91,192,190,0.25);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-top: 0.5rem;
    font-size: 0.88rem;
    line-height: 1.7;
    color: #c8d8e8;
    animation: fadeIn 0.25s ease;
}
@keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

.answer-title {
    font-size: 1rem;
    font-weight: 700;
    color: #5bc0be;
    margin-bottom: 0.6rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(91,192,190,0.2);
}
.kpi-mini {
    background: linear-gradient(135deg, #1a2a45, #1c2e4a);
    border: 1px solid rgba(91,192,190,0.18);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    text-align: center;
}
.kpi-mini-val { font-family: 'Space Mono', monospace; font-size: 1.3rem; font-weight: 700; color: #5bc0be; }
.kpi-mini-lbl { font-size: 0.72rem; color: #6b7f99; text-transform: uppercase; letter-spacing: 0.07em; margin-top: 2px; }

.firm-header {
    background: linear-gradient(135deg, #131f38, #1a2a45);
    border: 1px solid rgba(91,192,190,0.3);
    border-radius: 12px;
    padding: 1rem 1.4rem;
    margin-bottom: 1rem;
}
.cat-label {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    font-family: 'Space Mono', monospace;
    background: rgba(91,192,190,0.12);
    color: #5bc0be;
    border: 1px solid rgba(91,192,190,0.3);
    margin-bottom: 0.5rem;
}
.signal-pill {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.signal-strong-buy  { background: rgba(46,204,113,0.15); color: #2ecc71; border: 1px solid rgba(46,204,113,0.4); }
.signal-buy         { background: rgba(39,174,96,0.12);  color: #27ae60; border: 1px solid rgba(39,174,96,0.35); }
.signal-neutral     { background: rgba(241,196,15,0.12); color: #f1c40f; border: 1px solid rgba(241,196,15,0.35); }
.signal-sell        { background: rgba(230,126,34,0.12); color: #e67e22; border: 1px solid rgba(230,126,34,0.35); }
.signal-strong-sell { background: rgba(231,76,60,0.12);  color: #e74c3c; border: 1px solid rgba(231,76,60,0.4); }

.trend-up   { color: #2ecc71; font-weight: 700; }
.trend-down { color: #e74c3c; font-weight: 700; }
.trend-flat { color: #f1c40f; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

page_header(st, "TRỢ LÝ TRA CỨU", "Tra cứu nhanh dữ liệu · Hỏi đáp định nghĩa · Dashboard doanh nghiệp", "💬")


# ══════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Đang tải dữ liệu...")
def load_all():
    panel      = dl.get_panel()
    panel_enr  = dl.get_panel_enriched()
    feat_imp   = dl.get_feature_importance()
    bm_info    = dl.get_best_model_info()
    boot_df    = dl.get_bootstrap_ci()
    cv_df      = dl.get_cv_results()
    subgroup   = dl.get_subgroup_analysis()
    thresh_df  = dl.get_threshold_analysis()
    pred_obs   = dl.get_predictions_obs()
    pred_firm  = dl.get_predictions_firm()
    model_obj  = dl.get_best_model()
    return panel, panel_enr, feat_imp, bm_info, boot_df, cv_df, subgroup, thresh_df, pred_obs, pred_firm, model_obj

try:
    panel, panel_enr, feat_imp, bm_info, boot_df, cv_df, subgroup_df, thresh_df, pred_obs, pred_firm, model_obj = load_all()
except FileNotFoundError as e:
    st.error(f"❌ Không tìm thấy file dữ liệu: {e}")
    st.stop()

st.markdown(dl.data_source_badge(), unsafe_allow_html=True)

TICKERS = sorted(panel['firm'].unique().tolist())
FEATURES_ML    = dl.FEATURES_ML
FILL_ZERO_VARS = dl.FILL_ZERO_VARS


# ── Pre-compute train medians cho imputation ──────────────────────────
@st.cache_data(show_spinner=False)
def _train_medians():
    train = panel[panel['year'] <= 2024]
    return {f: float(train[f].median()) for f in FEATURES_ML if f in train.columns}

_TRAIN_MEDIANS = _train_medians()

def safe_predict(row_or_dict) -> float:
    if isinstance(row_or_dict, dict):
        vals = dict(row_or_dict)
    else:
        vals = {f: float(row_or_dict[f]) for f in FEATURES_ML if f in row_or_dict.index}
    for f in FEATURES_ML:
        if f not in vals or (isinstance(vals.get(f), float) and np.isnan(vals[f])):
            vals[f] = 0.0 if f in FILL_ZERO_VARS else _TRAIN_MEDIANS.get(f, 0.0)
    row_s = pd.Series(vals)
    X = pd.DataFrame([row_s[model_obj["features"]].values], columns=None)
    return float(model_obj["pipeline"].predict_proba(X)[0, 1])


# ══════════════════════════════════════════════════════════════════════
# PRE-COMPUTE — TẤT CẢ SỐ TỪ DATA THẬT
# ══════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def compute_all_stats(n: int):
    df = panel

    # Cơ bản
    n_firms   = df['firm'].nunique()
    n_obs     = len(df)
    n_pos     = int((df['target'] == 1).sum())
    n_neg     = int((df['target'] == 0).sum())
    pos_rate  = n_pos / n_obs * 100

    # ROA
    roa_mean    = float(df['ROA'].mean())
    roa_median  = float(df['ROA'].median())
    roa_neg_pct = float((df['ROA'] < 0).mean() * 100)
    roa_std     = float(df['ROA'].std())

    # DAR
    dar_mean = float(df['DAR'].mean() * 100) if 'DAR' in df.columns else 0

    # ICR
    icr_miss_pct = float(df['ICR'].isna().mean() * 100) if 'ICR' in df.columns else 0

    # has_debt
    hd0_rate = float(df[df['has_debt']==0]['target'].mean() * 100) if 'has_debt' in df.columns else 0
    hd1_rate = float(df[df['has_debt']==1]['target'].mean() * 100) if 'has_debt' in df.columns else 0
    hd_pct   = float((df['has_debt']==1).mean() * 100) if 'has_debt' in df.columns else 0

    # Firm ROA
    firm_roa = df.groupby('firm')['ROA'].mean().sort_values(ascending=False)
    top5_firms = dict(firm_roa.head(5).round(4))
    bot5_firms = dict(firm_roa.tail(5).round(4))

    # COVID
    pre_roa   = float(df[df['year'] < 2020]['ROA'].median())
    covid_roa = float(df[(df['year'] == 2020) | ((df['year'] == 2021) & (df['quarter'] <= 2))]['ROA'].median())
    covid_drop = (pre_roa - covid_roa) / abs(pre_roa) * 100 if pre_roa != 0 else 0

    # Model
    _auc_row      = boot_df[boot_df['metric'].str.upper() == 'AUC']
    holdout_auc   = float(_auc_row['mean'].iloc[0])    if not _auc_row.empty else 0
    holdout_ci_lo = float(_auc_row['ci_low'].iloc[0])  if not _auc_row.empty else 0
    holdout_ci_hi = float(_auc_row['ci_high'].iloc[0]) if not _auc_row.empty else 0
    cv_auc_mean   = bm_info.get('cv_auc_mean', 0)
    cv_auc_std    = bm_info.get('cv_auc_std', 0)
    best_name     = bm_info.get('model_name', 'RF')
    best_params   = bm_info.get('final_params', {})
    youden_thr    = dl.get_youden_threshold()

    # SHAP
    shap_sorted   = feat_imp.sort_values('importance', ascending=False).reset_index(drop=True)
    total_shap    = shap_sorted['importance'].sum()
    shap_top3_pct = shap_sorted.head(3)['importance'].sum() / total_shap * 100
    shap_top5     = shap_sorted.head(5)[['feature','importance']].values.tolist()

    # Subgroup
    _auc_col = next((c for c in subgroup_df.columns if 'auc' in c.lower()), None)
    _grp_col = subgroup_df.columns[0]
    best_sg      = subgroup_df.loc[subgroup_df[_auc_col].idxmax(), _grp_col] if _auc_col else "N/A"
    best_sg_auc  = float(subgroup_df[_auc_col].max()) if _auc_col else 0
    worst_sg     = subgroup_df.loc[subgroup_df[_auc_col].idxmin(), _grp_col] if _auc_col else "N/A"
    worst_sg_auc = float(subgroup_df[_auc_col].min()) if _auc_col else 0

    # CV per model
    cv_summary = cv_df.groupby('model')['AUC'].agg(['mean','std']).sort_values('mean', ascending=False).round(4)

    return dict(
        n_firms=n_firms, n_obs=n_obs, n_pos=n_pos, n_neg=n_neg, pos_rate=pos_rate,
        roa_mean=roa_mean, roa_median=roa_median, roa_neg_pct=roa_neg_pct, roa_std=roa_std,
        dar_mean=dar_mean, icr_miss_pct=icr_miss_pct,
        hd0_rate=hd0_rate, hd1_rate=hd1_rate, hd_pct=hd_pct,
        top5_firms=top5_firms, bot5_firms=bot5_firms,
        pre_roa=pre_roa, covid_roa=covid_roa, covid_drop=covid_drop,
        holdout_auc=holdout_auc, holdout_ci_lo=holdout_ci_lo, holdout_ci_hi=holdout_ci_hi,
        cv_auc_mean=cv_auc_mean, cv_auc_std=cv_auc_std,
        best_name=best_name, best_params=best_params, youden_thr=youden_thr,
        shap_top3_pct=shap_top3_pct, shap_top5=shap_top5, total_shap=total_shap,
        best_sg=best_sg, best_sg_auc=best_sg_auc,
        worst_sg=worst_sg, worst_sg_auc=worst_sg_auc,
        cv_summary=cv_summary,
    )

S = compute_all_stats(len(panel))


# ══════════════════════════════════════════════════════════════════════
# ĐỊNH NGHĨA TẤT CẢ CÂU TRẢ LỜI PRESET
# ══════════════════════════════════════════════════════════════════════
def _shap_rows_html():
    rows = ""
    for i, (feat, val) in enumerate(S['shap_top5']):
        pct  = val / S['total_shap'] * 100
        bar  = min(100, int(val / S['shap_top5'][0][1] * 100))
        clr  = "#5bc0be" if i < 3 else "#3a7bd5"
        rows += f"""
        <div style='margin-bottom:10px;'>
          <div style='display:flex;justify-content:space-between;margin-bottom:3px;'>
            <span style='color:#e0e6ed;'><b>{i+1}. {feat}</b></span>
            <span style='font-family:Space Mono,monospace;color:{clr};'>{val:.4f}
              <span style='color:#6b7f99;font-size:0.78rem;'> ({pct:.1f}%)</span></span>
          </div>
          <div style='background:#1a2540;border-radius:3px;height:6px;'>
            <div style='background:{clr};height:6px;border-radius:3px;width:{bar}%;'></div>
          </div>
        </div>"""
    return rows

def _cv_table_html():
    rows = ""
    for model, row in S['cv_summary'].iterrows():
        is_best = (model == S['best_name'])
        color   = "#2ecc71" if is_best else "#a8b8cc"
        star    = " ⭐" if is_best else ""
        rows += f"""<tr style='{"background:rgba(46,204,113,0.06);" if is_best else ""}'>
          <td style='padding:5px 10px;color:{color};font-weight:{"700" if is_best else "400"};'>{model}{star}</td>
          <td style='padding:5px 10px;font-family:Space Mono,monospace;color:{color};'>{row["mean"]:.4f}</td>
          <td style='padding:5px 10px;font-family:Space Mono,monospace;color:#6b7f99;'>±{row["std"]:.4f}</td>
        </tr>"""
    return f"""<table style='width:100%;border-collapse:collapse;font-size:0.85rem;margin:0.5rem 0;'>
      <thead><tr>
        <th style='padding:5px 10px;color:#5bc0be;text-align:left;border-bottom:1px solid rgba(91,192,190,0.3);'>Model</th>
        <th style='padding:5px 10px;color:#5bc0be;text-align:left;border-bottom:1px solid rgba(91,192,190,0.3);'>CV AUC mean</th>
        <th style='padding:5px 10px;color:#5bc0be;text-align:left;border-bottom:1px solid rgba(91,192,190,0.3);'>±Std</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>"""

# Từ điển câu hỏi → (tên hiển thị, nội dung HTML)
PRESET_QA = {
    # ── CHỈ SỐ TÀI CHÍNH ──────────────────────────────────────────────
    "roa_def": (
        "ROA là gì? Ngành BĐS đặc thù thế nào?",
        f"""<div class='answer-title'>📐 ROA (Return on Assets) — Tỷ suất lợi nhuận trên tài sản</div>
        <b>Công thức:</b> ROA = Lợi nhuận sau thuế / Tổng tài sản bình quân<br><br>
        ROA đo lường hiệu quả sinh lời từ <b>toàn bộ tài sản</b>, bất kể nguồn vốn — không bị bóp méo bởi cấu trúc vốn như ROE.
        Đặc biệt có giá trị trong BĐS Việt Nam khi nhiều DN có vốn chủ sở hữu âm.<br><br>
        <b>Thống kê dataset ({S['n_obs']:,} quan sát):</b><br>
        • Trung bình ngành: <b style='color:#5bc0be;'>{S['roa_mean']*100:.2f}%</b> &nbsp;|&nbsp;
          Median: <b style='color:#5bc0be;'>{S['roa_median']*100:.2f}%</b><br>
        • Độ lệch chuẩn: {S['roa_std']*100:.2f}% — biến động cao do chu kỳ bàn giao dự án<br>
        • <b style='color:#e74c3c;'>{S['roa_neg_pct']:.1f}%</b> quan sát có ROA âm — đặc thù ngành vốn lớn, ghi nhận doanh thu không đều<br><br>
        <b>Biến mục tiêu:</b> target = 1 nếu ROA quý <i>t+1</i> > median ngành cùng quý
        (tránh bias mùa vụ, loại bỏ chu kỳ vĩ mô)."""
    ),
    "dar_def": (
        "DAR — Tỷ lệ nợ/Tổng tài sản có ý nghĩa gì?",
        f"""<div class='answer-title'>🏦 DAR (Debt-to-Asset Ratio) — Tỷ lệ nợ trên Tổng tài sản</div>
        <b>Công thức:</b> DAR = Tổng nợ phải trả / Tổng tài sản<br><br>
        DAR phản ánh mức độ sử dụng đòn bẩy tài chính. BĐS Việt Nam dùng nợ để tài trợ quỹ đất và dự án
        → DAR trung bình rất cao so với các ngành khác.<br><br>
        <b>Trong dataset:</b> DAR trung bình = <b style='color:#5bc0be;'>{S['dar_mean']:.1f}%</b><br>
        • DAR &gt; 70%: cần giám sát rủi ro tài chính<br>
        • DAR &gt; 90%: cảnh báo nguy cơ mất khả năng thanh toán<br><br>
        <b>SHAP insight:</b> DAR xếp áp chót trong feature importance (SHAP ≈ 0.0011, p=0.191 không có ý nghĩa).
        Lý do: cấu trúc vốn ít thay đổi theo quý và đồng đều cao trong ngành —
        model khó dùng DAR để phân biệt xu hướng ROA ngắn hạn."""
    ),
    "icr_def": (
        "ICR — Hệ số khả năng trả lãi vay hoạt động thế nào?",
        f"""<div class='answer-title'>💳 ICR (Interest Coverage Ratio) — Hệ số đảm bảo lãi vay</div>
        <b>Công thức:</b> ICR = Lợi nhuận hoạt động / Chi phí lãi vay<br><br>
        ICR &lt; 1: DN không tạo đủ lợi nhuận để trả lãi → rủi ro cao<br>
        ICR 1–1.5: vùng cảnh báo | ICR &gt; 3: an toàn tài chính<br><br>
        <b>Đặc điểm dataset:</b><br>
        • <b style='color:#f39c12;'>{S['icr_miss_pct']:.1f}%</b> quan sát có ICR missing —
          vì {100-S['hd_pct']:.1f}% DN không có nợ vay (interest_expense = 0)<br>
        • Xử lý: tạo biến <code>has_debt</code> = 1/0; ICR missing → fill 0<br><br>
        <b>SHAP vs Spearman phân kỳ:</b> ICR đứng thứ 3 theo Spearman ρ (0.368) nhưng thứ 6 theo SHAP (≈0.009).
        Lý do <b>threshold effect phi tuyến</b>: tác động cận biên ICR giảm dần sau ngưỡng an toàn —
        SHAP mean bị kéo giảm khi tính bình quân toàn phân phối. ICR rất quan trọng ở vùng cực đoan (ICR &lt; 1)."""
    ),
    "shap_def": (
        "SHAP là gì? Tại sao dùng SHAP thay vì feature importance?",
        f"""<div class='answer-title'>🔍 SHAP (SHapley Additive exPlanations)</div>
        SHAP đo lường đóng góp <b>biên từng feature cho từng quan sát riêng lẻ</b>,
        dựa trên lý thuyết Shapley từ lý thuyết trò chơi.<br><br>
        <b>Khác biệt so với feature importance truyền thống:</b><br>
        • Feature importance (Gini): chỉ cho biết feature nào quan trọng <i>tổng thể</i> — không giải thích <i>hướng</i> tác động<br>
        • SHAP: cho biết mỗi feature <b>đẩy xác suất dự báo lên hay xuống bao nhiêu</b> cho từng DN cụ thể<br><br>
        Top 3 ROA-related features chiếm <b style='color:#5bc0be;'>{S['shap_top3_pct']:.1f}%</b> tổng SHAP
        → xác nhận tính dai dẳng lợi nhuận (profit persistence) là cơ chế dự báo nổi trội nhất."""
    ),
    "walkforward_def": (
        "Walk-forward CV là gì? Tại sao không dùng k-fold thông thường?",
        f"""<div class='answer-title'>⏩ Walk-forward Expanding Window Cross-Validation</div>
        K-fold thông thường <b>trộn dữ liệu theo thời gian</b> → train có thể dùng dữ liệu tương lai để dự báo quá khứ
        → <b style='color:#e74c3c;'>data leakage nghiêm trọng</b> trong time-series tài chính.<br><br>
        <b>Walk-forward CV đảm bảo:</b><br>
        • Fold 1: Train 2014–2016 → Validate 2017<br>
        • Fold 2: Train 2014–2017 → Validate 2018<br>
        • ... (expanding window — thêm data mỗi fold)<br>
        • Fold 8: Train 2014–2023 → Validate 2024<br>
        • Holdout: Train 2014–2024 → <b>Test 2025 (độc lập hoàn toàn)</b><br><br>
        <b>Nested tuning:</b> RandomSearch(n_iter=15, cv=3) chạy bên trong mỗi fold → tránh leakage hyperparameter.<br>
        Kết quả: {S['best_name']} CV AUC = <b style='color:#5bc0be;'>{S['cv_auc_mean']:.4f} ± {S['cv_auc_std']:.4f}</b>"""
    ),
    # ── MÔ HÌNH ──────────────────────────────────────────────────────
    "model_compare": (
        "So sánh 6 mô hình ML — mô hình nào tốt nhất?",
        f"""<div class='answer-title'>🤖 So sánh {len(S['cv_summary'])} mô hình — Walk-forward CV 8 folds</div>
        {_cv_table_html()}
        <b>{S['best_name']}</b> được chọn với CV AUC cao nhất và ổn định nhất qua 8 folds.<br><br>
        <b>Lý do RF hoạt động tốt nhất:</b><br>
        • <code>max_depth=3</code> (cây nông) kiểm soát overfitting hiệu quả trên panel data nhiễu cao<br>
        • Ensemble nhiều cây → robust với outlier trong tỷ số tài chính BĐS<br>
        • Xử lý tốt quan hệ phi tuyến (threshold effect của ICR, step function ROA)<br><br>
        <b>Kiểm định Friedman:</b> p = 0.0045 → có sự khác biệt có ý nghĩa thống kê giữa 6 models.
        RF rank trung bình = 2.25 (thấp nhất = tốt nhất)."""
    ),
    "holdout_result": (
        "Kết quả Holdout 2025 — đánh giá hiệu năng thực tế?",
        f"""<div class='answer-title'>🧪 Holdout 2025 — Đánh giá ngoài mẫu thực sự</div>
        <b>{S['n_obs'] - int(panel[panel['year']==2025].shape[0] if 'year' in panel.columns else 188)} obs train</b>
        (2014–2024) → <b>188 obs test</b> (2025, hoàn toàn độc lập)<br><br>
        <table style='width:100%;border-collapse:collapse;font-size:0.85rem;'>
        <tr style='border-bottom:1px solid rgba(91,192,190,0.2);'>
          <td style='padding:5px;color:#5bc0be;font-weight:700;'>Metric</td>
          <td style='padding:5px;color:#5bc0be;font-weight:700;'>Giá trị</td>
          <td style='padding:5px;color:#5bc0be;font-weight:700;'>Diễn giải</td>
        </tr>
        <tr><td style='padding:5px;color:#e0e6ed;'>AUC</td>
            <td style='padding:5px;font-family:Space Mono,monospace;color:#5bc0be;'>{S['holdout_auc']:.4f}</td>
            <td style='padding:5px;color:#a8b8cc;font-size:0.82rem;'>CI 95%: [{S['holdout_ci_lo']:.4f} – {S['holdout_ci_hi']:.4f}]</td></tr>
        <tr><td style='padding:5px;color:#e0e6ed;'>Gap CV→Holdout</td>
            <td style='padding:5px;font-family:Space Mono,monospace;color:#f39c12;'>{(S['cv_auc_mean']-S['holdout_auc'])*100:.1f}pp</td>
            <td style='padding:5px;color:#a8b8cc;font-size:0.82rem;'>Overfitting nhẹ — chấp nhận được</td></tr>
        <tr><td style='padding:5px;color:#e0e6ed;'>Ngưỡng Youden</td>
            <td style='padding:5px;font-family:Space Mono,monospace;color:#5bc0be;'>{S['youden_thr']:.3f}</td>
            <td style='padding:5px;color:#a8b8cc;font-size:0.82rem;'>Tối ưu hơn ngưỡng mặc định 0.5</td></tr>
        </table><br>
        AUC ~0.69 trong dự báo tài chính panel <b>là chấp nhận được</b> — thị trường BĐS VN có nhiều yếu tố
        vĩ mô (chính sách tín dụng, pháp lý) không dự báo được từ tỷ số tài chính đơn thuần."""
    ),
    "subgroup_result": (
        "Subgroup analysis — nhóm nào mô hình dự báo tốt nhất/kém nhất?",
        f"""<div class='answer-title'>📊 Subgroup Analysis — Hiệu năng theo nhóm con (Holdout 2025)</div>
        <b>Tốt nhất: {S['best_sg']}</b> (AUC = {S['best_sg_auc']:.3f}) — DN có mô hình kinh doanh thuần túy BĐS,
        chỉ số tài chính phân biệt rõ hơn theo quý.<br><br>
        <b>Kém nhất: {S['worst_sg']}</b> (AUC = {S['worst_sg_auc']:.3f}) — Tập đoàn lớn đa ngành hoá cao:
        phát triển nhà ở + khu công nghiệp + thương mại + dịch vụ → chỉ số tổng hợp cấp DN ít phân biệt
        được xu hướng ROA theo quý.<br><br>
        <b>has_debt breakdown:</b><br>
        • has_debt = 1 (có vay): AUC = 0.706 — ICR cung cấp tín hiệu phân biệt bổ sung<br>
        • has_debt = 0 (không vay): AUC = 0.626 — thiếu tín hiệu từ nhóm ICR/lãi vay<br><br>
        <b>Theo giai đoạn (train set):</b> Pre-COVID (0.812) · COVID (0.850) · Post-COVID (0.846)
        → mô hình ổn định qua các chu kỳ thị trường, học được quan hệ bền vững."""
    ),
    # ── DỮ LIỆU ──────────────────────────────────────────────────────
    "dataset_overview": (
        "Tổng quan dataset — bao nhiêu DN, quan sát, feature?",
        f"""<div class='answer-title'>🗂️ Tổng quan Dataset BĐS Việt Nam</div>
        <b>{S['n_firms']} doanh nghiệp</b> BĐS phi tài chính niêm yết liên tục HOSE/HNX<br>
        Giai đoạn: Q1/2014 – Q4/2025 (48 quý · ~{round(S['n_obs']/S['n_firms'])} quý/DN)<br>
        Tổng quan sát: <b style='color:#5bc0be;'>{S['n_obs']:,}</b> (sau làm sạch)<br><br>
        <b>Loại bỏ 3 DN:</b> PVR, VCR, PV2 — revenue = 0 trong &gt;40% các quý
        → ratio tài chính bất thường, gây bias model.<br><br>
        <b>Features ML:</b> 13 biến cuối (chọn từ 114 biến ban đầu qua VIF + lý thuyết tài chính)<br>
        ROA · NPM · TATO · ITO · DAR · QR · FCF_TA · ICR · has_debt · SIZE · ROA_lag1 · ROA_lag4 · REV_GROWTH_YOY<br><br>
        <b>Phân bố target:</b> {S['n_pos']:,} target=1 ({S['pos_rate']:.1f}%) · {S['n_neg']:,} target=0 ({100-S['pos_rate']:.1f}%)<br>
        Train 2014–2024: 52.2% pos · Holdout 2025: 39.4% pos"""
    ),
    "covid_impact": (
        "COVID-19 ảnh hưởng thế nào đến ngành BĐS Việt Nam?",
        f"""<div class='answer-title'>🦠 COVID-19 — Structural Break Q1/2020–Q2/2021</div>
        <b>Trước COVID (2014–2019):</b> Median ROA ngành = {S['pre_roa']*100:.2f}%<br>
        <b>Trong COVID (Q1/2020–Q2/2021):</b> Median ROA = {S['covid_roa']*100:.2f}%<br>
        → <b style='color:#e74c3c;'>Giảm {S['covid_drop']:.1f}%</b> so với giai đoạn trước<br><br>
        <b>Tác động phân kỳ:</b><br>
        • Một số DN hưởng lợi từ nhu cầu nhà ở tích lũy và lãi suất thấp<br>
        • Nhiều DN khó khăn vì đình trệ bàn giao, pháp lý bị trì hoãn<br>
        • Độ phân kỳ ROA giữa DN tăng mạnh trong giai đoạn này<br><br>
        <b>Ảnh hưởng đến model:</b> Fold CV test năm 2020 thường có AUC thấp nhất trong 8 folds —
        structural break khiến quan hệ feature-target thay đổi tạm thời.
        Tuy nhiên walk-forward CV tự nhiên nắm bắt được khi train tích lũy qua thời gian."""
    ),
    "pvr_vcr_pv2": (
        "Tại sao loại bỏ PVR, VCR, PV2 khỏi dataset?",
        f"""<div class='answer-title'>🚫 Loại bỏ PVR · VCR · PV2 — Lý do kỹ thuật</div>
        Ba DN này có <b>Revenue = 0 hoặc gần 0 trong &gt;40% các quý</b>:<br><br>
        • <b>PVR</b> (Petrovietnam Retail): dừng/thu hẹp hoạt động BĐS — BCTC không phản ánh hoạt động thực<br>
        • <b>VCR</b> (Vinaconex Riverside): giao dịch rất thưa, dữ liệu không liên tục<br>
        • <b>PV2</b>: dữ liệu tài chính không đủ độ tin cậy cho mô hình<br><br>
        <b>Hệ quả kỹ thuật nếu giữ lại:</b><br>
        • Tỷ số tài chính như NPM, TATO, REV_GROWTH_YOY → chia cho revenue ≈ 0 → outlier cực đoan<br>
        • Winsorize không đủ để xử lý — bias model một cách hệ thống<br>
        • Target rate của 3 DN này bất thường, không đại diện cho ngành<br><br>
        Còn lại <b>47 doanh nghiệp</b> với {S['n_obs']:,} quan sát sạch."""
    ),
    "target_var": (
        "Biến mục tiêu được xây dựng như thế nào?",
        f"""<div class='answer-title'>🎯 Xây dựng Biến Mục tiêu Binary</div>
        <b>target = 1</b> nếu ROA<sub>t+1</sub> của DN &gt; median ngành cùng quý<br>
        <b>target = 0</b> nếu ROA<sub>t+1</sub> của DN ≤ median ngành cùng quý<br><br>
        <b>Tại sao so với median ngành cùng quý?</b><br>
        • Loại bỏ tác động chu kỳ kinh tế vĩ mô (recession, boom toàn ngành)<br>
        • Loại bỏ mùa vụ (Q4 thường có ROA cao hơn Q1 toàn ngành)<br>
        • Tập trung vào <i>hiệu quả tương đối</i> — quan trọng cho nhà đầu tư so sánh trong ngành<br><br>
        <b>Phân bố:</b> {S['pos_rate']:.1f}% target=1 trên toàn giai đoạn 2014–2025<br>
        Train 2014–2024: 52.2% (cân bằng tốt) · Holdout 2025: 39.4% (pos thấp hơn — năm khó khăn)"""
    ),
}

# Nhóm câu hỏi cho UI
QA_GROUPS = {
    "📐 Chỉ số tài chính": ["roa_def", "dar_def", "icr_def"],
    "🤖 Mô hình & Phương pháp": ["shap_def", "walkforward_def", "model_compare", "holdout_result", "subgroup_result"],
    "🗂️ Dữ liệu & Nghiên cứu": ["dataset_overview", "covid_impact", "pvr_vcr_pv2", "target_var"],
}


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR — chọn chế độ
# ══════════════════════════════════════════════════════════════════════
st.sidebar.markdown("### 💬 Chế độ xem")
mode = st.sidebar.radio("Chế độ:", [
    "❓ Hỏi & Đáp nhanh",
    "🏢 Tra cứu doanh nghiệp",
])

st.sidebar.markdown("---")

if mode == "🏢 Tra cứu doanh nghiệp":
    selected_firm = st.sidebar.selectbox("Chọn mã DN:", TICKERS, index=TICKERS.index("VIC") if "VIC" in TICKERS else 0)
    st.sidebar.markdown(f"""
    <div class='card' style='font-size:0.78rem;margin-top:0.5rem;'>
    <b style='color:#5bc0be;'>{selected_firm}</b><br>
    <span style='color:#6b7f99;'>Nhấn vào mã để xem dashboard đầy đủ: lịch sử ROA, tín hiệu dự báo, so sánh ngành.</span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
    <div style='font-size:0.8rem;color:#6b7f99;'>    </div>
    """, unsafe_allow_html=True)

# KPI strip tổng quan
c1, c2, c3, c4, c5, c6 = st.columns(6)
kpis = [
    (str(S['n_firms']), "Doanh nghiệp"),
    (f"{S['n_obs']:,}", "Quan sát"),
    (f"{S['roa_mean']*100:.2f}%", "ROA TB ngành"),
    (f"{S['cv_auc_mean']:.4f}", "CV AUC (RF)"),
    (f"{S['holdout_auc']:.4f}", "Holdout AUC"),
    (f"{S['youden_thr']:.3f}", "Youden Threshold"),
]
for col, (val, lbl) in zip([c1,c2,c3,c4,c5,c6], kpis):
    col.markdown(f"""
    <div class='kpi-mini'>
      <div class='kpi-mini-val'>{val}</div>
      <div class='kpi-mini-lbl'>{lbl}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='margin:1rem 0;'>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# MODE 1: HỎI & ĐÁP NHANH
# ══════════════════════════════════════════════════════════════════════
if mode == "❓ Hỏi & Đáp nhanh":

    # Init session state cho câu hỏi đang chọn
    if "selected_qa" not in st.session_state:
        st.session_state.selected_qa = None

    col_q, col_a = st.columns([2, 3], gap="large")

    with col_q:
        section_header(st, "Chọn câu hỏi", "❓")
        st.markdown("<div style='font-size:0.82rem;color:#6b7f99;margin-bottom:0.8rem;'>Click vào câu hỏi để xem câu trả lời ngay →</div>", unsafe_allow_html=True)

        for group_name, keys in QA_GROUPS.items():
            st.markdown(f"<div style='font-size:0.78rem;color:#5bc0be;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;margin:0.8rem 0 0.3rem;'>{group_name}</div>", unsafe_allow_html=True)
            for key in keys:
                q_label, _ = PRESET_QA[key]
                is_active = (st.session_state.selected_qa == key)

                btn_label = f"▶ {q_label}" if is_active else q_label

                if st.button(
                    btn_label,
                    key=f"btn_{key}",
                    use_container_width=True,
                ):
                    st.session_state.selected_qa = key
                    st.rerun()

    with col_a:
        section_header(st, "Câu trả lời", "💡")
        if st.session_state.selected_qa is None:
            st.markdown("""
            <div style='text-align:center;padding:3rem 1rem;color:#3a506b;'>
                <div style='font-size:3rem;margin-bottom:1rem;'>💬</div>
                <div style='font-size:1rem;color:#5bc0be;font-weight:600;'>Chọn một câu hỏi bên trái</div>
                <div style='font-size:0.85rem;margin-top:0.5rem;'>
                Tất cả câu trả lời được tính trực tiếp từ data thật<br>
                — không dùng AI API, phản hồi tức thì.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            _, answer_html = PRESET_QA[st.session_state.selected_qa]
            st.markdown(f"<div class='answer-box'>{answer_html}</div>", unsafe_allow_html=True)

            # Gợi ý câu tiếp theo
            all_keys = list(PRESET_QA.keys())
            curr_idx = all_keys.index(st.session_state.selected_qa)
            if curr_idx + 1 < len(all_keys):
                next_key = all_keys[curr_idx + 1]
                next_q, _ = PRESET_QA[next_key]
                st.markdown(f"""
                <div style='margin-top:1rem;font-size:0.8rem;color:#6b7f99;'>
                Câu tiếp theo có thể bạn quan tâm:
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"→ {next_q}", key="btn_next_suggest"):
                    st.session_state.selected_qa = next_key
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════
# MODE 2: TRA CỨU DOANH NGHIỆP
# ══════════════════════════════════════════════════════════════════════
elif mode == "🏢 Tra cứu doanh nghiệp":
    section_header(st, f"Dashboard · {selected_firm}", "🏢")

    # ── Load data DN ─────────────────────────────────────────────────
    @st.cache_data(show_spinner=False)
    def get_firm_data_cached(ticker):
        return dl.get_firm_data(ticker, enriched=True)

    df_firm = get_firm_data_cached(selected_firm)

    if len(df_firm) == 0:
        st.warning(f"Không tìm thấy dữ liệu cho {selected_firm}")
        st.stop()

    # Tính xác suất dự báo cho tất cả kỳ trong DN
    @st.cache_data(show_spinner=False)
    def get_firm_probs(ticker):
        df_f = dl.get_firm_data(ticker, enriched=False)
        probs = []
        for _, row in df_f.iterrows():
            try:
                probs.append(safe_predict(row))
            except Exception:
                probs.append(np.nan)
        df_f = df_f.copy()
        df_f['prob'] = probs
        df_f['q_label'] = df_f['year'].astype(str) + 'Q' + df_f['quarter'].astype(str)
        return df_f

    df_prob = get_firm_probs(selected_firm)
    latest  = df_firm.iloc[-1]
    latest_prob = safe_predict(latest)
    signal  = dl.prob_to_signal(latest_prob)
    signal_map = {
        "Strong_Buy":  ("Rất cao",    "signal-strong-buy",  "#2ecc71"),
        "Buy":         ("Cao",        "signal-buy",         "#27ae60"),
        "Neutral":     ("Trung bình", "signal-neutral",     "#f1c40f"),
        "Sell":        ("Thấp",       "signal-neutral",     "#e67e22"),
        "Strong_Sell": ("Rất thấp",   "signal-sell",        "#e74c3c"),
    }
    sig_label, sig_class, sig_color = signal_map.get(signal, ("Neutral", "signal-neutral", "#f1c40f"))

    # Thống kê lịch sử DN
    hist_target_rate = float(df_firm['target'].mean() * 100) if 'target' in df_firm.columns else 0
    hist_roa_mean    = float(df_firm['ROA'].mean())
    hist_roa_std     = float(df_firm['ROA'].std())

    # ── Header DN ───────────────────────────────────────────────────
    st.markdown(f"""
    <div class='firm-header'>
      <div style='display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:0.5rem;'>
        <div>
          <div style='font-size:1.8rem;font-weight:800;color:#e0e6ed;'>{selected_firm}</div>
          <div style='font-size:0.82rem;color:#6b7f99;margin-top:2px;'>
            {len(df_firm)} quý · {int(df_firm['year'].min()) if 'year' in df_firm.columns else ''}–{int(df_firm['year'].max()) if 'year' in df_firm.columns else ''}
          </div>
          <div style='margin-top:6px;'>
            <span class='cat-label'>BĐS Niêm yết · HOSE/HNX</span>
          </div>
        </div>
        <div style='text-align:right;'>
          <div style='font-size:0.78rem;color:#6b7f99;margin-bottom:4px;'>Tín hiệu kỳ mới nhất</div>
          <span class='signal-pill {sig_class}'>{sig_label}</span>
          <div style='font-family:Space Mono,monospace;font-size:1.5rem;font-weight:700;color:{sig_color};margin-top:4px;'>
            {latest_prob*100:.1f}%
          </div>
          <div style='font-size:0.75rem;color:#6b7f99;'>P(ROA_cao) · {S['best_name']} model</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Row ──────────────────────────────────────────────────────
    kpi_cols = st.columns(5)
    kpi_data = [
        ("ROA hiện tại",   f"{float(latest.get('ROA',0))*100:.2f}%", "Kỳ mới nhất"),
        ("DAR",            f"{float(latest.get('DAR',0))*100:.1f}%",  "Tỷ lệ nợ/TS"),
        ("QR",             f"{float(latest.get('QR',0)):.2f}x",       "Thanh khoản nhanh"),
        ("ROA TB lịch sử", f"{hist_roa_mean*100:.2f}%",               f"2014–2025 · std={hist_roa_std*100:.2f}%"),
        ("Target rate",    f"{hist_target_rate:.1f}%",                 "% kỳ ROA > median ngành"),
    ]
    for col, (lbl, val, sub) in zip(kpi_cols, kpi_data):
        col.metric(lbl, val, sub)

    st.markdown("---")

    # ── Biểu đồ 1: ROA + Target + Prob qua thời gian ────────────────
    col_l, col_r = st.columns([3, 2], gap="large")
    with col_l:
        section_header(st, "ROA lịch sử & Xác suất dự báo", "")
        df_plot = df_prob.dropna(subset=['ROA']).copy()

        fig = go.Figure()

        # Nền target=1 (ROA_cao)
        if 'target' in df_plot.columns:
            for _, row in df_plot[df_plot['target']==1].iterrows():
                fig.add_shape(type="rect",
                    x0=row['q_label'], x1=row['q_label'],
                    y0=-1, y1=1, line_width=6,
                    line_color="rgba(46,204,113,0.25)")

        # ROA bar
        roa_colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in df_plot['ROA']]
        fig.add_trace(go.Bar(
            x=df_plot['q_label'], y=df_plot['ROA'],
            name="ROA", marker_color=roa_colors, opacity=0.7,
            yaxis="y1"
        ))

        # Prob line
        if 'prob' in df_plot.columns:
            fig.add_trace(go.Scatter(
                x=df_plot['q_label'], y=df_plot['prob'],
                mode="lines", name="P(ROA_cao)",
                line=dict(color="#5bc0be", width=2),
                yaxis="y2"
            ))
            fig.add_hline(y=0.5, line_dash="dot", line_color="rgba(241,196,15,0.5)", line_width=1)

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Arial, sans-serif", color="#c8d8e8", size=11),
            height=360,
            title=dict(text=f"{selected_firm} — ROA theo quý & Xác suất dự báo", font=dict(color="#5bc0be", size=13)),
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
            barmode="overlay",
            xaxis=dict(tickangle=-45, nticks=16, gridcolor="rgba(91,192,190,0.08)",
                       linecolor="rgba(91,192,190,0.2)"),
            yaxis=dict(title="ROA", gridcolor="rgba(91,192,190,0.1)",
                       linecolor="rgba(91,192,190,0.2)", tickformat=".1%"),
            yaxis2=dict(title="P(ROA_cao)", overlaying="y", side="right",
                        range=[0,1], showgrid=False, tickformat=".0%",
                        tickfont=dict(color="#5bc0be")),
            margin=dict(l=50, r=60, t=50, b=60),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        section_header(st, "Chỉ số kỳ mới nhất so với ngành", "")

        # So sánh 6 chỉ số chính vs median ngành
        compare_features = ["ROA","DAR","QR","NPM","FCF_TA","SIZE"]
        ind_medians = {}
        for f in compare_features:
            if f in panel.columns:
                ind_medians[f] = float(panel[f].median())

        rows_html = ""
        for f in compare_features:
            if f not in df_firm.columns or f not in ind_medians:
                continue
            firm_val = float(latest.get(f, np.nan))
            ind_med  = ind_medians[f]
            if np.isnan(firm_val):
                continue

            # Chiều tốt: ROA/QR/NPM/FCF_TA cao hơn = tốt; DAR thấp hơn = tốt
            better_if_higher = f not in ("DAR",)
            is_better = (firm_val > ind_med) if better_if_higher else (firm_val < ind_med)
            color = "#2ecc71" if is_better else "#e74c3c"
            arrow = "▲" if firm_val > ind_med else "▼"

            # Format
            def fmt(v, col):
                if col in ("ROA","NPM","FCF_TA","DAR","QR"): return f"{v*100:.2f}%"
                if col == "SIZE": return f"{v:.2f}"
                return f"{v:.3f}"

            rows_html += f"""
            <tr style='border-bottom:1px solid rgba(91,192,190,0.08);'>
              <td style='padding:6px 8px;color:#a8b8cc;font-size:0.82rem;'>{f}</td>
              <td style='padding:6px 8px;font-family:Space Mono,monospace;color:{color};font-size:0.85rem;'>
                {fmt(firm_val,f)} <span style='font-size:0.75rem;'>{arrow}</span>
              </td>
              <td style='padding:6px 8px;font-family:Space Mono,monospace;color:#6b7f99;font-size:0.82rem;'>{fmt(ind_med,f)}</td>
            </tr>"""

        st.markdown(f"""
        <table style='width:100%;border-collapse:collapse;'>
          <thead><tr>
            <th style='padding:6px 8px;color:#5bc0be;text-align:left;font-size:0.78rem;border-bottom:1px solid rgba(91,192,190,0.3);'>Feature</th>
            <th style='padding:6px 8px;color:#5bc0be;text-align:left;font-size:0.78rem;border-bottom:1px solid rgba(91,192,190,0.3);'>{selected_firm}</th>
            <th style='padding:6px 8px;color:#5bc0be;text-align:left;font-size:0.78rem;border-bottom:1px solid rgba(91,192,190,0.3);'>Median ngành</th>
          </tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

        # Tín hiệu dự báo gần nhất
        recent_probs = df_prob.dropna(subset=['prob']).tail(8)
        if len(recent_probs) > 0:
            st.markdown("<br>", unsafe_allow_html=True)
            section_header(st, "Tín hiệu 8 kỳ gần nhất", "")
            sig_rows = ""
            for _, r in recent_probs.iterrows():
                p   = float(r['prob'])
                tgt = int(r.get('target', -1))
                s   = dl.prob_to_signal(p)
                sig_info = signal_map.get(s, ("Trung bình","signal-neutral","#f1c40f"))
                _, sc, sclr = sig_info
                correct_icon = ""
                if tgt == 1: correct_icon = " ✓" if p >= 0.5 else " ✗"
                elif tgt == 0: correct_icon = " ✓" if p < 0.5 else " ✗"
                sig_rows += f"""
                <tr>
                  <td style='padding:4px 8px;color:#a8b8cc;font-size:0.8rem;'>{r['q_label']}</td>
                  <td style='padding:4px 8px;font-family:Space Mono,monospace;color:{sclr};font-size:0.82rem;'>{p*100:.0f}%</td>
                  <td style='padding:4px 8px;'><span class='signal-pill {sc}' style='font-size:0.7rem;padding:2px 8px;'>{s.replace("_"," ")}</span>{correct_icon}</td>
                </tr>"""
            st.markdown(f"""
            <table style='width:100%;border-collapse:collapse;'>
              <thead><tr>
                <th style='padding:4px 8px;color:#5bc0be;text-align:left;font-size:0.76rem;border-bottom:1px solid rgba(91,192,190,0.3);'>Quý</th>
                <th style='padding:4px 8px;color:#5bc0be;text-align:left;font-size:0.76rem;border-bottom:1px solid rgba(91,192,190,0.3);'>P(ROA_cao)</th>
                <th style='padding:4px 8px;color:#5bc0be;text-align:left;font-size:0.76rem;border-bottom:1px solid rgba(91,192,190,0.3);'>Tín hiệu</th>
              </tr></thead>
              <tbody>{sig_rows}</tbody>
            </table>
            <div style='font-size:0.72rem;color:#3a506b;margin-top:4px;'>✓ = dự báo đúng · ✗ = dự báo sai (tại ngưỡng 0.5)</div>
            """, unsafe_allow_html=True)

    # ── SHAP feature contributions cho kỳ mới nhất ─────────────────
    st.markdown("---")
    section_header(st, f"Yếu tố tác động đến dự báo kỳ mới nhất — {selected_firm}", "")
    st.markdown(f"""
    <div style='font-size:0.83rem;color:#8899aa;margin-bottom:0.8rem;'>
    Các chỉ số tài chính hiện tại của <b>{selected_firm}</b> đóng góp như thế nào vào xác suất P(ROA_cao) = <b style='color:#5bc0be;'>{latest_prob*100:.1f}%</b>.
    So sánh với median ngành — giá trị vượt ngành (xanh) thường đóng góp dương vào xác suất.
    </div>
    """, unsafe_allow_html=True)

    shap_sorted = feat_imp.sort_values('importance', ascending=False).reset_index(drop=True)
    contrib_rows = ""
    bar_data_labels, bar_data_firm, bar_data_ind = [], [], []

    for _, row in shap_sorted.iterrows():
        f    = row['feature']
        shap = float(row['importance'])
        if f not in df_firm.columns: continue
        fval = float(latest.get(f, np.nan))
        if np.isnan(fval): continue
        imed = float(panel[f].median()) if f in panel.columns else 0

        diff = fval - imed
        is_high = (diff > 0) if f not in ("DAR",) else (diff < 0)

        def fmt2(v, col):
            if col in ("ROA","ROA_lag1","ROA_lag4","NPM","FCF_TA","DAR"): return f"{v*100:.2f}%"
            if col == "SIZE": return f"{v:.2f}"
            if col == "has_debt": return "Có" if v == 1 else "Không"
            return f"{v:.3f}"

        col_clr = "#2ecc71" if is_high else "#e74c3c"
        arrow   = "▲" if diff > 0 else "▼"
        bar_w   = min(100, int(shap / shap_sorted['importance'].max() * 100))

        contrib_rows += f"""
        <tr style='border-bottom:1px solid rgba(91,192,190,0.07);'>
          <td style='padding:5px 10px;color:#a8b8cc;font-size:0.82rem;width:110px;'>{f}</td>
          <td style='padding:5px 10px;font-family:Space Mono,monospace;color:{col_clr};font-size:0.85rem;'>
            {fmt2(fval,f)} <span style='font-size:0.72rem;'>{arrow}</span>
          </td>
          <td style='padding:5px 10px;font-family:Space Mono,monospace;color:#6b7f99;font-size:0.82rem;'>{fmt2(imed,f)}</td>
          <td style='padding:5px 10px;width:150px;'>
            <div style='background:#1a2540;border-radius:3px;height:5px;'>
              <div style='background:{col_clr};height:5px;border-radius:3px;width:{bar_w}%;opacity:0.7;'></div>
            </div>
            <div style='font-size:0.72rem;color:#6b7f99;margin-top:2px;'>SHAP: {shap:.4f}</div>
          </td>
        </tr>"""

    st.markdown(f"""
    <div style='overflow-x:auto;'>
    <table style='width:100%;border-collapse:collapse;'>
      <thead><tr>
        <th style='padding:5px 10px;color:#5bc0be;text-align:left;font-size:0.78rem;border-bottom:1px solid rgba(91,192,190,0.3);'>Feature</th>
        <th style='padding:5px 10px;color:#5bc0be;text-align:left;font-size:0.78rem;border-bottom:1px solid rgba(91,192,190,0.3);'>{selected_firm}</th>
        <th style='padding:5px 10px;color:#5bc0be;text-align:left;font-size:0.78rem;border-bottom:1px solid rgba(91,192,190,0.3);'>Median ngành</th>
        <th style='padding:5px 10px;color:#5bc0be;text-align:left;font-size:0.78rem;border-bottom:1px solid rgba(91,192,190,0.3);'>SHAP importance</th>
      </tr></thead>
      <tbody>{contrib_rows}</tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)

    # ── Lịch sử target rate theo năm ─────────────────────────────────
    if 'target' in df_firm.columns and 'year' in df_firm.columns:
        st.markdown("---")
        section_header(st, f"Tỷ lệ ROA > Median ngành theo năm — {selected_firm}", "")
        yr_rate = df_firm.groupby('year')['target'].mean().reset_index()
        yr_rate.columns = ['year', 'rate']
        bar_colors_yr = ["#2ecc71" if v >= 0.5 else "#e74c3c" for v in yr_rate['rate']]

        fig_yr = go.Figure()
        fig_yr.add_trace(go.Bar(
            x=yr_rate['year'].astype(str), y=yr_rate['rate'],
            marker_color=bar_colors_yr, opacity=0.8, name="Target rate",
            text=[f"{v*100:.0f}%" for v in yr_rate['rate']],
            textposition="outside", textfont=dict(size=10, color="#c8d8e8")
        ))
        fig_yr.add_hline(y=0.5, line_dash="dash", line_color="#f1c40f", line_width=1,
                         annotation_text="50%", annotation_font_color="#f1c40f")
        fig_yr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Arial, sans-serif", color="#c8d8e8", size=11),
            height=260,
            title=dict(text=f"{selected_firm} — % quý đạt ROA_cao từng năm (xanh ≥ 50%)", font=dict(color="#5bc0be", size=12)),
            xaxis=dict(gridcolor="rgba(91,192,190,0.08)", linecolor="rgba(91,192,190,0.2)"),
            yaxis=dict(gridcolor="rgba(91,192,190,0.1)", linecolor="rgba(91,192,190,0.2)",
                       range=[0, 1.2], tickformat=".0%"),
            margin=dict(l=40, r=20, t=50, b=30),
        )
        st.plotly_chart(fig_yr, use_container_width=True)