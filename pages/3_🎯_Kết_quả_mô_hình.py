import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from style import apply_style, PLOTLY_THEME, page_header, section_header
import data_loader as dl

st.set_page_config(page_title="Kết quả mô hình | BĐS", page_icon="🤖", layout="wide")
apply_style(st)

# ══════════════════════════════════════════════════════════════════════
# LOAD DATA THẬT
# ══════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Đang tải kết quả mô hình...")
def load_all():
    cv_df      = dl.get_cv_results()
    boot_df    = dl.get_bootstrap_ci()
    subgroup   = dl.get_subgroup_analysis()
    delong_df  = dl.get_delong_test()
    nemenyi_df = dl.get_nemenyi_test()
    thresh_df  = dl.get_threshold_analysis()
    feat_imp   = dl.get_feature_importance()
    pred_obs   = dl.get_predictions_obs()
    pred_firm  = dl.get_predictions_firm()
    bm_info    = dl.get_best_model_info()
    return cv_df, boot_df, subgroup, delong_df, nemenyi_df, thresh_df, feat_imp, pred_obs, pred_firm, bm_info

try:
    cv_df, boot_df, subgroup_df, delong_df, nemenyi_df, thresh_df, feat_imp, pred_obs, pred_firm, bm_info = load_all()
except FileNotFoundError as e:
    st.error(f"❌ Không tìm thấy file dữ liệu: {e}")
    st.stop()

# ── Derived từ cv_df ────────────────────────────────────────────────────────
MODELS = cv_df['model'].unique().tolist()
MODEL_COLORS = {
    "LR_L2":"#3498db","LR_L1":"#2980b9","RF":"#2ecc71",
    "GBM":"#27ae60","SVM":"#9b59b6","XGB":"#e67e22"
}
import itertools
_default_colors = ["#5bc0be","#f39c12","#e74c3c","#1abc9c","#8e44ad","#2c3e50"]
for m, c in zip(MODELS, itertools.cycle(_default_colors)):
    if m not in MODEL_COLORS:
        MODEL_COLORS[m] = c

cv_pivot = cv_df.pivot_table(index='fold', columns='model', values='AUC')

cv_summary = cv_df.groupby('model')[['AUC','PR_AUC','F1','Accuracy']].mean().round(4)
cv_std     = cv_df.groupby('model')['AUC'].std().round(4).rename('AUC_std')
cv_summary = cv_summary.join(cv_std)

# Holdout metrics
_rf_obs = pred_obs.copy()
if 'prob_roa_cao' in _rf_obs.columns and 'target' in _rf_obs.columns:
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, average_precision_score
    _y    = _rf_obs['target'].values
    _prob = _rf_obs['prob_roa_cao'].values
    _pred50 = (_prob >= 0.5).astype(int)
    _youden_thr = dl.get_youden_threshold()
    _pred_j = (_prob >= _youden_thr).astype(int)
    # Gán lại vào pred_obs để Section 2 dùng được (column pred_50 có thể không có trong CSV)
    pred_obs = _rf_obs.copy()
    pred_obs['pred_50'] = _pred50
    pred_obs['pred_j']  = _pred_j
    RF_HOLDOUT = {
        "AUC":    round(float(roc_auc_score(_y, _prob)), 4),
        "PR_AUC": round(float(average_precision_score(_y, _prob)), 4),
        "F1":     round(float(f1_score(_y, _pred50)), 4),
        "Acc":    round(float(accuracy_score(_y, _pred50)), 4),
    }
    _n_holdout  = len(_rf_obs)
    _n_pos_hold = int(_y.sum())
    _n_neg_hold = _n_holdout - _n_pos_hold
else:
    _auc_row = boot_df[boot_df['metric'].str.upper()=='AUC']
    RF_HOLDOUT = {
        "AUC":    float(_auc_row['mean'].iloc[0]) if not _auc_row.empty else 0.692,
        "PR_AUC": 0.5806, "F1": 0.5594, "Acc": 0.6649
    }
    _n_holdout, _n_pos_hold, _n_neg_hold = 188, 74, 114

_best_name    = bm_info.get('model_name', 'RF')
_cv_auc_mean  = bm_info.get('cv_auc_mean', 0.0)
_cv_auc_std   = bm_info.get('cv_auc_std', 0.0)
_final_params = bm_info.get('final_params', {})
_cv_holdout_gap = RF_HOLDOUT["AUC"] - _cv_auc_mean

page_header(st, "KẾT QUẢ MÔ HÌNH MACHINE LEARNING",
            f"Walk-forward CV · {len(MODELS)} models · {_best_name} tốt nhất · Holdout 2025", "🤖")

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("### 🤖 Phần kết quả")
section = st.sidebar.radio("Chọn:", [
    "📊 CV Walk-forward",
    "🧪 Holdout 2025",
    "🌟 SHAP Feature Importance",
    "🔬 Kiểm định thống kê",
    "🏢 Dự đoán theo DN",
])
st.sidebar.markdown("---")

_params_text = " · ".join([f"{k.replace('clf__','')}={v}" for k,v in list(_final_params.items())[:4]])
st.sidebar.markdown(f"""
<div class='card'>
<b style='color:#5bc0be;'>🏆 Best Model: {_best_name}</b><br>
<span style='font-size:0.82rem;color:#a8b8cc;'>
CV AUC: {_cv_auc_mean:.4f} ± {_cv_auc_std:.4f}<br>
Holdout AUC: {RF_HOLDOUT['AUC']:.4f}<br>
{_params_text}
</span>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# SECTION 1: CV WALK-FORWARD
# ══════════════════════════════════════════════════════════════════════
if section == "📊 CV Walk-forward":
    _n_folds   = cv_df['fold'].nunique()
    _val_years = sorted(cv_df['val_year'].unique()) if 'val_year' in cv_df.columns else []
    _yr_range  = f"{_val_years[0]}–{_val_years[-1]}" if _val_years else "CV"

    section_header(st, f"Walk-forward Cross-Validation ({_n_folds} folds, {_yr_range})", "📊")

    st.markdown("""
    <div class='highlight-box'>
    <b>Phương pháp Walk-forward Expanding Window:</b>
    Mỗi fold train trên toàn bộ dữ liệu lịch sử trước năm validation — đảm bảo <b>không rò rỉ thông tin tương lai</b>.
    Khác với k-fold thông thường, phương pháp này mô phỏng deployment thực tế: mô hình chỉ được phép "biết" quá khứ.
    Hyperparameter tuning thực hiện bằng <b>nested RandomSearch (n_iter=15, cv=3)</b> bên trong mỗi fold.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 3], gap="large")
    with col1:
        st.markdown("**📋 Tổng hợp kết quả CV (trung bình qua các folds)**")

        df_sum = cv_summary.reset_index()
        df_sum.columns = ["Model","AUC mean","PR-AUC","F1","Accuracy","AUC std"]
        df_sum = df_sum[["Model","AUC mean","AUC std","PR-AUC","F1","Accuracy"]]
        df_sum = df_sum.sort_values("AUC mean", ascending=False)

        # ✅ Không dùng .style — render bằng HTML thuần
        rows_html = ""
        for _, row in df_sum.iterrows():
            is_best = row["Model"] == _best_name
            bg = "rgba(46,204,113,0.08)" if is_best else "transparent"
            star = " ⭐" if is_best else ""
            rows_html += f"""
            <tr style='background:{bg};'>
              <td style='color:#5bc0be;font-weight:{"700" if is_best else "400"};'>{row["Model"]}{star}</td>
              <td style='color:#e0e6ed;font-family:Space Mono,monospace;'>{row["AUC mean"]:.4f}</td>
              <td style='color:#6b7f99;font-family:Space Mono,monospace;'>±{row["AUC std"]:.4f}</td>
              <td style='color:#a8b8cc;font-family:Space Mono,monospace;'>{row["PR-AUC"]:.4f}</td>
              <td style='color:#a8b8cc;font-family:Space Mono,monospace;'>{row["F1"]:.4f}</td>
              <td style='color:#a8b8cc;font-family:Space Mono,monospace;'>{row["Accuracy"]:.4f}</td>
            </tr>"""
        st.markdown(f"""
        <div style='overflow-x:auto; overflow-y:auto; max-height:260px;
                    border:1px solid rgba(91,192,190,0.15); border-radius:8px;
                    background:#131f38; padding:0 4px;'>
        <table style='width:100%;border-collapse:collapse;font-size:0.83rem;'>
          <thead style='position:sticky;top:0;background:#1a2a45;z-index:1;'>
            <tr style='border-bottom:1px solid rgba(91,192,190,0.3);'>
              <th style='color:#5bc0be;text-align:left;padding:6px 8px;'>Model</th>
              <th style='color:#5bc0be;text-align:left;padding:6px 8px;'>AUC</th>
              <th style='color:#5bc0be;text-align:left;padding:6px 8px;'>±Std</th>
              <th style='color:#5bc0be;text-align:left;padding:6px 8px;'>PR-AUC</th>
              <th style='color:#5bc0be;text-align:left;padding:6px 8px;'>F1</th>
              <th style='color:#5bc0be;text-align:left;padding:6px 8px;'>Acc</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)

        _n_train_min = cv_df['n_train'].min() if 'n_train' in cv_df.columns else "—"
        st.markdown(f"""
        <div class='highlight-box' style='margin-top:0.8rem;'>
        <b>Sơ đồ Walk-forward:</b><br>
        <span style='font-size:0.82rem;color:#a8b8cc;'>
        Fold 1: Train 2014–2016 → Val 2017<br>
        Fold 2: Train 2014–2017 → Val 2018<br>
        ... (expanding window)<br>
        Fold {_n_folds}: Train 2014–2023 → Val 2024<br>
        Holdout: Train 2014–2024 → Test 2025<br><br>
        Train tối thiểu: {_n_train_min} obs
        </span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # AUC per fold line chart
        fig = go.Figure()
        fold_labels = cv_pivot.index.tolist()
        if 'val_year' in cv_df.columns:
            fold_year_map = cv_df.groupby('fold')['val_year'].first().to_dict()
            fold_labels = [f"Fold {f}\n(val:{fold_year_map.get(f,'')})" for f in cv_pivot.index]

        for model in cv_pivot.columns:
            fig.add_trace(go.Scatter(
                x=fold_labels, y=cv_pivot[model].values,
                mode="lines+markers", name=model,
                line=dict(color=MODEL_COLORS.get(model,"#5bc0be"),
                          width=3.0 if model==_best_name else 1.5),
                marker=dict(size=8 if model==_best_name else 5),
                opacity=1.0 if model==_best_name else 0.6
            ))
        _auc_min = max(0.3, cv_df['AUC'].min() - 0.03)
        _auc_max = min(1.0, cv_df['AUC'].max() + 0.03)

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Arial, sans-serif", color="#c8d8e8", size=12),
            height=420,
            title=dict(text=f"AUC từng fold · {_best_name} nổi bật (đường đậm)", font=dict(color="#5bc0be")),
            legend=dict(x=0.01, y=0.01, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
            xaxis=dict(gridcolor="rgba(91,192,190,0.1)", linecolor="rgba(91,192,190,0.2)", tickangle=-30),
            yaxis=dict(gridcolor="rgba(91,192,190,0.1)", linecolor="rgba(91,192,190,0.2)",
                       title="AUC", range=[_auc_min, _auc_max]),
            margin=dict(l=40, r=20, t=50, b=60),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Boxplot phân phối AUC
    section_header(st, f"Phân phối AUC qua {_n_folds} folds — {_best_name} ổn định nhất", "")
    st.markdown("""
    <div style='font-size:0.85rem;color:#8899aa;margin-bottom:0.6rem;'>
    Boxplot thể hiện median, IQR và outlier của AUC từng model qua các folds.
    Model tốt cần AUC <b>cao</b> và <b>ổn định</b> (hộp hẹp). RF đạt cả hai tiêu chí.
    </div>
    """, unsafe_allow_html=True)

    fig_box = go.Figure()
    for model in cv_pivot.columns:
        fig_box.add_trace(go.Box(
            y=cv_pivot[model].dropna().values,
            name=model,
            marker_color=MODEL_COLORS.get(model,"#5bc0be"),
            boxmean=True,
        ))
    fig_box.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", color="#c8d8e8", size=12),
        height=360,
        title=dict(text=f"Boxplot AUC · {_n_folds} folds · {len(cv_pivot.columns)} models", font=dict(color="#5bc0be")),
        xaxis=dict(gridcolor="rgba(91,192,190,0.1)", linecolor="rgba(91,192,190,0.2)"),
        yaxis=dict(gridcolor="rgba(91,192,190,0.1)", linecolor="rgba(91,192,190,0.2)", title="AUC"),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    st.plotly_chart(fig_box, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# SECTION 2: HOLDOUT 2025
# ══════════════════════════════════════════════════════════════════════
elif section == "🧪 Holdout 2025":
    section_header(st, f"Đánh giá Holdout năm 2025 (n={_n_holdout})", "🧪")

    st.markdown(f"""
    <div class='highlight-box'>
    <b>Tập Holdout 2025 là bộ kiểm tra hoàn toàn độc lập</b> — mô hình <b>không được nhìn thấy</b>
    bất kỳ dữ liệu nào từ năm 2025 trong suốt quá trình huấn luyện và tuning.
    Đây là bằng chứng ngoài mẫu thực sự (true out-of-sample), phân biệt với CV walk-forward
    vẫn còn nằm trong khoảng thời gian có thể bị ảnh hưởng gián tiếp.
    <br><br>
    <b>{_n_holdout} quan sát</b>: {_n_pos_hold} target=1 (ROA_cao) · {_n_neg_hold} target=0 (ROA_thấp) · pos rate = {_n_pos_hold/_n_holdout*100:.1f}%
    </div>
    """, unsafe_allow_html=True)

    _gap_pp = _cv_holdout_gap * 100
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("AUC Holdout", f"{RF_HOLDOUT['AUC']:.4f}",
              f"{_gap_pp:+.1f}pp so CV",
              help="AUC = Diện tích dưới đường ROC. AUC=0.5 là ngẫu nhiên, AUC=1 là hoàn hảo. AUC~0.69 trong tài chính panel được coi là chấp nhận được.")
    c2.metric("PR-AUC", f"{RF_HOLDOUT['PR_AUC']:.4f}", "",
              help="Precision-Recall AUC — quan trọng khi lớp mất cân bằng. Baseline = tỷ lệ pos (39.4%).")
    c3.metric("F1 (ngưỡng 0.5)", f"{RF_HOLDOUT['F1']:.4f}", "",
              help="Trung bình điều hòa Precision-Recall tại ngưỡng 0.5.")
    c4.metric("Accuracy", f"{RF_HOLDOUT['Acc']*100:.1f}%", "",
              help="Tỷ lệ phân loại đúng. Baseline naive = max(39.4%, 60.6%) = 60.6%.")
    c5.metric("Gap CV→Holdout", f"{abs(_gap_pp):.1f}pp",
              "Overfitting nhẹ" if abs(_gap_pp) > 5 else "Ổn định",
              help="Gap < 10pp được coi là chấp nhận được trong dự báo tài chính panel dữ liệu cao chiều.")

    st.markdown("---")

    col1, col2 = st.columns(2, gap="large")
    with col1:
        section_header(st, "Kết quả phân loại chi tiết (Classification Report)", "")
        st.markdown("""
        <div style='font-size:0.83rem;color:#8899aa;margin-bottom:0.5rem;'>
        Precision = trong số DN được dự đoán ROA_cao, bao nhiêu % đúng thực tế.<br>
        Recall = trong số DN thực sự ROA_cao, mô hình tìm ra được bao nhiêu %.<br>
        F1 = trung bình điều hòa Precision và Recall.
        </div>
        """, unsafe_allow_html=True)

        if 'target' in pred_obs.columns and 'pred_50' in pred_obs.columns:
            from sklearn.metrics import classification_report
            _cr_dict = classification_report(
                pred_obs['target'], pred_obs['pred_50'],
                target_names=["ROA_thap","ROA_cao"], output_dict=True
            )
            # Build HTML table từ dict — không dùng <pre>
            _cr_rows = ""
            _row_defs = [
                ("ROA_thap",    "ROA_thap",    "#e74c3c", False),
                ("ROA_cao",     "ROA_cao",     "#5bc0be", False),
                ("macro avg",   "macro avg",   "#a8b8cc", True),
                ("weighted avg","weighted avg","#a8b8cc", True),
            ]
            for key, label, color, is_avg in _row_defs:
                if key not in _cr_dict:
                    continue
                r = _cr_dict[key]
                prec = float(r["precision"])
                rec  = float(r["recall"])
                f1   = float(r["f1-score"])
                sup  = int(r["support"])
                border_top = "border-top:1px solid rgba(91,192,190,0.2);" if is_avg else ""
                _cr_rows += f"""
                <tr style='background:{"rgba(255,255,255,0.02)" if is_avg else "transparent"};{border_top}'>
                  <td style='padding:8px 12px;color:{color};font-weight:{"600" if not is_avg else "400"};
                             font-family:Space Mono,monospace;font-size:0.83rem;'>{label}</td>
                  <td style='padding:8px 12px;text-align:right;font-family:Space Mono,monospace;
                             font-size:0.86rem;color:#e0e6ed;'>{prec:.2f}</td>
                  <td style='padding:8px 12px;text-align:right;font-family:Space Mono,monospace;
                             font-size:0.86rem;color:#e0e6ed;'>{rec:.2f}</td>
                  <td style='padding:8px 12px;text-align:right;font-family:Space Mono,monospace;
                             font-size:0.86rem;color:{color};font-weight:600;'>{f1:.2f}</td>
                  <td style='padding:8px 12px;text-align:right;font-family:Space Mono,monospace;
                             font-size:0.86rem;color:#6b7f99;'>{sup}</td>
                </tr>"""
            # Accuracy row riêng
            _acc_val = float(_cr_dict.get("accuracy", RF_HOLDOUT["Acc"]))
            _cr_rows += f"""
                <tr style='background:rgba(91,192,190,0.04);border-top:1px solid rgba(91,192,190,0.2);'>
                  <td style='padding:8px 12px;color:#f1c40f;font-family:Space Mono,monospace;
                             font-size:0.83rem;font-weight:600;'>accuracy</td>
                  <td colspan='2'></td>
                  <td style='padding:8px 12px;text-align:right;font-family:Space Mono,monospace;
                             font-size:0.86rem;color:#f1c40f;font-weight:700;'>{_acc_val:.2f}</td>
                  <td style='padding:8px 12px;text-align:right;font-family:Space Mono,monospace;
                             font-size:0.86rem;color:#6b7f99;'>{_n_holdout}</td>
                </tr>"""

            st.markdown(f"""
            <div style='background:#131f38; border:1px solid rgba(91,192,190,0.25);
                        border-radius:10px; overflow:hidden; margin-bottom:0.4rem;'>
              <table style='width:100%;border-collapse:collapse;'>
                <thead>
                  <tr style='background:#1a2a45;border-bottom:1px solid rgba(91,192,190,0.3);'>
                    <th style='padding:8px 12px;color:#5bc0be;text-align:left;font-size:0.8rem;
                               letter-spacing:0.08em;text-transform:uppercase;'>Nhãn</th>
                    <th style='padding:8px 12px;color:#5bc0be;text-align:right;font-size:0.8rem;
                               letter-spacing:0.08em;text-transform:uppercase;'>Precision</th>
                    <th style='padding:8px 12px;color:#5bc0be;text-align:right;font-size:0.8rem;
                               letter-spacing:0.08em;text-transform:uppercase;'>Recall</th>
                    <th style='padding:8px 12px;color:#5bc0be;text-align:right;font-size:0.8rem;
                               letter-spacing:0.08em;text-transform:uppercase;'>F1-score</th>
                    <th style='padding:8px 12px;color:#5bc0be;text-align:right;font-size:0.8rem;
                               letter-spacing:0.08em;text-transform:uppercase;'>Support</th>
                  </tr>
                </thead>
                <tbody>{_cr_rows}</tbody>
              </table>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Không tìm thấy cột pred_50 trong predictions_2025.csv")

        # Youden threshold note
        st.markdown(f"""
        <div class='card' style='margin-top:0.8rem;'>
        <b style='color:#5bc0be;'>💡 Ngưỡng Youden's J = {_youden_thr:.3f}</b><br>
        <span style='font-size:0.83rem;color:#a8b8cc;'>
        Sử dụng ngưỡng {_youden_thr:.3f} thay vì 0.5 cải thiện F1 từ {RF_HOLDOUT['F1']:.3f} lên ~0.605
        và Accuracy lên ~68.1% — cân bằng tối ưu Sensitivity (True Positive Rate) và Specificity.
        </span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        section_header(st, "Bootstrap 95% CI (B=1000) — Độ tin cậy kết quả", "")
        st.markdown("""
        <div style='font-size:0.83rem;color:#8899aa;margin-bottom:0.8rem;'>
        Bootstrap lấy mẫu lại 1000 lần từ holdout 2025 → ước lượng khoảng tin cậy 95%.
        CI hẹp = kết quả ổn định; CI rộng = biến động cao do mẫu nhỏ.
        </div>
        """, unsafe_allow_html=True)

        # Định nghĩa tiếng Việt cho metric
        metric_explain = {
            "AUC":      ("Diện tích dưới đường ROC", "#5bc0be"),
            "PR_AUC":   ("Diện tích dưới đường Precision-Recall", "#3a7bd5"),
            "F1":       ("F1 tại ngưỡng 0.5", "#2ecc71"),
            "Accuracy": ("Tỷ lệ phân loại đúng", "#f39c12"),
        }
        for _, row in boot_df.iterrows():
            metric = str(row['metric']).upper()
            mean_  = float(row['mean'])
            lo_    = float(row['ci_low'])
            hi_    = float(row['ci_high'])
            label, color = metric_explain.get(metric, (metric, "#5bc0be"))
            pct = max(0, min(100, (mean_ - 0.4) / 0.45 * 100))
            st.markdown(f"""
            <div style='margin-bottom:14px;'>
                <div style='display:flex; justify-content:space-between; font-size:0.84rem; margin-bottom:3px;'>
                    <span style='color:#e0e6ed;font-weight:600;'>{metric}
                        <span style='color:#6b7f99;font-weight:400;font-size:0.78rem;'> — {label}</span>
                    </span>
                    <span style='font-family:Space Mono,monospace;color:{color};'>{mean_:.4f}
                        <span style='color:#6b7f99;font-size:0.78rem;'> [{lo_:.4f} – {hi_:.4f}]</span>
                    </span>
                </div>
                <div style='background:rgba(58,80,107,0.4);border-radius:4px;height:7px;'>
                    <div style='background:linear-gradient(90deg,{color},{color}99);
                                height:7px;border-radius:4px;width:{pct:.0f}%;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Subgroup analysis
    st.markdown("---")
    section_header(st, "Subgroup Analysis — Hiệu năng mô hình theo nhóm con", "")
    st.markdown("""
    <div style='font-size:0.83rem;color:#8899aa;margin-bottom:0.8rem;'>
    Phân tích theo nhóm giúp xác định nhóm DN mô hình dự báo tốt/kém nhất.
    <b>Mid_cap đạt AUC cao nhất (0.754)</b> — DN có mô hình kinh doanh thuần túy BĐS, chỉ số tài chính phân biệt rõ hơn.
    Large_cap thấp nhất (0.619) vì đa dạng hoá cao làm chỉ số tổng hợp ít ý nghĩa theo quý.
    </div>
    """, unsafe_allow_html=True)

    _sg_cols = subgroup_df.columns.tolist()
    _n_col   = next((c for c in _sg_cols if c.lower() in ('n','count','n_obs')), _sg_cols[1])
    _auc_col = next((c for c in _sg_cols if 'auc' in c.lower()), None)
    _f1_col  = next((c for c in _sg_cols if 'f1' in c.lower()), None)
    _acc_col = next((c for c in _sg_cols if 'acc' in c.lower()), None)
    _grp_col = _sg_cols[0]

    if _auc_col:
        _sg_sorted = subgroup_df.sort_values(_auc_col, ascending=False).reset_index(drop=True)
        _sg_max_auc = float(_sg_sorted[_auc_col].max())
        cols = st.columns(min(len(_sg_sorted), 4))
        for col, (_, row) in zip(itertools.cycle(cols), _sg_sorted.iterrows()):
            auc_val = float(row[_auc_col])
            color = "#2ecc71" if auc_val >= _sg_max_auc * 0.95 else "#5bc0be" if auc_val >= 0.68 else "#f39c12" if auc_val >= 0.62 else "#e74c3c"
            pos_rate = float(row.get('pos_rate', row.get('% pos', 0)))
            if pos_rate <= 1: pos_rate *= 100
            with col:
                st.markdown(f"""
                <div class='card'>
                <b style='color:{color};font-size:0.88rem;'>{row[_grp_col]}</b><br>
                <span style='font-size:0.78rem;color:#6b7f99;'>N={int(row[_n_col])} · pos={pos_rate:.1f}%</span><br><br>
                <span style='font-family:Space Mono,monospace;font-size:1.15rem;color:{color};'>AUC: {auc_val:.3f}</span><br>
                {'<span style="font-size:0.82rem;color:#a8b8cc;">F1: '+f"{float(row[_f1_col]):.3f}"+'</span> · ' if _f1_col else ''}
                {'<span style="font-size:0.82rem;color:#a8b8cc;">Acc: '+f"{float(row[_acc_col]):.3f}"+'</span>' if _acc_col else ''}
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# SECTION 3: SHAP
# ══════════════════════════════════════════════════════════════════════
elif section == "🌟 SHAP Feature Importance":
    section_header(st, "SHAP Feature Importance — Diễn giải mô hình Random Forest", "🌟")

    st.markdown("""
    <div class='highlight-box'>
    <b>SHAP (SHapley Additive exPlanations)</b> đo lường đóng góp biên trung bình của từng đặc trưng
    vào xác suất dự báo <b>cho từng quan sát riêng lẻ</b> — khác với feature importance truyền thống
    chỉ cho giá trị tổng hợp toàn mô hình. Mean |SHAP value| = trung bình giá trị tuyệt đối SHAP
    trên toàn bộ 2256 quan sát, thể hiện tầm quan trọng tổng thể của mỗi đặc trưng.
    </div>
    """, unsafe_allow_html=True)

    # ── Vẽ SHAP bar chart — dùng data từ file thật, sort ascending cho horizontal bar
    shap_df = feat_imp.sort_values("importance", ascending=True).copy()

    # Bảng chú giải ý nghĩa feature
    FEAT_VI = {
        "ROA_lag1":       ("ROA quý trước (t-1)", "Tính dai dẳng lợi nhuận ngắn hạn"),
        "ROA":            ("ROA kỳ hiện tại (t)", "Hiệu quả sử dụng tài sản hiện tại"),
        "ROA_lag4":       ("ROA cùng quý năm trước (t-4)", "Tính mùa vụ — chu kỳ bàn giao dự án"),
        "NPM":            ("Biên lợi nhuận ròng", "Khả năng sinh lời trên doanh thu"),
        "SIZE":           ("Quy mô DN (log Tổng TS)", "Tác động quy mô đến hiệu quả"),
        "ICR":            ("Hệ số khả năng trả lãi", "Ngưỡng phi tuyến: ICR < 1.5 = rủi ro cao"),
        "TATO":           ("Vòng quay tổng tài sản", "Hiệu quả sử dụng toàn bộ tài sản"),
        "QR":             ("Tỷ số thanh toán nhanh", "Thanh khoản ngắn hạn"),
        "ITO":            ("Vòng quay hàng tồn kho", "Hiệu quả quản lý tồn kho"),
        "REV_GROWTH_YOY": ("Tăng trưởng DT YoY", "Động lực tăng trưởng doanh thu"),
        "has_debt":       ("Có nợ vay (binary)", "Phân tách DN có/không có lãi vay"),
        "FCF_TA":         ("Dòng tiền tự do / Tổng TS", "FCF âm phổ biến trong BĐS VN"),
        "DAR":            ("Tỷ lệ nợ/Tổng tài sản", "Cấu trúc vốn — ít biến động theo quý"),
    }

    # Màu: top 3 = xanh đậm, mid = xanh nhạt, bottom = xám
    n = len(shap_df)
    bar_colors = []
    for i, row in enumerate(shap_df.itertuples()):
        rank = n - i  # rank từ cao xuống thấp (vì sorted ascending, cuối = cao nhất)
        if rank <= 3:
            bar_colors.append("#5bc0be")
        elif rank <= 7:
            bar_colors.append("#3a7bd5")
        else:
            bar_colors.append("#3a506b")

    # Custom text: value + tên tiếng Việt ngắn
    bar_text = []
    for _, row in shap_df.iterrows():
        bar_text.append(f"{row['importance']:.4f}")

    fig_shap = go.Figure()
    fig_shap.add_trace(go.Bar(
        y=shap_df["feature"],
        x=shap_df["importance"],
        orientation="h",
        marker_color=bar_colors,
        text=bar_text,
        textposition="outside",
        textfont=dict(size=10, color="#e0e6ed", family="Space Mono, monospace"),
    ))
    _x_max = shap_df["importance"].max() * 1.25
    fig_shap.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", color="#c8d8e8", size=12),
        height=520,
        title=dict(text=f"{_best_name} — Mean |SHAP value| · TreeSHAP trên 2,256 quan sát", font=dict(color="#5bc0be")),
        xaxis=dict(gridcolor="rgba(91,192,190,0.1)", linecolor="rgba(91,192,190,0.2)",
                   title="Mean |SHAP value|", range=[0, _x_max]),
        yaxis=dict(gridcolor="rgba(91,192,190,0.1)", linecolor="rgba(91,192,190,0.2)",
                   tickfont=dict(family="Space Mono, monospace", size=11)),
        margin=dict(l=130, r=100, t=60, b=40),
    )
    st.plotly_chart(fig_shap, use_container_width=True)

    # ── Diễn giải chi tiết theo từng nhóm ─────────────────────────────────────
    col1, col2 = st.columns(2, gap="large")

    with col1:
        section_header(st, "Top 5 features — Diễn giải kinh tế", "")
        top5 = shap_df.sort_values("importance", ascending=False).head(5)
        max_val = float(shap_df["importance"].max())
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            feat = row["feature"]
            val  = float(row["importance"])
            pct_of_max = val / max_val * 100
            vi_name, vi_explain = FEAT_VI.get(feat, (feat, ""))
            _total_shap = shap_df["importance"].sum()
            _pct_total  = val / _total_shap * 100
            st.markdown(f"""
            <div style='margin-bottom:14px;'>
                <div style='display:flex; justify-content:space-between; margin-bottom:4px;align-items:baseline;'>
                    <div>
                        <b style='color:#e0e6ed;'>{i}. {feat}</b>
                        <span style='font-size:0.78rem;color:#6b7f99;'> — {vi_name}</span>
                    </div>
                    <span style='font-family:Space Mono,monospace;color:#5bc0be;font-size:0.95rem;'>{val:.4f}
                        <span style='color:#6b7f99;font-size:0.75rem;'> ({_pct_total:.1f}%)</span>
                    </span>
                </div>
                <div style='background:#1a2540;border-radius:4px;height:8px;margin-bottom:4px;'>
                    <div style='background:linear-gradient(90deg,#5bc0be,#3a7bd5);
                                height:8px;border-radius:4px;width:{pct_of_max:.0f}%;'></div>
                </div>
                <span style='font-size:0.78rem;color:#6b7f99;font-style:italic;'>{vi_explain}</span>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        section_header(st, "Nhóm features & Insight nghiên cứu", "")

        # Tính nhóm từ data thật
        _roa_feats  = ["ROA_lag1","ROA","ROA_lag4"]
        _roa_shap   = shap_df[shap_df['feature'].isin(_roa_feats)]['importance'].sum()
        _total_shap = shap_df['importance'].sum()
        _roa_pct    = _roa_shap / _total_shap * 100

        _roa_lag1_val = float(shap_df[shap_df['feature']=='ROA_lag1']['importance'].iloc[0]) if 'ROA_lag1' in shap_df['feature'].values else 0.1083
        _roa_val      = float(shap_df[shap_df['feature']=='ROA']['importance'].iloc[0])      if 'ROA'      in shap_df['feature'].values else 0.0887
        _roa_lag4_val = float(shap_df[shap_df['feature']=='ROA_lag4']['importance'].iloc[0]) if 'ROA_lag4' in shap_df['feature'].values else 0.0587
        _npm_val      = float(shap_df[shap_df['feature']=='NPM']['importance'].iloc[0])       if 'NPM'      in shap_df['feature'].values else 0.0241
        _icr_val      = float(shap_df[shap_df['feature']=='ICR']['importance'].iloc[0])       if 'ICR'      in shap_df['feature'].values else 0.0091
        _dar_val      = float(shap_df[shap_df['feature']=='DAR']['importance'].iloc[0])       if 'DAR'      in shap_df['feature'].values else 0.0011

        st.markdown(f"""
        <div class='highlight-box'>
        <b style='color:#5bc0be;'>📌 Nhóm Sinh lời (ROA Persistence)</b><br>
        <span style='font-size:0.85rem;color:#a8b8cc;'>
        ROA_lag1 ({_roa_lag1_val:.4f}) + ROA ({_roa_val:.4f}) + ROA_lag4 ({_roa_lag4_val:.4f})<br>
        → Chiếm <b style='color:#5bc0be;'>{_roa_pct:.1f}%</b> tổng SHAP importance<br>
        Xác nhận <b>Giả thuyết H3</b>: tính dai dẳng lợi nhuận là cơ chế dự báo nổi trội nhất.<br>
        ROA_lag4 cao hơn NPM ({_npm_val:.4f}) — phản ánh chu kỳ bàn giao dự án theo năm đặc thù ngành BĐS VN.
        </span>
        </div>
        <div class='card' style='margin-top:0.8rem;'>
        <b style='color:#f39c12;'>⚠️ Sự phân kỳ SHAP vs Spearman ρ của ICR</b><br>
        <span style='font-size:0.83rem;color:#a8b8cc;'>
        ICR đứng <b>thứ 3 theo Spearman</b> (ρ=0.368) nhưng chỉ <b>thứ 6 theo SHAP</b> ({_icr_val:.4f}).<br>
        Lý do: ICR có <b>threshold effect phi tuyến</b> — tác động cận biên giảm dần sau ngưỡng an toàn,
        khiến SHAP mean bị kéo giảm khi tính bình quân trên toàn phân phối.
        ICR rất quan trọng ở vùng giá trị cực đoan (ICR &lt; 1).
        </span>
        </div>
        <div class='card' style='margin-top:0.8rem;'>
        <b style='color:#6b7f99;'>📉 DAR xếp áp chót (SHAP = {_dar_val:.4f})</b><br>
        <span style='font-size:0.83rem;color:#a8b8cc;'>
        Nhất quán với tương quan không có ý nghĩa (p=0.191) — cấu trúc vốn dài hạn ít thay đổi theo quý
        và ít phân biệt được xu hướng ROA ngắn hạn trong ngành vốn có DAR cao đồng đều (~47.6%).
        </span>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# SECTION 4: KIỂM ĐỊNH THỐNG KÊ
# ══════════════════════════════════════════════════════════════════════
elif section == "🔬 Kiểm định thống kê":
    section_header(st, "Kiểm định thống kê — Xác nhận độ tin cậy mô hình", "🔬")

    tab1, tab2, tab3 = st.tabs(["DeLong Test", "Friedman + Nemenyi", "Threshold Analysis"])

    with tab1:
        st.markdown(f"**DeLong Test: So sánh AUC của {_best_name} vs các model khác trên Holdout 2025**")
        st.markdown("""
        <div style='font-size:0.83rem;color:#8899aa;margin-bottom:0.8rem;'>
        <b>DeLong Test</b> kiểm định xem sự chênh lệch AUC giữa hai mô hình có ý nghĩa thống kê không.
        H₀: AUC(RF) = AUC(model_khác). p &lt; 0.05 → bác bỏ H₀, RF vượt trội có ý nghĩa.
        </div>
        """, unsafe_allow_html=True)

        # Render bảng DeLong không dùng .style
        dl_rows = ""
        for _, row in delong_df.iterrows():
            p_val = float(row.get('p', row.get('p_value', 1.0)))
            color = "#2ecc71" if p_val < 0.05 else "#6b7f99"
            cells = ""
            for col in delong_df.columns:
                cell_color = color if col.lower() in ('p','p_value','sig') else '#c8d8e8'
                ff = "Space Mono,monospace" if isinstance(row[col], (int, float)) else "Arial"
                cells += f"<td style='padding:5px 10px;color:{cell_color};font-family:{ff};font-size:0.82rem;'>{row[col]}</td>"
            dl_rows += f"<tr>{cells}</tr>"
        headers = "".join(f"<th style='padding:5px 10px;color:#5bc0be;text-align:left;border-bottom:1px solid rgba(91,192,190,0.3);'>{c}</th>" for c in delong_df.columns)
        st.markdown(f"""
        <div style='overflow-x:auto;'>
        <table style='width:100%;border-collapse:collapse;font-size:0.82rem;'>
          <thead><tr>{headers}</tr></thead>
          <tbody>{dl_rows}</tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)

        _sig_col = next((c for c in delong_df.columns if 'sig' in c.lower()), None)
        _n_sig   = int((delong_df[_sig_col].astype(str).str.contains(r'\*')).sum()) if _sig_col else 0
        st.markdown(f"""
        <div class='highlight-box' style='margin-top:1rem;'>
        <b>Kết quả:</b> {_n_sig}/5 cặp có p &lt; 0.05 trên holdout 2025.<br>
        <span style='font-size:0.85rem;color:#a8b8cc;'>
        {"Không có sự khác biệt AUC có ý nghĩa thống kê giữa RF và các model khác trên holdout." if _n_sig == 0
         else f"RF vượt trội có ý nghĩa thống kê trong {_n_sig}/5 so sánh."}
        <br>Tree-based models (RF, GBM, XGB) cho AUC tương đương nhau — sự phân biệt rõ hơn ở
        <b>Friedman Test trên CV folds</b> (xem tab kế tiếp).
        </span>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("**Friedman Test + Nemenyi Post-hoc (so sánh 6 models trên 8 CV folds)**")
        st.markdown("""
        <div style='font-size:0.83rem;color:#8899aa;margin-bottom:0.8rem;'>
        <b>Friedman Test</b> (non-parametric) kiểm định xem rank AUC của các model có khác biệt nhau không trên 8 folds.
        Nếu Friedman có ý nghĩa (p &lt; 0.05), <b>Nemenyi post-hoc</b> xác định cặp nào cụ thể khác nhau.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Ma trận AUC theo fold (từ cv_results.csv):**")
        auc_matrix_df = cv_pivot.copy()
        auc_matrix_df.index = [f"Fold {i}" for i in auc_matrix_df.index]

        # Render ma trận AUC dạng HTML (không dùng .style)
        mat_rows = ""
        for idx, row in auc_matrix_df.round(4).iterrows():
            mat_rows += f"<tr><td style='color:#5bc0be;font-weight:600;padding:5px 10px;'>{idx}</td>"
            for model in auc_matrix_df.columns:
                val = float(row[model])
                col_color = "#2ecc71" if val >= 0.8 else "#5bc0be" if val >= 0.7 else "#f39c12" if val >= 0.6 else "#e74c3c"
                mat_rows += f"<td style='font-family:Space Mono,monospace;color:{col_color};padding:5px 10px;'>{val:.4f}</td>"
            mat_rows += "</tr>"
        mat_headers = "<th style='color:#5bc0be;padding:5px 10px;'></th>" + "".join(
            f"<th style='color:#5bc0be;padding:5px 10px;{'font-weight:700;' if m==_best_name else ''}'>{m}{'⭐' if m==_best_name else ''}</th>"
            for m in auc_matrix_df.columns)
        st.markdown(f"""
        <div style='overflow-x:auto;margin-bottom:1rem;'>
        <table style='border-collapse:collapse;font-size:0.82rem;'>
          <thead><tr>{mat_headers}</tr></thead>
          <tbody>{mat_rows}</tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Nemenyi pairwise test:**")
        st.dataframe(nemenyi_df, use_container_width=True, height=250)

        st.markdown(f"""
        <div class='highlight-box' style='margin-top:1rem;'>
        <b>Kết quả Friedman Test:</b> χ² = 17.00, df = 5, <b>p = 0.0045</b><br>
        → Có sự khác biệt có ý nghĩa thống kê giữa {len(MODELS)} mô hình trên {cv_df['fold'].nunique()} folds CV.<br>
        <span style='font-size:0.85rem;color:#a8b8cc;'>
        RF rank trung bình = <b>2.25</b> (thấp nhất = tốt nhất) · LR_L1 = 5.25 (kém nhất) · CD = 2.6659<br>
        Xác nhận <b>Giả thuyết H2</b>: mô hình phi tuyến (RF, GBM, XGB) vượt trội so với Logistic Regression.
        </span>
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown("**Threshold Analysis — Lựa chọn ngưỡng quyết định theo mục tiêu ứng dụng**")
        st.markdown("""
        <div style='font-size:0.83rem;color:#8899aa;margin-bottom:0.8rem;'>
        Ngưỡng mặc định 0.5 không tối ưu khi lớp mất cân bằng. Ba chiến lược ngưỡng phục vụ
        các mục tiêu đầu tư khác nhau:
        </div>
        """, unsafe_allow_html=True)

        # Render bảng threshold với giải thích
        thresh_rows = ""
        for _, row in thresh_df.iterrows():
            strat = str(row.get('Strategy',''))
            thr   = float(row.get('Threshold', 0.5))
            rec   = float(row.get('Recall', 0))
            prec  = float(row.get('Precision', 0))
            f1    = float(row.get('F1', 0))
            acc   = float(row.get('Acc', row.get('Accuracy', 0)))
            is_youden = 'youden' in strat.lower() or abs(thr - 0.419) < 0.01
            bg = "rgba(91,192,190,0.06)" if is_youden else "transparent"
            thresh_rows += f"""
            <tr style='background:{bg};border-bottom:1px solid rgba(91,192,190,0.1);'>
              <td style='padding:7px 10px;color:{"#5bc0be" if is_youden else "#c8d8e8"};font-weight:{"700" if is_youden else "400"};'>{strat}{"  ← được chọn" if is_youden else ""}</td>
              <td style='padding:7px 10px;font-family:Space Mono,monospace;color:#e0e6ed;'>{thr:.3f}</td>
              <td style='padding:7px 10px;font-family:Space Mono,monospace;color:#a8b8cc;'>{prec:.3f}</td>
              <td style='padding:7px 10px;font-family:Space Mono,monospace;color:#a8b8cc;'>{rec:.3f}</td>
              <td style='padding:7px 10px;font-family:Space Mono,monospace;color:#a8b8cc;'>{f1:.3f}</td>
              <td style='padding:7px 10px;font-family:Space Mono,monospace;color:#a8b8cc;'>{acc:.3f}</td>
            </tr>"""
        t_headers = "".join(f"<th style='padding:7px 10px;color:#5bc0be;text-align:left;border-bottom:1px solid rgba(91,192,190,0.3);'>{h}</th>"
                            for h in ["Strategy","Threshold","Precision","Recall","F1","Accuracy"])
        st.markdown(f"""
        <div style='overflow-x:auto;'>
        <table style='width:100%;border-collapse:collapse;font-size:0.83rem;'>
          <thead><tr>{t_headers}</tr></thead>
          <tbody>{thresh_rows}</tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)

        _youden_thr = dl.get_youden_threshold()
        st.markdown(f"""
        <div class='highlight-box' style='margin-top:1rem;'>
        <b>Hướng dẫn chọn ngưỡng theo mục tiêu:</b><br>
        <span style='font-size:0.85rem;color:#a8b8cc;'>
        • <b style='color:#5bc0be;'>Youden's J ({_youden_thr:.3f})</b> — Cân bằng cả hai sai số.
          Phù hợp chiến lược đầu tư cân bằng, không hy sinh quá nhiều ở cả hai phía.<br>
        • <b style='color:#f39c12;'>F1-optimal (~0.242)</b> — Recall rất cao (93.2%), Precision thấp (46.6%).
          Dùng để <em>lọc danh sách theo dõi mở rộng</em>: ưu tiên không bỏ sót DN tiềm năng.<br>
        • <b style='color:#e74c3c;'>Ngưỡng 0.5</b> — Mặc định, ít tối ưu khi pos rate = 39.4%.
        </span>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# SECTION 5: FIRM HEATMAP 2025 — tone màu và thứ tự theo báo cáo
# ══════════════════════════════════════════════════════════════════════
elif section == "🏢 Dự đoán theo DN":
    section_header(st, "Xác suất P(ROA_cao) theo DN — Holdout 2025", "🏢")

    st.markdown("""
    <div class='highlight-box'>
    Heatmap thể hiện xác suất mô hình dự báo ROA quý tiếp theo của từng DN <b>vượt trung vị ngành</b>.
    Thứ tự sắp xếp: DN có xác suất <b>trung bình 4 quý cao nhất ở trên cùng</b>.
    Ba tầng tín hiệu từ báo cáo: <b style='color:#2ecc71;'>Cao ổn định (P&gt;80%)</b> · 
    <b style='color:#f39c12;'>Trung gian (40–75%)</b> · 
    <b style='color:#e74c3c;'>Thấp ổn định (P&lt;30%)</b>
    </div>
    """, unsafe_allow_html=True)

    # Detect Q cols và firm col
    _q_cols  = [c for c in pred_firm.columns if c.upper().startswith('Q') and c not in pred_firm.columns[:1]]
    _mean_col = next((c for c in pred_firm.columns if 'mean' in c.lower()), None)
    _firm_col = pred_firm.columns[0]

    if not _q_cols:
        # Thử detect theo % trong tên
        _q_cols = [c for c in pred_firm.columns if '%' in c and c != _firm_col]

    if _q_cols:
        heatmap_df = pred_firm.set_index(_firm_col)[_q_cols].copy()

        # Chuẩn hóa về [0,1] nếu đang là %
        _sample_val = heatmap_df.iloc[0,0]
        if isinstance(_sample_val, (int,float)) and float(_sample_val) > 1.0:
            heatmap_df = heatmap_df / 100.0

        # Tính mean nếu chưa có
        heatmap_df['_mean'] = heatmap_df.mean(axis=1)

        # Sort theo mean DESCENDING — DN tốt nhất ở trên, kém nhất ở dưới
        # Plotly heatmap y=[bottom...top] nên cần reverse để top DN hiện ở trên
        heatmap_df = heatmap_df.sort_values('_mean', ascending=True)  # ascending=True vì autorange reversed
        _mean_vals = heatmap_df['_mean'].values
        heatmap_df = heatmap_df.drop(columns=['_mean'])

        # ── Colorscale theo 3 tầng của báo cáo ──────────────────────────────
        # Đỏ (thấp) → vàng/cam (trung gian) → xanh lá (cao)
        # Khớp với: <30% đỏ, 40-75% vàng/cam, >80% xanh
        COLORSCALE = [
            [0.00, "#c0392b"],   # đỏ đậm — P < 20%
            [0.25, "#e74c3c"],   # đỏ — P ~25%
            [0.35, "#e67e22"],   # cam — P ~35%
            [0.45, "#f39c12"],   # vàng — P ~45%
            [0.55, "#f1c40f"],   # vàng nhạt — P ~55%
            [0.65, "#2ecc71"],   # xanh lá nhạt — P ~65%
            [0.80, "#27ae60"],   # xanh lá — P ~80%
            [1.00, "#1a8040"],   # xanh lá đậm — P ~100%
        ]

        z_vals = heatmap_df.values
        x_labels = [f"{c} 2025" if '2025' not in str(c) else str(c) for c in _q_cols]
        y_labels = heatmap_df.index.tolist()

        fig_hm = go.Figure(go.Heatmap(
            z=z_vals,
            x=x_labels,
            y=y_labels,
            colorscale=COLORSCALE,
            zmin=0, zmax=1,
            text=[[f"{v*100:.0f}%" for v in row] for row in z_vals],
            texttemplate="%{text}",
            textfont=dict(size=9, color="white"),
            colorbar=dict(
                title=dict(text="P(ROA_cao)", font=dict(color="#e0e6ed")),
                tickcolor="#e0e6ed",
                tickfont=dict(color="#e0e6ed"),
                tickformat=".0%",
                tickvals=[0, 0.3, 0.5, 0.7, 1.0],
                ticktext=["0%","30%","50%","70%","100%"],
            )
        ))

        fig_hm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Arial, sans-serif", color="#c8d8e8", size=11),
            height=880,
            title=dict(
                text=f"Xác suất dự đoán P(ROA_cao) · {_best_name} · 47 DN × 4 quý năm 2025",
                font=dict(color="#5bc0be", size=14)
            ),
            xaxis=dict(
                side="top",
                tickfont=dict(size=12, color="#e0e6ed"),
                linecolor="rgba(91,192,190,0.2)"
            ),
            yaxis=dict(
                autorange="reversed",  # DN sort cao nhất ở trên
                tickfont=dict(size=10, family="Space Mono, monospace", color="#c8d8e8"),
                linecolor="rgba(91,192,190,0.2)"
            ),
            margin=dict(l=70, r=80, t=80, b=20),
        )
        st.plotly_chart(fig_hm, use_container_width=True)

        st.markdown("---")

        # ── Phân tích 3 tầng theo báo cáo ────────────────────────────────────
        _last_q  = _q_cols[-1]
        _mean_series = pred_firm.set_index(_firm_col)[_q_cols].mean(axis=1)
        if isinstance(_mean_series.iloc[0], float) and _mean_series.iloc[0] > 1:
            _mean_series = _mean_series / 100.0

        _high_stable = _mean_series[_mean_series >= 0.80].sort_values(ascending=False)
        _mid_var     = _mean_series[(_mean_series >= 0.40) & (_mean_series < 0.80)].sort_values(ascending=False)
        _low_stable  = _mean_series[_mean_series < 0.30].sort_values(ascending=True)

        col1, col2, col3 = st.columns(3, gap="large")

        with col1:
            section_header(st, f"Nhóm Cao Ổn định (P>80%)", "🟢")
            st.markdown(f"""
            <div style='font-size:0.82rem;color:#8899aa;margin-bottom:0.5rem;'>
            {len(_high_stable)} DN · Persistence mạnh · Nhất quán lịch sử 2014–2024
            </div>
            """, unsafe_allow_html=True)
            for firm, prob in _high_stable.items():
                bar_w = min(100, int(prob * 100))
                st.markdown(f"""
                <div style='margin-bottom:7px;background:rgba(46,204,113,0.04);border-radius:6px;padding:5px 8px;border-left:3px solid #2ecc71;'>
                  <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <b style='color:#e0e6ed;font-size:0.9rem;'>{firm}</b>
                    <span style='font-family:Space Mono,monospace;color:#2ecc71;font-size:0.9rem;'>{prob*100:.1f}%</span>
                  </div>
                  <div style='background:rgba(46,204,113,0.15);border-radius:3px;height:4px;margin-top:4px;'>
                    <div style='background:#2ecc71;height:4px;border-radius:3px;width:{bar_w}%;'></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            section_header(st, f"Nhóm Trung gian (40–75%)", "🟡")
            st.markdown(f"""
            <div style='font-size:0.82rem;color:#8899aa;margin-bottom:0.5rem;'>
            {len(_mid_var)} DN · Biến động theo quý · Đặc thù ghi nhận doanh thu theo đợt
            </div>
            """, unsafe_allow_html=True)
            for firm, prob in _mid_var.items():
                bar_w = min(100, int(prob * 100))
                color = "#f39c12" if prob >= 0.55 else "#e67e22"
                st.markdown(f"""
                <div style='margin-bottom:7px;background:rgba(243,156,18,0.04);border-radius:6px;padding:5px 8px;border-left:3px solid {color};'>
                  <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <b style='color:#e0e6ed;font-size:0.9rem;'>{firm}</b>
                    <span style='font-family:Space Mono,monospace;color:{color};font-size:0.9rem;'>{prob*100:.1f}%</span>
                  </div>
                  <div style='background:rgba(243,156,18,0.15);border-radius:3px;height:4px;margin-top:4px;'>
                    <div style='background:{color};height:4px;border-radius:3px;width:{bar_w}%;'></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        with col3:
            section_header(st, f"Nhóm Thấp Ổn định (P<30%)", "🔴")
            st.markdown(f"""
            <div style='font-size:0.82rem;color:#8899aa;margin-bottom:0.5rem;'>
            {len(_low_stable)} DN · Cảnh báo sức khỏe tài chính · Theo dõi chặt
            </div>
            """, unsafe_allow_html=True)
            for firm, prob in _low_stable.items():
                bar_w = max(2, int(prob * 100))
                st.markdown(f"""
                <div style='margin-bottom:7px;background:rgba(231,76,60,0.04);border-radius:6px;padding:5px 8px;border-left:3px solid #e74c3c;'>
                  <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <b style='color:#e0e6ed;font-size:0.9rem;'>{firm}</b>
                    <span style='font-family:Space Mono,monospace;color:#e74c3c;font-size:0.9rem;'>{prob*100:.1f}%</span>
                  </div>
                  <div style='background:rgba(231,76,60,0.15);border-radius:3px;height:4px;margin-top:4px;'>
                    <div style='background:#e74c3c;height:4px;border-radius:3px;width:{bar_w}%;'></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        # ── Histogram phân phối xác suất ────────────────────────────────────
        st.markdown("---")
        section_header(st, f"Phân phối xác suất P(ROA_cao) — Q4/2025", "")

        last_q_raw = pred_firm.set_index(_firm_col)[_last_q].copy()
        if float(last_q_raw.iloc[0]) > 1.0:
            last_q_raw = last_q_raw / 100.0
        last_q_vals = last_q_raw.values

        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=last_q_vals, nbinsx=15,
            marker_color="#3a7bd5", opacity=0.75,
            name=f"Phân phối xác suất"
        ))
        fig_dist.add_vline(x=0.5,   line_dash="dash", line_color="#f1c40f", line_width=1.5,
                           annotation_text="Ngưỡng 0.5", annotation_font_color="#f1c40f")
        fig_dist.add_vline(x=_youden_thr, line_dash="dash", line_color="#5bc0be", line_width=1.5,
                           annotation_text=f"Youden {_youden_thr:.3f}", annotation_font_color="#5bc0be")
        fig_dist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Arial, sans-serif", color="#c8d8e8", size=12),
            height=280,
            title=dict(text=f"Phân phối P(ROA_cao) · {_last_q}/2025 · 47 DN", font=dict(color="#5bc0be")),
            xaxis=dict(gridcolor="rgba(91,192,190,0.1)", linecolor="rgba(91,192,190,0.2)",
                       title="Xác suất P(ROA_cao)", tickformat=".0%"),
            yaxis=dict(gridcolor="rgba(91,192,190,0.1)", linecolor="rgba(91,192,190,0.2)", title="Số DN"),
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        n_high  = int((last_q_vals > 0.5).sum())
        n_youden= int((last_q_vals > _youden_thr).sum())
        n_total = len(last_q_vals)
        c1, c2, c3 = st.columns(3)
        c1.metric(f"DN P>50% ({_last_q})", f"{n_high}/{n_total}", f"{n_high/n_total*100:.0f}%")
        c2.metric(f"DN P>Youden ({_youden_thr:.3f})", f"{n_youden}/{n_total}", f"{n_youden/n_total*100:.0f}%")
        c3.metric("Note: Q4/2025", "prob_mean ≈ 48%", "vs actual 0% → hệ thống risk",
                  help="Model không thể dự báo sự kiện hệ thống toàn ngành — hạn chế cần lưu ý khi ứng dụng thực tiễn.")

    else:
        st.warning("Không tìm thấy cột Q1/Q2/Q3/Q4 trong firm_predictions_2025.csv. Kiểm tra tên cột file.")
        st.dataframe(pred_firm.head(10), use_container_width=True)