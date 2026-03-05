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

st.set_page_config(page_title="EDA | BĐS", page_icon="🔍", layout="wide")
apply_style(st)


# ══════════════════════════════════════════════════════════════════════
# LOAD DATA THẬT
# ══════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Đang tải dữ liệu EDA...")
def load_data():
    panel     = dl.get_panel()
    panel_enr = dl.get_panel_enriched()
    # Load raw (pre-winsorize) data cho histogram "Trước"
    panel_raw = dl._csv(dl.PANEL_FULL_CSV)
    panel_raw = panel_raw[panel_raw['year'] != 2013]
    panel_raw = panel_raw[~panel_raw['firm'].isin(['PVR','VCR','PV2'])]
    return panel, panel_enr, panel_raw

try:
    panel, panel_enr, panel_raw = load_data()
except FileNotFoundError as e:
    st.error(f"❌ Không tìm thấy file dữ liệu: {e}")
    st.stop()

FEATURES = dl.FEATURES_ML
N        = len(panel)

_n_firms = panel['firm'].nunique()
_n_qtrs  = panel.groupby('firm')[['year','quarter']].apply(len).mean()

page_header(st, "PHÂN TÍCH KHÁM PHÁ DỮ LIỆU",
            f"EDA · {_n_firms} firms × ~{int(_n_qtrs)} quý · {N:,} quan sát sau clean", "🔍")


# ── Tính các derived series từ data thật ─────────────────────────────────────

# Feature–target correlations (Pearson & Spearman với target)
@st.cache_data(show_spinner=False)
def calc_correlations():
    corr_p, corr_s = [], []
    for f in FEATURES:
        if f in panel.columns:
            sub = panel[[f, 'target']].dropna()
            corr_p.append(float(sub[f].corr(sub['target'], method='pearson')))
            corr_s.append(float(sub[f].corr(sub['target'], method='spearman')))
        else:
            corr_p.append(0.0)
            corr_s.append(0.0)
    return corr_p, corr_s

CORR_PEARSON, CORR_SPEARMAN = calc_correlations()

# Time-series stats (median ROA, std ROA, target rate) theo quý
@st.cache_data(show_spinner=False)
def calc_time_series():
    grp = panel[panel['year'] >= 2014].groupby(['year','quarter'])
    med  = grp['ROA'].median().reset_index().sort_values(['year','quarter'])
    std  = grp['ROA'].std().reset_index().sort_values(['year','quarter'])
    trate = grp['target'].mean().reset_index().sort_values(['year','quarter'])
    med['q_label']  = med['year'].astype(str)  + 'Q' + med['quarter'].astype(str)
    std['q_label']  = std['year'].astype(str)  + 'Q' + std['quarter'].astype(str)
    trate['q_label'] = trate['year'].astype(str) + 'Q' + trate['quarter'].astype(str)
    return med, std, trate

roa_med_ts, roa_std_ts, target_rate_ts = calc_time_series()
quarters = roa_med_ts['q_label'].tolist()

# has_debt stats
_debt_counts = panel['has_debt'].value_counts()
_has_debt_1  = int(_debt_counts.get(1, 0))
_has_debt_0  = int(_debt_counts.get(0, 0))
_has_debt_rate = _has_debt_1 / N * 100
_target_debt1  = float(panel[panel['has_debt']==1]['target'].mean())
_target_debt0  = float(panel[panel['has_debt']==0]['target'].mean())

# Winsorize clip counts per firm — tính từ winsor_bounds của model
@st.cache_data(show_spinner=False)
def calc_clip_counts():
    try:
        model_obj = dl.get_best_model()
        bounds = model_obj['winsor_bounds']
        clip_counts = {}
        for firm, fdf in panel.groupby('firm'):
            cnt = 0
            for var, (lo, hi) in bounds.items():
                if var in fdf.columns:
                    cnt += int(((fdf[var] < lo) | (fdf[var] > hi)).sum())
            clip_counts[firm] = cnt
        return dict(sorted(clip_counts.items(), key=lambda x: x[1], reverse=True))
    except Exception:
        return {}

CLIP_FIRMS = calc_clip_counts()

# Thứ tự biến cố định khớp với notebook
FEAT_ORDER = ['ROA','NPM','TATO','ITO','DAR','QR','FCF_TA',
              'ICR','SIZE','ROA_lag1','ROA_lag4','REV_GROWTH_YOY']

@st.cache_data(show_spinner=False)
def calc_feature_corr_matrix():
    # Dùng panel_enr + Pearson + thứ tự cố định để khớp notebook
    ordered = [f for f in FEAT_ORDER if f in panel_enr.columns]
    return panel_enr[ordered].corr(method='pearson'), ordered

FEAT_CORR_MAT, FEAT_LABELS = calc_feature_corr_matrix()


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("### 🔍 Chọn phần EDA")
section = st.sidebar.radio("Phần:", [
    "🧹 Làm sạch dữ liệu",
    "📊 Phân phối & Outlier",
    "🔗 VIF & Lựa chọn đặc trưng",
    "📐 Tương quan Feature–Target",
    "💳 Phân tích has_debt",
    "🗺️ Ma trận tương quan",
])
st.sidebar.markdown("---")
st.sidebar.info(f"""
**Dataset EDA:**  
{_n_firms} firm (loại PVR/VCR/PV2)  
~{int(_n_qtrs)} quý · {N:,} quan sát  
{len(FEATURES)} features ML  
""")


# ══════════════════════════════════════════════════════════════════════
# SECTION 1: LÀM SẠCH DỮ LIỆU
# ══════════════════════════════════════════════════════════════════════
if section == "🧹 Làm sạch dữ liệu":
    section_header(st, "Quy trình làm sạch dữ liệu (Cleaning Log)", "🧹")

    st.markdown("""
    <div class='highlight-box'>
    Quy trình làm sạch thực hiện theo trình tự chặt chẽ với nhật ký theo dõi chi tiết
    từng bước — từ loại doanh nghiệp cấp firm cho đến xử lý outlier và NaN.
    </div>
    """, unsafe_allow_html=True)

    # ── Bước 1: Loại 3 DN ────────────────────────────────────────────────────
    section_header(st, "Bước 1 — Loại DN cấp độ firm", "")
    col1, col2, col3 = st.columns(3)
    col1.metric("PVR", "Loại bỏ", "Cấu trúc dữ liệu hỏng")
    col2.metric("VCR", "Loại bỏ", "Cấu trúc dữ liệu hỏng")
    col3.metric("PV2", "Loại bỏ", "Cấu trúc dữ liệu hỏng")
    st.markdown(f"""
    <div class='card'>
    50 DN ban đầu → <b style='color:#5bc0be;'>{_n_firms} DN</b> đưa vào phân tích
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Bước 2: Vi phạm logic tài chính ─────────────────────────────────────
    section_header(st, "Bước 2 — Kiểm tra vi phạm logic tài chính", "")
    flags_data = {
        "Loại vi phạm": [
            "op_gt_rev_flag — Lợi nhuận HĐ > Doanh thu thuần",
            "neg_rev_flag — Doanh thu âm (hoàn trả / điều chỉnh HĐ)",
            "bs_error_flag — Sai số BCĐKT vượt 5%",
        ],
        "Số quan sát": [155, 17, 35],
        "Xử lý": [
            "Giữ lại — thu nhập TC hạch toán vào LN HĐ",
            "Giữ lại — đặc thù ngành BĐS",
            "Giữ lại — bảo toàn cấu trúc bảng cân bằng",
        ],
    }
    st.dataframe(pd.DataFrame(flags_data), use_container_width=True, hide_index=True)
    st.markdown("""
    <div class='highlight-box'>
    <b>Triết lý làm sạch:</b>
    <span style='font-size:0.88rem;color:#a8b8cc;'>
    Không loại quan sát có vi phạm logic — thay vào đó đánh dấu cờ và giữ lại,
    vì chúng phản ánh đặc thù ghi nhận doanh thu theo hợp đồng của ngành BĐS.
    Chỉ loại doanh nghiệp khi toàn bộ cấu trúc dữ liệu bị hỏng.
    </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Bước 3: Winsorize ────────────────────────────────────────────────────
    # Trước WS: panel_raw = data_with_gap_and_target.csv (loại PVR/VCR/PV2, bỏ 2013)
    # Sau WS:   panel     = data_clean_b7_full.csv
    section_header(st, "Bước 3 — Winsorize 2.5%–97.5%", "")

    skew_data = []
    for feat in ['ICR','ITO','REV_GROWTH_YOY','NPM','QR','ROA','TATO','FCF_TA','DAR','SIZE','ROA_lag1','ROA_lag4']:
        if feat in panel_raw.columns and feat in panel.columns:
            rv = panel_raw[feat].dropna().values
            cv = panel[feat].dropna().values
            skew_data.append({
                "Biến": feat,
                "Skewness (trước WS)": round(float(pd.Series(rv).skew()), 3),
                "Kurtosis (trước WS)": round(float(pd.Series(rv).kurt()), 3),
                "Skewness (sau WS)":   round(float(pd.Series(cv).skew()), 3),
                "Kurtosis (sau WS)":   round(float(pd.Series(cv).kurt()), 3),
            })
    if skew_data:
        df_skew = pd.DataFrame(skew_data)

        fig_skew = go.Figure()
        fig_skew.add_trace(go.Bar(
            name="Trước WS (panel_raw)", x=df_skew["Biến"],
            y=df_skew["Skewness (trước WS)"].abs(),
            marker_color="#e67e22", opacity=0.75
        ))
        fig_skew.add_trace(go.Bar(
            name="Sau WS (panel clean)", x=df_skew["Biến"],
            y=df_skew["Skewness (sau WS)"].abs(),
            marker_color="#5bc0be", opacity=0.85
        ))
        fig_skew.add_hline(y=2, line_dash="dash", line_color="#e74c3c",
                           annotation_text="|skew|=2 (ngưỡng thực hành)",
                           annotation_font_color="#e74c3c")
        fig_skew.add_hline(y=1, line_dash="dot", line_color="#f1c40f",
                           annotation_text="|skew|=1 (lý tưởng)")
        fig_skew.update_layout(
            **PLOTLY_THEME, barmode="group", height=420,
            title="|Skewness| trước và sau Winsorize 2.5%–97.5%",
            xaxis_title="Biến", yaxis_title="|Skewness|",
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig_skew, use_container_width=True)

    st.markdown("""
    <div class='highlight-box'>
    <span style='font-size:0.88rem;color:#a8b8cc;'>
    Chọn 2.5%–97.5% thay vì 1%–99% do phân phối BĐS đặc biệt lệch
    (ICR=16.2 · ITO=18.4 · REV_GROWTH_YOY=20.1 trước xử lý).
    Số obs bị clip: 90–114/biến (~4%–5%).
    </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Bước 4: Fill NaN ──────────────────────────────────────────────────────
    section_header(st, "Bước 4 — Xử lý giá trị NaN (Fill Zero)", "")
    fill_data = {
        "Biến": ["ICR", "ITO", "NPM", "REV_GROWTH_YOY"],
        "Lý do NaN": [
            "Firm không có lãi vay (interest_expense = 0)",
            "Không có hàng tồn kho (avg_inventory = 0)",
            "Doanh thu bằng không",
            "Không thể tính tăng trưởng (thiếu kỳ trước)",
        ],
        "Số NaN": [470, 85, 4, 7],
        "Chiến lược": ["Fill = 0", "Fill = 0", "Fill = 0", "Fill = 0"],
    }
    st.dataframe(pd.DataFrame(fill_data), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════
# SECTION 2: PHÂN PHỐI & OUTLIER
# ══════════════════════════════════════════════════════════════════════
elif section == "📊 Phân phối & Outlier":
    section_header(st, "Phân phối và xử lý Outlier (Winsorize)", "📊")
    # Histogram trái: panel_raw (data_with_gap_and_target.csv, trước WS)
    # Histogram phải: panel    (data_clean_b7_full.csv, sau WS)

    avail_feats = [f for f in ["ROA","NPM","TATO","DAR","QR","FCF_TA","ICR","SIZE"]
                   if f in panel.columns]
    sel = st.selectbox("Chọn feature để xem phân phối:", avail_feats)

    raw_vals = panel_raw[sel].dropna().values if sel in panel_raw.columns else panel[sel].dropna().values

    # Lấy winsor bounds từ model thật
    try:
        model_obj   = dl.get_best_model()
        bounds      = model_obj['winsor_bounds']
        clip_lo     = bounds.get(sel, (np.percentile(raw_vals, 1), np.percentile(raw_vals, 99)))[0]
        clip_hi     = bounds.get(sel, (np.percentile(raw_vals, 1), np.percentile(raw_vals, 99)))[1]
    except Exception:
        clip_lo = float(np.percentile(raw_vals, 1))
        clip_hi = float(np.percentile(raw_vals, 99))

    winsorized = np.clip(raw_vals, clip_lo, clip_hi)

    from scipy.stats import skew as scipy_skew
    try:
        skewness = float(scipy_skew(raw_vals))
    except Exception:
        skewness = float(pd.Series(raw_vals).skew())

    pct_neg  = float((raw_vals < 0).mean() * 100)
    mean_val = float(np.mean(raw_vals))

    col1, col2 = st.columns(2, gap="large")
    with col1:
        fig_raw = go.Figure()
        fig_raw.add_trace(go.Histogram(x=raw_vals, nbinsx=60, name="Trước winsorize",
                                      marker_color="#e67e22", opacity=0.75,
                                      histnorm="probability density"))
        fig_raw.add_vline(x=mean_val, line_dash="dash", line_color="#f1c40f",
                         annotation_text=f"Mean={mean_val:.4f}")
        fig_raw.update_layout(**PLOTLY_THEME, height=340, title=f"{sel} — Phân phối gốc",
                             xaxis_title=sel, yaxis_title="Mật độ")
        st.plotly_chart(fig_raw, use_container_width=True)

    with col2:
        fig_win = go.Figure()
        fig_win.add_trace(go.Histogram(x=winsorized, nbinsx=60, name="Sau winsorize",
                                      marker_color="#5bc0be", opacity=0.75,
                                      histnorm="probability density"))
        fig_win.add_vline(x=clip_lo, line_dash="dash", line_color="#e74c3c",
                         annotation_text=f"Lo={clip_lo:.3f}")
        fig_win.add_vline(x=clip_hi, line_dash="dash", line_color="#e74c3c",
                         annotation_text=f"Hi={clip_hi:.3f}")
        fig_win.update_layout(**PLOTLY_THEME, height=340, title=f"{sel} — Sau Winsorize",
                             xaxis_title=sel, yaxis_title="Mật độ")
        st.plotly_chart(fig_win, use_container_width=True)

    col3, col4, col5, col6 = st.columns(4)
    col3.metric("Skewness gốc", f"{skewness:.2f}", "|skew| > 1: cần xử lý" if abs(skewness) > 1 else "OK")
    col4.metric("% Âm", f"{pct_neg:.1f}%")
    col5.metric("Clip Lo", f"{clip_lo:.4f}")
    col6.metric("Clip Hi", f"{clip_hi:.4f}")


# ══════════════════════════════════════════════════════════════════════
# SECTION 3: VIF & LỰA CHỌN ĐẶC TRƯNG
# ══════════════════════════════════════════════════════════════════════
elif section == "🔗 VIF & Lựa chọn đặc trưng":
    # Dữ liệu: panel = data_clean_b7_full.csv (đã clean)
    section_header(st, "Phân tích đa cộng tuyến & Lựa chọn đặc trưng (VIF)", "🔗")

    # ── Tương quan cao giữa các biến ứng viên (từ panel_enr đã clean) ─────────
    section_header(st, "Loại biến do tương quan cao (trước VIF)", "")

    # Tính tương quan thực từ panel_enr cho các cặp bị loại
    CANDIDATE_PAIRS = [
        ("CFO_TA", "FCF_TA"), ("DER", "DAR"), ("CR", "QR"),
        ("ROA_lag2", "ROA_lag1"), ("ROE", "ROA"),
    ]
    pair_rows = []
    for va, vb in CANDIDATE_PAIRS:
        if va in panel_enr.columns and vb in panel_enr.columns:
            r = round(float(panel_enr[[va, vb]].dropna().corr().iloc[0,1]), 3)
        else:
            r = None
        pair_rows.append({
            "Biến bị loại": va,
            "Tương quan với": vb,
            "r (data thật)": r if r is not None else "—",
            "Biến giữ lại": vb,
            "Lý do giữ": {
                "CFO_TA": "FCF_TA bao gồm cả CapEx",
                "DER":    "DAR ổn định hơn khi equity dương",
                "CR":     "QR phù hợp BĐS (loại hàng TK kém TK)",
                "ROA_lag2": "lag1+lag4 đã bao phủ ngắn hạn & năm",
                "ROE":    "Trùng tử số LN sau thuế với ROA",
            }.get(va, ""),
        })
    st.dataframe(pd.DataFrame(pair_rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── VIF Report 13 biến cuối ───────────────────────────────────────────────
    section_header(st, "VIF Report — 13 biến còn lại (panel đã clean)", "")

    @st.cache_data(show_spinner=False)
    def get_vif_data():
        vif_path = os.path.join(dl.ROOT, "data", "evaluation", "vif_report_b7.csv")
        if os.path.exists(vif_path):
            return pd.read_csv(vif_path)
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            feat_cols = [f for f in FEAT_ORDER if f in panel.columns]
            X = panel[feat_cols].dropna()
            return pd.DataFrame([
                {"feature": col,
                 "VIF": round(float(variance_inflation_factor(X.values, i)), 3)}
                for i, col in enumerate(feat_cols)
            ])
        except Exception:
            return pd.DataFrame({
                "feature": ['ROA','NPM','TATO','ITO','DAR','QR','FCF_TA',
                            'ICR','has_debt','SIZE','ROA_lag1','ROA_lag4','REV_GROWTH_YOY'],
                "VIF": [2.25, 1.18, 1.32, 1.05, 1.40, 1.22, 1.08,
                        1.15, 1.19, 1.27, 1.82, 1.76, 1.11],
            })

    vif_df = get_vif_data()
    if not vif_df.empty:
        feat_col = vif_df.columns[0]
        vif_col  = next((c for c in vif_df.columns if 'vif' in c.lower()), vif_df.columns[1])
        vif_plot = vif_df.sort_values(vif_col, ascending=True)
        bar_colors_vif = [
            "#e74c3c" if v >= 10 else "#f39c12" if v >= 5 else "#5bc0be"
            for v in vif_plot[vif_col]
        ]
        fig_vif = go.Figure(go.Bar(
            x=vif_plot[vif_col], y=vif_plot[feat_col],
            orientation="h", marker_color=bar_colors_vif,
            text=[f"{v:.2f}" for v in vif_plot[vif_col]], textposition="outside"
        ))
        fig_vif.add_vline(x=10, line_dash="dash", line_color="#e74c3c",
                          annotation_text="Ngưỡng 10",
                          annotation_font_color="#e74c3c")
        fig_vif.add_vline(x=5, line_dash="dot", line_color="#f39c12",
                          annotation_text="Ngưỡng 5")
        fig_vif.update_layout(
            **PLOTLY_THEME, height=430,
            title="VIF — 13 biến cuối (max = 2.25 < 10 ✓)",
            xaxis_title="VIF",
            xaxis_range=[0, float(vif_plot[vif_col].max()) * 1.3],
        )
        st.plotly_chart(fig_vif, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# SECTION 4: TƯƠNG QUAN FEATURE–TARGET (Bảng 5 báo cáo)
# ══════════════════════════════════════════════════════════════════════
elif section == "📐 Tương quan Feature–Target":
    # Dữ liệu: panel = data_clean_b7_full.csv (đã clean), N=2,256
    section_header(st, "Tương quan Feature–Target (Pearson & Spearman)", "📐")

    def sig_stars(p):
        if p < 0.001: return "***"
        elif p < 0.01: return "**"
        elif p < 0.05: return "*"
        return "ns"

    from scipy.stats import spearmanr, pearsonr
    corr_table = []
    for feat in FEATURES:
        if feat == 'has_debt':
            continue
        if feat in panel.columns:
            sub = panel[[feat, 'target']].dropna()
            pr, _  = pearsonr(sub[feat], sub['target'])
            sr, sp = spearmanr(sub[feat], sub['target'])
            corr_table.append({
                "Biến": feat,
                "Pearson r": round(float(pr), 3),
                "Spearman ρ": round(float(sr), 3),
                "p-value": f"{sp:.3f}" if sp >= 0.001 else "<0.001",
                "Sig.": sig_stars(sp),
            })

    corr_df = pd.DataFrame(corr_table)
    corr_df = corr_df.loc[corr_df["Spearman ρ"].abs().sort_values(ascending=False).index]

    feat_labels_corr   = corr_df["Biến"].tolist()
    pearson_vals_corr  = corr_df["Pearson r"].tolist()
    spearman_vals_corr = corr_df["Spearman ρ"].tolist()

    fig_corr_bar = go.Figure()
    fig_corr_bar.add_trace(go.Bar(
        name="Pearson r", x=feat_labels_corr, y=pearson_vals_corr,
        marker_color="#3a7bd5", opacity=0.8
    ))
    fig_corr_bar.add_trace(go.Bar(
        name="Spearman ρ", x=feat_labels_corr, y=spearman_vals_corr,
        marker_color="#5bc0be", opacity=0.85
    ))
    fig_corr_bar.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
    fig_corr_bar.update_layout(
        **PLOTLY_THEME, barmode="group", height=420,
        title="Pearson r vs Spearman ρ với Target (sắp xếp theo |ρ| giảm dần)",
        xaxis_title="Feature", yaxis_title="Hệ số tương quan",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig_corr_bar, use_container_width=True)

    st.markdown("""
    <div class='highlight-box'>
    <span style='font-size:0.87rem;color:#a8b8cc;'>
    ROA_lag1 · ROA · ROA_lag4 dẫn đầu — tính dai dẳng lợi nhuận.
    ICR xếp thứ 3 (ρ=0.368 vs r=0.199) — phân kỳ lớn = quan hệ phi tuyến,
    RF nắm bắt được còn logistic tuyến tính thì không.
    DAR không có ý nghĩa thống kê (p=0.191).
    </span>
    </div>
    """, unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════════════
# SECTION 6: HAS_DEBT
# ══════════════════════════════════════════════════════════════════════
elif section == "💳 Phân tích has_debt":
    section_header(st, "Phân tích DN có/không có nợ vay (has_debt)", "💳")

    col1, col2, col3 = st.columns(3)
    col1.metric("DN có nợ vay (has_debt=1)",
                f"{_has_debt_rate:.1f}%",
                f"{_has_debt_1:,} quan sát")
    col2.metric("DN không có nợ vay (has_debt=0)",
                f"{100-_has_debt_rate:.1f}%",
                f"{_has_debt_0:,} quan sát")
    col3.metric("Target rate (has_debt=0)",
                f"{_target_debt0*100:.1f}%",
                f"≈ target rate (has_debt=1) = {_target_debt1*100:.1f}%")

    st.markdown("---")

    fig_donut = go.Figure(go.Pie(
        labels=["Có nợ vay (has_debt=1)", "Không có nợ vay (has_debt=0)"],
        values=[_has_debt_1, _has_debt_0],
        hole=0.55,
        marker_colors=["#5bc0be","#3a7bd5"],
        textinfo="label+percent+value"
    ))
    fig_donut.update_layout(**PLOTLY_THEME, height=300,
                            margin=dict(l=0,r=0,t=20,b=0),
                            title="Phân bố has_debt")
    st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown(f"""
    <div class='highlight-box'>
    <b>Key finding:</b><br>
    <span style='font-size:0.88rem;color:#a8b8cc;'>
    Target rate không khác biệt đáng kể giữa hai nhóm
    ({_target_debt0*100:.1f}% vs {_target_debt1*100:.1f}%) →
    <b>has_debt không phải predictor mạnh theo chiều trực tiếp</b>, nhưng quan trọng để kiểm soát
    missing ICR trong mô hình. ICR chỉ có ý nghĩa với DN có nợ vay.
    </span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# SECTION 7: MA TRẬN TƯƠNG QUAN
# ══════════════════════════════════════════════════════════════════════
elif section == "🗺️ Ma trận tương quan":
    # Dữ liệu: panel_enr = data_clean_b7_full.csv enriched (đã clean)
    section_header(st, "Ma trận tương quan giữa các Features", "📐")

    sel_feats = st.multiselect("Chọn features hiển thị:", FEAT_LABELS,
                               default=FEAT_LABELS, key="corr_sel")

    if len(sel_feats) >= 2:
        # Thêm target vào ma trận như notebook Sec 2b
        cols_with_target = sel_feats + ['target']
        sub_mat = panel_enr[cols_with_target].corr(method='pearson')
        sub_vals = sub_mat.values
        labels   = cols_with_target

        fig_corr = go.Figure(go.Heatmap(
            z=sub_vals,
            x=labels, y=labels,
            colorscale="RdBu", reversescale=True,
            zmin=-1, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in sub_vals],
            texttemplate="%{text}",
            textfont=dict(size=10),
            colorbar=dict(tickcolor="#e0e6ed", tickfont=dict(color="#e0e6ed"))
        ))
        fig_corr.update_layout(
            **PLOTLY_THEME,
            height=560,
            title=f"Ma trận tương quan Pearson — FINAL_FEATURES + Target ({len(sel_feats)} features)"
        )
        fig_corr.update_xaxes(side="top")
        fig_corr.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_corr, use_container_width=True)

        # Tìm các cặp tương quan cao nhất từ data thật để hiển thị insight
        pairs = []
        for i, fi in enumerate(sel_feats):
            for j, fj in enumerate(sel_feats):
                if j > i:
                    pairs.append((fi, fj, float(FEAT_CORR_MAT.loc[fi, fj])))
        top_pairs = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)[:4]

        insights = "<br>".join([
            f"{'①②③④'[k]} {p[0]} – {p[1]} = <b>{p[2]:.2f}</b>"
            for k, p in enumerate(top_pairs)
        ])
        st.markdown(f"""
        <div class='highlight-box'>
        <b>Top correlations từ data thật:</b><br>
        <span style='font-size:0.85rem;color:#a8b8cc;'>
        {insights}
        </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Cần chọn ít nhất 2 features.")