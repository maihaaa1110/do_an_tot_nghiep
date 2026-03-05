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

st.set_page_config(page_title="Tổng quan dữ liệu | BĐS", page_icon="📊", layout="wide")
apply_style(st)
page_header(st, "TỔNG QUAN DỮ LIỆU", "47 doanh nghiệp BĐS · Q1/2014 – Q4/2025", "📊")


# ══════════════════════════════════════════════════════════════════════
# LOAD DATA THẬT
# ══════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Đang tải dữ liệu...")
def load_data():
    panel       = dl.get_panel()
    panel_enr   = dl.get_panel_enriched()
    firm_size   = dl.get_firm_size_map()
    return panel, panel_enr, firm_size

try:
    panel, panel_enr, firm_size = load_data()
    DATA_OK = True
except FileNotFoundError as e:
    st.error(f"❌ Không tìm thấy file dữ liệu: {e}")
    st.stop()


# ── Derived constants ────────────────────────────────────────────────────────
TICKERS_50   = dl.TICKERS_50
TICKERS_CLEAN = dl.TICKERS_CLEAN  # 47 firms

# Firm size classification từ xlsx thật
_size_col = [c for c in firm_size.columns if 'quy' in c.lower() or 'size' in c.lower() or 'mo' in c.lower()][-1]
SIZE_LARGE = firm_size[firm_size[_size_col].str.contains('ớn', na=False)]['firm'].tolist()
SIZE_SMALL  = firm_size[firm_size[_size_col].str.contains('hỏ', na=False)]['firm'].tolist()

# P75 threshold từ firm_size xlsx (cột total_assets_bn)
_asset_col = [c for c in firm_size.columns if 'asset' in c.lower() or 'ta' in c.lower() or 'tổng' in c.lower()][0] \
    if any('asset' in c.lower() or 'ta' in c.lower() or 'tổng' in c.lower() for c in firm_size.columns) \
    else firm_size.columns[1]
P75_THRESHOLD = float(np.percentile(firm_size[_asset_col].dropna(), 75))

# Descriptive stats từ panel thật
RATIO_COLS_AVAIL = [c for c in dl.RATIO_COLS if c in panel_enr.columns]

def _desc(col):
    s = panel_enr[col].dropna()
    return {
        "mean":    round(float(s.mean()), 4),
        "std":     round(float(s.std()),  4),
        "min":     round(float(s.min()),  4),
        "max":     round(float(s.max()),  4),
        "pct_neg": round(float((s < 0).mean() * 100), 1),
    }

DESC_STATS = {c: _desc(c) for c in RATIO_COLS_AVAIL}

# Missing values từ panel_enr
_missing_raw = panel_enr[dl.FEATURES_ML + ['ICR_gap'] if 'ICR_gap' in panel_enr.columns else dl.FEATURES_ML].isnull().mean() * 100
MISSING = _missing_raw[_missing_raw > 0].sort_values(ascending=False).to_dict()

# BS Error — tính từ panel nếu có cột liên quan, ngược lại bỏ qua
_bs_col = [c for c in panel.columns if 'bs_error' in c.lower() or 'bs_err' in c.lower()]
if _bs_col:
    bs_firm = panel.groupby('firm')[_bs_col[0]].mean().sort_values(ascending=False).head(10) * 100
    BS_ERROR_FIRMS = bs_firm.index.tolist()
    BS_ERROR_VALS  = bs_firm.values.tolist()
else:
    BS_ERROR_FIRMS, BS_ERROR_VALS = [], []

# Zero revenue firms
_rev_col = [c for c in panel.columns if 'rev' in c.lower() and 'growth' not in c.lower()]
if 'revenue_zero_flag' in panel.columns:
    _zero_rev = (panel.groupby('firm')['revenue_zero_flag']
                 .apply(lambda x: (x == 1).mean() * 100)
                 .sort_values(ascending=False))
    ZERO_REVENUE_FIRMS = _zero_rev[_zero_rev > 0].to_dict()
else:
    ZERO_REVENUE_FIRMS = {}

# Industry median ROA over time (từ panel_enr thật)
_roa_trend = (panel_enr[panel_enr['year'] >= 2014]
              .groupby(['year', 'quarter'])['ROA']
              .median()
              .reset_index()
              .sort_values(['year', 'quarter']))
_roa_trend['q_label'] = _roa_trend['year'].astype(str) + 'Q' + _roa_trend['quarter'].astype(str)

# Firm-level ROA stats từ panel thật
_firm_roa_df = (panel.groupby('firm')
                .agg(mean_roa=('ROA', 'mean'),
                     std_roa=('ROA', 'std'),
                     target_rate=('target', 'mean'))
                .reset_index())
FIRM_ROA = {
    row['firm']: {
        'mean': float(row['mean_roa']),
        'std':  float(row['std_roa']),
        'target_rate': float(row['target_rate'])
    }
    for _, row in _firm_roa_df.iterrows()
}

# Target counts
_pos = int((panel['target'] == 1).sum())
_neg = int((panel['target'] == 0).sum())

# ── Ánh xạ tên cột sang tiếng Việt cho bảng data_clean_sort ─────────
COL_VI = {
    "firm":           "Mã DN",
    "year":           "Năm",
    "quarter":        "Quý",
    "ROA":            "ROA",
    "NPM":            "Biên LN ròng",
    "TATO":           "Vòng quay TS",
    "ITO":            "Vòng quay HK",
    "DAR":            "Tỷ lệ nợ/TS",
    "QR":             "Hệ số TK nhanh",
    "FCF_TA":         "FCF/Tổng TS",
    "ICR":            "Hệ số đảm bảo LS",
    "has_debt":       "Có nợ vay",
    "SIZE":           "Quy mô (log TS)",
    "ROA_lag1":       "ROA lag 1 quý",
    "ROA_lag4":       "ROA lag 4 quý",
    "REV_GROWTH_YOY": "Tăng trưởng DT YoY",
    "target":         "Mục tiêu (ROA > ngành)",
}

@st.cache_data(show_spinner="Đang tải bảng dữ liệu...")
def load_clean_csv():
    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data", "clean", "data_clean_sort.csv"
    )
    return pd.read_csv(csv_path)


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("### 🗂️ Điều hướng")
section = st.sidebar.radio("Chọn phần xem:", [
    "📋 Tổng quan dataset",
    "📏 Phân loại quy mô",
    "📉 Tỷ lệ missing dữ liệu",
    "📊 Thống kê mô tả",
    "📈 Xu hướng theo thời gian",
])
st.sidebar.markdown("---")

_n_firms   = panel['firm'].nunique()
_n_obs     = len(panel)
_q_per_dn  = round(_n_obs / _n_firms)
_n_years   = panel['year'].nunique()
_n_cols    = 114

st.sidebar.markdown(f"""
<div style='font-size:0.8rem; color:#6b7f99;'>
<b style='color:#5bc0be;'>Dataset:</b><br>
{_n_firms} DN · {_q_per_dn} quý · {_n_obs:,} obs<br>
(Loại bỏ PVR/VCR/PV2)<br>
{_n_cols} biến ban đầu<br>
{len(dl.FEATURES_ML)} features cuối cùng
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# SECTION 1: TỔNG QUAN DATASET
# ══════════════════════════════════════════════════════════════════════
if section == "📋 Tổng quan dataset":
    section_header(st, "Tổng quan Dataset", "📋")

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    kpis = [
        (str(_n_firms),         "Doanh nghiệp BĐS"),
        (f"{_n_obs:,}",         "Quan sát (sau clean)"),
        (str(_q_per_dn),        "Quý / DN"),
        (str(_n_years),         f"Năm ({panel['year'].min()}-{panel['year'].max()})"),
        (str(_n_cols),          "Biến ban đầu"),
        (str(len(dl.FEATURES_ML)), "Features ML"),
    ]
    for col, (val, lbl) in zip([k1,k2,k3,k4,k5,k6], kpis):
        col.metric(lbl, val)

    st.markdown("---")

    col_left, col_right = st.columns([2,1], gap="large")
    with col_left:
        section_header(st, "Danh sách 50 mã chứng khoán", "")
        html = "<div style='display:flex; flex-wrap:wrap; gap:6px; margin-bottom:1rem;'>"
        for t in TICKERS_50:
            is_large = t in SIZE_LARGE
            color = "#5bc0be" if is_large else "#3a506b"
            bg    = "rgba(91,192,190,0.12)" if is_large else "rgba(58,80,107,0.12)"
            html += (f"<span style='background:{bg};border:1px solid {color};color:{color};"
                     f"border-radius:6px;padding:3px 10px;font-family:Space Mono,monospace;"
                     f"font-size:0.82rem;font-weight:700;'>{t}</span>")
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)
        st.markdown(f"""
        <div class='highlight-box'>
        <span class='badge badge-teal'>Xanh đậm</span> = Doanh nghiệp <b>Lớn</b>
        (tổng tài sản ≥ P75 ngành = {P75_THRESHOLD:,.0f} tỷ VND)<br>
        <span class='badge' style='background:rgba(58,80,107,0.15);color:#3a7bd5;
        border:1px solid rgba(58,80,107,0.4);'>Xanh nhạt</span> = Doanh nghiệp <b>Nhỏ</b>
        (tổng tài sản &lt; P75 ngành)
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        section_header(st, "Cấu trúc dữ liệu", "")
        st.markdown(f"""
        <div class='card'>
            <b style='color:#5bc0be;'>Nguồn dữ liệu</b><br>
            <span style='font-size:0.85rem;color:#a8b8cc;'>vnstock API (VCI source)<br>BCTC Quý: BS · IS · CF · Ratio</span>
        </div>
        <div class='card'>
            <b style='color:#5bc0be;'>Biến mục tiêu</b><br>
            <span style='font-size:0.85rem;color:#a8b8cc;'>ROA_next > median ngành<br>Binary: 0 (thấp) / 1 (cao)<br>Phân bố: {_pos:,} pos · {_neg:,} neg</span>
        </div>
        <div class='card'>
            <b style='color:#5bc0be;'>Loại bỏ năm 2013</b><br>
            <span style='font-size:0.85rem;color:#a8b8cc;'>Tránh NaN từ lag variables<br>→ {_n_obs:,} quan sát sạch</span>
        </div>
        """, unsafe_allow_html=True)

        # Target distribution donut từ data thật
        fig_pie = go.Figure(go.Pie(
            labels=["ROA_cao (=1)", "ROA_thap (=0)"],
            values=[_pos, _neg],
            hole=0.6,
            marker_colors=["#5bc0be","#3a506b"],
            textinfo="label+percent",
            textfont=dict(size=11),
        ))
        fig_pie.update_layout(
            **PLOTLY_THEME,
            height=200, margin=dict(l=0,r=0,t=10,b=0),
            showlegend=False,
            annotations=[dict(text="Target", x=0.5, y=0.5, showarrow=False,
                             font=dict(size=13, color="#5bc0be"))]
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────
    # BẢNG DỮ LIỆU PANEL SAU XỬ LÝ (data_clean_sort.csv)
    # ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    section_header(st, "Dữ liệu panel sau xử lý", "🗃️")

    try:
        df_clean = load_clean_csv()

        # Đổi tên cột sang tiếng Việt
        df_clean_vi = df_clean.rename(columns=COL_VI)
        target_col_vi = COL_VI.get("target", "target")
        df_clean_vi["Năm"] = df_clean_vi["Năm"].astype(int).astype(str)
        df_clean_vi["Quý"] = df_clean_vi["Quý"].astype(int).astype(str)

        # ── Bộ lọc nhanh ────────────────────────────────────────────
        fc1, fc2, fc3 = st.columns([2, 2, 4])
        with fc1:
            firms_avail = ["Tất cả"] + sorted(df_clean["firm"].unique().tolist())
            sel_firm = st.selectbox("Lọc theo mã DN:", firms_avail, key="tbl_firm")
        with fc2:
            years_avail = ["Tất cả"] + sorted(df_clean["year"].unique().tolist())
            sel_year = st.selectbox("Lọc theo năm:", years_avail, key="tbl_year")
        with fc3:
            n_rows = st.slider(
                "Số dòng hiển thị:", min_value=20, max_value=2256,
                value=60, step=10, key="tbl_nrows"
            )

        # ── Áp dụng bộ lọc ──────────────────────────────────────────
        df_view = df_clean_vi.copy()
        if sel_firm != "Tất cả":
            df_view = df_view[df_view["Mã DN"] == sel_firm]
        if sel_year != "Tất cả":
            df_view = df_view[df_view["Năm"] == sel_year]
        df_view = df_view.head(n_rows).reset_index(drop=True)

        # ── Highlight cột Mục tiêu ───────────────────────────────────
        # Các cột số cần format (loại trừ id/flag/target)
        _skip_fmt = {"Năm", "Quý", "Có nợ vay", target_col_vi}
        _fmt_cols = {
            c: "{:.4f}" for c in df_view.select_dtypes(include="number").columns
            if c not in _skip_fmt
        }

        def _highlight_target(col):
            """Tô màu cột Mục tiêu theo tone nền tối."""
            if col.name != target_col_vi:
                return [""] * len(col)
            styles = []
            for v in col:
                if v == 1:
                    styles.append(
                        "background-color: rgba(91,192,190,0.22);"
                        "color: #5bc0be;"
                        "font-weight: 700;"
                    )
                else:
                    styles.append(
                        "background-color: rgba(58,80,107,0.20);"
                        "color: #7a96b8;"
                    )
            return styles

        def _highlight_target(col):
            """Tô màu ô cột Mục tiêu theo tone nền tối."""
            if col.name != target_col_vi:
                return [""] * len(col)
            return [
                (
                    "background-color: rgba(91,192,190,0.25);"
                    "color: #5bc0be;"
                    "font-weight: 700;"
                )
                if v == 1 else
                (
                    "background-color: rgba(58,80,107,0.22);"
                    "color: #7a96b8;"
                )
                for v in col
            ]

        df_show = df_view.copy()

        # format số
        for c, fmt in _fmt_cols.items():
            df_show[c] = df_show[c].map(lambda x: fmt.format(x) if pd.notna(x) else "—")

        st.dataframe(
            df_show,
            use_container_width=True,
            height=500,
            column_config={
                target_col_vi: st.column_config.NumberColumn(
                    label=target_col_vi,
                    help="1 = ROA cao hơn median ngành, 0 = thấp hơn",
                    format="%d",
                )
            }
        )
        
    except FileNotFoundError:
        st.warning(
            "⚠️ Không tìm thấy file `data/clean/data_clean_sort.csv`. "
            "Kiểm tra lại đường dẫn tương đối từ thư mục `pages/`."
        )


# ══════════════════════════════════════════════════════════════════════
# SECTION 2: PHÂN LOẠI QUY MÔ
# ══════════════════════════════════════════════════════════════════════
elif section == "📏 Phân loại quy mô":
    section_header(st, "Phân loại quy mô doanh nghiệp", "📏")

    _n_large = len(SIZE_LARGE)
    _n_small = len(SIZE_SMALL)
    _n_total = len(firm_size)

    col1, col2, col3 = st.columns(3)
    col1.metric("Ngưỡng Pháp lý", "100 tỷ VND", "Điều kiện tối thiểu")
    col2.metric("Ngưỡng P75 ngành", f"{P75_THRESHOLD:,.0f} tỷ VND", "Phân vị 75 toàn ngành")
    col3.metric("Số DN đủ điều kiện", f"{_n_total} / 118", "Lọc từ toàn bộ BĐS niêm yết")

    st.markdown("---")

    col_chart, col_info = st.columns([3,2], gap="large")
    with col_chart:
        section_header(st, f"Phân bố tổng tài sản ({_n_total} DN)", "")

        # Dùng data thật từ firm_size xlsx
        df_size = firm_size.copy()
        df_size = df_size.rename(columns={_size_col: 'Quy_Mo', _asset_col: 'total_assets_bn'})
        df_size = df_size.sort_values('total_assets_bn', ascending=False)

        fig_bar = go.Figure()
        for qm, color in [("Lớn","#5bc0be"), ("Nhỏ","#3a7bd5")]:
            sub = df_size[df_size['Quy_Mo'].str.contains(qm[:2], na=False)]
            fig_bar.add_trace(go.Bar(
                x=sub['firm'], y=sub['total_assets_bn'],
                name=f"DN {qm}", marker_color=color, opacity=0.85
            ))

        fig_bar.add_hline(y=P75_THRESHOLD, line_dash="dash", line_color="#e74c3c",
                         annotation_text=f"P75 = {P75_THRESHOLD:,.0f} tỷ", annotation_position="top right")
        fig_bar.add_hline(y=100, line_dash="dot", line_color="#f1c40f",
                         annotation_text="Ngưỡng pháp lý = 100 tỷ")
        fig_bar.update_layout(**PLOTLY_THEME, height=420,
                              yaxis_title="Tổng tài sản (tỷ VND)", xaxis_title="",
                              title="Tổng tài sản từ dữ liệu thực tế (tỷ VND)")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_info:
        section_header(st, "Tiêu chí phân loại", "")
        st.markdown(f"""
        <div class='highlight-box'>
            <b style='color:#5bc0be;'>Phân loại theo 2 ngưỡng kép:</b><br><br>
            <b>Doanh nghiệp Lớn</b> khi:<br>
            <span style='color:#a8b8cc; font-size:0.88rem;'>
            ① Tổng tài sản ≥ <b style='color:#5bc0be;'>{P75_THRESHOLD:,.0f} tỷ</b> (P75 ngành)<br>
            <i>VÀ</i><br>
            ② Tổng tài sản ≥ <b style='color:#5bc0be;'>100 tỷ</b> (ngưỡng pháp lý)
            </span><br><br>
            <b>Doanh nghiệp Nhỏ</b>: không thỏa một trong hai điều kiện trên
        </div>
        """, unsafe_allow_html=True)

        fig_donut = go.Figure(go.Pie(
            labels=["DN Lớn", "DN Nhỏ"],
            values=[_n_large, _n_small],
            hole=0.55,
            marker_colors=["#5bc0be","#3a7bd5"],
            textinfo="label+value+percent",
        ))
        fig_donut.update_layout(**PLOTLY_THEME, height=250, margin=dict(l=0,r=0,t=20,b=0),
                               title="Cơ cấu mẫu nghiên cứu")
        st.plotly_chart(fig_donut, use_container_width=True)

        st.markdown("""
        <div class='card'>
            <b style='color:#5bc0be;'>📌 Lưu ý phân tích</b><br>
            <span style='font-size:0.83rem;color:#a8b8cc;'>
            Biến SIZE = log(total_assets) được dùng trong model để kiểm soát quy mô.
            Subgroup analysis cho thấy model hoạt động tốt hơn với DN Mid-cap.
            </span>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# SECTION 3: CHẤT LƯỢNG DỮ LIỆU
# ══════════════════════════════════════════════════════════════════════
elif section == "📉 Tỷ lệ missing dữ liệu":
    section_header(st, "Phân tích tỷ lệ missing dữ liệu", "📉")

    col1, col2 = st.columns([3,2], gap="large")
    with col1:
        if MISSING:
            MISSING_FILTERED = {k: v for k, v in MISSING.items() if k != "ICR_gap"}
            missing_data = pd.DataFrame({
                "Biến": list(MISSING_FILTERED.keys()),
                "% Missing": list(MISSING_FILTERED.values()),
                "Phân loại": ["Cấu trúc" if v > 20 else "Lag" for v in MISSING_FILTERED.values()]
            }).sort_values("% Missing", ascending=True)

            fig_miss = go.Figure(go.Bar(
                x=missing_data["% Missing"], y=missing_data["Biến"],
                orientation="h",
                marker_color=["#e74c3c" if v > 20 else "#f39c12"
                                for v in missing_data["% Missing"]],
                text=[f"{v:.1f}%" for v in missing_data["% Missing"]],
                textposition="outside",
            ))
            fig_miss.add_vline(x=20, line_dash="dash", line_color="#e74c3c",
                                annotation_text="Ngưỡng 20%")
            fig_miss.update_layout(**PLOTLY_THEME, height=360,
                                    title="Tỷ lệ Missing theo biến",
                                    xaxis_title="% Missing", yaxis_title="")
            st.plotly_chart(fig_miss, use_container_width=True)
        else:
            st.info("Không phát hiện missing values trong features ML.")

    with col2:
        icr_miss = MISSING.get('ICR', 0)
        st.markdown(f"""
        <div class='highlight-box'>
        <b style='color:#e74c3c;'>🔴 ICR: {icr_miss:.2f}% missing</b><br>
        <span style='font-size:0.85rem;color:#a8b8cc;'>
        Interest Coverage Ratio = Operating_profit / Interest_expense<br>
        Missing khi <b>interest_expense = 0</b> (DN không có nợ vay) → hợp lệ về tài chính, không phải lỗi dữ liệu
        </span>
        </div>
        <div class='highlight-box' style='margin-top:0.8rem;'>
        <b style='color:#f39c12;'>⚠️ Lag variables</b><br>
        <span style='font-size:0.85rem;color:#a8b8cc;'>
        ROA_lag1, lag2, lag4 có missing tự nhiên ở đầu chuỗi → đã xử lý bằng cách loại 2013
        </span>
        </div>
        <div class='card' style='margin-top:0.8rem;'>
        <b style='color:#5bc0be;'>✅ Xử lý:</b><br>
        <span style='font-size:0.83rem;color:#a8b8cc;'>
        Tạo biến <code>has_debt</code> = 1/0 để mã hóa ICR missing<br>
        ICR missing → điền = 0 (không có nghĩa vụ lãi vay)
        </span>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# SECTION 4: THỐNG KÊ MÔ TẢ
# ══════════════════════════════════════════════════════════════════════
elif section == "📊 Thống kê mô tả":
    section_header(st, "Thống kê mô tả các chỉ số tài chính", "📊")

    df_desc = pd.DataFrame(DESC_STATS).T.reset_index()
    df_desc.columns = ["Chỉ số","Trung bình","Độ lệch chuẩn","Min","Max","% Âm"]
    df_desc = df_desc.round(4)

    df_show = df_desc.copy()
    df_show["Trung bình"] = df_show["Trung bình"].map("{:.4f}".format)
    df_show["Độ lệch chuẩn"] = df_show["Độ lệch chuẩn"].map("{:.4f}".format)
    df_show["Min"] = df_show["Min"].map("{:.4f}".format)
    df_show["Max"] = df_show["Max"].map("{:.4f}".format)
    df_show["% Âm"] = df_show["% Âm"].map("{:.1f}%".format)

    st.dataframe(df_show, use_container_width=True, height=450)

    st.markdown("---")
    section_header(st, "Phân phối so sánh các tỷ số", "📈")

    sel_metric = st.selectbox("Chọn chỉ số:", list(DESC_STATS.keys()), key="desc_sel")
    stats = DESC_STATS[sel_metric]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Trung bình", f"{stats['mean']:.4f}")
    col2.metric("Độ lệch chuẩn", f"{stats['std']:.4f}")
    col3.metric("Min / Max", f"{stats['min']:.3f} / {stats['max']:.3f}")
    col4.metric("% Âm", f"{stats['pct_neg']:.1f}%")

    _real_vals = panel_enr[sel_metric].dropna().values

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=_real_vals, nbinsx=50, name=sel_metric,
        marker_color="#5bc0be", opacity=0.7,
        histnorm="probability density"
    ))
    fig_hist.add_vline(x=stats['mean'], line_dash="dash", line_color="#f1c40f",
                      annotation_text=f"Mean={stats['mean']:.3f}")
    if stats['pct_neg'] > 0:
        fig_hist.add_vline(x=0, line_dash="dash", line_color="#e74c3c", annotation_text="0")
    fig_hist.update_layout(**PLOTLY_THEME, height=380,
                          title=f"Phân phối thực tế: {sel_metric} ({len(_real_vals):,} obs)",
                          xaxis_title=sel_metric, yaxis_title="Mật độ")
    st.plotly_chart(fig_hist, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# SECTION 5: XU HƯỚNG THEO THỜI GIAN
# ══════════════════════════════════════════════════════════════════════
elif section == "📈 Xu hướng theo thời gian":
    section_header(st, "Xu hướng ROA ngành theo thời gian", "📈")

    col1, col2 = st.columns([3,1])
    with col2:
        show_ci    = st.checkbox("Hiển thị khoảng tin cậy", value=True)
        show_covid = st.checkbox("Đánh dấu COVID-19", value=True)

    _roa_q25 = (panel_enr[panel_enr['year'] >= 2014]
                .groupby(['year','quarter'])['ROA']
                .quantile(0.25).reset_index()
                .sort_values(['year','quarter']))
    _roa_q75 = (panel_enr[panel_enr['year'] >= 2014]
                .groupby(['year','quarter'])['ROA']
                .quantile(0.75).reset_index()
                .sort_values(['year','quarter']))

    q_labels = _roa_trend['q_label'].tolist()

    fig_trend = go.Figure()

    if show_ci:
        upper = _roa_q75['ROA'].values
        lower = _roa_q25['ROA'].values
        fig_trend.add_trace(go.Scatter(
            x=q_labels + q_labels[::-1],
            y=list(upper) + list(lower[::-1]),
            fill='toself', fillcolor='rgba(91,192,190,0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=True, name='IQR (Q25–Q75)'
        ))

    fig_trend.add_trace(go.Scatter(
        x=q_labels, y=_roa_trend['ROA'].values,
        mode='lines+markers',
        line=dict(color='#5bc0be', width=2.5),
        marker=dict(size=4),
        name='Median ROA ngành',
    ))

    if show_covid:
        fig_trend.add_vrect(x0="2020Q1", x1="2021Q2",
                           fillcolor="rgba(231,76,60,0.1)",
                           annotation_text="COVID-19",
                           annotation_position="top left",
                           line_width=0)

    fig_trend.update_layout(
        **PLOTLY_THEME,
        height=420,
        title=f"Median ROA ngành BĐS ({q_labels[0]} – {q_labels[-1]})",
        xaxis_title="Quý",
        yaxis_title="ROA",
    )
    fig_trend.update_xaxes(
        tickangle=-45,
        tickmode="array",
        tickvals=q_labels[::4],
        ticktext=q_labels[::4],
    )

    st.plotly_chart(fig_trend, use_container_width=True)
    st.markdown("---")
    section_header(st, "ROA theo firm – Top & Bottom performers", "")

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("**Top 8 DN có ROA trung bình cao nhất**")
        top_firms = sorted(FIRM_ROA.items(), key=lambda x: x[1]['mean'], reverse=True)[:8]
        fig_top = go.Figure(go.Bar(
            y=[f[0] for f in top_firms],
            x=[f[1]['mean']*100 for f in top_firms],
            orientation='h',
            marker_color="#5bc0be",
            text=[f"{f[1]['mean']*100:.2f}%" for f in top_firms],
            textposition="outside"
        ))
        fig_top.update_layout(**PLOTLY_THEME, height=280, margin=dict(l=0,r=60,t=20,b=0),
                             xaxis_title="ROA trung bình (%)", yaxis_title="")
        st.plotly_chart(fig_top, use_container_width=True)

    with col2:
        st.markdown("**Top 8 DN có tỷ lệ target=1 cao nhất**")
        top_target = sorted(FIRM_ROA.items(), key=lambda x: x[1]['target_rate'], reverse=True)[:8]
        colors_t = ["#5bc0be" if f[1]['target_rate'] >= 0.7 else "#3a7bd5" for f in top_target]
        fig_target = go.Figure(go.Bar(
            y=[f[0] for f in top_target],
            x=[f[1]['target_rate']*100 for f in top_target],
            orientation='h',
            marker_color=colors_t,
            text=[f"{f[1]['target_rate']*100:.1f}%" for f in top_target],
            textposition="outside"
        ))
        fig_target.add_vline(x=50, line_dash="dash", line_color="#f1c40f",
                            annotation_text="50% (ngẫu nhiên)")
        fig_target.update_layout(**PLOTLY_THEME, height=280, margin=dict(l=0,r=60,t=20,b=0),
                                xaxis_title="Tỷ lệ quý ROA cao hơn ngành (%)", yaxis_title="")
        st.plotly_chart(fig_target, use_container_width=True)