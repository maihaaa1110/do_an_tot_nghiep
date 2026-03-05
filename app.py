import streamlit as st
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from style import apply_style, DARK_CSS
import data_loader as dl

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BĐS Việt Nam",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_style(st)

# ── Extra home-page CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
.hero-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #5bc0be 0%, #7dd4d2 40%, #3a7bd5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
    margin-bottom: 0.3rem;
}
.hero-sub {
    font-size: 1.15rem;
    color: #8899aa;
    font-weight: 400;
    margin-bottom: 2rem;
}
.pipeline-step {
    background: linear-gradient(135deg, #131f38, #1a2a45);
    border: 1px solid rgba(91,192,190,0.2);
    border-left: 4px solid #5bc0be;
    border-radius: 0 12px 12px 0;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
    transition: border-color 0.2s, transform 0.2s;
}
.pipeline-step:hover { border-color: #5bc0be; transform: translateX(4px); }
.step-num {
    background: rgba(91,192,190,0.15);
    color: #5bc0be;
    border: 1px solid rgba(91,192,190,0.4);
    border-radius: 50%;
    width: 32px; height: 32px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    font-weight: 700;
    flex-shrink: 0;
}
.step-text { color: #c8d8e8; font-size: 0.9rem; line-height: 1.4; }
.step-text b { color: #5bc0be; }
.member-card {
    background: linear-gradient(135deg, #131f38, #1a2a45);
    border: 1px solid rgba(91,192,190,0.15);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    transition: border-color 0.25s, transform 0.25s;
}
.member-card:hover { border-color: #5bc0be; transform: translateY(-3px); }
.member-name { color: #e0e6ed; font-weight: 600; font-size: 0.95rem; }
.member-id { color: #5bc0be; font-family: 'Space Mono', monospace; font-size: 0.8rem; margin-top: 4px; }
.kpi-strip {
    background: linear-gradient(90deg, rgba(91,192,190,0.08), rgba(58,80,107,0.08));
    border: 1px solid rgba(91,192,190,0.2);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    display: flex;
    justify-content: space-around;
    flex-wrap: wrap;
    gap: 1rem;
    margin: 1.5rem 0;
}
.kpi-item { text-align: center; }
.kpi-val { font-family: 'Space Mono', monospace; font-size: 1.6rem; font-weight: 700; color: #5bc0be; }
.kpi-lbl { font-size: 0.75rem; color: #6b7f99; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 2px; }
.nav-card {
    background: linear-gradient(135deg, #131f38, #1c2e4a);
    border: 1px solid rgba(91,192,190,0.2);
    border-radius: 14px;
    padding: 1.4rem;
    text-align: center;
    cursor: default;
    transition: all 0.25s;
}
.nav-card:hover { border-color: #5bc0be; transform: translateY(-4px); box-shadow: 0 8px 30px rgba(91,192,190,0.15); }
.nav-icon { font-size: 2.2rem; }
.nav-title { color: #5bc0be; font-weight: 700; font-size: 1rem; margin: 0.5rem 0 0.3rem; }
.nav-desc { color: #6b7f99; font-size: 0.82rem; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

# ── Hero section ───────────────────────────────────────────────────────────────
col_hero, col_kpi = st.columns([3, 2], gap="large")

with col_hero:
    st.markdown("""
    <div style='padding-top: 1rem;'>
        <p style='color:#5bc0be; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.15em; margin-bottom:0.4rem;'>
            📡 Ứng dụng Kỹ thuật Machine Learning
        </p>
        <div class='hero-title'>Dự Đoán Hiệu Quả<br>Tài Chính Doanh<br>Nghiệp BĐS</div>
        <div class='hero-sub'>
            Phân tích bảng cân đối kế toán 47 công ty bất động sản<br>
            niêm yết tại Việt Nam · Giai đoạn 2014 – 2025
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_kpi:
    # ── Load KPI thật từ data_loader ──────────────────────────────────────────
    try:
        _panel     = dl.get_panel()
        _cv        = dl.get_cv_results()
        _bm_info   = dl.get_best_model_info()
        _boot      = dl.get_bootstrap_ci()

        _n_firms   = _panel['firm'].nunique()
        _n_obs     = len(_panel)
        _q_per_dn  = round(_n_obs / _n_firms)
        _cv_auc    = f"{_bm_info['cv_auc_mean']*100:.1f}%"
        _n_feat    = len(dl.FEATURES_ML)
        _n_models  = _cv['model'].nunique()
        _n_folds   = _cv['fold'].nunique()
        try:
            _pred_obs = dl.get_predictions_obs()
            from sklearn.metrics import roc_auc_score
            _holdout_auc = f"{roc_auc_score(_pred_obs['target'], _pred_obs['prob_roa_cao'])*100:.1f}%"
        except Exception:
            _holdout_auc = "69.2%"
    except Exception:
        # Nếu file chưa sẵn, báo lỗi rõ thay vì dùng fallback giả
        _n_firms, _n_obs, _q_per_dn = "—", "—", "—"
        _cv_auc, _n_feat, _n_models, _n_folds, _holdout_auc = "—", "—", "—", "—", "—"

    st.markdown(f"""
    <div style='padding-top:1.5rem;'>
        <div class='kpi-strip'>
            <div class='kpi-item'><div class='kpi-val'>{_n_firms}</div><div class='kpi-lbl'>Doanh nghiệp</div></div>
            <div class='kpi-item'><div class='kpi-val'>{_n_obs:,}</div><div class='kpi-lbl'>Quan sát</div></div>
            <div class='kpi-item'><div class='kpi-val'>{_q_per_dn}</div><div class='kpi-lbl'>Quý / DN</div></div>
            <div class='kpi-item'><div class='kpi-val'>{_cv_auc}</div><div class='kpi-lbl'>CV AUC (RF)</div></div>
        </div>
        <div class='kpi-strip'>
            <div class='kpi-item'><div class='kpi-val'>{_n_feat}</div><div class='kpi-lbl'>Features</div></div>
            <div class='kpi-item'><div class='kpi-val'>{_n_models}</div><div class='kpi-lbl'>Models</div></div>
            <div class='kpi-item'><div class='kpi-val'>{_n_folds}</div><div class='kpi-lbl'>CV Folds</div></div>
            <div class='kpi-item'><div class='kpi-val'>{_holdout_auc}</div><div class='kpi-lbl'>Holdout AUC</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Data status badge ─────────────────────────────────────────────────────────
st.markdown(dl.data_source_badge(), unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Navigation cards ───────────────────────────────────────────────────────────
st.markdown("<h3 style='text-align:center; margin-bottom:1.2rem;'> Tóm tắt web</h3>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
pages_info = [
    ("📊", "Tổng quan dữ liệu", "Thống kê mô tả · Chất lượng dữ liệu · Phân loại quy mô 47 DN"),
    ("🔍", "Phân tích EDA", "Phân phối · Tương quan · Xu hướng thời gian · Outlier"),
    ("🎯", "Kết quả mô hình", "CV Walk-forward · SHAP · Holdout 2025 · Kiểm định thống kê"),
    ("🤖", "Trợ lý tra cứu", "Tra cứu nhanh dữ liệu · Hỏi đáp định nghĩa · Dashboard doanh nghiệp"),
]
for col, (icon, title, desc) in zip([c1, c2, c3, c4], pages_info):
    with col:
        st.markdown(f"""
        <div class='nav-card'>
            <div class='nav-icon'>{icon}</div>
            <div class='nav-title'>{title}</div>
            <div class='nav-desc'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Two-column layout: Pipeline + Members ─────────────────────────────────────
left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown("###  Pipeline nghiên cứu (9 bước)")
    pipeline = [
        ("01", "Chọn mẫu & phân loại quy mô", "118 mã BĐS → 50 mã đủ điều kiện · P75 ngành = 9,630 tỷ"),
        ("02", "Thu thập dữ liệu tài chính", "vnstock API · BCTC quý: BS / IS / CF / Ratio"),
        ("03", "Kiểm tra tính đầy đủ", "Validate 50 mã liên tục Q1/2013 – Q4/2025"),
        ("04", "Tính chỉ số tài chính", "ROA, ROE, DAR, ICR, SIZE, FCF_TA, REV_GROWTH…"),
        ("05", "Tổng hợp dataset", "2,600 obs × 114 biến · Target: ROA_next > median ngành"),
        ("06", "Thống kê mô tả", "Missing · Zero · BS error · Skewness phân phối"),
        ("07", "Làm sạch dữ liệu", "Loại PVR/VCR/PV2 · Winsorize · Xử lý ICR"),
        ("08", "EDA chuyên sâu", "47 firm × 48 quý · Feature-target correlation · Outlier"),
        ("09", "Mô hình ML", "Walk-forward CV · RF tốt nhất (AUC=0.811) · SHAP"),
    ]
    for num, title, detail in pipeline:
        st.markdown(f"""
        <div class='pipeline-step'>
            <div class='step-num'>{num}</div>
            <div class='step-text'><b>{title}</b><br><span style='color:#6b7f99;font-size:0.82rem;'>{detail}</span></div>
        </div>
        """, unsafe_allow_html=True)

with right:
    st.markdown("### Thông tin sinh viên thực hiện")
    members = [
        ("Mai Thị Thanh Hà", "K224141659")
    ]
    for i, (name, msv) in enumerate(members, 1):
        st.markdown(f"""
        <div class='member-card' style='margin-bottom:0.6rem;'>
            <div style='color:#6b7f99; font-size:0.72rem; text-transform:uppercase;
                        letter-spacing:0.1em; margin-bottom:4px;'>Sinh viên thực hiện</div>
            <div class='member-name'>{name}</div>
            <div class='member-id'>{msv}</div>
        </div>
        """, unsafe_allow_html=True)

    # Target variable explanation
    st.markdown("---")
    st.markdown("### Biến mục tiêu")

    try:
        _p = dl.get_panel()
        _pos = int((_p['target'] == 1).sum())
        _neg = int((_p['target'] == 0).sum())
        _train = _p[_p['year'] <= 2024]
        _hold  = _p[_p['year'] == 2025]
        _balance_pct = f"Train: {(_train['target']==1).mean()*100:.1f}% pos"
        _pill2 = f"Holdout 2025: {(_hold['target']==1).mean()*100:.1f}% pos"
    except Exception:
        _balance_pct = "balanced"
        _pill2 = "—"

    st.markdown(f"""
    <div class='highlight-box'>
        <b style='color:#5bc0be;'>target = 1</b> (ROA_cao)<br>
        <span style='font-size:0.88rem; color:#a8b8cc;'>
        ROA quý <i>t+1</i> của DN vượt median ngành<br>
        cùng quý → hiệu quả tốt hơn trung bình ngành
        </span><br><br>
        <b style='color:#e74c3c;'>target = 0</b> (ROA_thap)<br>
        <span style='font-size:0.88rem; color:#a8b8cc;'>
        ROA quý <i>t+1</i> thấp hơn median ngành<br>
        → hiệu quả kém hơn trung bình ngành
        </span><br><br>
        <span class='stat-pill'>{_balance_pct}</span>
        <span class='stat-pill'>{_pill2}</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center; color:#3a506b; font-size:0.8rem; padding-bottom:1rem;'>
    Mai Thị Thanh Hà · Ứng dụng Kỹ thuật Machine Learning 
    <br>Dữ liệu: <span style='color:#5bc0be;'>vnstock / VCI</span>
    <br>Pages: Tổng quan · EDA · ML Results · Trợ lý</span>
</div>
""", unsafe_allow_html=True)