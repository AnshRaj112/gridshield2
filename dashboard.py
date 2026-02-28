"""
GridShield – CEO-Grade Interactive Dashboard
Multi-page Streamlit app with live penalty recalculation and train vs test comparison.
"""
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from config import (
    DOCS_DIR, FINANCIAL_CAP, MAX_RELIABILITY_VIOLATIONS,
    BIAS_LOWER_BOUND, BIAS_UPPER_BOUND, PEAK_START_HOUR, PEAK_END_HOUR,
    PENALTY_UNDER_BASE, PENALTY_OVER_BASE,
)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GridShield – Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_CHART = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 1.5rem 2rem; border-radius: 12px; margin-bottom: 1.5rem;
        color: white; text-align: center;
    }
    .main-header h1 { font-size: 2rem; font-weight: 700; margin: 0; letter-spacing: 1px; }
    .main-header p { font-size: 0.9rem; opacity: 0.8; margin: 0.3rem 0 0 0; }
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255,255,255,0.08); border-radius: 12px;
        padding: 1.2rem; text-align: center; color: white; margin-bottom: 0.8rem;
    }
    .metric-card .value { font-size: 1.6rem; font-weight: 700; margin: 0.3rem 0; }
    .metric-card .label {
        font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; opacity: 0.7;
    }
    .status-good { color: #00e676; }
    .status-warn { color: #ffa726; }
    .status-bad  { color: #ef5350; }
    .badge-pass { background: rgba(0,230,118,0.15); color: #00e676;
                  border: 1px solid #00e676; padding: 4px 12px; border-radius: 20px;
                  font-size: 0.8rem; font-weight: 600; display: inline-block; }
    .badge-fail { background: rgba(239,83,80,0.15); color: #ef5350;
                  border: 1px solid #ef5350; padding: 4px 12px; border-radius: 20px;
                  font-size: 0.8rem; font-weight: 600; display: inline-block; }
    .comparison-header {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border-radius: 8px; padding: 0.6rem 1rem; color: white;
        font-weight: 600; text-align: center; margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    state_path = os.path.join(DOCS_DIR, "pipeline_state.json")
    interval_path = os.path.join(DOCS_DIR, "interval_penalties.csv")
    train_interval_path = os.path.join(DOCS_DIR, "train_interval_penalties.csv")
    comparison_path = os.path.join(DOCS_DIR, "dataset_comparison.json")

    if not os.path.exists(state_path) or not os.path.exists(interval_path):
        st.error("⚠️ Pipeline data not found. Run `python main.py` first.")
        st.stop()

    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    test_intervals = pd.read_csv(interval_path)
    test_intervals["timestamp"] = pd.to_datetime(test_intervals["timestamp"])

    train_intervals = None
    if os.path.exists(train_interval_path):
        train_intervals = pd.read_csv(train_interval_path)
        train_intervals["timestamp"] = pd.to_datetime(train_intervals["timestamp"])

    comparison = None
    if os.path.exists(comparison_path):
        with open(comparison_path, "r", encoding="utf-8") as f:
            comparison = json.load(f)

    return state, test_intervals, train_intervals, comparison


def calc_live_penalty(df, pu_offpeak, pu_peak, po):
    """Recalculate penalties with user-adjusted rates."""
    dev = df["forecast"].values - df["actual"].values
    is_pk = df["is_peak"].values.astype(bool)
    under = dev < 0
    over = dev > 0
    pen = np.zeros(len(df))
    # Off-peak underforecast
    pen[under & ~is_pk] = np.abs(dev[under & ~is_pk]) * pu_offpeak
    # Peak underforecast
    pen[under & is_pk] = np.abs(dev[under & is_pk]) * pu_peak
    # Overforecast
    pen[over] = np.abs(dev[over]) * po
    return pen


def mc(txt, cls=""):
    """Shorthand for metric card HTML."""
    return f'<div class="metric-card">{txt}</div>'


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 : EXECUTIVE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
def page_executive_summary(state, test_iv, cap_val):
    st.markdown("## 📊 Executive Summary")

    total_penalty = test_iv["penalty"].sum()
    bt = state.get("backtest_tiered", {})
    mc_data = state.get("mc_summary", {})
    mape = bt.get("mape_pct", 0)
    violations = bt.get("reliability_violations", 0)
    bias = bt.get("forecast_bias_pct", 0)
    in_bounds = BIAS_LOWER_BOUND * 100 <= bias <= BIAS_UPPER_BOUND * 100

    # ── KPI cards ─────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        s = "status-good" if total_penalty < cap_val else "status-bad"
        st.markdown(mc(f'<div class="label">Total Penalty</div><div class="value {s}">₹{total_penalty:,.0f}</div>'), unsafe_allow_html=True)
    with c2:
        s = "status-good" if mape < 5 else "status-bad"
        st.markdown(mc(f'<div class="label">MAPE</div><div class="value {s}">{mape:.2f}%</div>'), unsafe_allow_html=True)
    with c3:
        s = "status-good" if violations <= MAX_RELIABILITY_VIOLATIONS else "status-bad"
        st.markdown(mc(f'<div class="label">Reliability Violations</div><div class="value {s}">{violations} / {MAX_RELIABILITY_VIOLATIONS}</div>'), unsafe_allow_html=True)
    with c4:
        s = "status-good" if in_bounds else "status-bad"
        st.markdown(mc(f'<div class="label">Forecast Bias</div><div class="value {s}">{bias:+.2f}%</div>'), unsafe_allow_html=True)
    with c5:
        var95 = mc_data.get("var_95", 0)
        st.markdown(mc(f'<div class="label">VaR (95%)</div><div class="value" style="color:#42a5f5">₹{var95:,.0f}</div>'), unsafe_allow_html=True)

    # ── Gauge row ─────────────────────────────────────────────────────────
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("### Financial Cap Utilization")
        cu = total_penalty / cap_val * 100
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=cu,
            number={"suffix": "%", "font": {"size": 36}},
            delta={"reference": 100, "decreasing": {"color": "#00e676"}},
            gauge={"axis": {"range": [0, max(150, cu + 20)]},
                   "steps": [{"range": [0, 90], "color": "rgba(0,230,118,0.15)"},
                             {"range": [90, max(150, cu + 20)], "color": "rgba(239,83,80,0.15)"}],
                   "threshold": {"line": {"color": "#ef5350", "width": 3}, "value": 100}}
        ))
        fig.update_layout(**DARK_CHART, height=320, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with g2:
        st.markdown("### Bias Tracking")
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=bias,
            number={"suffix": "%", "font": {"size": 36}},
            gauge={"axis": {"range": [-5, 8]},
                   "steps": [{"range": [-5, -2], "color": "rgba(239,83,80,0.25)"},
                             {"range": [-2, 3], "color": "rgba(0,230,118,0.15)"},
                             {"range": [3, 8], "color": "rgba(239,83,80,0.25)"}],
                   "threshold": {"line": {"color": "#00e676", "width": 3}, "value": 0}}
        ))
        fig.update_layout(**DARK_CHART, height=320, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # ── Monetary exposure ─────────────────────────────────────────────────
    st.markdown("### 💵 Expected Monetary Exposure")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Mean Penalty", f"₹{mc_data.get('mean_penalty', 0):,.0f}")
    e2.metric("VaR (95%)", f"₹{mc_data.get('var_95', 0):,.0f}")
    e3.metric("CVaR (95%)", f"₹{mc_data.get('cvar_95', 0):,.0f}")
    e4.metric("Cap Breach Prob", f"{mc_data.get('cap_breach_prob', 0) * 100:.1f}%")

    # ── Model Performance ─────────────────────────────────────────────────
    st.markdown("### 📋 Model Performance by Horizon")
    metrics = state.get("model_metrics", {})
    if metrics:
        rows = [{"Horizon": k, "MAE (kW)": f"{v['mae']:.1f}",
                 "RMSE (kW)": f"{v['rmse']:.1f}", "MAPE (%)": f"{v['mape']:.2f}",
                 "Bias (%)": f"{v['bias_pct']:+.2f}"} for k, v in metrics.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 : FORECAST ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
def page_forecast_analysis(state, test_iv, show_days):
    st.markdown("## 📈 Forecast Analysis")
    display = test_iv.tail(show_days * 96).copy()

    # ── Forecast vs Actual ────────────────────────────────────────────────
    st.markdown("### Demand: Actual vs Forecast")
    fig = go.Figure()
    sd = display["actual"].rolling(96, min_periods=1).std().fillna(0)
    fig.add_trace(go.Scatter(x=display["timestamp"], y=display["forecast"] + 1.96 * sd,
                             mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=display["timestamp"], y=display["forecast"] - 1.96 * sd,
                             mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(66,165,245,0.1)", name="95% CI"))
    fig.add_trace(go.Scatter(x=display["timestamp"], y=display["actual"],
                             mode="lines", name="Actual", line=dict(color="#00e676", width=1.5)))
    fig.add_trace(go.Scatter(x=display["timestamp"], y=display["forecast"],
                             mode="lines", name="Forecast", line=dict(color="#42a5f5", width=1.5)))
    fig.update_layout(**DARK_CHART, height=420, hovermode="x unified",
                      yaxis_title="Load (kW)", legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True)

    # ── Penalty + Heatmap row ─────────────────────────────────────────────
    cl, cr = st.columns(2)
    with cl:
        st.markdown("### 💰 Daily Penalty Breakdown")
        d = display.copy()
        d["date"] = d["timestamp"].dt.date
        daily = d.groupby("date")["penalty"].sum().reset_index()
        fig_b = px.bar(daily, x="date", y="penalty", color_discrete_sequence=["#ef5350"])
        fig_b.update_layout(**DARK_CHART, height=350, yaxis_title="Penalty (₹)")
        st.plotly_chart(fig_b, use_container_width=True)

    with cr:
        st.markdown("### 🔥 Risk Heatmap (Hour × Day)")
        hm = display.copy()
        hm["hour"] = hm["timestamp"].dt.hour
        hm["dow"] = hm["timestamp"].dt.day_name()
        piv = hm.pivot_table(values="penalty", index="hour", columns="dow", aggfunc="mean")
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        piv = piv.reindex(columns=[d for d in days_order if d in piv.columns])
        fig_h = go.Figure(data=go.Heatmap(z=piv.values, x=piv.columns, y=piv.index, colorscale="YlOrRd"))
        fig_h.update_layout(**DARK_CHART, height=350, yaxis=dict(dtick=2))
        st.plotly_chart(fig_h, use_container_width=True)

    # ── Cumulative penalty ────────────────────────────────────────────────
    st.markdown("### 📉 Cumulative Penalty Over Time")
    display["cum_penalty"] = display["penalty"].cumsum()
    fig_c = go.Figure()
    fig_c.add_trace(go.Scatter(x=display["timestamp"], y=display["cum_penalty"],
                               fill="tozeroy", line=dict(color="#ffa726")))
    fig_c.update_layout(**DARK_CHART, height=300, yaxis_title="₹ Cumulative")
    st.plotly_chart(fig_c, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 : TRAIN vs TEST COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
def page_train_vs_test(comparison, train_iv, test_iv, show_days):
    st.markdown("## 🔬 Train vs Test Dataset Comparison")

    if comparison is None:
        st.warning("Run `python main.py` to generate comparison data.")
        return

    tr = comparison["train"]
    te = comparison["test"]

    # ── Side-by-side dataset stats ────────────────────────────────────────
    st.markdown("### 📋 Dataset Overview")
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="comparison-header">🟢 TRAIN DATASET</div>', unsafe_allow_html=True)
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | **Records** | {tr['n_rows']:,} |
        | **Period** | {tr['date_start'][:10]} → {tr['date_end'][:10]} |
        | **Features** | {tr['n_features']} |
        | **Load Mean** | {tr['load_mean']:.1f} kW |
        | **Load Std** | {tr['load_std']:.1f} kW |
        | **Load Min** | {tr['load_min']:.1f} kW |
        | **Load Max** | {tr['load_max']:.1f} kW |
        | **Peak %** | {tr['peak_pct']:.1f}% |
        """)
    with col_r:
        st.markdown('<div class="comparison-header">🔵 TEST DATASET</div>', unsafe_allow_html=True)
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | **Records** | {te['n_rows']:,} |
        | **Period** | {te['date_start'][:10]} → {te['date_end'][:10]} |
        | **Features** | {te['n_features']} |
        | **Load Mean** | {te['load_mean']:.1f} kW |
        | **Load Std** | {te['load_std']:.1f} kW |
        | **Load Min** | {te['load_min']:.1f} kW |
        | **Load Max** | {te['load_max']:.1f} kW |
        | **Peak %** | {te['peak_pct']:.1f}% |
        """)

    # ── Performance comparison table ──────────────────────────────────────
    st.markdown("### ⚡ Performance Comparison")
    tr_m = tr.get("metrics", {})
    te_m = te.get("metrics", {})
    comp_rows = []
    metric_pairs = [
        ("MAPE (%)", "mape_pct"), ("Forecast Bias (%)", "forecast_bias_pct"),
        ("Total Penalty (₹)", "total_penalty"), ("Peak Penalty (₹)", "peak_penalty"),
        ("Off-Peak Penalty (₹)", "offpeak_penalty"),
        ("Mean Abs Deviation (kW)", "mean_abs_deviation_kw"),
        ("P95 Abs Deviation (kW)", "p95_abs_deviation_kw"),
        ("Reliability Violations", "reliability_violations"),
    ]
    for label, key in metric_pairs:
        tv = tr_m.get(key, 0)
        ev = te_m.get(key, 0)
        if "₹" in label:
            comp_rows.append({"Metric": label, "Train": f"₹{tv:,.2f}", "Test": f"₹{ev:,.2f}"})
        elif "%" in label:
            comp_rows.append({"Metric": label, "Train": f"{tv:.2f}%", "Test": f"{ev:.2f}%"})
        else:
            comp_rows.append({"Metric": label, "Train": f"{tv:.2f}", "Test": f"{ev:.2f}"})
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    # ── Penalty Report Cards (Train vs Test) ──────────────────────────────
    st.markdown("### 💰 Deviation Penalty Report")
    rpt_l, rpt_r = st.columns(2)

    with rpt_l:
        st.markdown('<div class="comparison-header">🟢 TRAIN PERIOD</div>', unsafe_allow_html=True)
        trp = tr_m.get("total_penalty", 0)
        trpk = tr_m.get("peak_penalty", 0)
        trop = tr_m.get("offpeak_penalty", 0)
        trbias = tr_m.get("forecast_bias_pct", 0)
        trp95 = tr_m.get("p95_abs_deviation_kw", 0)
        st.markdown(mc(f'<div class="label">Total Deviation Penalty</div>'
                       f'<div class="value status-warn">₹{trp:,.2f}</div>'), unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        with r1:
            st.markdown(mc(f'<div class="label">Peak-Hour Penalty</div>'
                           f'<div class="value" style="color:#ef5350">₹{trpk:,.2f}</div>'), unsafe_allow_html=True)
        with r2:
            st.markdown(mc(f'<div class="label">Off-Peak Penalty</div>'
                           f'<div class="value" style="color:#42a5f5">₹{trop:,.2f}</div>'), unsafe_allow_html=True)
        r3, r4 = st.columns(2)
        with r3:
            st.markdown(mc(f'<div class="label">Forecast Bias</div>'
                           f'<div class="value" style="color:#ffa726">{trbias:+.2f}%</div>'), unsafe_allow_html=True)
        with r4:
            st.markdown(mc(f'<div class="label">P95 Abs Deviation</div>'
                           f'<div class="value" style="color:#ce93d8">{trp95:.1f} kW</div>'), unsafe_allow_html=True)

    with rpt_r:
        st.markdown('<div class="comparison-header">🔵 TEST PERIOD</div>', unsafe_allow_html=True)
        tep = te_m.get("total_penalty", 0)
        tepk = te_m.get("peak_penalty", 0)
        teop = te_m.get("offpeak_penalty", 0)
        tebias = te_m.get("forecast_bias_pct", 0)
        tep95 = te_m.get("p95_abs_deviation_kw", 0)
        st.markdown(mc(f'<div class="label">Total Deviation Penalty</div>'
                       f'<div class="value status-warn">₹{tep:,.2f}</div>'), unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        with r1:
            st.markdown(mc(f'<div class="label">Peak-Hour Penalty</div>'
                           f'<div class="value" style="color:#ef5350">₹{tepk:,.2f}</div>'), unsafe_allow_html=True)
        with r2:
            st.markdown(mc(f'<div class="label">Off-Peak Penalty</div>'
                           f'<div class="value" style="color:#42a5f5">₹{teop:,.2f}</div>'), unsafe_allow_html=True)
        r3, r4 = st.columns(2)
        with r3:
            st.markdown(mc(f'<div class="label">Forecast Bias</div>'
                           f'<div class="value" style="color:#ffa726">{tebias:+.2f}%</div>'), unsafe_allow_html=True)
        with r4:
            st.markdown(mc(f'<div class="label">P95 Abs Deviation</div>'
                           f'<div class="value" style="color:#ce93d8">{tep95:.1f} kW</div>'), unsafe_allow_html=True)

    # ── Actual Peak-Hour Heatmap (Hour × Day) ─────────────────────────────
    st.markdown("### 🔥 Actual Load Heatmap – Peak Hours Highlighted")
    hm_l, hm_r = st.columns(2)

    for col_widget, df_src, title, cscale in [
        (hm_l, train_iv, "Train – Actual Load (Hour × Day)", "YlOrRd"),
        (hm_r, test_iv, "Test – Actual Load (Hour × Day)", "YlGnBu"),
    ]:
        with col_widget:
            st.markdown(f'<div class="comparison-header">{title}</div>', unsafe_allow_html=True)
            if df_src is not None:
                hm = df_src.copy()
                hm["hour"] = hm["timestamp"].dt.hour
                hm["dow"] = hm["timestamp"].dt.day_name()
                piv = hm.pivot_table(values="actual", index="hour", columns="dow", aggfunc="mean")
                days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                piv = piv.reindex(columns=[d for d in days_order if d in piv.columns])
                fig_hm = go.Figure(data=go.Heatmap(
                    z=piv.values, x=piv.columns, y=piv.index,
                    colorscale=cscale, colorbar_title="kW",
                ))
                # Highlight peak hours with a horizontal band
                fig_hm.add_hrect(
                    y0=PEAK_START_HOUR - 0.5, y1=PEAK_END_HOUR + 0.5,
                    line=dict(color="#ffa726", width=2, dash="dash"),
                    fillcolor="rgba(0,0,0,0)",
                    annotation_text="PEAK", annotation_position="top left",
                    annotation=dict(font=dict(color="#ffa726", size=12)),
                )
                fig_hm.update_layout(**DARK_CHART, height=400, yaxis=dict(dtick=1, title="Hour"),
                                     xaxis=dict(title=""))
                st.plotly_chart(fig_hm, use_container_width=True)

    # ── Side-by-side load distribution ────────────────────────────────────
    st.markdown("### 📊 Load Distribution Comparison")
    fig_dist = make_subplots(rows=1, cols=2, subplot_titles=("Train Load Distribution", "Test Load Distribution"))

    if train_iv is not None:
        fig_dist.add_trace(go.Histogram(x=train_iv["actual"], nbinsx=80,
                                        marker_color="#00e676", opacity=0.7, name="Train"), row=1, col=1)
    fig_dist.add_trace(go.Histogram(x=test_iv["actual"], nbinsx=80,
                                    marker_color="#42a5f5", opacity=0.7, name="Test"), row=1, col=2)
    fig_dist.update_layout(**DARK_CHART, height=350, showlegend=False)
    fig_dist.update_xaxes(title_text="Load (kW)")
    fig_dist.update_yaxes(title_text="Frequency")
    st.plotly_chart(fig_dist, use_container_width=True)

    # ── Side-by-side box plot of key stats ────────────────────────────────
    st.markdown("### 📦 Load Statistics Box Plot")
    box_data = []
    if train_iv is not None:
        for v in train_iv["actual"].values:
            box_data.append({"Dataset": "Train", "Load (kW)": v})
    for v in test_iv["actual"].values:
        box_data.append({"Dataset": "Test", "Load (kW)": v})
    box_df = pd.DataFrame(box_data)
    fig_box = px.box(box_df, x="Dataset", y="Load (kW)",
                     color="Dataset", color_discrete_map={"Train": "#00e676", "Test": "#42a5f5"})
    fig_box.update_layout(**DARK_CHART, height=350, showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

    # ── Side-by-side Forecast vs Actual ───────────────────────────────────
    st.markdown("### 📈 Forecast vs Actual — Side by Side")
    f1, f2 = st.columns(2)

    with f1:
        st.markdown('<div class="comparison-header">🟢 TRAIN (last N days)</div>', unsafe_allow_html=True)
        if train_iv is not None:
            t_disp = train_iv.tail(show_days * 96)
            fig_tl = go.Figure()
            fig_tl.add_trace(go.Scatter(x=t_disp["timestamp"], y=t_disp["actual"],
                                        mode="lines", name="Actual", line=dict(color="#00e676", width=1)))
            fig_tl.add_trace(go.Scatter(x=t_disp["timestamp"], y=t_disp["forecast"],
                                        mode="lines", name="Forecast", line=dict(color="#ffa726", width=1)))
            fig_tl.update_layout(**DARK_CHART, height=320, yaxis_title="Load (kW)")
            st.plotly_chart(fig_tl, use_container_width=True)

    with f2:
        st.markdown('<div class="comparison-header">🔵 TEST (all data)</div>', unsafe_allow_html=True)
        fig_te = go.Figure()
        fig_te.add_trace(go.Scatter(x=test_iv["timestamp"], y=test_iv["actual"],
                                    mode="lines", name="Actual", line=dict(color="#00e676", width=1)))
        fig_te.add_trace(go.Scatter(x=test_iv["timestamp"], y=test_iv["forecast"],
                                    mode="lines", name="Forecast", line=dict(color="#42a5f5", width=1)))
        fig_te.update_layout(**DARK_CHART, height=320, yaxis_title="Load (kW)")
        st.plotly_chart(fig_te, use_container_width=True)

    # ── Side-by-side Error Distribution ───────────────────────────────────
    st.markdown("### 📉 Prediction Error Distribution")
    err_fig = make_subplots(rows=1, cols=2, subplot_titles=("Train Errors", "Test Errors"))
    if train_iv is not None:
        train_err = train_iv["forecast"] - train_iv["actual"]
        err_fig.add_trace(go.Histogram(x=train_err, nbinsx=80,
                                       marker_color="#00e676", opacity=0.7), row=1, col=1)
    test_err = test_iv["forecast"] - test_iv["actual"]
    err_fig.add_trace(go.Histogram(x=test_err, nbinsx=80,
                                   marker_color="#42a5f5", opacity=0.7), row=1, col=2)
    err_fig.update_layout(**DARK_CHART, height=320, showlegend=False)
    err_fig.update_xaxes(title_text="Error (kW)")
    st.plotly_chart(err_fig, use_container_width=True)

    # ── Daily MAPE comparison ─────────────────────────────────────────────
    st.markdown("### 📅 Daily MAPE Comparison")
    daily_mapes = []
    for label, df, color in [("Train", train_iv, "#00e676"), ("Test", test_iv, "#42a5f5")]:
        if df is None:
            continue
        d = df.copy()
        d["date"] = d["timestamp"].dt.date
        d["ape"] = np.abs((d["forecast"] - d["actual"]) / d["actual"].replace(0, np.nan)) * 100
        dm = d.groupby("date")["ape"].mean().reset_index()
        dm["dataset"] = label
        dm["color"] = color
        daily_mapes.append(dm)
    if daily_mapes:
        fig_dm = go.Figure()
        for dm in daily_mapes:
            fig_dm.add_trace(go.Scatter(x=dm["date"], y=dm["ape"], mode="lines+markers",
                                        name=dm["dataset"].iloc[0],
                                        line=dict(color=dm["color"].iloc[0], width=2),
                                        marker=dict(size=4)))
        fig_dm.update_layout(**DARK_CHART, height=350, yaxis_title="MAPE (%)")
        st.plotly_chart(fig_dm, use_container_width=True)

    # ── Hourly profile comparison ─────────────────────────────────────────
    st.markdown("### 🕐 Hourly Load Profile")
    hourly_fig = go.Figure()
    for label, df, color in [("Train", train_iv, "#00e676"), ("Test", test_iv, "#42a5f5")]:
        if df is None:
            continue
        d = df.copy()
        d["hour"] = d["timestamp"].dt.hour
        hp = d.groupby("hour")["actual"].mean().reset_index()
        hourly_fig.add_trace(go.Scatter(x=hp["hour"], y=hp["actual"], mode="lines+markers",
                                        name=label, line=dict(color=color, width=2)))
    hourly_fig.update_layout(**DARK_CHART, height=350, xaxis_title="Hour of Day",
                             yaxis_title="Avg Load (kW)", xaxis=dict(dtick=2))
    st.plotly_chart(hourly_fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 : RISK  &  SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════
def page_risk_scenarios(state, cap_val, mult):
    st.markdown("## 🌪️ Risk & Scenario Analysis")

    # ── Pareto frontier ───────────────────────────────────────────────────
    st.markdown("### Pareto Frontier (Penalty vs Violations)")
    pts = state.get("pareto_points", [])
    if pts:
        pdf = pd.DataFrame(pts)
        fig_p = px.scatter(pdf, x=pdf["total_penalty"] * mult, y="reliability_violations",
                           color="forecast_bias_pct", color_continuous_scale="Viridis",
                           labels={"x": "Total Penalty", "reliability_violations": "Violations"})
        fig_p.add_hline(y=MAX_RELIABILITY_VIOLATIONS, line_dash="dash", line_color="#ef5350",
                        annotation_text="Reliability Limit")
        fig_p.update_layout(**DARK_CHART, height=400)
        st.plotly_chart(fig_p, use_container_width=True)

    cl, cr = st.columns(2)
    with cl:
        st.markdown("### Scenario Impact")
        scenarios = state.get("scenario_results", [])
        if scenarios:
            names = [s["scenario_name"] for s in scenarios]
            pens = [s["total_penalty"] * mult for s in scenarios]
            colors = ["#42a5f5" if p < cap_val else "#ef5350" for p in pens]
            fig_sc = go.Figure(go.Bar(
                x=pens, y=names, orientation="h", marker_color=colors,
                text=[f"₹{p:,.0f}" for p in pens], textposition="outside"))
            fig_sc.add_vline(x=cap_val, line_dash="dash", line_color="#ef5350", annotation_text="Cap")
            fig_sc.update_layout(**DARK_CHART, height=300, xaxis_title="Penalty (₹)")
            st.plotly_chart(fig_sc, use_container_width=True)

    with cr:
        st.markdown("### Monte Carlo Percentiles")
        mc_data = state.get("mc_summary", {})
        pctls = mc_data.get("percentiles", {})
        if pctls:
            labels = list(pctls.keys())
            vals = [v * mult for v in pctls.values()]
            colors_mc = ["#4dd0e1", "#42a5f5", "#7e57c2", "#ffa726", "#ef5350"][:len(vals)]
            fig_mc = go.Figure(go.Bar(x=labels, y=vals, marker_color=colors_mc))
            fig_mc.add_hline(y=cap_val, line_dash="dash", line_color="#ef5350",
                             annotation_text="Cap")
            fig_mc.update_layout(**DARK_CHART, height=300, yaxis_title="Penalty (₹)")
            st.plotly_chart(fig_mc, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    st.markdown(
        '<div class="main-header"><h1>⚡ GRIDSHIELD</h1>'
        '<p>Regulatory Demand Forecasting & Penalty Optimization</p></div>',
        unsafe_allow_html=True,
    )

    state, test_intervals, train_intervals, comparison = load_data()

    # ── Sidebar ───────────────────────────────────────────────────────────
    st.sidebar.markdown("## 🧭 Navigation")
    page = st.sidebar.radio("Page", [
        "Executive Summary", "Forecast Analysis",
        "Train vs Test", "Risk & Scenarios"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🎛️ Penalty Controls")
    st.sidebar.caption("Adjust cost parameters to instantly compute optimal quantiles: `q = Cu / (Cu + Co)`.")
    
    pu_offpeak = st.sidebar.number_input("🌙 Off-Peak Under-forecast (₹)",
                                  value=4.0, min_value=0.0, step=0.5)
    pu_peak = st.sidebar.number_input("☀️ Peak Under-forecast (₹)",
                                  value=6.0, min_value=0.0, step=0.5,
                                  help="Stage 2 Shock increases this to ₹6")
    po = st.sidebar.number_input("⬆️ Over-forecast Penalty (₹/kW)",
                                  value=2.0, min_value=0.0, step=0.5)
                                  
    q_offpeak = pu_offpeak / (pu_offpeak + po) if (pu_offpeak + po) > 0 else 0
    q_peak = pu_peak / (pu_peak + po) if (pu_peak + po) > 0 else 0
    st.sidebar.info(f"**Derived Target Quantiles:**\n* Off-peak: `{q_offpeak:.3f}`\n* Peak: `{q_peak:.3f}`")
    
    st.sidebar.markdown("---")
    cap_val = st.sidebar.number_input("💰 Financial Cap (₹)",
                                       value=int(FINANCIAL_CAP), step=10000)
    show_days = st.sidebar.slider("📅 Days to Display", 7, 365, 30)

    # ── Live recalculation ────────────────────────────────────────────────
    test_iv = test_intervals.copy()
    test_iv["penalty"] = calc_live_penalty(test_iv, pu_offpeak, pu_peak, po)

    train_iv = None
    if train_intervals is not None:
        train_iv = train_intervals.copy()
        train_iv["penalty"] = calc_live_penalty(train_iv, pu_offpeak, pu_peak, po)

    # Effective multiplier for scenario/pareto scaling
    base_pen = calc_live_penalty(test_intervals, 4.0, 6.0, 2.0).sum()
    current_pen = test_iv["penalty"].sum()
    mult = current_pen / base_pen if base_pen > 0 else 1.0

    # ── Page routing ──────────────────────────────────────────────────────
    if page == "Executive Summary":
        page_executive_summary(state, test_iv, cap_val)
    elif page == "Forecast Analysis":
        page_forecast_analysis(state, test_iv, show_days)
    elif page == "Train vs Test":
        page_train_vs_test(comparison, train_iv, test_iv, show_days)
    elif page == "Risk & Scenarios":
        page_risk_scenarios(state, cap_val, mult)

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div style="text-align:center;opacity:0.5;font-size:0.8rem">'
        'GridShield • Regulatory Forecasting & Risk Optimization Engine'
        '</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
