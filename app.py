"""
CartLift: Checkout Redesign A/B Test — Dashboard
==================================================
Run: streamlit run app.py

Requires data/checkout_sessions.csv (generate via `python simulate_data.py`).
"""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportions_ztest

# ----------------------------------------------------------------------------
# Config & brand
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="CartLift · Checkout A/B Test",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

CRIMSON = "#A6192E"
GOLD = "#E8A33D"
CREAM = "#FAF6EF"
CHARCOAL = "#2B2320"
MUTED = "#8C7B6B"

ALPHA = 0.05
DEFAULT_MDE_PP = 1.5  # smallest lift (in pp) worth caring about

CUSTOM_CSS = f"""
<style>
    .stApp {{
        background-color: {CREAM};
    }}
    h1, h2, h3 {{
        color: {CHARCOAL};
        font-weight: 700;
    }}
    div[data-testid="stMetric"] {{
        background-color: white;
        border: 1px solid #ECE3D6;
        border-radius: 10px;
        padding: 14px 16px;
    }}
    div[data-testid="stMetricLabel"] {{
        color: {MUTED};
    }}
    .verdict-box {{
        border-left: 4px solid {CRIMSON};
        background-color: white;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 10px;
    }}
    .verdict-ok {{
        border-left-color: {GOLD};
    }}
    .footnote {{
        color: {MUTED};
        font-size: 0.85rem;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

DATA_PATH = "data/checkout_sessions.csv"


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


if not os.path.exists(DATA_PATH):
    st.error(
        f"**No dataset found at `{DATA_PATH}`.**\n\n"
        "Generate it first by running `python simulate_data.py` from the project "
        "root, then reload this page."
    )
    st.stop()

try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Couldn't read `{DATA_PATH}`: {e}")
    st.stop()

required_cols = {"date", "device", "user_type", "arm", "converted", "page_load_ms"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Dataset is missing expected column(s): {sorted(missing)}")
    st.stop()


# ----------------------------------------------------------------------------
# Analysis functions (cached — recompute only when df or params change)
# ----------------------------------------------------------------------------
@st.cache_data
def compute_srm(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    overall = df["arm"].value_counts()
    _, p = stats.chisquare(overall.values)
    rows.append({"Scope": "Overall", "Control": overall.get("control", 0),
                 "Treatment": overall.get("treatment", 0), "p-value": p})
    for device, sub in df.groupby("device"):
        counts = sub["arm"].value_counts()
        _, p = stats.chisquare(counts.values)
        rows.append({"Scope": f"Device: {device.title()}", "Control": counts.get("control", 0),
                      "Treatment": counts.get("treatment", 0), "p-value": p})
    out = pd.DataFrame(rows)
    out["SRM Flag"] = out["p-value"].apply(lambda p: "⚠️ SRM Detected" if p < 0.01 else "✅ OK")
    return out


@st.cache_data
def compute_segments(df: pd.DataFrame, mde_pp: float) -> pd.DataFrame:
    mde = mde_pp / 100
    rows = []
    for (device, user_type), sub in df.groupby(["device", "user_type"]):
        ctrl, treat = sub[sub.arm == "control"], sub[sub.arm == "treatment"]
        n_c, n_t = len(ctrl), len(treat)
        if n_c == 0 or n_t == 0:
            continue
        x_c, x_t = ctrl["converted"].sum(), treat["converted"].sum()
        cvr_c, cvr_t = x_c / n_c, x_t / n_t

        _, p_val = proportions_ztest(np.array([x_t, x_c]), np.array([n_t, n_c]))

        h = 2 * np.arcsin(np.sqrt(min(cvr_c + mde, 0.999))) - 2 * np.arcsin(np.sqrt(cvr_c))
        power = NormalIndPower().solve_power(
            effect_size=abs(h), nobs1=min(n_c, n_t), alpha=ALPHA, ratio=1.0,
            alternative="two-sided",
        )

        rows.append({
            "Device": device.title(), "User Type": user_type.title(),
            "n (control)": n_c, "n (treatment)": n_t,
            "CVR Control": cvr_c, "CVR Treatment": cvr_t,
            "Lift (pp)": (cvr_t - cvr_c) * 100, "p-value": p_val,
            "Significant": p_val < ALPHA, "Achieved Power": power,
            "Adequately Powered": power >= 0.80,
        })
    return pd.DataFrame(rows)


@st.cache_data
def compute_load_regression(df: pd.DataFrame):
    log_load = np.log(df["page_load_ms"].clip(lower=1))
    ctrl, treat = log_load[df.arm == "control"], log_load[df.arm == "treatment"]
    _, p_val = stats.ttest_ind(treat, ctrl, equal_var=False)
    med_c = df.loc[df.arm == "control", "page_load_ms"].median()
    med_t = df.loc[df.arm == "treatment", "page_load_ms"].median()
    pct_change = (med_t - med_c) / med_c * 100
    return {
        "median_control": med_c, "median_treatment": med_t,
        "pct_change": pct_change, "p_value": p_val,
        "is_regression": (p_val < ALPHA) and (pct_change > 0),
    }


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛒 CartLift")
    st.caption("Checkout Redesign A/B Test")
    st.divider()
    mde_pp = st.slider(
        "Minimum effect worth shipping for (pp)", min_value=0.5, max_value=3.0,
        value=DEFAULT_MDE_PP, step=0.25,
        help="Used in the power analysis: the smallest lift that would justify shipping.",
    )
    st.divider()
    st.caption(
        "**Data note:** synthetic dataset engineered to reproduce a realistic "
        "sample-ratio-mismatch bug. Not real production data."
    )
    st.caption("Stack: pandas · scipy.stats · statsmodels · Streamlit · Plotly")

seg_df = compute_segments(df, mde_pp)
srm_df = compute_srm(df)
load = compute_load_regression(df)

naive_cvr = df.groupby("arm")["converted"].mean()
naive_lift_pp = (naive_cvr.get("treatment", 0) - naive_cvr.get("control", 0)) * 100

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("🛒 CartLift: Checkout Redesign A/B Test")
st.markdown(
    f"<span class='footnote'>One-page checkout vs. multi-step flow · "
    f"{len(df):,} sessions · {df['date'].nunique()} days · "
    f"<b>synthetic dataset</b> — see sidebar</span>",
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Sessions", f"{len(df):,}")
c2.metric("Naive Headline Lift", f"{naive_lift_pp:+.2f} pp",
          help="Computed before checking randomization health — see Section 1.")
c3.metric("Page Load Change", f"{load['pct_change']:+.1f}%",
          delta=f"{load['median_treatment'] - load['median_control']:.0f} ms",
          delta_color="inverse")
c4.metric("Test Duration", f"{df['date'].nunique()} days")

st.divider()

# ----------------------------------------------------------------------------
# 1. Randomization health check
# ----------------------------------------------------------------------------
st.header("1 · Randomization Health Check")
st.write(
    "Before trusting the headline number, check whether users landed in "
    "treatment/control at the expected ~50/50 rate. A significant deviation "
    "(Sample Ratio Mismatch) means randomization was broken and the naive "
    "comparison above is confounded."
)

col1, col2 = st.columns([3, 2])
with col1:
    fig_srm = go.Figure()
    fig_srm.add_trace(go.Bar(name="Control", x=srm_df["Scope"], y=srm_df["Control"],
                              marker_color=GOLD))
    fig_srm.add_trace(go.Bar(name="Treatment", x=srm_df["Scope"], y=srm_df["Treatment"],
                              marker_color=CRIMSON))
    fig_srm.update_layout(barmode="group", title="Sample Sizes by Arm",
                          plot_bgcolor="white", paper_bgcolor="white",
                          legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_srm, use_container_width=True)
with col2:
    st.dataframe(
        srm_df.style.format({"p-value": "{:.2e}"}),
        hide_index=True, use_container_width=True,
    )

flagged = srm_df[srm_df["SRM Flag"].str.contains("Detected")]
if not flagged.empty:
    scopes = ", ".join(flagged["Scope"])
    st.warning(
        f"**SRM detected in:** {scopes}. Any naive comparison including this "
        "traffic is confounded — segment before trusting the result."
    )
else:
    st.success("No sample ratio mismatch detected. Randomization looks healthy.")

st.divider()

# ----------------------------------------------------------------------------
# 2. Segmented analysis
# ----------------------------------------------------------------------------
st.header("2 · Segmented Analysis")
st.write("Two-proportion z-test within each device × user-type segment, isolating the true effect from any mix-shift.")

fig_seg = go.Figure()
for device, color in [("Mobile", CRIMSON), ("Desktop", GOLD)]:
    sub = seg_df[seg_df["Device"] == device]
    if sub.empty:
        continue
    fig_seg.add_trace(go.Bar(
        name=device, x=sub["User Type"], y=sub["Lift (pp)"], marker_color=color,
        text=sub["Lift (pp)"].round(2), textposition="outside",
    ))
fig_seg.add_hline(y=0, line_dash="dash", line_color=MUTED)
fig_seg.update_layout(barmode="group", title="True Lift by Segment (pp)",
                      plot_bgcolor="white", paper_bgcolor="white",
                      legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig_seg, use_container_width=True)

display_seg = seg_df.copy()
for col in ["CVR Control", "CVR Treatment"]:
    display_seg[col] = (display_seg[col] * 100).round(2).astype(str) + "%"
display_seg["Lift (pp)"] = display_seg["Lift (pp)"].round(2)
display_seg["Achieved Power"] = (display_seg["Achieved Power"] * 100).round(1).astype(str) + "%"
display_seg["p-value"] = display_seg["p-value"].apply(lambda p: f"{p:.2e}")
st.dataframe(display_seg, hide_index=True, use_container_width=True)

st.divider()

# ----------------------------------------------------------------------------
# 3. Power analysis — verdicts
# ----------------------------------------------------------------------------
st.header("3 · Power Analysis: \"No Effect\" vs. \"Inconclusive\"")
st.write(
    f"For segments that weren't significant, achieved power to detect a "
    f"{mde_pp:.2f}pp lift tells us whether it's a true null or just underpowered."
)

for _, row in seg_df.iterrows():
    if row["Significant"]:
        continue
    powered = row["Adequately Powered"]
    cls = "verdict-box verdict-ok" if powered else "verdict-box"
    verdict = ("Adequately powered — reasonable to treat as a true null."
               if powered else
               "**Underpowered** — inconclusive, not proven zero.")
    st.markdown(
        f"<div class='{cls}'><b>{row['Device']} / {row['User Type']}</b> — "
        f"achieved power {row['Achieved Power']*100:.1f}%. {verdict}</div>",
        unsafe_allow_html=True,
    )

if seg_df["Significant"].all():
    st.info("Every segment reached significance at the current threshold.")

st.divider()

# ----------------------------------------------------------------------------
# 4. Page load regression
# ----------------------------------------------------------------------------
st.header("4 · Page Load Regression Check")
col1, col2 = st.columns([3, 2])
with col1:
    fig_load = go.Figure()
    fig_load.add_trace(go.Box(y=df.loc[df.arm == "control", "page_load_ms"],
                              name="Control", marker_color=GOLD))
    fig_load.add_trace(go.Box(y=df.loc[df.arm == "treatment", "page_load_ms"],
                              name="Treatment", marker_color=CRIMSON))
    fig_load.update_layout(title="Page Load Time Distribution (ms)",
                           plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig_load, width='stretch')
with col2:
    st.metric("Median — Control", f"{load['median_control']:.0f} ms")
    st.metric("Median — Treatment", f"{load['median_treatment']:.0f} ms",
              delta=f"{load['pct_change']:+.1f}%", delta_color="inverse")
    st.caption(f"Welch's t-test p-value: {load['p_value']:.2e}")
    if load["is_regression"]:
        st.error(
            "The one-page checkout is measurably slower to load. This is a "
            "real cost that should be weighed against the conversion gain."
        )
    else:
        st.success("No significant load-time regression detected.")

st.divider()

# ----------------------------------------------------------------------------
# 5. Recommendation
# ----------------------------------------------------------------------------
st.header("✅ Recommendation")

sig_mobile = seg_df[(seg_df.Device == "Mobile") & (seg_df.Significant)]
underpowered = seg_df[(~seg_df.Significant) & (~seg_df["Adequately Powered"])]

rec_lines = []
if not sig_mobile.empty:
    lo, hi = sig_mobile["Lift (pp)"].min(), sig_mobile["Lift (pp)"].max()
    rec_lines.append(f"**Ship on mobile.** Significant lift of +{lo:.1f} to +{hi:.1f}pp across mobile segments.")
if not underpowered.empty:
    segs = ", ".join(f"{r.Device}/{r['User Type']}" for _, r in underpowered.iterrows())
    rec_lines.append(f"**Hold {segs} for a clean re-test** — inconclusive, not proven zero.")
if load["is_regression"]:
    rec_lines.append(f"**Investigate the {load['pct_change']:.0f}% page-load regression** before a full launch.")
rec_lines.append("**Fix the flag-rollout bug** so future tests don't inherit a mix-shift.")

st.success("\n\n".join(rec_lines))

with st.expander("📋 Methodology & data notes"):
    st.markdown(
        """
- **Dataset:** synthetic, generated to reproduce a realistic checkout A/B test
  with an intentional randomization bug — not real production data.
- **SRM check:** chi-square goodness-of-fit test on arm counts, overall and by device.
- **Segment lift:** two-proportion z-test (`statsmodels.stats.proportion.proportions_ztest`).
- **Power:** post-hoc power to detect the sidebar-selected minimum effect
  (`statsmodels.stats.power.NormalIndPower`, Cohen's h for proportions).
- **Page load:** Welch's t-test on log-transformed load times.
"""
    )