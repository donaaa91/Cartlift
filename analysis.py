

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportions_ztest

DEFAULT_ALPHA = 0.05
DEFAULT_MDE = 0.015  # smallest lift (as a decimal) considered worth shipping for
REQUIRED_COLUMNS = {"date", "device", "user_type", "arm", "converted", "page_load_ms"}


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """Loads and validates the session-level dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No dataset found at '{path}'. Generate it first with:\n"
            f"  python simulate_data.py --out {path}"
        )
    df = pd.read_csv(path, parse_dates=["date"])

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected column(s): {sorted(missing)}")
    if df.empty:
        raise ValueError(f"Dataset at '{path}' is empty.")
    if not df["arm"].isin(["control", "treatment"]).all():
        raise ValueError("'arm' column must only contain 'control'/'treatment'.")

    return df


# ----------------------------------------------------------------------------
# 1. Randomization health check (Sample Ratio Mismatch)
# ----------------------------------------------------------------------------
def srm_check(df: pd.DataFrame, alpha: float = 0.01) -> dict:
    """Chi-square goodness-of-fit test: is the treatment/control split
    ~50/50 as expected, overall and within each device segment?

    A low p-value (below `alpha`) flags a Sample Ratio Mismatch — a sign
    the randomization mechanism itself is broken.
    """
    results = {}

    overall = df["arm"].value_counts()
    chi2, p = stats.chisquare(overall.values)
    results["overall"] = {"counts": overall.to_dict(), "chi2": chi2, "p_value": p,
                           "srm_flag": p < alpha}

    for device, sub in df.groupby("device"):
        counts = sub["arm"].value_counts()
        chi2, p = stats.chisquare(counts.values)
        results[f"device={device}"] = {"counts": counts.to_dict(), "chi2": chi2,
                                        "p_value": p, "srm_flag": p < alpha}
    return results


# ----------------------------------------------------------------------------
# 2. Naive headline
# ----------------------------------------------------------------------------
def naive_headline(df: pd.DataFrame) -> dict:
    """The overall lift you'd report if you skipped the SRM check entirely."""
    cvr = df.groupby("arm")["converted"].mean()
    lift = cvr.get("treatment", np.nan) - cvr.get("control", np.nan)
    return {
        "control_cvr": cvr.get("control", np.nan),
        "treatment_cvr": cvr.get("treatment", np.nan),
        "naive_lift_pp": lift * 100,
    }


# ----------------------------------------------------------------------------
# 3. Segmented analysis
# ----------------------------------------------------------------------------
def segment_analysis(df: pd.DataFrame, alpha: float = DEFAULT_ALPHA) -> pd.DataFrame:
    """Two-proportion z-test within each device x user_type segment,
    isolating the true effect from any overall mix-shift."""
    rows = []
    for (device, user_type), sub in df.groupby(["device", "user_type"]):
        ctrl = sub[sub.arm == "control"]
        treat = sub[sub.arm == "treatment"]
        n_c, n_t = len(ctrl), len(treat)
        if n_c == 0 or n_t == 0:
            continue

        x_c, x_t = ctrl["converted"].sum(), treat["converted"].sum()
        cvr_c, cvr_t = x_c / n_c, x_t / n_t
        lift_pp = (cvr_t - cvr_c) * 100

        z_stat, p_val = proportions_ztest(np.array([x_t, x_c]), np.array([n_t, n_c]))

        rows.append({
            "device": device, "user_type": user_type,
            "n_control": n_c, "n_treatment": n_t,
            "cvr_control": cvr_c, "cvr_treatment": cvr_t,
            "lift_pp": lift_pp, "z_stat": z_stat, "p_value": p_val,
            "significant": p_val < alpha,
        })
    return pd.DataFrame(rows).sort_values(["device", "user_type"]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# 4. Power analysis
# ----------------------------------------------------------------------------
def power_analysis(df: pd.DataFrame, mde: float = DEFAULT_MDE,
                    alpha: float = DEFAULT_ALPHA) -> pd.DataFrame:
    """For each segment, computes achieved power to detect a lift of size
    `mde` at the current sample size — distinguishing 'true null' from
    'underpowered / inconclusive' for non-significant segments."""
    analysis = NormalIndPower()
    rows = []
    for (device, user_type), sub in df.groupby(["device", "user_type"]):
        ctrl = sub[sub.arm == "control"]
        n_c = len(ctrl)
        n_t = len(sub[sub.arm == "treatment"])
        if n_c == 0 or n_t == 0:
            continue
        n_per_arm = min(n_c, n_t)
        p_base = ctrl["converted"].mean()

        # Cohen's h effect size for two proportions
        h = 2 * np.arcsin(np.sqrt(min(p_base + mde, 0.999))) - 2 * np.arcsin(np.sqrt(p_base))
        power = analysis.solve_power(effect_size=abs(h), nobs1=n_per_arm, alpha=alpha,
                                      ratio=1.0, alternative="two-sided")

        rows.append({
            "device": device, "user_type": user_type, "n_per_arm": n_per_arm,
            "baseline_cvr": p_base, "mde_pp": mde * 100,
            "achieved_power": power, "adequately_powered": power >= 0.80,
        })
    return pd.DataFrame(rows).sort_values(["device", "user_type"]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# 5. Page load regression check
# ----------------------------------------------------------------------------
def page_load_regression(df: pd.DataFrame, alpha: float = DEFAULT_ALPHA) -> dict:
    """Welch's t-test on log page load times, treatment vs. control.
    Page load is right-skewed, so we test on the log scale."""
    log_load = np.log(df["page_load_ms"].clip(lower=1))
    ctrl = log_load[df.arm == "control"]
    treat = log_load[df.arm == "treatment"]

    t_stat, p_val = stats.ttest_ind(treat, ctrl, equal_var=False)

    median_ctrl = df.loc[df.arm == "control", "page_load_ms"].median()
    median_treat = df.loc[df.arm == "treatment", "page_load_ms"].median()
    pct_change = (median_treat - median_ctrl) / median_ctrl * 100

    return {
        "median_control_ms": median_ctrl, "median_treatment_ms": median_treat,
        "pct_change": pct_change, "t_stat": t_stat, "p_value": p_val,
        "significant_regression": (p_val < alpha) and (pct_change > 0),
    }


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
def run_full_analysis(data_path: str, mde: float = DEFAULT_MDE, alpha: float = DEFAULT_ALPHA,
                       output_dir: str = "outputs") -> dict:
    df = load_data(data_path)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("1. RANDOMIZATION HEALTH CHECK (SRM)")
    print("=" * 60)
    srm = srm_check(df)
    for key, val in srm.items():
        flag = "⚠️ SRM DETECTED" if val["srm_flag"] else "OK"
        print(f"  {key}: counts={val['counts']}  p={val['p_value']:.2e}  [{flag}]")

    print("\n" + "=" * 60)
    print("2. NAIVE HEADLINE (pre-segmentation — do not trust)")
    print("=" * 60)
    naive = naive_headline(df)
    print(f"  Control CVR:   {naive['control_cvr']:.4f}")
    print(f"  Treatment CVR: {naive['treatment_cvr']:.4f}")
    print(f"  Naive lift:    {naive['naive_lift_pp']:.2f} pp  <- may be confounded by mix-shift")

    print("\n" + "=" * 60)
    print("3. SEGMENTED ANALYSIS (the real numbers)")
    print("=" * 60)
    seg = segment_analysis(df, alpha=alpha)
    print(seg.to_string(index=False))

    print("\n" + "=" * 60)
    print("4. POWER ANALYSIS (no-effect vs. inconclusive)")
    print("=" * 60)
    power = power_analysis(df, mde=mde, alpha=alpha)
    print(power.to_string(index=False))

    print("\n" + "=" * 60)
    print("5. PAGE LOAD REGRESSION CHECK")
    print("=" * 60)
    load = page_load_regression(df, alpha=alpha)
    for k, v in load.items():
        print(f"  {k}: {v}")

    seg.to_csv(os.path.join(output_dir, "segment_results.csv"), index=False)
    power.to_csv(os.path.join(output_dir, "power_results.csv"), index=False)
    pd.DataFrame([load]).to_csv(os.path.join(output_dir, "load_regression.csv"), index=False)

    srm_rows = [{"scope": k, "p_value": v["p_value"], "srm_flag": v["srm_flag"]}
                for k, v in srm.items()]
    pd.DataFrame(srm_rows).to_csv(os.path.join(output_dir, "srm_results.csv"), index=False)

    print(f"\nResults saved to '{output_dir}/'.")

    return {"df": df, "srm": srm, "naive": naive, "segments": seg,
             "power": power, "load_regression": load}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CartLift A/B test analysis pipeline.")
    parser.add_argument("--data", type=str, default="data/checkout_sessions.csv",
                         help="Path to input CSV (default: data/checkout_sessions.csv)")
    parser.add_argument("--mde", type=float, default=DEFAULT_MDE,
                         help=f"Minimum detectable effect, as a decimal (default: {DEFAULT_MDE})")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                         help=f"Significance threshold (default: {DEFAULT_ALPHA})")
    parser.add_argument("--output-dir", type=str, default="outputs",
                         help="Directory to save result CSVs (default: outputs)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        run_full_analysis(args.data, mde=args.mde, alpha=args.alpha, output_dir=args.output_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
