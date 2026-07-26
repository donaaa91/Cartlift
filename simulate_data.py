"""
CartLift: Checkout Redesign A/B Test — Data Simulator
=======================================================

Generates a SYNTHETIC dataset of ~42,000 checkout sessions over 21 days,
engineered to reproduce a specific real-world failure mode: a mobile app
that ships the treatment flag a few days early, causing a sample-ratio
mismatch (mobile + new users become over-represented in treatment).

The true effect is baked in on purpose:
  - Mobile users get a genuine conversion lift from the redesign.
  - Desktop users get ~no true effect.
  - Treatment is genuinely slower to load (real page-load regression).

This lets `analysis.py` be checked against a known ground truth rather
than an unverifiable real dataset.

Usage:
    python simulate_data.py
    python simulate_data.py --seed 7 --out data/checkout_sessions.csv

Output:
    data/checkout_sessions.csv
"""

import argparse
import sys

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Simulation parameters
# ----------------------------------------------------------------------------
N_DAYS = 21
START_DATE = pd.Timestamp("2026-06-01")
DAILY_SESSIONS_MEAN = 2000  # ~2,000/day * 21 days ≈ 42,000 sessions total

# Traffic composition, absent any bug
P_MOBILE = 0.55
P_NEW_USER = 0.40

# Window in which the mobile app shipped the treatment flag early (the bug)
BUG_START_DAY = 1
BUG_END_DAY = 5
BUG_MOBILE_TREATMENT_BOOST = 0.30  # extra P(treatment) for mobile during the bug window
BUG_MOBILE_TRAFFIC_BOOST = 0.12   # mobile share also ticks up during the bug window
BUG_NEW_USER_TRAFFIC_BOOST = 0.10  # new-user share also ticks up during the bug window

# Baseline conversion rates by segment (control arm), as decimals
BASE_CVR = {
    ("mobile", "new"): 0.062,
    ("mobile", "returning"): 0.081,
    ("desktop", "new"): 0.071,
    ("desktop", "returning"): 0.093,
}

# TRUE treatment effects by segment (additive, in decimal pp).
# Real lift on mobile; ~flat on desktop.
TRUE_LIFT = {
    ("mobile", "new"): 0.058,
    ("mobile", "returning"): 0.048,
    ("desktop", "new"): 0.004,
    ("desktop", "returning"): -0.002,
}

# Page load time (ms). Treatment is heavier (one-page = bigger payload).
BASE_LOAD_MS = {"mobile": 1850, "desktop": 1400}
TREATMENT_LOAD_REGRESSION = 0.085  # ~+10-11% median load time on treatment
LOAD_SIGMA = 0.22  # lognormal shape parameter


def assign_arm(rng: np.random.Generator, day: int, device: str) -> str:
    """Assigns a session to 'treatment' or 'control'.

    Reproduces the flag-rollout bug: during BUG_START_DAY..BUG_END_DAY,
    mobile sessions have an inflated chance of landing in treatment.
    """
    p_treat = 0.5
    if device == "mobile" and BUG_START_DAY <= day <= BUG_END_DAY:
        p_treat = min(0.95, 0.5 + BUG_MOBILE_TREATMENT_BOOST)
    return "treatment" if rng.random() < p_treat else "control"


def simulate(seed: int = 42, n_days: int = N_DAYS,
             daily_sessions_mean: int = DAILY_SESSIONS_MEAN) -> pd.DataFrame:
    """Generates the full synthetic session-level dataset.

    Parameters
    ----------
    seed : random seed for reproducibility.
    n_days : number of days to simulate.
    daily_sessions_mean : mean sessions/day (actual count is Poisson-distributed).

    Returns
    -------
    pd.DataFrame with one row per session.
    """
    rng = np.random.default_rng(seed)
    rows = []
    session_id = 1

    for day in range(n_days):
        date = START_DATE + pd.Timedelta(days=day)
        n_sessions = rng.poisson(daily_sessions_mean)

        p_mobile_today = P_MOBILE
        p_new_today = P_NEW_USER
        if BUG_START_DAY <= day <= BUG_END_DAY:
            p_mobile_today = min(0.95, P_MOBILE + BUG_MOBILE_TRAFFIC_BOOST)
            p_new_today = min(0.95, P_NEW_USER + BUG_NEW_USER_TRAFFIC_BOOST)

        for _ in range(n_sessions):
            device = "mobile" if rng.random() < p_mobile_today else "desktop"
            user_type = "new" if rng.random() < p_new_today else "returning"
            arm = assign_arm(rng, day, device)

            base_cvr = BASE_CVR[(device, user_type)]
            lift = TRUE_LIFT[(device, user_type)] if arm == "treatment" else 0.0
            p_convert = float(np.clip(base_cvr + lift, 0.001, 0.999))
            converted = rng.random() < p_convert

            base_load = BASE_LOAD_MS[device]
            load_multiplier = (1 + TREATMENT_LOAD_REGRESSION) if arm == "treatment" else 1.0
            page_load_ms = rng.lognormal(mean=np.log(base_load * load_multiplier), sigma=LOAD_SIGMA)

            rows.append({
                "session_id": session_id,
                "date": date,
                "day_index": day,
                "device": device,
                "user_type": user_type,
                "arm": arm,
                "converted": int(converted),
                "page_load_ms": round(float(page_load_ms), 1),
            })
            session_id += 1

    df = pd.DataFrame(rows)
    return df


def validate(df: pd.DataFrame) -> None:
    """Basic sanity checks on the generated dataset. Raises AssertionError on failure."""
    assert len(df) > 0, "Generated dataset is empty."
    assert df["converted"].isin([0, 1]).all(), "converted column must be 0/1."
    assert df["arm"].isin(["control", "treatment"]).all(), "arm must be 'control'/'treatment'."
    assert df["device"].isin(["mobile", "desktop"]).all(), "device must be 'mobile'/'desktop'."
    assert df["user_type"].isin(["new", "returning"]).all(), "user_type must be 'new'/'returning'."
    assert (df["page_load_ms"] > 0).all(), "page_load_ms must be positive."
    assert df["session_id"].is_unique, "session_id must be unique."


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the synthetic CartLift A/B test dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--days", type=int, default=N_DAYS, help=f"Number of days to simulate (default: {N_DAYS})")
    parser.add_argument("--daily-sessions", type=int, default=DAILY_SESSIONS_MEAN,
                         help=f"Mean sessions per day (default: {DAILY_SESSIONS_MEAN})")
    parser.add_argument("--out", type=str, default="data/checkout_sessions.csv",
                         help="Output CSV path (default: data/checkout_sessions.csv)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    print(f"Simulating ~{args.days * args.daily_sessions:,} sessions over {args.days} days "
          f"(seed={args.seed})...")
    df = simulate(seed=args.seed, n_days=args.days, daily_sessions_mean=args.daily_sessions)

    try:
        validate(df)
    except AssertionError as e:
        print(f"Validation failed: {e}", file=sys.stderr)
        return 1

    df.to_csv(args.out, index=False)
    print(f"Generated {len(df):,} sessions -> {args.out}")

    print("\nSessions by device x arm (check: mobile counts should look imbalanced,"
          " desktop roughly even):")
    print(df.groupby(["device", "arm"]).size().unstack())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
