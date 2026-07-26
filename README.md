# 🛒 CartLift: Checkout Redesign A/B Test

A statistical case study on a checkout-flow experiment: a one-page checkout vs. the
existing multi-step flow, ~42,000 sessions over 21 days.



> **Data note:** This project uses a **synthetic dataset** (`simulate_data.py`),
> engineered to reproduce a realistic and common experimentation failure mode — a
> mobile app that ships a treatment flag a few days early, creating a sample
> ratio mismatch. It is not real production data from any company. The
> statistical methods (SRM check, segmentation, two-proportion z-tests,
> post-hoc power analysis) are applied exactly as they would be on a real
> dataset.

---

## The problem

A naive comparison of the two checkout flows shows a **+3.37pp lift** — a
number clean enough to justify shipping immediately. But before trusting any
A/B test result, you have to check whether the randomization that produced it
was actually working. In this case, it wasn't.

## What I did

1. **Randomization health check** — ran a chi-square goodness-of-fit test on
   arm counts, overall and by device. Caught a sample ratio mismatch on
   mobile (p ≈ 7e-141): the mobile app had shipped the treatment flag ~5 days
   early, over-representing mobile and new users — who convert differently
   regardless of the redesign.
2. **Segmented the analysis** by device × user type and re-ran two-proportion
   z-tests within each segment, isolating the true effect from the mix-shift
   artifact.
3. **Ran a power analysis** on the segments that weren't significant, to tell
   "no effect" apart from "not enough data to know."
4. **Checked a secondary metric** — page load time — to see whether the
   redesign cost anything it didn't show up in the conversion number.

## What I found

| Check | Result |
|---|---|
| Naive headline lift | +3.37pp — confounded, don't trust it |
| Mobile, new users | **+5.2pp**, significant (p ≈ 5e-19) |
| Mobile, returning users | **+5.65pp**, significant (p ≈ 3e-25) |
| Desktop (both segments) | Not significant, but **underpowered** (~66–70% achieved power at a 1.5pp threshold) — inconclusive, not proven zero |
| Page load time | **+10.9%** regression on treatment, significant (p < 0.001) |

## Recommendation

- **Ship on mobile** — the lift is real and holds up under segmentation.
- **Hold desktop for a clean, dedicated re-test** rather than treating the
  non-significant result as proof there's no effect.
- **Fix the flag-rollout bug** before running any future test on this
  platform.
- **Investigate the page load regression** before a full launch — it's a real
  cost that could erode the conversion gain over time.


## Project structure

```
cartlift/
├── simulate_data.py      # generates the synthetic dataset (bakes in the SRM bug + true effects)
├── analysis.py           # SRM check, naive lift, segmentation, power analysis, load regression
├── app.py                # Streamlit dashboard
├── data/
│   └── checkout_sessions.csv   # generated dataset (regenerate anytime)
├── outputs/               # saved analysis result CSVs + charts
├── requirements.txt
└── README.md
```

## Running it locally

```bash
git clone https://github.com/donaaa91/Cartlift.git
cd Cartlift
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python simulate_data.py         # generates data/checkout_sessions.csv
python analysis.py              # prints the full analysis, saves CSVs to outputs/
streamlit run app.py            # launches the interactive dashboard
```

## Methodology notes

- **SRM check:** chi-square goodness-of-fit test on arm counts
  (`scipy.stats.chisquare`), overall and by device.
- **Segment lift:** two-proportion z-test
  (`statsmodels.stats.proportion.proportions_ztest`).
- **Power analysis:** post-hoc power to detect a business-relevant minimum
  effect, using Cohen's h for proportions
  (`statsmodels.stats.power.NormalIndPower`).
- **Page load:** Welch's t-test on log-transformed load times (right-skewed,
  lognormal-shaped).

## Stack

Python · pandas · scipy.stats · statsmodels · Streamlit · Plotly

## Author

Dona Mariya Manoj
