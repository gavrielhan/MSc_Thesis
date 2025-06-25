from lifelines import CoxPHFitter
import pandas as pd
import numpy as np

def get_closest_times(index, targets):
    """Map each target time to the closest available index value."""
    index_vals = index.to_numpy()
    return [index_vals[np.argmin(np.abs(index_vals - t))] for t in targets]

def fit_absolute_risk_calibrator(risk_scores, durations, events, times=[30, 90, 180, 270, 360]):
    """
    Fits a baseline survival curve using lifelines and returns calibrated risks.
    """
    num_conditions = risk_scores.shape[1]
    calibrators = []

    for ci in range(num_conditions):
        df = pd.DataFrame({
            'duration': durations[:, ci],
            'event': events[:, ci],
            'risk': risk_scores[:, ci],
        })

        # Standardise risk to avoid near-zero variance which harms Cox fitting
        mu = df['risk'].mean()
        sigma = df['risk'].std()
        if sigma < 1e-6:
            sigma = 1.0  # avoid division by zero and keep some variance
        df['risk_z'] = (df['risk'] - mu) / sigma

        cph = CoxPHFitter(penalizer=0.1)
        try:
            cph.fit(
                df[['duration', 'event', 'risk_z']],
                duration_col="duration",
                event_col="event",
                formula="risk_z",
            )
        except Exception as e:
            print(f"Skipping condition {ci} due to error: {e}")
            calibrators.append((None, None, {}))
            continue

        # Map requested times to available closest times
        available_times = cph.baseline_survival_.index.to_numpy()
        time_mapping = {}
        for t in times:
            closest = available_times[np.argmin(np.abs(available_times - t))]
            time_mapping[t] = closest

        S0_t = cph.baseline_survival_.loc[list(time_mapping.values())]
        calibrators.append((cph, S0_t, time_mapping))

    return calibrators
