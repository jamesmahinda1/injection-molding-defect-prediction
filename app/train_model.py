"""Train the XGBoost scrap predictor and save everything the app needs.

The raw dataset is not deployed with the app, so this script bakes the trained
model, the input ranges for the sliders, and a couple of example cycles into
small files under app/artifacts/. Run it once from the project root:

    python app/train_model.py
"""
import json
import os
import pandas as pd
from xgboost import XGBClassifier

HERE = os.path.dirname(__file__)
ART = os.path.join(HERE, 'artifacts')
os.makedirs(ART, exist_ok=True)

RENAME = {
    'Melt temperature': 'melt_temp',
    'Mold temperature': 'mold_temp',
    'time_to_fill': 'fill_time',
    'ZDx - Plasticizing time': 'plasticizing_time',
    'ZUx - Cycle time': 'cycle_time',
    'SKx - Closing force': 'closing_force',
    'SKs - Clamping force peak value': 'clamping_force',
    'Ms - Torque peak value current cycle': 'torque_peak',
    'Mm - Torque mean value current cycle': 'torque_mean',
    'APSs - Specific back pressure peak value': 'back_pressure',
    'APVs - Specific injection pressure peak value': 'injection_pressure',
    'CPn - Screw position at the end of hold pressure': 'screw_position',
    'SVo - Shot volume': 'shot_volume',
}
UNITS = {
    'melt_temp': 'C', 'mold_temp': 'C', 'fill_time': 's', 'plasticizing_time': 's',
    'cycle_time': 's', 'closing_force': 'N', 'clamping_force': 'N', 'torque_peak': 'Nm',
    'torque_mean': 'Nm', 'back_pressure': 'Bar', 'injection_pressure': 'Bar',
    'screw_position': 'mm', 'shot_volume': 'cm3',
}


def main():
    df = pd.read_csv(os.path.join(HERE, '..', 'data', 'raw', 'injection_molding.csv'), sep=';')
    df = df.rename(columns=RENAME)
    features = list(RENAME.values())
    df['quality'] = df['quality'].astype(int)
    df['is_scrap'] = (df['quality'] == 1).astype(int)

    X, y = df[features], df['is_scrap']

    scrap_weight = (y == 0).sum() / (y == 1).sum()
    model = XGBClassifier(scale_pos_weight=scrap_weight, eval_metric='logloss',
                          random_state=42)
    model.fit(X, y)
    model.save_model(os.path.join(ART, 'model.json'))

    good = df[df['is_scrap'] == 0]
    target = df[df['quality'] == 3]

    # example cycles for the demo buttons (defined first so the sliders can be
    # clipped without ever excluding an example value)
    examples = {
        'good': {f: round(float(v), 3) for f, v in good[features].iloc[0].items()},
        'scrap': {f: round(float(v), 3)
                  for f, v in df[df['is_scrap'] == 1][features].iloc[0].items()},
    }

    # slider ranges, defaults, and the target-quality reference band per setting.
    # Slider ends are clipped to the 1st-99th percentile so a few outlier cycles
    # do not stretch the sliders, but always widened to include the example
    # cycles so those load cleanly. "target" parts (quality 3) are the optimal
    # grade, so their 10th-90th percentile is the ideal band to aim inside.
    meta = {}
    for f in features:
        lo = min(df[f].quantile(0.01), examples['good'][f], examples['scrap'][f])
        hi = max(df[f].quantile(0.99), examples['good'][f], examples['scrap'][f])
        meta[f] = {
            'min': round(float(lo), 3),
            'max': round(float(hi), 3),
            'default': round(float(target[f].median()), 3),
            'good_low': round(float(target[f].quantile(0.10)), 3),
            'good_high': round(float(target[f].quantile(0.90)), 3),
            'unit': UNITS.get(f, ''),
        }

    with open(os.path.join(ART, 'feature_meta.json'), 'w') as fh:
        json.dump({'features': features, 'meta': meta, 'examples': examples}, fh, indent=2)

    print('Saved model and metadata to', ART)
    print('Features:', len(features))


if __name__ == '__main__':
    main()
