"""Injection Molding Defect Prediction - story app and live demo.

One app that walks the whole project (the problem, the machine, the data, the
analysis, the models) and ends with a live predictor. Present by moving through
the sections in the sidebar. Run from the project root:

    streamlit run app/app.py
"""
import json
import os

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier

HERE = os.path.dirname(__file__)
ART = os.path.join(HERE, 'artifacts')
IMG = os.path.join(HERE, '..', 'presentation', 'assets')

st.set_page_config(page_title='Injection Molding Defect Prediction',
                   page_icon='wrench', layout='wide',
                   initial_sidebar_state='expanded')

st.markdown("""
<style>
  .stApp {
    background: radial-gradient(1200px 600px at 20% -10%, #16243f 0%, #0b1220 55%);
  }
  header[data-testid="stHeader"] { background: transparent; }
  #MainMenu, footer { visibility: hidden; }
  [data-testid="stToolbar"], [data-testid="stAppDeployButton"] { display: none; }
  [data-testid="stStatusWidget"] { display: none; }
  section[data-testid="stSidebar"] { background: #0d1626; border-right: 1px solid #1e2c47; }
  h1, h2, h3 { letter-spacing: .2px; }
  h1 { font-weight: 800; }
  .kicker {
    font-family: ui-monospace, "Cascadia Code", Consolas, monospace;
    font-size: .8rem; letter-spacing: .18em; text-transform: uppercase;
    color: #4f9dde; margin: 0 0 .2rem 0;
  }
  .hero {
    font-size: 2.9rem; font-weight: 800; line-height: 1.1; margin: .2rem 0 .3rem;
  }
  .hero-sub { color: #9fb0c8; font-size: 1.25rem; margin-bottom: 1.2rem; }
  div[data-testid="stMetric"] {
    background: #111c33; border: 1px solid #1e2c47; border-radius: 14px;
    padding: 16px 18px;
  }
  div[data-testid="stMetricValue"] { color: #4f9dde; }
  .stButton > button {
    border-radius: 10px; border: 1px solid #2a3c5e; background: #142443;
    color: #e8eef7; font-weight: 600; padding: .55rem 1rem;
  }
  .stButton > button:hover { border-color: #4f9dde; color: #4f9dde; }
  .stAlert { border-radius: 12px; }
  .block-container { padding-top: 2.2rem; max-width: 1150px; }
  img { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


def kicker(text):
    st.markdown(f'<div class="kicker">{text}</div>', unsafe_allow_html=True)


PRETTY = {
    'melt_temp': 'Melt temperature', 'mold_temp': 'Mold temperature',
    'fill_time': 'Fill time', 'plasticizing_time': 'Plasticizing time',
    'cycle_time': 'Cycle time', 'closing_force': 'Closing force',
    'clamping_force': 'Clamping force', 'torque_peak': 'Torque peak',
    'torque_mean': 'Torque mean', 'back_pressure': 'Back pressure',
    'injection_pressure': 'Injection pressure', 'screw_position': 'Screw position',
    'shot_volume': 'Shot volume',
}


@st.cache_resource
def load():
    model = XGBClassifier()
    model.load_model(os.path.join(ART, 'model.json'))
    with open(os.path.join(ART, 'feature_meta.json')) as fh:
        info = json.load(fh)
    return model, info, shap.TreeExplainer(model)


model, info, explainer = load()
features = info['features']
meta = info['meta']
examples = info['examples']


def img(name):
    path = os.path.join(IMG, name)
    if os.path.exists(path):
        st.image(path, width='stretch')


SECTIONS = [
    'Start here',
    '1. The problem',
    '2. How the machine works',
    '3. The data',
    '4. Do the settings differ?',
    '5. Can a straight line split them?',
    '6. The models',
    '7. Explaining the model',
    '8. Try it live',
    '9. Honest note and next steps',
]

st.sidebar.title('Injection Molding')
st.sidebar.caption('Defect prediction')
page = st.sidebar.radio('Go to', SECTIONS, label_visibility='collapsed')
st.sidebar.divider()
st.sidebar.caption('James Mahinda')
st.sidebar.caption('Advanced ML capstone')


# ---------------------------------------------------------------- sections
if page == 'Start here':
    kicker('Advanced ML Capstone // James Mahinda')
    st.markdown('<div class="hero">Injection Molding<br>Defect Prediction</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Catching defective plastic parts the moment '
                "they are made, from the machine's own settings.</div>",
                unsafe_allow_html=True)

    st.write(
        'Injection molding machines make one plastic part per cycle. When the '
        'settings drift out of range, the part comes out defective, but the '
        'factory usually only finds out later during inspection, after the '
        'material and machine time are already wasted.')
    st.write(
        'This project uses the settings the machine already records to predict, '
        'right away, whether a part is good or scrap, and to point to the settings '
        'that caused the risk so an operator can fix the machine before making '
        'more bad parts. Everything is built on 1,451 real production cycles.')

    st.write('')
    c = st.columns(3)
    c[0].metric('Cycles studied', '1,451')
    c[1].metric('Machine settings used', '13')
    c[2].metric('Scrap-catching score (out of 1)', '0.97')

    st.write('')
    st.subheader('What this walkthrough covers')
    a, b = st.columns(2)
    a.markdown(
        '- The problem and the machine behind it\n'
        '- The data and how good and scrap parts differ\n'
        '- Whether a simple straight line can tell them apart')
    b.markdown(
        '- The models, and why XGBoost wins\n'
        '- How the model explains each prediction\n'
        '- A live predictor you can try yourself')

elif page == '1. The problem':
    kicker('SEC 01 // The problem')
    st.header('The problem, and why it matters')
    a, b = st.columns([1, 1])
    with a:
        st.write('Almost everything plastic around us, from bottle caps to car '
                 'light lenses, is made on a machine like this. The problem is what '
                 'happens when a part comes out defective.')
        st.markdown(
            '- Factories catch defective parts late, during inspection\n'
            '- By then the material and machine time are already wasted\n'
            '- Every bad cycle is money lost\n'
            '- A machine drifting out of range can scrap a whole batch')
        st.write('')
        st.info('The idea: use the settings the machine already records to catch '
                'scrap right away, and point to the setting that caused it.')
    with b:
        img('machine_photo.jpg')

elif page == '2. How the machine works':
    kicker('SEC 02 // The machine')
    st.header('How the machine works')
    a, b = st.columns([1, 1])
    with a:
        st.markdown(
            '1. **Feed** plastic pellets in\n'
            '2. **Melt** them with a heated screw, like melted chocolate\n'
            '3. **Inject** into a mold under high pressure\n'
            '4. **Cool** so the plastic hardens into shape\n'
            '5. **Eject** the finished part')
        st.write('')
        st.write('Each repeat is one **cycle**, one part. Every cycle the machine '
                 'records its own settings (temperatures, pressures, times). '
                 'Those settings are the data.')
    with b:
        img('machine_diagram.png')

elif page == '3. The data':
    kicker('SEC 03 // The data')
    st.header('The data')
    st.write('I used a real, public dataset of production cycles from a plastics '
             'company. It is the record of what the machine was doing on each '
             'cycle, paired with how the part turned out.')
    st.markdown(
        '- Real production data from a plastics company (iGuzzini)\n'
        '- **1,451** molding cycles\n'
        '- **13** process parameters per cycle, plus a quality label\n'
        '- Clean: no missing values, no duplicates')
    st.write('')
    st.write('Each part is graded good or scrap. Scrap means it failed the quality '
             'standard and has to be thrown away. About a quarter of the parts in '
             'the data are scrap, so there is plenty of both to learn from.')

elif page == '4. Do the settings differ?':
    kicker('SEC 04 // Exploring the data')
    st.header('Do the settings differ between good and scrap?')
    st.write('If the cycles that made good parts and the cycles that made scrap had '
             'the same settings, no model could tell them apart. So first I checked '
             'whether the settings actually differ.')
    img('boxplots.png')
    st.success('They do. Several settings clearly differ, cycle time most of all. '
               'So there is real signal for a model to learn.')

elif page == '5. Can a straight line split them?':
    kicker('SEC 05 // Linear separability')
    st.header('Can a straight line separate good from scrap?')
    a, b = st.columns([1, 1])
    with a:
        st.write('Before choosing a model I checked if a simple straight line '
                 'could split the two classes. I squashed the 13 settings into a '
                 '2D picture (using PCA, a dimensionality reduction method) to look, '
                 'then confirmed with models.')
        st.markdown(
            '- Good and scrap **overlap**, no straight line splits them\n'
            '- Straight line model: scrap F1 **0.74**\n'
            '- Curved model: scrap F1 **0.85**')
        st.caption('F1 is a score from 0 to 1 for how well a model catches scrap, '
                   'higher is better. More on it in section 6.')
        st.info('Because a straight line is not enough, I move to stronger models. '
                'Not because they sound advanced, but because the data needs them.')
    with b:
        img('pca_scatter.png')

elif page == '6. The models':
    kicker('SEC 06 // Modeling')
    st.header('The models')
    a, b = st.columns([1, 1])
    with a:
        st.table(pd.DataFrame({
            'Model': ['Logistic Regression (baseline)', 'SVM (curved)', 'XGBoost'],
            'Scrap F1': [0.71, 0.85, 0.97],
        }))
        st.markdown('**XGBoost** (a boosting model) **wins:** about **98% '
                    'accuracy**, catching around 71 of 74 scrap parts and missing '
                    'only about 3.')
        st.markdown('**What the F1 score means, in plain words**')
        st.markdown(
            'A model can be wrong two ways: it can **miss** real scrap, or it can '
            'raise a **false alarm** on a good part. The F1 score is a single number '
            'from 0 to 1 that balances both: how much scrap it catches, and how often '
            'it is right when it shouts scrap. **1 is perfect.** So XGBoost at 0.97 '
            'catches almost all the scrap while rarely crying wolf.')
        img('confusion.png')
        st.caption('In the grid, the up-and-down axis is what the part really was, '
                   'and the left-to-right axis is what the model guessed. The two '
                   'dark boxes running corner to corner are the correct calls, the '
                   'lighter boxes off that line are its few mistakes.')
    with b:
        img('model_comparison.png')

elif page == '7. Explaining the model':
    kicker('SEC 07 // Explainability')
    st.header('Explaining the model')
    a, b = st.columns([1, 1])
    with a:
        st.write('A prediction alone is not useful on a factory floor. The operator '
                 'needs to know **why** a cycle was flagged and what to change.')
        st.write('I use a tool called **SHAP**. For any single cycle it gives each '
                 'setting a simple score: how much that setting pushed the cycle '
                 'toward scrap, or toward good. A big push means that setting '
                 'mattered a lot for that cycle.')
        st.write('The chart on the right averages that over **all 1,451 cycles**, so '
                 'it shows which settings the model leans on the most overall. Here '
                 '**cycle time, injection pressure and fill time** are the biggest '
                 'drivers.')
        st.info('The live demo does exactly this for a single cycle: flag it, and '
                'it lists the exact settings that drove the result.')
    with b:
        img('shap_importance.png')

elif page == '9. Honest note and next steps':
    kicker('SEC 09 // Honesty and roadmap')
    st.header('Being honest, and what comes next')
    st.info('These results are measured on a held-out test set the model never saw '
            'during training, so they are a fair score on this dataset. To be clear '
            'about scope: the model is trained and validated on one public dataset, '
            'a single product on a single machine, recorded over five production '
            'days. Using it on a different product, machine, or plant would need '
            'retraining on that data and fresh validation there. The numbers are '
            'solid for this data, and the next step is proving them on more machines.')
    st.subheader('One thing this data could not do: watch the machine over time')
    st.write('Each row records a cycle\'s settings, but not the time it happened. '
             'There are no timestamps, and the cycles are not stored in time order. '
             'So one thing I could not build is a time-based forecast: watching the '
             'machine as it runs and warning that it is slowly drifting toward making '
             'scrap, before it actually does.')
    st.write('This is a real gap in the wider industry too, not just this dataset. '
             'Many factory machines record their settings but never save the time of '
             'each cycle. The fix is simple: a plant would just need to store a '
             'timestamp with every cycle. Then a time-series method (such as ARIMA) '
             'could forecast where a setting is heading and raise an early warning '
             'before any scrap is made.')

    st.subheader('Next steps')
    st.markdown(
        '- Save a timestamp per cycle, then forecast drift and warn before scrap happens\n'
        '- Connect the model to a live machine feed inside a factory system\n'
        '- Suggest the exact setting change to bring a cycle back to target')

# ---------------------------------------------------------------- live demo
elif page == '8. Try it live':
    kicker('SEC 08 // Live demo')
    st.header('Try it live')
    st.write('This is the model in action. Give it a molding cycle and it tells you '
             'whether that cycle produced a good or a scrap part, and which settings '
             'pushed it that way.')
    st.markdown(
        '**How to use it**\n'
        '1. Click a button below to load a real example cycle (good or scrap), '
        'or drag the sliders to build your own\n'
        '2. Read the result, it updates the moment anything changes\n'
        '3. Look at "Why the model decided this" to see which settings pushed the '
        'result, and which to check first')

    # slider values live in session state under the s_<feature> keys, so the
    # example buttons can set them and the sliders actually move
    for f in features:
        st.session_state.setdefault('s_' + f, float(meta[f]['default']))

    def set_example(kind):
        for f in features:
            st.session_state['s_' + f] = float(examples[kind][f])

    c1, c2 = st.columns(2)
    c1.button('Load an example good cycle', on_click=set_example, args=('good',),
              width='stretch')
    c2.button('Load an example scrap cycle', on_click=set_example, args=('scrap',),
              width='stretch')

    vals = {f: st.session_state['s_' + f] for f in features}
    row = pd.DataFrame([vals])[features]
    proba = float(model.predict_proba(row)[0, 1])
    is_scrap = proba >= 0.5

    # result banner, right under the buttons so a click shows an instant change
    st.write('')
    if is_scrap:
        st.error(f"### Prediction: SCRAP  -  scrap risk {proba*100:.0f}%")
    else:
        st.success(f"### Prediction: GOOD  -  scrap risk {proba*100:.0f}%")
    st.progress(min(max(proba, 0.0), 1.0))

    st.divider()
    left, right = st.columns([1, 1])
    with left:
        st.subheader('Cycle settings')
        st.caption('Drag any slider and watch the prediction above change.')
        cols = st.columns(2)
        for idx, f in enumerate(features):
            m = meta[f]
            span = m['max'] - m['min']
            step = round(span / 100, 3) if span else 0.01
            label = PRETTY[f] + (f" ({m['unit']})" if m['unit'] else '')
            cols[idx % 2].slider(label, float(m['min']), float(m['max']),
                                 step=float(step or 0.01), key='s_' + f)

    with right:
        st.subheader('Why the model decided this')
        st.caption('Red settings pushed this cycle toward scrap, blue toward good. '
                   'The longest bars mattered most.')
        shap_vals = explainer.shap_values(row)[0]
        contrib = pd.DataFrame({
            'setting': [PRETTY[f] for f in features],
            'push': shap_vals,
        }).sort_values('push', key=np.abs, ascending=False).head(6).iloc[::-1]
        fig, ax = plt.subplots(figsize=(6, 3.4))
        colors = ['#e05563' if v > 0 else '#4f9dde' for v in contrib['push']]
        ax.barh(contrib['setting'], contrib['push'], color=colors)
        ax.axvline(0, color='#888', lw=0.8)
        ax.set_xlabel('toward scrap (right) / toward good (left)')
        st.pyplot(fig)
        plt.close(fig)
        st.caption('These are the settings to look at first for this cycle.')
