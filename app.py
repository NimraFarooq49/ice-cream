import os
import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Ice Cream Revenue — Dashboard", page_icon="🍦", layout="wide")

def header():
	st.markdown(
		"<div style='display:flex;align-items:center;gap:16px'>"
		"<h1 style='margin:0'>🍦 Ice Cream Revenue Dashboard</h1>"
		"<div style='color:#6c757d;margin-left:12px'>A simple, non-technical view of model results and live predictions</div>"
		"</div>",
		unsafe_allow_html=True,
	)


header()

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')


def models_exist():
	required = ['svm_model.joblib', 'scaler.joblib', 'results.joblib']
	return all(os.path.exists(os.path.join(MODEL_DIR, r)) for r in required)


@st.cache_data
def load_data(path="Ice_Cream.csv"):
	df = pd.read_csv(path)
	df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce')
	df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce')
	df = df.dropna()
	return df


df = load_data()

# Basic explanation for non-technical users
st.markdown(
	"<div style='background:#black;padding:12px;border-radius:8px'>"
	"<strong>What this does:</strong> We trained two models that predict whether daily revenue will be <em>High</em> or <em>Low</em> based on the day's temperature. "
	"<br>High means revenue is at or above the historical median. The SVM model is recommended for live prediction here."
	"</div>",
	unsafe_allow_html=True,
)

st.write("")

if not models_exist():
	st.error("Model artifacts not found. Run `training_notebook.ipynb` to create `models/` before using this dashboard.")
	with st.expander("How to create models"):
		st.markdown(
			"1. Open `training_notebook.ipynb` and run all cells.\n2. That creates `models/` with trained artifacts.\n3. Refresh this page."
		)
	st.stop()

# Load saved artifacts
svm = joblib.load(os.path.join(MODEL_DIR, 'svm_model.joblib'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
results = joblib.load(os.path.join(MODEL_DIR, 'results.joblib'))

# KPIs row
col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 2])
with col1:
	st.metric(label="SVM Accuracy (test)", value=f"{results['svm']['accuracy']:.2%}")
with col2:
	st.metric(label="Decision Tree Accuracy (test)", value=f"{results['decision_tree']['accuracy']:.2%}")
with col3:
	median_rev = df['Revenue'].median()
	st.metric(label="Median Revenue", value=f"{median_rev:.0f}")
with col4:
	st.markdown("<div style='font-size:14px;color:#495057'>Predict whether revenue will be High (>= median) or Low for a given temperature. Use the interactive panel to the right to try values.</div>", unsafe_allow_html=True)

st.markdown("---")

# Charts row: left = data overview, right = prediction panel
left, right = st.columns([2, 1])

with left:
	st.subheader("Data overview — friendly visuals")
	# Aggregated average revenue by rounded temperature bins for clarity
	df['TempRound'] = df['Temperature'].round(0).astype(int)
	agg = df.groupby('TempRound', as_index=False)['Revenue'].mean().rename(columns={'Revenue': 'AvgRevenue'})
	agg = agg.sort_values('TempRound')
	st.markdown("**Average revenue by temperature (°C)**")
	st.bar_chart(data=agg.set_index('TempRound')['AvgRevenue'])

	st.write("---")
	st.markdown("**How to read this**: Each bar shows the average revenue observed historically at that temperature (rounded). Higher bars mean days at that temperature tended to bring more revenue.")

	st.write("")
	st.subheader("Scatter: Temperature vs Revenue (sample)")
	sample = df.sample(min(300, len(df)), random_state=1)
	# color by label for intuitive view
	sample['Label'] = (sample['Revenue'] >= median_rev).astype(int)
	# Use Streamlit's built-in chart for simplicity
	spec = {
		"mark": "point",
		"encoding": {
			"x": {"field": "Temperature", "type": "quantitative", "title": "Temperature (°C)"},
			"y": {"field": "Revenue", "type": "quantitative", "title": "Revenue"},
			"color": {"field": "Label", "type": "nominal", "title": "High revenue", "scale": {"domain": [0, 1], "range": ["#636EFA", "#EF553B"]}},
			"tooltip": [{"field": "Temperature"}, {"field": "Revenue"}]
		}
	}
	# Pass the sampled DataFrame as the data source so Vega-Lite can render correctly
	st.vega_lite_chart(data=sample, spec=spec, width='stretch')

with right:
	st.subheader("Live prediction")
	st.markdown("Enter a temperature to see the model's prediction and confidence. Changes update instantly.")
	temp = st.number_input("Temperature (°C)", value=float(df['Temperature'].median()), step=0.1, format="%.1f")

	# Compute prediction immediately whenever `temp` changes (Streamlit reruns on widget change)
	x = np.array([[temp]])
	x_s = scaler.transform(x)
	pred = svm.predict(x_s)[0]
	proba = svm.predict_proba(x_s)[0]
	prob_high = float(proba[1])
	label = "High revenue" if pred == 1 else "Low revenue"

	# Prominent result box
	st.write("")
	if pred == 1:
		st.success(f"Prediction: {label}")
	else:
		st.info(f"Prediction: {label}")

	# Show both class probabilities clearly for non-technical users
	p_low = float(proba[0])
	p_high = float(proba[1])
	pcol1, pcol2 = st.columns(2)
	pcol1.metric(label="P(Low revenue)", value=f"{p_low:.1%}")
	pcol2.metric(label="P(High revenue)", value=f"{p_high:.1%}")
	# visual progress bar for High
	st.progress(min(max(p_high, 0.0), 1.0))

	st.markdown("---")
	st.markdown("**What this means (plain language):**")
	if prob_high >= 0.8:
		st.write("Very likely this temperature will produce High revenue compared to historical days.")
	elif prob_high >= 0.6:
		st.write("Somewhat likely — above average chance of High revenue.")
	elif prob_high >= 0.4:
		st.write("Uncertain — about a 50/50 chance.")
	else:
		st.write("Likely Low revenue for this temperature.")

	st.write("")
	st.subheader("Quick actions")
	# Provide a direct download button for the model
	model_path = os.path.join(MODEL_DIR, 'svm_model.joblib')
	with open(model_path, 'rb') as fh:
		model_bytes = fh.read()
	st.download_button('Download SVM model', data=model_bytes, file_name='svm_model.joblib')

st.markdown("---")

with st.expander("Technical details (classification reports & confusion matrices)"):
	st.subheader("Decision Tree")
	# (classification_report not shown here by default)
	if 'decision_tree' in results:
		st.write("Confusion matrix:")
		st.write(results['decision_tree'].get('confusion_matrix'))
		# show classification report if available in saved results
		if isinstance(results['decision_tree'].get('classification_report'), dict):
			st.json(results['decision_tree']['classification_report'])

	st.subheader("SVM")
	if 'svm' in results:
		st.write("Confusion matrix:")
		st.write(results['svm'].get('confusion_matrix'))
		if isinstance(results['svm'].get('classification_report'), dict):
			st.json(results['svm']['classification_report'])

st.caption("Built from `training_notebook.ipynb` — models trained on the provided `Ice_Cream.csv` dataset.")

# Small centered footer credit
st.markdown(
	"<div style='text-align:center;margin-top:18px;color:#6c757d;font-size:14px'> Developed by **Nimra** for Sir ZEESHAN with ❤️</div>",
	unsafe_allow_html=True,
)



