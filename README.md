
# Telco Customer Churn Engine

An end-to-end machine learning system that predicts customer churn and, more importantly, translates those predictions into practical business actions.

🔗 Live App: https://e42pk238crgmv4qgrfx29s.streamlit.app/

---

## Why I built this

Most churn prediction projects stop at:

> “Model accuracy = X%”

But in reality, that’s not useful unless it answers:

* Which customers are at risk?
* Why are they leaving?
* What should we actually do about it?

I built this project to bridge that gap — turning a machine learning model into something closer to a **decision-making tool**.

---

## What this system does

This is not just a model — it’s a small pipeline that goes from raw input → prediction → explanation → action.

### 1. Predict churn

* Uses Logistic Regression trained on a class-imbalanced dataset
* SMOTE is applied to improve recall (catch more churners)
* Final model prioritizes **recall over accuracy**, which makes sense in a retention context

---

### 2. Explain the prediction

Instead of treating the model like a black box, the app shows which features are pushing a customer toward churn.

For example:

* Short tenure → increases risk
* Month-to-month contract → increases risk
* Higher engagement / longer tenure → reduces risk

This makes the output interpretable, not just numerical.

---

### 3. Classify customer risk

Each prediction is mapped into a business-friendly category:

* Low risk
* Medium risk
* High / critical risk

---

### 4. Recommend what to do

This is the part I focused on the most.

Based on:

* churn probability
* customer value (monthly charges)
* tenure

The system suggests actions like:

* proactive support
* retention offers
* onboarding improvements
* or no action if risk is low

---

## How it is structured

I tried to keep the project modular instead of putting everything in one file.

```text
CustomerChurn/
│
├── app.py                # Streamlit UI
├── models/               # Saved model, scaler, feature columns
├── src/
│   ├── preprocess.py     # Feature engineering + encoding
│   ├── predict.py        # Model inference
│   └── strategy.py       # Business logic layer
│
├── notebook/             # Model development
├── requirements.txt
└── README.md
```

---

## Model notes

* Algorithm: Logistic Regression
* SMOTE used to handle imbalance
* Recall (churn): ~0.79

I intentionally didn’t chase maximum accuracy —
missing a churner is more expensive than a false alarm.

---

## Running locally

```bash
git clone https://github.com/amaanzz/telco-churn-prediction-engine.git
cd telco-churn-prediction-engine

pip install -r requirements.txt
streamlit run app.py
```

---

## What I learned

* Fixing pipelines (SMOTE + scaling order) matters more than trying new models
* Model evaluation should reflect business cost, not just metrics
* Explainability makes a huge difference in usability
* The hardest part isn’t training the model — it’s turning it into a usable system

---

## What I’d improve next

* Add a more advanced model (XGBoost / ensemble)
* Improve the explainability visuals (true SHAP integration)
* Add user history / batch predictions
* Deploy with a proper backend instead of a single Streamlit app

---

## Author

Amaan Shaikh
B.Tech student exploring applied machine learning and real-world systems

---

If you check this out, feel free to share feedback or ideas — always open to improving this further.
