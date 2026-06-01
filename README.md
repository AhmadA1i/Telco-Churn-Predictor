# Customer Churn Prediction

End-to-end customer churn prediction project for the Telco dataset. The notebook trains and evaluates multiple classification models, then saves a tuned Gradient Boosting model for use in a Streamlit frontend.

## Project Overview

- **Notebook:** [customer_churn_updated.ipynb](customer_churn_updated.ipynb)
- **Frontend:** [app.py](app.py)
- **Dataset:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Saved artifacts:** `artifacts/churn_model.pkl`, `artifacts/scaler.pkl`, `artifacts/churn_preprocess_meta.pkl`

## What the Project Does

- Cleans and prepares the Telco churn dataset
- Engineers features such as `AvgMonthlySpend` and `TenureBucket`
- Benchmarks multiple classification models
- Tunes a Gradient Boosting classifier
- Saves the trained model and preprocessing metadata
- Serves interactive churn predictions in a Streamlit app

## Recommended Environment

Use **Python 3.10** with conda.

### Create the environment

```bash
conda create -n churn-app python=3.10 -y
conda activate churn-app
```

### Install app dependencies

```bash
conda install -c conda-forge pandas numpy scikit-learn joblib streamlit -y
```

### Install notebook / modeling dependencies

```bash
conda install -c conda-forge matplotlib seaborn plotly xgboost catboost missingno jupyterlab ipykernel -y
```

If conda has trouble resolving any package, install that package with pip:

```bash
pip install streamlit xgboost catboost missingno
```

## Running the Notebook

1. Open [customer_churn_updated.ipynb](customer_churn_updated.ipynb)
2. Set `DATA_PATH` to the local CSV file if needed
3. Run the cells in order
4. Execute the deployment cells so the `artifacts/` files are created

The notebook saves:

- `artifacts/churn_model.pkl`
- `artifacts/scaler.pkl`
- `artifacts/churn_preprocess_meta.pkl`

## Running the Streamlit App

After the notebook has created the artifacts, start the app with:

```bash
streamlit run app.py
```

The app loads the saved model and preprocessing metadata, then applies the same feature engineering and scaling used during training before making a prediction.

## Model Notes

- Target variable: `Churn`
- Best deployed model: tuned `GradientBoostingClassifier`
- Decision threshold is saved in `artifacts/churn_preprocess_meta.pkl`
- The app expects the same customer fields used in the Telco dataset

## Suggested Test Inputs

To try a high-risk churn profile, use values like:

- `Contract = Month-to-month`
- `InternetService = Fiber optic`
- `PaymentMethod = Electronic check`
- `PaperlessBilling = Yes`
- `Partner = No`
- `Dependents = No`
- `SeniorCitizen = 1`
- `tenure = 0 to 3`
- `MonthlyCharges = high`
- `TotalCharges = low`

To try a low-risk profile, use values like:

- `Contract = Two year`
- `InternetService = DSL`
- `PaymentMethod = Bank transfer (automatic)` or `Credit card (automatic)`
- `PaperlessBilling = No`
- `Partner = Yes`
- `Dependents = Yes`
- `tenure = 60+`
- lower monthly charges

## Project Structure

```text
Customer_Churn_Prediction/
|-- app.py
|-- customer_churn_updated.ipynb
|-- requirements.txt
|-- WA_Fn-UseC_-Telco-Customer-Churn.csv
|-- artifacts/
```

## Notes

- Keep the notebook and Streamlit app in sync if you retrain the model or change the feature set.
- If the artifacts are missing, rerun the notebook deployment cells before starting Streamlit.