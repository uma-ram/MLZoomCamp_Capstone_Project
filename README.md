
# Bank Marketing – Term Deposit Subscription Classification 

## Problem Description

Banks run marketing campaigns to encourage customers to subscribe to
term deposit products. Contacting all customers is expensive and often
inefficient.

The goal of this project is to build a machine learning classification
model that predicts whether a client will subscribe to a term deposit
(`yes` or `no`) based on customer demographics, past interactions, and
campaign information.

This prediction can help banks:
- Target high-probability customers
- Reduce marketing costs
- Improve campaign success rates

## Dataset

The dataset is taken from the UCI Bank Marketing dataset and is available
on Kaggle. The dataset is provided as separate training and test files.
Exploratory data analysis is performed on the training data only,
as it contains the target variable.

Each row represents a client contacted during a marketing campaign, and
the target variable `y` indicates whether the client subscribed to a
term deposit.

## Project Structure

bank-marketing-ml/<br>
│<br>
├── data/<br>
│   ├── train.csv<br>
│   └── test.csv<br>
├── models/<br>
│   └── model.bin<br>
├── notebooks/<br>
│   ├── 01_eda.ipynb<br>
│   └── 02_modeling.ipynb│<br>
├── screenshots/<br>
│   ├── Screenshot 2026-01-13 141004.png<br>
│   └── Screenshot 2026-01-13 141015.png<br>
│   ├── Screenshot 2026-01-13 141647.png<br>
│   └── Screenshot_Render.png<br>
├── src/<br>
│   └── train.py│<br>
├── app.py<br>
├── Dockerfile<br>
├── README.md<br>
├── requirements.txt<br>
├── sampledata.py<br>

## Exploratory Data Analysis

![alt text](<Screenshots/Screenshot 2026-01-13 141004.png>)

![alt text](<Screenshots/Screenshot 2026-01-13 141415.png>)

![alt text](<Screenshots/Screenshot 2026-01-13 141647.png>)

## Model Selection

Several models were evaluated, including Logistic Regression, Random Forest, Gradient Boosting, and XGBoost.

XGBoost achieved the highest ROC-AUC score (0.93) on the validation set, indicating superior  discriminative performance.

<table>
  <tr style="background-color:#f2f2f2; font-weight:bold;">
    <td>Model</td>
    <td>Precision</td>
    <td>Recall</td>
    <td>F1-Score</td>
    <td>ROC-AUC</td>
  </tr>

  <tr style="background-color:white;">
    <td>XGBoost</td>
    <td>0.65</td>
    <td>0.48</td>
    <td>0.55</td>
    <td>0.933</td>
  </tr>

  <tr style="background-color:#e7f0fd;">
    <td>Random Forest</td>
    <td>0.69</td>
    <td>0.39</td>
    <td>0.45</td>
    <td>0.928</td>
  </tr>

  <tr style="background-color:white;">
    <td>Gradient Boosting</td>
    <td>0.65</td>
    <td>0.41</td>
    <td>0.50</td>
    <td>0.924</td>
  </tr>

  <tr style="background-color:#e7f0fd;">
    <td>Logistic Regression</td>
    <td>0.42</td>
    <td>0.81</td>
    <td>0.55</td>
    <td>0.907</td>
  </tr>
</table>

## Hyperparameter Tuning

Hyperparameter tuning was performed for the XGBoost model using GridSearchCV, as it showed strong  baseline performance and offered a good balance between interpretability and accuracy.

Other models such as Gradient Boosting and Random Forest were trained using reasonable default parameters to compare their performance. Due to time and computational constraints, further tuning was not performed, as the improvement was marginal compared to the tuned Random Forest model.

<table>
  <tr style="background-color:#f2f2f2; font-weight:bold;">
    <td>Model</td>
    <td>Precision</td>
    <td>Recall</td>
    <td>F1-Score</td>
    <td>ROC-AUC</td>
  </tr>

  <tr style="background-color:white;">
    <td>XGBoost</td>
    <td>0.65</td>
    <td>0.48</td>
    <td>0.55</td>
    <td>0.935</td>
  </tr>

  </table>

## Model Evaluation

Due to class imbalance, model performance was evaluated using
precision, recall, F1-score for the positive class (`y = yes`), and
ROC-AUC.

- Precision (class 1): 0.65
- Recall (class 1): 0.48
- F1-score (class 1): 0.55
- ROC-AUC: 0.93

Recall was prioritized as the key metric, as the business objective
is to identify as many potential subscribers as possible.

## How to Run Locally

The model training logic is exported from the notebook into a Python
script for reproducibility.

### Environment Setup

Create and activate a virtual environment
```
  python -m venv venv
```

Windows
```
  venv\Scripts\activate
```

Linux / Mac
```
  source venv/bin/activate
```

Install dependencies:
```
  pip install -r requirements.txt
```

### Local Setup

* Clone the repository and navigate to the project directory.
```
  https://github.com/uma-ram/MLZoomCamp_Capstone_Project
```

* Train the model
```
  python src/train.py
```
* Run the app:
```
  uvicorn app:app --reload
```

* Open browser:
```
  http://127.0.0.1:8000/docs
```

## API Call

You can make a POST request to the /predict endpoint to get a prediction.

Using curl: <br>

```
curl -X 'POST' \
  'https://bank-marketing-ml.onrender.com/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "age": 26,
  "job": "admin.",
  "marital": "married",
  "education": "secondary",
  "default": "no",
  "balance": 45,
  "housing": "no",
  "loan": "no",
  "contact": "unknown",
  "day": 5,
  "month": "may",
  "duration": 1467,
  "campaign": 1,
  "pdays": -1,
  "previous": 0,
  "poutcome": "unknown"
}'

```

 Response

   ```
{
    "subscription_probability": 0.5527550578117371,
    "will_subscribe": true
}
   ```

## Cloud Deployment

The model is deployed using Render and is publicly accessible.

API Endpoint:

https://bank-marketing-ml.onrender.com/docs

![alt text](<Screenshots/Screenshot_Render.png>)
