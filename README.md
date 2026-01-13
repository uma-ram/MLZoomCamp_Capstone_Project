
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

- `notebooks/` – Exploratory data analysis and experiments
- `data/` – Dataset files
- `src/` – Training and evaluation scripts (added later)


## EDA

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