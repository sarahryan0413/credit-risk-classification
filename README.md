# Credit Risk Analysis Report

## Overview of the Analysis

The purpose of this analysis was to build and evaluate machine learning models to predict the credit risk of loans. Specifically, we aimed to determine whether a loan was likely to be a healthy loan (0) or a high-risk loan (1) using a set of financial features.

The dataset provided financial information about loans, such as loan size, interest rate, borrower income, debt to income, number of accounts, derogatory marks, total debt, and loan status. The target variable was loan status, where:
- 0 represents a healthy loan
- 1 represents a high-risk loan

To better understand the target variable, I used value_counts() to identify class distribution. The data was imbalanced, with far more healthy loans than high-risk ones.

We followed the standard machine learning process:
1. Data preparation – Separated features (X) and labels (y), and split the data into training and testing sets.
2. Modeling – Trained a Logistic Regression model using LogisticRegression from sklearn.linear_model.
3. Evaluation – Used a confusion matrix and a classification report to assess model performance, including precision, recall, and accuracy.

## Results

Machine Learning Model 1: Logistic Regression
- Accuracy: 0.99
- Precision (class 0 - healthy loans): 1.00
- Recall (class 0 - healthy loans): 0.99
- Precision (class 1 - high-risk loans): 0.84
- Recall (class 1 - high-risk loans): 0.94

## Summary

The logistic regression model performed very well overall, achieving 99% accuracy on the test data. However, because of the class imbalance, accuracy alone can be misleading. That’s why we looked at precision and recall, especially for high-risk loans.

The model is particularly strong at identifying healthy loans (perfect precision), but also does a solid job identifying high-risk loans, with 94% recall. This means it correctly flags most high-risk loans — which is critical in financial risk analysis.

## Recommendation:

Given the high recall for high-risk loans and overall strong performance, I recommend using the Logistic Regression model as a baseline for production or further tuning. It balances precision and recall well, especially for the minority class (1), which is often the most important in credit risk analysis.
