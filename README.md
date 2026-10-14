# House Price Prediction

Predicts house prices using Linear Regression on the California Housing dataset. Boston Housing was removed from scikit-learn, so California Housing is used as the standard replacement.

## What the script does
1. Loads the California Housing dataset (target: MedHouseVal, in $100,000s)
2. Explores the data with summary statistics, histograms and a correlation matrix
3. Checks for and fills missing values
4. Splits the data into 80% train and 20% test
5. Normalizes features with StandardScaler
6. Trains a Linear Regression model
7. Evaluates it with MSE, RMSE, MAE and R2 score
8. Prints feature coefficients and plots actual vs predicted prices

## Files
- task3_house_price_prediction.py: main script
- house_histograms.png: feature distributions
- house_correlation.png: correlation matrix
- house_actual_vs_predicted.png: actual vs predicted prices

## Libraries
numpy, pandas, matplotlib, scikit-learn

## How to run
pip install numpy pandas matplotlib scikit-learn
python task3_house_price_prediction.py
