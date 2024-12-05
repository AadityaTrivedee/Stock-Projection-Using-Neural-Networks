# Stock-Projection-Using-Neural-Networks

## Overview
This project focuses on predicting stock prices using neural networks, specifically leveraging TensorFlow and Keras. The model is trained on historical stock data of Nvidia (NVDA) to predict future stock prices. The project includes data preprocessing, feature engineering, model training, and evaluation.


## Table of Contents
- [Overview](#overview)
- [Libraries Used](#libraries-used)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Model Training](#model-training)
- [Results](#results)
- [Prediction](#prediction)
- [Conclusion & Random Walk Theory](#conclusion-and-the-random-walk-theory)

## Libraries Used
- **NumPy**: For numerical operations.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib & Seaborn**: For data visualization.
- **Scikit-learn**: For data splitting.
- **TensorFlow**: For building and training the neural network model.

## Data Preprocessing
1. **Loading Data**: The dataset `nvda.csv` is loaded using Pandas.
2. **Feature Selection**: Relevant features such as `Open`, `High`, `Low`, `Close`, and `Volume` are selected.
3. **Data Cleaning**: Checking for null values and ensuring data integrity.
4. **Feature Engineering**:
   - Adding new features like `open-close` and `high-low`.
   - Creating a `target` feature which is the `Close` price shifted by one day.
   - Adding a feature `is_quarter_end` to indicate the end of a quarter.
5. **Data Visualization**:
   - Plotting the closing price over time.
   - Visualizing the distribution and box plots of selected features.
   - Heatmap to show the correlation between features.
  
## Model Architecture
The neural network model is built using TensorFlow and Keras. The architecture includes:
- **Input Layer**: 7 input features.
- **Hidden Layers**: Two dense layers with 256 units each and ReLU activation.
- **Output Layer**: A single unit to predict the stock price.

## Model Training
- **Optimizer**: Adam.
- **Loss Function**: Mean Absolute Error (MAE).
- **Batch Size**: 64.
- **Epochs**: 100.

## Results
- **Training Loss**: The model achieves a low training loss, indicating a good fit on the training data.
- **Validation Loss**: The validation loss is also low, suggesting that the model generalizes well to unseen data.
- **MAPE**:
  - **Training Set**: 2.3475% (**_Accuracy:97.6525_**)
  - **Development Set**: 2.3773% (**_Accuracy:97.6227_**)
  - **Test Set**: 2.5689% (**_Accuracy:97.4311_**)

## Prediction
The model is used to predict the stock price for the next day based on the latest data point. For example:
```python
Predicted stock price for the next day: [[142.729]]
```

## Conclusion and The Random Walk Theory
This project demonstrates the application of neural networks for stock price prediction. The model is trained on historical data. It is important to note that while the model showcases high accuracy in predictions, the Random Walk Theory suggests that stock prices follow a random walk and are inherently unpredictable. This theory posits that past movements or trends in stock prices cannot be used to predict future movements. Therefore, relying solely on historical data for stock price prediction may not be a foolproof strategy.

Despite the model's high accuracy, it is crucial to consider the Random Walk Theory and understand that stock prices are influenced by a multitude of factors, many of which are unpredictable. Historical data should not be the sole focus for making investment decisions. Instead, a comprehensive approach that includes fundamental analysis, market sentiment, and other economic indicators should be adopted.
