Golden State Housing Predictor
Objective:
To develop a machine learning model that predicts median housing values in California districts, based on a set of features. This project aims to understand the key factors influencing house prices in different regions of the state and to provide insights through data visualization and modeling.

Technologies Used:
Programming Language: Python
Data Manipulation: pandas
Data Visualization: matplotlib, seaborn
Machine Learning: scikit-learn
Methodology:
Data Exploration: Initial dive into the California housing dataset to understand its structure, features, and target variables. Extensive visualizations were performed to gain insights into the distribution and relationships within the data.

Data Preprocessing: The dataset was split into training and testing sets, ensuring a good mix of data points in both.

Model Selection: Opted for a Linear Regression model given its simplicity and appropriateness for predicting a continuous target variable.

Evaluation: The model's performance was evaluated using the Mean Squared Error (MSE) to understand the prediction accuracy.

Results:
The Linear Regression model was trained on the dataset, and an evaluation was performed on the test set. Key findings include:

A Mean Squared Error of 0.56.
A visualization comparing actual vs. predicted median house values showed a general correlation between the predicted and true values, indicating that the model captures the underlying trend in the data. While many of the points were clustered around a line of perfect prediction, indicating accurate estimations, there were outliers and deviations that signify areas where the model might have struggled.

