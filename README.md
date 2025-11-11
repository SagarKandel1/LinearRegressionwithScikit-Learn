# Salary Prediction using Linear Regression

## Project Overview
This project demonstrates a **Linear Regression** model to predict salaries based on years of experience. It uses **synthetic data** to simulate real-world variations, and it provides insights into how experience correlates with salary. The project includes visualizations and evaluation metrics to analyze the model's performance and accuracy.

## Purpose
The purpose of this project is to:
- Understand the relationship between years of experience and salary.
- Learn how to generate and work with synthetic datasets.
- Apply Linear Regression to predict continuous values.
- Evaluate model performance using standard regression metrics.
- Visualize data and predictions for better understanding.

## Features
- Synthetic dataset generation for 500 individuals with random noise.
- Scatter plots showing salaries versus years of experience.
- Linear Regression modeling to fit the salary data.
- Evaluation of predictions using:
  - **Mean Absolute Error (MAE)**
  - **Mean Squared Error (MSE)**
  - **R² Score**
- Visual representation of regression lines to show predicted trends.

## Libraries Used
- **NumPy (`np`)**: for numerical operations and generating random numbers.
- **pandas (`pd`)**: for creating and managing structured data.
- **Matplotlib (`plt`)**: for plotting and visualizing data.
- **Seaborn (`sns`)**: for advanced statistical plots and visualizations.
- **scikit-learn**: for building and evaluating the Linear Regression model, splitting datasets, and calculating performance metrics.

## Dataset
The dataset is synthetic, generated using random years of experience (between 2 and 20 years) and a salary formula with added noise. This simulates real-world salary variations and provides a controlled environment for testing regression models.

## Visualizations
- **Scatter Plot**: Red dots show the relationship between years of experience and salary, illustrating the distribution of data points.
- **Regression Line**: Blue line demonstrates the predicted trend of salary as experience increases, helping to visualize the model’s predictions.

## Model Evaluation
The Linear Regression model is evaluated using:
- **Mean Absolute Error (MAE)**: average magnitude of errors between predicted and actual salaries.
- **Mean Squared Error (MSE)**: measures the average squared difference between predicted and actual salaries.
- **R² Score**: indicates how well the regression model explains the variance in the data.

These metrics provide a comprehensive understanding of the model’s accuracy and reliability.

## Example Output
**Summary Statistics of Data:**
- Count: 500 data points
- Mean Years of Experience: ~11 years
- Mean Salary: ~€134,000
- Standard Deviation (Salary): ~€29,000
- Minimum Salary: €60,000
- Maximum Salary: €200,000

**Regression Metrics (Example):**
- Mean Absolute Error: ~€7,900
- Mean Squared Error: ~€101,000,000
- R² Score: ~0.95

These outputs demonstrate that the model effectively captures the linear relationship between experience and salary.

## Conclusion
This project illustrates how Linear Regression can be applied to predict continuous outcomes like salary. By generating synthetic data, visualizing trends, and evaluating model performance, it provides a complete workflow for regression analysis. It serves as a practical example for learning predictive modeling, data visualization, and model evaluation techniques.

## License
This project is licensed under the MIT License.
