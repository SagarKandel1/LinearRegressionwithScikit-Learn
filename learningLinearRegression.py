# Import necessary libraries
import pandas as pd        # For creating and managing dataframes
import numpy as np         # For numerical operations and generating random numbers
import matplotlib.pyplot as plt  # For plotting graphs
import seaborn as sns      # For advanced statistical plots (scatterplots, regression lines)
from sklearn.linear_model import LinearRegression # For performing linear regression
from sklearn.model_selection import train_test_split # For splitting data into training and testing sets
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error  # For evaluating the regression model
# -----------------------------
# Step 1: Set seed and generate synthetic data
# -----------------------------
np.random.seed(42)  # Ensures the random numbers are the same each time you run the code

num_samples = 500  # Number of data points

# Random years of experience between 2 and 20
years_of_experience = np.random.randint(2, 21, size=num_samples)

# Linear relationship for salaries: slope and intercept
slope = (200_000 - 60_000) / 18  # salary increase per year of experience
intercept = 60_000                # starting salary

# Generate salaries with some random noise to simulate real-world variation
salaries = slope * years_of_experience + intercept + np.random.normal(0, 10_000, size=num_samples)

# -----------------------------
# Step 2: Create DataFrame
# -----------------------------
df = pd.DataFrame({
    'Years_of_Experience': years_of_experience,
    'Salary': salaries
})

# Check the summary statistics of our synthetic dataset
print(df.describe())

# -----------------------------
# Step 3: Plotting
# -----------------------------
plt.figure(figsize=(10, 6))  # Set the size of the figure

# Scatter plot of the data points
# Note: 'scatter=False' was removed because seaborn's scatterplot does not have this argument
sns.scatterplot(
    x='Years_of_Experience',
    y='Salary',
    data=df,
    color='red',
    label='Data Points'
)

# Optional: Plot regression line using seaborn's regplot
# We set scatter=False because we already plotted points above
sns.regplot(
    x='Years_of_Experience',
    y='Salary',
    data=df,
    scatter=False,  # Only draw the regression line, not extra points
    color='blue',
    label='Regression Line'
)

# Add labels and title
plt.xlabel('Years of Experience')
plt.ylabel('Salary (€)')
plt.title('Linear Regression: Years of Experience vs Salary')

# Show the plot
plt.show()


# -----------------------------
# Step 3: Regression
# -----------------------------
X = df[['Years_of_Experience']]  # Feature matrix
y = df['Salary']                 # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Split data into training and testing sets

Lr = LinearRegression()  # Create a Linear Regression model
Lr.fit(X_train, y_train)  # Fit the model to the training data

Lr.score(X_test, y_test) # Evaluate the model on the test data

# -----------------------------
# Step 4: Predictions
# -----------------------------
y_pred = Lr.predict(X_test)  # Make predictions on the test data

print(f'Mean Absolute Error:', mean_absolute_error(y_test, y_pred),"\n") # Mean Absolute Error to evaluate the model

print(f'Mean Squared Error:', mean_squared_error(y_test, y_pred),"\n") # Mean Squared Error to evaluate the model

print(f'R² score:', r2_score(y_test, y_pred),"\n" )# R² score to evaluate the model

print(f'Intercept:', Lr.intercept_,"\n")
print(f'Coefficients:', Lr.coef_,"\n")

coefficients = Lr.coef_
intercept = Lr.intercept_

# Generate 100 points along X-axis for plotting the line
X_line = np.linspace(0, 20, 100)  # 0 to 20 years
y_line = coefficients * X_line + intercept  # regression line

# Plot the regression line
plt.plot(X_line, y_line.flatten(), color='blue', label=f'y = {coefficients[0]:.2f}x + {intercept:.2f}')
plt.xlabel('Years of Experience')
plt.ylabel('Salary (€)')
plt.title('Linear Regression Line')
plt.legend()
plt.grid()
plt.show()


