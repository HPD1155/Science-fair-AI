import numpy as np
import random
from sklearn.model_selection import train_test_split
import pandas as pd
# import losses
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import math
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


df = pd.read_csv('/kaggle/input/science-fair/infection_data.csv')

X = df[['population_size', 'population_density', 'contact_rate', 'contact_through_animal', 'timeframe']]
y = df['infected_population']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=True)

plt.scatter(X_train['population_size'], y_train)
plt.scatter(X_valid['population_size'], y_valid)
plt.scatter(X_test['population_size'], y_test)
plt.show()

# show train test val split on bar graph
plt.bar(['train', 'valid', 'test'], [len(X_train), len(X_valid), len(X_test)])
plt.show()

model = DecisionTreeRegressor(max_depth=50, random_state=64)

model.fit(X_train, y_train)

X_test_new = X_test[:1]
y_predicted = model.predict(X_test_new)
print(f"Score for current predicted values is {model.score(X_test_new, y_predicted)}")
print(y_predicted)


print("R-squared score on training set:", model.score(X_train, y_train))
print("R-squared score on test set:", model.score(X_test, y_test))

import seaborn as sns

corr_matrix = df.corr()  # Calculate correlation matrix
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')  # Customize as needed
plt.title("Correlation Heatmap")
plt.show()
# Plotting the test data and predicted values
plt.scatter(X_test['population_size'], y_test, label='Actual values')
plt.scatter(X_test['population_size'], model.predict(X_test), label='Predicted values', color='r')

# Calculating and plotting the regression line for visualization
m, b = np.polyfit(X_test['population_size'], y_test, 1)
plt.plot(X_test['population_size'], m*X_test['population_size'] + b, color='g', label='Regression Line')

# Adding labels and title
plt.xlabel('Population Size')
plt.ylabel('Infected Population')
plt.title('Test Data and Predicted Values')
plt.legend()

# Display R-squared value as a measure of accuracy
r_squared = model.score(X_test, y_test)
plt.figtext(0.15, 0.95, f'R-squared = {r_squared:.2f}', fontsize=10, ha='left')

plt.show()
