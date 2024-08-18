# Importing the necessary libraries
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Fetching the California Housing dataset
housing = fetch_california_housing()

# Splitting the data into training, validation, and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

# Standardizing the data using StandardScaler
scaler = StandardScaler()

# Fitting the scaler on the training data and transforming the validation and test data
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Data preprocessing will be further improved and will be ready to be fed into a TensorFlow model.
# We will define and train our model in the next steps.
