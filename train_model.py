import pandas as pd
#import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import uniform
import joblib

# Load dataset
file_path = "Global Health Statistics.csv"
data = pd.read_csv(file_path)

# Define target and predictors
target = "Mortality Rate (%)"
predictors = [
    "Prevalence Rate (%)", "Incidence Rate (%)", "Age Group", "Gender",
    "Healthcare Access (%)", "Doctors per 1000", "Hospital Beds per 1000",
    "Treatment Type", "Average Treatment Cost (USD)", "Availability of Vaccines/Treatment",
    "Recovery Rate (%)", "DALYs", "Improvement in 5 Years (%)",
    "Per Capita Income (USD)", "Education Index", "Urbanization Rate (%)",
]

X = data[predictors]
y = data[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
numerical_features = X.select_dtypes(include=["float64", "int64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Model pipeline
model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", GradientBoostingRegressor(random_state=42)),
    ]
)

# Distribution for RandomizedSearchCV
param_dist = {
    "regressor__n_estimators": [50, 100, 150],
    "regressor__learning_rate": uniform(0.05, 0.05),
    "regressor__max_depth": [3, 5, 7],
    "regressor__subsample": uniform(0.8, 0.2),
}

# Randomized search with cross-validation
random_search = RandomizedSearchCV(
    estimator=model_pipeline,
    param_distributions=param_dist,
    n_iter=5,
    cv=2,
    scoring="r2",
    n_jobs=-1,
    verbose=2,
    random_state=42,
)

# Fit the randomized search to the training data
random_search.fit(X_train, y_train)

# Best model
best_model = random_search.best_estimator_

# Save the trained model
joblib.dump(best_model, "mortality_rate_model.pkl")

# Evaluate the model
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Feature importance
regressor = best_model.named_steps["regressor"]
feature_importances = regressor.feature_importances_

# Get the feature names after preprocessing
preprocessed_feature_names = (
    numerical_features.tolist()
    + best_model.named_steps["preprocessor"]
    .transformers_[1][1]
    .get_feature_names_out(categorical_features)
    .tolist()
)

# Create a DataFrame for feature importances
importance_df = pd.DataFrame({
    "Feature": preprocessed_feature_names,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)

print("Feature Importances:")
print(importance_df)

