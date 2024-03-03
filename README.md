# Spotify Data Analysis and Stream Prediction 🎵📊

Welcome to the Spotify Data Analysis and Stream Prediction project! In this innovative endeavor, we delve into the world of Spotify datasets to uncover insights and predict the number of streams for songs based on various features. Let's embark on this exciting journey!

## Data Exploration and Preprocessing 🚀🔍

To begin our journey, we import necessary libraries and load the Spotify dataset. We explore the data, handle missing values, and preprocess it for analysis.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set display options for better visibility
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

# Load the dataset
spotify_data = pd.read_csv('/content/spotify-2023.csv', encoding='ISO-8859-1')

# Explore the dataset
print('Dataframe shape:', spotify_data.shape)
print(spotify_data.dtypes)
print('Missing values in each column:\n', spotify_data.isnull().sum())
```

## Exploratory Data Analysis (EDA) 📈🔬

Let's visualize and analyze Spotify data to gain valuable insights.

```python
# Summary statistics of 'streams'
print(spotify_data['streams'].describe())

# Plotting a histogram of 'streams'
plt.figure(figsize=(10, 5))
sns.histplot(data=spotify_data['streams'], color='g', bins=100, alpha=0.4, kde=True)
plt.show()

# Visualizing histograms of various features
spotify_data.hist(figsize=(14, 16), bins=50, xlabelsize=8, ylabelsize=8)
```

## Feature Engineering 🛠️🧩

We engineer features and preprocess data to prepare it for modeling.

```python
# Combine playlist and chart features
spotify_data['total_playlist'] = spotify_data[['in_spotify_playlists', 'in_apple_playlists', 'in_deezer_playlists']].sum(axis=1)
spotify_data['total_charts'] = spotify_data[['in_spotify_charts', 'in_apple_charts', 'in_deezer_charts', 'in_shazam_charts']].sum(axis=1)

# Correlation matrix
correlation_matrix = spotify_data.corr()

# Heatmap of correlations
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .5})
plt.show()
```

## Model Building and Evaluation 🤖📊

We train machine learning models to predict the number of streams for songs.

```python
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Set up features and target variables
X = spotify_data.drop('streams', axis=1)
y = spotify_data['streams']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# XGBRegressor
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# Model evaluation
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

y_pred_xgb = xgb_model.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Model evaluation after hyperparameter tuning
y_pred = grid_search.best_estimator_.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Feature importance
feature_importance = pd.Series(grid_search.best_estimator_.feature_importances_, index=X.columns).sort_values(ascending=False)

# Save the model
import pickle
filename = 'spotify_model.sav'
pickle.dump(grid_search.best_estimator_, open(filename, 'wb'))

# Load the model
loaded_model = pickle.load(open(filename, 'rb'))

# Prediction
input_df = pd.DataFrame(columns=X.columns)
for col in input_df.columns:
    value = input(f'Enter the value for {col}: ')
    input_df.loc[0, col] = value
input_df = input_df.apply(pd.to_numeric)
prediction = loaded_model.predict(input_df)
print(f'The predicted number of streams for the song is {prediction[0]:.0f}')
```

## Conclusion 🎉📝

In this mind-blowing journey, we explored Spotify data, engineered features, built machine learning models, and predicted the number of streams for songs with remarkable accuracy. 🎶✨
