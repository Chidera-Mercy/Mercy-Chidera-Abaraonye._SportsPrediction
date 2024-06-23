#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, VotingRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV
import joblib


# In[2]:


# Load the datasets
df_train = pd.read_csv(r"male_players (legacy).csv")
df_test = pd.read_csv(r"players_22.csv")


# In[3]:


# Drop irrelevant features from the datasets
irrelevant_features_train= [
    'player_id', 'player_url', 'fifa_version', 'fifa_update','short_name', 'long_name', 'player_positions', 'dob', 'league_id', 'league_name', 'league_level', 'club_jersey_number', 'club_loaned_from',
    'club_joined_date', 'club_contract_valid_until_year', 'nationality_id', 'nationality_name', 'nation_team_id', 'nation_position',	'nation_jersey_number', 'real_face', 'release_clause_eur',	'player_tags',	'player_traits',
    'ls', 'st', 'rs', 'lw', 'lf', 'cf', 'rf', 'rw', 'lam', 'cam', 'ram', 'lm', 'lcm', 'cm', 'rcm', 'rm', 'lwb', 'ldm', 'cdm', 'rdm', 'rwb', 'lb', 'lcb', 'cb', 'rcb', 'rb', 'gk',
    'player_face_url', 'fifa_update_date', 'club_name', 'club_team_id'
]
irrelevant_features_test= [
    'sofifa_id', 'player_url', 'short_name', 'long_name', 'player_positions', 'dob', 'league_name', 'league_level', 'club_jersey_number', 'club_loaned_from',
    'club_joined', 'club_contract_valid_until', 'nationality_id', 'nationality_name', 'nation_team_id', 'nation_position',	'nation_jersey_number', 'real_face', 'release_clause_eur',	'player_tags',	'player_traits',
    'ls', 'st', 'rs', 'lw', 'lf', 'cf', 'rf', 'rw', 'lam', 'cam', 'ram', 'lm', 'lcm', 'cm', 'rcm', 'rm', 'lwb', 'ldm', 'cdm', 'rdm', 'rwb', 'lb', 'lcb', 'cb', 'rcb', 'rb', 'gk',
    'player_face_url', 'club_team_id', 'club_logo_url', 'club_flag_url', 'nation_logo_url', 'nation_flag_url', 'club_name'
]

df_train.drop(irrelevant_features_train, axis=1, inplace=True)
df_test.drop(irrelevant_features_test, axis=1, inplace=True)


# In[4]:


# Visualize distribution of 'overall' (target variable)
plt.figure(figsize=(10, 6))
sns.histplot(df_train['overall'], kde=True)
plt.title('Distribution of Overall Ratings')
plt.xlabel('Overall Rating')
plt.ylabel('Frequency')
plt.show()


# In[5]:


# Separate the numeric and non numeric features
numeric_features = df_train.select_dtypes(include=[np.number])
categorical_features = df_train.select_dtypes(include=[object])


# In[6]:


# Scatter plots for numeric features vs 'overall'
for feature in numeric_features:
    if feature != 'overall':
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_train, x=feature, y='overall')
        plt.title(f'{feature} vs Overall')
        plt.xlabel(feature)
        plt.ylabel('Overall')
        plt.show()

# Box plots for categorical features vs 'overall'
for feature in categorical_features:
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=df_train, x=feature, y='overall')
    plt.title(f'{feature} vs Overall')
    plt.xlabel(feature)
    plt.ylabel('Overall')
    plt.show()


# In[7]:


# One-hot encoding for categorical features
df_train_encoded = df_train.copy()
df_train_encoded = pd.get_dummies(df_train_encoded, columns=categorical_features.columns, drop_first=True)


# In[8]:


# Compute the correlation matrix
corr_matrix = df_train_encoded.corr()


# In[9]:


# Filter features with a high correlation to 'overall'
correlation_threshold = 0.5
high_corr_features = corr_matrix.index[abs(corr_matrix["overall"]) > correlation_threshold].tolist()

# Create a dataset with selected features and some categorical features
selected_features_train = df_train[high_corr_features + ['work_rate', 'body_type']]
selected_features_test = df_test[high_corr_features + ['work_rate', 'body_type']]


# In[10]:


# Filter columns with 30% or more missing values
L =[]
L_less =[]
for i in selected_features_train.columns:
    if((selected_features_train[i].isnull().sum()) < (0.3*(df_train.shape[0]))):
        L.append(i)
    else:
        L_less.append(i)

selected_features_train = selected_features_train[L]
selected_features_test = selected_features_test[L]


# In[11]:


# Separate numeric and non_numeric data
numeric_data_train = selected_features_train.select_dtypes(include=np.number)
numeric_data_test = selected_features_test.select_dtypes(include=np.number)

non_numeric_train = selected_features_train.select_dtypes(include = ['object'])
non_numeric_test = selected_features_test.select_dtypes(include = ['object'])


# In[12]:


# Fill missing values in the numeric dataset
imp = IterativeImputer(max_iter=10, random_state=0)
numeric_data_train = pd.DataFrame(np.round(imp.fit_transform(numeric_data_train)), columns=numeric_data_train.columns)
numeric_data_test = pd.DataFrame(np.round(imp.fit_transform(numeric_data_test)), columns=numeric_data_test.columns)


# In[13]:


# Fill missing values in the non_numeric dataset
cat_imputer = SimpleImputer(strategy='most_frequent')
non_numeric_train = pd.DataFrame(cat_imputer.fit_transform(non_numeric_train), columns=non_numeric_train.columns)
non_numeric_test = pd.DataFrame(cat_imputer.fit_transform(non_numeric_test), columns=non_numeric_test.columns)


# In[14]:


# One Hot Encode non-numeric features
non_numeric_train = pd.get_dummies(non_numeric_train).astype(int)
non_numeric_test = pd.get_dummies(non_numeric_test).astype(int)


# In[15]:


# Prepare final datasets
YTrain = numeric_data_train['overall']
YTest = numeric_data_test['overall']
XTrain = pd.concat([numeric_data_train.drop('overall', axis=1), non_numeric_train],axis=1)
XTest = pd.concat([numeric_data_test.drop('overall', axis=1), non_numeric_test],axis=1)


# In[16]:


# Scale features
scaler = StandardScaler()
XTrain = scaler.fit_transform(XTrain)
XTest = scaler.fit_transform(XTest)


# In[17]:


# Function to train and evaluate models
def evaluate_models(models, XTrain, YTrain, XTest, YTest):
    results = {}
    predictions = {}
    
    for name, model in models.items():
        model.fit(XTrain, YTrain)
        y_pred = model.predict(XTest)
        mse = mean_squared_error(YTest, y_pred)
        results[name] = mse
        predictions[name] = pd.DataFrame({'Actual': YTest, 'Predicted': y_pred})
        print(f"{name}: Mean Squared Error = {mse}")
    
    return results, predictions


# In[18]:


# Train and evaluate models
models = {
    'Gradient Boosting': GradientBoostingRegressor(),
    'Bagging': BaggingRegressor(),
    'Random Forest': RandomForestRegressor(),
    'XGB': XGBRegressor()
}
# Evaluate models
results, predictions = evaluate_models(models, XTrain, YTrain, XTest, YTest)

print(results)
print(predictions)


# In[19]:


# Prepare another dataset for train and test with just the earlier selected numeric features
YTrain = numeric_data_train['overall']
YTest = numeric_data_test['overall']
XTrain = numeric_data_train.drop('overall', axis=1)
XTest = numeric_data_test.drop('overall', axis=1)


# In[20]:


# Scale features
scaler = StandardScaler()
XTrain = scaler.fit_transform(XTrain)
XTest = scaler.fit_transform(XTest)


# In[21]:


joblib.dump(scaler, 'scaler.pkl')


# In[22]:


# Train and evaluate earlier define models
results, predictions = evaluate_models(models, XTrain, YTrain, XTest, YTest)

print(results)
print(predictions)


# In[23]:


# Function to evaluate voting ensemble model
def evaluate_voting_ensemble(voting_ensemble, XTrain, YTrain, XTest, YTest):
    # Make predictions
    train_predictions = voting_ensemble.predict(XTrain)
    test_predictions = voting_ensemble.predict(XTest)

    predictions= pd.DataFrame({'Actual': YTest, 'Predicted': test_predictions})

    # Evaluate predictions
    mse_train = mean_squared_error(YTrain, train_predictions)
    mse_test = mean_squared_error(YTest, test_predictions)
    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)

    return rmse_train, rmse_test, mse_train, mse_test, predictions


# In[24]:


# Initialize the models
rf = RandomForestRegressor(random_state=42)
bagging = BaggingRegressor(random_state=42)
xgb = XGBRegressor(random_state=42, use_label_encoder=False, eval_metric='rmse')


# In[27]:


# Define the parameter grids for different models
rf_param_grid = {
    'n_estimators': [100, 200, 400],
    'max_depth': [10, 20, 30],
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [2, 4, 6]
}

bagging_param_grid = {
    'n_estimators': [10, 20, 40],
    'max_samples': [0.5, 0.75, 1.0],
    'max_features': [0.5, 0.75, 1.0]
}

xgb_param_grid = {
    'n_estimators': [100, 200, 400],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}


# In[29]:


# Apply GridSearchCV for RandomForestRegressor
rf_grid_search = GridSearchCV(estimator=rf, param_grid=rf_param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
rf_grid_search.fit(XTrain, YTrain)
best_rf_model = rf_grid_search.best_estimator_

# Apply GridSearchCV for BaggingRegressor
bagging_grid_search = GridSearchCV(estimator=bagging, param_grid=bagging_param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
bagging_grid_search.fit(XTrain, YTrain)
best_bagging_model = bagging_grid_search.best_estimator_

# Apply GridSearchCV for XGBRegressor
xgb_grid_search = GridSearchCV(estimator=xgb, param_grid=xgb_param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
xgb_grid_search.fit(XTrain, YTrain)
best_xgb_model = xgb_grid_search.best_estimator_


# In[31]:


# Print best parameters
print(f'Best parameters for Random Forest: {rf_grid_search.best_params_}')
print(f'Best parameters for Bagging Regressor: {bagging_grid_search.best_params_}')
print(f'Best parameters for XGB Regressor: {xgb_grid_search.best_params_}')


# In[45]:


# Create a Voting Regressor with the best models
voting_ensemble = VotingRegressor(estimators=[
    ('rf', best_rf_model),
    ('bagging', best_bagging_model),
    ('xgb', best_xgb_model)
])

# Train the voting ensemble
voting_ensemble.fit(XTrain, YTrain)


# In[47]:


# Evaluate the ensemble model
rmse_train, rmse_test, mse_train, mse_test, predictions = evaluate_voting_ensemble(voting_ensemble, XTrain, YTrain, XTest, YTest)

# Print the results
print(f'Voting Ensemble Train RMSE: {rmse_train}')
print(f'Voting Ensemble Test RMSE: {rmse_test}')
print(f'Voting Ensemble Train MSE: {mse_train}')
print(f'Voting Ensemble Test MSE: {mse_test}')

print(predictions)


# In[51]:


# Create a Voting Regressor with the best models
voting_ensemble1 = VotingRegressor(estimators=[
    ('rf', best_rf_model),
    ('xgb', best_xgb_model)
])

# Train the voting ensemble
voting_ensemble1.fit(XTrain, YTrain)


# In[53]:


# Evaluate the ensemble model
rmse_train, rmse_test, mse_train, mse_test, predictions = evaluate_voting_ensemble(voting_ensemble1, XTrain, YTrain, XTest, YTest)

# Print the results
print(f'Voting Ensemble Train RMSE: {rmse_train}')
print(f'Voting Ensemble Test RMSE: {rmse_test}')
print(f'Voting Ensemble Train MSE: {mse_train}')
print(f'Voting Ensemble Test MSE: {mse_test}')

print(predictions)


# In[41]:


# Evaluate the  model
rmse_train, rmse_test, mse_train, mse_test, predictions = evaluate_voting_ensemble(best_rf_model, XTrain, YTrain, XTest, YTest)

# Print the results
print(f'Voting Ensemble Train RMSE: {rmse_train}')
print(f'Voting Ensemble Test RMSE: {rmse_test}')
print(f'Voting Ensemble Train MSE: {mse_train}')
print(f'Voting Ensemble Test MSE: {mse_test}')

print(predictions)


# In[43]:


# Evaluate the ensemble model
rmse_train, rmse_test, mse_train, mse_test, predictions = evaluate_voting_ensemble(best_xgb_model, XTrain, YTrain, XTest, YTest)

# Print the results
print(f'Voting Ensemble Train RMSE: {rmse_train}')
print(f'Voting Ensemble Test RMSE: {rmse_test}')
print(f'Voting Ensemble Train MSE: {mse_train}')
print(f'Voting Ensemble Test MSE: {mse_test}')

print(predictions)


# In[55]:


# Evaluate the ensemble model
rmse_train, rmse_test, mse_train, mse_test, predictions = evaluate_voting_ensemble(best_bagging_model, XTrain, YTrain, XTest, YTest)

# Print the results
print(f'Voting Ensemble Train RMSE: {rmse_train}')
print(f'Voting Ensemble Test RMSE: {rmse_test}')
print(f'Voting Ensemble Train MSE: {mse_train}')
print(f'Voting Ensemble Test MSE: {mse_test}')

print(predictions)


# In[57]:


voting_ensemble2 = VotingRegressor(estimators=[
    ('rf', best_rf_model),
    ('bagging', best_bagging_model)
])

# Train the voting ensemble
voting_ensemble2.fit(XTrain, YTrain)


# In[59]:


# Evaluate the ensemble model
rmse_train, rmse_test, mse_train, mse_test, predictions = evaluate_voting_ensemble(voting_ensemble2, XTrain, YTrain, XTest, YTest)

# Print the results
print(f'Voting Ensemble Train RMSE: {rmse_train}')
print(f'Voting Ensemble Test RMSE: {rmse_test}')
print(f'Voting Ensemble Train MSE: {mse_train}')
print(f'Voting Ensemble Test MSE: {mse_test}')

print(predictions)


# In[61]:


# Save the model
joblib.dump(voting_ensemble1, 'model.pkl')

