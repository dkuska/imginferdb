import sys
import os
from pathlib import Path 
x_folder = Path(__file__).resolve().parents[2]
src_folder = os.path.join(x_folder, 'src')
sys.path.append(str(src_folder))
featurizer_folder = os.path.join(src_folder, 'featurizers')
sys.path.append(str(featurizer_folder))
from transpiler import Standalone
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, OrdinalEncoder
from scipy.stats import uniform, randint
from sklearn.metrics import mean_squared_log_error, make_scorer
from pathlib import Path
from nyc_rides_featurizer import NYC_Featurizer
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from datetime import time
from lightgbm import LGBMRegressor
import numpy as np
from sklearn.impute import SimpleImputer

# data_path = os.path.join(x_folder, 'data', 'nyc_rides', 'train.csv')
data_path = os.path.join(x_folder, 'data', 'nyc_rides', 'nyc_rides_augmented.csv')
# augmented_data_path = os.path.join(x_folder, 'data', 'nyc_rides', 'train_augmented.csv')
# fastest_routes_path_1 = os.path.join(x_folder, 'data', 'nyc_rides', 'fastest_routes_train_part_1.csv')
# fastest_routes_path_2 = os.path.join(x_folder, 'data', 'nyc_rides', 'fastest_routes_train_part_2.csv')

df = pd.read_csv(data_path)
# augmented_df = pd.read_csv(augmented_data_path)
# fastest_routes_1 = pd.read_csv(fastest_routes_path_1)
# fastest_routes_2 = pd.read_csv(fastest_routes_path_2)
# frames = [fastest_routes_1, fastest_routes_2]
# fastest_routes_df = pd.concat(frames)

# df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
# df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])

# df = df.set_index('id').join(augmented_df.set_index('id'), how='right')
# df = df.join(fastest_routes_df.set_index('id'), how='right')
# df.reset_index(drop=False, inplace=True)

# ##### Data Cleaning
# df = df.loc[df['dropoff_datetime'] > df['pickup_datetime'], :]
# df = df.loc[(df['trip_duration'] > 0) & (df['trip_duration'] < (24 * 60 * 60)), :]
# threshold = pd.Timedelta('00:00:00')
# df = df.loc[df['dropoff_datetime'].dt.time > time(0,0,0), :]

# write_path = os.path.join(x_folder, 'data', 'nyc_rides', 'nyc_rides_augmented.csv')
# df.to_csv(write_path, index=False)

training_features = [i for i in list(df) if i != 'trip_duration']
X = df[training_features]
y = df['trip_duration'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)

numerical_transformer = Pipeline(
                    steps=
                            [
                                ('scaler', RobustScaler())
                            ]
                    )
categorical_transformer = Pipeline(
    steps=
            [
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ]
    )

categorical_transformer_trees = Pipeline(
    steps=
            [
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ]
    )

column_transformer = ColumnTransformer(
                                        transformers=[
                                                        ('num', numerical_transformer, [])
                                                    ]
                                        , remainder='drop'
                                        , n_jobs=-1

                                    )

featurizer = NYC_Featurizer('deep')

pipeline = Pipeline(
                        steps=
                                [   
                                    ('featurizer', featurizer),
                                    ('imputer', SimpleImputer()),
                                    ('column_transformer', column_transformer),
                                    
                                    # , ('stacking_regressor', StackingRegressor(estimators=estimators, passthrough=True, final_estimator=model, n_jobs=-1))
                                    # ('clf', model)
                                ]
                    )

xgboost = xgb.XGBRegressor(n_estimators=1000, objective="reg:squaredlogerror", random_state=42, n_jobs=-1, min_child_weight=10, subsample=0.8, eta=0.05)
lr = LinearRegression(n_jobs=-1)
dt = DecisionTreeRegressor()
nn = MLPRegressor(hidden_layer_sizes=(100,), max_iter=10000, activation='logistic', alpha=0.005)
knn = KNeighborsRegressor(algorithm='kd_tree', n_jobs=-1)
lgbm = LGBMRegressor(n_estimators=1000, n_jobs=-1, objective='regression', reg_lambda=1, reg_alpha=1)

models = [
    # xgboost, 
    lr, 
    # dt, 
    # nn, 
    # knn, 
    # lgbm
    ]
tree_based_models = [xgboost.__class__.__name__, dt.__class__.__name__, lgbm.__class__.__name__]

df = pd.DataFrame()
for model in models:

    if model.__class__.__name__ in tree_based_models:
        params = {'remainder':'passthrough'}
        column_transformer.set_params(**params)
        pipeline.named_steps.column_transformer.transformers = [('num', numerical_transformer, [])]
    else:
        params = {'remainder':'drop'}
        column_transformer.set_params(**params)
        pipeline.named_steps.column_transformer.transformers = [
                                                                ('num', numerical_transformer, [i for i in range(len(featurizer.num_features))])
                                                                ]
        
    pipeline.steps.append(['clf', model])

    inferdb_categorical = []
    
    exp = Standalone(X_train, X_test, y_train, y_test, 'nyc_rides', 'regression', True, True, pipeline)
    for i in range(5):
        d = exp.create_report(cat_mask=inferdb_categorical)
        d['iteration'] = i
        df = pd.concat([df, d])
        export_path = os.path.join(x_folder, 'experiments', 'output', exp.experiment_name)
        df.to_csv(export_path + '_complex_standalone.csv', index=False)
    
    pipeline.steps.pop(-1)