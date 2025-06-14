import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, explained_variance_score,
                             max_error, mean_absolute_percentage_error)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import SelectKBest, f_regression


# Import ALL Regression Models

# Linear Models
from sklearn.linear_model import (LinearRegression,
                                  Ridge,
                                  Lasso,
                                  ElasticNet,
                                  BayesianRidge,
                                  ARDRegression,
                                  SGDRegressor,
                                  HuberRegressor,
                                  TheilSenRegressor,
                                  RANSACRegressor,
                                  PassiveAggressiveRegressor,
                                  OrthogonalMatchingPursuit)

# Tree-based Models
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor,
                              GradientBoostingRegressor,
                              AdaBoostRegressor,
                              ExtraTreesRegressor,
                              BaggingRegressor,
                              StackingRegressor,
                              VotingRegressor,
                              HistGradientBoostingRegressor)

# SVM Models
from sklearn.svm import SVR, LinearSVR, NuSVR

# Nearest Neighbors
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor

# Neural Networks
from sklearn.neural_network import MLPRegressor

# Gaussian Processes
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, RationalQuadratic

# Ensemble Methods
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Other regressors
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_decomposition import PLSRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.dummy import DummyRegressor
from sklearn.compose import TransformedTargetRegressor

# Data Preparation

# Load your data (replace with your actual data loading)
# X, y = load_data()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
preprocessor = Pipeline([
    ('scaler', StandardScaler()),  # Standard scaling works well for most models
    ('feature_selector', SelectKBest(f_regression, k='all'))  # Can adjust 'k' as needed
])

# Define ALL Models with Parameter Grids

# Define ALL Models with Parameter Grid
models = {
    # ========== Linear Models ==========
    'Linear Regression': {
        'model': LinearRegression(),
        'params': {
            'fit_intercept': [True, False],
            'positive': [True, False]
        }
    },

    'Ridge Regression': {
        'model': Ridge(random_state=42),
        'params': {
            'alpha': [0.1, 0.5, 1.0, 2.0, 5.0],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        }
    },

    'Lasso Regression': {
        'model': Lasso(random_state=42),
        'params': {
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
            'selection': ['cyclic', 'random']
        }
    },

    'ElasticNet': {
        'model': ElasticNet(random_state=42),
        'params': {
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
            'selection': ['cyclic', 'random']
        }
    },

    'Bayesian Ridge': {
        'model': BayesianRidge(),
        'params': {
            'alpha_1': [1e-6, 1e-5, 1e-4],
            'alpha_2': [1e-6, 1e-5, 1e-4],
            'lambda_1': [1e-6, 1e-5, 1e-4],
            'lambda_2': [1e-6, 1e-5, 1e-4]
        }
    },

    'ARD Regression': {
        'model': ARDRegression(),
        'params': {
            'alpha_1': [1e-6, 1e-5, 1e-4],
            'alpha_2': [1e-6, 1e-5, 1e-4],
            'lambda_1': [1e-6, 1e-5, 1e-4],
            'lambda_2': [1e-6, 1e-5, 1e-4],
            'threshold_lambda': [100, 1000, 10000]
        }
    },

    'SGD Regressor': {
        'model': SGDRegressor(random_state=42),
        'params': {
            'loss': ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive']
        }
    },

    'Huber Regressor': {
        'model': HuberRegressor(),
        'params': {
            'epsilon': [1.1, 1.35, 1.5],
            'alpha': [0.0001, 0.001, 0.01],
            'max_iter': [100, 200, 300]
        }
    },

    'Theil-Sen Regressor': {
        'model': TheilSenRegressor(random_state=42),
        'params': {
            'max_subpopulation': [1000, 5000, 10000],
            'n_subsamples': [None, 100, 200],
            'max_iter': [100, 300, 500]
        }
    },

    'RANSAC Regressor': {
        'model': RANSACRegressor(random_state=42),
        'params': {
            'min_samples': [None, 0.1, 0.5, 0.9],
            'residual_threshold': [None, 1.0, 2.0],
            'max_trials': [50, 100, 200]
        }
    },

    # ========== Tree-based Models ==========
    'Decision Tree': {
        'model': DecisionTreeRegressor(random_state=42),
        'params': {
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            'max_depth': [None, 5, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
    },

    'Random Forest': {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
    },

    'Extra Trees': {
        'model': ExtraTreesRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
            'max_features': ['sqrt', 'log2', None]
        }
    },

    # ========== Boosting Models ==========
    'Gradient Boosting': {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0],
            'min_samples_split': [2, 5]
        }
    },

    'XGBoost': {
        'model': XGBRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 6, 9],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'gamma': [0, 0.1, 0.2],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [0, 0.1, 1]
        }
    },

    'LightGBM': {
        'model': LGBMRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [31, 63, 127],
            'max_depth': [-1, 10, 20],
            'min_child_samples': [20, 50],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [0, 0.1, 1]
        }
    },

    'CatBoost': {
        'model': CatBoostRegressor(random_state=42, verbose=0),
        'params': {
            'iterations': [100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'depth': [4, 6, 8],
            'l2_leaf_reg': [1, 3, 5]
        }
    },

    'AdaBoost': {
        'model': AdaBoostRegressor(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1.0],
            'loss': ['linear', 'square', 'exponential']
        }
    },

    'Hist Gradient Boosting': {
        'model': HistGradientBoostingRegressor(random_state=42),
        'params': {
            'learning_rate': [0.01, 0.1, 0.2],
            'max_iter': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_leaf': [20, 50, 100]
        }
    },

    # ========== SVM Models ==========
    'SVR': {
        'model': SVR(),
        'params': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.1, 1],
            'degree': [2, 3, 4],
            'epsilon': [0.01, 0.1, 0.5]
        }
    },

    'Linear SVR': {
        'model': LinearSVR(random_state=42),
        'params': {
            'C': [0.1, 1, 10],
            'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
            'dual': [True, False],
            'epsilon': [0.01, 0.1, 0.5]
        }
    },

    'NuSVR': {
        'model': NuSVR(),
        'params': {
            'nu': [0.1, 0.5, 0.8],
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.1, 1]
        }
    },

    # ========== Nearest Neighbors ==========
    'KNN Regressor': {
        'model': KNeighborsRegressor(),
        'params': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'p': [1, 2]
        }
    },

    'Radius Neighbors Regressor': {
        'model': RadiusNeighborsRegressor(),
        'params': {
            'radius': [1.0, 2.0, 5.0],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
    },

    # ========== Neural Networks ==========
    'MLP Regressor': {
        'model': MLPRegressor(random_state=42, early_stopping=True),
        'params': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['relu', 'tanh', 'logistic'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'learning_rate_init': [0.001, 0.01]
        }
    },

    # ========== Gaussian Processes ==========
    'Gaussian Process': {
        'model': GaussianProcessRegressor(random_state=42),
        'params': {
            'kernel': [None,
                      RBF(),
                      ConstantKernel() * RBF(),
                      RationalQuadratic()],
            'alpha': [1e-10, 1e-5, 1e-2],
            'normalize_y': [True, False]
        }
    },

    # ========== Other Regressors ==========
    'Kernel Ridge': {
        'model': KernelRidge(),
        'params': {
            'alpha': [0.1, 1.0, 10.0],
            'kernel': ['linear', 'rbf', 'polynomial'],
            'gamma': [None, 0.1, 1.0],
            'degree': [2, 3, 4]
        }
    },

    'PLS Regression': {
        'model': PLSRegression(),
        'params': {
            'n_components': [1, 2, 3, 5],
            'scale': [True, False],
            'max_iter': [500, 1000]
        }
    },

    'Dummy Regressor': {
        'model': DummyRegressor(),
        'params': {
            'strategy': ['mean', 'median', 'quantile', 'constant']
        }
    }
}


# Evaluation Function

def evaluate_regression_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'explained_variance': explained_variance_score(y_test, y_pred),
        'max_error': max_error(y_test, y_pred),
        'mape': mean_absolute_percentage_error(y_test, y_pred)
    }

    print("\nRegression Metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    return metrics


# Model Training and Evaluation

results = {}
best_models = {}
cv = KFold(n_splits=5, shuffle=True, random_state=42)

for name, config in models.items():
    print(f"\n{'=' * 50}")
    print(f"Training and tuning {name}")
    print(f"{'=' * 50}")

    try:
        # Create pipeline with preprocessing and model
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', config['model'])
        ])

        # Update params to include pipeline prefix
        params = {f'model__{key}': value for key, value in config['params'].items()}

        grid_search = GridSearchCV(
            pipeline,
            param_grid=params,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(x_train, y_train)

        best_models[name] = grid_search.best_estimator_
        results[name] = evaluate_regression_model(best_models[name], x_test, y_test)

        print(f"\nBest parameters for {name}:")
        print(grid_search.best_params_)

    except Exception as e:
        print(f"Error with {name}: {str(e)}")
        results[name] = {'error': str(e)}


# Model Comparison


print("\nModel Comparison:")
comparison = pd.DataFrame.from_dict(results, orient='index')
# Remove models that failed
comparison = comparison[~comparison.index.isin([k for k, v in results.items() if 'error' in v])]
print(comparison.sort_values(by='r2', ascending=False))


# Ensemble Methods


print("\nBuilding Ensemble Models...")

# Select top models for ensemble
top_models = comparison.nlargest(5, 'r2').index.tolist()

# Create a list of base estimators for stacking
estimators = [(name, best_models[name].named_steps['model']) for name in top_models]

# Stacking Regressor
stacking_reg = Pipeline([
    ('preprocessor', preprocessor),
    ('model', StackingRegressor(
        estimators=estimators,
        final_estimator=LinearRegression(),
        cv=5,
        n_jobs=-1
    ))
])

stacking_reg.fit(x_train, y_train)
print("\nStacking Regressor Performance:")
stacking_metrics = evaluate_regression_model(stacking_reg, x_test, y_test)
results['Stacking'] = stacking_metrics

# Voting Regressor
voting_reg = Pipeline([
    ('preprocessor', preprocessor),
    ('model', VotingRegressor(
        estimators=estimators,
        n_jobs=-1
    ))
])

voting_reg.fit(x_train, y_train)
print("\nVoting Regressor Performance:")
voting_metrics = evaluate_regression_model(voting_reg, x_test, y_test)
results['Voting'] = voting_metrics


# Final Comparison

print("\nFinal Model Comparison:")
final_comparison = pd.DataFrame.from_dict(results, orient='index')
final_comparison = final_comparison[~final_comparison.index.isin([k for k, v in results.items() if 'error' in v])]
print(final_comparison.sort_values(by='r2', ascending=False))

# Save Best Model

best_model_name = final_comparison['r2'].idxmax()
best_model = best_models.get(best_model_name,
                             stacking_reg if best_model_name == 'Stacking' else voting_reg)

print(f"\nBest model is: {best_model_name}")

# Save the best model
from joblib import dump

dump(best_model, 'best_regressor_model.joblib')

# Save all results to CSV
final_comparison.sort_values(by='r2', ascending=False).to_csv('regression_model_comparison.csv')

print("\nModel training and evaluation complete!")