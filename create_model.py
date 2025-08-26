# Training uses a subset of sales columns + zipcode-joined demographics.
# Extra raw sales columns (e.g., waterfront, view) are ignored by design but
# may be leveraged in future model versions without changing client contracts.
import json
import pathlib
import pickle
import os
import argparse
from datetime import datetime
import subprocess
from typing import List
from typing import Tuple

# Limit BLAS/numexpr threads early (before importing numpy/pandas)
DEFAULT_MAX_WORKERS = int(os.getenv("MAX_WORKERS", "2"))
os.environ.setdefault("OMP_NUM_THREADS", str(DEFAULT_MAX_WORKERS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(DEFAULT_MAX_WORKERS))
os.environ.setdefault("MKL_NUM_THREADS", str(DEFAULT_MAX_WORKERS))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(DEFAULT_MAX_WORKERS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(DEFAULT_MAX_WORKERS))

import pandas
from sklearn import model_selection
from sklearn import neighbors
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold

SALES_PATH = "data/kc_house_data.csv"  # path to CSV with home sale data
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"  # path to CSV with demographics
# List of columns (subset) that will be taken from home sale data
SALES_COLUMN_SELECTION = [
	'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
	'sqft_above', 'sqft_basement', 'zipcode'
]
OUTPUT_DIR = "model"  # Directory where output artifacts will be saved


def load_data(
	sales_path: str, demographics_path: str, sales_column_selection: List[str]
) -> Tuple[pandas.DataFrame, pandas.Series]:
	"""Load the target and feature data by merging sales and demographics.

	Args:
		sales_path: path to CSV file with home sale data
		demographics_path: path to CSV file with home sale data
		sales_column_selection: list of columns from sales data to be used as
			features

	Returns:
		Tuple containg with two elements: a DataFrame and a Series of the same
		length.  The DataFrame contains features for machine learning, the
		series contains the target variable (home sale price).

	"""
	data = pandas.read_csv(sales_path,
						   usecols=sales_column_selection,
						   dtype={'zipcode': str})
	demographics = pandas.read_csv(demographics_path,
								   dtype={'zipcode': str})

	merged_data = data.merge(demographics, how="left",
							 on="zipcode").drop(columns="zipcode")
	# Remove the target variable from the dataframe, features will remain
	y = merged_data.pop('price')
	x = merged_data

	return x, y


def build_model(algorithm: str, max_workers: int):
	"""Create a regression model pipeline based on the chosen algorithm.

	Uses a RobustScaler to reduce the impact of outliers. Supports:
	- knn (default)
	- xgboost (requires the optional xgboost dependency)
	"""
	alg = algorithm.lower()
	if alg == "knn":
		return pipeline.make_pipeline(
			preprocessing.RobustScaler(),
			neighbors.KNeighborsRegressor(n_jobs=max_workers),
		)
	elif alg == "xgboost":
		try:
			# Lazy import so xgboost remains optional
			from xgboost import XGBRegressor  # type: ignore
		except Exception as e:
			raise ImportError(
				"xgboost is not installed. Install with 'pip install xgboost' or use requirements-xgb.txt"
			) from e
		# Reasonable defaults; tree method auto on CPU; scale_pos_weight not needed
		model = XGBRegressor(
			n_estimators=500,
			max_depth=8,
			learning_rate=0.05,
			subsample=0.9,
			colsample_bytree=0.9,
			early_stopping_rounds=50,
			random_state=42,
			n_jobs=max_workers
		)
		# For tree models we typically do not scale features
		return pipeline.make_pipeline(model)
	else:
		raise ValueError(f"Unsupported algorithm: {algorithm}")


def tune_knn(x, y, cv_folds: int, max_workers: int):
	"""Grid-search KNN hyperparameters with CV; returns best estimator and info."""
	scoring = "neg_root_mean_squared_error"
	cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
	pipe = pipeline.make_pipeline(
		preprocessing.RobustScaler(),
		neighbors.KNeighborsRegressor(n_jobs=max_workers),
	)
	param_grid = {
		"kneighborsregressor__n_neighbors": [3, 5, 7, 9, 11, 15, 21, 31],
		"kneighborsregressor__weights": ["uniform", "distance"],
		"kneighborsregressor__p": [1, 2],
		"kneighborsregressor__leaf_size": [15, 30, 60],
	}
	gs = GridSearchCV(
		estimator=pipe,
		param_grid=param_grid,
		cv=cv,
		scoring=scoring,
		n_jobs=max_workers,
		refit=True,
	)
	gs.fit(x, y)
	return gs.best_estimator_, gs.best_params_, float(gs.best_score_)


def tune_xgb(x, y, cv_folds: int, n_iter: int, max_workers: int):
	"""Randomized-search XGBoost hyperparameters with CV; returns best and info."""
	from xgboost import XGBRegressor  # type: ignore
	scoring = "neg_root_mean_squared_error"
	cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
	pipe = pipeline.make_pipeline(
		XGBRegressor(
			objective="reg:squarederror",
			n_estimators=600,
			random_state=42,
			n_jobs=max_workers,
		)
	)
	param_dist = {
		"xgbregressor__n_estimators": [400, 600, 800, 1000],
		"xgbregressor__max_depth": [4, 6, 8, 10],
		"xgbregressor__learning_rate": [0.03, 0.05, 0.07, 0.1],
		"xgbregressor__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
		"xgbregressor__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
		"xgbregressor__min_child_weight": [1, 3, 5, 7],
		"xgbregressor__reg_lambda": [0, 1, 5, 10],
		"xgbregressor__reg_alpha": [0, 0.1, 0.5, 1.0],
	}
	rs = RandomizedSearchCV(
		estimator=pipe,
		param_distributions=param_dist,
		n_iter=n_iter,
		cv=cv,
		scoring=scoring,
		n_jobs=max_workers,
		random_state=42,
		refit=True,
	)
	rs.fit(x, y)
	return rs.best_estimator_, rs.best_params_, float(rs.best_score_)


def main():
	"""Load data, train model, evaluate, and export artifacts.

	Algorithm can be selected via CLI flag '--algo' or env var MODEL_ALGO.
	Defaults to 'knn'. Example: `python create_model.py --algo xgboost`.
	Use --tune to run CV: `--tune knn` or `--tune xgb`; control folds and
	iterations via --cv-folds and --n-iter.
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument("--algo", default=os.getenv("MODEL_ALGO", "knn"),
				help="Algorithm to use: knn (default) or xgboost")
	parser.add_argument("--tune", choices=["knn", "xgb"], default=None,
				help="Enable CV tuning for knn or xgb")
	parser.add_argument("--cv-folds", type=int, default=5,
				help="Number of CV folds (default 5)")
	parser.add_argument("--n-iter", type=int, default=30,
				help="Randomized search iterations for xgb (default 30)")
	parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
				help="Max parallel workers (threads/processes). Default from MAX_WORKERS env or 2.")
	args = parser.parse_args()
	algorithm = args.algo
	max_workers = max(1, int(args.max_workers))

	x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
	x_train, x_test, y_train, y_test = model_selection.train_test_split(
		x, y, random_state=42)

	best_params = None
	cv_score = None

	if args.tune == "knn":
		model, best_params, cv_score = tune_knn(x_train, y_train, args.cv_folds, max_workers)
		used_algorithm = f"{algorithm}-tuned"
	elif args.tune == "xgb":
		model, best_params, cv_score = tune_xgb(x_train, y_train, args.cv_folds, args.n_iter, max_workers)
		# Refit best with early stopping on held-out split
		model = model.fit(
			x_train,
			y_train,
			**{"xgbregressor__eval_set": [(x_test, y_test)], "xgbregressor__verbose": False},
		)
		used_algorithm = f"{algorithm}-tuned"
	else:
		model = build_model(algorithm, max_workers)
		# Fit model (pass eval_set for xgboost to enable early stopping)
		if algorithm.lower() == "xgboost":
			model = model.fit(
				x_train,
				y_train,
				**{"xgbregressor__eval_set": [(x_test, y_test)], "xgbregressor__verbose": False},
			)
		else:
			model = model.fit(x_train, y_train)
		used_algorithm = algorithm

	# Evaluate model performance on train and test splits
	y_pred_train = model.predict(x_train)
	y_pred_test = model.predict(x_test)
	r2_train = float(metrics.r2_score(y_train, y_pred_train))
	r2_test = float(metrics.r2_score(y_test, y_pred_test))
	rmse_train = float(metrics.mean_squared_error(y_train, y_pred_train, squared=False))
	rmse_test = float(metrics.mean_squared_error(y_test, y_pred_test, squared=False))
	mae_train = float(metrics.mean_absolute_error(y_train, y_pred_train))
	mae_test = float(metrics.mean_absolute_error(y_test, y_pred_test))
	medae_train = float(metrics.median_absolute_error(y_train, y_pred_train))
	medae_test = float(metrics.median_absolute_error(y_test, y_pred_test))

	output_dir = pathlib.Path(OUTPUT_DIR)
	output_dir.mkdir(exist_ok=True)

	# Output model artifacts: pickled model and JSON list of features
	pickle.dump(model, open(output_dir / "model.pkl", 'wb'))
	json.dump(list(x_train.columns),
			  open(output_dir / "model_features.json", 'w'))

	# Save simple metrics for reference
	def _git_sha_short() -> str:
		try:
			return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
		except Exception:
			return "nogit"

	model_version = os.getenv("MODEL_VERSION") or f"{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{_git_sha_short()}"

	json.dump(
		{
			"model_version": model_version,
			"algorithm": used_algorithm,
			"best_params": best_params,
			"cv_score_neg_rmse": cv_score,
			"r2_train": r2_train,
			"r2_test": r2_test,
			"rmse_train": rmse_train,
			"rmse_test": rmse_test,
			"mae_train": mae_train,
			"mae_test": mae_test,
			"medae_train": medae_train,
			"medae_test": medae_test,
			"n_train": int(len(y_train)),
			"n_test": int(len(y_test))
		},
		open(output_dir / "metrics.json", 'w')
	)


if __name__ == "__main__":
	main()
