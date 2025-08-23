# Training uses a subset of sales columns + zipcode-joined demographics.
# Extra raw sales columns (e.g., waterfront, view) are ignored by design but
# may be leveraged in future model versions without changing client contracts.
import json
import pathlib
import pickle
import os
import argparse
from typing import List
from typing import Tuple

import pandas
from sklearn import model_selection
from sklearn import neighbors
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import metrics

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


def build_model(algorithm: str):
	"""Create a regression model pipeline based on the chosen algorithm.

	Uses a RobustScaler to reduce the impact of outliers. Supports:
	- knn (default)
	- xgboost (requires the optional xgboost dependency)
	"""
	alg = algorithm.lower()
	if alg == "knn":
		return pipeline.make_pipeline(preprocessing.RobustScaler(),
									 neighbors.KNeighborsRegressor())
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
			n_jobs=0
		)
		# For tree models we typically do not scale features
		return pipeline.make_pipeline(model)
	else:
		raise ValueError(f"Unsupported algorithm: {algorithm}")


def main():
	"""Load data, train model, evaluate, and export artifacts.

	Algorithm can be selected via CLI flag '--algo' or env var MODEL_ALGO.
	Defaults to 'knn'. Example: `python create_model.py --algo xgboost`.
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument("--algo", default=os.getenv("MODEL_ALGO", "knn"),
					help="Algorithm to use: knn (default) or xgboost")
	args = parser.parse_args()
	algorithm = args.algo

	x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
	x_train, x_test, y_train, y_test = model_selection.train_test_split(
		x, y, random_state=42)

	model = build_model(algorithm).fit(x_train, y_train)

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
	json.dump(
		{
			"algorithm": algorithm,
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
