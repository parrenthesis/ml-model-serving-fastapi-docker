import csv
import json
import os
from typing import List, Dict

import requests

API_URL = os.getenv("API_URL", "http://localhost:8000/predict")
CSV_PATH = os.getenv("CSV_PATH", "data/future_unseen_examples.csv")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))


def load_examples(path: str, limit: int = 0) -> List[Dict[str, str]]:
	rows: List[Dict[str, str]] = []
	with open(path, newline="") as f:
		reader = csv.DictReader(f)
		for i, row in enumerate(reader):
			rows.append(row)
			if limit and len(rows) >= limit:
				break
	return rows


def test_prediction_response_structure(response_data: Dict) -> None:
	"""Test that prediction response has the expected structure."""
	assert "predictions" in response_data, "Response missing 'predictions' field"
	assert isinstance(response_data["predictions"], list), "Predictions should be a list"
	
	for i, prediction in enumerate(response_data["predictions"]):
		assert "prediction" in prediction, f"Prediction {i} missing 'prediction' field"
		assert isinstance(prediction["prediction"], (int, float)), f"Prediction {i} should be numeric"
		assert prediction["prediction"] > 0, f"Prediction {i} should be positive"
		
		# Check for confidence interval fields (may be None if not available)
		assert "confidence_interval" in prediction, f"Prediction {i} missing 'confidence_interval' field"
		assert "confidence_score" in prediction, f"Prediction {i} missing 'confidence_score' field"
		assert "confidence_type" in prediction, f"Prediction {i} missing 'confidence_type' field"
		
		# If confidence interval exists, validate its structure
		if prediction["confidence_interval"] is not None:
			assert "lower" in prediction["confidence_interval"], f"Confidence interval {i} missing 'lower'"
			assert "upper" in prediction["confidence_interval"], f"Confidence interval {i} missing 'upper'"
			assert prediction["confidence_interval"]["lower"] <= prediction["confidence_interval"]["upper"], \
				f"Confidence interval {i} has invalid bounds"


def main() -> None:
	examples = load_examples(CSV_PATH, limit=BATCH_SIZE)
	
	if not examples:
		print(f"No examples found in {CSV_PATH}")
		return
	
	print(f"Testing with {len(examples)} examples from {CSV_PATH}")
	
	try:
		resp = requests.post(API_URL, json=examples, timeout=30)
		resp.raise_for_status()
		
		response_data = resp.json()
		print("API request successful")
		
		# Test response structure
		test_prediction_response_structure(response_data)
		print("Response structure validation passed")
		
		# Print results
		print(f"\nPredictions with confidence intervals:")
		print(json.dumps(response_data, indent=2))
		
		# Summary statistics
		predictions = [p["prediction"] for p in response_data["predictions"]]
		confidence_scores = [p.get("confidence_score", 0) for p in response_data["predictions"] if p.get("confidence_score") is not None]
		
		print(f"\nSummary:")
		print(f"  - Number of predictions: {len(predictions)}")
		print(f"  - Average prediction: ${sum(predictions)/len(predictions):,.0f}")
		print(f"  - Min prediction: ${min(predictions):,.0f}")
		print(f"  - Max prediction: ${max(predictions):,.0f}")
		
		if confidence_scores:
			print(f"  - Average confidence score: {sum(confidence_scores)/len(confidence_scores):.3f}")
			print(f"  - Min confidence score: {min(confidence_scores):.3f}")
			print(f"  - Max confidence score: {max(confidence_scores):.3f}")
		
	except requests.exceptions.ConnectionError:
		print(f"Could not connect to API at {API_URL}")
		print("Make sure the API is running with: make api")
		return
	except requests.exceptions.Timeout:
		print("API request timed out")
		return
	except requests.exceptions.HTTPError as e:
		print(f"API request failed with status {e.response.status_code}")
		try:
			error_data = e.response.json()
			print(f"Error details: {json.dumps(error_data, indent=2)}")
		except:
			print(f"Error response: {e.response.text}")
		return
	except Exception as e:
		print(f"Unexpected error: {e}")
		return


if __name__ == "__main__":
	main()
