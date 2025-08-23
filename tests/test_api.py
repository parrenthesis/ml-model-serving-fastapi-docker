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


def main() -> None:
	examples = load_examples(CSV_PATH, limit=BATCH_SIZE)
	resp = requests.post(API_URL, json=examples)
	resp.raise_for_status()
	print(json.dumps(resp.json(), indent=2))


if __name__ == "__main__":
	main()
