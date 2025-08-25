# Design:
# - Load artifacts at startup for performance and stable schema enforcement.
# - Align inputs to model_features.json to prevent schema drift.
# - Provide both full-schema and minimal-schema endpoints to fit client workflows.
# - Enrich demographics on the server to reduce client complexity and errors
#   (clients send zipcode; we join demographics here).
import json
import os
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from typing import Sequence

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.responses import JSONResponse
try:
	import orjson
	from fastapi.responses import ORJSONResponse  # type: ignore
	DefaultResponseClass = ORJSONResponse
except Exception:
	DefaultResponseClass = JSONResponse
from pydantic import BaseModel, Field, ConfigDict
import pickle

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent


class Settings(BaseModel):
	data_dir: Path = Path(os.getenv("DATA_DIR", str(ROOT_DIR / "data")))
	model_dir: Path = Path(os.getenv("MODEL_DIR", str(ROOT_DIR / "model")))
	input_extra_policy: str = os.getenv("INPUT_EXTRA_POLICY", "allow").lower()
	api_keys: List[str] = [k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()]
	rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "0"))
	model_version: Optional[str] = os.getenv("MODEL_VERSION")


SETTINGS = Settings()
DATA_DIR = SETTINGS.data_dir
MODEL_DIR = SETTINGS.model_dir

app = FastAPI(
	title="Real Estate Price Prediction API",
	version="1.0.0",
	default_response_class=DefaultResponseClass,
)

MODEL: Any = None
MODEL_FEATURES: List[str] = []
DEMOGRAPHICS: Optional[pd.DataFrame] = None
METRICS: Optional[Dict[str, Any]] = None
MODEL_VERSION: Optional[str] = SETTINGS.model_version

# Type alias for clarity: acceptable containers for feature name lists
FeatureNames = Sequence[str]

# Input policy for handling extra fields on full-schema requests: allow|ignore|forbid
INPUT_EXTRA_POLICY = SETTINGS.input_extra_policy
if INPUT_EXTRA_POLICY not in {"allow", "ignore", "forbid"}:
	INPUT_EXTRA_POLICY = "allow"


class SaleMinimal(BaseModel):
	"""Minimal fields the model needs from the sales file.

	The service will enrich these with demographics and align to
	model_features.json before prediction.
	"""
	bedrooms: int
	bathrooms: float
	sqft_living: int
	sqft_lot: int
	floors: float
	sqft_above: int
	sqft_basement: int
	zipcode: str = Field(..., pattern=r"^\d{5}$")


class _SaleFullBase(BaseModel):
	"""Base full-schema input matching future_unseen_examples.csv."""
	bedrooms: int
	bathrooms: float
	sqft_living: int
	sqft_lot: int
	floors: float
	waterfront: int
	view: int
	condition: int
	grade: int
	sqft_above: int
	sqft_basement: int
	yr_built: int
	yr_renovated: int
	zipcode: str = Field(..., pattern=r"^\d{5}$")
	lat: float
	long: float
	sqft_living15: int
	sqft_lot15: int


class SaleFullAllow(_SaleFullBase):
	model_config = ConfigDict(extra='allow')


class SaleFullIgnore(_SaleFullBase):
	model_config = ConfigDict(extra='ignore')


class SaleFullForbid(_SaleFullBase):
	model_config = ConfigDict(extra='forbid')


# Alias selected by env for endpoint typing/validation
SaleFull = (
	SaleFullAllow if INPUT_EXTRA_POLICY == "allow" else
	SaleFullIgnore if INPUT_EXTRA_POLICY == "ignore" else
	SaleFullForbid
)


class PredictResponse(BaseModel):
	"""Lean prediction payload."""
	prediction: float


class BatchPredictResponse(BaseModel):
	"""Batch container for predictions."""
	predictions: List[PredictResponse]


def _load_pickle(path: Path):
	with open(path, "rb") as f:
		return pickle.load(f)


def load_artifacts(model_dir: Path, data_dir: Path) -> Tuple[Any, List[str], pd.DataFrame]:
	"""Load model, feature list, and demographics table from disk."""
	model_path = model_dir / "model.pkl"
	features_path = model_dir / "model_features.json"
	demographics_path = data_dir / "zipcode_demographics.csv"

	if not model_path.exists() or not features_path.exists():
		raise FileNotFoundError(
			f"Model artifacts not found in {model_dir}. Run 'python create_model.py' first.")

	model = _load_pickle(model_path)
	model_features: List[str] = json.load(open(features_path))

	demographics = pd.read_csv(demographics_path, dtype={"zipcode": str})
	demographics = demographics.set_index("zipcode", drop=True)

	return model, model_features, demographics


@app.on_event("startup")
def startup_event() -> None:
	"""Load heavy artifacts once at process start to minimize per-request latency."""
	global MODEL, MODEL_FEATURES, DEMOGRAPHICS, METRICS, MODEL_VERSION
	MODEL, MODEL_FEATURES, DEMOGRAPHICS = load_artifacts(MODEL_DIR, DATA_DIR)
	metrics_path = MODEL_DIR / "metrics.json"
	if metrics_path.exists():
		try:
			METRICS = json.load(open(metrics_path))
			if MODEL_VERSION is None:
				MODEL_VERSION = (METRICS or {}).get("model_version")
		except Exception:
			METRICS = None


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
	"""Liveness probe with lightweight model metadata for diagnostics."""
	return {
		"status": "ok",
		"num_features": len(MODEL_FEATURES) if MODEL_FEATURES else 0,
		"algorithm": (METRICS or {}).get("algorithm"),
		"model_version": MODEL_VERSION,
	}


@app.get("/readyz")
def readyz() -> Dict[str, Any]:
	"""Readiness probe indicating whether artifacts are loaded and usable."""
	return {
		"model_loaded": MODEL is not None,
		"features_loaded": bool(MODEL_FEATURES),
		"demographics_loaded": DEMOGRAPHICS is not None,
	}


@app.get("/metrics")
def metrics() -> Dict[str, Any]:
	"""Return model/service metrics and metadata without doing inference."""
	return {
		"metrics": METRICS or {},
		"feature_count": len(MODEL_FEATURES) if MODEL_FEATURES else 0,
		"algorithm": (METRICS or {}).get("algorithm"),
		"model_version": MODEL_VERSION,
	}


def _ensure_required_fields(df: pd.DataFrame, required: FeatureNames) -> None:
	"""Validate required fields are present; raise 422 with details if not."""
	missing = [c for c in required if c not in df.columns]
	if missing:
		raise HTTPException(status_code=422, detail={
			"error": "Missing required fields",
			"missing": missing
		})


def _enrich_with_demographics(df: pd.DataFrame, demographics: pd.DataFrame) -> pd.DataFrame:
	"""Join zipcode demographics and drop the zipcode column post-join.

	Raises 422 if zipcode is missing or unknown.
	"""
	if "zipcode" not in df.columns:
		raise HTTPException(status_code=422, detail="Field 'zipcode' is required for enrichment")
	df = df.copy()
	df["zipcode"] = df["zipcode"].astype(str)
	merged = df.merge(demographics.reset_index(), how="left", on="zipcode")
	# Only consider missing values in demographic columns for zipcode validation
	dem_cols = [c for c in demographics.reset_index().columns if c != "zipcode"]
	if merged[dem_cols].isna().any(axis=1).any():
		unknown = merged.loc[merged[dem_cols].isna().any(axis=1), "zipcode"].astype(str).unique().tolist()
		raise HTTPException(status_code=422, detail={
			"error": "Unknown zipcode(s) for demographics enrichment",
			"zipcodes": unknown
		})
	merged = merged.drop(columns=["zipcode"])  # zipcode not used by model after join
	return merged


def _align_to_model_features(df: pd.DataFrame, model_features: FeatureNames) -> pd.DataFrame:
	"""Filter and order columns to exactly match the training schema.

	Any extra columns provided by clients are dropped here, and missing columns
	are reported in a 422 error.
	"""
	missing = [c for c in model_features if c not in df.columns]
	if missing:
		raise HTTPException(status_code=422, detail={
			"error": "Missing features after enrichment",
			"missing": missing
		})
	# Keep only model features and order columns
	return df[model_features]


def _predict_dataframe(df: pd.DataFrame) -> List[float]:
	"""Run model inference and cast to JSON-serializable floats."""
	preds = MODEL.predict(df)
	return [float(x) for x in preds]


# --- Auth and simple rate limiting (optional; enabled by env) ---

def require_api_key(x_api_key: Optional[str] = Header(None)) -> None:
	if not SETTINGS.api_keys:
		return
	if x_api_key is None or x_api_key not in SETTINGS.api_keys:
		raise HTTPException(status_code=401, detail="Unauthorized")


class _SimpleRateLimiter:
	def __init__(self, per_minute: int) -> None:
		self.per_minute = max(0, per_minute)
		self._counts: Dict[str, Tuple[float, int]] = {}
		self._lock = threading.Lock()

	def allow(self, key: str) -> bool:
		if self.per_minute <= 0:
			return True
		now = time.monotonic()
		window_start = int(now // 60)
		with self._lock:
			prev = self._counts.get(key)
			if prev is None or prev[0] != window_start:
				self._counts[key] = (window_start, 1)
				return True
			if prev[1] >= self.per_minute:
				return False
			self._counts[key] = (window_start, prev[1] + 1)
			return True


_RATE_LIMITER = _SimpleRateLimiter(SETTINGS.rate_limit_per_minute)


def rate_limit_dependency(request: Request, x_api_key: Optional[str] = Header(None)) -> None:
	identity = x_api_key or (request.client.host if request.client else "unknown")
	if not _RATE_LIMITER.allow(identity):
		raise HTTPException(status_code=429, detail="Rate limit exceeded")


@app.post("/predict", response_model=BatchPredictResponse)
async def predict(payload: Union[SaleFull, List[SaleFull]], _: None = Depends(require_api_key), __: None = Depends(rate_limit_dependency)) -> BatchPredictResponse:
	"""Predict from full-schema payload(s)."""
	records: List[Dict[str, Any]] = (
		[p.model_dump() for p in payload] if isinstance(payload, list) else [payload.model_dump()]
	)
	df = pd.DataFrame.from_records(records)

	_ensure_required_fields(df, required=["zipcode"])  # at minimum need zipcode to enrich
	enriched = _enrich_with_demographics(df, DEMOGRAPHICS)
	aligned = _align_to_model_features(enriched, MODEL_FEATURES) # pre-existing model_features.json

	preds = _predict_dataframe(aligned)
	responses = [PredictResponse(prediction=p) for p in preds]
	return BatchPredictResponse(predictions=responses)


@app.post("/predict_minimal", response_model=BatchPredictResponse)
async def predict_minimal(payload: Union[SaleMinimal, List[SaleMinimal]], _: None = Depends(require_api_key), __: None = Depends(rate_limit_dependency)) -> BatchPredictResponse:
	"""Predict from a minimal set of sales fields; server fills demographics."""
	records: List[Dict[str, Any]] = (
		[p.model_dump() for p in payload] if isinstance(payload, list) else [payload.model_dump()]
	)
	df = pd.DataFrame.from_records(records)

	# Derive minimal required field names from the Pydantic model
	minimal_fields = list(SaleMinimal.model_fields.keys())
	_ensure_required_fields(df, required=minimal_fields)

	df_sale_only = df[minimal_fields] 
	enriched = _enrich_with_demographics(df_sale_only, DEMOGRAPHICS)
	aligned = _align_to_model_features(enriched, MODEL_FEATURES) # pre-existing model_features.json

	preds = _predict_dataframe(aligned) 
	responses = [PredictResponse(prediction=p) for p in preds] 
	return BatchPredictResponse(predictions=responses) 


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
	"""Return a generic 500 error; avoid leaking internals to clients."""
	return JSONResponse(status_code=500, content={"error": "Internal Server Error"})
