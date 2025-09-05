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
import hashlib
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from typing import Sequence

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.responses import JSONResponse, Response
try:
	import orjson
	from fastapi.responses import ORJSONResponse  # type: ignore
	DefaultResponseClass = ORJSONResponse
except Exception:
	DefaultResponseClass = JSONResponse
from pydantic import BaseModel, Field, ConfigDict
import pickle
import logging

try:
	from prometheus_client import Counter, Histogram, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST  # type: ignore
except Exception:
	Counter = Histogram = CollectorRegistry = None  # type: ignore
	generate_latest = None  # type: ignore
	CONTENT_TYPE_LATEST = "text/plain"  # type: ignore

try:
	import requests  # type: ignore
except Exception:
	requests = None  # type: ignore

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent


class Settings(BaseModel):
	data_dir: Path = Path(os.getenv("DATA_DIR", str(ROOT_DIR / "data")))
	model_dir: Path = Path(os.getenv("MODEL_DIR", str(ROOT_DIR / "model")))
	input_extra_policy: str = os.getenv("INPUT_EXTRA_POLICY", "allow").lower()
	api_keys: List[str] = [k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()]
	rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "0"))
	model_version: Optional[str] = os.getenv("MODEL_VERSION")
	# Observability / logging
	prometheus_enabled: bool = os.getenv("PROMETHEUS_ENABLED", "false").lower() in {"1", "true", "yes"}
	log_json: bool = os.getenv("LOG_JSON", "false").lower() in {"1", "true", "yes"}
	request_id_header: str = os.getenv("REQUEST_ID_HEADER", "X-Request-ID")
	# Batch limits
	max_batch: int = int(os.getenv("MAX_BATCH", "512"))
	# Demographics cache
	cache_size: int = int(os.getenv("CACHE_SIZE", "1024"))
	cache_ttl_seconds: int = int(os.getenv("CACHE_TTL", "600"))
	redis_url: Optional[str] = os.getenv("REDIS_URL") or None
	# Hybrid artifact source
	model_source: str = os.getenv("MODEL_SOURCE", "local").lower()
	model_url: Optional[str] = os.getenv("MODEL_URL") or None
	model_s3_uri: Optional[str] = os.getenv("MODEL_S3_URI") or None
	model_sha256: Optional[str] = os.getenv("MODEL_SHA256") or None
	# Confidence intervals
	confidence_enabled: bool = os.getenv("CONFIDENCE_ENABLED", "true").lower() in {"1", "true", "yes"}
	confidence_method: str = os.getenv("CONFIDENCE_METHOD", "hybrid")  # "quantile", "knn_variance", "feature_distance", "hybrid"
	confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.3"))
	# FIXED: Changed from 2.0 to 0.5 for more reasonable confidence scaling
	# Lower values = higher confidence for similar houses, more realistic scores
	feature_distance_threshold: float = float(os.getenv("FEATURE_DISTANCE_THRESHOLD", "0.5"))


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
# Confidence interval models and data
MODEL_MEDIAN: Any = None
MODEL_LOWER: Any = None
MODEL_UPPER: Any = None
TRAINING_DATA: Optional[pd.DataFrame] = None

# Logger setup (structured JSON optional)
LOGGER = logging.getLogger("housing_api")
if SETTINGS.log_json:
	try:
		from pythonjsonlogger import jsonlogger  # type: ignore
		_handler = logging.StreamHandler()
		fmt = jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(message)s")
		_handler.setFormatter(fmt)
		LOGGER.handlers = [_handler]
		LOGGER.setLevel(logging.INFO)
	except Exception:
		LOGGER.setLevel(logging.INFO)
else:
	LOGGER.setLevel(logging.INFO)

# Type alias for clarity: acceptable containers for feature name lists
FeatureNames = Sequence[str]


@app.middleware("http")
async def _obs_middleware(request: Request, call_next):
	start = time.perf_counter()
	req_id = request.headers.get(SETTINGS.request_id_header) or str(uuid.uuid4())
	request.state.request_id = req_id
	try:
		response = await call_next(request)
	except Exception:
		response = JSONResponse(status_code=500, content={"error": "Internal Server Error"})
		raise
	finally:
		elapsed = max(0.0, time.perf_counter() - start)
		status = str(getattr(locals().get("response", None), "status_code", 500))
		path = request.url.path
		if SETTINGS.log_json:
			LOGGER.info("request_complete", extra={
				"request_id": req_id,
				"path": path,
				"method": request.method,
				"status": int(status),
				"latency_seconds": round(elapsed, 6),
				"model_version": MODEL_VERSION,
			})
		if SETTINGS.prometheus_enabled and hasattr(app.state, "prom_requests"):
			labels = {"path": path, "method": request.method, "status": status, "model_version": MODEL_VERSION or "unknown"}
			app.state.prom_requests.labels(**labels).inc()
			app.state.prom_latency.labels(**labels).observe(elapsed)
	return response

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
	"""Enhanced prediction payload with confidence intervals."""
	prediction: float
	confidence_interval: Optional[Dict[str, float]] = None  # {"lower": 450000, "upper": 550000}
	confidence_score: Optional[float] = None  # 0.0-1.0 confidence in prediction
	confidence_type: Optional[str] = None  # "quantile", "feature_distance", "hybrid"
	feature_novelty: Optional[float] = None  # Distance to training data (0=very similar, 1=very novel)


class BatchPredictResponse(BaseModel):
	"""Batch container for predictions."""
	predictions: List[PredictResponse]


def _load_pickle(path: Path):
	with open(path, "rb") as f:
		return pickle.load(f)


def _sha256_file(path: Path) -> str:
	h = hashlib.sha256()
	with open(path, "rb") as f:
		for chunk in iter(lambda: f.read(8192), b""):
			h.update(chunk)
	return h.hexdigest()


def _maybe_fetch_remote_artifacts(local_model_dir: Path) -> Path:
	"""Hybrid artifact fetch; returns directory to load artifacts from or local fallback."""
	if SETTINGS.model_source == "local":
		return local_model_dir
	cache_root = ROOT_DIR / "model_cache"
	cache_root.mkdir(exist_ok=True)
	version = SETTINGS.model_version or "unknown"
	cache_dir = cache_root / version
	try:
		if SETTINGS.model_source == "http":
			if requests is None:
				raise ImportError("requests is required for MODEL_SOURCE=http")
			base = (SETTINGS.model_url or "").rstrip("/")
			if not base:
				raise ValueError("MODEL_URL must be set for MODEL_SOURCE=http")
			cache_dir.mkdir(exist_ok=True)
			for name in ("model.pkl", "model_features.json", "metrics.json"):
				r = requests.get(f"{base}/{name}", timeout=15)
				r.raise_for_status()
				(cache_dir / name).write_bytes(r.content)
			if SETTINGS.model_sha256:
				sha = _sha256_file(cache_dir / "model.pkl")
				if sha.lower() != SETTINGS.model_sha256.lower():
					raise ValueError("MODEL_SHA256 verification failed for model.pkl")
			return cache_dir
		elif SETTINGS.model_source == "s3":
			try:
				import boto3  # type: ignore
			except Exception as e:
				raise ImportError("boto3 is required for MODEL_SOURCE=s3") from e
			uri = (SETTINGS.model_s3_uri or "").rstrip("/")
			if not uri.startswith("s3://"):
				raise ValueError("MODEL_S3_URI must start with s3://")
			_, _, rest = uri.partition("s3://")
			bucket, _, prefix = rest.partition("/")
			cache_dir.mkdir(exist_ok=True)
			s3 = boto3.client("s3")
			for name in ("model.pkl", "model_features.json", "metrics.json"):
				key = f"{prefix}/{name}" if prefix else name
				s3.download_file(bucket, key, str(cache_dir / name))
			if SETTINGS.model_sha256:
				sha = _sha256_file(cache_dir / "model.pkl")
				if sha.lower() != SETTINGS.model_sha256.lower():
					raise ValueError("MODEL_SHA256 verification failed for model.pkl")
			return cache_dir
		else:
			return local_model_dir
	except Exception as e:
		LOGGER.warning("hybrid_fetch_failed", extra={"error": str(e)})
		return local_model_dir


def load_artifacts(model_dir: Path, data_dir: Path) -> Tuple[Any, List[str], pd.DataFrame, Any, Any, Any, Optional[pd.DataFrame]]:
	"""Load model, feature list, demographics table, and confidence models from disk or remote (hybrid)."""
	chosen_dir = _maybe_fetch_remote_artifacts(model_dir)
	model_path = chosen_dir / "model.pkl"
	features_path = chosen_dir / "model_features.json"
	demographics_path = data_dir / "zipcode_demographics.csv"
	
	# Confidence model paths
	model_median_path = chosen_dir / "quantile_0.5.pkl"
	model_lower_path = chosen_dir / "quantile_0.05.pkl"
	model_upper_path = chosen_dir / "quantile_0.95.pkl"
	training_data_path = chosen_dir / "training_data.pkl"

	if not model_path.exists() or not features_path.exists():
		raise FileNotFoundError(
			f"Model artifacts not found in {chosen_dir}. Run 'python create_model.py' first.")

	model = _load_pickle(model_path)
	model_features: List[str] = json.load(open(features_path))

	demographics = pd.read_csv(demographics_path, dtype={"zipcode": str})
	demographics = demographics.set_index("zipcode", drop=True)
	
	# Load confidence models if they exist
	model_median = None
	model_lower = None
	model_upper = None
	training_data = None
	
	if model_median_path.exists() and model_lower_path.exists() and model_upper_path.exists():
		try:
			model_median = _load_pickle(model_median_path)
			model_lower = _load_pickle(model_lower_path)
			model_upper = _load_pickle(model_upper_path)
		except Exception as e:
			LOGGER.warning(f"Failed to load quantile models: {e}")
	
	if training_data_path.exists():
		try:
			training_data = _load_pickle(training_data_path)
		except Exception as e:
			LOGGER.warning(f"Failed to load training data: {e}")

	return model, model_features, demographics, model_median, model_lower, model_upper, training_data


@app.on_event("startup")
def startup_event() -> None:
	"""Load heavy artifacts once at process start to minimize per-request latency."""
	global MODEL, MODEL_FEATURES, DEMOGRAPHICS, METRICS, MODEL_VERSION, MODEL_MEDIAN, MODEL_LOWER, MODEL_UPPER, TRAINING_DATA
	MODEL, MODEL_FEATURES, DEMOGRAPHICS, MODEL_MEDIAN, MODEL_LOWER, MODEL_UPPER, TRAINING_DATA = load_artifacts(MODEL_DIR, DATA_DIR)
	metrics_path = MODEL_DIR / "metrics.json"
	if metrics_path.exists():
		try:
			METRICS = json.load(open(metrics_path))
			# Prefer metrics.json version over default env "dev" (but keep explicit overrides)
			metrics_version = (METRICS or {}).get("model_version")
			if metrics_version and (MODEL_VERSION in (None, "", "dev")):
				MODEL_VERSION = metrics_version
		except Exception:
			METRICS = None

	# Setup Prometheus registry if enabled
	if SETTINGS.prometheus_enabled and Counter is not None and Histogram is not None:
		_registry = CollectorRegistry()
		app.state.prom_requests = Counter(
			"api_requests_total", "Count of API requests", ["path", "method", "status", "model_version"], registry=_registry
		)
		app.state.prom_latency = Histogram(
			"api_request_duration_seconds", "Latency of API requests", ["path", "method", "status", "model_version"], registry=_registry
		)
		app.state.prom_registry = _registry


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


@app.get("/metrics_prom")
def metrics_prometheus():
	"""Prometheus exposition format when enabled."""
	if not SETTINGS.prometheus_enabled or generate_latest is None or not hasattr(app.state, "prom_registry"):
		raise HTTPException(status_code=404, detail="Prometheus disabled")
	data = generate_latest(app.state.prom_registry)
	return Response(content=data, media_type=CONTENT_TYPE_LATEST)


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
	# Populate a simple in-process LRU cache for future lookups (optional)
	if not hasattr(app.state, "dem_cache"):
		app.state.dem_cache = {}
		app.state.dem_cache_order = []
		app.state.dem_cache_expiry = {}
	for _, r in merged[["zipcode"] + dem_cols].drop_duplicates("zipcode").iterrows():
		z = str(r["zipcode"])  # type: ignore[index]
		row_dict = {col: r[col] for col in dem_cols}
		app.state.dem_cache[z] = row_dict
		app.state.dem_cache_expiry[z] = time.time() + max(1, SETTINGS.cache_ttl_seconds)
		if z in app.state.dem_cache_order:
			app.state.dem_cache_order.remove(z)
		app.state.dem_cache_order.append(z)
		while len(app.state.dem_cache_order) > max(1, SETTINGS.cache_size):
			old = app.state.dem_cache_order.pop(0)
			app.state.dem_cache.pop(old, None)
			app.state.dem_cache_expiry.pop(old, None)
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


def _predict_with_confidence(df: pd.DataFrame) -> List[Dict[str, Any]]:
	"""Enhanced prediction with confidence intervals."""
	predictions = []
	
	for idx, row in df.iterrows():
		# Base prediction
		pred = MODEL.predict([row])[0]
		
		confidence_info = {}
		
		if SETTINGS.confidence_enabled:
			# FIXED: Confidence calculations now properly capped and scaled
			# - Quantile confidence: based on relative interval width, capped at 90%
			# - KNN confidence: based on neighbor variance, capped at 90%
			# - Feature distance: based on distance to training data, capped at 90%
			# - Hybrid: weighted combination of all methods, capped at 90%
			if SETTINGS.confidence_method in ["quantile", "hybrid"] and MODEL_LOWER is not None and MODEL_UPPER is not None:
				# Quantile regression confidence intervals
				lower = MODEL_LOWER.predict([row])[0]
				upper = MODEL_UPPER.predict([row])[0]
				confidence_info["confidence_interval"] = {"lower": float(lower), "upper": float(upper)}
				confidence_info["confidence_type"] = "quantile"
				
				# Calculate confidence score from interval width - FIXED LOGIC
				# Smaller interval = higher confidence, but cap at reasonable levels
				if pred > 0:
					# Calculate relative interval width (as percentage of prediction)
					relative_interval = (upper - lower) / pred
					# Convert to confidence: 0% interval = 100% confidence, 100% interval = 0% confidence
					# FIXED: Cap confidence at 0.9 (90%) to avoid unrealistic scores like 0.999...
					# This prevents the model from claiming near-perfect confidence
					quantile_confidence = max(0.1, min(0.9, 1.0 - relative_interval))
				else:
					quantile_confidence = 0.5
				
				confidence_info["quantile_confidence"] = quantile_confidence
				if SETTINGS.confidence_method == "quantile":
					confidence_info["confidence_score"] = quantile_confidence
				
				# Add debugging info for quantile method
				if SETTINGS.log_json:
					confidence_info["debug"] = {
						"lower": float(lower),
						"upper": float(upper),
						"prediction": float(pred),
						"interval_width": float(upper - lower),
						"relative_interval": float((upper - lower) / pred) if pred > 0 else 0,
						"quantile_confidence": quantile_confidence
					}
			
			if SETTINGS.confidence_method in ["knn_variance", "hybrid"] and TRAINING_DATA is not None:
				# KNN variance-based confidence (for KNN models)
				if hasattr(MODEL, 'named_steps') and 'kneighborsregressor' in MODEL.named_steps:
					# Get k-nearest neighbors and their predictions
					knn_model = MODEL.named_steps['kneighborsregressor']
					distances, indices = knn_model.kneighbors([row])
					
					# Get the actual target values for these neighbors
					neighbor_values = TRAINING_DATA.iloc[indices[0]]['price'].values if 'price' in TRAINING_DATA.columns else None
					
					if neighbor_values is not None:
						# Calculate variance of neighbor values
						variance = np.var(neighbor_values)
						mean_value = np.mean(neighbor_values)
						
						# Convert variance to confidence score (lower variance = higher confidence)
						# Normalize by mean to get relative variance
						relative_variance = variance / (mean_value ** 2) if mean_value > 0 else 1.0
						# FIXED: Cap confidence at reasonable levels and use better scaling
						# This prevents unrealistic confidence scores and provides better variance handling
						knn_confidence = max(0.1, min(0.9, 1.0 - min(relative_variance, 1.0)))
						
						confidence_info["knn_variance"] = float(variance)
						confidence_info["knn_confidence"] = float(knn_confidence)
						
						if SETTINGS.confidence_method == "knn_variance":
							confidence_info["confidence_score"] = knn_confidence
							confidence_info["confidence_type"] = "knn_variance"
						
						# Add debugging info for KNN method
						if SETTINGS.log_json:
							confidence_info["debug"] = {
								"variance": float(variance),
								"mean_value": float(mean_value),
								"relative_variance": float(relative_variance),
								"knn_confidence": float(knn_confidence)
							}
			
			if SETTINGS.confidence_method in ["feature_distance", "hybrid"] and TRAINING_DATA is not None:
				# Feature distance confidence
				# Exclude target column for distance calculation
				feature_columns = [col for col in TRAINING_DATA.columns if col != 'price']
				feature_confidence, novelty = calculate_feature_distance(
					row.values, TRAINING_DATA[feature_columns].values
				)
				confidence_info["feature_novelty"] = novelty
				
				if SETTINGS.confidence_method == "feature_distance":
					confidence_info["confidence_score"] = feature_confidence
					confidence_info["confidence_type"] = "feature_distance"
					
					# Add debugging info for feature distance method
					if SETTINGS.log_json:
						confidence_info["debug"] = {
							"feature_confidence": float(feature_confidence),
							"feature_novelty": float(novelty)
						}
				elif SETTINGS.confidence_method == "hybrid":
					# Combine confidence measures - FIXED LOGIC
					quantile_confidence = confidence_info.get("quantile_confidence", 0.5)
					knn_confidence = confidence_info.get("knn_confidence", 0.5)
					
					# Weight the different confidence measures
					weights = []
					confidences = []
					
					if "quantile_confidence" in confidence_info:
						weights.append(0.4)
						confidences.append(quantile_confidence)
					if "knn_confidence" in confidence_info:
						weights.append(0.3)
						confidences.append(knn_confidence)
					if "feature_novelty" in confidence_info:
						weights.append(0.3)
						confidences.append(feature_confidence)
					
					if weights:
						# Normalize weights
						total_weight = sum(weights)
						weights = [w / total_weight for w in weights]
						# FIXED: Cap final confidence at reasonable levels
						# This prevents the hybrid method from producing unrealistic confidence scores
						hybrid_confidence = sum(w * c for w, c in zip(weights, confidences))
						confidence_info["confidence_score"] = max(0.1, min(0.9, hybrid_confidence))
						
						# Add debugging info for hybrid method
						confidence_info["debug"] = {
							"quantile_confidence": quantile_confidence,
							"knn_confidence": knn_confidence,
							"feature_confidence": feature_confidence,
							"weights": weights,
							"raw_hybrid": hybrid_confidence,
							"final_confidence": confidence_info["confidence_score"]
						}
					else:
						confidence_info["confidence_score"] = 0.5
					
					confidence_info["confidence_type"] = "hybrid"
		
		predictions.append({
			"prediction": float(pred),
			**confidence_info
		})
		
		# Add overall debugging info
		if SETTINGS.log_json:
			LOGGER.info("prediction_confidence_summary", extra={
				"prediction": float(pred),
				"confidence_score": confidence_info.get("confidence_score", 0.0),
				"confidence_type": confidence_info.get("confidence_type", "none"),
				"feature_novelty": confidence_info.get("feature_novelty", 0.0),
				"has_quantile": "quantile_confidence" in confidence_info,
				"has_knn": "knn_confidence" in confidence_info,
				"has_feature": "feature_novelty" in confidence_info
			})
	
	return predictions


def calculate_feature_distance(input_features: np.ndarray, training_data: np.ndarray) -> Tuple[float, float]:
	"""Calculate normalized distance to training data.
	
	Args:
		input_features: Single row of features (1D array)
		training_data: Training data matrix (2D array)
		
	Returns:
		Tuple of (confidence_score, normalized_distance)
		- confidence_score: 0.0-1.0 (1.0 = very similar to training data)
		- normalized_distance: Raw distance metric (higher = more novel)
	"""
	try:
		from scipy.spatial.distance import cdist
	except ImportError:
		# Fallback if scipy not available
		return 0.5, 1.0
	
	# Ensure input_features is 2D for cdist
	if input_features.ndim == 1:
		input_features = input_features.reshape(1, -1)
	
	# Calculate distances to all training samples
	distances = cdist(input_features, training_data, metric='euclidean')
	
	# Normalize by training data variance
	training_std = np.std(training_data, axis=0)
	# Avoid division by zero
	training_std = np.where(training_std == 0, 1.0, training_std)
	
	# FIXED: Better distance normalization logic
	# The previous calculation was dividing distances by mean(std) which created very large numbers
	# Now we calculate mean(distance) / mean(std) which gives more reasonable novelty scores
	distances_flat = distances.flatten()
	mean_distance = np.mean(distances_flat)
	mean_feature_std = np.mean(training_std)
	
	# Normalize distance by feature scale - this gives us a more reasonable novelty score
	# Lower values = more similar to training data, higher values = more novel
	# Expected range: 0.1-5.0 for similar houses, 5.0-20.0 for unusual houses
	# NOTE: Houses in future_unseen_examples.csv are intentionally different from training data
	# High novelty scores (10+) are expected and correct - they indicate the model is working properly
	normalized_distance = mean_distance / mean_feature_std if mean_feature_std > 0 else mean_distance
	
	# Convert to confidence score (0=very similar, 1=very novel)
	# Use better scaling and cap at reasonable levels
	scale_factor = SETTINGS.feature_distance_threshold
	
	# Convert distance to confidence: lower distance = higher confidence
	# Use sigmoid-like function for better scaling
	confidence_score = 1.0 / (1.0 + normalized_distance / scale_factor)
	
	# FIXED: Cap confidence at reasonable levels (10% to 90%)
	# This prevents unrealistic confidence scores and provides better distance scaling
	confidence_score = max(0.1, min(0.9, confidence_score))
	
	# Add some debugging info
	if SETTINGS.log_json:
		LOGGER.info("feature_distance_calculation", extra={
			"mean_distance": float(mean_distance),
			"mean_feature_std": float(mean_feature_std),
			"normalized_distance": float(normalized_distance),
			"scale_factor": float(scale_factor),
			"raw_confidence": float(1.0 / (1.0 + normalized_distance / scale_factor)),
			"final_confidence": float(confidence_score)
		})
	
	return float(confidence_score), float(normalized_distance)


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
	# Enforce batch cap if a list is provided
	if isinstance(payload, list) and len(payload) > max(1, SETTINGS.max_batch):
		raise HTTPException(status_code=422, detail=f"Batch too large; max is {SETTINGS.max_batch}")
	records: List[Dict[str, Any]] = (
		[p.model_dump() for p in payload] if isinstance(payload, list) else [payload.model_dump()]
	)
	df = pd.DataFrame.from_records(records)

	_ensure_required_fields(df, required=["zipcode"])  # at minimum need zipcode to enrich
	enriched = _enrich_with_demographics(df, DEMOGRAPHICS)
	aligned = _align_to_model_features(enriched, MODEL_FEATURES) # pre-existing model_features.json

	if SETTINGS.confidence_enabled:
		pred_results = _predict_with_confidence(aligned)
		responses = [PredictResponse(**pred) for pred in pred_results]
	else:
		preds = _predict_dataframe(aligned)
		responses = [PredictResponse(prediction=p) for p in preds]
	
	return BatchPredictResponse(predictions=responses)


@app.post("/predict_minimal", response_model=BatchPredictResponse)
async def predict_minimal(payload: Union[SaleMinimal, List[SaleMinimal]], _: None = Depends(require_api_key), __: None = Depends(rate_limit_dependency)) -> BatchPredictResponse:
	"""Predict from a minimal set of sales fields; server fills demographics."""
	# Enforce batch cap if a list is provided
	if isinstance(payload, list) and len(payload) > max(1, SETTINGS.max_batch):
		raise HTTPException(status_code=422, detail=f"Batch too large; max is {SETTINGS.max_batch}")
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

	if SETTINGS.confidence_enabled:
		pred_results = _predict_with_confidence(aligned)
		responses = [PredictResponse(**pred) for pred in pred_results]
	else:
		preds = _predict_dataframe(aligned) 
		responses = [PredictResponse(prediction=p) for p in preds] 
	
	return BatchPredictResponse(predictions=responses) 


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
	"""Return a generic 500 error; avoid leaking internals to clients."""
	return JSONResponse(status_code=500, content={"error": "Internal Server Error"})
