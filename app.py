from __future__ import annotations

import base64
import json
import math
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


FITBIT_AUTHORIZE_URL = "https://www.fitbit.com/oauth2/authorize"
FITBIT_TOKEN_URL = "https://api.fitbit.com/oauth2/token"
FITBIT_API_BASE_URL = "https://api.fitbit.com"
GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
OPTIONAL_HEALTH_SCOPES = ("temperature", "respiratory_rate", "oxygen_saturation", "cardio_fitness")
DEFAULT_SCOPES = ("activity", "heartrate", "sleep", "profile", *OPTIONAL_HEALTH_SCOPES)
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
REQUEST_TIMEOUT = 30
API_CACHE_TTL_SECONDS = 900
LOOKBACK_MIN_DAYS = 7
LOOKBACK_MAX_DAYS = 30
DEFAULT_LOOKBACK_DAYS = 7
ACTIVITY_HEATMAP_MAX_DAYS = 7
FORECAST_HORIZON_DAYS = 7
WEEKDAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
PLOT_FONT_FAMILY = "IBM Plex Sans, sans-serif"


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FitbitConfig:
    client_id: str
    client_secret: str
    redirect_uri: str
    scopes: tuple[str, ...] = DEFAULT_SCOPES

    @property
    def is_configured(self) -> bool:
        return all([self.client_id, self.client_secret, self.redirect_uri])


@dataclass
class LLMConfig:
    groq_api_key: str
    groq_model: str = DEFAULT_GROQ_MODEL

    @property
    def is_configured(self) -> bool:
        return bool(self.groq_api_key and self.groq_model)


# ---------------------------------------------------------------------------
# Environment / config loading
# ---------------------------------------------------------------------------

def bootstrap_environment() -> None:
    if load_dotenv:
        load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True)


def read_secret(name: str, default: str = "") -> str:
    """Read from Streamlit secrets or environment variable (fallback for local dev)."""
    try:
        if name in st.secrets:
            value = st.secrets[name]
            return str(value) if value is not None else default
    except StreamlitSecretNotFoundError:
        pass
    return os.getenv(name, default)


def load_config() -> FitbitConfig:
    """
    Load Fitbit config with priority:
    1. Session state (user entered in UI)
    2. Streamlit secrets / .env (local / server config)
    """
    scopes_raw = read_secret("FITBIT_SCOPES", " ".join(DEFAULT_SCOPES))
    scopes = tuple(s.strip() for s in scopes_raw.split() if s.strip()) or DEFAULT_SCOPES
    return FitbitConfig(
        client_id=st.session_state.get("fitbit_client_id") or read_secret("FITBIT_CLIENT_ID"),
        client_secret=st.session_state.get("fitbit_client_secret") or read_secret("FITBIT_CLIENT_SECRET"),
        redirect_uri=st.session_state.get("fitbit_redirect_uri") or read_secret("FITBIT_REDIRECT_URI"),
        scopes=scopes,
    )


def load_llm_config() -> LLMConfig:
    return LLMConfig(
        groq_api_key=st.session_state.get("groq_api_key") or read_secret("GROQ_API_KEY"),
        groq_model=st.session_state.get("groq_model") or read_secret("GROQ_MODEL", DEFAULT_GROQ_MODEL),
    )


# ---------------------------------------------------------------------------
# Token management (session-state only — no file I/O for Streamlit Cloud)
# ---------------------------------------------------------------------------

def auth_headers(config: FitbitConfig) -> dict[str, str]:
    encoded = base64.b64encode(
        f"{config.client_id}:{config.client_secret}".encode("utf-8")
    ).decode("utf-8")
    return {
        "Authorization": f"Basic {encoded}",
        "Content-Type": "application/x-www-form-urlencoded",
    }


def build_authorize_url(config: FitbitConfig) -> str:
    import urllib.parse
    # Encode credentials into the state param so they survive the redirect
    state_data = {
        "client_id": config.client_id,
        "client_secret": config.client_secret,
        "redirect_uri": config.redirect_uri,
        "groq_api_key": st.session_state.get("groq_api_key", ""),
        "groq_model": st.session_state.get("groq_model", DEFAULT_GROQ_MODEL),
    }
    state = base64.urlsafe_b64encode(
        json.dumps(state_data).encode()
    ).decode()
    params = {
        "response_type": "code",
        "client_id": config.client_id,
        "redirect_uri": config.redirect_uri,
        "scope": " ".join(config.scopes),
        "expires_in": "31536000",
        "state": state,
    }
    return f"{FITBIT_AUTHORIZE_URL}?{urlencode(params)}"


def persist_token_bundle(token_bundle: dict[str, Any]) -> None:
    st.session_state["fitbit_token_bundle"] = token_bundle


def clear_token_bundle() -> None:
    st.session_state.pop("fitbit_token_bundle", None)


def compute_expiry(expires_in: int | str | None) -> str:
    lifetime = int(expires_in or 0)
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=lifetime)
    return expires_at.isoformat()


def token_expired(token_bundle: dict[str, Any], skew_seconds: int = 120) -> bool:
    expires_at_raw = token_bundle.get("expires_at")
    if not expires_at_raw:
        return True
    try:
        expires_at = datetime.fromisoformat(expires_at_raw)
    except ValueError:
        return True
    return datetime.now(timezone.utc) >= expires_at - timedelta(seconds=skew_seconds)


def exchange_code_for_token(config: FitbitConfig, code: str) -> dict[str, Any]:
    response = requests.post(
        FITBIT_TOKEN_URL,
        headers=auth_headers(config),
        data={
            "client_id": config.client_id,
            "grant_type": "authorization_code",
            "redirect_uri": config.redirect_uri,
            "code": code,
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    payload["expires_at"] = compute_expiry(payload.get("expires_in"))
    return payload


def refresh_access_token(config: FitbitConfig, refresh_token: str) -> dict[str, Any]:
    response = requests.post(
        FITBIT_TOKEN_URL,
        headers=auth_headers(config),
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    payload["expires_at"] = compute_expiry(payload.get("expires_in"))
    return payload


def get_active_token_bundle(config: FitbitConfig) -> dict[str, Any] | None:
    token_bundle = st.session_state.get("fitbit_token_bundle")
    if not token_bundle:
        return None
    if token_expired(token_bundle):
        refreshed = refresh_access_token(config, token_bundle["refresh_token"])
        persist_token_bundle(refreshed)
        return refreshed
    return token_bundle


# ---------------------------------------------------------------------------
# API cache (session-state only)
# ---------------------------------------------------------------------------

def load_api_cache_store() -> dict[str, Any]:
    store = st.session_state.get("fitbit_api_cache")
    if isinstance(store, dict):
        return store
    store = {}
    st.session_state["fitbit_api_cache"] = store
    return store


def persist_api_cache_store(cache: dict[str, Any]) -> None:
    st.session_state["fitbit_api_cache"] = cache


def clear_api_cache() -> None:
    st.session_state.pop("fitbit_api_cache", None)


def clear_ai_insights_cache() -> None:
    st.session_state.pop("fitbit_ai_insights", None)


def api_cache_key(token_bundle: dict[str, Any], path: str, params: dict[str, Any] | None = None) -> str:
    payload = {
        "user_id": token_bundle.get("user_id"),
        "path": path,
        "params": params or {},
    }
    return json.dumps(payload, sort_keys=True)


def cached_fitbit_get(
    config: FitbitConfig,
    token_bundle: dict[str, Any],
    path: str,
    params: dict[str, Any] | None = None,
) -> Any:
    cache = load_api_cache_store()
    key = api_cache_key(token_bundle, path, params)
    now = datetime.now(timezone.utc)
    cached_entry = cache.get(key)
    if cached_entry:
        expires_at = datetime.fromisoformat(cached_entry["expires_at"])
        if now < expires_at:
            return cached_entry["payload"]

    try:
        payload = fitbit_get(config, token_bundle, path, params)
    except requests.HTTPError as exc:
        if cached_entry and exc.response is not None and exc.response.status_code == 429:
            return cached_entry["payload"]
        raise
    cache[key] = {
        "payload": payload,
        "expires_at": (now + timedelta(seconds=API_CACHE_TTL_SECONDS)).isoformat(),
    }
    persist_api_cache_store(cache)
    return payload


def fitbit_get(
    config: FitbitConfig,
    token_bundle: dict[str, Any],
    path: str,
    params: dict[str, Any] | None = None,
) -> Any:
    headers = {"Authorization": f"Bearer {token_bundle['access_token']}"}
    response = requests.get(
        f"{FITBIT_API_BASE_URL}{path}",
        headers=headers,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code == 401 and token_bundle.get("refresh_token"):
        refreshed = refresh_access_token(config, token_bundle["refresh_token"])
        persist_token_bundle(refreshed)
        headers["Authorization"] = f"Bearer {refreshed['access_token']}"
        response = requests.get(
            f"{FITBIT_API_BASE_URL}{path}",
            headers=headers,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
    response.raise_for_status()
    return response.json()


def format_request_error(exc: requests.RequestException) -> str:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        if exc.response.status_code == 429:
            reset_seconds = exc.response.headers.get("fitbit-rate-limit-reset")
            if reset_seconds:
                return (
                    f"Fitbit rate limit reached. Retry in about {reset_seconds} seconds. "
                    f"Response: {exc.response.text}"
                )
        return exc.response.text
    return str(exc)


def parse_token_scopes(token_bundle: dict[str, Any] | None) -> set[str]:
    if not token_bundle:
        return set()
    scopes_raw = token_bundle.get("scope", "")
    if isinstance(scopes_raw, str):
        return {s.strip() for s in scopes_raw.split() if s.strip()}
    if isinstance(scopes_raw, list):
        return {str(s).strip() for s in scopes_raw if str(s).strip()}
    return set()


def missing_scopes(token_bundle: dict[str, Any] | None, required_scopes: tuple[str, ...]) -> list[str]:
    token_scopes = parse_token_scopes(token_bundle)
    return [s for s in required_scopes if s not in token_scopes]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def extract_first_numeric(value: Any, candidate_keys: tuple[str, ...] = ()) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, dict):
        for key in candidate_keys:
            if key in value:
                extracted = extract_first_numeric(value.get(key), candidate_keys)
                if extracted is not None:
                    return extracted
        for nested in value.values():
            extracted = extract_first_numeric(nested, candidate_keys)
            if extracted is not None:
                return extracted
        return None
    if isinstance(value, list):
        for item in value:
            extracted = extract_first_numeric(item, candidate_keys)
            if extracted is not None:
                return extracted
    return None


def fetch_optional_summary_rows(
    config: FitbitConfig,
    token_bundle: dict[str, Any],
    path: str,
    payload_keys: tuple[str, ...],
    empty_columns: list[str],
    row_builder: Any,
) -> pd.DataFrame:
    try:
        payload = cached_fitbit_get(config, token_bundle, path)
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code in {400, 403, 404}:
            return pd.DataFrame(columns=empty_columns)
        raise

    # Handle endpoints that return a bare list instead of a dict
    dataset: list[dict[str, Any]] = []
    if isinstance(payload, list):
        dataset = payload
    else:
        for key in payload_keys:
            candidate = payload.get(key)
            if isinstance(candidate, list):
                dataset = candidate
                break

    rows: list[dict[str, Any]] = []
    for entry in dataset:
        row = row_builder(entry)
        if row:
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=empty_columns)
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values("date")


# ---------------------------------------------------------------------------
# Fitbit data fetchers
# ---------------------------------------------------------------------------

def fetch_profile(config: FitbitConfig, token_bundle: dict[str, Any]) -> dict[str, Any]:
    return cached_fitbit_get(config, token_bundle, "/1/user/-/profile.json").get("user", {})


def fetch_daily_steps(
    config: FitbitConfig,
    token_bundle: dict[str, Any],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    payload = cached_fitbit_get(
        config,
        token_bundle,
        f"/1/user/-/activities/steps/date/{start_date.isoformat()}/{end_date.isoformat()}.json",
    )
    rows = payload.get("activities-steps", [])
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["date", "steps"])
    frame["date"] = pd.to_datetime(frame["dateTime"])
    frame["steps"] = pd.to_numeric(frame["value"], errors="coerce").fillna(0).astype(int)
    return frame[["date", "steps"]].sort_values("date")


def fetch_intraday_steps(
    config: FitbitConfig,
    token_bundle: dict[str, Any],
    selected_date: date,
) -> pd.DataFrame:
    payload = cached_fitbit_get(
        config,
        token_bundle,
        f"/1/user/-/activities/steps/date/{selected_date.isoformat()}/1d/1min.json",
    )
    dataset = payload.get("activities-steps-intraday", {}).get("dataset", [])
    frame = pd.DataFrame(dataset)
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "steps"])
    frame["timestamp"] = pd.to_datetime(
        selected_date.isoformat() + " " + frame["time"].astype(str),
        errors="coerce",
    )
    frame["steps"] = pd.to_numeric(frame["value"], errors="coerce").fillna(0)
    return frame[["timestamp", "steps"]].dropna().sort_values("timestamp")

def fetch_intraday_heart_rate(
    config: FitbitConfig,
    token_bundle: dict[str, Any],
    selected_date: date,
) -> pd.DataFrame:
    try:
        payload = cached_fitbit_get(
            config,
            token_bundle,
            f"/1/user/-/activities/heart/date/{selected_date.isoformat()}/1d/1sec.json",
        )
    except requests.HTTPError:
        try:
            payload = cached_fitbit_get(
                config,
                token_bundle,
                f"/1/user/-/activities/heart/date/{selected_date.isoformat()}/1d/1min.json",
            )
        except requests.HTTPError:
            return pd.DataFrame(columns=["timestamp", "heart_rate"])

    dataset = payload.get("activities-heart-intraday", {}).get("dataset", [])
    frame = pd.DataFrame(dataset)
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "heart_rate"])
    frame["timestamp"] = pd.to_datetime(
        selected_date.isoformat() + " " + frame["time"].astype(str),
        errors="coerce",
    )
    frame["heart_rate"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame[["timestamp", "heart_rate"]].dropna().sort_values("timestamp")

def fetch_intraday_steps_window(
    config: FitbitConfig,
    token_bundle: dict[str, Any],
    start_date: date,
    end_date: date,
    max_days: int = ACTIVITY_HEATMAP_MAX_DAYS,
) -> pd.DataFrame:
    capped_start_date = max(start_date, end_date - timedelta(days=max_days - 1))
    frames: list[pd.DataFrame] = []
    current_date = capped_start_date
    while current_date <= end_date:
        daily_frame = fetch_intraday_steps(config, token_bundle, current_date)
        if not daily_frame.empty:
            frames.append(daily_frame)
        current_date += timedelta(days=1)
    if not frames:
        return pd.DataFrame(columns=["timestamp", "steps"])
    return pd.concat(frames, ignore_index=True).sort_values("timestamp")


def fetch_heart_rate_summary(
    config: FitbitConfig,
    token_bundle: dict[str, Any],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    payload = cached_fitbit_get(
        config,
        token_bundle,
        f"/1/user/-/activities/heart/date/{start_date.isoformat()}/{end_date.isoformat()}.json",
    )
    rows: list[dict[str, Any]] = []
    for entry in payload.get("activities-heart", []):
        heart_value = entry.get("value", {})
        rows.append(
            {
                "date": entry.get("dateTime"),
                "resting_heart_rate": pd.to_numeric(
                    heart_value.get("restingHeartRate"), errors="coerce"
                ),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["date", "resting_heart_rate"])
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values("date")


def fetch_sleep_logs(
    config: FitbitConfig,
    token_bundle: dict[str, Any],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    payload = cached_fitbit_get(
        config,
        token_bundle,
        f"/1.2/user/-/sleep/date/{start_date.isoformat()}/{end_date.isoformat()}.json",
    )
    sleep_logs = payload.get("sleep", [])
    logs_by_date: dict[str, list[dict[str, Any]]] = {}
    for entry in sleep_logs:
        date_of_sleep = entry.get("dateOfSleep")
        if not date_of_sleep:
            continue
        logs_by_date.setdefault(date_of_sleep, []).append(entry)

    rows: list[dict[str, Any]] = []
    for date_of_sleep in sorted(logs_by_date):
        entries = logs_by_date[date_of_sleep]
        if entries:
            rows.append(normalize_sleep_log(pick_main_sleep_log(entries)))
    if not rows:
        return pd.DataFrame(
            columns=[
                "date", "start_time", "duration_hours", "time_in_bed_hours",
                "minutes_asleep", "minutes_awake", "efficiency",
                "deep_hours", "light_hours", "rem_hours", "wake_hours",
            ]
        )
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values("date")


def fetch_temperature_summary(
    config: FitbitConfig,
    token_bundle: dict[str, Any],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    return fetch_optional_summary_rows(
        config=config,
        token_bundle=token_bundle,
        path=f"/1/user/-/temp/skin/date/{start_date.isoformat()}/{end_date.isoformat()}.json",
        payload_keys=("tempSkin", "temperatureSkin"),
        empty_columns=["date", "temperature_variation"],
        row_builder=lambda entry: {
            "date": entry.get("dateTime"),
            "temperature_variation": extract_first_numeric(
                entry.get("value"), ("nightlyRelative", "value", "temperature")
            ),
        }
        if entry.get("dateTime")
        else None,
    )


def fetch_respiratory_rate_summary(
    config: FitbitConfig,
    token_bundle: dict[str, Any],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    return fetch_optional_summary_rows(
        config=config,
        token_bundle=token_bundle,
        path=f"/1/user/-/br/date/{start_date.isoformat()}/{end_date.isoformat()}.json",
        payload_keys=("br", "breathingRate"),
        empty_columns=["date", "respiratory_rate"],
        row_builder=lambda entry: {
            "date": entry.get("dateTime"),
            "respiratory_rate": extract_first_numeric(
                entry.get("value"), ("breathingRate", "br", "value")
            ),
        }
        if entry.get("dateTime")
        else None,
    )


def fetch_oxygen_saturation_summary(
    config: FitbitConfig,
    token_bundle: dict[str, Any],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    return fetch_optional_summary_rows(
        config=config,
        token_bundle=token_bundle,
        path=f"/1/user/-/spo2/date/{start_date.isoformat()}/{end_date.isoformat()}.json",
        payload_keys=("spo2", "spO2"),
        empty_columns=["date", "oxygen_saturation_avg", "oxygen_saturation_min", "oxygen_saturation_max"],
        row_builder=lambda entry: {
            "date": entry.get("dateTime"),
            "oxygen_saturation_avg": extract_first_numeric(
                entry.get("value"), ("avg", "average", "value")
            ),
            "oxygen_saturation_min": extract_first_numeric(
                entry.get("value"), ("min", "minimum")
            ),
            "oxygen_saturation_max": extract_first_numeric(
                entry.get("value"), ("max", "maximum")
            ),
        }
        if entry.get("dateTime")
        else None,
    )


def fetch_cardio_fitness_summary(
    config: FitbitConfig,
    token_bundle: dict[str, Any],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    return fetch_optional_summary_rows(
        config=config,
        token_bundle=token_bundle,
        path=f"/1/user/-/cardioscore/date/{start_date.isoformat()}/{end_date.isoformat()}.json",
        payload_keys=("cardioScore", "cardioscore"),
        empty_columns=["date", "cardio_fitness"],
        row_builder=lambda entry: {
            "date": entry.get("dateTime"),
            "cardio_fitness": extract_first_numeric(
                entry.get("value"), ("vo2Max", "cardioFitnessScore", "value")
            ),
        }
        if entry.get("dateTime")
        else None,
    )


# ---------------------------------------------------------------------------
# Sleep helpers
# ---------------------------------------------------------------------------

def pick_main_sleep_log(sleep_logs: list[dict[str, Any]]) -> dict[str, Any]:
    main_sleeps = [e for e in sleep_logs if e.get("isMainSleep")]
    candidates = main_sleeps or sleep_logs
    return max(candidates, key=lambda e: e.get("duration", 0))


def normalize_sleep_log(sleep_log: dict[str, Any]) -> dict[str, Any]:
    summary = sleep_log.get("levels", {}).get("summary", {})

    def stage_minutes(stage: str) -> int:
        return int(summary.get(stage, {}).get("minutes", 0) or 0)

    return {
        "date": sleep_log.get("dateOfSleep"),
        "start_time": sleep_log.get("startTime"),
        "duration_hours": round((sleep_log.get("duration", 0) or 0) / 3_600_000, 2),
        "time_in_bed_hours": round((sleep_log.get("timeInBed", 0) or 0) / 60, 2),
        "minutes_asleep": int(sleep_log.get("minutesAsleep", 0) or 0),
        "minutes_awake": int(sleep_log.get("minutesAwake", 0) or 0),
        "efficiency": int(sleep_log.get("efficiency", 0) or 0),
        "deep_hours": round(stage_minutes("deep") / 60, 2),
        "light_hours": round(stage_minutes("light") / 60, 2),
        "rem_hours": round(stage_minutes("rem") / 60, 2),
        "wake_hours": round(stage_minutes("wake") / 60, 2),
    }


# ---------------------------------------------------------------------------
# Analytics helpers
# ---------------------------------------------------------------------------

def aggregate_intraday_steps_by_hour(intraday_steps: pd.DataFrame) -> pd.DataFrame:
    if intraday_steps.empty:
        return pd.DataFrame(columns=["hour", "hour_label", "steps"])
    frame = intraday_steps.copy()
    frame["hour"] = frame["timestamp"].dt.hour
    hourly = frame.groupby("hour", as_index=False)["steps"].sum()
    hourly["hour_label"] = hourly["hour"].map(lambda h: f"{int(h):02d}:00")
    return hourly.sort_values("hour")


def build_activity_heatmap_frame(intraday_window: pd.DataFrame) -> pd.DataFrame:
    if intraday_window.empty:
        return pd.DataFrame(columns=["date", "day_label", "hour", "steps"])
    frame = intraday_window.copy()
    frame["date"] = frame["timestamp"].dt.normalize()
    frame["hour"] = frame["timestamp"].dt.hour.astype(int)
    grouped = frame.groupby(["date", "hour"], as_index=False)["steps"].sum()
    grouped["day_label"] = grouped["date"].dt.strftime("%a %d %b")
    return grouped.sort_values(["date", "hour"])


def normalize_bedtime_hours(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    bedtime_hours = parsed.dt.hour + (parsed.dt.minute.fillna(0) / 60)
    return bedtime_hours.where(bedtime_hours >= 18, bedtime_hours + 24)


def bedtime_axis_config(values: pd.Series) -> tuple[list[int], list[float]]:
    clean_values = pd.to_numeric(values, errors="coerce").dropna()
    if clean_values.empty:
        return list(range(20, 31, 2)), [18, 30]
    min_tick = max(18, int(math.floor((clean_values.min() - 1) / 2) * 2))
    max_tick = min(42, int(math.ceil((clean_values.max() + 1) / 2) * 2))
    if max_tick <= min_tick:
        max_tick = min(42, min_tick + 4)
    tick_values = list(range(min_tick, max_tick + 1, 2))
    return tick_values, [min_tick - 0.5, max_tick + 0.5]


def build_consistency_score_frame(
    steps_daily: pd.DataFrame,
    sleep_frame: pd.DataFrame,
    heart_daily: pd.DataFrame,
) -> pd.DataFrame:
    merged = merge_daily_metrics(steps_daily, heart_daily, sleep_frame)
    if merged.empty:
        return pd.DataFrame(
            columns=[
                "date", "consistency_score", "steps_score", "sleep_score",
                "efficiency_score", "bedtime_score", "recovery_score",
            ]
        )

    score_frame = merged.copy()
    score_frame["steps_score"] = (
        pd.to_numeric(score_frame["steps"], errors="coerce").fillna(0) / 10_000
    ).clip(0, 1) * 45
    score_frame["sleep_score"] = (
        pd.to_numeric(score_frame["duration_hours"], errors="coerce").fillna(0) / 7
    ).clip(0, 1) * 25
    score_frame["efficiency_score"] = (
        pd.to_numeric(score_frame["efficiency"], errors="coerce").fillna(0) / 100
    ).clip(0, 1) * 15

    bedtime_frame = (
        sleep_frame[["date", "start_time"]].copy()
        if not sleep_frame.empty
        else pd.DataFrame(columns=["date", "start_time"])
    )
    if not bedtime_frame.empty:
        bedtime_frame["bedtime_hour"] = normalize_bedtime_hours(bedtime_frame["start_time"])
        median_bedtime = bedtime_frame["bedtime_hour"].dropna().median()
        bedtime_frame["bedtime_score"] = (
            1 - (bedtime_frame["bedtime_hour"] - median_bedtime).abs().fillna(3) / 3
        ).clip(lower=0) * 10
        score_frame = score_frame.merge(
            bedtime_frame[["date", "bedtime_score"]], on="date", how="left"
        )
    else:
        score_frame["bedtime_score"] = 0.0

    resting_series = pd.to_numeric(score_frame["resting_heart_rate"], errors="coerce")
    resting_baseline = resting_series.dropna().median()
    if pd.isna(resting_baseline):
        score_frame["recovery_score"] = 0.0
    else:
        resting_delta = (resting_series - resting_baseline).clip(lower=0).fillna(8)
        score_frame["recovery_score"] = (1 - resting_delta / 8).clip(lower=0) * 5

    component_columns = [
        "steps_score", "sleep_score", "efficiency_score", "bedtime_score", "recovery_score"
    ]
    score_frame["bedtime_display"] = (
        bedtime_frame.set_index("date")["bedtime_hour"].reindex(score_frame["date"]).values
        if not bedtime_frame.empty
        else None
    )
    score_frame["consistency_score"] = score_frame[component_columns].sum(axis=1).round(1)
    return score_frame[["date", "consistency_score", *component_columns, "bedtime_display"]].sort_values("date")


def merge_daily_metrics(
    steps_daily: pd.DataFrame,
    heart_daily: pd.DataFrame,
    sleep_frame: pd.DataFrame,
    temperature_frame: pd.DataFrame | None = None,
    respiratory_frame: pd.DataFrame | None = None,
    oxygen_frame: pd.DataFrame | None = None,
    cardio_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    merged = steps_daily.copy()
    if not heart_daily.empty:
        merged = merged.merge(heart_daily, on="date", how="left")
    if not sleep_frame.empty:
        merged = merged.merge(
            sleep_frame[["date", "duration_hours", "efficiency", "deep_hours", "rem_hours"]],
            on="date",
            how="left",
        )
    if temperature_frame is not None and not temperature_frame.empty:
        merged = merged.merge(temperature_frame, on="date", how="left")
    if respiratory_frame is not None and not respiratory_frame.empty:
        merged = merged.merge(respiratory_frame, on="date", how="left")
    if oxygen_frame is not None and not oxygen_frame.empty:
        merged = merged.merge(oxygen_frame, on="date", how="left")
    if cardio_frame is not None and not cardio_frame.empty:
        merged = merged.merge(cardio_frame, on="date", how="left")
    return merged.sort_values("date")


def build_selected_day_snapshot(
    steps_daily: pd.DataFrame,
    intraday_steps: pd.DataFrame,
    heart_daily: pd.DataFrame,
    sleep_frame: pd.DataFrame,
    temperature_frame: pd.DataFrame,
    respiratory_frame: pd.DataFrame,
    oxygen_frame: pd.DataFrame,
    cardio_frame: pd.DataFrame,
    selected_date: date,
) -> dict[str, Any]:
    selected_day = pd.Timestamp(selected_date)
    steps_row = steps_daily.loc[steps_daily["date"] == selected_day]
    heart_row = heart_daily.loc[heart_daily["date"] == selected_day] if not heart_daily.empty else pd.DataFrame()
    sleep_row = sleep_frame.loc[sleep_frame["date"] == selected_day] if not sleep_frame.empty else pd.DataFrame()
    temperature_row = temperature_frame.loc[temperature_frame["date"] == selected_day] if not temperature_frame.empty else pd.DataFrame()
    respiratory_row = respiratory_frame.loc[respiratory_frame["date"] == selected_day] if not respiratory_frame.empty else pd.DataFrame()
    oxygen_row = oxygen_frame.loc[oxygen_frame["date"] == selected_day] if not oxygen_frame.empty else pd.DataFrame()
    cardio_row = cardio_frame.loc[cardio_frame["date"] == selected_day] if not cardio_frame.empty else pd.DataFrame()
    hourly_steps = aggregate_intraday_steps_by_hour(intraday_steps)

    return {
        "date": selected_date.isoformat(),
        "steps": int(steps_row["steps"].iloc[0]) if not steps_row.empty else None,
        "peak_steps_hour": hourly_steps.sort_values("steps", ascending=False)["hour_label"].iloc[0] if not hourly_steps.empty else None,
        "peak_steps_hour_value": int(hourly_steps["steps"].max()) if not hourly_steps.empty else None,
        "resting_heart_rate": int(heart_row["resting_heart_rate"].iloc[0]) if not heart_row.empty and pd.notna(heart_row["resting_heart_rate"].iloc[0]) else None,
        "sleep_hours": round(float(sleep_row["duration_hours"].iloc[0]), 2) if not sleep_row.empty else None,
        "sleep_efficiency": int(sleep_row["efficiency"].iloc[0]) if not sleep_row.empty else None,
        "bedtime": sleep_row["start_time"].iloc[0] if not sleep_row.empty else None,
        "temperature_variation": round(float(temperature_row["temperature_variation"].iloc[0]), 2) if not temperature_row.empty and pd.notna(temperature_row["temperature_variation"].iloc[0]) else None,
        "respiratory_rate": round(float(respiratory_row["respiratory_rate"].iloc[0]), 2) if not respiratory_row.empty and pd.notna(respiratory_row["respiratory_rate"].iloc[0]) else None,
        "oxygen_saturation_avg": round(float(oxygen_row["oxygen_saturation_avg"].iloc[0]), 2) if not oxygen_row.empty and pd.notna(oxygen_row["oxygen_saturation_avg"].iloc[0]) else None,
        "cardio_fitness": round(float(cardio_row["cardio_fitness"].iloc[0]), 2) if not cardio_row.empty and pd.notna(cardio_row["cardio_fitness"].iloc[0]) else None,
    }


def compare_recent_vs_prior(
    frame: pd.DataFrame, value_column: str, window: int = 7
) -> tuple[float | None, float | None]:
    if len(frame) < window * 2:
        return None, None
    recent = float(frame[value_column].tail(window).mean())
    prior = float(frame[value_column].tail(window * 2).head(window).mean())
    if prior == 0:
        return recent, None
    return recent, ((recent - prior) / prior) * 100


def latest_z_score(frame: pd.DataFrame, value_column: str) -> float | None:
    if len(frame) < 3:
        return None
    series = pd.to_numeric(frame[value_column], errors="coerce").dropna()
    if len(series) < 3:
        return None
    std = float(series.std(ddof=0))
    if std == 0:
        return None
    return float((series.iloc[-1] - series.mean()) / std)


def correlation_strength(correlation: float) -> str:
    absolute = abs(correlation)
    if absolute >= 0.7:
        return "strong"
    if absolute >= 0.4:
        return "moderate"
    if absolute >= 0.2:
        return "light"
    return "weak"


def linear_forecast(
    frame: pd.DataFrame, value_column: str, horizon: int = FORECAST_HORIZON_DAYS
) -> dict[str, Any] | None:
    working = frame[["date", value_column]].dropna().copy()
    if len(working) < 5:
        return None

    working = working.sort_values("date").reset_index(drop=True)
    x_values = list(range(len(working)))
    y_values = [float(v) for v in working[value_column]]
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    denominator = sum((x - x_mean) ** 2 for x in x_values)
    if denominator == 0:
        return None
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values)) / denominator
    intercept = y_mean - slope * x_mean
    trend_values = [intercept + slope * x for x in x_values]

    ss_res = sum((a - f) ** 2 for a, f in zip(y_values, trend_values))
    ss_tot = sum((a - y_mean) ** 2 for a in y_values)
    residual_std = (ss_res / max(len(y_values) - 2, 1)) ** 0.5 if len(y_values) > 2 else 0.0
    r_squared = 1 - (ss_res / ss_tot) if ss_tot else 1.0

    working["trend"] = trend_values
    last_date = pd.Timestamp(working["date"].iloc[-1]).to_pydatetime()
    forecast_rows: list[dict[str, Any]] = []
    for offset in range(1, horizon + 1):
        prediction = max(0.0, intercept + slope * (len(working) + offset - 1))
        forecast_rows.append(
            {
                "date": last_date + timedelta(days=offset),
                value_column: prediction,
                "lower": max(0.0, prediction - 1.96 * residual_std),
                "upper": max(0.0, prediction + 1.96 * residual_std),
            }
        )
    return {
        "history": working,
        "forecast": pd.DataFrame(forecast_rows),
        "slope": slope,
        "r_squared": r_squared,
        "next_value": forecast_rows[0][value_column] if forecast_rows else None,
    }


def build_statistical_insights(
    steps_daily: pd.DataFrame,
    heart_daily: pd.DataFrame,
    sleep_frame: pd.DataFrame,
    temperature_frame: pd.DataFrame,
    respiratory_frame: pd.DataFrame,
    oxygen_frame: pd.DataFrame,
    cardio_frame: pd.DataFrame,
    selected_date: date,
) -> list[str]:
    insights: list[str] = []

    recent_steps, steps_delta_pct = compare_recent_vs_prior(steps_daily, "steps")
    if recent_steps is not None:
        direction = "above" if (steps_delta_pct or 0) >= 0 else "below"
        insights.append(
            f"Recent 7-day steps average is {recent_steps:,.0f}, "
            f"{abs(steps_delta_pct or 0):.1f}% {direction} the prior week."
        )

    if not heart_daily.empty:
        recent_hr, heart_delta_pct = compare_recent_vs_prior(
            heart_daily.dropna(subset=["resting_heart_rate"]), "resting_heart_rate"
        )
        if recent_hr is not None:
            direction = "higher" if (heart_delta_pct or 0) >= 0 else "lower"
            insights.append(
                f"Recent resting heart rate averages {recent_hr:.1f} bpm, "
                f"{abs(heart_delta_pct or 0):.1f}% {direction} than the prior week."
            )

    merged = merge_daily_metrics(
        steps_daily, heart_daily, sleep_frame,
        temperature_frame=temperature_frame,
        respiratory_frame=respiratory_frame,
        oxygen_frame=oxygen_frame,
        cardio_frame=cardio_frame,
    )
    if {"steps", "duration_hours"}.issubset(merged.columns):
        joined = merged[["steps", "duration_hours"]].dropna()
        correlation = float(joined.corr().iloc[0, 1]) if len(joined) >= 3 else None
        if correlation is not None and pd.notna(correlation):
            direction = "positive" if correlation >= 0 else "negative"
            insights.append(
                f"Sleep duration and steps show a {correlation_strength(correlation)} "
                f"{direction} correlation ({correlation:.2f}) over the selected window."
            )

    step_z_score = latest_z_score(steps_daily, "steps")
    if step_z_score is not None and abs(step_z_score) >= 1.0:
        direction = "above" if step_z_score > 0 else "below"
        insights.append(
            f"Latest daily steps are {abs(step_z_score):.1f} standard deviations "
            f"{direction} your recent average."
        )

    if not oxygen_frame.empty and oxygen_frame["oxygen_saturation_avg"].dropna().min() < 95:
        insights.append(
            "At least one selected-window day shows average oxygen saturation below 95%; "
            "treat that as observational only, not diagnostic."
        )

    if not cardio_frame.empty:
        cardio_recent, cardio_delta_pct = compare_recent_vs_prior(
            cardio_frame.dropna(subset=["cardio_fitness"]), "cardio_fitness"
        )
        if cardio_recent is not None:
            direction = "higher" if (cardio_delta_pct or 0) >= 0 else "lower"
            insights.append(
                f"Recent cardio fitness averages {cardio_recent:.1f}, "
                f"{abs(cardio_delta_pct or 0):.1f}% {direction} than the prior week."
            )

    if not insights:
        insights.append(
            "Not enough history is available yet for comparative statistics."
        )
    return insights


# ---------------------------------------------------------------------------
# AI insights
# ---------------------------------------------------------------------------

def summarize_fitness_data(
    steps_daily: pd.DataFrame,
    intraday_steps: pd.DataFrame,
    heart_daily: pd.DataFrame,
    sleep_frame: pd.DataFrame,
    temperature_frame: pd.DataFrame,
    respiratory_frame: pd.DataFrame,
    oxygen_frame: pd.DataFrame,
    cardio_frame: pd.DataFrame,
    selected_date: date,
    lookback_days: int,
    insight_scope: str,
) -> dict[str, Any]:
    latest_steps = int(steps_daily["steps"].iloc[-1]) if not steps_daily.empty else 0
    avg_steps = round(float(steps_daily["steps"].mean()), 1) if not steps_daily.empty else 0.0
    step_trend = (
        round(float(steps_daily["steps"].tail(min(7, len(steps_daily))).mean()), 1)
        if not steps_daily.empty
        else 0.0
    )
    avg_sleep_hours = round(float(sleep_frame["duration_hours"].mean()), 2) if not sleep_frame.empty else 0.0
    avg_sleep_efficiency = round(float(sleep_frame["efficiency"].mean()), 1) if not sleep_frame.empty else 0.0
    merged = merge_daily_metrics(
        steps_daily, heart_daily, sleep_frame,
        temperature_frame=temperature_frame,
        respiratory_frame=respiratory_frame,
        oxygen_frame=oxygen_frame,
        cardio_frame=cardio_frame,
    )
    recent_days = []
    if not merged.empty:
        for _, row in merged.iterrows():
            recent_days.append(
                {
                    "date": pd.Timestamp(row["date"]).date().isoformat(),
                    "steps": int(row["steps"]) if pd.notna(row.get("steps")) else None,
                    "resting_heart_rate": int(row["resting_heart_rate"]) if pd.notna(row.get("resting_heart_rate")) else None,
                    "sleep_hours": round(float(row["duration_hours"]), 2) if pd.notna(row.get("duration_hours")) else None,
                    "sleep_efficiency": int(row["efficiency"]) if pd.notna(row.get("efficiency")) else None,
                    "temperature_variation": round(float(row["temperature_variation"]), 2) if pd.notna(row.get("temperature_variation")) else None,
                    "respiratory_rate": round(float(row["respiratory_rate"]), 2) if pd.notna(row.get("respiratory_rate")) else None,
                    "oxygen_saturation_avg": round(float(row["oxygen_saturation_avg"]), 2) if pd.notna(row.get("oxygen_saturation_avg")) else None,
                    "cardio_fitness": round(float(row["cardio_fitness"]), 2) if pd.notna(row.get("cardio_fitness")) else None,
                }
            )
    selected_day = build_selected_day_snapshot(
        steps_daily=steps_daily,
        intraday_steps=intraday_steps,
        heart_daily=heart_daily,
        sleep_frame=sleep_frame,
        temperature_frame=temperature_frame,
        respiratory_frame=respiratory_frame,
        oxygen_frame=oxygen_frame,
        cardio_frame=cardio_frame,
        selected_date=selected_date,
    )
    best_steps_day = None
    if not steps_daily.empty:
        best_row = steps_daily.sort_values("steps", ascending=False).iloc[0]
        best_steps_day = {
            "date": pd.Timestamp(best_row["date"]).date().isoformat(),
            "steps": int(best_row["steps"]),
        }
    lowest_steps_day = None
    if not steps_daily.empty:
        low_row = steps_daily.sort_values("steps").iloc[0]
        lowest_steps_day = {
            "date": pd.Timestamp(low_row["date"]).date().isoformat(),
            "steps": int(low_row["steps"]),
        }

    return {
        "insight_scope": insight_scope,
        "lookback_days": lookback_days,
        "selected_date": selected_date.isoformat(),
        "daily_steps": {
            "days_with_data": int(len(steps_daily)),
            "total_steps": int(steps_daily["steps"].sum()) if not steps_daily.empty else 0,
            "average_steps": avg_steps,
            "latest_day_steps": latest_steps,
            "recent_7_day_average_steps": step_trend,
            "best_steps_day": best_steps_day,
            "lowest_steps_day": lowest_steps_day,
        },
        "heart_rate": {
            "average_resting_heart_rate": round(
                float(heart_daily["resting_heart_rate"].dropna().mean()), 1
            )
            if not heart_daily.dropna(subset=["resting_heart_rate"]).empty
            else None,
            "latest_resting_heart_rate": int(heart_daily["resting_heart_rate"].dropna().iloc[-1])
            if not heart_daily.dropna(subset=["resting_heart_rate"]).empty
            else None,
        },
        "sleep": {
            "days_with_logs": int(len(sleep_frame)),
            "average_sleep_hours": avg_sleep_hours,
            "average_efficiency": avg_sleep_efficiency,
            "latest_sleep_hours": round(float(sleep_frame["duration_hours"].iloc[-1]), 2)
            if not sleep_frame.empty
            else None,
        },
        "wellness": {
            "days_with_temperature": int(len(temperature_frame)),
            "days_with_respiratory_rate": int(len(respiratory_frame)),
            "days_with_oxygen_saturation": int(len(oxygen_frame)),
            "days_with_cardio_fitness": int(len(cardio_frame)),
            "average_temperature_variation": round(float(temperature_frame["temperature_variation"].mean()), 2) if not temperature_frame.empty else None,
            "average_respiratory_rate": round(float(respiratory_frame["respiratory_rate"].mean()), 2) if not respiratory_frame.empty else None,
            "average_oxygen_saturation": round(float(oxygen_frame["oxygen_saturation_avg"].mean()), 2) if not oxygen_frame.empty else None,
            "average_cardio_fitness": round(float(cardio_frame["cardio_fitness"].mean()), 2) if not cardio_frame.empty else None,
        },
        "selected_day": selected_day if insight_scope == "selected_day" else None,
        "window_daily_records": recent_days
        if insight_scope == "window_summary"
        else recent_days[-min(7, len(recent_days)):],
    }


def generate_ai_insights(
    llm_config: LLMConfig,
    steps_daily: pd.DataFrame,
    intraday_steps: pd.DataFrame,
    heart_daily: pd.DataFrame,
    sleep_frame: pd.DataFrame,
    temperature_frame: pd.DataFrame,
    respiratory_frame: pd.DataFrame,
    oxygen_frame: pd.DataFrame,
    cardio_frame: pd.DataFrame,
    selected_date: date,
    lookback_days: int,
    insight_scope: str,
) -> str:
    summary = summarize_fitness_data(
        steps_daily=steps_daily,
        intraday_steps=intraday_steps,
        heart_daily=heart_daily,
        sleep_frame=sleep_frame,
        temperature_frame=temperature_frame,
        respiratory_frame=respiratory_frame,
        oxygen_frame=oxygen_frame,
        cardio_frame=cardio_frame,
        selected_date=selected_date,
        lookback_days=lookback_days,
        insight_scope=insight_scope,
    )
    response = requests.post(
        GROQ_CHAT_COMPLETIONS_URL,
        headers={
            "Authorization": f"Bearer {llm_config.groq_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": llm_config.groq_model,
            "temperature": 0.2,
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a professional health and fitness insights assistant analyzing Fitbit data. "
                        "Write with the judgment of a medically literate clinician and an experienced fitness coach: calm, precise, evidence-oriented, and respectful.\n\n"
                        "TONE & STYLE:\n"
                        "- Be professional, clear, supportive, and concise\n"
                        "- Do not be offensive, sarcastic, insulting, or mocking\n"
                        "- Sound like a highly qualified health professional reviewing real data\n"
                        "- Prefer medically accurate wording over hype or drama\n\n"
                        "ANALYSIS:\n"
                        "- Identify trends, spikes, drops, and inconsistencies\n"
                        "- Compare recent behavior vs baseline\n"
                        "- Highlight what the user is doing well and where improvement is needed\n"
                        "- If data is weak or missing, say so clearly and cautiously\n"
                        "- Numeric comparisons must be mathematically correct\n"
                        "- If insight_scope is window_summary, focus on the entire selected window\n"
                        "- If insight_scope is selected_day, focus on the chosen day vs baseline\n\n"
                        "MEDICAL ACCURACY:\n"
                        "- Prioritize health accuracy and conservative interpretation\n"
                        "- Do not diagnose diseases or prescribe medication\n"
                        "- Use careful language: 'may suggest', 'can be associated with', 'worth monitoring'\n\n"
                        "OUTPUT FORMAT (STRICT):\n"
                        "1. One short professional headline\n"
                        "2. 3 bullet insights (specific, data-backed)\n"
                        "3. 2 bullet action steps (clear, practical)\n\n"
                        "- Keep it concise and high-signal\n"
                        "- Make every point traceable to the provided data\n"
                        "- Do not mention intraday or minute-level heart-rate data\n"
                        "- Base heart commentary only on resting heart rate\n"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Analyze this Fitbit data and follow the required format exactly:\n{json.dumps(summary, indent=2)}",
                },
            ],
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    choices = payload.get("choices", [])
    if not choices:
        raise ValueError("Groq returned no choices.")
    content = choices[0].get("message", {}).get("content", "")
    if not content:
        raise ValueError("Groq returned an empty response.")
    return content


# ---------------------------------------------------------------------------
# OAuth helpers
# ---------------------------------------------------------------------------

def format_query_param(name: str) -> str | None:
    value = st.query_params.get(name)
    if isinstance(value, list):
        return value[0] if value else None
    return value


def clear_auth_query_params() -> None:
    for key in ("code", "state", "error", "error_description"):
        try:
            del st.query_params[key]
        except KeyError:
            pass


def handle_oauth_callback(config: FitbitConfig) -> None:
    error = format_query_param("error")
    if error:
        description = format_query_param("error_description") or "Authorization failed."
        st.error(f"Fitbit authorization error: {description}")
        clear_auth_query_params()
        st.stop()

    code = format_query_param("code")
    if not code:
        return

    # Restore credentials from state param if session state was wiped
    state = format_query_param("state")
    if state:
        try:
            state_data = json.loads(base64.urlsafe_b64decode(state.encode()).decode())
            if not st.session_state.get("fitbit_client_id"):
                st.session_state["fitbit_client_id"] = state_data.get("client_id", "")
            if not st.session_state.get("fitbit_client_secret"):
                st.session_state["fitbit_client_secret"] = state_data.get("client_secret", "")
            if not st.session_state.get("fitbit_redirect_uri"):
                st.session_state["fitbit_redirect_uri"] = state_data.get("redirect_uri", "")
            if not st.session_state.get("groq_api_key") and state_data.get("groq_api_key"):
                st.session_state["groq_api_key"] = state_data.get("groq_api_key", "")
            # Reload config with restored credentials
            config = load_config()
        except Exception:
            pass

    try:
        token_bundle = exchange_code_for_token(config, code)
    except requests.HTTPError as exc:
        body = exc.response.text if exc.response is not None else str(exc)
        st.error(f"Unable to exchange Fitbit authorization code: {body}")
        st.stop()
    persist_token_bundle(token_bundle)
    clear_auth_query_params()
    st.rerun()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def metric_display(value: Any, formatter: Any, missing_text: str = "Not available") -> str:
    if value is None:
        return missing_text
    return formatter(value)


def format_bedtime_display(value: Any) -> str:
    if value is None:
        return "No bedtime log"
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return "No bedtime log"
    return parsed.strftime("%I:%M %p").lstrip("0")


def format_datetime_display(value: Any) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return "No timestamp"
    return f"{parsed.day} {parsed.strftime('%b')} {parsed.strftime('%I:%M %p').lstrip('0')}"


def format_date_heading(value: date) -> str:
    return f"{value.strftime('%A')}, {value.day} {value.strftime('%b %Y')}"


def format_date_range_label(start_value: date, end_value: date) -> str:
    return f"{start_value.day} {start_value.strftime('%b %Y')} to {end_value.day} {end_value.strftime('%b %Y')}"


def format_date_compact(value: date | pd.Timestamp) -> str:
    timestamp = pd.Timestamp(value)
    return f"{timestamp.day} {timestamp.strftime('%b %Y')}"


def format_clock_tick(hour_value: int) -> str:
    hour = hour_value % 24
    suffix = "AM" if hour < 12 else "PM"
    display_hour = hour % 12 or 12
    return f"{display_hour} {suffix}"


def format_bedtime_hour_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return "No bedtime"
    total_minutes = int(round((float(value) % 24) * 60)) % (24 * 60)
    hour = total_minutes // 60
    minute = total_minutes % 60
    suffix = "AM" if hour < 12 else "PM"
    display_hour = hour % 12 or 12
    return f"{display_hour}:{minute:02d} {suffix}"


# ---------------------------------------------------------------------------
# Plotly helpers
# ---------------------------------------------------------------------------

def style_figure(
    fig: go.Figure,
    *,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    height: int = 380,
    legend_title: str | None = None,
) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left"),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        hovermode="x unified",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        margin=dict(l=18, r=18, t=62, b=18),
        font=dict(family=PLOT_FONT_FAMILY, color="#0f172a"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title_text=legend_title,
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            tickfont=dict(color="#526071"),
            title_font=dict(color="#526071"),
        ),
        yaxis=dict(
            gridcolor="rgba(15, 23, 42, 0.08)",
            zeroline=False,
            tickfont=dict(color="#526071"),
            title_font=dict(color="#526071"),
        ),
    )
    return fig


def build_forecast_figure(
    forecast_bundle: dict[str, Any],
    value_column: str,
    title: str,
    yaxis_title: str,
    actual_color: str,
    band_color: str,
) -> go.Figure:
    history = forecast_bundle["history"]
    forecast = forecast_bundle["forecast"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history["date"], y=history[value_column], mode="lines+markers", name="Actual", line=dict(color=actual_color, width=2)))
    fig.add_trace(go.Scatter(x=history["date"], y=history["trend"], mode="lines", name="Trend", line=dict(color=actual_color, width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=forecast["date"], y=forecast["upper"], mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=forecast["date"], y=forecast["lower"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor=band_color, name="95% band", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=forecast["date"], y=forecast[value_column], mode="lines+markers", name="Forecast", line=dict(color=actual_color, width=2, dash="dash")))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title=yaxis_title, hovermode="x unified")
    return fig


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

def apply_app_style() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
        :root {
            --bg-soft: #edf4f7;
            --card: rgba(255, 255, 255, 0.88);
            --card-strong: rgba(255, 255, 255, 0.96);
            --border: rgba(255, 255, 255, 0.0);
            --ink: #0f172a;
            --muted: #526071;
            --accent: #0f766e;
            --accent-dark: #115e59;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(14, 165, 233, 0.12), transparent 30%),
                radial-gradient(circle at top right, rgba(16, 185, 129, 0.11), transparent 24%),
                linear-gradient(180deg, #f8fbfc 0%, var(--bg-soft) 100%);
            color: var(--ink);
            font-family: "IBM Plex Sans", sans-serif;
        }
        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, var(--card-strong) 0%, rgba(248, 251, 252, 0.98) 100%);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 0.85rem 1rem;
            box-shadow: 0 16px 38px rgba(15, 23, 42, 0.07);
        }
        .block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1240px; }
        [data-testid="stSidebar"] > div:first-child {
            background: linear-gradient(180deg, rgba(255,255,255,0.94) 0%, rgba(242,247,249,0.98) 100%);
        }
        h1, h2, h3 { font-family: "Space Grotesk", sans-serif; color: var(--ink); letter-spacing: -0.03em; }
        h2 { font-size: 1.45rem; margin-top: 0.5rem; }
        h3 { font-size: 1.15rem; }
        p, label, [data-testid="stCaptionContainer"] { color: var(--muted); }
        div[data-testid="stMetricLabel"] { color: var(--muted); }
        div[data-testid="stPlotlyChart"], div[data-testid="stDataFrame"] {
            background: var(--card);
            border-radius: 20px;
            padding: 0.75rem;
            box-shadow: 0 16px 34px rgba(15, 23, 42, 0.05);
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: linear-gradient(180deg, rgba(255,255,255,0.76) 0%, rgba(248,251,252,0.92) 100%);
            border-radius: 24px;
            padding: 0.4rem 0.45rem;
            box-shadow: 0 18px 38px rgba(15, 23, 42, 0.06);
        }
        div[data-testid="stButton"] > button {
            border-radius: 999px;
            background: linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(242,251,250,0.98) 100%);
            color: var(--accent-dark);
            font-weight: 600;
            box-shadow: 0 10px 22px rgba(15, 23, 42, 0.05);
        }
        .fitbit-hero {
            background:
                radial-gradient(circle at top right, rgba(16, 185, 129, 0.16), transparent 26%),
                linear-gradient(135deg, rgba(255,255,255,0.96) 0%, rgba(244,250,251,0.98) 100%);
            border-radius: 28px;
            padding: 1.35rem 1.45rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 20px 44px rgba(15, 23, 42, 0.07);
        }
        .fitbit-hero__top { display: flex; align-items: center; justify-content: space-between; gap: 1rem; flex-wrap: wrap; }
        .fitbit-brand { display: flex; align-items: center; gap: 1rem; }
        .fitbit-logo { position: relative; width: 48px; height: 48px; border-radius: 16px; background: rgba(255,255,255,0.9); box-shadow: 0 12px 24px rgba(15, 23, 42, 0.06); }
        .fitbit-logo span { position: absolute; width: 7px; height: 7px; border-radius: 999px; background: linear-gradient(180deg, rgba(45,212,191,1) 0%, rgba(15,118,110,0.95) 100%); }
        .fitbit-logo span:nth-child(1){left:20px;top:7px;} .fitbit-logo span:nth-child(2){left:14px;top:15px;} .fitbit-logo span:nth-child(3){left:26px;top:15px;}
        .fitbit-logo span:nth-child(4){left:8px;top:23px;} .fitbit-logo span:nth-child(5){left:20px;top:23px;} .fitbit-logo span:nth-child(6){left:32px;top:23px;}
        .fitbit-logo span:nth-child(7){left:14px;top:31px;} .fitbit-logo span:nth-child(8){left:26px;top:31px;} .fitbit-logo span:nth-child(9){left:20px;top:39px;}
        .fitbit-logo span:nth-child(10){left:38px;top:23px;opacity:0.55;} .fitbit-logo span:nth-child(11){left:2px;top:23px;opacity:0.55;}
        .fitbit-hero__eyebrow { margin: 0 0 0.35rem 0; font-size: 0.78rem; letter-spacing: 0.16em; text-transform: uppercase; color: var(--accent); font-weight: 700; }
        .fitbit-hero h1 { margin: 0; font-size: 2.35rem; line-height: 1.02; }
        .fitbit-hero__subtitle { margin: 0.65rem 0 0 0; font-size: 1rem; color: var(--muted); }
        .fitbit-chip-row { display: flex; gap: 0.55rem; flex-wrap: wrap; align-items: center; }
        .fitbit-chip { border-radius: 999px; padding: 0.45rem 0.8rem; background: rgba(255,255,255,0.72); color: var(--ink); font-size: 0.86rem; font-weight: 600; box-shadow: 0 8px 18px rgba(15,23,42,0.04); }
        .fitbit-chip--muted { color: var(--muted); }
        .fitbit-empty { background: linear-gradient(180deg, rgba(255,255,255,0.94) 0%, rgba(246,249,250,0.98) 100%); border-radius: 18px; padding: 1rem 1.1rem; color: var(--muted); margin: 0.35rem 0 0.4rem 0; }
        .fitbit-empty p { margin: 0; }
        details[data-testid="stExpander"] { background: rgba(255,255,255,0.72); border-radius: 18px; }
        details[data-testid="stExpander"] summary { font-weight: 600; color: var(--ink); }

        /* Credentials form */
        .creds-card {
            max-width: 520px;
            margin: 3rem auto;
            background: white;
            border-radius: 24px;
            padding: 2rem 2.25rem;
            box-shadow: 0 24px 56px rgba(15,23,42,0.10);
        }
        .creds-card h2 { margin-top: 0; }
        .creds-hint {
            background: #f0fdf9;
            border-left: 3px solid #0f766e;
            border-radius: 0 8px 8px 0;
            padding: 0.75rem 1rem;
            font-size: 0.88rem;
            color: #0f766e;
            margin-bottom: 1.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Credentials setup UI
# ---------------------------------------------------------------------------

def render_credentials_setup() -> None:
    """Full-page credentials form shown when no config is present."""
    st.markdown(
        """
        <div class="creds-card">
            <p class="fitbit-hero__eyebrow">Setup</p>
            <h2>Connect your Fitbit</h2>
            <div class="creds-hint">
                You need a free Fitbit developer app to get your Client ID and Secret.<br>
                Create one at <strong>dev.fitbit.com → Manage → Register An App</strong>.<br>
                Set the <em>OAuth 2.0 Application Type</em> to <strong>Personal</strong> and
                the <em>Redirect URL</em> to this app's URL.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("fitbit_credentials_form"):
        col1, col2 = st.columns(2)
        client_id = col1.text_input("Fitbit Client ID", placeholder="23ABCD")
        client_secret = col2.text_input("Fitbit Client Secret", type="password", placeholder="abc123...")
        groq_key = st.text_input(
            "Groq API Key (optional — for AI insights)",
            type="password",
            placeholder="gsk_...",
        )
        submitted = st.form_submit_button("Save and continue", use_container_width=True)

    if submitted:
        if not client_id or not client_secret:
            st.error("Client ID and Client Secret are required.")
            return
        st.session_state["fitbit_client_id"] = client_id.strip()
        st.session_state["fitbit_client_secret"] = client_secret.strip()
        # redirect_uri comes from secrets automatically via load_config()
        if groq_key:
            st.session_state["groq_api_key"] = groq_key.strip()
        st.rerun()

def render_credentials_sidebar_editor() -> None:
    """Sidebar expander to update credentials mid-session."""
    with st.sidebar.expander("Edit credentials"):
        with st.form("update_creds_form"):
            client_id = st.text_input(
                "Client ID",
                value=st.session_state.get("fitbit_client_id", ""),
            )
            client_secret = st.text_input(
                "Client Secret",
                type="password",
                value=st.session_state.get("fitbit_client_secret", ""),
            )
            redirect_uri = st.text_input(
                "Redirect URI",
                value=st.session_state.get("fitbit_redirect_uri", ""),
            )
            groq_key = st.text_input(
                "Groq API Key",
                type="password",
                value=st.session_state.get("groq_api_key", ""),
            )
            groq_model = st.text_input(
                "Groq model",
                value=st.session_state.get("groq_model", DEFAULT_GROQ_MODEL),
            )
            if st.form_submit_button("Update", use_container_width=True):
                st.session_state["fitbit_client_id"] = client_id.strip()
                st.session_state["fitbit_client_secret"] = client_secret.strip()
                st.session_state["fitbit_redirect_uri"] = redirect_uri.strip()
                if groq_key:
                    st.session_state["groq_api_key"] = groq_key.strip()
                if groq_model:
                    st.session_state["groq_model"] = groq_model.strip()
                clear_api_cache()
                clear_token_bundle()
                st.rerun()

        if st.button("Clear all credentials", use_container_width=True):
            for key in ("fitbit_client_id", "fitbit_client_secret", "fitbit_redirect_uri",
                        "groq_api_key", "groq_model"):
                st.session_state.pop(key, None)
            clear_api_cache()
            clear_token_bundle()
            st.rerun()


# ---------------------------------------------------------------------------
# Dashboard render functions
# ---------------------------------------------------------------------------

def render_empty_state(message: str) -> None:
    st.markdown(
        f'<div class="fitbit-empty"><p>{message}</p></div>',
        unsafe_allow_html=True,
    )

def render_connection_panel(config: FitbitConfig, token_bundle: dict[str, Any] | None) -> None:
    st.sidebar.header("Fitbit connection")
    auth_url = build_authorize_url(config)
    if token_bundle:
        st.sidebar.success("Connected")
        if st.sidebar.button("Disconnect Fitbit"):
            clear_api_cache()
            clear_token_bundle()
            clear_auth_query_params()
            st.rerun()
    else:
        st.sidebar.link_button("Connect Fitbit", auth_url, use_container_width=True)
        html = """
<div style="max-width:560px;margin:5rem auto;background:white;border-radius:24px;padding:2.5rem 2.75rem;box-shadow:0 24px 56px rgba(15,23,42,0.10);text-align:center;">
<div style="font-size:2.5rem;margin-bottom:1rem;">💪</div>
<h2 style="margin:0 0 0.5rem 0;font-size:1.6rem;">Welcome to your Health Dashboard</h2>
<p style="color:#526071;margin-bottom:2rem;font-size:1rem;line-height:1.6;">Your personal Fitbit analytics — steps, sleep, heart rate, and AI-powered insights, all in one place.</p>
<div style="text-align:left;background:#f8fbfc;border-radius:16px;padding:1.25rem 1.5rem;margin-bottom:1.5rem;">
<p style="margin:0 0 0.75rem 0;font-weight:600;color:#0f172a;font-size:0.95rem;">Get started in 2 steps:</p>
<div style="display:flex;align-items:flex-start;gap:0.75rem;margin-bottom:0.75rem;">
<div style="min-width:28px;height:28px;background:#0f766e;color:white;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:0.8rem;font-weight:700;">1</div>
<p style="margin:0;color:#526071;font-size:0.92rem;padding-top:4px;">Open the <strong style="color:#0f172a;">sidebar on the left</strong> and enter your Fitbit Client ID and Client Secret in the <strong style="color:#0f172a;">Edit credentials</strong> panel. Don't have them yet? See the <a href="https://github.com/vasanthbhaskara/fitbit-dashboard#how-to-use-the-live-app" target="_blank" style="color:#0f766e;">setup guide</a>.</p>
</div>
<div style="display:flex;align-items:flex-start;gap:0.75rem;">
<div style="min-width:28px;height:28px;background:#0f766e;color:white;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:0.8rem;font-weight:700;">2</div>
<p style="margin:0;color:#526071;font-size:0.92rem;padding-top:4px;">Click the <strong style="color:#0f172a;">Connect Fitbit</strong> button in the sidebar to authorise access to your data. You will be redirected to Fitbit and back automatically.</p>
</div>
</div>
<p style="margin:0;font-size:0.8rem;color:#94a3b8;">Your credentials are stored only in your browser session and never saved to any database.</p>
</div>
"""
        st.markdown(html, unsafe_allow_html=True)
        st.stop()


def render_header(
    profile: dict[str, Any],
    selected_date: date,
    start_date: date,
    end_date: date,
    connected: bool,
) -> None:
    full_name = profile.get("fullName") or "Fitbit Dashboard"
    member_since = profile.get("memberSince")
    subtitle = f"Live Fitbit metrics for {full_name}"
    if member_since:
        subtitle += f" • member since {member_since}"
    connection_label = "Connected" if connected else "Not connected"
    st.markdown(
        f"""
        <section class="fitbit-hero">
            <div class="fitbit-hero__top">
                <div class="fitbit-brand">
                    <div class="fitbit-logo" aria-hidden="true">
                        <span></span><span></span><span></span>
                        <span></span><span></span><span></span>
                        <span></span><span></span><span></span>
                        <span></span><span></span>
                    </div>
                    <div>
                        <p class="fitbit-hero__eyebrow">Fitbit dashboard</p>
                        <h1>{full_name}</h1>
                    </div>
                </div>
                <div class="fitbit-chip-row">
                    <span class="fitbit-chip">{format_date_heading(selected_date)}</span>
                    <span class="fitbit-chip fitbit-chip--muted">{format_date_range_label(start_date, end_date)}</span>
                    <span class="fitbit-chip">{connection_label}</span>
                </div>
            </div>
            <p class="fitbit-hero__subtitle">{subtitle}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_kpis(
    steps_daily: pd.DataFrame,
    heart_daily: pd.DataFrame,
    sleep_frame: pd.DataFrame,
) -> None:
    total_steps = int(steps_daily["steps"].sum()) if not steps_daily.empty else 0
    avg_steps = int(round(steps_daily["steps"].mean())) if not steps_daily.empty else 0
    avg_sleep_hours = round(float(sleep_frame["duration_hours"].mean()), 2) if not sleep_frame.empty else 0.0
    latest_resting_hr = (
        int(heart_daily["resting_heart_rate"].dropna().iloc[-1])
        if not heart_daily.dropna(subset=["resting_heart_rate"]).empty
        else 0
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Window steps", f"{total_steps:,}")
    col2.metric("Average daily steps", f"{avg_steps:,}")
    col3.metric("Resting heart rate", f"{latest_resting_hr} bpm" if latest_resting_hr else "No data")
    col4.metric("Average sleep", f"{avg_sleep_hours:.2f} h")


def render_selected_day_section(
    steps_daily: pd.DataFrame,
    intraday_steps: pd.DataFrame,
    heart_daily: pd.DataFrame,
    sleep_frame: pd.DataFrame,
    selected_date: date,
) -> None:
    st.subheader(format_date_heading(selected_date))
    snapshot = build_selected_day_snapshot(
        steps_daily=steps_daily,
        intraday_steps=intraday_steps,
        heart_daily=heart_daily,
        sleep_frame=sleep_frame,
        temperature_frame=pd.DataFrame(columns=["date", "temperature_variation"]),
        respiratory_frame=pd.DataFrame(columns=["date", "respiratory_rate"]),
        oxygen_frame=pd.DataFrame(columns=["date", "oxygen_saturation_avg"]),
        cardio_frame=pd.DataFrame(columns=["date", "cardio_fitness"]),
        selected_date=selected_date,
    )
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Steps", metric_display(snapshot["steps"], lambda v: f"{int(v):,}", "No step total"))
    col2.metric("Resting HR", metric_display(snapshot["resting_heart_rate"], lambda v: f"{int(v)} bpm", "No resting HR"))
    col3.metric("Sleep", metric_display(snapshot["sleep_hours"], lambda v: f"{float(v):.2f} h", "No sleep log"))
    col4.metric("Sleep efficiency", metric_display(snapshot["sleep_efficiency"], lambda v: f"{int(v)}%", "No data"))
    col5.metric("Bedtime", format_bedtime_display(snapshot["bedtime"]))


def render_statistical_insights_section(
    steps_daily: pd.DataFrame,
    heart_daily: pd.DataFrame,
    sleep_frame: pd.DataFrame,
    temperature_frame: pd.DataFrame,
    respiratory_frame: pd.DataFrame,
    oxygen_frame: pd.DataFrame,
    cardio_frame: pd.DataFrame,
    selected_date: date,
) -> None:
    st.subheader("Statistical insights")
    insights = build_statistical_insights(
        steps_daily, heart_daily, sleep_frame,
        temperature_frame, respiratory_frame, oxygen_frame, cardio_frame,
        selected_date,
    )
    steps_forecast = linear_forecast(steps_daily, "steps")
    resting_hr_forecast = linear_forecast(
        heart_daily.dropna(subset=["resting_heart_rate"]), "resting_heart_rate"
    )
    merged = merge_daily_metrics(
        steps_daily, heart_daily, sleep_frame,
        temperature_frame=temperature_frame,
        respiratory_frame=respiratory_frame,
        oxygen_frame=oxygen_frame,
        cardio_frame=cardio_frame,
    )
    correlation = None
    if {"steps", "duration_hours"}.issubset(merged.columns):
        joined = merged[["steps", "duration_hours"]].dropna()
        if len(joined) >= 3:
            correlation = float(joined.corr().iloc[0, 1])

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Next-day step forecast",
        f"{steps_forecast['next_value']:,.0f}" if steps_forecast and steps_forecast["next_value"] is not None else "N/A",
        delta=f"R² {steps_forecast['r_squared']:.2f}" if steps_forecast else None,
    )
    col2.metric(
        "Next-day resting HR",
        f"{resting_hr_forecast['next_value']:.1f} bpm" if resting_hr_forecast and resting_hr_forecast["next_value"] is not None else "N/A",
        delta=f"R² {resting_hr_forecast['r_squared']:.2f}" if resting_hr_forecast else None,
    )
    col3.metric(
        "Sleep/steps correlation",
        f"{correlation:.2f}" if correlation is not None else "N/A",
        delta=f"{correlation_strength(correlation)}" if correlation is not None else None,
    )
    st.markdown("\n".join(f"- {insight}" for insight in insights))


def render_prediction_section(
    steps_daily: pd.DataFrame, heart_daily: pd.DataFrame
) -> None:
    st.subheader("Predictions")
    steps_forecast = linear_forecast(steps_daily, "steps")
    resting_hr_forecast = linear_forecast(
        heart_daily.dropna(subset=["resting_heart_rate"]), "resting_heart_rate"
    )
    if not steps_forecast:
        st.info("Need at least 5 daily step observations for the regression forecast.")
    else:
        st.plotly_chart(
            build_forecast_figure(steps_forecast, value_column="steps", title="7-day step forecast", yaxis_title="Steps", actual_color="#0f766e", band_color="rgba(15,118,110,0.14)"),
            use_container_width=True,
        )
    if not resting_hr_forecast:
        st.info("Need at least 5 resting heart-rate observations for the regression forecast.")
    else:
        st.plotly_chart(
            build_forecast_figure(resting_hr_forecast, value_column="resting_heart_rate", title="7-day resting heart-rate forecast", yaxis_title="BPM", actual_color="#b91c1c", band_color="rgba(185,28,28,0.14)"),
            use_container_width=True,
        )


def render_steps_section(
    steps_daily: pd.DataFrame,
    intraday_steps: pd.DataFrame,
    selected_date: date,
) -> None:
    st.subheader("Steps")
    if steps_daily.empty:
        render_empty_state("No steps data returned for this date range.")
        return

    daily_steps = steps_daily.copy().sort_values("date")
    daily_steps["rolling_7"] = daily_steps["steps"].rolling(window=7, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=daily_steps["date"], y=daily_steps["steps"], name="Daily steps", marker_color="#99f6e4"))
    fig.add_trace(go.Scatter(x=daily_steps["date"], y=daily_steps["rolling_7"], mode="lines+markers", name="7-day average", line=dict(color="#0f766e", width=3)))
    fig.add_hline(y=10_000, line_dash="dot", line_color="#1d4ed8", annotation_text="10k goal")
    style_figure(fig, title="Daily steps with rolling trend", xaxis_title="Date", yaxis_title="Steps")
    st.plotly_chart(fig, use_container_width=True)

    if intraday_steps.empty:
        return

    hourly_steps = aggregate_intraday_steps_by_hour(intraday_steps)
    cumulative = intraday_steps.copy()
    cumulative["cumulative_steps"] = cumulative["steps"].cumsum()

    fig = px.bar(
        hourly_steps, x="hour_label", y="steps",
        title=f"Hourly steps on {format_date_compact(selected_date)}",
        labels={"hour_label": "Hour", "steps": "Steps"},
        color="steps", color_continuous_scale=["#a7f3d0", "#047857"],
    )
    fig.update_layout(coloraxis_showscale=False)
    style_figure(fig, title=f"Hourly steps on {format_date_compact(selected_date)}", xaxis_title="Hour", yaxis_title="Steps")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.area(
        cumulative, x="timestamp", y="cumulative_steps",
        title=f"Cumulative steps on {format_date_compact(selected_date)}",
        labels={"timestamp": "Time", "cumulative_steps": "Cumulative steps"},
    )
    fig.update_traces(line_color="#0f766e", fillcolor="rgba(15,118,110,0.18)")
    style_figure(fig, title=f"Cumulative steps on {format_date_compact(selected_date)}", xaxis_title="Time", yaxis_title="Cumulative steps")
    st.plotly_chart(fig, use_container_width=True)


def render_activity_patterns_section(
    intraday_steps_window: pd.DataFrame,
    steps_daily: pd.DataFrame,
    sleep_frame: pd.DataFrame,
    heart_daily: pd.DataFrame,
) -> None:
    st.subheader("Activity patterns")

    heatmap_frame = build_activity_heatmap_frame(intraday_steps_window)
    if not heatmap_frame.empty:
        day_order = heatmap_frame[["date", "day_label"]].drop_duplicates().sort_values("date")
        heatmap_matrix = (
            heatmap_frame.pivot(index="day_label", columns="hour", values="steps")
            .reindex(index=day_order["day_label"], columns=list(range(24)))
            .fillna(0)
        )
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_matrix.values,
            x=[format_clock_tick(h) for h in heatmap_matrix.columns],
            y=heatmap_matrix.index.tolist(),
            colorscale=[[0.0, "#ecfeff"], [0.25, "#a5f3fc"], [0.5, "#22d3ee"], [0.75, "#0891b2"], [1.0, "#164e63"]],
            hovertemplate="Day: %{y}<br>Hour: %{x}<br>Steps: %{z:.0f}<extra></extra>",
        ))
        fig.update_layout(
            title=dict(text="Hourly activity heatmap", x=0.02, xanchor="left"),
            xaxis_title="Hour of day", yaxis_title="Date", height=410,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=18, r=18, t=62, b=18),
            font=dict(family=PLOT_FONT_FAMILY, color="#0f172a"),
        )
        st.plotly_chart(fig, use_container_width=True)

    consistency_frame = build_consistency_score_frame(steps_daily, sleep_frame, heart_daily)
    if consistency_frame.empty:
        render_empty_state("Not enough daily data is available yet to score consistency.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=consistency_frame["date"], y=consistency_frame["consistency_score"], mode="lines+markers", name="Consistency score", line=dict(color="#0f766e", width=3)))
    fig.add_trace(go.Bar(x=consistency_frame["date"], y=consistency_frame["steps_score"], name="Steps", marker_color="rgba(15,118,110,0.18)", hovertemplate="Date: %{x|%d %b %Y}<br>Steps score: %{y:.1f}<extra></extra>"))
    fig.add_hline(y=70, line_dash="dot", line_color="#0f766e", annotation_text="Solid routine")
    style_figure(fig, title="Consistency score trend", xaxis_title="Date", yaxis_title="Score")
    fig.update_layout(barmode="overlay")
    st.plotly_chart(fig, use_container_width=True)

    component_labels = {"steps_score": "Steps", "sleep_score": "Sleep", "efficiency_score": "Efficiency", "bedtime_score": "Bedtime", "recovery_score": "Recovery"}
    component_colors = {"steps_score": "#0f766e", "sleep_score": "#2563eb", "efficiency_score": "#7c3aed", "bedtime_score": "#d97706", "recovery_score": "#dc2626"}
    component_chart = go.Figure()
    for col_name in component_labels:
        component_chart.add_trace(go.Bar(x=consistency_frame["date"], y=consistency_frame[col_name], name=component_labels[col_name], marker_color=component_colors[col_name]))
    style_figure(component_chart, title="Consistency score components by day", xaxis_title="Date", yaxis_title="Points")
    component_chart.update_layout(barmode="stack")
    st.plotly_chart(component_chart, use_container_width=True)

    score_display = consistency_frame.copy()
    score_display["date"] = score_display["date"].dt.date
    score_display["bedtime_display"] = score_display["bedtime_display"].map(
        lambda v: format_bedtime_hour_value(v) if pd.notna(v) else "No bedtime"
    )
    score_display = score_display.rename(columns={
        "date": "Date", "consistency_score": "Total", "steps_score": "Steps",
        "sleep_score": "Sleep", "efficiency_score": "Efficiency",
        "bedtime_score": "Bedtime", "recovery_score": "Recovery", "bedtime_display": "Clock time",
    })
    st.dataframe(
        score_display, use_container_width=True, hide_index=True,
        column_config={
            "Total": st.column_config.NumberColumn("Total", format="%.1f"),
            "Steps": st.column_config.NumberColumn("Steps", format="%.1f"),
            "Sleep": st.column_config.NumberColumn("Sleep", format="%.1f"),
            "Efficiency": st.column_config.NumberColumn("Efficiency", format="%.1f"),
            "Bedtime": st.column_config.NumberColumn("Bedtime", format="%.1f"),
            "Recovery": st.column_config.NumberColumn("Recovery", format="%.1f"),
        },
    )
    with st.expander("Consistency score formula", expanded=False):
        with st.container(border=True):
            st.markdown("**Consistency Score Formula**")
            st.latex(r"S = S_{\mathrm{steps}} + S_{\mathrm{sleep}} + S_{\mathrm{eff}} + S_{\mathrm{bed}} + S_{\mathrm{rec}}")
            st.latex(r"S_{\mathrm{steps}} = 45 \cdot \min\left(\frac{\mathrm{steps}}{10000}, 1\right)")
            st.latex(r"S_{\mathrm{sleep}} = 25 \cdot \min\left(\frac{\mathrm{sleep\ hours}}{7}, 1\right)")
            st.latex(r"S_{\mathrm{eff}} = 15 \cdot \min\left(\frac{\mathrm{sleep\ efficiency}}{100}, 1\right)")
            st.latex(r"S_{\mathrm{bed}} = 10 \cdot \max\left(0,\ 1 - \frac{\left|\mathrm{bedtime} - \mathrm{median\ bedtime}\right|}{3}\right)")
            st.latex(r"S_{\mathrm{rec}} = 5 \cdot \max\left(0,\ 1 - \frac{\max(\mathrm{RHR} - \mathrm{median\ RHR},\ 0)}{8}\right)")


def render_heart_section(heart_daily: pd.DataFrame) -> None:
    st.subheader("Heart rate")
    resting_daily = heart_daily.dropna(subset=["resting_heart_rate"])
    if resting_daily.empty:
        render_empty_state("No daily resting heart-rate history returned for this window.")
        return

    daily_view = resting_daily.copy()
    daily_view["rolling_3"] = daily_view["resting_heart_rate"].rolling(window=3, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_view["date"], y=daily_view["resting_heart_rate"], mode="lines+markers", name="Resting HR", line=dict(color="#ef4444", width=2)))
    fig.add_trace(go.Scatter(x=daily_view["date"], y=daily_view["rolling_3"], mode="lines", name="3-day average", line=dict(color="#7f1d1d", width=2, dash="dot")))
    style_figure(fig, title="Daily resting heart-rate trend", xaxis_title="Date", yaxis_title="BPM")
    st.plotly_chart(fig, use_container_width=True)

def render_intraday_section(
    config: FitbitConfig,
    token_bundle: dict[str, Any],
    intraday_steps: pd.DataFrame,
    selected_date: date,
) -> None:
    st.subheader(f"Intraday detail — {format_date_compact(selected_date)}")

    intraday_hr = fetch_intraday_heart_rate(config, token_bundle, selected_date)

    # --- Heart rate zones reference lines ---
    ZONES = {
        "Fat burn": (114, "#f59e0b"),
        "Cardio":   (133, "#f97316"),
        "Peak":     (152, "#ef4444"),
    }

    # Heart rate over the day
    if not intraday_hr.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=intraday_hr["timestamp"],
            y=intraday_hr["heart_rate"],
            mode="lines",
            name="Heart rate",
            line=dict(color="#ef4444", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(239,68,68,0.08)",
        ))
        for zone_name, (bpm, color) in ZONES.items():
            fig.add_hline(
                y=bpm,
                line_dash="dot",
                line_color=color,
                annotation_text=zone_name,
                annotation_position="right",
            )
        style_figure(fig, title="Heart rate throughout the day", xaxis_title="Time", yaxis_title="BPM")
        st.plotly_chart(fig, use_container_width=True)
    else:
        render_empty_state("No intraday heart rate data available for this date.")

    # Heart rate distribution histogram
    if not intraday_hr.empty:
        fig = px.histogram(
            intraday_hr,
            x="heart_rate",
            nbins=40,
            title="Heart rate distribution",
            labels={"heart_rate": "BPM", "count": "Minutes"},
            color_discrete_sequence=["#ef4444"],
        )
        style_figure(fig, title="Heart rate distribution", xaxis_title="BPM", yaxis_title="Minutes")
        st.plotly_chart(fig, use_container_width=True)

    # Steps vs heart rate overlay
    if not intraday_hr.empty and not intraday_steps.empty:
        # Resample both to 5-min buckets for cleaner overlay
        hr = intraday_hr.set_index("timestamp").resample("5min").mean().reset_index()
        steps = intraday_steps.set_index("timestamp").resample("5min").sum().reset_index()
        merged = hr.merge(steps, on="timestamp", how="inner")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=merged["timestamp"],
            y=merged["steps"],
            name="Steps",
            marker_color="rgba(15,118,110,0.35)",
            yaxis="y2",
        ))
        fig.add_trace(go.Scatter(
            x=merged["timestamp"],
            y=merged["heart_rate"],
            mode="lines",
            name="Heart rate",
            line=dict(color="#ef4444", width=2),
        ))
        fig.update_layout(
            yaxis=dict(title="BPM", tickfont=dict(color="#526071")),
            yaxis2=dict(title="Steps", overlaying="y", side="right", tickfont=dict(color="#526071")),
            hovermode="x unified",
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0)",
            margin=dict(l=18, r=18, t=62, b=18),
            font=dict(family=PLOT_FONT_FAMILY, color="#0f172a"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            title=dict(text="Steps vs heart rate", x=0.02, xanchor="left"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Hourly heart rate summary bar
    if not intraday_hr.empty:
        hourly_hr = intraday_hr.copy()
        hourly_hr["hour"] = hourly_hr["timestamp"].dt.hour
        hourly_avg = hourly_hr.groupby("hour")["heart_rate"].agg(["mean", "min", "max"]).reset_index()
        hourly_avg["hour_label"] = hourly_avg["hour"].map(lambda h: f"{int(h):02d}:00")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=hourly_avg["hour_label"],
            y=hourly_avg["mean"],
            name="Avg HR",
            marker_color="#ef4444",
            error_y=dict(
                type="data",
                symmetric=False,
                array=(hourly_avg["max"] - hourly_avg["mean"]).tolist(),
                arrayminus=(hourly_avg["mean"] - hourly_avg["min"]).tolist(),
                color="#7f1d1d",
            ),
        ))
        style_figure(fig, title="Hourly heart rate (avg with min/max range)", xaxis_title="Hour", yaxis_title="BPM")
        st.plotly_chart(fig, use_container_width=True)

def render_sleep_section(sleep_frame: pd.DataFrame) -> None:
    st.subheader("Sleep")
    if sleep_frame.empty:
        render_empty_state("No sleep logs returned for this date range.")
        return

    stage_frame = sleep_frame.melt(
        id_vars="date",
        value_vars=["deep_hours", "light_hours", "rem_hours", "wake_hours"],
        var_name="stage", value_name="hours",
    )
    fig = px.bar(
        sleep_frame, x="date", y="duration_hours",
        labels={"date": "Date", "duration_hours": "Hours"},
        title="Sleep duration by night",
        color="efficiency", color_continuous_scale=["#bfdbfe", "#1d4ed8"],
    )
    fig.add_hline(y=7, line_dash="dot", line_color="#1d4ed8", annotation_text="7h target")
    fig.update_layout(coloraxis_colorbar_title="Efficiency")
    style_figure(fig, title="Sleep duration by night", xaxis_title="Date", yaxis_title="Hours")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(
        stage_frame, x="date", y="hours", color="stage",
        title="Sleep stages", labels={"date": "Date", "hours": "Hours", "stage": "Stage"},
        color_discrete_map={"deep_hours": "#1d4ed8", "light_hours": "#38bdf8", "rem_hours": "#f59e0b", "wake_hours": "#ef4444"},
    )
    style_figure(fig, title="Sleep stages", xaxis_title="Date", yaxis_title="Hours")
    fig.update_layout(barmode="stack")
    st.plotly_chart(fig, use_container_width=True)

    sleep_scatter = sleep_frame.copy()
    sleep_scatter["bedtime_plot_hour"] = normalize_bedtime_hours(sleep_scatter["start_time"])
    bedtime_tick_values, bedtime_range = bedtime_axis_config(sleep_scatter["bedtime_plot_hour"])
    fig = px.scatter(
        sleep_scatter.dropna(subset=["bedtime_plot_hour"]),
        x="date", y="bedtime_plot_hour", size="duration_hours", color="efficiency",
        title="Bedtime consistency", labels={"date": "Date", "bedtime_plot_hour": "Bedtime"},
        color_continuous_scale=["#bfdbfe", "#1d4ed8"],
    )
    style_figure(fig, title="Bedtime consistency", xaxis_title="Date", yaxis_title="Bedtime")
    fig.update_yaxes(
        tickmode="array", tickvals=bedtime_tick_values,
        ticktext=[format_clock_tick(v) for v in bedtime_tick_values],
        range=bedtime_range, title_text="Bedtime",
    )
    st.plotly_chart(fig, use_container_width=True)

    sleep_display = sleep_frame.copy()
    sleep_display["date"] = sleep_display["date"].dt.date
    sleep_display["start_time"] = sleep_display["start_time"].map(format_datetime_display)
    st.dataframe(
        sleep_display, use_container_width=True, hide_index=True,
        column_config={
            "date": "Date",
            "start_time": "Sleep started",
            "duration_hours": st.column_config.NumberColumn("Sleep hours", format="%.2f"),
            "time_in_bed_hours": st.column_config.NumberColumn("Time in bed", format="%.2f"),
            "efficiency": st.column_config.ProgressColumn("Efficiency", min_value=0, max_value=100),
        },
    )


def render_daily_optional_metric_chart(
    frame: pd.DataFrame,
    value_column: str,
    title: str,
    yaxis_title: str,
    color: str,
    empty_message: str,
) -> None:
    if frame.empty or frame[value_column].dropna().empty:
        render_empty_state(empty_message)
        return
    chart_frame = frame.dropna(subset=[value_column]).copy().sort_values("date")
    chart_frame["rolling_3"] = chart_frame[value_column].rolling(window=3, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_frame["date"], y=chart_frame[value_column], mode="lines+markers", name="Daily", line=dict(color=color, width=2)))
    fig.add_trace(go.Scatter(x=chart_frame["date"], y=chart_frame["rolling_3"], mode="lines", name="3-day average", line=dict(color=color, width=2, dash="dot")))
    style_figure(fig, title=title, xaxis_title="Date", yaxis_title=yaxis_title)
    st.plotly_chart(fig, use_container_width=True)


def render_ai_insights_section(
    llm_config: LLMConfig,
    steps_daily: pd.DataFrame,
    intraday_steps: pd.DataFrame,
    heart_daily: pd.DataFrame,
    sleep_frame: pd.DataFrame,
    temperature_frame: pd.DataFrame,
    respiratory_frame: pd.DataFrame,
    oxygen_frame: pd.DataFrame,
    cardio_frame: pd.DataFrame,
    selected_date: date,
    lookback_days: int,
) -> None:
    st.subheader("AI insights")
    st.caption("Optional LLM summary based on aggregated Fitbit metrics. Not medical advice.")

    if not llm_config.is_configured:
        render_empty_state(
            "Add a Groq API key in the 'Edit credentials' panel in the sidebar to enable AI insights. "
            f"The default model is `{DEFAULT_GROQ_MODEL}` (free at console.groq.com)."
        )
        return

    insight_scope = st.radio(
        "Insight scope",
        options=["window_summary", "selected_day"],
        format_func=lambda v: "Window summary" if v == "window_summary" else f"Selected day: {format_date_compact(selected_date)}",
        horizontal=True,
    )

    cache_key = json.dumps(
        {
            "selected_date": selected_date.isoformat(),
            "lookback_days": lookback_days,
            "insight_scope": insight_scope,
            "steps_points": len(steps_daily),
            "heart_points": len(heart_daily),
            "sleep_points": len(sleep_frame),
            "resting_heart_rate": (
                int(heart_daily["resting_heart_rate"].dropna().iloc[-1])
                if not heart_daily.dropna(subset=["resting_heart_rate"]).empty
                else None
            ),
            "temperature_points": len(temperature_frame),
            "respiratory_points": len(respiratory_frame),
            "oxygen_points": len(oxygen_frame),
            "cardio_points": len(cardio_frame),
        },
        sort_keys=True,
    )
    cached_insight = st.session_state.get("fitbit_ai_insights")

    col1, col2 = st.columns([1, 3])
    with col1:
        generate = st.button("Generate insights", use_container_width=True)
    with col2:
        regenerate = st.button("Regenerate", use_container_width=True)

    if regenerate:
        clear_ai_insights_cache()
        cached_insight = None

    if generate or regenerate:
        try:
            with st.spinner("Generating AI insights..."):
                insight_text = generate_ai_insights(
                    llm_config=llm_config,
                    steps_daily=steps_daily,
                    intraday_steps=intraday_steps,
                    heart_daily=heart_daily,
                    sleep_frame=sleep_frame,
                    temperature_frame=temperature_frame,
                    respiratory_frame=respiratory_frame,
                    oxygen_frame=oxygen_frame,
                    cardio_frame=cardio_frame,
                    selected_date=selected_date,
                    lookback_days=lookback_days,
                    insight_scope=insight_scope,
                )
            st.session_state["fitbit_ai_insights"] = {"cache_key": cache_key, "content": insight_text}
            cached_insight = st.session_state["fitbit_ai_insights"]
        except (requests.RequestException, ValueError) as exc:
            st.error(f"Unable to generate AI insights: {exc}")
            return

    if cached_insight and cached_insight.get("cache_key") == cache_key:
        st.markdown(cached_insight["content"])
    else:
        render_empty_state("Click `Generate insights` to create a short summary of your current dashboard data.")

def strip_url_fragment() -> None:
    st.components.v1.html(
        """
        <script>
        (function() {
            var parent = window.parent || window;
            if (parent.location.hash && parent.location.hash !== '') {
                parent.location.replace(
                    parent.location.href.split('#')[0]
                );
            }
        })();
        </script>
        """,
        height=0,
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    bootstrap_environment()
    st.set_page_config(page_title="Fitbit Dashboard", page_icon=":heartbeat:", layout="wide")
    apply_app_style()

    config = load_config()
    llm_config = load_llm_config()

    # Restore credentials from OAuth state param if session was wiped on redirect
    if not config.is_configured:
        state = format_query_param("state")
        if state and format_query_param("code"):
            try:
                state_data = json.loads(base64.urlsafe_b64decode(state.encode()).decode())
                st.session_state["fitbit_client_id"] = state_data.get("client_id", "")
                st.session_state["fitbit_client_secret"] = state_data.get("client_secret", "")
                st.session_state["fitbit_redirect_uri"] = state_data.get("redirect_uri", "")
                if state_data.get("groq_api_key"):
                    st.session_state["groq_api_key"] = state_data.get("groq_api_key", "")
                config = load_config()
            except Exception:
                pass

    if not config.is_configured:
        render_credentials_setup()
        st.stop()

    render_credentials_sidebar_editor()
    handle_oauth_callback(config)

    try:
        token_bundle = get_active_token_bundle(config)
    except requests.RequestException as exc:
        st.error(f"Unable to refresh Fitbit access token: {format_request_error(exc)}")
        clear_token_bundle()
        st.stop()

    render_connection_panel(config, token_bundle)

    st.sidebar.header("Filters")
    lookback_days = st.sidebar.slider(
        "Lookback window (days)",
        min_value=LOOKBACK_MIN_DAYS,
        max_value=LOOKBACK_MAX_DAYS,
        value=DEFAULT_LOOKBACK_DAYS,
        key="lookback_days_v2",
        help="Capped to reduce Fitbit API quota pressure while keeping enough history for trends.",
    )
    selected_date = st.sidebar.date_input(
        "Date (select a day to view health details)",
        value=date.today(),
        max_value=date.today(),
    )
    if isinstance(selected_date, tuple):
        selected_date = selected_date[-1]

    if st.sidebar.button("Refresh Fitbit data", use_container_width=True):
        clear_api_cache()
        st.rerun()

    try:
        with st.spinner("Loading Fitbit profile and metrics..."):
            profile = fetch_profile(config, token_bundle)
            end_date = date.today()
            start_date = end_date - timedelta(days=lookback_days - 1)
            missing_optional_scopes_list = missing_scopes(token_bundle, OPTIONAL_HEALTH_SCOPES)

            steps_daily = fetch_daily_steps(config, token_bundle, start_date, end_date)
            intraday_steps = fetch_intraday_steps(config, token_bundle, selected_date)
            intraday_steps_window = fetch_intraday_steps_window(config, token_bundle, start_date, end_date)
            heart_daily = fetch_heart_rate_summary(config, token_bundle, start_date, end_date)
            sleep_frame = fetch_sleep_logs(config, token_bundle, start_date, end_date)
            temperature_frame = (
                fetch_temperature_summary(config, token_bundle, start_date, end_date)
                if "temperature" not in missing_optional_scopes_list
                else pd.DataFrame(columns=["date", "temperature_variation"])
            )
            respiratory_frame = (
                fetch_respiratory_rate_summary(config, token_bundle, start_date, end_date)
                if "respiratory_rate" not in missing_optional_scopes_list
                else pd.DataFrame(columns=["date", "respiratory_rate"])
            )
            oxygen_frame = (
                fetch_oxygen_saturation_summary(config, token_bundle, start_date, end_date)
                if "oxygen_saturation" not in missing_optional_scopes_list
                else pd.DataFrame(columns=["date", "oxygen_saturation_avg", "oxygen_saturation_min", "oxygen_saturation_max"])
            )
            cardio_frame = (
                fetch_cardio_fitness_summary(config, token_bundle, start_date, end_date)
                if "cardio_fitness" not in missing_optional_scopes_list
                else pd.DataFrame(columns=["date", "cardio_fitness"])
            )
    except requests.RequestException as exc:
        st.error(f"Fitbit API request failed: {format_request_error(exc)}")
        st.stop()

    render_header(profile, selected_date, start_date, end_date, connected=token_bundle is not None)

    with st.container(border=True):
        render_selected_day_section(steps_daily, intraday_steps, heart_daily, sleep_frame, selected_date)
    st.divider()
    with st.container(border=True):
        st.subheader(format_date_range_label(start_date, end_date))
        render_kpis(steps_daily, heart_daily, sleep_frame)
    st.divider()
    with st.container(border=True):
        render_ai_insights_section(
            llm_config=llm_config,
            steps_daily=steps_daily,
            intraday_steps=intraday_steps,
            heart_daily=heart_daily,
            sleep_frame=sleep_frame,
            temperature_frame=temperature_frame,
            respiratory_frame=respiratory_frame,
            oxygen_frame=oxygen_frame,
            cardio_frame=cardio_frame,
            selected_date=selected_date,
            lookback_days=lookback_days,
        )
    st.divider()
    with st.container(border=True):
        render_statistical_insights_section(
            steps_daily, heart_daily, sleep_frame,
            temperature_frame, respiratory_frame, oxygen_frame, cardio_frame,
            selected_date,
        )
    st.divider()
    with st.container(border=True):
        render_prediction_section(steps_daily, heart_daily)
    st.divider()
    with st.container(border=True):
        render_activity_patterns_section(intraday_steps_window, steps_daily, sleep_frame, heart_daily)
    st.divider()
    with st.container(border=True):
        render_steps_section(steps_daily, intraday_steps, selected_date)
    st.divider()
    with st.container(border=True):
        render_heart_section(heart_daily)
    st.divider()
    st.divider()
    with st.container(border=True):
        render_sleep_section(sleep_frame)
    st.divider()


if __name__ == "__main__":
    main()