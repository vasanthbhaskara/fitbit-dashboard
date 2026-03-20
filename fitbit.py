from __future__ import annotations

import base64
import json
import os
import secrets
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

AUTH_BASE_URL = "https://www.fitbit.com/oauth2/authorize"
TOKEN_URL = "https://api.fitbit.com/oauth2/token"
API_BASE_URL = "https://api.fitbit.com"
DEFAULT_SCOPE = "activity heartrate sleep profile"

PALETTE = {
    "steps": "#127475",
    "steps_fill": "#8DD8C7",
    "heart": "#C84B31",
    "heart_fill": "#F7C9C0",
    "sleep": "#355070",
    "sleep_fill": "#B9C6D8",
    "accent": "#E9C46A",
    "surface": "#FFFDF8",
    "surface_alt": "#F4F7FB",
}


class FitbitAuthError(RuntimeError):
    pass


class FitbitAPIError(RuntimeError):
    pass


@dataclass
class FitbitConfig:
    client_id: str
    client_secret: str
    redirect_uri: str
    scope: str
    token_path: Path


def load_config() -> FitbitConfig:
    load_dotenv()
    return FitbitConfig(
        client_id=os.getenv("FITBIT_CLIENT_ID", "").strip(),
        client_secret=os.getenv("FITBIT_CLIENT_SECRET", "").strip(),
        redirect_uri=os.getenv("FITBIT_REDIRECT_URI", "http://localhost:8501").strip(),
        scope=os.getenv("FITBIT_SCOPE", DEFAULT_SCOPE).strip() or DEFAULT_SCOPE,
        token_path=Path(os.getenv("FITBIT_TOKEN_PATH", ".fitbit_tokens.json")).expanduser(),
    )


def apply_styles() -> None:
    st.set_page_config(page_title="Fitbit Dashboard", page_icon="activity", layout="wide")
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background:
                radial-gradient(circle at top left, rgba(233, 196, 106, 0.18), transparent 24%),
                radial-gradient(circle at top right, rgba(18, 116, 117, 0.14), transparent 26%),
                linear-gradient(180deg, #faf4eb 0%, #fffdf8 32%, #f1f5fa 100%);
        }}
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}
        div[data-testid="stMetric"] {{
            background: rgba(255, 253, 248, 0.86);
            border: 1px solid rgba(53, 80, 112, 0.12);
            border-radius: 18px;
            padding: 0.8rem;
            box-shadow: 0 14px 30px rgba(53, 80, 112, 0.06);
        }}
        div[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #fffaf0 0%, #f7fbff 100%);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session(config: FitbitConfig) -> None:
    if "fitbit_tokens" not in st.session_state:
        st.session_state["fitbit_tokens"] = load_token_cache(config.token_path)
    if "fitbit_oauth_state" not in st.session_state:
        st.session_state["fitbit_oauth_state"] = secrets.token_urlsafe(24)


def load_token_cache(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def save_token_cache(path: Path, tokens: dict[str, Any]) -> None:
    path.write_text(json.dumps(tokens, indent=2))


def clear_token_cache(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def build_authorize_url(config: FitbitConfig) -> str:
    state = st.session_state["fitbit_oauth_state"]
    request = requests.Request(
        "GET",
        AUTH_BASE_URL,
        params={
            "client_id": config.client_id,
            "response_type": "code",
            "redirect_uri": config.redirect_uri,
            "scope": config.scope,
            "expires_in": "604800",
            "state": state,
        },
    )
    return request.prepare().url


def parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def token_expired(tokens: dict[str, Any]) -> bool:
    expires_at = parse_iso_datetime(tokens.get("expires_at"))
    if not expires_at:
        return True
    return datetime.now(timezone.utc) >= expires_at


def store_tokens(config: FitbitConfig, payload: dict[str, Any]) -> dict[str, Any]:
    token_data = dict(payload)
    expires_in = int(token_data.get("expires_in", 3600))
    token_data["expires_at"] = (
        datetime.now(timezone.utc) + timedelta(seconds=max(expires_in - 120, 60))
    ).isoformat()
    st.session_state["fitbit_tokens"] = token_data
    save_token_cache(config.token_path, token_data)
    return token_data


def build_token_headers(config: FitbitConfig) -> dict[str, str]:
    basic = base64.b64encode(
        f"{config.client_id}:{config.client_secret}".encode("utf-8")
    ).decode("utf-8")
    return {
        "Authorization": f"Basic {basic}",
        "Content-Type": "application/x-www-form-urlencoded",
    }


def raise_for_fitbit_error(response: requests.Response) -> None:
    if response.ok:
        return
    message = response.text
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict) and payload.get("errors"):
        message = "; ".join(error.get("message", "Unknown Fitbit error") for error in payload["errors"])
    raise FitbitAPIError(f"{response.status_code}: {message}")


def exchange_code_for_tokens(config: FitbitConfig, code: str) -> dict[str, Any]:
    response = requests.post(
        TOKEN_URL,
        headers=build_token_headers(config),
        data={
            "client_id": config.client_id,
            "grant_type": "authorization_code",
            "redirect_uri": config.redirect_uri,
            "code": code,
        },
        timeout=30,
    )
    raise_for_fitbit_error(response)
    return store_tokens(config, response.json())


def refresh_access_token(config: FitbitConfig, refresh_token: str) -> dict[str, Any]:
    response = requests.post(
        TOKEN_URL,
        headers=build_token_headers(config),
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        },
        timeout=30,
    )
    raise_for_fitbit_error(response)
    return store_tokens(config, response.json())


def handle_oauth_callback(config: FitbitConfig) -> None:
    code = st.query_params.get("code")
    returned_state = st.query_params.get("state")
    error = st.query_params.get("error")

    if error:
        st.error(f"Fitbit authorization failed: {error}")
        st.query_params.clear()
        return

    if not code:
        return

    expected_state = st.session_state.get("fitbit_oauth_state")
    if expected_state and returned_state and expected_state != returned_state:
        st.error("OAuth state mismatch. Start the Fitbit connection flow again.")
        st.query_params.clear()
        return

    with st.spinner("Exchanging Fitbit authorization code..."):
        try:
            exchange_code_for_tokens(config, code)
        except (FitbitAuthError, FitbitAPIError) as exc:
            st.error(f"Could not complete Fitbit login: {exc}")
            st.query_params.clear()
            return

    st.session_state["fitbit_oauth_state"] = secrets.token_urlsafe(24)
    st.query_params.clear()
    st.rerun()


def clear_connection(config: FitbitConfig) -> None:
    st.session_state["fitbit_tokens"] = None
    st.session_state["fitbit_oauth_state"] = secrets.token_urlsafe(24)
    clear_token_cache(config.token_path)
    st.cache_data.clear()


def ensure_access_token(config: FitbitConfig) -> str | None:
    tokens = st.session_state.get("fitbit_tokens")
    if not tokens:
        return None

    if token_expired(tokens):
        refresh_token = tokens.get("refresh_token")
        if not refresh_token:
            clear_connection(config)
            return None
        try:
            tokens = refresh_access_token(config, refresh_token)
        except FitbitAPIError as exc:
            clear_connection(config)
            raise FitbitAuthError(f"Token refresh failed: {exc}") from exc

    return tokens.get("access_token")


@st.cache_data(ttl=900, show_spinner=False)
def fitbit_get_cached(
    access_token: str,
    endpoint: str,
    params: tuple[tuple[str, str], ...] = (),
) -> dict[str, Any]:
    response = requests.get(
        f"{API_BASE_URL}{endpoint}",
        headers={"Authorization": f"Bearer {access_token}"},
        params=dict(params),
        timeout=30,
    )
    raise_for_fitbit_error(response)
    return response.json()


def safe_fetch(access_token: str, endpoint: str, params: tuple[tuple[str, str], ...] = ()) -> dict[str, Any]:
    try:
        return fitbit_get_cached(access_token, endpoint, params)
    except FitbitAPIError as exc:
        st.warning(f"Fitbit API call failed for `{endpoint}`: {exc}")
        return {}


def parse_steps_daily(payload: dict[str, Any]) -> pd.DataFrame:
    records = payload.get("activities-steps", [])
    if not records:
        return pd.DataFrame(columns=["date", "steps"])
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["dateTime"], errors="coerce")
    df["steps"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
    return df[["date", "steps"]].sort_values("date")


def parse_steps_intraday(payload: dict[str, Any], fallback_date: date) -> pd.DataFrame:
    dataset = payload.get("activities-steps-intraday", {}).get("dataset", [])
    base_date = fallback_date.isoformat()
    if payload.get("activities-steps"):
        base_date = payload["activities-steps"][0].get("dateTime", base_date)
    if not dataset:
        return pd.DataFrame(columns=["timestamp", "steps", "cumulative_steps"])

    df = pd.DataFrame(dataset)
    df["steps"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
    df["timestamp"] = pd.to_datetime(base_date + " " + df["time"], errors="coerce")
    df["cumulative_steps"] = df["steps"].cumsum()
    return df[["timestamp", "steps", "cumulative_steps"]]


def parse_heart_daily(payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in payload.get("activities-heart", []):
        value = item.get("value", {})
        rows.append(
            {
                "date": pd.to_datetime(item.get("dateTime"), errors="coerce"),
                "resting_hr": pd.to_numeric(value.get("restingHeartRate"), errors="coerce"),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["date", "resting_hr"])
    return pd.DataFrame(rows).sort_values("date")


def parse_heart_intraday(payload: dict[str, Any], fallback_date: date) -> pd.DataFrame:
    dataset = payload.get("activities-heart-intraday", {}).get("dataset", [])
    base_date = fallback_date.isoformat()
    if payload.get("activities-heart"):
        base_date = payload["activities-heart"][0].get("dateTime", base_date)
    if not dataset:
        return pd.DataFrame(columns=["timestamp", "bpm", "bpm_smooth"])

    df = pd.DataFrame(dataset)
    df["bpm"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
    df["timestamp"] = pd.to_datetime(base_date + " " + df["time"], errors="coerce")
    df["bpm_smooth"] = df["bpm"].rolling(window=5, min_periods=1).mean()
    return df[["timestamp", "bpm", "bpm_smooth"]]


def parse_heart_zones(payload: dict[str, Any]) -> pd.DataFrame:
    records = payload.get("activities-heart", [])
    if not records:
        return pd.DataFrame(columns=["zone", "minutes", "calories"])

    zones = records[0].get("value", {}).get("heartRateZones", [])
    rows = []
    for zone in zones:
        rows.append(
            {
                "zone": zone.get("name", "Unknown"),
                "minutes": pd.to_numeric(zone.get("minutes"), errors="coerce"),
                "calories": pd.to_numeric(zone.get("caloriesOut"), errors="coerce"),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["zone", "minutes", "calories"])
    return pd.DataFrame(rows)


def parse_sleep_series(payload: dict[str, Any], column_name: str) -> pd.DataFrame:
    series_key = next((key for key in payload if key.startswith("sleep-")), None)
    if not series_key:
        return pd.DataFrame(columns=["date", column_name])

    df = pd.DataFrame(payload.get(series_key, []))
    if df.empty:
        return pd.DataFrame(columns=["date", column_name])

    df["date"] = pd.to_datetime(df["dateTime"], errors="coerce")
    df[column_name] = pd.to_numeric(df["value"], errors="coerce")
    return df[["date", column_name]].sort_values("date")


def select_primary_sleep_log(payload: dict[str, Any]) -> dict[str, Any] | None:
    logs = payload.get("sleep", [])
    if not logs:
        return None
    main_logs = [log for log in logs if log.get("isMainSleep")]
    candidates = main_logs or logs
    return max(candidates, key=lambda log: log.get("duration", 0))


def parse_sleep_stage_segments(log: dict[str, Any] | None) -> pd.DataFrame:
    if not log:
        return pd.DataFrame(columns=["start", "end", "state", "minutes"])

    if log.get("minuteData"):
        start_time = pd.to_datetime(log.get("startTime"), errors="coerce")
        stage_map = {"1": "Asleep", "2": "Restless", "3": "Awake"}
        minute_rows = []
        for index, item in enumerate(log["minuteData"]):
            minute_rows.append(
                {
                    "timestamp": start_time + pd.to_timedelta(index, unit="m"),
                    "state": stage_map.get(str(item.get("value")), "Unknown"),
                }
            )

        if not minute_rows:
            return pd.DataFrame(columns=["start", "end", "state", "minutes"])

        df = pd.DataFrame(minute_rows)
        change_group = (df["state"] != df["state"].shift()).cumsum()
        segments = (
            df.groupby(change_group)
            .agg(start=("timestamp", "first"), end=("timestamp", "last"), state=("state", "first"))
            .reset_index(drop=True)
        )
        segments["end"] = segments["end"] + pd.Timedelta(minutes=1)
        segments["minutes"] = (segments["end"] - segments["start"]).dt.total_seconds() / 60
        return segments

    levels = log.get("levels", {}).get("data", [])
    if levels:
        rows = []
        for item in levels:
            start = pd.to_datetime(item.get("dateTime"), errors="coerce")
            seconds = int(item.get("seconds", 0))
            rows.append(
                {
                    "start": start,
                    "end": start + pd.to_timedelta(seconds, unit="s"),
                    "state": str(item.get("level", "Unknown")).title(),
                    "minutes": seconds / 60,
                }
            )
        return pd.DataFrame(rows)

    return pd.DataFrame(columns=["start", "end", "state", "minutes"])


def merge_time_series(frames: list[pd.DataFrame]) -> pd.DataFrame:
    usable = [frame.set_index("date") for frame in frames if not frame.empty]
    if not usable:
        return pd.DataFrame(columns=["date"])
    merged = pd.concat(usable, axis=1).reset_index()
    return merged.sort_values("date")


def build_sleep_daily(
    minutes_payload: dict[str, Any],
    time_in_bed_payload: dict[str, Any],
    efficiency_payload: dict[str, Any],
) -> pd.DataFrame:
    sleep_daily = merge_time_series(
        [
            parse_sleep_series(minutes_payload, "minutes_asleep"),
            parse_sleep_series(time_in_bed_payload, "time_in_bed"),
            parse_sleep_series(efficiency_payload, "efficiency"),
        ]
    )
    if sleep_daily.empty:
        return pd.DataFrame(
            columns=["date", "minutes_asleep", "time_in_bed", "efficiency", "hours_asleep", "hours_in_bed"]
        )
    sleep_daily["hours_asleep"] = sleep_daily["minutes_asleep"] / 60
    sleep_daily["hours_in_bed"] = sleep_daily["time_in_bed"] / 60
    return sleep_daily


def format_int(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{int(round(float(value))):,}"


def format_hours(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.1f} h"


def render_setup_help(config: FitbitConfig) -> None:
    st.subheader("Fitbit App Setup")
    st.markdown(
        """
        1. Create a Fitbit Web API application in the Fitbit developer portal.
        2. Set the OAuth 2.0 redirect URI to match this Streamlit app, usually `http://localhost:8501`.
        3. Request at least these scopes: `activity heartrate sleep profile`.
        4. Add your credentials to a local `.env` file, then restart Streamlit.
        """
    )
    st.code(
        "\n".join(
            [
                f"FITBIT_CLIENT_ID={config.client_id or 'your_client_id'}",
                "FITBIT_CLIENT_SECRET=your_client_secret",
                f"FITBIT_REDIRECT_URI={config.redirect_uri}",
                f"FITBIT_SCOPE={config.scope}",
            ]
        ),
        language="bash",
    )


def render_connection_panel(config: FitbitConfig) -> None:
    st.sidebar.header("Connection")
    if st.sidebar.button("Clear cached data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    if st.session_state.get("fitbit_tokens"):
        if st.sidebar.button("Disconnect Fitbit", use_container_width=True):
            clear_connection(config)
            st.rerun()
    else:
        st.sidebar.link_button("Connect Fitbit", build_authorize_url(config), use_container_width=True)
        st.sidebar.caption("Authorize Fitbit to load activity, heart rate, and sleep data.")


def render_overview(
    profile: dict[str, Any],
    steps_daily: pd.DataFrame,
    heart_daily: pd.DataFrame,
    sleep_daily: pd.DataFrame,
) -> None:
    profile_data = profile.get("user", {})
    st.title("Fitbit Health Dashboard")
    st.caption(
        f"{profile_data.get('displayName', 'Fitbit User')} | "
        f"Timezone: {profile_data.get('timezone', 'Unknown')} | "
        f"Member since {profile_data.get('memberSince', 'N/A')}"
    )

    total_steps = steps_daily["steps"].sum() if not steps_daily.empty else None
    avg_steps = steps_daily["steps"].mean() if not steps_daily.empty else None
    latest_resting_hr = (
        heart_daily.dropna(subset=["resting_hr"]).iloc[-1]["resting_hr"]
        if not heart_daily.dropna(subset=["resting_hr"]).empty
        else None
    )
    avg_sleep = sleep_daily["hours_asleep"].mean() if not sleep_daily.empty else None

    metric_cols = st.columns(4)
    metric_cols[0].metric("Total Steps", format_int(total_steps))
    metric_cols[1].metric("Average Daily Steps", format_int(avg_steps))
    metric_cols[2].metric(
        "Latest Resting HR",
        f"{format_int(latest_resting_hr)} bpm" if latest_resting_hr is not None and not pd.isna(latest_resting_hr) else "N/A",
    )
    metric_cols[3].metric("Average Sleep", format_hours(avg_sleep))


def render_steps_section(steps_daily: pd.DataFrame, steps_intraday: pd.DataFrame, intraday_day: date) -> None:
    if steps_daily.empty:
        st.info("No steps data returned for the selected range.")
        return

    daily_chart = px.bar(
        steps_daily,
        x="date",
        y="steps",
        title="Daily Steps",
        color_discrete_sequence=[PALETTE["steps"]],
    )
    daily_chart.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.65)",
        xaxis_title="Date",
        yaxis_title="Steps",
    )
    st.plotly_chart(daily_chart, use_container_width=True)

    if steps_intraday.empty:
        st.info(
            f"No intraday steps data returned for {intraday_day.isoformat()}. "
            "Fitbit intraday access may require a Personal app or approved access."
        )
        return

    intraday_chart = go.Figure()
    intraday_chart.add_trace(
        go.Bar(
            x=steps_intraday["timestamp"],
            y=steps_intraday["steps"],
            name="Steps per Minute",
            marker_color=PALETTE["steps_fill"],
            opacity=0.8,
        )
    )
    intraday_chart.add_trace(
        go.Scatter(
            x=steps_intraday["timestamp"],
            y=steps_intraday["cumulative_steps"],
            mode="lines",
            name="Cumulative Steps",
            line={"color": PALETTE["steps"], "width": 3},
            yaxis="y2",
        )
    )
    intraday_chart.update_layout(
        title=f"Intraday Steps | {intraday_day.isoformat()}",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.65)",
        xaxis_title="Time",
        yaxis_title="Steps per Minute",
        yaxis2={"title": "Cumulative Steps", "overlaying": "y", "side": "right"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    st.plotly_chart(intraday_chart, use_container_width=True)


def render_heart_section(heart_daily: pd.DataFrame, heart_intraday: pd.DataFrame, heart_zones: pd.DataFrame, intraday_day: date) -> None:
    if not heart_daily.empty and not heart_daily.dropna(subset=["resting_hr"]).empty:
        resting_chart = px.line(
            heart_daily.dropna(subset=["resting_hr"]),
            x="date",
            y="resting_hr",
            markers=True,
            title="Resting Heart Rate Trend",
            color_discrete_sequence=[PALETTE["heart"]],
        )
        resting_chart.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.65)",
            xaxis_title="Date",
            yaxis_title="Resting BPM",
        )
        st.plotly_chart(resting_chart, use_container_width=True)
    else:
        st.info("No resting heart rate summary returned for the selected range.")

    if heart_intraday.empty:
        st.info(
            f"No intraday heart rate data returned for {intraday_day.isoformat()}. "
            "Check app permissions and Fitbit intraday access."
        )
    else:
        intraday_chart = go.Figure()
        intraday_chart.add_trace(
            go.Scatter(
                x=heart_intraday["timestamp"],
                y=heart_intraday["bpm"],
                mode="lines",
                name="BPM",
                line={"color": PALETTE["heart_fill"], "width": 1.5},
            )
        )
        intraday_chart.add_trace(
            go.Scatter(
                x=heart_intraday["timestamp"],
                y=heart_intraday["bpm_smooth"],
                mode="lines",
                name="5-Minute Avg",
                line={"color": PALETTE["heart"], "width": 3},
            )
        )
        intraday_chart.update_layout(
            title=f"Intraday Heart Rate | {intraday_day.isoformat()}",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.65)",
            xaxis_title="Time",
            yaxis_title="BPM",
        )
        st.plotly_chart(intraday_chart, use_container_width=True)

    if not heart_zones.empty:
        zone_chart = px.bar(
            heart_zones,
            x="zone",
            y="minutes",
            title=f"Heart Rate Zones | {intraday_day.isoformat()}",
            color="zone",
            color_discrete_sequence=[PALETTE["heart"], PALETTE["accent"], PALETTE["steps"], PALETTE["sleep"]],
        )
        zone_chart.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.65)",
            xaxis_title="Zone",
            yaxis_title="Minutes",
            showlegend=False,
        )
        st.plotly_chart(zone_chart, use_container_width=True)


def render_sleep_section(
    sleep_daily: pd.DataFrame,
    sleep_segments: pd.DataFrame,
    sleep_log: dict[str, Any] | None,
    sleep_day: date,
    show_details: bool = True,
) -> None:
    if not sleep_daily.empty:
        sleep_chart = go.Figure()
        sleep_chart.add_trace(
            go.Bar(
                x=sleep_daily["date"],
                y=sleep_daily["hours_asleep"],
                name="Hours Asleep",
                marker_color=PALETTE["sleep_fill"],
            )
        )
        sleep_chart.add_trace(
            go.Scatter(
                x=sleep_daily["date"],
                y=sleep_daily["hours_in_bed"],
                mode="lines+markers",
                name="Hours in Bed",
                line={"color": PALETTE["sleep"], "width": 3},
            )
        )
        sleep_chart.update_layout(
            title="Sleep Trend",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.65)",
            xaxis_title="Date",
            yaxis_title="Hours",
        )
        st.plotly_chart(sleep_chart, use_container_width=True)
    else:
        st.info("No sleep summary data returned for the selected range.")

    if sleep_log:
        minutes_asleep = pd.to_numeric(sleep_log.get("minutesAsleep"), errors="coerce")
        time_in_bed = pd.to_numeric(sleep_log.get("timeInBed"), errors="coerce")
        efficiency = format_int(pd.to_numeric(sleep_log.get("efficiency"), errors="coerce"))
        summary_cols = st.columns(4)
        summary_cols[0].metric("Time Asleep", format_hours(minutes_asleep / 60 if not pd.isna(minutes_asleep) else None))
        summary_cols[1].metric("Time in Bed", format_hours(time_in_bed / 60 if not pd.isna(time_in_bed) else None))
        summary_cols[2].metric("Efficiency", f"{efficiency}%" if efficiency != "N/A" else "N/A")
        summary_cols[3].metric("Start Time", sleep_log.get("startTime", "N/A"))

    if not show_details:
        return

    if sleep_segments.empty:
        st.info(f"No detailed sleep stages were returned for {sleep_day.isoformat()}.")
        return

    timeline_data = sleep_segments.copy()
    timeline_data["track"] = "Sleep"
    stage_chart = px.timeline(
        timeline_data,
        x_start="start",
        x_end="end",
        y="track",
        color="state",
        title=f"Sleep Stages | {sleep_day.isoformat()}",
        color_discrete_map={
            "Awake": "#D65A31",
            "Restless": "#E9C46A",
            "Asleep": "#355070",
            "Light": "#7EA0B7",
            "Deep": "#355070",
            "Rem": "#6D597A",
        },
    )
    stage_chart.update_yaxes(title="", showticklabels=False)
    stage_chart.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.65)",
        xaxis_title="Time",
        legend_title="Stage",
    )
    st.plotly_chart(stage_chart, use_container_width=True)

    stage_summary = sleep_segments.groupby("state", as_index=False)["minutes"].sum().sort_values("minutes", ascending=False)
    summary_chart = px.bar(
        stage_summary,
        x="state",
        y="minutes",
        title="Sleep Stage Duration",
        color="state",
        color_discrete_map={
            "Awake": "#D65A31",
            "Restless": "#E9C46A",
            "Asleep": "#355070",
            "Light": "#7EA0B7",
            "Deep": "#355070",
            "Rem": "#6D597A",
        },
    )
    summary_chart.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.65)",
        xaxis_title="Stage",
        yaxis_title="Minutes",
        showlegend=False,
    )
    st.plotly_chart(summary_chart, use_container_width=True)


def require_config(config: FitbitConfig) -> None:
    if config.client_id and config.client_secret and config.redirect_uri:
        return
    st.error("Missing Fitbit OAuth configuration. Add your Fitbit app credentials before running the dashboard.")
    render_setup_help(config)
    st.stop()


def normalize_date_range(value: Any) -> tuple[date, date]:
    if isinstance(value, tuple) and len(value) == 2:
        start_date, end_date = value
    else:
        start_date = value
        end_date = value
    if start_date > end_date:
        start_date, end_date = end_date, start_date
    return start_date, end_date


def main() -> None:
    apply_styles()
    config = load_config()
    init_session(config)
    handle_oauth_callback(config)
    require_config(config)
    render_connection_panel(config)

    if not st.session_state.get("fitbit_tokens"):
        st.title("Fitbit Health Dashboard")
        st.caption("Streamlit dashboard powered by Fitbit Web API data.")
        render_setup_help(config)
        st.info("Connect Fitbit from the sidebar after adding your OAuth credentials.")
        st.stop()

    try:
        access_token = ensure_access_token(config)
    except FitbitAuthError as exc:
        st.error(str(exc))
        st.stop()

    if not access_token:
        st.info("Connect Fitbit from the sidebar to load your data.")
        st.stop()

    today = date.today()
    default_start = today - timedelta(days=29)

    st.sidebar.header("Filters")
    selected_range = st.sidebar.date_input(
        "Date range",
        value=(default_start, today),
        min_value=today - timedelta(days=365),
        max_value=today,
    )
    start_date, end_date = normalize_date_range(selected_range)

    intraday_day = st.sidebar.date_input(
        "Intraday day",
        value=end_date,
        min_value=start_date,
        max_value=end_date,
        key="intraday_day",
    )
    sleep_day = st.sidebar.date_input(
        "Sleep detail day",
        value=end_date,
        min_value=start_date,
        max_value=end_date,
        key="sleep_day",
    )

    with st.spinner("Loading Fitbit data..."):
        profile = safe_fetch(access_token, "/1/user/-/profile.json")
        steps_daily_payload = safe_fetch(
            access_token,
            f"/1/user/-/activities/steps/date/{start_date.isoformat()}/{end_date.isoformat()}.json",
        )
        steps_intraday_payload = safe_fetch(
            access_token,
            f"/1/user/-/activities/steps/date/{intraday_day.isoformat()}/1d/1min.json",
        )
        heart_range_payload = safe_fetch(
            access_token,
            f"/1/user/-/activities/heart/date/{start_date.isoformat()}/{end_date.isoformat()}/1min.json",
        )
        heart_intraday_payload = safe_fetch(
            access_token,
            f"/1/user/-/activities/heart/date/{intraday_day.isoformat()}/1d/1sec.json",
        )
        sleep_minutes_payload = safe_fetch(
            access_token,
            f"/1/user/-/sleep/minutesAsleep/date/{start_date.isoformat()}/{end_date.isoformat()}.json",
        )
        sleep_time_in_bed_payload = safe_fetch(
            access_token,
            f"/1/user/-/sleep/timeInBed/date/{start_date.isoformat()}/{end_date.isoformat()}.json",
        )
        sleep_efficiency_payload = safe_fetch(
            access_token,
            f"/1/user/-/sleep/efficiency/date/{start_date.isoformat()}/{end_date.isoformat()}.json",
        )
        sleep_log_payload = safe_fetch(
            access_token,
            f"/1/user/-/sleep/date/{sleep_day.isoformat()}.json",
        )

    steps_daily = parse_steps_daily(steps_daily_payload)
    steps_intraday = parse_steps_intraday(steps_intraday_payload, intraday_day)
    heart_daily = parse_heart_daily(heart_range_payload)
    heart_intraday = parse_heart_intraday(heart_intraday_payload, intraday_day)
    heart_zones = parse_heart_zones(heart_intraday_payload or heart_range_payload)
    sleep_daily = build_sleep_daily(
        sleep_minutes_payload,
        sleep_time_in_bed_payload,
        sleep_efficiency_payload,
    )
    sleep_log = select_primary_sleep_log(sleep_log_payload)
    sleep_segments = parse_sleep_stage_segments(sleep_log)

    render_overview(profile, steps_daily, heart_daily, sleep_daily)

    overview_tab, steps_tab, heart_tab, sleep_tab = st.tabs(
        ["Overview", "Steps", "Heart Rate", "Sleep"]
    )

    with overview_tab:
        summary_left, summary_right = st.columns(2)
        with summary_left:
            render_steps_section(steps_daily, steps_intraday, intraday_day)
        with summary_right:
            render_sleep_section(sleep_daily, pd.DataFrame(), sleep_log, sleep_day, show_details=False)

    with steps_tab:
        render_steps_section(steps_daily, steps_intraday, intraday_day)

    with heart_tab:
        render_heart_section(heart_daily, heart_intraday, heart_zones, intraday_day)

    with sleep_tab:
        render_sleep_section(sleep_daily, sleep_segments, sleep_log, sleep_day)


if __name__ == "__main__":
    main()
