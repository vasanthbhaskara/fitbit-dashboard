# Fitbit Streamlit Dashboard

Streamlit dashboard for live Fitbit activity, heart-rate, and sleep analytics.

## Features

- Fitbit OAuth 2.0 login with refresh-token persistence
- Daily and intraday steps charts
- Daily and intraday heart-rate analysis with summary fallback when intraday data is missing
- Sleep duration, stage breakdown, and bedtime consistency charts
- Temperature, respiratory rate, oxygen saturation, and cardio-fitness trend panels
- Regression-based step and resting heart-rate forecasts
- Statistical insight cards for trends, correlation, and outliers
- Selected-day drilldown metrics and AI day-level insights
- Optional AI insights panel powered by Groq

## Setup

1. Create a Fitbit developer app in the Fitbit developer portal.
2. Configure the redirect URI to match your Streamlit app URL.
   - Local default: `http://localhost:8501`
3. Copy `.env.example` to `.env` and fill in your Fitbit client credentials.
4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Run the app:

```bash
streamlit run app.py
```

## Optional AI insights

Add a Groq API key to `.env` if you want a free-tier LLM summary of the current dashboard metrics:

```bash
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.1-8b-instant
```

Then use the `Generate insights` button inside the app. The LLM sees only aggregated Fitbit metrics prepared by the dashboard, not your full raw intraday dataset.

## Fitbit scopes

The app expects these scopes:

- `activity`
- `heartrate`
- `sleep`
- `profile`
- `temperature`
- `respiratory_rate`
- `oxygen_saturation`
- `cardio_fitness`

If you add new scopes after first authorizing the app, disconnect and reconnect Fitbit so the token is reissued with the expanded scope set.

## Notes

- Detailed intraday endpoints are typically intended for personal-use applications connected to your own Fitbit account.
- OAuth tokens are stored locally in `.fitbit_tokens.json` so the app can refresh access automatically.
- Fitbit API responses are cached locally and in the Streamlit session for 15 minutes to reduce quota pressure, and the lookback slider is capped to keep request volume predictable.
- The original `fitbit.py` file in this repo is an older notebook export and is not used by the Streamlit app.
