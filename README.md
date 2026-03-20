# Fitbit Dashboard

An open-source, LLM-powered personal health dashboard built with Streamlit. Connect your Fitbit, explore your activity, sleep, and heart rate data through interactive charts, and get AI-generated insights powered by Groq.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)

---

## Overview

This dashboard connects to the Fitbit Web API via OAuth 2.0 and visualises your personal health data in real time. No backend server required — credentials and tokens are stored entirely in your browser session.

Anyone can use it by entering their own Fitbit developer credentials directly in the app UI.

---

## Features

| Category | Details |
|---|---|
| **Activity** | Daily steps, 7-day rolling average, hourly breakdown, cumulative chart |
| **Heart rate** | Resting HR trend, 3-day rolling average, next-day forecast |
| **Sleep** | Duration, stage breakdown (deep/light/REM/wake), bedtime consistency |
| **Analytics** | Linear regression forecasts, z-score outlier detection, sleep/steps correlation |
| **Consistency score** | Composite daily score across steps, sleep, efficiency, bedtime, and recovery |
| **AI insights** | Groq LLM summary of your metrics — window summary or selected-day drilldown |

---

## Getting started

### 1. Create a Fitbit developer app

- Go to [dev.fitbit.com](https://dev.fitbit.com) → Manage → Register An App
- Set **OAuth 2.0 Application Type** to `Personal`
- Set **Redirect URL** to `http://localhost:8501/`
- Note your **Client ID** and **Client Secret**

### 2. Connect in the UI

On first load the app shows a credentials form. Enter your Client ID, Client Secret, and Redirect URI — no `.env` file needed.

### 4. Authorize Fitbit

Click **Connect Fitbit** in the sidebar. After authorizing, you'll be redirected back and the dashboard loads automatically.

---

### AI insights (optional)

Get a free API key at [console.groq.com](https://console.groq.com) and enter it in the credentials form. The default model is `llama-3.1-8b-instant`
---

## Project structure

```
fitbit-dashboard/
├── app.py              # Main Streamlit app
├── requirements.txt    # Python dependencies
├── .env.example        # Optional local config template
└── README.md
```

---

## Tech stack

- [Streamlit](https://streamlit.io) — UI and deployment
- [Fitbit Web API](https://dev.fitbit.com/build/reference/web-api/) — health data
- [Plotly](https://plotly.com/python/) — interactive charts
- [Pandas](https://pandas.pydata.org) — data processing
- [Groq](https://console.groq.com) — LLM inference for AI insights

---

## Notes

- API responses are cached in session state for 15 minutes to reduce quota pressure
- The lookback window is capped at 30 days to keep requests predictable
- Tokens are stored in session state only — nothing is written to disk on Streamlit Cloud

---