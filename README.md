# EEG_DV_Tool

A web-based tool for visualizing and analyzing EEG engagement and attention indices from participants watching Data Videos (DVs) and Micro Data Videos (mDVs). Enables researchers to explore the relationship between DV elements (stimuli) and EEG brain activity patterns.

## Features

- Synchronized video player and EEG time-series chart
- Filter by participant, brain region, and index type (engagement / attention)
- Toggle between individual participant data and averaged data
- Optional z-score normalization and outlier clipping
- DV element overlays showing when specific video elements occurred (zoom, animation, sound effects, etc.)
- Vertical sync line tracking video playback position on the chart

## Tech Stack

- **Backend**: Python, FastAPI, Pandas
- **Frontend**: HTML5, JavaScript, Chart.js
- **Templating**: Jinja2

## Setup

Requires Python 3 and a virtual environment with dependencies installed.

```bash
pip install -r requirement.txt
```

## Running the App

```bash
cd <repo-folder>
.\venv\Scripts\activate.bat
uvicorn app.main:app --reload --port 8080
```

Then Ctrl+click the localhost link in the terminal.

## Project Structure

```
app/
  main.py              # FastAPI backend and EEG processing logic
  static/
    style.css          # Dashboard styles
    player.css         # Player page styles
    data/
      videos.csv       # Video metadata
      video_stimuli.csv
      eeg_*.csv        # EEG data files
      mdvs/            # Video files
  templates/
    index.html         # Video listing dashboard
    player.html        # Video player and analysis interface
requirement.txt
```

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Dashboard listing all videos |
| `GET /player` | Video player and analysis view |
| `GET /api/eeg` | EEG data with optional filters (participant, region, normalization) |
| `GET /api/stimuli` | Stimulus timing data for a given video |