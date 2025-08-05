from fastapi import FastAPI, Request, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse


import pandas as pd
import numpy as np
from typing import Union
import os

app = FastAPI()
base_dir = os.path.dirname(__file__)
static_path = os.path.join(base_dir, "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")
templates = Jinja2Templates(directory="app/templates")

VIDEO_CSV_PATH = os.path.join("app", "static", "data", "videos.csv")
EEG_CSV_PATH = os.path.join("app", "static", "data", "eeg.csv")

def home():
    if os.path.exists(VIDEO_CSV_PATH):
        df = pd.read_csv(VIDEO_CSV_PATH)
        return df.to_dict(orient='records')
    else:
        return []

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    videos = home()
    return templates.TemplateResponse("index.html", {"request": request, "videos": videos})

@app.get("/player", response_class=HTMLResponse)
def player(request: Request, id: str):
    df = pd.read_csv(VIDEO_CSV_PATH)
    video = df[df["id"] == int(id)].to_dict(orient="records")
    if not video:
        return HTMLResponse("Video not found", status_code=404)
    eeg_data = get_eeg_summary(video_id=video[0]["id"], participant_id=[105, 107, 2, 7], region="all", index="engagement")
    eeg_list = eeg_data.tolist()
    return templates.TemplateResponse("player.html", {"request": request, "video": video[0], "eeg_data": eeg_list})

def smooth_signal(signal, window_len=5):
    return np.convolve(signal, np.ones(window_len)/window_len, mode='same')

def get_eeg_summary(video_id: int, participant_id: list | int, region: str = "all", index: str = "engagement"):
    """
    This function calculates the average index using the data in eeg.csv based on the video_id, participant_id, desired index, and region.
   
    video_id: One of the videos: 0, 1, 2, 3. //Ghazaleh: I just changed this to int because in other files it is int.
    participant_id: Currently: 105, 107, 2, 7, 10, 14. Accepts a list of IDs or just one integer.
    region: Brain regions: all, frontal, frontocentral, parietal.
    index: engagement, attention.

    Returns: The average index values across the region over the participants
    """ 
    # @app.get("/eeg-summary/{video_id}")
    # Load EEG CSV
    print("Loading EEG Data...")
    df = pd.read_csv(EEG_CSV_PATH)
    print("EEG Data Loaded")

    # Convert string lists in PSD columns back to Python lists
    for col in ['theta_psd', 'alpha_psd', 'beta_psd']:
        df[col] = df[col].apply(eval)

    # Filter rows based on video ID
    df = df[df['video'] == video_id]

    # Filter by participant(s)
    if isinstance(participant_id, int):
        df = df[df['participant'] == participant_id]
    else:
        df = df[df['participant'].isin(participant_id)]

    # Filter by brain region unless "all" regions are requested
    if region.lower() != "all":
        df = df[df['region'].str.lower() == region.lower()]

    # Error if no data remains after filtering
    if df.empty:
        raise ValueError("No data found for given parameters.")

    # --- Process participant-wise data ---
    participant_data = []

    # Loop through each participant's data
    for pid, group in df.groupby('participant'):

        # Compute engagement or attention index for each row in the group (each row = one region)
        if index == "engagement":
            vals = [
                np.array(beta) / (np.array(theta) + np.array(alpha))
                for theta, alpha, beta in zip(group['theta_psd'], group['alpha_psd'], group['beta_psd'])
            ]
        elif index == "attention":
            vals = [
                np.array(beta) / np.array(alpha)
                for alpha, beta in zip(group['alpha_psd'], group['beta_psd'])
            ]
        else:
            raise ValueError("Invalid index. Use 'engagement' or 'attention'.")

        # --- Average across regions ---
        flattened = np.mean(vals, axis=0)  # shape: (n_timepoints,)

        # Optional: smooth the signal
        smoothed = smooth_signal(flattened)

        # Add to list for averaging later across participants
        participant_data.append(smoothed)

    # --- Average across participants ---
    # First align all participant signals to the same length (minimum time length)
    min_len = min(len(arr) for arr in participant_data)
    trimmed = np.array([arr[:min_len] for arr in participant_data])

    # Final average across participants
    return np.mean(trimmed, axis=0)

@app.get("/api/eeg")
def get_filtered_eeg_api(
    video: int,
    participant: list[int] = Query(None),
    region: str = "all",
    index: str = "engagement"
):
    try:
        if participant is None:
            participant = [105, 107, 2, 7]
        result = get_eeg_summary(video, participant, region, index)
        return JSONResponse(content=result.tolist())
    except ValueError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})