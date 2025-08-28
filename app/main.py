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
VIDEO_STIMULI_CSV_PATH = os.path.join("app", "static", "data", "video_stimuli.csv")

# we should extract this from the database or a config file - need to be revised later
participants = [105, 107, 2, 7, 10, 14, 15, 16, 17,18,19,20,21,22,23,24,25,26]

stimuli = {
    0: {"id": 0, "label": "Sound effect", "color": "#e6194b"},
    1: {"id": 1, "label": "Zoom", "color": "#3cb44b"},
    2: {"id": 2, "label": "Pointing", "color": "#ffe119"},
    3: {"id": 3, "label": "Highlighting", "color": "#0082c8"},
    4: {"id": 4, "label": "Gradual Text", "color": "#f58231"},
    5: {"id": 5, "label": "Gradual appearance", "color": "#911eb4"},
    6: {"id": 6, "label": "Camera Angle", "color": "#46f0f0"},
    7: {"id": 7, "label": "Animation", "color": "#f032e6"},
    8: {"id": 8, "label": "Map", "color": "#d2f53c"},
    9: {"id": 9, "label": "Area", "color": "#fabebe"},
    10: {"id": 10, "label": "Pictograph", "color": "#008080"},
    11: {"id": 11, "label": "Size Change", "color": "#e6beff"},
}

# Helper functions
def smooth_signal(signal, window_len=3):
    return np.convolve(signal, np.ones(window_len)/window_len, mode='same')

# Reads indices, Normalizes using min max OR zscore to make sure all indices are in the same range before across PPs
def get_eeg_summary(video_id: int, participant_id: list | int, region: str = "all",
                    index: str = "engagement", normalize: str | None = None):
    """
    Summarizes EEG indices (engagement or attention) for given video(s), participant(s), and region(s), with optional normalization per participant.
    Reads directly from the precomputed "engagement" and "attention" columns in eeg.csv.

    Each row contains a list of index values per clean 0.5s window for a single channel.
    video_id: One of the videos: 0, 1, 2, 3 (these are mDVs v3-v6).
    participant_id: Accepts a list of IDs or just one integer. The list is in variable <participants> above which should be updated with new data
    region: Brain regions: all, frontal, frontocentral, parietal, other. "all" means the 4 regions combined
    index: engagement, attention.
    normalize: None, "minmax", or "zscore"

    Returns: The average index values across the region over the participants
    """
    # Load EEG CSV
    df = pd.read_csv(EEG_CSV_PATH)

    # Convert stringified lists back to Python lists
    df['engagement'] = df['engagement'].apply(eval)
    df['attention']  = df['attention'].apply(eval)

    # Filter by video
    df = df[df['video'] == video_id]

    # Filter by participant(s)
    if isinstance(participant_id, int):
        df = df[df['participant'] == participant_id]
    else:
        df = df[df['participant'].isin(participant_id)]

    # Filter by region if not "all"
    if region != "all":
        df = df[df['region'].str.lower() == region.lower()]

    if df.empty:
        raise ValueError("No data found for given parameters.")

    # --- Process participant-wise data ---
    participant_data = []

    for pid, group in df.groupby('participant'):
        # Pick engagement or attention
        vals = group[index].tolist()  # each element = list of windows per channel
        # Average across channels/regions for this participant
        channel_means = np.mean([np.array(v) for v in vals], axis=0)

        # Optional normalization per participant
        if normalize == "minmax":
            min_val = np.min(channel_means)
            max_val = np.max(channel_means)
            if max_val != min_val:
                channel_means = (channel_means - min_val) / (max_val - min_val)
        elif normalize == "zscore":
            mean_val = np.mean(channel_means)
            std_val = np.std(channel_means)
            if std_val != 0:
                channel_means = (channel_means - mean_val) / std_val

        # Smooth signal
        smoothed = smooth_signal(channel_means)
        participant_data.append(smoothed)

    # --- Average across participants ---
    min_len = min(len(arr) for arr in participant_data)
    trimmed = np.array([arr[:min_len] for arr in participant_data])
    return np.mean(trimmed, axis=0)

# FastAPI routes
# Index page - shows all videos
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

def get_video_by_video_id(video_id: str):
    df = pd.read_csv(VIDEO_CSV_PATH)
    video = df[df["id"] == int(video_id)].to_dict(orient="records")
    if not video:
        return HTMLResponse("Video not found", status_code=404)
    return video[0]

def get_stimuli_by_video_id(video_id: str):
    video_stimuli = pd.read_csv(VIDEO_STIMULI_CSV_PATH)
    stimuli_list = video_stimuli[video_stimuli["video_id"] == int(video_id)].to_dict(orient="records")
    unique_stimuli_ids = set(item["stimulus_id"] for item in stimuli_list)
    vide_stim_info = [stimuli[s] for s in unique_stimuli_ids]
    return stimuli_list, vide_stim_info

def get_eeg_data_by_video_id(video_id: str):
    engagement_data = get_eeg_summary(video_id=int(video_id), participant_id=participants, region="all", index="engagement", normalize='minmax')
    engagement_list = engagement_data.tolist()
    attention_data = get_eeg_summary(video_id=int(video_id), participant_id=participants, region="all", index="attention", normalize='minmax')
    attention_list = attention_data.tolist()
    return engagement_list, attention_list

@app.get("/player", response_class=HTMLResponse)
async def player(request: Request, id: str):
    video = get_video_by_video_id(id)
    stimuli_list, vide_stim_info = get_stimuli_by_video_id(id)
    eng_list, atn_list = get_eeg_data_by_video_id(id)
    return templates.TemplateResponse("player.html", 
                                        {"request": request, 
                                        "video": video, 
                                        "eng_data":  eng_list, 
                                        "atn_data": atn_list,
                                        "stimuli_data": stimuli_list, 
                                        "stimuli_info": vide_stim_info,
                                        "participants": participants})

@app.get("/api/stimuli")
def get_video_stimuli(video_id: int):
    if os.path.exists(VIDEO_STIMULI_CSV_PATH):
        stimuli_df = pd.read_csv(VIDEO_STIMULI_CSV_PATH)
        try:
            filtered = stimuli_df[stimuli_df["video_id"] == video_id]
            return JSONResponse(content=filtered.to_dict(orient="records"))
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})
    else:
        return []
    
@app.get("/api/eeg")
def get_filtered_eeg_api(
    video: int,
    participant: list[int] = Query(None),
    region: list[str] = Query(None)
):
    try:
        if participant is None:
            participant = participants
        results = {}

        for p in participant:
            if region:
                for r in region:
                    engagement = get_eeg_summary(video, p, r, index="engagement", normalize='minmax')
                    attention = get_eeg_summary(video, p, r, index="attention", normalize='minmax')
                    results[f"participant_{p}_{r}"] = {
                        "engagement": engagement.tolist(),
                        "attention": attention.tolist()
                    }
            else:
                engagement = get_eeg_summary(video, p, "all", index="engagement", normalize='minmax')
                attention = get_eeg_summary(video, p, "all", index="attention", normalize='minmax')
                results[f"participant_{p}_all"] = {
                    "engagement": engagement.tolist(),
                    "attention": attention.tolist()
                }
        results = {}
        
        engagement_all = get_eeg_summary(video, participant, "all", index="engagement", normalize='minmax')
        attention_all = get_eeg_summary(video, participant, "all", index="attention", normalize='minmax')
        results["all"] = {
            "engagement": engagement_all.tolist(),
            "attention": attention_all.tolist()
        }
    except ValueError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})

    
