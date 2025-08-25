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
participants = [105, 107, 2, 7, 10, 14]

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
    engagement_data = get_eeg_summary(video_id=int(video_id), participant_id=participants, region="all", index="engagement")
    engagement_list = engagement_data.tolist()
    attention_data = get_eeg_summary(video_id=int(video_id), participant_id=participants, region="all", index="attention")
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
<<<<<<< Updated upstream
    region: str = "all"
=======
    region: list[str] = Query(None)
>>>>>>> Stashed changes
):
    try:
        if participant is None:
            participant = participants
<<<<<<< Updated upstream
        result_engagement = get_eeg_summary(video, participant, region, index="engagement")
        result_attention = get_eeg_summary(video, participant, region, index="attention")
        return JSONResponse(content={
            "engagement": result_engagement.tolist(),
            "attention": result_attention.tolist()
        })
=======

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


        engagement_all = get_eeg_summary(video, participant, "all", index="engagement", normalize='minmax')
        attention_all = get_eeg_summary(video, participant, "all", index="attention", normalize='minmax')
        results["all"] = {
            "engagement": engagement_all.tolist(),
            "attention": attention_all.tolist()
        }
        return JSONResponse(content=results)

>>>>>>> Stashed changes
    except ValueError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})
    
