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
# EEG_BASELINE_SUBTRACTED_CSV_PATH = os.path.join("app", "static", "data", "eeg_baseline_subtracted.csv")
VIDEO_STIMULI_CSV_PATH = os.path.join("app", "static", "data", "video_stimuli.csv")

# we should extract this from the database or a config file - need to be revised later
participants = [105, 107, 2, 7, 14, 15, 16, 17,18,19,20,21,22,23,24,25,26,27]

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
                    index: str = "engagement", normalize: str = "false"):
    """
    Summarizes EEG indices (engagement or attention) for given video(s), participant(s), and region(s),
    with optional normalization per participant. Reads from precomputed eeg.csv.

    Parameters
    ----------
    video_id : int
        One of the videos: 0, 1, 2, 3 (corresponding to v3–v6).
    participant_id : list[int] or int
        Either a single participant ID or a list of participants.
        The list is in variable <participants> above which should be updated with new data
    region : str
        Brain region: 'all', 'frontal', 'frontocentral', 'parietal', 'other'.
        'all' means region = "all" as precomputed in the CSV.
    index : str
        'engagement' or 'attention'.
    normalize : None | 'minmax' | 'zscore'
        Whether to normalize per participant before averaging.

    Returns
    -------
    np.ndarray
        Group-level average index time series, index-aligned across participants.
    """

    # Load CSV
    df = pd.read_csv(EEG_CSV_PATH) #  EEG_CSV_PATH OR EEG_BASELINE_SUBTRACTED_CSV_PATH
    # Convert stringified lists back into Python lists
    df['engagement'] = df['engagement'].apply(eval)
    df['attention']  = df['attention'].apply(eval)

    # Filter by video
    df = df[df['video'] == video_id]

    # Filter by participant(s)
    if isinstance(participant_id, int):
        df = df[df['participant'] == participant_id]
    else:
        df = df[df['participant'].isin(participant_id)]

    # Filter by region
    df = df[df['region'].str.lower() == region.lower()]

    if df.empty:
        raise ValueError("No data found for given parameters.")

    # --- Per participant processing ---
    participant_data = []
    for pid, group in df.groupby('participant'):
        # Each row is already region-averaged → just take it
        vals = group[index].tolist()[0]  # only 1 row per participant per region
        vals = np.array(vals)

        # Optional normalization
        # if normalize == "minmax":
        #     min_val, max_val = np.min(vals), np.max(vals)
        #     if max_val > min_val:
        #         vals = (vals - min_val) / (max_val - min_val)
        if normalize == "true":
            mean_val, std_val = np.mean(vals), np.std(vals)
            if std_val > 0:
                vals = (vals - mean_val) / std_val

        # Already smoothed in preprocessing, but can smooth again if desired
        vals = smooth_signal(vals)
        participant_data.append(vals)

    # --- Index alignment & group average ---
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

def get_eeg_data_by_video_id(video_id: str, normalize: str | None = None):
    engagement_data = get_eeg_summary(video_id=int(video_id), participant_id=participants, region="all", index="engagement", normalize=normalize)
    engagement_list = engagement_data.tolist()
    attention_data = get_eeg_summary(video_id=int(video_id), participant_id=participants, region="all", index="attention", normalize=normalize)
    attention_list = attention_data.tolist()
    return engagement_list, attention_list

@app.get("/player", response_class=HTMLResponse)
async def player(request: Request, id: str, normalize: bool = Query(False)):
    video = get_video_by_video_id(id)
    stimuli_list, vide_stim_info = get_stimuli_by_video_id(id)
    eng_list, atn_list = get_eeg_data_by_video_id(id, normalize)
    print("Engagement data length:", len(eng_list))
    print("Attention data length:", len(atn_list))
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
    region: list[str] = Query(None),
    normalize: str = Query('false'),
    merge: str = Query('false')
):
    try:
        if participant is None:
            participant = participants
        results = {}
        if merge == 'true':
            print("Merging all participants")
            if region:
                for r in region:
                    engagement = get_eeg_summary(video, participant, r, index="engagement", normalize=normalize)
                    attention = get_eeg_summary(video, participant, r, index="attention", normalize=normalize)
                    results[f"all_{r}"] = {
                        "engagement": engagement.tolist(),
                        "attention": attention.tolist(),
                        "region": r,
                        "participant": "all"
                    }
                    print(f"Computed all participants for region {r}")
            else:
                engagement = get_eeg_summary(video, participant, "all", index="engagement", normalize=normalize)
                attention = get_eeg_summary(video, participant, "all", index="attention", normalize=normalize)
                results["all"] = {
                    "engagement": engagement.tolist(),
                    "attention": attention.tolist(),
                    "region": "all",
                    "participant": "all"
                }
                print("Computed all participants for all regions")
        else:
            print("Processing individual participants")
            for p in participant:
                if region:
                    for r in region:
                        engagement = get_eeg_summary(video, p, r, index="engagement", normalize=normalize)
                        attention = get_eeg_summary(video, p, r, index="attention", normalize=normalize)
                        # region R for participant P
                        results[f"{p}_{r}"] = {
                            "engagement": engagement.tolist(),
                            "attention": attention.tolist(),
                            "region": r,
                            "participant": p
                        }
                        print(f"Computed participant {p} for region {r}")
                else:
                    engagement = get_eeg_summary(video, p, "all", index="engagement", normalize=normalize)
                    attention = get_eeg_summary(video, p, "all", index="attention", normalize=normalize)
                    # all regions for participant P
                    results[f"{p}"] = {
                        "engagement": engagement.tolist(),
                        "attention": attention.tolist(),
                        "region": "all",
                        "participant": p
                    }
                    print(f"Computed participant {p} for all regions")
        results["meta"] = {
            "video": video,
            "participants": participant,
            "regions": region if region else ["all"],
            "normalize": normalize,
            "length": len(next(iter(results.values()))["engagement"]) if results else 0
        }
        # overall average across all selected participants and regions
        # engagement_all = get_eeg_summary(video, participant, "all", index="engagement", normalize=normalize)
        # attention_all = get_eeg_summary(video, participant, "all", index="attention", normalize=normalize)
        # results["all"] = {
        #     "engagement": engagement_all.tolist(),
        #     "attention": attention_all.tolist()
        # }
        return JSONResponse(content=results)
    except ValueError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})

    
