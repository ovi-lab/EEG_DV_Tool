from fastapi import FastAPI, Request, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse

import pandas as pd
import numpy as np
from typing import Union
import os

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

base_dir = os.path.dirname(__file__)
static_path = os.path.join(base_dir, "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")
templates = Jinja2Templates(directory="app/templates")

VIDEO_CSV_PATH = os.path.join("app", "static", "data", "videos.csv")
VIDEO_STIMULI_CSV_PATH = os.path.join("app", "static", "data", "video_stimuli.csv")

EEG_RAW_CROSS_BASELINE_SUBTRACTED_CSV_PATH_TRIMMED_POWER_NORM = os.path.join("app", "static", "data", "eeg_cross_power_normalized.csv")
EEG_RAW_CROSS_BASELINE_SUBTRACTED_CSV_PATH_TRIMMED_AVG_POWER_NORM_VIDEO_SPECIFIC_BASELINE = os.path.join("app", "static", "data", "eeg_cross_power_normalized_avgpower.csv")
EEG_RAW_CROSS_BASELINE_SUBTRACTED_CSV_PATH_TRIMMED_AVG_POWER_NORM_VIDEO_SPECIFIC_BASELINE_NO_END_TRIM = os.path.join("app", "static", "data", "eeg_cross_power_normalized_avgpower_no-end-trim.csv")

EEG_RAW_CROSS_BASELINE_SUBTRACTED_CSV_PATH_TRIMMED_POWER_NORM_dB = os.path.join("app", "static", "data", "eeg_cross_power_normalized_dB.csv")

EEG_RAW_VIDEO_BASELINE_SUBTRACTED_CSV_PATH_TRIMMED_POWER_NORM = os.path.join("app", "static", "data", "eeg_video_power_normalized.csv")

CLIP_STD = 2 # For Option 2 of Z-normalization method

# we should extract this from the database or a config file - need to be revised later
participants = [105, 107, 2, 7, 14, 15, 16, 17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38] 

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
    13: {"id": 13, "label": "Human Appearing Transition", "color": "#77ef77"},
    14: {"id": 14, "label": "Graphic Appearing Transition", "color": "#006400"}
}

# Helper functions
def smooth_signal(signal, window_len=3):
    """Smooth signal while handling NaN values properly"""
    if len(signal) == 0:
        return signal
    
    # Create a copy to avoid modifying the original
    smoothed = np.copy(signal)
    
    for i in range(len(signal)):
        start_idx = max(0, i - window_len // 2)
        end_idx = min(len(signal), i + window_len // 2 + 1)
        window = signal[start_idx:end_idx]
        
        # Calculate mean ignoring NaN values
        valid_values = window[~np.isnan(window)]
        if len(valid_values) > 0:
            smoothed[i] = np.mean(valid_values)
        else:
            smoothed[i] = np.nan
    
    return smoothed

def to_list(x):
    if isinstance(x, str):
        try:
            return eval(x, {"nan": np.nan, "np": np})
        except:
            return [np.nan]
    return x

def nanmean_along_axis(data, axis=0):
    """Calculate mean along axis ignoring NaN values"""
    return np.nanmean(data, axis=axis)

def align_and_average_with_nan(participant_data):
    """
    Align participant data to the same length and compute average ignoring NaN values.
    
    Parameters
    ----------
    participant_data : list of np.ndarray
        List of participant time series data, potentially with different lengths and NaN values.
    
    Returns
    -------
    np.ndarray
        Group-level average with NaN handling.
    """
    if not participant_data:
        return np.array([])
    
    # Find the minimum length for alignment
    min_len = min(len(arr) for arr in participant_data)
    
    # Trim all arrays to the same length
    trimmed_data = []
    for arr in participant_data:
        trimmed = arr[:min_len]
        trimmed_data.append(trimmed)
    
    # Stack into 2D array (participants x time)
    stacked = np.array(trimmed_data)
    
    # Calculate mean across participants (axis=0) ignoring NaN values
    group_average = np.nanmean(stacked, axis=0)
    
    return group_average

# Reads indices, Normalizes using min max OR zscore to make sure all indices are in the same range before across PPs
def get_eeg_summary(video_id: int, participant_id: list | int, region: str = "all",
                    index: str = "engagement", normalize: str = "false"):
    """
    Summarizes EEG indices (engagement or attention) for given video(s), participant(s), and region(s),
    with optional normalization per participant. Reads from precomputed eeg.csv.
    Now handles NaN values properly during averaging.

    Parameters
    ----------
    video_id : int
        One of the videos: 0, 1, 2, 3 (corresponding to v3–v6).
    participant_id : list[int] or int
        Either a single participant ID or a list of participants.
        The list is in variable <participants> above which should be updated with new data
    region : str
        Brain region: 'all', 'frontal', 'frontocentral', 'parietal', 'occipital', 'other'.
        'all' means region = "all" as precomputed in the CSV.
    index : str
        'engagement' or 'attention'.
    normalize : None | 'minmax' | 'zscore'
        Whether to normalize per participant before averaging.

    Returns
    -------
    np.ndarray
        Group-level average index time series, index-aligned across participants with NaN handling.
    """

    # Load CSV
    df = pd.read_csv(EEG_RAW_CROSS_BASELINE_SUBTRACTED_CSV_PATH_TRIMMED_AVG_POWER_NORM_VIDEO_SPECIFIC_BASELINE_NO_END_TRIM) #  Options present above
    # Convert stringified lists back into Python lists
    df['engagement'] = df['engagement'].apply(to_list)
    df['attention']  = df['attention'].apply(to_list)

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
    NORM = 2 # [0: Normal z-norm, 1: Robust using median and IQR, 2: Clip outliers]
    if NORM == 0:
        for pid, group in df.groupby('participant'):
            # Each row is already region-averaged → just take it
            vals = group[index].tolist()[0]  # only 1 row per participant per region
            vals = np.array(vals, dtype=float)  # Ensure float type for NaN handling

            # Optional normalization - only on non-NaN values
            if normalize == "true":
                # Only normalize if we have valid (non-NaN) values
                valid_mask = ~np.isnan(vals)
                if np.sum(valid_mask) > 0:  # Check if we have any valid values
                    valid_values = vals[valid_mask]
                    mean_val, std_val = np.mean(valid_values), np.std(valid_values)
                    if std_val > 0:
                        # Create normalized array preserving NaN positions
                        normalized_vals = np.full_like(vals, np.nan)
                        normalized_vals[valid_mask] = (valid_values - mean_val) / std_val
                        np.set_printoptions(precision=4, suppress=True)
                        # print("raw values", vals)
                        vals = normalized_vals
                        # print("normalized values", vals)

            # Smooth signal while preserving NaN values
            # vals = smooth_signal(vals)
            participant_data.append(vals)

    elif NORM == 1:
        for pid, group in df.groupby('participant'):
            # Each row is already region-averaged → just take it
            vals = group[index].tolist()[0]  # only 1 row per participant per region
            vals = np.array(vals, dtype=float)  # Ensure float type for NaN handling

            # Optional normalization - only on non-NaN values
            if normalize == "true":
                # Only normalize if we have valid (non-NaN) values
                valid_mask = ~np.isnan(vals)
                if np.sum(valid_mask) > 0:  # Check if we have any valid values
                    valid_values = vals[valid_mask]
                    
                    # --- ROBUST Z-NORM (Option 1: Median + IQR) ---
                    from scipy.stats import iqr
                    
                    median_val = np.median(valid_values)
                    iqr_val = iqr(valid_values)
                    
                    if iqr_val > 0:
                        # Robust z-norm: use median and IQR instead of mean and std
                        normalized_vals = np.full_like(vals, np.nan)
                        normalized_vals[valid_mask] = (valid_values - median_val) / iqr_val
                        
                        np.set_printoptions(precision=4, suppress=True)
                        # print("raw values", vals)
                        # print(f"  median: {median_val:.4f}, IQR: {iqr_val:.4f}")
                        vals = normalized_vals
                        # print("normalized values", vals)

            # Smooth signal while preserving NaN values
            # vals = smooth_signal(vals)
            participant_data.append(vals)

    elif NORM == 2: # Clip outliers
        for pid, group in df.groupby('participant'):
            # Each row is already region-averaged → just take it
            vals = group[index].tolist()[0]  # only 1 row per participant per region
            vals = np.array(vals, dtype=float)  # Ensure float type for NaN handling

            # Optional normalization - only on non-NaN values
            if normalize == "true":
                # Only normalize if we have valid (non-NaN) values
                valid_mask = ~np.isnan(vals)
                if np.sum(valid_mask) > 0:  # Check if we have any valid values
                    valid_values = vals[valid_mask]
                    ZNORM = 1 # [1 = Z-NORMALIZE, 0 = ONLY CLIP]
                    if ZNORM == 1: 
                        mean_val, std_val = np.mean(valid_values), np.std(valid_values)
                        if std_val > 0:
                            # --- CLIP OUTLIERS BEFORE Z-NORM (Option 2) ---
                            clip_lower = mean_val - CLIP_STD * std_val
                            clip_upper = mean_val + CLIP_STD * std_val
                            clipped_values = np.clip(valid_values, clip_lower, clip_upper)
                            
                            # Recalculate mean and std AFTER clipping
                            mean_clipped = np.mean(clipped_values)
                            std_clipped = np.std(clipped_values)
                            
                            # Z-norm on clipped values
                            normalized_vals = np.full_like(vals, np.nan)
                            if std_clipped > 0:
                                normalized_vals[valid_mask] = (clipped_values - mean_clipped) / std_clipped
                            
                            np.set_printoptions(precision=4, suppress=True)
                            # print("raw values", vals)
                            # print("clipped values", clipped_values)
                            vals = normalized_vals
                            # print("normalized values", vals)
                    elif ZNORM == 0:
                        mean_val = np.mean(valid_values)
                        std_val = np.std(valid_values)
                        
                        if std_val > 0:
                            clip_lower = mean_val - CLIP_STD * std_val
                            clip_upper = mean_val + CLIP_STD * std_val
                            
                            clipped_vals = np.full_like(vals, np.nan)
                            clipped_vals[valid_mask] = np.clip(valid_values, clip_lower, clip_upper)
                            
                            np.set_printoptions(precision=4, suppress=True)
                            # print("raw values", vals)
                            # print(f"  clip range: [{clip_lower:.4f}, {clip_upper:.4f}]")
                            vals = clipped_vals
                            # print("clipped values", vals)

            # Smooth signal while preserving NaN values
            # vals = smooth_signal(vals)
            participant_data.append(vals)

    # --- Index alignment & group average with NaN handling ---
    group_average = align_and_average_with_nan(participant_data)
    
    return group_average

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

def clean_data_for_json(data):
    """Clean data by replacing NaN and infinity values for JSON serialization"""
    if isinstance(data, list):
        return [clean_data_for_json(item) for item in data]
    elif isinstance(data, np.ndarray):
        # Replace NaN and inf with None (which becomes null in JSON)
        cleaned = np.where(np.isfinite(data), data, None)
        return cleaned.tolist()
    elif isinstance(data, (float, np.float64, np.float32)):
        if np.isnan(data) or np.isinf(data):
            return None
        return float(data)
    else:
        return data

def get_eeg_data_by_video_id(video_id: str, normalize: str | None = None):
    engagement_data = get_eeg_summary(video_id=int(video_id), participant_id=participants, region="all", index="engagement", normalize=normalize)
    engagement_list = clean_data_for_json(engagement_data)
    attention_data = get_eeg_summary(video_id=int(video_id), participant_id=participants, region="all", index="attention", normalize=normalize)
    attention_list = clean_data_for_json(attention_data)
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
                        "engagement": clean_data_for_json(engagement),
                        "attention": clean_data_for_json(attention),
                        "region": r,
                        "participant": "all"
                    }
                    print(f"Computed all participants for region {r}")
            else:
                engagement = get_eeg_summary(video, participant, "all", index="engagement", normalize=normalize)
                attention = get_eeg_summary(video, participant, "all", index="attention", normalize=normalize)
                results["all"] = {
                    "engagement": clean_data_for_json(engagement),
                    "attention": clean_data_for_json(attention),
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
                            "engagement": clean_data_for_json(engagement),
                            "attention": clean_data_for_json(attention),
                            "region": r,
                            "participant": p
                        }
                        print(f"Computed participant {p} for region {r}")
                else:
                    engagement = get_eeg_summary(video, p, "all", index="engagement", normalize=normalize)
                    attention = get_eeg_summary(video, p, "all", index="attention", normalize=normalize)
                    # all regions for participant P
                    results[f"{p}"] = {
                        "engagement": clean_data_for_json(engagement),
                        "attention": clean_data_for_json(attention),
                        "region": "all",
                        "participant": p
                    }
                    print(f"Computed participant {p} for all regions")
        
        # Get a sample length from the first result for metadata
        sample_length = 0
        if results:
            first_key = next(iter(results.keys()))
            sample_length = len(results[first_key]["engagement"])
        
        results["meta"] = {
            "video": video,
            "participants": participant,
            "regions": region if region else ["all"],
            "normalize": normalize,
            "length": sample_length
        }
        
        return JSONResponse(content=results)
    except ValueError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})