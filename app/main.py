"""FastAPI backend for the EEG Data Video (EEGDV) Analysis Tool.

Serves the video player UI and exposes JSON APIs for EEG engagement/attention
time series, aligned and averaged across participants, with optional
per-participant normalization.
"""
import logging
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

load_dotenv()

logger = logging.getLogger("eegdv")
logging.basicConfig(level=logging.INFO)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(__file__)
STATIC_PATH = os.path.join(BASE_DIR, "static")

# DATA_DIR points at the (gitignored) folder holding participant data and
# videos; configure it in .env (not committed). Mounted at /static/data
# ahead of the general /static mount so it resolves correctly even if it
# lives outside app/static.
DATA_DIR = os.getenv("DATA_DIR", os.path.join("app", "static", "data"))
app.mount("/static/data", StaticFiles(directory=DATA_DIR), name="data")
app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")
templates = Jinja2Templates(directory="app/templates")

# --- Data file paths ---
VIDEO_CSV_PATH = os.path.join(DATA_DIR, "videos.csv")
VIDEO_STIMULI_CSV_PATH = os.path.join(DATA_DIR, "video_stimuli.csv")

# Precomputed EEG engagement/attention series: cross-baseline-subtracted,
# power-normalized, using a video-specific baseline, with no end-trim.
# Covers both mDVs and long DVs (mDV+LDV).
EEG_DATA_CSV_PATH = os.path.join(
    DATA_DIR, "video_grouped_indices_REGIONS_CROSS_WELCH_RATIO_ALL_DVs.csv" # video_grouped_indices_REGIONS_CROSS_WELCH_RATIO_ALL_DVs eeg_cross_power_normalized_avgpower_no-end-trim - mDV+LDV
) # EEG_RAW_CROSS_BASELINE_SUBTRACTED_CSV_PATH_TRIMMED_AVG_POWER_NORM_VIDEO_SPECIFIC_BASELINE_NO_END_TRIM_ALL_DVS

# --- Normalization config ---
# Method used to rescale each participant's series onto a comparable range
# before averaging across participants. See `_normalize_participant_values`.
NORMALIZATION_METHOD = "clip_zscore"  # one of: zscore, robust_zscore, clip_zscore, clip_only
CLIP_STD = 2  # outlier clipping threshold, in standard deviations (clip_zscore / clip_only)

# Set to False to hide the Attention index in the player UI (dev toggle).
SHOW_ATTENTION = False

# TODO: extract from a database or config file once participant lists grow further.
PARTICIPANTS = [105, 107, 2, 7, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]

STIMULI = {
    0: {"id": 0, "label": "Sound effect", "abbrev": "SE", "color": "#C2CDFF"},
    1: {"id": 1, "label": "Zooming", "abbrev": "ZM", "color": "#FFF3C2"},
    2: {"id": 2, "label": "Pointing", "abbrev": "PT", "color": "#D9F4CD"},
    3: {"id": 3, "label": "Highlighting", "abbrev": "HL", "color": "#C7FAE9"},
    4: {"id": 4, "label": "Gradual Text", "abbrev": "GT", "color": "#FFC2E8"},
    5: {"id": 5, "label": "Gradual appearance", "abbrev": "GA", "color": "#EBD5FF"},
    6: {"id": 6, "label": "Camera Angle", "abbrev": "CA", "color": "#A9CEFF"},
    7: {"id": 7, "label": "Animation", "abbrev": "AN", "color": "#FFADAD"},
    10: {"id": 10, "label": "Data Visualization", "abbrev": "DVIS", "color": "#FFD699"},
    11: {"id": 11, "label": "Size Change", "abbrev": "SC", "color": "#DBD5D5"},
    13: {"id": 13, "label": "Abrupt Transition", "abbrev": "ATR", "color": "#FFB88B"},
    14: {"id": 14, "label": "Gradual Transition", "abbrev": "GTR", "color": "#C5A8FF"},
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def to_list(value):
    """Parse a stringified Python list (e.g. "[1.0, nan, 2.0]") back into a list.

    Returns the value unchanged if it isn't a string, and `[nan]` if parsing fails.
    """
    if not isinstance(value, str):
        return value
    try:
        # Restricted eval: only "nan"/"np" are exposed, no builtins.
        return eval(value, {"__builtins__": {}, "nan": np.nan, "np": np})
    except Exception:
        return [np.nan]


def align_and_average_with_nan(participant_series):
    """Trim a list of per-participant arrays to a common length and average
    across participants, ignoring NaNs.

    Parameters
    ----------
    participant_series : list of np.ndarray
        Per-participant time series, potentially of different lengths and
        containing NaNs.

    Returns
    -------
    np.ndarray
        Group-level average time series (empty array if input is empty).
    """
    if not participant_series:
        return np.array([])

    min_len = min(len(series) for series in participant_series)
    trimmed = np.array([series[:min_len] for series in participant_series])
    return np.nanmean(trimmed, axis=0)


def _normalize_participant_values(values, method):
    """Rescale a single participant's series using `method`, ignoring NaNs
    and preserving their positions.

    Methods
    -------
    zscore        : standard (mean, std) z-normalization.
    robust_zscore : z-normalization using median and IQR instead of mean/std,
                    more robust to outliers.
    clip_zscore   : clip outliers to +/-CLIP_STD std devs, then z-normalize
                    the clipped values. (Current default.)
    clip_only     : clip outliers to +/-CLIP_STD std devs without normalizing.

    Returns the original array unchanged if there are no valid values, or if
    the spread (std/IQR) is zero.
    """
    valid_mask = ~np.isnan(values)
    if not np.any(valid_mask):
        return values
    valid = values[valid_mask]

    if method == "zscore":
        mean, std = np.mean(valid), np.std(valid)
        if std <= 0:
            return values
        out = np.full_like(values, np.nan)
        out[valid_mask] = (valid - mean) / std
        return out

    if method == "robust_zscore":
        from scipy.stats import iqr as scipy_iqr

        median, spread = np.median(valid), scipy_iqr(valid)
        if spread <= 0:
            return values
        out = np.full_like(values, np.nan)
        out[valid_mask] = (valid - median) / spread
        return out

    if method == "clip_zscore":
        mean, std = np.mean(valid), np.std(valid)
        if std <= 0:
            return values
        clipped = np.clip(valid, mean - CLIP_STD * std, mean + CLIP_STD * std)
        clipped_mean, clipped_std = np.mean(clipped), np.std(clipped)
        if clipped_std <= 0:
            return values
        out = np.full_like(values, np.nan)
        out[valid_mask] = (clipped - clipped_mean) / clipped_std
        return out

    if method == "clip_only":
        mean, std = np.mean(valid), np.std(valid)
        if std <= 0:
            return values
        out = np.full_like(values, np.nan)
        out[valid_mask] = np.clip(valid, mean - CLIP_STD * std, mean + CLIP_STD * std)
        return out

    raise ValueError(f"Unknown normalization method: {method!r}")


EEG_INDICES = ("engagement", "attention")


def get_eeg_series(video_id: int, participant_id: list | int, region: str = "all",
                    normalize: str = "false"):
    """Summarize engagement AND attention across participant(s) for one video
    and region in a single pass, optionally normalizing each participant's
    series before averaging. Reads from the precomputed EEG CSV.

    Parameters
    ----------
    video_id : int
        Video identifier.
    participant_id : int or list[int]
        A single participant ID, or a list of them (see `PARTICIPANTS`).
    region : str
        Brain region: 'all', 'frontal', 'frontocentral', 'parietal',
        'occipital', 'other'. 'all' matches the precomputed "all" rows.
    normalize : str
        'true' to normalize each participant's series (via
        `NORMALIZATION_METHOD`) before averaging; anything else skips it.

    Returns
    -------
    dict[str, np.ndarray]
        {"engagement": ..., "attention": ...} group-level average time
        series, aligned across participants with NaN handling.
    """
    print(f"[get_eeg_series] video={video_id} region={region} "
          f"participants={participant_id} normalize={normalize}")

    df = pd.read_csv(EEG_DATA_CSV_PATH)
    for idx in EEG_INDICES:
        df[idx] = df[idx].apply(to_list)

    df = df[df["video"] == video_id]
    if isinstance(participant_id, int):
        df = df[df["participant"] == participant_id]
    else:
        df = df[df["participant"].isin(participant_id)]
    df = df[df["region"].str.lower() == region.lower()]

    if df.empty:
        raise ValueError("No data found for given parameters.")

    print(f"[get_eeg_series] {df['participant'].nunique()} participant row(s) matched")

    participant_series = {idx: [] for idx in EEG_INDICES}
    for pid, group in df.groupby("participant"):
        # Each row is already region-averaged, so there's exactly one row
        # per participant per region.
        lengths = []
        for idx in EEG_INDICES:
            vals = np.array(group[idx].tolist()[0], dtype=float)
            if normalize == "true":
                vals = _normalize_participant_values(vals, NORMALIZATION_METHOD)
            participant_series[idx].append(vals)
            lengths.append(f"{idx}={len(vals)}")
        print(f"[get_eeg_series]   participant={pid} {' '.join(lengths)}")

    result = {idx: align_and_average_with_nan(participant_series[idx]) for idx in EEG_INDICES}
    print(f"[get_eeg_series] group average length={len(result['engagement'])}")
    return result


def clean_data_for_json(data):
    """Recursively replace NaN/Inf with None so data can be safely JSON-serialized."""
    if isinstance(data, list):
        return [clean_data_for_json(item) for item in data]
    if isinstance(data, np.ndarray):
        return np.where(np.isfinite(data), data, None).tolist()
    if isinstance(data, (float, np.float64, np.float32)):
        return None if (np.isnan(data) or np.isinf(data)) else float(data)
    return data


def get_video_by_video_id(video_id: str):
    """Look up a single video's metadata row by ID."""
    df = pd.read_csv(VIDEO_CSV_PATH)
    matches = df[df["id"] == int(video_id)].to_dict(orient="records")
    if not matches:
        return HTMLResponse("Video not found", status_code=404)
    return matches[0]


def get_stimuli_by_video_id(video_id: str):
    """Return the raw stimuli-event rows and the distinct stimulus type
    definitions (label/abbrev/color) that occur in a given video."""
    video_stimuli = pd.read_csv(VIDEO_STIMULI_CSV_PATH)
    stimuli_list = video_stimuli[video_stimuli["video_id"] == int(video_id)].to_dict(orient="records")
    unique_stimulus_ids = {item["stimulus_id"] for item in stimuli_list}
    stimulus_types = [STIMULI[s] for s in unique_stimulus_ids]
    return stimuli_list, stimulus_types


def get_eeg_data_by_video_id(video_id: str, normalize: str | None = None):
    """Fetch group-averaged engagement and attention series (all participants,
    all regions) for a video, ready for JSON serialization."""
    series = get_eeg_series(int(video_id), PARTICIPANTS, region="all", normalize=normalize)
    return clean_data_for_json(series["engagement"]), clean_data_for_json(series["attention"])


def _build_eeg_result_entry(video, participant_spec, participant_label, region, normalize):
    """Compute engagement/attention series for one participant-spec/region
    combination, in the shape expected by the /api/eeg response."""
    series = get_eeg_series(video, participant_spec, region, normalize=normalize)
    return {
        "engagement": clean_data_for_json(series["engagement"]),
        "attention": clean_data_for_json(series["attention"]),
        "region": region,
        "participant": participant_label,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Index page listing all videos."""
    videos = pd.read_csv(VIDEO_CSV_PATH).to_dict(orient="records") if os.path.exists(VIDEO_CSV_PATH) else []
    print(f"[/ read_root] listing {len(videos)} video(s)")
    return templates.TemplateResponse("index.html", {"request": request, "videos": videos})


@app.get("/player", response_class=HTMLResponse)
async def player(request: Request, id: str, normalize: bool = Query(False)):
    """Video player page with synchronized EEG engagement/attention overlays
    and stimulus annotations."""
    print(f"[/player] video_id={id} normalize={normalize}")
    video = get_video_by_video_id(id)
    stimuli_list, stimulus_types = get_stimuli_by_video_id(id)
    print(f"[/player] {len(stimuli_list)} stimulus event(s), {len(stimulus_types)} stimulus type(s)")
    eng_data, atn_data = get_eeg_data_by_video_id(id, str(normalize).lower())
    print(f"[/player] engagement length={len(eng_data)} attention length={len(atn_data)}")
    logger.info("Loaded player data for video %s: engagement=%d, attention=%d",
                id, len(eng_data), len(atn_data))
    return templates.TemplateResponse("player.html", {
        "request": request,
        "video": video,
        "eng_data": eng_data,
        "atn_data": atn_data,
        "stimuli_data": stimuli_list,
        "stimuli_info": stimulus_types,
        "participants": PARTICIPANTS,
        "show_attention": SHOW_ATTENTION,
    })


@app.get("/api/stimuli")
def get_video_stimuli(video_id: int):
    """Raw stimulus-event rows for a video."""
    print(f"[/api/stimuli] video_id={video_id}")
    if not os.path.exists(VIDEO_STIMULI_CSV_PATH):
        return []
    try:
        stimuli_df = pd.read_csv(VIDEO_STIMULI_CSV_PATH)
        filtered = stimuli_df[stimuli_df["video_id"] == video_id]
        print(f"[/api/stimuli] {len(filtered)} row(s) found")
        return JSONResponse(content=filtered.to_dict(orient="records"))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/eeg")
def get_filtered_eeg_api(
    video: int,
    participant: list[int] = Query(None),
    region: list[str] = Query(None),
    normalize: str = Query("false"),
    merge: str = Query("false"),
):
    """Group-averaged (or per-participant) EEG engagement/attention series
    for a video, optionally split by region and/or normalized.

    - merge='true': averages across all requested participants into one series.
    - merge='false' (default): returns one series per participant.
    - region: if provided, computes one entry per region; otherwise 'all'.
    """
    try:
        participants = participant if participant is not None else PARTICIPANTS
        regions = region  # None means "all regions" (single "all" entry)

        print(f"[/api/eeg] video={video} merge={merge} normalize={normalize} "
              f"participants={participants} regions={regions or ['all']}")

        if merge == "true":
            print(f"[/api/eeg] merging across {len(participants)} participant(s)")
            participant_groups = [(participants, "all")]
        else:
            print(f"[/api/eeg] computing per-participant results for {len(participants)} participant(s)")
            participant_groups = [(p, p) for p in participants]

        results = {}
        for participant_spec, label in participant_groups:
            key_prefix = str(label)
            if regions:
                for r in regions:
                    print(f"[/api/eeg] computing entry '{key_prefix}_{r}'")
                    results[f"{key_prefix}_{r}"] = _build_eeg_result_entry(
                        video, participant_spec, label, r, normalize)
            else:
                print(f"[/api/eeg] computing entry '{key_prefix}' (region=all)")
                results[key_prefix] = _build_eeg_result_entry(
                    video, participant_spec, label, "all", normalize)

        sample_length = len(next(iter(results.values()))["engagement"]) if results else 0
        print(f"[/api/eeg] done: {len(results)} entrie(s), sample length={sample_length}")
        results["meta"] = {
            "video": video,
            "participants": participants,
            "regions": regions if regions else ["all"],
            "normalize": normalize,
            "length": sample_length,
        }
        return JSONResponse(content=results)
    except ValueError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})
