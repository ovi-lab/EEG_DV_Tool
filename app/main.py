from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import pandas as pd
import os

app = FastAPI()
base_dir = os.path.dirname(__file__)
static_path = os.path.join(base_dir, "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")
templates = Jinja2Templates(directory="app/templates")

VALID_CSV_PATH = os.path.join("app", "static", "data", "videos.csv")

def get_videos():
    if os.path.exists(VALID_CSV_PATH):
        df = pd.read_csv(VALID_CSV_PATH)
        return df.to_dict(orient='records')
    else:
        return []

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    videos = get_videos()
    return templates.TemplateResponse("index.html", {"request": request, "videos": videos})



# def get_eeg_summary(video_id: int, region: str = "all", index: str = "engagement", particpant_id: int = 0):
# @Hamza please complete the function below
# This function calculates the signal - your function needs to filter the data in eeg.csv based on the video_id, region, index, and participant_id.
    
# @app.get("/eeg-summary/{video_id}")