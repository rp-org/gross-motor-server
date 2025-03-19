from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from gross import predict_video, predict_action
import json
import os

# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change this for security)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

TEMP_DIR = "temp_dir"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/predict-gross")
async def gross_assess_endpoint(file: UploadFile = File(...), action: str = Form(...)):
    print('File', file)
    print('Expected Action: ', action)

    # ensure temp_dir exists
    os.makedirs(TEMP_DIR, exist_ok=True)

    # define video path
    video_path = os.path.join(TEMP_DIR, f"temp_{file.filename}")
    
    # save uploaded video file to the temp_dir folder
    video_bytes = await file.read()

    with open(video_path, "wb") as f:
        f.write(video_bytes)

    # assess the video
    result = predict_video(video_path, action)

    # cleanup - delete file after processing
    os.remove(video_path)

    # return result
    return JSONResponse(content=result)

# test api
@app.get("/")
def read_root():
    return {"message": "Hello, World!"}