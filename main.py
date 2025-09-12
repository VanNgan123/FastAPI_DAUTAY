from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import HTTPException
import io
from ultralytics import YOLO
import os
import shutil
from starlette.responses import FileResponse, JSONResponse


dir_model = r"best.pt"

model= YOLO(dir_model)


app =FastAPI()



UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    # Lưu file upload tạm thời
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Chạy YOLO dự đoán
    results = model.predict(source=file_path, conf=0.2, save=True, project=RESULT_DIR, name="run")

    # YOLO tự lưu ảnh kết quả vào thư mục runs, mình lấy ảnh đầu tiên ra
    save_dir = results[0].save_dir
    result_image = os.path.join(save_dir, file.filename)

    # Trả về ảnh kết quả
    if os.path.exists(result_image):
        return FileResponse(result_image, media_type="image/jpeg")
    else:
        return JSONResponse(content={"error": "Không tìm thấy ảnh kết quả."})