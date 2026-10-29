from pathlib import Path
from io import BytesIO
from pathlib import Path
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image
from ultralytics import YOLO
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse

# ==========================================
# 1. Project Configuration
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "road_damage_best.pt"

CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 640


# ==========================================
# 2. Initialize FastAPI
# ==========================================

app = FastAPI(
    title="Road Damage Detection API",
    description=(
        "Upload a road image to detect road damage "
        "using a trained YOLO11n model."
    ),
    version="1.0.0",
)
BASE_DIR = Path(__file__).resolve().parent

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static"
)

# ==========================================
# 3. Load the Trained YOLO Model
# ==========================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file was not found: {MODEL_PATH}"
    )

model = YOLO(str(MODEL_PATH))


# ==========================================
# 4. Home Endpoint
# ==========================================

@app.get("/")
async def home():
    return FileResponse(BASE_DIR / "static" / "index.html")


# ==========================================
# 5. Health Check Endpoint
# ==========================================

@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "model": "road_damage_best.pt",
        "available_classes": model.names
    }


# ==========================================
# 6. Predict Road Damage
# ==========================================

@app.post("/predict")
async def predict_road_damage(
    file: UploadFile = File(...)
):
    # Validate uploaded file type
    if file.content_type not in {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/bmp",
    }:
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid JPG, PNG, WEBP, or BMP image.",
        )

    try:
        # Read uploaded image
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        # Convert image to RGB
        image = Image.open(
            BytesIO(contents)
        ).convert("RGB")

        image_array = np.array(image)

        # Run YOLO prediction
        results = model.predict(
            source=image_array,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )

        result = results[0]

        # Extract detections
        detections = []

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())

                x1, y1, x2, y2 = (
                    float(value)
                    for value in box.xyxy[0].tolist()
                )

                detections.append({
                    "class_id": class_id,
                    "class_name": model.names[class_id],
                    "confidence": round(confidence, 4),
                    "bounding_box": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                    },
                })

        # Generate image with annotated detections
        annotated_image = result.plot()

        # Convert annotated image to JPEG bytes
        from PIL import Image as PILImage

        annotated_rgb = annotated_image[:, :, ::-1]

        output_image = BytesIO()

        PILImage.fromarray(
            annotated_rgb
        ).save(
            output_image,
            format="JPEG",
            quality=90,
        )

        output_image.seek(0)

        # Return prediction results and annotated image
        return {
            "filename": file.filename,
            "image_width": image.width,
            "image_height": image.height,
            "total_detections": len(detections),
            "detections": detections,
            "annotated_image_endpoint": "/predict/image",
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}",
        )

@app.post("/predict/image")
async def predict_road_damage_image(
    file: UploadFile = File(...)
):
    if file.content_type not in {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/bmp",
    }:
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid JPG, PNG, WEBP, or BMP image.",
        )

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        image = Image.open(
            BytesIO(contents)
        ).convert("RGB")

        image_array = np.array(image)

        results = model.predict(
            source=image_array,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )

        result = results[0]

        # Draw bounding boxes and labels
        annotated_image = result.plot()

        # Convert BGR to RGB
        annotated_rgb = annotated_image[:, :, ::-1]

        # Encode the image as JPEG
        output_image = BytesIO()

        Image.fromarray(
            annotated_rgb
        ).save(
            output_image,
            format="JPEG",
            quality=95,
        )

        output_image.seek(0)

        return StreamingResponse(
            output_image,
            media_type="image/jpeg",
            headers={
                "Content-Disposition": (
                    'inline; filename="road_damage_result.jpg"'
                )
            },
        )

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Image prediction failed: {str(error)}",
        )