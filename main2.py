from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Load the YOLOv8 model
model = YOLO("harras.pt")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the image file
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    image = np.array(image)  # Convert PIL image to numpy array
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

    # Perform prediction
    results = model(image)

    # Parse results
    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            class_name = model.names[class_id]
            confidence = float(box.conf)
            bbox = box.xyxy[0].tolist()  # Get bounding box coordinates

            detections.append({
                "class": class_name,
                "confidence": confidence,
                "bbox": bbox
            })

    return {"detections": detections}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)