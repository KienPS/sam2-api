import io
import os
import numpy as np
from typing import List
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from PIL import Image
from dotenv import load_dotenv
import torch
from ultralytics.models.sam import SAM2DynamicInteractivePredictor

load_dotenv()

app = FastAPI()

overrides = dict(
    conf=float(os.getenv("SAM2_CONF", "0.01")),
    task=os.getenv("SAM2_TASK", "segment"),
    mode=os.getenv("SAM2_MODE", "predict"),
    imgsz=int(os.getenv("SAM2_IMGSZ", "1024")),
    model=os.getenv("SAM2_MODEL", "sam2_t.pt"),
    save=os.getenv("SAM2_SAVE", "false").lower() == "true"
)
predictor = SAM2DynamicInteractivePredictor(overrides=overrides)


class BoxInput(BaseModel):
    boxes: List[List[float]]


class BoxOutput(BaseModel):
    boxes: List[List[float]]


def image_to_numpy(file: UploadFile) -> np.ndarray:
    image = Image.open(io.BytesIO(file.file.read()))
    return np.array(image)


@app.post("/predict")
async def predict(
    current_frame: UploadFile = File(...),
    boxes: str = Form(...),
    next_frame: UploadFile = File(...)
) -> BoxOutput:
    import json
    
    current_frame_np = image_to_numpy(current_frame)
    next_frame_np = image_to_numpy(next_frame)
    
    boxes_list = json.loads(boxes)
    num_boxes = len(boxes_list)
    
    obj_ids = list(range(num_boxes))
    max_obj_num = num_boxes
    
    predictor.max_obj_num = max_obj_num
    
    results = predictor(
        source=current_frame_np,
        bboxes=boxes_list,
        obj_ids=obj_ids,
        update_memory=True
    )
    
    results = predictor(source=next_frame_np)
    predictor.reset_image()
    torch.cuda.empty_cache()

    output_boxes = []
    for obj_id in obj_ids:
        bbox = results[0][obj_id].boxes.xyxy
        bbox_array = bbox.tolist() if hasattr(bbox, 'tolist') else list(bbox)
        if isinstance(bbox_array, list) and len(bbox_array) == 1:
            output_boxes.append(bbox_array[0])
        else:
            output_boxes.append(bbox_array)
    
    return BoxOutput(boxes=output_boxes)
