import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import json
import copy
import torch
import base64
import uvicorn
import traceback
import threading
import mimetypes
from queue import Queue
from pathlib import Path
from loguru import logger
from setting import setting
from typing import Optional
from pydantic import BaseModel
from models.Embedding import Embedding
from face_processor import process_face
from fastapi.responses import FileResponse, Response
from utils.bicubic import BicubicDownSample
from utils.fan import extract_face_landmarks
from utils.model_utils import download_weight
from datasets.image_dataset import ImagesDataset
from fastapi.middleware.cors import CORSMiddleware
from models.Tunning import FineTuneContext, Tunning, AblationConfig
from utils.helpers import COLOR_MAP, NOSE_REGION, SKIN_REGION
from models.face_parsing.model import BiSeNet, seg_mean, seg_std
from file_process import get_all, error_field, set_inversion_image
from fastapi import FastAPI, Request, UploadFile, File, HTTPException, BackgroundTasks, Query

logger.add("app.log", rotation="5 MB", retention="10 days", level="INFO")
logger.info("🚀 Nose-AI FastAPI starting...")

app = FastAPI(
    title="Nose-AI API",
    description="A minimal studio for nose manipulation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",  
    openapi_url="/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],      # Allow all HTTP methods
    allow_headers=["*"],      # Allow all headers
)

configDetails = copy.deepcopy(setting)   

maxWorkers = 1
currentWorkers = 0
imageQueue = Queue()

downsample = BicubicDownSample(factor=setting.size // 512)

seg = BiSeNet(n_classes=16)
seg.to(setting.device)

if not os.path.exists(setting.seg_ckpt):
    download_weight(setting.seg_ckpt)
seg.load_state_dict(torch.load(setting.seg_ckpt))
for param in seg.parameters():
    param.requires_grad = False
seg.eval()


ii2s = Embedding(copy.deepcopy(setting))
def process_inversion(image_name:str):  
    try:
        global ii2s
        global imageQueue
        global currentWorkers

        torch.cuda.empty_cache()

        ii2s.invert_image(image_name)

        torch.cuda.empty_cache()

        if(imageQueue.empty()):
          currentWorkers = currentWorkers - 1
        else:
          process_inversion(imageQueue.get())

    except Exception as e: 
        print("Exception occurred:")
        traceback.print_exc()  # 🔍 shows full traceback
        logger.error(f"Inversion processing error: {e}")
        error_field(image_name)
        if(imageQueue.empty()):
            currentWorkers = currentWorkers - 1
        else:
            process_inversion(imageQueue.get())

def start_inversion(image_name:str):
    thread = threading.Thread(target=process_inversion, args = (image_name,))
    thread.start()

@app.get("/get-image")
def get_image(im_name: str = Query(..., description="Image path inside images-outputs folder")):
    image_path = os.path.join(setting.output_dir, im_name)
    if not os.path.exists(image_path):
        return {"error": f"Image '{im_name}' not found"}

    mime_type, _ = mimetypes.guess_type(image_path)
    with open(image_path, "rb") as f:
        data = f.read()
    return Response(content=data, media_type=mime_type)


class FineTuneRequest(BaseModel):
    fullPath: Optional[str] = None
    noseStyle: Optional[str] = None
    landmarks: Optional[str] = None
    segmentation: Optional[str] = None
    ablation: Optional[AblationConfig] = None   # ablation-study override; omitted by the frontend

tunning = Tunning(setting)
def process_fine_tune(item:FineTuneContext):  
    try:
        global imageQueue
        global currentWorkers

        output_folder_path = os.path.join(Path(item.ref_path).parent, "tuned")
        os.makedirs(output_folder_path, exist_ok=True)
        
        torch.cuda.empty_cache()

        tunning.tune_image(item)

        torch.cuda.empty_cache()

        if(imageQueue.empty()):
          currentWorkers = currentWorkers - 1
        else:
          process_fine_tune(imageQueue.get())
        
    except Exception as e: 
        print(e)
        logger.error(f"fine tunning processing error: {e}")

        error_field(item.inversion_name)

        if(imageQueue.empty()):
            currentWorkers = currentWorkers - 1
        else:
            process_fine_tune(imageQueue.get())


def start_fine_tune(item : FineTuneContext):
    thread = threading.Thread(target=process_fine_tune, args = (item,))
    thread.start()

@app.post("/fine-tune")
async def image_fine_tune(
    background_tasks: BackgroundTasks, body: FineTuneRequest):
    fullPath = body.fullPath
    noseStyle = body.noseStyle
    landmarks = body.landmarks
    segmentation = body.segmentation

    if segmentation is None and landmarks is None and ( noseStyle is None or noseStyle == fullPath ) :
        print("No Tune parameter is provided !!!!!!")
        return "Please send tunning parameter to tune the image based on it"


    try:
        if landmarks is not None:
            landmarks = json.loads(landmarks)  # should be like "[[1,2,3,1], [4,5,6,0]]"
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format for mask or landmarks")
    
    try:
        if segmentation is not None:
            segmentation = json.loads(segmentation)  # should be like 512  512
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format for mask or segmentation")


    ref_full_path = os.path.join(setting.output_dir, fullPath)
    
    if not fullPath or not os.path.exists(ref_full_path):
        raise HTTPException(status_code=400, detail="Ref Image file does not exist.")

    full_nose_path = None
    if noseStyle != 'self' and noseStyle != fullPath:
        full_nose_path = os.path.join(setting.output_dir, noseStyle)

        if not noseStyle or not os.path.exists(full_nose_path):
            raise HTTPException(status_code=400, detail="Nose Image file does not exist.")

    
    global maxWorkers
    global imageQueue
    global currentWorkers

    if full_nose_path == "self" or ref_full_path == full_nose_path or full_nose_path is None:
        inversion_name =  "Tunning-" + Path(ref_full_path).stem
        inversion = {
        "current_step": 0,        
        "total_steps": setting.Tune_steps,
        }
    else:
        inversion = {
        "current_step": 0,        
        "total_steps": setting.Transfer_restructure_steps + setting.Transfer_perceptual_steps,
        }
        inversion_name =  "Transfer-" + Path(ref_full_path).stem + Path(full_nose_path).stem

    if body.ablation is not None:
        inversion_name = inversion_name + "-" + body.ablation.name

    set_inversion_image(inversion_name, inversion)
    
    imageItem = FineTuneContext(inversion_name = inversion_name, ref_path = ref_full_path, nose_path = full_nose_path, landmarks = landmarks, segmentation = segmentation, ablation = body.ablation)

    if (currentWorkers == maxWorkers):
        imageQueue.put(imageItem)
        return "Queued"
    else: 
        currentWorkers = currentWorkers + 1
        background_tasks.add_task(start_fine_tune, imageItem)
        return "Started"


@app.get("/image-data")
def get_image_data(im_name: str = Query(..., description="Image path inside images-outputs folder")):
    image_path = os.path.join(setting.output_dir, im_name)
    
    if not os.path.exists(image_path):
        return {"error": f"Image '{im_name}' not found"}

    with open(image_path, "rb") as f:
        buffer = f.read()

    ext = os.path.splitext(image_path)[1][1:]  # e.g. 'png'
    base64_data = base64.b64encode(buffer).decode("utf-8")

    base64_image = f"data:image/{ext};base64,{base64_data}"

    global seg, downsample  

    dataset = ImagesDataset(opts=setting, image_path=image_path)

    img_H, _, _ = dataset[0]

    im_tensor = img_H.unsqueeze(0).to(setting.device)

    im = ((im_tensor[0] + 1) / 2).detach().cpu().clamp(0, 1)
    im = (downsample(im_tensor).clamp(0, 1) - seg_mean) / seg_std
    segmentted_im, _, _ = seg(im)
    segmentted_im = torch.argmax(segmentted_im, dim=1).long().squeeze(0)

    segmentationData =  {
        "segmentedImage": segmentted_im.detach().cpu().numpy().tolist(),
        "COLOR_MAP": COLOR_MAP.tolist(),
        "regions" : {
            "nose": NOSE_REGION,
            "skin": SKIN_REGION,
            } 
        }

    coords_tensor = extract_face_landmarks(
        image_path=image_path,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    return {
        "image": base64_image,
        "segmentationData": segmentationData,
        "landmarks": coords_tensor.detach().cpu().tolist(),
      }

@app.post("/upload-image")
async def upload_image(background_tasks: BackgroundTasks, image: UploadFile = File(...)):
    allowed_types = ["image/png", "image/jpeg"]
    
    if image.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Only PNG and JPG images are allowed")

    file_path = os.path.join(setting.unprocessed_dir, image.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await image.read())

    process_face(file_path)

    global maxWorkers
    global imageQueue
    global currentWorkers

    inversion = {
    "current_step": 0,        
    "total_steps": setting.W_steps + setting.FS_steps,
    }

    image_name = Path(file_path).stem

    set_inversion_image(image_name, inversion)
    
    if (currentWorkers == maxWorkers):
        imageQueue.put(image_name)
        return "Queued"
    else: 
        currentWorkers = currentWorkers + 1
        background_tasks.add_task(start_inversion, image_name)
        return "Started"

@app.get("/get-images")
def get_images():
    extensions = {".png"}

    BASE_IMAGE_DIR = Path(setting.output_dir)

    image_files = [
        str(path.relative_to(BASE_IMAGE_DIR))
        for path in BASE_IMAGE_DIR.rglob("*")
        if path.suffix.lower() in extensions
    ]

    return image_files

@app.get("/processes")
def files_status(request: Request):
    return get_all()

@app.get("/", tags=["Welcome"])
def welcome(request: Request):
    print(request.client.host)
    logger.info(f"Home called by:: {request.client.host}")
    
    return "Hi Welcome to Nose Manipulation!!"


@app.get("/health", tags=["System"])
def health_check():
    logger.info("Health check endpoint called.")
    return {"status": "ok", "message": "System healthy ✅"}

# -----------------------------------------------------
# Run server
# -----------------------------------------------------
if __name__ == "__main__":
    logger.info("✅ Listening on 0.0.0.0:3001")
    uvicorn.run("main:app", host="0.0.0.0", port=3001, reload=False, access_log=False)