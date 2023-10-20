# Initialize Vertex AI
import vertexai
import logging
import base64
import shutil
import tempfile
import requests
from vertexai.vision_models import ImageCaptioningModel
from vertexai.vision_models import Image
from vertexai.language_models import TextGenerationModel
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel

app = FastAPI()
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

class CaptionByImage(BaseModel):
    img_url: str
    language: str

class CaptionByTitle(BaseModel):
    title: str
    language: str

class HealthCheck(BaseModel):
    status: str = "OK"

@app.get("/health")
def get_health() -> HealthCheck:
    return HealthCheck(status="OK")

@app.post("/desc_from_img")
def description_from_image(request: CaptionByImage):
    """Get the caption for an image.

    Args:
        image: The image to be captioned.
        language: The language in which the caption should be returned.

    Returns:
        The caption for the image.
    """
    PROJECT_ID = "vital-octagon-19612" # @param {type:"string"}
    LOCATION = "us-central1" # @param {type:"string"}
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    image_captioning_model = ImageCaptioningModel.from_pretrained("imagetext@001")
    # Load the image file as Image object
    
    image = requests.get(request.img_url, stream=True)
    with tempfile.NamedTemporaryFile() as f:
        shutil.copyfileobj(image.raw, f)
        img=Image.load_from_file(f.name)
    #logging.info(language)
    # Get the caption for the image
    response = image_captioning_model.get_captions(
        image=img,
        number_of_results=1,
        language=request.language,
    )
    # Return the caption
    return response


@app.post("/desc_from_title")
def description_from_title(request: CaptionByTitle):
    """Get the caption for an image.

    Args:
        title: The title for which description to be generated.
        language: The language in which the description should be returned.

    Returns:
        The description for the title.
    """
    PROJECT_ID = "vital-octagon-19612" # @param {type:"string"}
    LOCATION = "us-central1" # @param {type:"string"}
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    desc_generation_model = TextGenerationModel.from_pretrained("text-bison@001")
    # Load the image file as Image object
    prompt=""" 
        Generate a short Online Product Description 
        Stands out from the competition.
        Improve search engine optimization (SEO) and organic traffic 
        Increase conversion rates
        Helps customer understand value of product  
        Limit returns for
        """ + request.title + """
        in the language""" + request.language
    #logging.info(language)
    # Get the caption for the image
    response = desc_generation_model.predict(
        prompt=prompt,
        max_output_tokens=256,
        temperature=0.2,
    ).text
    # Return the caption
    return response
