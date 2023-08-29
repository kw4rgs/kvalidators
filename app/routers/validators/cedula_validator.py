from fastapi import APIRouter, status, HTTPException
import cv2
import numpy as np
import base64
import pytesseract
import re
from fastapi.responses import JSONResponse

# Main method
class CedulaBackValidator:
    def __init__(self):
        self.logo_images = self.load_references()
        self.threshold = 0.70

    def load_references(self):
        dnrpa_logo = cv2.imread('app/assets/kw4rgs_logo_collection/cedula_back/cd_dnrpa.png', cv2.IMREAD_UNCHANGED)
        cedula_logo = cv2.imread('app/assets/kw4rgs_logo_collection/cedula_back/cd_logo.png', cv2.IMREAD_UNCHANGED)
        return [dnrpa_logo, cedula_logo]

    def validate(self, image_data):
        target_image = base64.b64decode(image_data)
        nparr = np.frombuffer(target_image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        found_logos = []

        for logo in self.logo_images:
            result = cv2.matchTemplate(img, logo, cv2.TM_CCOEFF_NORMED)
            loc = np.where(result >= self.threshold)
            
            if loc[0].size > 0:
                found_logos.append(logo)

        return len(found_logos) > 0
    
# Alternative method
class ImageProcessor:
    def process_image(self, image_base64):
        image_bytes = base64.b64decode(image_base64)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)

        decoded_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(decoded_image, cv2.COLOR_BGR2RGB)
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        alpha = -2
        beta = 60
        prepro_image = cv2.convertScaleAbs(image_gray, alpha=alpha, beta=beta)

        return prepro_image

class TextExtractor:
    def extract_text(self, image):
        return pytesseract.image_to_string(image, lang="spa")

class KeywordChecker:
    def __init__(self):
        self.keywords = ['AUTOMOTOR', 'DNRPA']

    def check_keywords(self, extracted_text):
        words = extracted_text.split()
        found_words = [word for word in self.keywords if re.search(r'\b' + re.escape(word) + r'\b', extracted_text, re.IGNORECASE)]
        return found_words

class CedulaBackValidatorAlt():
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.text_extractor = TextExtractor()
        self.keyword_checker = KeywordChecker()
        
    def validate(self, image_base64):
        prepro_image = self.image_processor.process_image(image_base64)
        extracted_text = self.text_extractor.extract_text(prepro_image)
        found_keywords = self.keyword_checker.check_keywords(extracted_text)

        return bool(found_keywords)
                

router = APIRouter(prefix="/api/v1/validator/cedula", tags=["Cedula validator"], responses={404: {"description": "Not found"}})

validator_instance = CedulaBackValidator()
validator_instance_alt = CedulaBackValidatorAlt()

@router.post("/back", status_code=status.HTTP_200_OK, response_description="kw4rgs's cedula's back validator")
async def cedula_back_validator(data: dict):
    try:
        image_data = data.get('data')
        is_valid = validator_instance.validate(image_data)

        if is_valid or validator_instance_alt.validate(image_data):
            response_content = {'error': False, 'data': 'This is a valid cedula back image'}
        else:
            response_content = {'error': True, 'data': 'This is not a valid cedula back image'}

        return JSONResponse(content=response_content, status_code=status.HTTP_200_OK)

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,detail=str(e))
