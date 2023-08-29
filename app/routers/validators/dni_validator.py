from fastapi import APIRouter, status, HTTPException
import cv2
import numpy as np
import base64
from fastapi.responses import JSONResponse

class DNIBackValidator:
    def __init__(self):
        self.logo_images = self.load_references()
        self.threshold = 0.80

    def load_references(self):
        pais_logo = cv2.imread('app/assets/kw4rgs_logo_collection/dni_back/dni_pais.png', cv2.IMREAD_UNCHANGED)
        arrows_logo = cv2.imread('app/assets/kw4rgs_logo_collection/dni_back/dni_arrows.png', cv2.IMREAD_UNCHANGED)
        pulgar_logo = cv2.imread('app/assets/kw4rgs_logo_collection/dni_back/dni_pulgar.png', cv2.IMREAD_UNCHANGED)
        return [pais_logo, arrows_logo, pulgar_logo]

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

router = APIRouter(prefix="/api/v1/validator/dni", tags=["DNI validator"], responses={404: {"description": "Not found"}})
validator_instance = DNIBackValidator()

@router.post("/back", status_code=status.HTTP_200_OK, response_description="kw4rgs's DNI's back validator")
async def dni_back_validator(data: dict):
    try:
        image_data = data.get('data')
        is_valid = validator_instance.validate(image_data)

        if is_valid:
            response_content = {'error': False, 'data': 'This is a valid DNI back image'}
        else:
            response_content = {'error': True, 'data': 'This is not a valid DNI back image'}

        return JSONResponse(content=response_content, status_code=status.HTTP_200_OK)

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
