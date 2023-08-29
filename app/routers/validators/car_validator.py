from fastapi import APIRouter, status, HTTPException
import tensorflow as tf
import cv2
import numpy as np
import base64

model_path = 'app/assets/kw4rgs_car_model_v1'

class CarValidator:
    def __init__(self, model_path: str):
        self.model = tf.keras.models.load_model(model_path)
        self.categories = {0: False, 1: True}

    def predict_base64_image(self, image_data: str) -> bool:
        try:
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is not None:
                img = cv2.resize(img, (224, 224))
                img = img.astype(float) / 255

                predictions = self.model.predict(img.reshape(-1, 224, 224, 3))
                predicted_class = np.argmax(predictions[0], axis=-1)

                return predicted_class == 1
            else:
                raise ValueError("Failed to process the image.")
        except Exception as e:
            raise ValueError(str(e))


router = APIRouter(prefix="/api/v1/validator", tags=["Car validator"], responses={404: {"description": "Not found"}})

car_validator_instance = CarValidator(model_path)

@router.post("/car", status_code=status.HTTP_200_OK, response_description="kw4rgs's car validator")
async def car_validator(data: dict) -> dict:
    try:
        image_data = data.get('data')
        result = car_validator_instance.predict_base64_image(image_data)
        
        if result:
            return {'error': False, 'data': 'This is a car'}
        else:
            return {'error': True, 'data': 'This is not a car'}
    
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
