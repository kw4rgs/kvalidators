from fastapi import APIRouter, status, HTTPException
import cv2
import base64
from zxing import BarCodeReader
import numpy as np
import tempfile
from fastapi.responses import JSONResponse


class BarcodeProcessor:
    def __init__(self):
        self.reader = BarCodeReader()

    def decode_barcode(self, image_path):
        return self.reader.decode(image_path)

class DNIExtractor:
    def __init__(self, barcode_processor):
        self.barcode_processor = barcode_processor

    def enhance_image(self, image):
        enhanced_image = cv2.convertScaleAbs(image, alpha=1.2, beta=50)
        return enhanced_image

    def extract_dni_data(self, image_data):
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            enhanced_image = self.enhance_image(image)
            cv2.imwrite(temp_file.name, enhanced_image)

        barcode = self.barcode_processor.decode_barcode(temp_file.name)
        
        if barcode is not None:
            barcode_data = barcode.raw
            if barcode_data is None:
                return None 
            formatted_data = self.format_data(barcode_data)
            return formatted_data

    def format_data(self, barcode_data):
        data_splitted = barcode_data.split(sep='@')
        data = {
            'apellido': data_splitted[1],
            'nombre': data_splitted[2],
            'sexo': data_splitted[3],
            'dni': data_splitted[4],
            'fecha_nacimiento': data_splitted[6],
        }
        return data
            
    
router = APIRouter(prefix="/api/v1/extractor/dni", tags=["DNI extractor"], responses={404: {"description": "Not found"}})

barcode_processor = BarcodeProcessor()
dni_extractor_instance = DNIExtractor(barcode_processor)

@router.post("/front", status_code=status.HTTP_200_OK, response_description="kw4rgs's front DNI extractor")
async def dni_front_extractor(data: dict) -> dict:
    try:
        image_data = data.get('data')
        image_bytes = base64.b64decode(image_data)
        extracted_data = dni_extractor_instance.extract_dni_data(image_bytes)
        
        if extracted_data:
            response_content = {'error': False, 'data': extracted_data}
            return JSONResponse(content=response_content, status_code=status.HTTP_200_OK)
        else:
            response_content = {'error': True, 'data': 'Impossible to extract data from the image.'}
            return JSONResponse(content=response_content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)
    
    except Exception:
        response_content = {'error': True, 'data': 'No image data provided to process.'}
        return JSONResponse(content=response_content, status_code=status.HTTP_400_BAD_REQUEST)
    