from fastapi import FastAPI, status, Depends, HTTPException
import uvicorn
from dotenv import load_dotenv
import os
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.routers.validators import car_validator, dni_validator, licencia_validator, cedula_validator
from app.routers.extractors import dni_extractor, cedula_extractor

load_dotenv()

bearer_scheme = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    token = credentials.credentials
    if token != os.getenv("SECRET_KEY"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    
app = FastAPI(
    title="Kw4rgs's Validators and Extractors",
    description="API for validating various documents, car pictures and extracting information from IDs.",
    version="1.0.0",
    responses={404: {"description": "Not found"}},
    dependencies=[Depends(verify_token)]
)

@app.get("/", status_code=status.HTTP_200_OK)
def health_Check():
    return {"message": "Welcome to Kw4rgs's validators and extractors"}

# validators
app.include_router(car_validator.router)
app.include_router(dni_validator.router)
app.include_router(licencia_validator.router)
app.include_router(cedula_validator.router)

# extractors
app.include_router(dni_extractor.router)
app.include_router(cedula_extractor.router)


if __name__ == "__main__":
    uvicorn.run("kvalidators:app", host="127.0.0.1", port=6969, reload=True, workers=1)