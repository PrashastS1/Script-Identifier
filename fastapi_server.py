from fastapi import FastAPI, HTTPException, UploadFile, File
import uvicorn
from fastapi.responses import JSONResponse
from loguru import logger
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from main import best_model  # Import your model function

# Create FastAPI app
app = FastAPI()

# Adding CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"], 
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "Authorization", "X-Requested-With"],
    expose_headers=["Content-Type", "Content-Length"],
    max_age=600,
)

@app.post("/pipeline/")
async def main_entry(image: UploadFile = File(...)) -> JSONResponse:
    logger.debug(f"Received image upload: {image.filename}")
    
    try:
        # Read image file
        contents = await image.read()
        
        # Convert to PIL Image
        img = Image.open(io.BytesIO(contents))
        
        # Process with your model
        language, confidence_score = best_model(image=img)
        
        return JSONResponse(
            status_code=200,
            content={
                "language": language,
                "confidence_score": confidence_score
            }
        )
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/health")
def health_check():
    return JSONResponse(status_code=200, content={"status": "OK"})

if __name__ == "__main__":
    uvicorn.run("fastapi_server:app", host="127.0.0.1", port=8002, reload=True)