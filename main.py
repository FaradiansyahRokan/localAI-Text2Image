import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, HTMLResponse
from fastapi.staticfiles import StaticFiles
from diffusers import AutoPipelineForText2Image
from pydantic import BaseModel
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware

# Pydantic model to validate incoming data from the frontend
class GenerateImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 896
    height: int = 896
    num_inference_steps: int = 20
    guidance_scale: float = 7.0

# --- Model Loading ---
# Load model outside the request function for efficiency
model_id = "lykon/dreamshaper-xl-v2-turbo"
device = "cuda"

pipe = AutoPipelineForText2Image.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)

pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

# --- API Endpoints ---
app = FastAPI()

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use a specific origin in production, e.g., "http://127.0.0.1:5500"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount a directory for static files (our HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# The main endpoint for generating the image
@app.post("/generate_image")
async def generate_image_endpoint(request_body: GenerateImageRequest):
    try:
        image = pipe(
            prompt=request_body.prompt,
            negative_prompt=request_body.negative_prompt,
            width=request_body.width,
            height=request_body.height,
            num_inference_steps=request_body.num_inference_steps,
            guidance_scale=request_body.guidance_scale,
        ).images[0]

        # Save the image to a byte stream and return it as a PNG response
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return Response(content=buffer.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Serve the main HTML file
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    try:
        with open("static/index.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html not found. Please ensure it is in the 'static' directory.")
