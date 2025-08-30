import torch
from diffusers import AutoPipelineForText2Image
import gradio as gr


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

def generate_image(prompt, negative_prompt, width, height, num_inference_steps, guidance_scale):
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]
    return image

iface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Prompt", value="a cinematic photo of a robot cat ordering coffee at a cafe, detailed, 8k"),
        gr.Textbox(label="Negative Prompt", value="cartoon, drawing, anime, blurry, low quality"),
        gr.Slider(minimum=256, maximum=1024, step=64, value=896, label="Width"),
        gr.Slider(minimum=256, maximum=1024, step=64, value=896, label="Height"),
        gr.Slider(minimum=1, maximum=50, step=1, value=20, label="Inference Steps"),
        gr.Slider(minimum=0.0, maximum=20.0, step=0.5, value=7.0, label="Guidance Scale")
    ],
    outputs="image",
    title="Dreamshaper XL v2 Turbo: Text-to-Image"
)

if __name__ == "__main__":
    iface.launch(share=True)