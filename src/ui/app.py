import gradio as gr
import torch
from PIL import Image
import io

# Load your PyTorch model

if torch.cuda.is_available():
    model = torch.load("./src/ui/best-model.ckpt")
else:
    model = torch.load("./src/ui/best-model.ckpt", map_location=torch.device('cpu'))
# model = torch.load("../model/BEST-epoch=23-step=864006.ckpt")
# model.eval()

# Function to convert PNG to WebP
def convert_to_webp(png_image):
    png_image = Image.open(io.BytesIO(png_image))
    webp_image = png_image.convert("RGB").save(io.BytesIO(), "WEBP")
    return webp_image.getvalue()

# Function to perform disparity estimation
def disparity_estimation(left_image, right_image):
    # Convert PNG to WebP
    left_webp = convert_to_webp(left_image)
    right_webp = convert_to_webp(right_image)

    # Process the images and perform disparity estimation with your model
    # You'll need to adjust this part according to your model's specific requirements
    disparity_map = model(left_webp, right_webp)

    # Return the disparity map as a PNG image
    return disparity_map

# Create a Gradio interface
iface = gr.Interface(
    fn=disparity_estimation,
    inputs=[
        gr.inputs.Image(type="pil", label="Left Image (PNG)"),
        gr.inputs.Image(type="pil", label="Right Image (PNG)"),
    ],
    outputs=gr.outputs.Image(type="pil", label="Disparity Map (PNG)"),
    title="Disparity Estimation",
    description="Upload two PNG images for left and right views to estimate disparity.",
)

# Start the Gradio UI
iface.launch()