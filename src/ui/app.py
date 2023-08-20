import gradio as gr
import torch
from torchvision.transforms import ToTensor
import io
# Load your PyTorch model

if torch.cuda.is_available():
    model = torch.load("./best-model.ckpt")
    # model = torch.load("./src/ui/best-model.ckpt")
else:
    model = torch.load("./best-model.ckpt", map_location=torch.device('cpu'))
    # model = torch.load("./src/ui/best-model.ckpt", map_location=torch.device('cpu'))
# model = torch.load("../model/BEST-epoch=23-step=864006.ckpt")
# model.eval()
class Rescale():
    """
    Rescales the left and right image tensors (initially ranged between [0, 1]) and rescales them to be between [-1, 1].
    """

    @staticmethod
    def __call__(sample):
        for name in ['left', 'right']:
            sample[name] = (sample[name] - 0.5) * 2
        return sample
    
# def png_loader(path):
#     img = io.imread(path)
#     return img

# Function to convert PNG to WebP
def prepare_sample(left_image, right_image):

    # left_image = png_loader(left_image)
    # right_imate = png_loader(right_imate)

    # transforms = [ToTensor(), Rescale()]
    transforms = [ToTensor()]
    # torch_sample = ToTensor()(sample)
    for transform in transforms:
        left_image = transform(left_image)
        right_image = transform(right_image)

    sample = {
        'left': left_image, 'right': right_image, 
        }

    return sample

def disparity_estimation(left_image, right_image):

    sample = prepare_sample(left_image, right_image)
    # disparity_map = model(sample['left'], sample['right'])
    with torch.no_grad():
        disparity_map = model(sample)
    return disparity_map

# Create a Gradio interface
demo = gr.Interface(
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
demo.launch()

