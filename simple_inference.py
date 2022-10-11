from torch import autocast
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
	"CompVis/stable-diffusion-v1-4", 
	use_auth_token=True
).to("cuda")

prompt = "김치"
with autocast("cuda"):
    image = pipe(prompt)["sample"][0]
    
image.save("a.png")