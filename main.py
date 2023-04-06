import torch
import datetime
from token_auto_concat_embeds import token_auto_concat_embeds
from diffusers import (
    DiffusionPipeline,
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    UniPCMultistepScheduler,
)

# Generate Settings
MODEL_ID=''
VAE_MODEL_ID=''
POSITIVE_PROMPT = ''
NEGATIVE_PROMPT = ''
HEIGHT = 768
WIDTH = 512
SCALE = 12.0
STEP = 28
SEED = 3788086447
SCHEDULER = 'EulerAncestralDiscrete'
DEVICE = 'cuda'
CACHE_DIR = './cache'

# Constant Values
DEFAULT_PROMPT = [
    'masterpiece, best quality, ',
    'lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, '
]
SCHEDULERS = {
    'DDIM' : DDIMScheduler,
    'DDPM' : DDPMScheduler,
    'DEISMultistep' : DEISMultistepScheduler,
    'DPMSolverMultistep' : DPMSolverMultistepScheduler,
    'DPMSolverSinglestep' : DPMSolverSinglestepScheduler,
    'EulerAncestralDiscrete' : EulerAncestralDiscreteScheduler,
    'EulerDiscrete' : EulerDiscreteScheduler,
    'HeunDiscrete' : HeunDiscreteScheduler,
    'KDPM2AncestralDiscrete' : KDPM2AncestralDiscreteScheduler,
    'KDPM2Discrete' : KDPM2DiscreteScheduler,
    'UniPCMultistep' : UniPCMultistepScheduler
}

# Create Pipeline
pipe = DiffusionPipeline.from_pretrained(
    pretrained_model_name_or_path=MODEL_ID,
    torch_dtype=torch.float16,
    cache_dir=CACHE_DIR,
)

# Add Scheduler
pipe.scheduler = SCHEDULERS[SCHEDULER].from_pretrained(
    pretrained_model_name_or_path=MODEL_ID,
    torch_dtype=torch.float16,
    cache_dir=CACHE_DIR,
    subfolder='scheduler'
)

# Add Vae
pipe.vae = AutoencoderKL.from_pretrained(
    pretrained_model_name_or_path=VAE_MODEL_ID,
    torch_dtype=torch.float16,
    cache_dir=CACHE_DIR,
)

# Other Setting
pipe.safety_checker = None if pipe.safety_checker is None else lambda images, **kwargs: (images, False)
pipe.enable_attention_slicing()
pipe = pipe.to(DEVICE)

# Toekn concatenated embedding
positive_embeds, negative_embeds = token_auto_concat_embeds(pipe, DEFAULT_PROMPT[0] + POSITIVE_PROMPT, DEFAULT_PROMPT[1] + NEGATIVE_PROMPT)

# Generate Image
image = pipe(
    prompt_embeds=positive_embeds,
    height=HEIGHT,
    width=WIDTH,
    num_inference_steps=STEP,
    guidance_scale=SCALE,
    negative_prompt_embeds=negative_embeds,
    generator=torch.Generator(device=DEVICE).manual_seed(SEED)
).images[0]

# Save Image
image.save("images/" + str(datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')) + ".png")