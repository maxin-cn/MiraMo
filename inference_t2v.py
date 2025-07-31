import os, sys
sys.path.append(os.path.split(sys.path[0])[0])
import time
import torch
import random
import imageio
import argparse
import torchvision

import numpy as np

from download import find_model
from pipelines.pipeline_miramo import MiraMoPipeline
from diffusers.models import AutoencoderDC, AutoencoderKLTemporalDecoder
from models import get_models
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers.schedulers import DPMSolverMultistepScheduler
from omegaconf import OmegaConf
from PIL import Image

def prepare_image(path, vae, transform_video, device, dtype=torch.float16):
    with open(path, 'rb') as f:
        image = Image.open(f).convert('RGB')
    image = torch.as_tensor(np.array(image, dtype=np.uint8, copy=True)).unsqueeze(0).permute(0, 3, 1, 2)
    image, ori_h, ori_w, crops_coords_top, crops_coords_left = transform_video(image)
    image = vae.encode(image.to(dtype=dtype, device=device))[0] * vae.config.scaling_factor
    image = image.unsqueeze(1)
    print(image.shape)
    return image

def main(args):

    if args.seed:
        torch.manual_seed(args.seed)
        print("Random Seed: ", args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 # torch.float16

    model = get_models(args).from_pretrained_2d(pretrained_model_name_or_path=args.pretrained_model_path, 
                                                subfolder="transformer", 
                                                use_motion_guidance=args.model.use_motion_guidance,
                                                ).to(device=device, dtype=dtype)

    t0 = time.time()
    state_dict = find_model(args.ckpt)
    print("Model download time: ", time.time() - t0)

    model.load_state_dict(state_dict)
    vae = AutoencoderKLTemporalDecoder.from_pretrained(args.pretrained_model_path, subfolder="vae_temporal_decoder", torch_dtype=dtype).to(device=device)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.pretrained_model_path, 'tokenizer'), use_fast=False)
    tokenizer.padding_side = "right"
    text_encoder = AutoModelForCausalLM.from_pretrained(os.path.join(args.pretrained_model_path, 'text_encoder'), torch_dtype=dtype).get_decoder().to(device)

    # set eval mode
    model.eval()
    vae.eval()
    text_encoder.eval()

    scheduler = DPMSolverMultistepScheduler.from_pretrained(pretrained_model_name_or_path=args.pretrained_model_path, subfolder="scheduler")

    videogen_pipeline = MiraMoPipeline(
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=model,
        scheduler=scheduler,
    ).to(device)
    # videogen_pipeline.enable_xformers_memory_efficient_attention()
    # videogen_pipeline.enable_vae_slicing()

    if not os.path.exists(args.save_img_path):
        os.makedirs(args.save_img_path)
    
    prompt_lists = [
              'A panda holding a board with the word AI Lab written on it.',
              'a cute raccoon playing guitar in the park at sunrise, oil painting style.',
              'Yellow and black tropical fish dart through the sea',
              'A very happy fuzzy panda dressed as a chef eating pizza in the New York street food truck.',
              'A high-quality 3D render of hyperrealist, super strong, multicolor stripped, and ﬂuffy bear with wings, highly detailed.',
              'a koala bear playing piano in the forest.',
              'an astronaut feeding ducks on a sunny afternoon, reflection from the water.',
              'an astronaut flying in space.',
              'a panda standing on a surfboard in the ocean in sunset.',
              'A cyborg koala dj in front of aturntable, in heavy raining futuristic tokyo rooftop cyberpunk night, sci-f, fantasy, intricate, neon light, soft light smooth, sharp focus, illustration.',
              'Iron Man dancing on the beach',
              'A cute funny robot dancing, centered, award winning watercolor pen illustration, detailed, isometric illustration, drawing.',
              'A cute anime girl looks at the beautiful nature through the window of amoving train, well rendered, 3D rendered.',
              'Two pandas discussing an academic paper.',
              'A boat sailing rapidly in a lake, in the style of a neo retro poster, matte drawing, outrun color palette.',
              'two teddy bears playing poker under water, oil painting style',
              'teddy bear playing the guitar',
              "A guineapig painter is painting on the canvas, anthro, very cute kid's film character, concept artwork, 3dconcept, detailed fur, high detail iconic character for upcoming film.", 
              'a cat wearing sunglasses and working as a lifeguard at pool.',
              'Teddy bear walking down 5th Avenue, front view, beautiful sunset, high definition, 4k',
              "A video of porsche 911 with mountain in the background, photorealistic vivid, sharp focus, reflection, refraction, sunrays, very detailed intricate, intense cinematic composition",
              'beer pouring into glass, low angle video shot.',
              'A skull burning while being held up by a skeletal hand, hyperdetailed, volumetric light, f8 aperture.',
              'Zoom in video of a robot warrior, ultra realistic, concept art, intricate details, highly detailed, photorealistic, 8k sharp focus, volumetric lighting unreal engine.',
              'A chihuahua in astronaut suit and sunglasses floating in space, earth andmoon in the background, photorealistic, 8k, cinematic lighting, hd, atmospheric, hyperdetailed, deviantart, photography, glow effect.',
              'An epic tornado attacking above aglowing city at night, the tornado is made of smoke, highly detailed.',
              'Baroque oil painting of shocked girl with long flowing purple hair, japanese light novel cover illustration, symmetrical perfect face, fine detail directed gaze.',
              'An emoji of a baby panda wearing a red hat, blue gloves, green shirt, and blue pants',
              'An oil painting of a couple in formal evening wear going home get caught in a heavy downpour with umbrellas',
              ]

    for i, prompt in enumerate(prompt_lists):

        print('Processing the ({}) prompt'.format(prompt))

        videos = videogen_pipeline(prompt, 
                                num_frames=args.video_length, 
                                height=args.image_size[0], 
                                width=args.image_size[1], 
                                num_inference_steps=args.num_sampling_steps,
                                guidance_scale=args.guidance_scale,
                                motion_bucket_id=args.motion_bucket_id,
                                use_resolution_binning=False,
                                ).images
        print('video shape: ', videos.shape)
        
        imageio.mimwrite(args.save_img_path + prompt.replace(' ', '_') + '_%04d' % args.motion_bucket_id + '_%04d' % i + '-imageio.mp4', videos[0], fps=8, quality=5) # highest quality is 10, lowest is 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/sample.yaml")
    parser.add_argument("--run-time", type=int, default=0)
    args = parser.parse_args()
    omega_conf = OmegaConf.load(args.config)
    omega_conf.run_time = args.run_time
    main(omega_conf)


