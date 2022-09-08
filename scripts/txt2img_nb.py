import argparse
import glob
import os
import sys
import time
from contextlib import contextmanager, nullcontext
from itertools import islice

import numpy as np
import torch
from einops import rearrange
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torch import autocast
from torchvision.utils import make_grid
from tqdm.auto import tqdm, trange


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    pl_sd = torch.load(ckpt, map_location=device)
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)

    model.cuda()
    model.eval()
    return model


def load_model(laion400m=False, config="configs/stable-diffusion/v1-inference.yaml", ckpt="models/ldm/stable-diffusion-v1/model.ckpt"):
    if laion400m:
        print("Falling back to LAION 400M model...")
        config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        ckpt = "models/ldm/text2img-large/model.ckpt"

    config = OmegaConf.load(f"{config}")
    model = load_model_from_config(config, f"{ckpt}")
    # use float16 model (for VRAM 8GB environment)
    model = model.to(torch.float16)

    return model


def main(model, prompt: str, ddim_steps=50, fixed_code=False, plms=False, ddim_eta=0.0, n_iter=1, H=512, W=512, C=4, f=8, n_samples=1, n_rows=0, scale=7.5, seed=42, precision="autocast", prompt_correction=[]):
    seed_everything(seed)

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    batch_size = n_samples
    n_rows = n_rows if n_rows > 0 else batch_size

    assert prompt is not None
    data = [batch_size * [prompt]]

    start_code = None
    if fixed_code:
        start_code = torch.randn(
            [n_samples, C, H // f, W // f], device=device)

    images = []

    precision_scope = autocast if precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                for n in trange(n_iter, desc="Sampling"):
                    for prompts in tqdm(data):
                        uc = None
                        if scale != 1.0:
                            uc = model.get_learned_conditioning(
                                batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        for pw in prompt_correction:
                            pw = pw.split('::')
                            p, weight = pw[:-1], float(pw[-1])
                            c += weight * \
                                model.get_learned_conditioning(list(p))
                        shape = [C, H // f, W // f]
                        samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                         conditioning=c,
                                                         batch_size=n_samples,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=scale,
                                                         unconditional_conditioning=uc,
                                                         eta=ddim_eta,
                                                         x_T=start_code)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp(
                            (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                        for x_sample in x_samples_ddim:
                            x_sample = 255. * \
                                rearrange(x_sample.cpu().numpy(),
                                          'c h w -> h w c')
                            images.append(Image.fromarray(
                                x_sample.astype(np.uint8)))

                toc = time.time()
    return images


if __name__ == "__main__":
    main()
