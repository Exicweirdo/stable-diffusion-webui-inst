from collections import namedtuple

import numpy as np
from tqdm import trange

import modules.scripts as scripts
import gradio as gr

from modules import processing, shared, sd_samplers, sd_samplers_common

import torch
import k_diffusion as K


def find_noise_for_image(p, cond, uncond, cfg_scale, strength):
    x = p.init_latent
    x_in = torch.cat([x] * 2)

    t_enc = int(strength * 1000)
    x_noisy = shared.sd_model.q_sample(x_start=x, t=t_enc)
    cond_in = torch.cat([uncond, cond])
    image_conditioning = torch.cat([p.image_conditioning] * 2)
    cond_in = {"c_concat": [image_conditioning], "c_crossattn": [cond_in]}
    noise_combined = shared.sd_model.apply_model(x_noisy, t=t_enc, cond=cond_in)

    noise_cond, noise_uncond = noise_combined.chunk(2)
    noise = noise_uncond + (noise_cond - noise_uncond) * cfg_scale

    sd_samplers_common.store_latent(x)

    del x, x_noisy, cond_in, noise_combined, noise_cond, noise_uncond

    shared.state.nextjob()

    return noise


Cached = namedtuple(
    "Cached",
    [
        "noise",
        "cfg_scale",
        "strength",
        "latent",
    ],
)


class Script(scripts.Script):
    def __init__(self):
        self.cache = None

    def title(self):
        return "stochastic inversion for InST"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        info = gr.Markdown(
            """
        * `CFG Scale` should be 2 or lower.
        """
        )

        override_sampler = gr.Checkbox(
            label="Override `Sampling method` to Euler?(this method is built for it)",
            value=True,
            elem_id=self.elem_id("override_sampler"),
        )

        strength = gr.Slider(
            label="Strenght",
            minimum=0,
            maximum=1,
            step=0.01,
            value=0.4,
            elem_id=self.elem_id("strength"),
        )

        override_strength = gr.Checkbox(
            label="Override `Denoising strength` to `Strength`?",
            value=True,
            elem_id=self.elem_id("override_strength"),
        )

        cfg = gr.Slider(
            label="Decode CFG scale",
            minimum=0.0,
            maximum=15.0,
            step=0.1,
            value=1.0,
            elem_id=self.elem_id("cfg"),
        )
        randomness = gr.Slider(
            label="Randomness",
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            value=0.0,
            elem_id=self.elem_id("randomness"),
        )

        return [
            info,
            override_sampler,
            strength,
            override_strength,
            cfg,
            randomness,
        ]

    def run(
        self,
        p,
        _,
        override_sampler,
        strength,
        override_strength,
        cfg,
        randomness,
    ):
        # Override
        if override_sampler:
            p.sampler_name = "Euler"
        if override_strength:
            p.denoising_strength = strength

        def sample_extra(
            conditioning,
            unconditional_conditioning,
            seeds,
            subseeds,
            subseed_strength,
            prompts,
        ):
            lat = (p.init_latent.cpu().numpy() * 10).astype(int)

            same_params = (
                self.cache is not None
                and self.cache.cfg_scale == cfg
                and self.cache.strength == strength
            )
            same_everything = (
                same_params
                and self.cache.latent.shape == lat.shape
                and np.abs(self.cache.latent - lat).sum() < 100
            )

            if same_everything:
                rec_noise = self.cache.noise
            else:
                shared.state.job_count += 1
                cond = p.sd_model.get_learned_conditioning(
                    p.batch_size * [original_prompt]
                )
                uncond = p.sd_model.get_learned_conditioning(
                    p.batch_size * [original_negative_prompt]
                )
                rec_noise = find_noise_for_image(p, cond, uncond, cfg, strength)
                self.cache = Cached(
                    rec_noise,
                    cfg,
                    strength,
                    lat,
                )

            rand_noise = processing.create_random_tensors(
                p.init_latent.shape[1:],
                seeds=seeds,
                subseeds=subseeds,
                subseed_strength=p.subseed_strength,
                seed_resize_from_h=p.seed_resize_from_h,
                seed_resize_from_w=p.seed_resize_from_w,
                p=p,
            )

            combined_noise = (
                (1 - randomness) * rec_noise + randomness * rand_noise
            ) / ((randomness**2 + (1 - randomness) ** 2) ** 0.5)

            sampler = sd_samplers.create_sampler(p.sampler_name, p.sd_model)

            sigmas = sampler.model_wrap.get_sigmas(p.steps)

            noise_dt = combined_noise - (p.init_latent / sigmas[0])

            p.seed = p.seed + 1

            return sampler.sample_img2img(
                p,
                p.init_latent,
                combined_noise,  # noise_dt,
                conditioning,
                unconditional_conditioning,
                image_conditioning=p.image_conditioning,
            )

        p.sample = sample_extra

        p.extra_generation_params["Decode CFG scale"] = cfg
        p.extra_generation_params["Strength"] = strength
        p.extra_generation_params["Randomness"] = randomness

        processed = processing.process_images(p)

        return processed
