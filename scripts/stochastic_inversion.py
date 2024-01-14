from collections import namedtuple

import numpy as np
from tqdm import trange

import modules.scripts as scripts
import gradio as gr
from modules.ui_common import create_refresh_button

from modules import processing, shared, sd_samplers, sd_samplers_common

from modules.sd_hijack import model_hijack

import torch
import k_diffusion as K


def find_noise_for_image(p, cond, uncond, cfg_scale, strength):
    x = p.init_latent
    x_in = torch.cat([x] * 2)

    t_enc = int(strength * 1000)
    t_enc = torch.tensor([t_enc], dtype=torch.long, device=x.device)
    x_noisy = shared.sd_model.q_sample(x_start=x_in, t=t_enc)
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
        "input_img",
        "embedding_name",
        "embedding_vec",
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
            label="Strength",
            minimum=0,
            maximum=1,
            step=0.01,
            value=0.4,
            elem_id=self.elem_id("strength"),
        )

        cfg = gr.Slider(
            label="Decode CFG scale",
            minimum=0.0,
            maximum=1.0,
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

        with gr.Accordion("InST Embedding Vector", open=False):
            input_img = gr.Image(
                label="Input image",
                sources=["upload", "clipboard"],
                image_mode="RGB",
                interactive=True,
                elem_id=self.elem_id("input_img"),
            )
            embedding_name = gr.Dropdown(
                label="Embedding",
                elem_id=self.elem_id("embedding_name"),
                choices=sorted(model_hijack.embedding_db.word_embeddings.keys()),
            )
            create_refresh_button(
                embedding_name,
                model_hijack.embedding_db.load_textual_inversion_embeddings,
                lambda: {
                    "choices": sorted(model_hijack.embedding_db.word_embeddings.keys())
                },
                self.elem_id("refresh_embedding_name"),
            )
            calc_vec = gr.Button(
                label="Calculate embedding vector",
                elem_id=self.elem_id("calc_vec"),
            )

        return [
            info,
            override_sampler,
            strength,
            cfg,
            randomness,
            input_img,
            embedding_name,
            calc_vec
        ]

    def run(
        self,
        p,
        _,
        override_sampler,
        strength,
        cfg,
        randomness,
        input_img,
        embedding_name,
        calc_vec,
    ):
        # Override
        if override_sampler:
            p.sampler_name = "Euler"
        embedding_vec = None
        """calc_vec.click(
            model_hijack.embedding_db.recalculate_embedding_vector_by_name,
            inputs=[embedding_name, input_img],
            outputs=[embedding_vec],
            show_progress=False,
        )"""
        input_img = torch.from_numpy(input_img) / 255.0
        model_hijack.embedding_db.recalculate_embedding_vector_by_name(embedding_name, input_img)

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
                and self.cache.embedding_name == embedding_name
                and self.cache.input_img.shape == input_img.shape
                and np.abs(self.cache.input_img - input_img).sum() < 100
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
                # model_hijack.embedding_db.#recalculate_embedding_vector_by_name(
                #    embedding_name, input_img
                # )
                cond = p.sd_model.get_learned_conditioning(p.batch_size * [p.prompt])
                uncond = p.sd_model.get_learned_conditioning(
                    p.batch_size * [p.negative_prompt]
                )
                rec_noise = find_noise_for_image(p, cond, uncond, cfg, strength)
                self.cache = Cached(
                    rec_noise,
                    cfg,
                    strength,
                    lat,
                    input_img,
                    embedding_name,
                    embedding_vec,
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
