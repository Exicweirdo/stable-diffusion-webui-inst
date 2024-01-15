import gradio as gr
from modules.ui_common import create_refresh_button
import modules.scripts as scripts
from modules import (
    script_callbacks,
)
from modules.sd_hijack import model_hijack
import torch

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui_component:
        input_img = gr.Image(
            label="Input image",
            sources=["upload", "clipboard"],
            image_mode="RGB",
            interactive=True,
            type = "pil",
            elem_id="InST_input_img",
        )
        with gr.Row():
            embedding_name = gr.Dropdown(
                label="Embedding",
                choices=sorted(model_hijack.embedding_db.word_embeddings.keys()),
                elem_id="InST_embedding_name",
            )
            create_refresh_button(
                embedding_name,
                model_hijack.embedding_db.load_textual_inversion_embeddings,
                lambda: {
                    "choices": sorted(model_hijack.embedding_db.word_embeddings.keys())
                },
                elem_id="InST_refresh_embeddings",
            )
        calc_vec = gr.Button(
            label="Calculate embedding vector",
            elem_id="InST_calc_vec",
        )
        #convert gr img to tensor
        
        def recalc_and_refresh(embedding_name, input_img):
            model_hijack.embedding_db.recalculate_embedding_vector_by_name(
                embedding_name, input_img, True
            )
            model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)
            return
        calc_vec.click(
            recalc_and_refresh,
            inputs=[embedding_name, input_img],
            outputs=[],
            show_progress=False,
        )

        return [(ui_component, "InST Embedding", "InST_tab")]


script_callbacks.on_ui_tabs(on_ui_tabs)
""" import modules.scripts as scripts
import gradio as gr
import os

from modules import script_callbacks


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui_component:
        with gr.Row():
            angle = gr.Slider(
                minimum=0.0,
                maximum=360.0,
                step=1,
                value=0,
                label="Angle"
            )
            checkbox = gr.Checkbox(
                False,
                label="Checkbox"
            )
            # TODO: add more UI components (cf. https://gradio.app/docs/#components)
        return [(ui_component, "Extension Template", "extension_template_tab")]

script_callbacks.on_ui_tabs(on_ui_tabs) """