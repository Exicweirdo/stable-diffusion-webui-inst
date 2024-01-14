import html

import gradio as gr

import modules.textual_inversion.textual_inversion
from modules import sd_hijack, shared


def create_embedding(name, initialization_text, nvpt, overwrite_old, embedding_type = "classical"):
    if embedding_type == "classical":
        filename = modules.textual_inversion.textual_inversion.create_embedding(name, nvpt, overwrite_old, init_text=initialization_text)
    elif embedding_type == "InST":
        filename = modules.textual_inversion.textual_inversion.create_embedding_with_attention(name, nvpt, overwrite_old, init_text=initialization_text)
    else:
        raise ValueError(f"Unknown embedding type {embedding_type}, must be 'classical' or 'InST'")

    sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)

    return gr.Dropdown.update(choices=sorted(sd_hijack.model_hijack.embedding_db.word_embeddings.keys())), f"Created: {filename}", ""


def train_embedding(*args):

    assert not shared.cmd_opts.lowvram, 'Training models with lowvram not possible'

    apply_optimizations = shared.opts.training_xattention_optimizations
    try:
        if not apply_optimizations:
            sd_hijack.undo_optimizations()

        embedding, filename = modules.textual_inversion.textual_inversion.train_embedding(*args)

        res = f"""
Training {'interrupted' if shared.state.interrupted else 'finished'} at {embedding.step} steps.
Embedding saved to {html.escape(filename)}
"""
        return res, ""
    except Exception:
        raise
    finally:
        if not apply_optimizations:
            sd_hijack.apply_optimizations()

