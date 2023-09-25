from __future__ import annotations
from typing import Iterable
import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
from gradio_im_to_3d import create_demo as create_im_to_3d_demo
import torch
import time

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('isl-org/ZoeDepth', "ZoeD_N", pretrained=True).to(DEVICE).eval()

class Seafoam(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.emerald,
        secondary_hue: colors.Color | str = colors.red,
        neutral_hue: colors.Color | str = colors.blue,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_md,
        text_size: sizes.Size | str = sizes.text_lg,
        text_color: colors.Color | str = colors.red,
        font: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Quicksand"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            body_background_fill="rgb(0,0,0)",
            body_background_fill_dark="",
            button_primary_background_fill="",
            button_primary_background_fill_hover="",
            button_primary_text_color="white",
            button_primary_background_fill_dark="",
            slider_color="*secondary_300",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_shadow="*shadow_drop_lg",
            button_large_padding="10px",
        )

seafoam = Seafoam()

title = "Image to 3D mesh"

with gr.Blocks(theme=seafoam) as demo:
    gr.Markdown(title)
    create_im_to_3d_demo(model)
print("Demo is running")
demo.launch(server_name="0.0.0.0")