from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Tuple, cast

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "model"

DCGAN_CKPT = MODEL_DIR / "dcgan_generator_final.pt"
WGANGP_CKPT = MODEL_DIR / "wgangp_checkpoint.pt"
WGANGP_GEN_FALLBACK = MODEL_DIR / "wgangp_generator_final.pt"
PIX2PIX_CKPT = MODEL_DIR / "pix2pix_export_q2.pt"
Q2_SAMPLE_IMAGE = MODEL_DIR / "q2_sample_input.png"

CYCLEGAN_GAB_CKPT = MODEL_DIR / "G_AB_final.pth"
CYCLEGAN_GBA_CKPT = MODEL_DIR / "G_BA_final.pth"
CYCLEGAN_FULL_CKPT = MODEL_DIR / "cyclegan_final.pth"

NOISE_SIZE = 100
CHANNELS = 3
Q2_IMAGE_SIZE = 256
Q3_IMAGE_SIZE = 128
Q3_N_RES = 6


# --------------------------
# Utility helpers
# --------------------------
def get_device(use_cuda_if_available: bool) -> torch.device:
    if use_cuda_if_available and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def is_tensor_dict(value: object) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    return all(torch.is_tensor(v) for v in value.values())


def extract_state_dict(payload: object, preferred_keys: Tuple[str, ...] = ()) -> dict:
    if is_tensor_dict(payload):
        return cast(dict, payload)  # already a state_dict

    if isinstance(payload, dict):
        for key in preferred_keys:
            if key in payload:
                return extract_state_dict(payload[key])

        fallback_keys = (
            "state_dict",
            "model_state_dict",
            "generator_state_dict",
            "generator",
            "model",
            "G",
        )
        for key in fallback_keys:
            if key in payload:
                try:
                    return extract_state_dict(payload[key])
                except Exception:
                    continue

    raise ValueError("Could not extract a valid state_dict from checkpoint.")


def strip_module_prefix(state_dict: dict) -> dict:
    if not state_dict:
        return state_dict

    has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
    if not has_module_prefix:
        return state_dict

    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def load_model_weights(model: nn.Module, state_dict: dict) -> None:
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass

    stripped = strip_module_prefix(state_dict)
    model.load_state_dict(stripped)


def tensor_to_pil_from_tanh(tensor: torch.Tensor) -> Image.Image:
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    img = (tensor.detach().cpu() * 0.5 + 0.5).clamp(0, 1)
    return transforms.ToPILImage()(img)


def batch_to_pil_from_tanh(batch: torch.Tensor) -> list[Image.Image]:
    images = []
    for i in range(batch.shape[0]):
        images.append(tensor_to_pil_from_tanh(batch[i]))
    return images


def read_uploaded_checkpoint(uploaded_file) -> Optional[bytes]:
    if uploaded_file is None:
        return None
    return uploaded_file.getvalue()


def load_torch_payload_from_bytes(blob: bytes, device: torch.device):
    return torch.load(io.BytesIO(blob), map_location=device)


def split_cyclegan_full_state_dict(payload: object) -> Tuple[dict, dict]:
    if not isinstance(payload, dict):
        raise ValueError("CycleGAN full checkpoint should be a dictionary.")

    # Common patterns in full checkpoints.
    direct_pairs = (
        ("G_AB", "G_BA"),
        ("g_ab", "g_ba"),
        ("generator_ab", "generator_ba"),
        ("G_AB_state_dict", "G_BA_state_dict"),
    )
    for key_ab, key_ba in direct_pairs:
        if key_ab in payload and key_ba in payload:
            sd_ab = extract_state_dict(payload[key_ab])
            sd_ba = extract_state_dict(payload[key_ba])
            return sd_ab, sd_ba

    # Single state_dict where keys are prefixed with generator names.
    if is_tensor_dict(payload):
        keys = list(payload.keys())
        if any(k.startswith("G_AB.") for k in keys) and any(k.startswith("G_BA.") for k in keys):
            sd_ab = {k.replace("G_AB.", "", 1): v for k, v in payload.items() if k.startswith("G_AB.")}
            sd_ba = {k.replace("G_BA.", "", 1): v for k, v in payload.items() if k.startswith("G_BA.")}
            return sd_ab, sd_ba

    # Nested state_dict fallback.
    if "state_dict" in payload:
        return split_cyclegan_full_state_dict(payload["state_dict"])

    raise ValueError("Unable to find both G_AB and G_BA in CycleGAN full checkpoint.")


# --------------------------
# Q1 models (DCGAN / WGAN-GP)
# --------------------------
class Q1Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(NOISE_SIZE, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.net(noise)


@st.cache_resource
def load_q1_dcgan_model(device_str: str) -> nn.Module:
    device = torch.device(device_str)
    if not DCGAN_CKPT.exists():
        raise FileNotFoundError(f"Missing checkpoint: {DCGAN_CKPT}")

    model = Q1Generator().to(device)
    payload = torch.load(DCGAN_CKPT, map_location=device)
    state_dict = extract_state_dict(payload)
    load_model_weights(model, state_dict)
    model.eval()
    return model


@st.cache_resource
def load_q1_wgangp_model(device_str: str) -> nn.Module:
    device = torch.device(device_str)
    model = Q1Generator().to(device)

    if WGANGP_CKPT.exists():
        payload = torch.load(WGANGP_CKPT, map_location=device)
        if isinstance(payload, dict) and "G" in payload:
            state_dict = extract_state_dict(payload["G"])
        else:
            state_dict = extract_state_dict(payload)
        load_model_weights(model, state_dict)
        model.eval()
        return model

    if WGANGP_GEN_FALLBACK.exists():
        payload = torch.load(WGANGP_GEN_FALLBACK, map_location=device)
        state_dict = extract_state_dict(payload)
        load_model_weights(model, state_dict)
        model.eval()
        return model

    raise FileNotFoundError(
        f"Missing checkpoints: {WGANGP_CKPT.name} and {WGANGP_GEN_FALLBACK.name}"
    )


# --------------------------
# Q2 model (Pix2Pix)
# --------------------------
class Q2UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Q2EDBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_bn=True, act="leaky", use_dropout=False):
        super().__init__()
        layers = []
        if down:
            layers.append(
                nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            )
        else:
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False))

        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))

        if act == "relu":
            layers.append(nn.ReLU())
        else:
            layers.append(nn.LeakyReLU(0.2))

        self.block = nn.Sequential(*layers)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.block(x)
        return self.dropout(x) if self.use_dropout else x


class Q2Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.down1 = Q2UNetBlock(features, features * 2, down=True, act="leaky", use_dropout=False)
        self.down2 = Q2UNetBlock(features * 2, features * 4, down=True, act="leaky", use_dropout=False)
        self.down3 = Q2UNetBlock(features * 4, features * 8, down=True, act="leaky", use_dropout=False)
        self.down4 = Q2UNetBlock(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.down5 = Q2UNetBlock(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.down6 = Q2UNetBlock(features * 8, features * 8, down=True, act="leaky", use_dropout=False)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1, padding_mode="reflect"),
            nn.ReLU(),
        )

        self.up1 = Q2UNetBlock(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = Q2UNetBlock(features * 16, features * 8, down=False, act="relu", use_dropout=True)
        self.up3 = Q2UNetBlock(features * 16, features * 8, down=False, act="relu", use_dropout=True)
        self.up4 = Q2UNetBlock(features * 16, features * 8, down=False, act="relu", use_dropout=True)
        self.up5 = Q2UNetBlock(features * 16, features * 4, down=False, act="relu", use_dropout=False)
        self.up6 = Q2UNetBlock(features * 8, features * 2, down=False, act="relu", use_dropout=False)
        self.up7 = Q2UNetBlock(features * 4, features, down=False, act="relu", use_dropout=False)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)

        bn = self.bottleneck(d7)
        u1 = self.up1(bn)
        u2 = self.up2(torch.cat([u1, d7], dim=1))
        u3 = self.up3(torch.cat([u2, d6], dim=1))
        u4 = self.up4(torch.cat([u3, d5], dim=1))
        u5 = self.up5(torch.cat([u4, d4], dim=1))
        u6 = self.up6(torch.cat([u5, d3], dim=1))
        u7 = self.up7(torch.cat([u6, d2], dim=1))

        return self.final_up(torch.cat([u7, d1], dim=1))


class Q2GeneratorAlt(nn.Module):
    """Pix2Pix generator variant with e*/d*/final naming used by exported Q2 checkpoint."""

    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.e0 = Q2EDBlock(in_channels, features, down=True, use_bn=False, act="leaky")
        self.e1 = Q2EDBlock(features, features * 2, down=True, use_bn=True, act="leaky")
        self.e2 = Q2EDBlock(features * 2, features * 4, down=True, use_bn=True, act="leaky")
        self.e3 = Q2EDBlock(features * 4, features * 8, down=True, use_bn=True, act="leaky")
        self.e4 = Q2EDBlock(features * 8, features * 8, down=True, use_bn=True, act="leaky")
        self.e5 = Q2EDBlock(features * 8, features * 8, down=True, use_bn=True, act="leaky")
        self.e6 = Q2EDBlock(features * 8, features * 8, down=True, use_bn=True, act="leaky")
        self.e7 = Q2EDBlock(features * 8, features * 8, down=True, use_bn=False, act="relu")

        self.d0 = Q2EDBlock(features * 8, features * 8, down=False, use_bn=True, act="relu", use_dropout=True)
        self.d1 = Q2EDBlock(
            features * 16,
            features * 8,
            down=False,
            use_bn=True,
            act="relu",
            use_dropout=True,
        )
        self.d2 = Q2EDBlock(
            features * 16,
            features * 8,
            down=False,
            use_bn=True,
            act="relu",
            use_dropout=True,
        )
        self.d3 = Q2EDBlock(features * 16, features * 8, down=False, use_bn=True, act="relu")
        self.d4 = Q2EDBlock(features * 16, features * 4, down=False, use_bn=True, act="relu")
        self.d5 = Q2EDBlock(features * 8, features * 2, down=False, use_bn=True, act="relu")
        self.d6 = Q2EDBlock(features * 4, features, down=False, use_bn=True, act="relu")

        self.final = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        e0 = self.e0(x)
        e1 = self.e1(e0)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)

        d0 = self.d0(e7)
        d1 = self.d1(torch.cat([d0, e6], dim=1))
        d2 = self.d2(torch.cat([d1, e5], dim=1))
        d3 = self.d3(torch.cat([d2, e4], dim=1))
        d4 = self.d4(torch.cat([d3, e3], dim=1))
        d5 = self.d5(torch.cat([d4, e2], dim=1))
        d6 = self.d6(torch.cat([d5, e1], dim=1))

        return self.final(torch.cat([d6, e0], dim=1))


Q2_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((Q2_IMAGE_SIZE, Q2_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


@st.cache_resource
def load_q2_pix2pix_model(device_str: str) -> nn.Module:
    device = torch.device(device_str)
    if not PIX2PIX_CKPT.exists():
        raise FileNotFoundError(f"Missing checkpoint: {PIX2PIX_CKPT}")

    payload = torch.load(PIX2PIX_CKPT, map_location=device)
    state_dict = extract_state_dict(payload)

    # Try both naming conventions used in this assignment's Pix2Pix exports.
    load_errors = []
    for model_ctor in (Q2Generator, Q2GeneratorAlt):
        model = model_ctor().to(device)
        try:
            load_model_weights(model, state_dict)
            model.eval()
            return model
        except RuntimeError as exc:
            load_errors.append(f"{model_ctor.__name__}: {exc}")

    joined_errors = "\n".join(load_errors)
    raise RuntimeError(
        "Unable to load Q2 Pix2Pix checkpoint with supported architectures.\n"
        f"Details:\n{joined_errors}"
    )


# --------------------------
# Q3 model (CycleGAN)
# --------------------------
class Q3ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3),
            nn.InstanceNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3),
            nn.InstanceNorm2d(ch),
        )

    def forward(self, x):
        return x + self.block(x)


class Q3Generator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, ngf=64, n_res=Q3_N_RES):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]

        ch = ngf
        for _ in range(2):
            layers += [
                nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(ch * 2),
                nn.ReLU(inplace=True),
            ]
            ch *= 2

        for _ in range(n_res):
            layers.append(Q3ResBlock(ch))

        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(ch, ch // 2, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(ch // 2),
                nn.ReLU(inplace=True),
            ]
            ch //= 2

        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ch, out_ch, 7), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


Q3_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((Q3_IMAGE_SIZE, Q3_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ]
)


@st.cache_resource
def load_q3_models(
    device_str: str,
    g_ab_blob: Optional[bytes],
    g_ba_blob: Optional[bytes],
    full_blob: Optional[bytes],
) -> Tuple[nn.Module, nn.Module]:
    device = torch.device(device_str)

    # Source priority:
    # 1) Uploaded G_AB + G_BA
    # 2) Uploaded full CycleGAN checkpoint
    # 3) Files in model/ directory

    sd_ab = None
    sd_ba = None

    if g_ab_blob is not None and g_ba_blob is not None:
        payload_ab = load_torch_payload_from_bytes(g_ab_blob, device)
        payload_ba = load_torch_payload_from_bytes(g_ba_blob, device)
        sd_ab = extract_state_dict(payload_ab)
        sd_ba = extract_state_dict(payload_ba)

    elif full_blob is not None:
        payload = load_torch_payload_from_bytes(full_blob, device)
        sd_ab, sd_ba = split_cyclegan_full_state_dict(payload)

    elif CYCLEGAN_GAB_CKPT.exists() and CYCLEGAN_GBA_CKPT.exists():
        payload_ab = torch.load(CYCLEGAN_GAB_CKPT, map_location=device)
        payload_ba = torch.load(CYCLEGAN_GBA_CKPT, map_location=device)
        sd_ab = extract_state_dict(payload_ab)
        sd_ba = extract_state_dict(payload_ba)

    elif CYCLEGAN_FULL_CKPT.exists():
        payload = torch.load(CYCLEGAN_FULL_CKPT, map_location=device)
        sd_ab, sd_ba = split_cyclegan_full_state_dict(payload)

    else:
        raise FileNotFoundError(
            "CycleGAN checkpoints are missing. Provide G_AB and G_BA checkpoints or a full CycleGAN checkpoint."
        )

    model_ab = Q3Generator().to(device)
    model_ba = Q3Generator().to(device)

    load_model_weights(model_ab, sd_ab)
    load_model_weights(model_ba, sd_ba)

    model_ab.eval()
    model_ba.eval()
    return model_ab, model_ba


# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="GenAI Assignment Inference", page_icon="AI", layout="wide")

theme_base = str(st.get_option("theme.base") or "light").lower()
is_dark_theme = theme_base == "dark"

if is_dark_theme:
    app_gradient = "radial-gradient(circle at top right, #101826 0%, #0d141f 45%, #0b1118 100%)"
    hero_gradient = "linear-gradient(135deg, #f5f7fa 0%, #e2e8f0 60%, #cbd5e1 100%)"
    hero_text = "#0f172a"
    hero_border = "#334155"
    card_bg = "rgba(15, 23, 42, 0.75)"
    card_border = "#334155"
else:
    app_gradient = "radial-gradient(circle at top right, #edf7ff 0%, #f8f5ee 40%, #f7fbf4 100%)"
    hero_gradient = "linear-gradient(135deg, #0f172a 0%, #1f2937 60%, #334155 100%)"
    hero_text = "#f8fafc"
    hero_border = "#64748b"
    card_bg = "rgba(255, 255, 255, 0.75)"
    card_border = "#cbd5e1"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;700;800&display=swap');

    .stApp {{
        background: {app_gradient};
        font-family: 'Manrope', sans-serif;
    }}
    .hero {{
        padding: 1rem 1.25rem;
        border-radius: 0.8rem;
        background: {hero_gradient};
        color: {hero_text};
        border: 1px solid {hero_border};
        margin-bottom: 1rem;
    }}
    .subcard {{
        padding: 0.8rem 1rem;
        border-radius: 0.6rem;
        border: 1px solid {card_border};
        background: {card_bg};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h2 style="margin:0;">GenAI Assignment 03 - Unified Inference Dashboard</h2>
      <p style="margin:0.35rem 0 0; opacity:0.9;">
        Run inference for Q1 (GAN generation), Q2 (Pix2Pix sketch-color), and Q3 (CycleGAN sketch-photo).
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Controls")
    use_cuda = st.checkbox("Use CUDA if available", value=True)
    device = get_device(use_cuda)
    st.caption(f"Device: `{device}`")

    task = st.radio(
        "Select Task",
        (
            "Q1: GAN Pokemon Generation",
            "Q2: Pix2Pix Sketch -> Color",
            "Q3: CycleGAN Sketch <-> Photo",
        ),
    )


if task == "Q1: GAN Pokemon Generation":
    st.subheader("Q1 - DCGAN / WGAN-GP Pokemon Image Generation")

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            model_choice = st.selectbox("Model", ["DCGAN", "WGAN-GP"], index=0)
        with c2:
            num_images = st.slider("Number of images", min_value=1, max_value=64, value=16, step=1)
        with c3:
            seed = st.number_input("Seed (-1 = random)", value=-1, step=1)

        generate_clicked = st.button("Generate Images", type="primary")

    if generate_clicked:
        try:
            if int(seed) >= 0:
                torch.manual_seed(int(seed))
                np.random.seed(int(seed))

            device_str = str(device)
            if model_choice == "DCGAN":
                model = load_q1_dcgan_model(device_str)
            else:
                model = load_q1_wgangp_model(device_str)

            noise = torch.randn(int(num_images), NOISE_SIZE, 1, 1, device=device)
            with torch.no_grad():
                fake_batch = model(noise)

            images = batch_to_pil_from_tanh(fake_batch)
            st.success("Inference complete.")
            st.image(images, caption=[f"{model_choice} #{i + 1}" for i in range(len(images))], width=128)

        except Exception as exc:
            st.error(f"Q1 inference failed: {exc}")


elif task == "Q2: Pix2Pix Sketch -> Color":
    st.subheader("Q2 - Pix2Pix Anime Sketch to Color")

    with st.container(border=True):
        input_mode = st.radio(
            "Input Source",
            ["Upload Sketch", "Use Built-in Sample"],
            horizontal=True,
        )

        left_up, right_up = st.columns(2)
        with left_up:
            sketch_file = st.file_uploader(
                "Upload sketch image",
                type=["png", "jpg", "jpeg", "webp"],
                disabled=input_mode != "Upload Sketch",
            )
        with right_up:
            ref_file = st.file_uploader(
                "Upload reference color image (optional)",
                type=["png", "jpg", "jpeg", "webp"],
            )

        sample_sketch = Image.new("RGB", (Q2_IMAGE_SIZE, Q2_IMAGE_SIZE), "white")
        # Create a deterministic sketch-like fallback sample when no file sample is available.
        for y in range(0, Q2_IMAGE_SIZE, 8):
            for x in range(0, Q2_IMAGE_SIZE, 8):
                if (x // 8 + y // 8) % 2 == 0:
                    sample_sketch.putpixel((x, y), (220, 220, 220))

        input_img = None
        sample_caption = "Built-in sample sketch"
        if input_mode == "Use Built-in Sample":
            if Q2_SAMPLE_IMAGE.exists():
                input_img = Image.open(Q2_SAMPLE_IMAGE).convert("RGB")
                sample_caption = "Provided sample sketch"
            else:
                input_img = sample_sketch
        elif sketch_file is not None:
            input_img = Image.open(sketch_file).convert("RGB")

        run_q2 = st.button("Colorize", type="primary", disabled=input_img is None)

    if input_img is not None:
        display_caption = sample_caption if input_mode == "Use Built-in Sample" else "Input sketch"
        st.image(input_img, caption=display_caption, width=300)

    if run_q2 and input_img is not None:
        try:
            model = load_q2_pix2pix_model(str(device))
            in_tensor = Q2_TRANSFORM(input_img).unsqueeze(0).to(device)

            with torch.no_grad():
                pred_tensor = model(in_tensor)

            output_img = tensor_to_pil_from_tanh(pred_tensor)

            left, right = st.columns(2)
            with left:
                st.image(input_img, caption="Input Sketch", use_container_width=True)
            with right:
                st.image(output_img, caption="Colorized Output", use_container_width=True)

            if ref_file is not None:
                ref_img = Image.open(ref_file).convert("RGB").resize((Q2_IMAGE_SIZE, Q2_IMAGE_SIZE))
                out_np = np.array(output_img.resize((Q2_IMAGE_SIZE, Q2_IMAGE_SIZE)), dtype=np.uint8)
                ref_np = np.array(ref_img, dtype=np.uint8)

                psnr = peak_signal_noise_ratio(ref_np, out_np, data_range=255)
                ssim = structural_similarity(ref_np, out_np, channel_axis=2, data_range=255)

                m1, m2 = st.columns(2)
                m1.metric("PSNR", f"{psnr:.2f}")
                m2.metric("SSIM", f"{ssim:.4f}")

        except Exception as exc:
            st.error(f"Q2 inference failed: {exc}")


else:
    st.subheader("Q3 - CycleGAN Sketch <-> Photo Translation")

    st.info(
        "Task 3 needs CycleGAN generator checkpoints. If not found in model/ directory, upload them below."
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        up_ab = st.file_uploader("Upload G_AB checkpoint (.pth/.pt)", type=["pth", "pt"])
    with col_b:
        up_ba = st.file_uploader("Upload G_BA checkpoint (.pth/.pt)", type=["pth", "pt"])
    with col_c:
        up_full = st.file_uploader("Or upload full CycleGAN checkpoint", type=["pth", "pt"])

    g_ab_blob = read_uploaded_checkpoint(up_ab)
    g_ba_blob = read_uploaded_checkpoint(up_ba)
    full_blob = read_uploaded_checkpoint(up_full)

    q3_available = (
        (g_ab_blob is not None and g_ba_blob is not None)
        or (full_blob is not None)
        or (CYCLEGAN_GAB_CKPT.exists() and CYCLEGAN_GBA_CKPT.exists())
        or CYCLEGAN_FULL_CKPT.exists()
    )

    if not q3_available:
        st.warning(
            "No usable Q3 checkpoints found. Upload G_AB + G_BA checkpoints or a full CycleGAN checkpoint to run inference."
        )

    with st.container(border=True):
        q3_image = st.file_uploader("Upload input image", type=["png", "jpg", "jpeg", "webp"])
        direction = st.radio(
            "Direction",
            ["Sketch -> Photo", "Photo -> Sketch"],
            horizontal=True,
        )
        show_cycle = st.checkbox("Show cycle consistency output", value=False)
        run_q3 = st.button("Translate", type="primary", disabled=(q3_image is None or not q3_available))

    if q3_image is not None:
        input_img = Image.open(q3_image).convert("RGB")
        st.image(input_img, caption="Input image", width=300)

    if run_q3 and q3_image is not None and q3_available:
        try:
            model_ab, model_ba = load_q3_models(str(device), g_ab_blob, g_ba_blob, full_blob)

            in_tensor = Q3_TRANSFORM(input_img).unsqueeze(0).to(device)
            with torch.no_grad():
                if direction == "Sketch -> Photo":
                    translated = model_ab(in_tensor)
                    cycled = model_ba(translated) if show_cycle else None
                else:
                    translated = model_ba(in_tensor)
                    cycled = model_ab(translated) if show_cycle else None

            out_img = tensor_to_pil_from_tanh(translated)

            if show_cycle and cycled is not None:
                cyc_img = tensor_to_pil_from_tanh(cycled)
                c1, c2, c3 = st.columns(3)
                c1.image(input_img.resize((Q3_IMAGE_SIZE, Q3_IMAGE_SIZE)), caption="Input", use_container_width=True)
                c2.image(out_img, caption="Translated", use_container_width=True)
                c3.image(cyc_img, caption="Cycle Reconstructed", use_container_width=True)
            else:
                c1, c2 = st.columns(2)
                c1.image(input_img.resize((Q3_IMAGE_SIZE, Q3_IMAGE_SIZE)), caption="Input", use_container_width=True)
                c2.image(out_img, caption="Translated", use_container_width=True)

        except Exception as exc:
            st.error(f"Q3 inference failed: {exc}")
