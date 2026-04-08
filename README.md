# gen_ai_project3

This repository contains GenAI Assignment 03 notebooks and checkpoints.

## Streamlit Inference App

A unified Streamlit interface is included in `app.py` for all three tasks:

1. Q1: GAN Pokemon generation (DCGAN / WGAN-GP)
2. Q2: Pix2Pix anime sketch to color translation
3. Q3: CycleGAN sketch-photo translation

### Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Checkpoint Status

Available in `model/`:

- `dcgan_generator_final.pt` (Q1 DCGAN)
- `wgangp_checkpoint.pt` (Q1 WGAN-GP, generator extracted from key `G`)
- `pix2pix_export_q2.pt` (Q2 Pix2Pix generator, both naming variants are supported)

Optional for Q2 sample mode:

- `q2_sample_input.png` (used when selecting "Use Built-in Sample" in the Q2 UI)

Available for Q3 in `model/`:

- `generator_sketch_to_photo.pth`
- `generator_photo_to_sketch.pth`

Also supported as fallback naming:

- `G_AB_final.pth`
- `G_BA_final.pth`
- or `cyclegan_final.pth`
