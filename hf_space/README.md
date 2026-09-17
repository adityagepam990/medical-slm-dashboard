---
title: Medical SLM Fine-Tuning Comparison
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: apache-2.0
---

# Medical SLM Fine-Tuning Comparison Dashboard

A Gradio dashboard for comparing small language models fine-tuned on
`FreedomIntelligence/medical-o1-reasoning-SFT`.

## Included experiment views

- Training configuration and parameter-efficient fine-tuning setup
- GPU and peak VRAM footprint
- Training wall-clock
- Base vs fine-tuned evaluation loss
- Base vs fine-tuned token accuracy when available
- Separate manual correctness view for experiments with a different evaluation protocol
- Per-model detail cards and Hugging Face repository links
- Raw metrics table for transparent reporting

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

## Deploy as a Hugging Face Space

Create a new **Gradio Space** and upload:

- `app.py`
- `requirements.txt`
- `README.md`
- `data/experiments.json`

The dashboard is deliberately data-driven. Add another record to
`data/experiments.json` to extend it with another model.

## Data quality note

The dashboard does not invent missing metrics. A field remains null when a run did not
capture it. Different evaluation protocols are displayed separately to avoid misleading
cross-model comparisons.

## Medical safety

Research artifact only. The dashboard compares training/evaluation behavior; it does not
establish clinical validity, diagnostic safety, or suitability for patient-facing use.
