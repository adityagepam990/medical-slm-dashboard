# Medical SLM Fine-Tuning Comparison Dashboard

A static dashboard comparing small language models fine-tuned on
`FreedomIntelligence/medical-o1-reasoning-SFT`. Runs entirely in the browser
(plain HTML + Plotly.js), so it can be hosted on GitHub Pages with no server.

## Host on GitHub Pages

1. Push this folder to a GitHub repository (default branch `main`).
2. In the repo go to **Settings → Pages**.
3. Under **Build and deployment → Source**, choose **GitHub Actions**.
4. The included workflow (`.github/workflows/pages.yml`) deploys on every push.
   Your site will be at `https://<user>.github.io/<repo>/`.

Alternative without Actions: choose **Deploy from a branch**, branch `main`, folder `/ (root)`.
The `.nojekyll` file is there so Jekyll doesn't interfere.

## Run locally

Browsers block `fetch` from `file://`, so serve the folder over HTTP:

```bash
python -m http.server 8000
# open http://localhost:8000/
```

## Add a model

Append a record to `data/experiments.json` using the same schema and push.
Leave a field `null` if the run didn't capture it; the dashboard shows `—` rather than estimating.

## Files

- `index.html` — the dashboard (all views, charts, tables)
- `data/experiments.json` — the data
- `vendor/plotly.min.js` — Plotly.js, bundled locally so the page works offline and without a CDN
- `hf_space/` — the original Gradio version, still deployable as a Hugging Face Space

## Interpretation guardrails

Research artifact only. The dashboard compares training/evaluation behavior; it does not
establish clinical validity, diagnostic safety, or suitability for patient-facing use.
