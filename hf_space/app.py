import json
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DATA = Path(__file__).parent / "data" / "experiments.json"

def load_data():
    with open(DATA, "r", encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))

df = load_data()

def fmt(v, digits=2, suffix=""):
    if pd.isna(v):
        return "—"
    if isinstance(v, (int, float)):
        return f"{v:.{digits}f}{suffix}"
    return str(v)

def repo_md(row):
    if pd.isna(row.get("hf_repo")) or not row.get("hf_repo"):
        return "Not linked"
    repo = row["hf_repo"]
    return f"[{repo}](https://huggingface.co/{repo})"

def overview_cards(data):
    models = len(data)
    linked = int(data["hf_repo"].notna().sum())
    total_train = int(data["train_samples"].fillna(0).sum())
    best_loss_drop = data["loss_reduction_pct"].dropna()
    loss_txt = f"{best_loss_drop.max():.2f}%" if len(best_loss_drop) else "—"
    return f"""
    <div class="cards">
      <div class="card"><div class="k">Models compared</div><div class="v">{models}</div></div>
      <div class="card"><div class="k">HF repos linked</div><div class="v">{linked}</div></div>
      <div class="card"><div class="k">Training samples</div><div class="v">{total_train:,}</div></div>
      <div class="card"><div class="k">Largest eval-loss drop</div><div class="v">{loss_txt}</div></div>
    </div>
    """

def make_training_table(data):
    cols = [
        "model","parameters_b","method","train_samples","epochs","lora_rank","lora_alpha",
        "effective_batch","max_seq_len","seed","gpu","training_minutes",
        "peak_vram_allocated_gb","final_train_loss"
    ]
    out = data[cols].copy()
    out.columns = [
        "Model","Params (B)","Method","Train samples","Epochs","LoRA r","LoRA α",
        "Effective batch","Max seq","Seed","GPU","Training (min)",
        "Peak VRAM alloc. (GB)","Final train loss"
    ]
    return out

def loss_chart(data):
    d = data.dropna(subset=["base_eval_loss","adapter_eval_loss"]).copy()
    if d.empty:
        return go.Figure().update_layout(title="No comparable loss metrics")
    long = d.melt(
        id_vars=["model"],
        value_vars=["base_eval_loss","adapter_eval_loss"],
        var_name="stage",
        value_name="eval_loss"
    )
    long["stage"] = long["stage"].map({
        "base_eval_loss":"Base",
        "adapter_eval_loss":"Fine-tuned"
    })
    fig = px.bar(long, x="model", y="eval_loss", barmode="group",
                 text_auto=".3f", labels={"model":"","eval_loss":"Evaluation loss","stage":""})
    fig.update_layout(title="Base vs Fine-tuned Evaluation Loss", legend_title_text="")
    return fig

def token_acc_chart(data):
    d = data.dropna(subset=["base_token_accuracy","adapter_token_accuracy"]).copy()
    if d.empty:
        return go.Figure().update_layout(title="No paired token-accuracy metrics")
    long = d.melt(
        id_vars=["model"],
        value_vars=["base_token_accuracy","adapter_token_accuracy"],
        var_name="stage",
        value_name="accuracy"
    )
    long["stage"] = long["stage"].map({
        "base_token_accuracy":"Base",
        "adapter_token_accuracy":"Fine-tuned"
    })
    long["accuracy_pct"] = long["accuracy"] * 100
    fig = px.bar(long, x="model", y="accuracy_pct", barmode="group",
                 text_auto=".2f", labels={"model":"","accuracy_pct":"Mean token accuracy (%)","stage":""})
    fig.update_layout(title="Paired Token Accuracy", legend_title_text="")
    return fig

def manual_acc_chart(data):
    d = data.dropna(subset=["base_manual_accuracy","adapter_manual_accuracy"]).copy()
    if d.empty:
        return go.Figure().update_layout(title="No manual-correctness metrics")
    long = d.melt(
        id_vars=["model"],
        value_vars=["base_manual_accuracy","adapter_manual_accuracy"],
        var_name="stage",
        value_name="accuracy"
    )
    long["stage"] = long["stage"].map({
        "base_manual_accuracy":"Base",
        "adapter_manual_accuracy":"Fine-tuned"
    })
    long["accuracy_pct"] = long["accuracy"] * 100
    fig = px.bar(long, x="model", y="accuracy_pct", barmode="group",
                 text_auto=".1f", labels={"model":"","accuracy_pct":"Manual correctness (%)","stage":""})
    fig.update_layout(title="Manual Held-out Correctness (separate protocol)", legend_title_text="")
    return fig

def vram_chart(data):
    d = data.dropna(subset=["peak_vram_allocated_gb"]).copy()
    fig = px.bar(d, x="model", y="peak_vram_allocated_gb", text_auto=".2f",
                 labels={"model":"","peak_vram_allocated_gb":"Peak allocated VRAM (GB)"})
    fig.update_layout(title="Peak Training VRAM (captured runs only)")
    return fig

def efficiency_chart(data):
    d = data.dropna(subset=["training_minutes"]).copy()
    d["hours"] = d["training_minutes"] / 60.0
    fig = px.scatter(
        d, x="hours", y="parameters_b", size="train_samples",
        hover_name="model", hover_data=["gpu","train_samples","epochs"],
        labels={"hours":"Training wall-clock (hours)","parameters_b":"Model size (B params)"}
    )
    fig.update_layout(title="Training Cost / Scale View")
    return fig

def model_detail(model):
    row = df[df["model"] == model].iloc[0]
    loss_delta = "—"
    if pd.notna(row["base_eval_loss"]) and pd.notna(row["adapter_eval_loss"]):
        loss_delta = f'{row["base_eval_loss"]:.4f} → {row["adapter_eval_loss"]:.4f} ({row["loss_reduction_pct"]:.2f}% reduction)'
    tok = "—"
    if pd.notna(row["base_token_accuracy"]) and pd.notna(row["adapter_token_accuracy"]):
        tok = f'{row["base_token_accuracy"]*100:.2f}% → {row["adapter_token_accuracy"]*100:.2f}% (+{row["token_accuracy_gain_pp"]:.2f} pp)'
    manual = "—"
    if pd.notna(row["base_manual_accuracy"]) and pd.notna(row["adapter_manual_accuracy"]):
        manual = f'{row["base_manual_accuracy"]*100:.1f}% → {row["adapter_manual_accuracy"]*100:.1f}% (+{row["manual_accuracy_gain_pp"]:.1f} pp)'
    return f"""
### {row['model']}

**Experiment:** {row['experiment']}  
**Base model:** `{row['base_model']}`  
**Hub repo:** {repo_md(row)}  
**Dataset:** `{row['dataset']}`  
**Method:** {row['method']}  
**Training:** {int(row['train_samples']):,} samples · {int(row['epochs'])} epoch(s) · {fmt(row['training_minutes'])} min  
**Hardware:** {row['gpu']} · peak allocated VRAM {fmt(row['peak_vram_allocated_gb'])} GB  
**LoRA:** r={fmt(row['lora_rank'],0)}, α={fmt(row['lora_alpha'],0)} · effective batch {fmt(row['effective_batch'],0)} · max seq {fmt(row['max_seq_len'],0)}  
**Final train loss:** {fmt(row['final_train_loss'],4)}  
**Eval loss:** {loss_delta}  
**Token accuracy:** {tok}  
**Manual correctness:** {manual}  
**Evaluation protocol:** {row['evaluation_type']}  

> {row['notes']}
"""

def raw_table(data):
    keep = [
        "model","hf_repo","base_model","train_samples","eval_samples","epochs","gpu",
        "peak_vram_allocated_gb","training_minutes","final_train_loss",
        "base_eval_loss","adapter_eval_loss","loss_reduction_pct",
        "base_token_accuracy","adapter_token_accuracy",
        "base_manual_accuracy","adapter_manual_accuracy","evaluation_type"
    ]
    return data[keep]

CSS = """
.gradio-container {max-width: 1350px !important;}
.hero {padding: 18px 4px 8px 4px;}
.hero h1 {font-size: 2.1rem; margin-bottom: .35rem;}
.sub {opacity:.75; font-size:1rem;}
.cards {display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:12px 0 18px;}
.card {border:1px solid rgba(128,128,128,.22); border-radius:14px; padding:16px 18px; background:rgba(128,128,128,.06);}
.k {font-size:.82rem; opacity:.68; margin-bottom:6px;}
.v {font-size:1.65rem; font-weight:700;}
.note {font-size:.92rem; opacity:.82; padding:10px 0;}
@media(max-width:800px){.cards{grid-template-columns:repeat(2,1fr);}}
"""

with gr.Blocks(css=CSS, title="Medical SLM Fine-Tuning Comparison") as demo:
    gr.HTML("""
    <div class="hero">
      <h1>Medical SLM Fine-Tuning Comparison</h1>
      <div class="sub">Medical-O1 reasoning adapters · training efficiency · evaluation behavior · hardware footprint</div>
    </div>
    """)
    gr.HTML(overview_cards(df))

    with gr.Tabs():
        with gr.Tab("Executive view"):
            with gr.Row():
                gr.Plot(loss_chart(df))
                gr.Plot(vram_chart(df))
            with gr.Row():
                gr.Plot(efficiency_chart(df))
                gr.Plot(token_acc_chart(df))
            gr.Markdown("""
**How to read this page:** evaluation protocols are intentionally kept separate.  
Loss and token accuracy are directly comparable only when the held-out set and scoring protocol are the same.  
The Qwen2.5-3B manual 8-question evaluation is therefore shown separately from the 500-example automated evaluations.
""")
            gr.Plot(manual_acc_chart(df))

        with gr.Tab("Training & hardware"):
            gr.Dataframe(make_training_table(df), interactive=False, wrap=True)
            gr.Markdown("""
Use this page to showcase **resource efficiency**: model size, effective batch size, training time,
GPU class, and peak allocated VRAM. Missing values are shown as `—` rather than estimated.
""")

        with gr.Tab("Model behavior"):
            selector = gr.Dropdown(
                choices=df["model"].tolist(),
                value=df["model"].tolist()[0],
                label="Choose a model"
            )
            detail = gr.Markdown(model_detail(df["model"].tolist()[0]))
            selector.change(model_detail, inputs=selector, outputs=detail)

        with gr.Tab("Raw comparison data"):
            gr.Dataframe(raw_table(df), interactive=False, wrap=True)
            gr.Markdown("""
The dashboard reads `data/experiments.json`. Add future models there using the same schema and restart the Space.
This makes the dashboard extensible without rewriting the visualization code.
""")

    gr.Markdown("""
---
### Interpretation guardrails
- This is a **research comparison dashboard**, not a clinical validation tool.
- A lower evaluation loss indicates better fit to the evaluation distribution, not clinical safety.
- Token accuracy and manually graded answer correctness measure different behavior.
- Small manually graded samples should be presented as exploratory evidence, not definitive model ranking.
""")

if __name__ == "__main__":
    demo.launch()
