import base64
import io
import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from src.config import (
    CLIP_MODELS,
    DEFAULT_CLIP_MODEL,
    EMBEDDINGS_DIR,
    RESULTS_DIR,
    TOP_K_VALUES,
    get_device,
)
from src.data import load_karpathy_splits, get_captions, get_image, get_all_captions_flat
from src.clip_embeddings import (
    encode_single_text,
    encode_single_text_tokenized,
    extract_and_cache_images,
    has_finetuned_checkpoint,
    load_embeddings,
    load_finetuned_model,
)


def _encode_text_for_kind(query: str, model, processor, kind: str):
    """Dispatch live-encoding of a single query based on the model kind."""
    if kind in {"scratch_vit", "scratch_cnn"}:
        tokenizer = getattr(model, "_scratch_tokenizer")
        return encode_single_text_tokenized(query, model, tokenizer)
    return encode_single_text(query, model, processor)
from src.retrieval import text_to_image_search
from src.evaluation import evaluate_text_to_image, random_baseline, save_results

st.set_page_config(page_title="Flickr30k Text-to-Image Retrieval", layout="wide")

# ─── Constants ───

MODEL_LABELS = {
    "clip-vit-b-32": "CLIP ViT-B/32 (512d)",
    "clip-vit-b-16": "CLIP ViT-B/16 (512d)",
}

MODEL_COLORS = {
    "clip-vit-b-32": "#4CAF50",
    "clip-vit-b-16": "#2196F3",
    "random": "#9E9E9E",
}

BLUE = "#005BA1"
DARK = "#1A1A2E"
LIGHT = "#E8F0FE"
MUTED = "#6B7280"
BORDER = "#E5E7EB"

IMAGES_PER_PAGE = 50
GRID_COLS = 5
CARD_HEIGHT = 200

# ─── CSS ───

st.markdown(f"""
<style>
#MainMenu, header, footer {{visibility: hidden;}}
.block-container {{padding-top: 0.5rem; max-width: 1400px;}}
.stRadio > div {{margin-bottom: -8px;}}
.stTextInput, .stNumberInput {{margin-bottom: -12px;}}
hr {{margin: 4px 0 8px 0 !important;}}

.result-card {{
    display: inline-block;
    position: relative;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid {BORDER};
    margin-bottom: 4px;
    background: white;
    transition: all 0.2s ease;
}}
.result-card:hover {{
    box-shadow: 0 4px 12px rgba(0,91,161,0.2);
    transform: translateY(-2px);
    border-color: {BLUE};
}}
.card-img {{
    width: 100% !important;
    min-width: 100%;
    object-fit: cover;
    display: block;
}}
.rank-badge {{
    position: absolute;
    top: 8px;
    left: 8px;
    background: rgba(0,91,161,0.85);
    color: white;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 4px;
}}
.sim-bar {{
    height: 4px;
    border-radius: 2px;
    background: linear-gradient(90deg, {BLUE} var(--pct), #e0e0e0 var(--pct));
}}
.score-text {{
    font-size: 0.8rem;
    color: {MUTED};
    text-align: center;
    padding: 4px 0;
}}
.gallery-card {{
    display: block;
    width: 100%;
    position: relative;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid {BORDER};
    margin-bottom: 4px;
    background: white;
}}
.caption-text {{
    font-size: 0.75rem;
    color: {MUTED};
    padding: 4px 6px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
</style>
""", unsafe_allow_html=True)


# ─── Helpers ───


def pil_to_base64(img: Image.Image, max_size: int = 300) -> str:
    img = img.copy()
    img.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def render_result_card(img: Image.Image, rank: int, score: float, height: int = CARD_HEIGHT) -> str:
    b64 = pil_to_base64(img, max_size=height)
    pct = max(0, min(100, int(score * 100)))
    return f"""
    <div class="result-card">
        <img src="data:image/jpeg;base64,{b64}" class="card-img" style="height:{height}px;">
        <div class="rank-badge">#{rank}</div>
        <div class="score-text">sim: {score:.3f}</div>
        <div class="sim-bar" style="--pct:{pct}%"></div>
    </div>
    """


def render_gallery_card(img: Image.Image, caption: str, idx: int, height: int = CARD_HEIGHT) -> str:
    b64 = pil_to_base64(img, max_size=height)
    cap = caption[:60] + ("..." if len(caption) > 60 else "")
    return f"""
    <div class="gallery-card">
        <img src="data:image/jpeg;base64,{b64}" class="card-img" style="height:{height}px;">
        <div class="caption-text" title="{caption}">{cap}</div>
    </div>
    """


# ─── Data Loading (cached) ───


@st.cache_resource
def load_data():
    splits = load_karpathy_splits()
    return splits


@st.cache_resource
def load_model_and_processor(model_key: str):
    """Load any model in the registry: base CLIP, projection, LoRA, full FT, or scratch."""
    model, processor, kind = load_finetuned_model(model_key)
    return model, processor, kind


@st.cache_resource
def load_image_embeddings(model_key: str, _dataset):
    """Load or extract image embeddings for the test split.

    Base models can lazily extract via ``extract_and_cache_images``. Fine-tuned
    models require pre-cached embeddings — they cannot be re-extracted with
    ``extract_and_cache_images`` since that always loads vanilla CLIP.
    """
    try:
        return load_embeddings(f"{model_key}_images", split="test")
    except FileNotFoundError:
        if model_key not in CLIP_MODELS:
            raise FileNotFoundError(
                f"No cached image embeddings for fine-tuned model {model_key!r}. "
                f"Re-run the corresponding training notebook to populate "
                f"data/embeddings/{model_key}_images_test.npy."
            )
        return extract_and_cache_images(_dataset, model_key, split="test")


# ─── Available models (check which have cached embeddings) ───


_FINETUNED_LABELS = {
    "projection_b32": "Projection Head (B/32, 256d)",
    "lora_b32": "LoRA (B/32, r=8)",
    "full_b32": "Full Fine-tune (B/32)",
    "scratch_vit_word": "Scratch ViT + Transformer (word vocab)",
    "scratch_vit_bpe":  "Scratch ViT + Transformer (BPE vocab)",
    "scratch_cnn_word": "Scratch CNN + Bi-LSTM (word vocab)",
    "scratch_cnn_bpe":  "Scratch CNN + Bi-LSTM (BPE vocab)",
}


def get_available_models() -> dict[str, str]:
    """Return models that are loadable: base CLIP + any fine-tuned with checkpoint."""
    available = {}
    for key, label in MODEL_LABELS.items():
        available[key] = label
    # Discover fine-tuned models from cached embeddings, but only include those
    # whose checkpoint also exists (so live encoding actually works).
    for npy_file in EMBEDDINGS_DIR.glob("*_images_test.npy"):
        name = npy_file.stem.replace("_images_test", "")
        if name in available:
            continue
        if not has_finetuned_checkpoint(name):
            continue
        available[name] = _FINETUNED_LABELS.get(name, f"{name} (fine-tuned)")
    return available


# ─── Pages ───


def page_search():
    """Main search page: text query -> ranked image results."""
    st.markdown("### Text-to-Image Search")

    splits = load_data()
    test_ds = splits["test"]

    available_models = get_available_models()
    col1, col2 = st.columns([3, 1])

    with col1:
        query = st.text_input(
            "Enter your search query",
            placeholder="e.g., A dog playing with a ball in the park",
            key="search_query",
        )

    with col2:
        model_key = st.selectbox(
            "Model",
            options=list(available_models.keys()),
            format_func=lambda k: available_models[k],
            key="search_model",
        )
        top_k = st.slider("Top-K", 1, 50, 10, key="search_topk")

    if not query:
        st.info("Enter a text query above to search for matching images.")

        # Show example queries
        st.markdown("**Example queries:**")
        examples = [
            "A man riding a bicycle on a road",
            "Children playing in a park",
            "A woman in a red dress",
            "A group of people sitting at a table",
            "A dog running on the beach",
        ]
        for ex in examples:
            if st.button(ex, key=f"example_{ex}"):
                st.session_state.search_query = ex
                st.rerun()
        return

    # Encode query
    model, processor, kind = load_model_and_processor(model_key)
    img_emb = load_image_embeddings(model_key, test_ds)

    text_emb = _encode_text_for_kind(query, model, processor, kind)

    # Retrieve
    results = text_to_image_search(text_emb, img_emb, top_k=top_k)

    st.markdown(f"**Results for:** *\"{query}\"* ({available_models[model_key]})")
    st.markdown("---")

    # Render results in grid
    cols = st.columns(min(GRID_COLS, len(results)))
    for i, (idx, score) in enumerate(results):
        col_idx = i % len(cols)
        with cols[col_idx]:
            img = get_image(test_ds, idx)
            st.markdown(
                render_result_card(img, rank=i + 1, score=score),
                unsafe_allow_html=True,
            )
            captions = get_captions(test_ds, idx)
            with st.expander(f"Captions #{i+1}"):
                for j, cap in enumerate(captions):
                    st.markdown(f"{j+1}. {cap}")


def page_gallery():
    """Browse gallery with pagination and caption search."""
    st.markdown("### Image Gallery")

    splits = load_data()
    test_ds = splits["test"]
    n_images = len(test_ds)

    # Search filter
    search = st.text_input("Filter by caption", placeholder="Search captions...", key="gallery_search")

    if search:
        search_lower = search.lower()
        matching_indices = []
        for i in range(n_images):
            captions_text = " ".join(get_captions(test_ds, i)).lower()
            if search_lower in captions_text:
                matching_indices.append(i)
        st.caption(f"Found {len(matching_indices)} images matching \"{search}\"")
    else:
        matching_indices = list(range(n_images))

    if not matching_indices:
        st.warning("No images match your search.")
        return

    # Pagination
    total_pages = max(1, (len(matching_indices) + IMAGES_PER_PAGE - 1) // IMAGES_PER_PAGE)
    page = st.number_input("Page", 1, total_pages, 1, key="gallery_page")
    start = (page - 1) * IMAGES_PER_PAGE
    end = min(start + IMAGES_PER_PAGE, len(matching_indices))
    page_indices = matching_indices[start:end]

    st.caption(f"Showing {start + 1}–{end} of {len(matching_indices)} images")

    # Render grid
    cols = st.columns(GRID_COLS)
    for i, idx in enumerate(page_indices):
        col_idx = i % GRID_COLS
        with cols[col_idx]:
            img = get_image(test_ds, idx)
            first_cap = get_captions(test_ds, idx)[0]
            st.markdown(
                render_gallery_card(img, first_cap, idx),
                unsafe_allow_html=True,
            )
            with st.expander(f"ID: {idx}"):
                for j, cap in enumerate(get_captions(test_ds, idx)):
                    st.markdown(f"{j+1}. {cap}")


def page_compare():
    """Compare retrieval results across models for the same query."""
    st.markdown("### Compare Models")

    splits = load_data()
    test_ds = splits["test"]

    available_models = get_available_models()

    query = st.text_input(
        "Search query",
        placeholder="e.g., Two dogs running in a field",
        key="compare_query",
    )

    selected_models = st.multiselect(
        "Models to compare",
        options=list(available_models.keys()),
        default=list(available_models.keys())[:2],
        format_func=lambda k: available_models[k],
        key="compare_models",
    )

    top_k = st.slider("Top-K", 1, 20, 5, key="compare_topk")

    if not query or not selected_models:
        st.info("Enter a query and select models to compare.")
        return

    st.markdown(f"**Query:** *\"{query}\"*")
    st.markdown("---")

    for model_key in selected_models:
        st.markdown(f"#### {available_models[model_key]}")

        model, processor, kind = load_model_and_processor(model_key)
        img_emb = load_image_embeddings(model_key, test_ds)
        text_emb = _encode_text_for_kind(query, model, processor, kind)
        results = text_to_image_search(text_emb, img_emb, top_k=top_k)

        cols = st.columns(min(top_k, GRID_COLS))
        for i, (idx, score) in enumerate(results):
            col_idx = i % len(cols)
            with cols[col_idx]:
                img = get_image(test_ds, idx)
                st.markdown(
                    render_result_card(img, rank=i + 1, score=score, height=180),
                    unsafe_allow_html=True,
                )
        st.markdown("---")


def page_evaluation():
    """Display Recall@K evaluation results."""
    st.markdown("### Evaluation Results")

    # Load any saved results
    all_results = {}
    if RESULTS_DIR.exists():
        for json_file in sorted(RESULTS_DIR.glob("*.json")):
            name = json_file.stem
            with open(json_file) as f:
                all_results[name] = json.load(f)

    if all_results:
        # Results table
        st.markdown("#### Recall@K Comparison")

        # Build table
        import pandas as pd
        rows = []
        for model_name, metrics in all_results.items():
            row = {"Model": model_name}
            for k, v in metrics.items():
                if k.startswith("R@"):
                    row[k] = f"{v * 100:.1f}%"
                else:
                    row[k] = f"{v:.1f}"
            rows.append(row)

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Bar chart
        st.markdown("#### Recall@K Bar Chart")
        import matplotlib.pyplot as plt

        recall_keys = sorted(
            [k for k in list(all_results.values())[0] if k.startswith("R@")],
            key=lambda x: int(x.split("@")[1]),
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(recall_keys))
        width = 0.8 / len(all_results)

        for i, (model_name, metrics) in enumerate(all_results.items()):
            values = [metrics.get(k, 0) * 100 for k in recall_keys]
            bars = ax.bar(x + i * width, values, width, label=model_name)
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7,
                )

        ax.set_ylabel("Recall (%)")
        ax.set_xticks(x + width * (len(all_results) - 1) / 2)
        ax.set_xticklabels(recall_keys)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8)
        ax.set_title("Text-to-Image Retrieval Performance")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    else:
        st.info("No evaluation results found. Run notebook 02 to generate baseline results.")

    # Run evaluation button
    st.markdown("---")
    st.markdown("#### Run Evaluation")

    splits = load_data()
    test_ds = splits["test"]
    available_models = get_available_models()

    eval_model = st.selectbox(
        "Model to evaluate",
        options=list(available_models.keys()),
        format_func=lambda k: available_models[k],
        key="eval_model",
    )

    if st.button("Run Evaluation", key="run_eval"):
        with st.spinner("Evaluating..."):
            img_emb = load_image_embeddings(eval_model, test_ds)
            captions, gt_indices = get_all_captions_flat(test_ds)

            model, processor, kind = load_model_and_processor(eval_model)
            if kind in {"scratch_vit", "scratch_cnn"}:
                from src.training import encode_texts_scratch
                tokenizer = getattr(model, "_scratch_tokenizer")
                device = next(model.parameters()).device
                txt_emb = encode_texts_scratch(captions, model, tokenizer, device=device)
            else:
                from src.clip_embeddings import encode_texts
                txt_emb = encode_texts(captions, model, processor)

            results = evaluate_text_to_image(txt_emb, img_emb, gt_indices)
            save_results(results, eval_model)

            st.success(f"Evaluation complete for {available_models[eval_model]}")
            for k, v in results.items():
                if k.startswith("R@"):
                    st.metric(k, f"{v * 100:.1f}%")
                else:
                    st.metric(k, f"{v:.1f}")


def page_explorer():
    """Embedding space explorer: t-SNE + similarity distributions."""
    st.markdown("### Embedding Explorer")

    splits = load_data()
    test_ds = splits["test"]

    available_models = get_available_models()
    model_key = st.selectbox(
        "Model",
        options=list(available_models.keys()),
        format_func=lambda k: available_models[k],
        key="explorer_model",
    )

    img_emb = load_image_embeddings(model_key, test_ds)

    st.markdown(f"**Embedding shape:** {img_emb.shape}")
    st.markdown(f"**Mean L2 norm:** {np.linalg.norm(img_emb, axis=1).mean():.4f}")

    tab1, tab2, tab3 = st.tabs(["t-SNE", "Similarity Distribution", "Statistics"])

    with tab1:
        sample_size = st.slider("Sample size", 200, 2000, 500, step=100, key="tsne_sample")

        if st.button("Generate t-SNE", key="gen_tsne"):
            with st.spinner("Computing t-SNE..."):
                from src.visualize import plot_tsne
                fig = plot_tsne(img_emb, test_ds, title=f"t-SNE: {available_models[model_key]}", sample_size=sample_size)
                st.pyplot(fig)
                import matplotlib.pyplot as plt
                plt.close(fig)

    with tab2:
        if st.button("Generate Distribution", key="gen_dist"):
            with st.spinner("Computing..."):
                from src.visualize import plot_similarity_distribution
                fig = plot_similarity_distribution(img_emb, title=f"Image-Image Similarity ({model_key})")
                st.pyplot(fig)
                import matplotlib.pyplot as plt
                plt.close(fig)

    with tab3:
        norms = np.linalg.norm(img_emb, axis=1)
        st.markdown(f"""
        | Statistic | Value |
        |-----------|-------|
        | Dimensions | {img_emb.shape[1]} |
        | Images | {img_emb.shape[0]} |
        | Mean norm | {norms.mean():.6f} |
        | Std norm | {norms.std():.6f} |
        | Min norm | {norms.min():.6f} |
        | Max norm | {norms.max():.6f} |
        """)


# ─── Navigation ───


PAGES = {
    "Search": page_search,
    "Gallery": page_gallery,
    "Compare": page_compare,
    "Evaluation": page_evaluation,
    "Explorer": page_explorer,
}


def main():
    # Compact header
    st.markdown(
        f"<h2 style='margin:0; padding:0 0 4px 0; color:{BLUE};'>"
        "Flickr30k Text-to-Image Retrieval</h2>",
        unsafe_allow_html=True,
    )

    # Horizontal page navigation
    page = st.radio(
        "Navigation",
        list(PAGES.keys()),
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown("---")

    PAGES[page]()


if __name__ == "__main__":
    main()
