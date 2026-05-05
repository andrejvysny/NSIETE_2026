"""Extend NB 06 cells 16 + 17 to include the 4 from-scratch rows."""

from __future__ import annotations

from pathlib import Path

from _nb_edit import load_nb, save_nb, set_cell_source

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "06_finetune_lora.ipynb"


CELL_16 = """\
TRAINING_COMPLETE = True  # Set to False to skip comparison

if TRAINING_COMPLETE:
    all_results: dict[str, dict] = {}

    # Locked row order: grouped by approach family.
    result_files = {
        "CLIP B/32 (baseline)":   ["baseline_b32", "baseline_clip-vit-b-32"],
        "CLIP B/16 (baseline)":   ["baseline_b16", "baseline_clip-vit-b-16"],
        "CLIP B/32 + Projection": ["projection_head_b32"],
        "CLIP B/32 Full FT":      ["full_finetune_b32"],
        "CLIP B/32 + LoRA (r=8)": ["lora_b32"],
        "Scratch ViT (word)":     ["scratch_vit_word"],
        "Scratch ViT (BPE)":      ["scratch_vit_bpe"],
        "Scratch CNN (word)":     ["scratch_cnn_word"],
        "Scratch CNN (BPE)":      ["scratch_cnn_bpe"],
    }

    for display_name, candidate_files in result_files.items():
        for fname in candidate_files:
            try:
                all_results[display_name] = load_results(fname)
                break
            except FileNotFoundError:
                continue
        else:
            print(f"Results not found for {display_name} (tried {candidate_files})")

    print_results_table(all_results, title="Text-to-Image Retrieval: All Approaches")
    save_results(all_results, "final_comparison")
"""


CELL_17 = """\
TRAINING_COMPLETE = True  # Set to False to skip bidirectional eval

if TRAINING_COMPLETE:
    # Image-to-text direction: for each image, find rank of its best caption.
    bidirectional_results: dict[str, dict] = {}

    embedding_pairs = {
        "CLIP B/32 (baseline)":   ("clip-vit-b-32_images_test",  "clip-vit-b-32_texts_test"),
        "CLIP B/16 (baseline)":   ("clip-vit-b-16_images_test",  "clip-vit-b-16_texts_test"),
        "CLIP B/32 + Projection": ("projection_b32_images_test", "projection_b32_texts_test"),
        "CLIP B/32 Full FT":      ("full_b32_images_test",       "full_b32_texts_test"),
        "CLIP B/32 + LoRA (r=8)": ("lora_b32_images_test",       "lora_b32_texts_test"),
        "Scratch ViT (word)":     ("scratch_vit_word_images_test", "scratch_vit_word_texts_test"),
        "Scratch ViT (BPE)":      ("scratch_vit_bpe_images_test",  "scratch_vit_bpe_texts_test"),
        "Scratch CNN (word)":     ("scratch_cnn_word_images_test", "scratch_cnn_word_texts_test"),
        "Scratch CNN (BPE)":      ("scratch_cnn_bpe_images_test",  "scratch_cnn_bpe_texts_test"),
    }

    for display_name, (img_file, txt_file) in embedding_pairs.items():
        img_path = EMBEDDINGS_DIR / f"{img_file}.npy"
        txt_path = EMBEDDINGS_DIR / f"{txt_file}.npy"
        if not (img_path.exists() and txt_path.exists()):
            print(f"Skipping {display_name}: embeddings not cached")
            continue
        img_e = np.load(img_path)
        txt_e = np.load(txt_path)
        i2t = evaluate_image_to_text(img_e, txt_e, test_gt)
        t2i = all_results.get(display_name, {})
        bidirectional_results[display_name] = {
            **{f"t2i/{k}": v for k, v in t2i.items()},
            **{f"i2t/{k}": v for k, v in i2t.items()},
        }

    save_results(bidirectional_results, "final_comparison_bidirectional")
    print_results_table(bidirectional_results, title="Bidirectional Retrieval (t2i + i2t)")
"""


def main() -> None:
    nb = load_nb(NB_PATH)
    set_cell_source(nb, 16, CELL_16)
    set_cell_source(nb, 17, CELL_17)
    save_nb(NB_PATH, nb)
    print(f"Extended cells 16+17 of {NB_PATH}")


if __name__ == "__main__":
    main()
