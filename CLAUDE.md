# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

University course repo for NSIETE (Neural Networks) @ FIIT STU. Weekly Jupyter notebook exercises building neural network concepts from scratch using NumPy. No frameworks like PyTorch/TensorFlow for core implementations — only NumPy for math, Plotly for visualization.

## Structure

- `week_N/` — each week's notebook(s) and supporting files
- `requirements.txt` — shared Python dependencies
- `course/` — lecture PDFs (read-only reference material)
- `NSIETE_Project/` — separate git repo for speed dating EDA project (independent from main repo)

## Setup

```bash
# Root-level venv (preferred)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Per-week `venv/` folders may also exist. Key deps: numpy, matplotlib, plotly, scikit-image, opencv-python, pandas, imageio, wandb, nbformat.

## Running Notebooks

```bash
# Execute notebook in-place (primary development workflow)
jupyter nbconvert --to notebook --execute --inplace week_N/notebook.ipynb

# Or use Jupyter for interactive work
jupyter notebook

# Export notebook to HTML+PDF (week_2 utility)
python week_2/export_notebook.py week_2/Task_1_perceptron+activations.ipynb
```

## Architecture Pattern (Week 2+)

Custom neural network framework mimicking PyTorch's structure, built incrementally across weeks:

- **`Module`** (in `utils.py`) — base class with `forward()`, `backward()`, `__call__` delegates to `forward()`, stores sub-modules in `OrderedDict`
- **`Linear`** — fully-connected layer: weights `W` shape `(out, in)`, bias `b` shape `(out, 1)`. Stores `dW`/`db` gradients after backward. Forward = `W @ input + b`
- **Activations** (`Sigmoid`, `Tanh`, `ReLU`, `LeakyReLU`) — each a `Module` subclass with `forward()`/`backward()`
- **Loss functions** (`SELoss`/`MSELoss`, `BCELoss`) — `Module` subclasses with `forward(input, target)` and `backward(input, target)` signatures (not standard Module signature)
- **`Model`** — container `Module` that iterates `self.modules` in order for forward, reversed for backward
- **Optimizers** (week 4+) — `SGD`, `SGDMomentum`, `RMSprop`, `Adam`. Call `layer.get_optimizer_context()` → `[[W, dW], [b, db]]`, update, then `layer.set_optimizer_context([W, b])`

### Shared utility files per week

Each week's directory may contain:

- `utils.py` — `Module` base class + `gradient_check()` function
- `dataset.py` — dataset generators (`dataset_Circles`, `dataset_Flower`, `MakeBatches`) + `draw_dataset()`

These are redefined per week (not shared across weeks) — classes evolve as new features are added.

### Input convention

Features along rows, samples along columns. Single sample shape `(features, 1)`, batch shape `(features, N)`.

### Training loop pattern (week 4+)

```python
for epoch in range(num_epochs):
    for mini_batch_x, mini_batch_y in dataset:
        y_hat = model.forward(mini_batch_x)
        loss = criterion(y_hat, mini_batch_y)
        model.backward(criterion.backward(y_hat, mini_batch_y))
        optimizer.step(model)
```

## Conventions

- All tensor math uses raw NumPy — no autograd, no framework shortcuts
- Document tensor shapes in comments (e.g. `# shape <10; 1>`)
- Notebooks contain TODO markers for student sections:
  - `# >>>>>>>>> add here` / `# <<<<<<<<<`
  - `# >>>> start_solution` / `# <<<< end_solution`
  - `# >>>> start here` / `# <<<< end here`
  - `###>>> start of solution` / `###<<< end of solution`
- Plotly preferred for visualization; matplotlib used occasionally (e.g. decision boundaries in week_4)
- Verify backward implementations with `gradient_check()` from `utils.py`
