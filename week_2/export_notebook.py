"""Execute notebook with Plotly rendered as static images, export HTML + PDF."""
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter, WebPDFExporter
import os

NB_PATH = "Task_1_perceptron+activations.ipynb"
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Force plotly to render as static images in notebook
os.environ["PLOTLY_RENDERER"] = "png"

nb = nbformat.read(NB_PATH, as_version=4)

# Inject plotly default renderer at the top
renderer_code = 'import plotly.io as pio; pio.renderers.default = "png"'
renderer_cell = nbformat.v4.new_code_cell(source=renderer_code)
renderer_cell.metadata["tags"] = ["injected"]
nb.cells.insert(0, renderer_cell)

ep = ExecutePreprocessor(timeout=120, kernel_name="python3")
ep.preprocess(nb, {"metadata": {"path": "."}})

# Remove injected cell before export
nb.cells.pop(0)

# HTML export
html_exp = HTMLExporter()
html_exp.embed_images = True
html_body, _ = html_exp.from_notebook_node(nb)
with open("Task_1_perceptron+activations.html", "w") as f:
    f.write(html_body)
print("HTML exported: Task_1_perceptron+activations.html")

# PDF export via webpdf (playwright/chromium)
try:
    pdf_exp = WebPDFExporter()
    pdf_exp.embed_images = True
    pdf_body, _ = pdf_exp.from_notebook_node(nb)
    with open("Task_1_perceptron+activations.pdf", "wb") as f:
        f.write(pdf_body)
    print("PDF exported: Task_1_perceptron+activations.pdf")
except Exception as e:
    print(f"PDF export failed: {e}")
    print("Trying HTML-to-PDF fallback via playwright...")
    from playwright.sync_api import sync_playwright
    html_path = os.path.abspath("Task_1_perceptron+activations.html")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(f"file://{html_path}")
        page.pdf(path="Task_1_perceptron+activations.pdf", format="A4",
                 print_background=True, margin={"top": "1cm", "bottom": "1cm",
                                                 "left": "1cm", "right": "1cm"})
        browser.close()
    print("PDF exported (fallback): Task_1_perceptron+activations.pdf")
