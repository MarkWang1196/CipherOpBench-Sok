# FHE Transformer Matmul Visualizer

An interactive web visualizer for understanding homomorphic-encryption-friendly matrix multiplication in Transformer models. The site presents matrix packing, alignment, slot-wise multiplication, reduction, and output reconstruction as an animated step-by-step workflow.

## Run locally

Requirements: Python 3.

```bash
cd website
python3 serve_visualizer.py
```

Then open:

```text
http://localhost:8000/transformer_visualizer.html
```

If port `8000` is occupied, choose another port:

```bash
python3 serve_visualizer.py --port 8765
```

Stop the local server with `Ctrl+C`.
