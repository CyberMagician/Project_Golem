# Project Golem: Neural Memory Visualizer

**A 3D interface for visualizing RAG (Retrieval-Augmented Generation) memory structures in real-time.**

![Project Golem Demo](https://via.placeholder.com/800x400?text=Insert+Your+Screenshot+Here)

## 🧠 What is this?
Project Golem is an experiment in **visualizing semantic space**. Instead of treating a Vector Database as a black box, Golem projects high-dimensional embeddings (768d) down to a 3D interactive "cortex."

When you query the system, it doesn't just return text—it "lights up" the specific neural pathways related to your query, allowing you to visually debug and understand how your AI associates concepts.

## 🛠️ Tech Stack
* **Embeddings:** Google `embedding-gemma-300m` (via `sentence-transformers`)
* **Dimensionality Reduction:** UMAP (Uniform Manifold Approximation and Projection)
* **Vector Database:** LanceDB (for storage), local NumPy (for fast cosine sim)
* **Frontend:** Three.js & WebGL
* **Backend:** Flask (Python)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
