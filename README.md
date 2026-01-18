![Demo](app/assets/golem.gif)

Project Golem: Neural Memory Visualizer

**A 3D interface for visualizing RAG (Retrieval-Augmented Generation) memory structures in real-time.**

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
```
2. The Ingest (Build the Brain)
This script scrapes Wikipedia for 20 distinct scientific domains, vectorizes them using Gemma, and maps them to 3D space.
```bash
python ingest.py
```
Note: This creates golem_cortex.json and golem_vectors.npy. This process requires a GPU (MPS on Mac) for speed.

3. The Server (Wake the Golem)
Start the neural server to handle queries and serve the frontend.
```bash
python GolemServer.py
```
4. Visualize
Open your browser to: http://localhost:8000

🕹️ Controls
Left Click + Drag: Rotate Camera

Right Click + Drag: Pan

Scroll: Zoom

Query Bar: Type a concept (e.g., "Julius Caesar") and hit Enter to see the brain fire.

⚙️ Customization
To change the "knowledge lobes," edit the TARGETS dictionary in ingest.py. You can point this at your own PDF folders, Obsidian vaults, or other datasets.

🤝 Integrating with Qdrant/Pinecone
To use an external Vector DB:

Fetch all vectors from your collection.

Pass them through UMAP to generate the golem_cortex.json map.

Update server.py to query your endpoint instead of the local NumPy matrix.
