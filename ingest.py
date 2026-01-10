import lancedb
import wikipediaapi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import umap
from sklearn.neighbors import NearestNeighbors
import json
import torch
import numpy as np

# --- CONFIG ---
DB_PATH = "./my_lancedb"
TABLE_NAME = "golem_memories"
JSON_OUTPUT_PATH = "./golem_cortex.json"
EMBEDDING_MODEL_ID = "google/embeddinggemma-300m"

# 20 DISTINCT DOMAINS FOR HIGH DENSITY
# We map them to 5 core color groups for visual clarity
COLOR_MAP = {
    "Bio": [0.29, 0.87, 0.50],   # Green
    "Tech": [0.22, 0.74, 0.97],  # Blue
    "Phys": [0.60, 0.20, 0.80],  # Purple
    "Hist": [0.94, 0.94, 0.20],  # Gold
    "Misc": [0.98, 0.55, 0.00]   # Orange
}

TARGETS = {
    # Biology / Green
    "Neurology": "Bio", "Immunology": "Bio", "Botany": "Bio", "Genetics": "Bio",
    # Tech / Blue
    "Artificial intelligence": "Tech", "Cybernetics": "Tech", "Cryptography": "Tech", "Robotics": "Tech",
    # Physics / Purple
    "Quantum mechanics": "Phys", "Astrophysics": "Phys", "Thermodynamics": "Phys", "Optics": "Phys",
    # History / Gold
    "Roman Empire": "Hist", "Ancient Egypt": "Hist", "Renaissance": "Hist", "Industrial Revolution": "Hist",
    # Misc / Orange
    "Basketball": "Misc", "Chess": "Misc", "Music theory": "Misc", "Game theory": "Misc"
}

def ingest_dense():
    print(f"🧠 INITIALIZING DENSE CORTEX BUILDER (20 CATEGORIES)...")
    
    # 1. Load Model
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"   ↳ Accelerator: {device.upper()}")
    encoder = SentenceTransformer(EMBEDDING_MODEL_ID, device=device, trust_remote_code=True)

    # 2. Harvest
    wiki = wikipediaapi.Wikipedia(user_agent='ProjectGolem/5.0', language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    
    docs = []
    raw_texts = []
    colors = []
    
    print("\n📚 HARVESTING...")
    for category, group in TARGETS.items():
        print(f"   ↳ {category}...")
        cat_page = wiki.page(f"Category:{category}")
        if not cat_page.exists(): continue
        
        count = 0
        for member in cat_page.categorymembers.values():
            if member.ns == wikipediaapi.Namespace.MAIN and count < 100: # 100 docs per category = ~2000 total
                chunks = splitter.create_documents([member.summary])
                if chunks:
                    chunk = chunks[0]
                    docs.append({
                        "title": member.title,
                        "text": chunk.page_content,
                        "cat": category
                    })
                    raw_texts.append("Represent this document for retrieval: " + chunk.page_content)
                    colors.append(COLOR_MAP[group])
                    count += 1

    print(f"\n📦 Acquired {len(docs)} nodes. Vectorizing...")
    vectors = encoder.encode(raw_texts, show_progress_bar=True)

    # 3. 3D Projection (UMAP)
    print("   ↳ Calculating 3D Manifold...")
    # n_neighbors=30 makes the global structure tighter for dense clouds
    reducer = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.1, metric='cosine')
    embeddings_3d = reducer.fit_transform(vectors)

    # 4. Wiring (KNN)
    print("   ↳ Wiring Synapses...")
    nbrs = NearestNeighbors(n_neighbors=8, metric='cosine').fit(vectors)
    distances, indices = nbrs.kneighbors(vectors)

    # 5. Output
    cortex_data = []
    lancedb_data = []

    for i in range(len(docs)):
        cortex_data.append({
            "id": i,
            "title": docs[i]['title'],
            "cat": docs[i]['cat'],
            "pos": embeddings_3d[i].tolist(),
            "col": colors[i],
            "nbs": indices[i][1:].tolist()
        })
        lancedb_data.append({
            "text": docs[i]['text'],
            "title": docs[i]['title'],
            "category": docs[i]['cat'],
            "vector": vectors[i],
            "json_id": i
        })

    with open(JSON_OUTPUT_PATH, 'w') as f:
        json.dump(cortex_data, f)
    
    # Save vectors to disk for the active server to load quickly without re-embedding everything
    np.save("golem_vectors.npy", vectors)

    print("✅ DENSE CORTEX GENERATED.")

if __name__ == "__main__":
    ingest_dense()