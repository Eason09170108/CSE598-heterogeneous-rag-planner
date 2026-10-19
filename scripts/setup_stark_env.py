
import subprocess, sys

# ===== 1. Install stark-qa with workarounds =====
subprocess.run([sys.executable, "-m", "pip", "install", "stark-qa", "--no-deps", "-q"])
subprocess.run([sys.executable, "-m", "pip", "install",
                 "ogb", "torch_geometric", "langchain_text_splitters", "PyTDC",
                 "fuzzywuzzy", "langdetect", "torchmetrics", "lightning_utilities",
                 "faiss-cpu", "sentence-transformers",
                 "--no-deps", "-q"])

from unittest.mock import MagicMock
for fake_mod in ["cellxgene_census", "gget", "tiledbsoma", "bm25s",
                  "colbert", "colbert.infra", "colbert.data", "anthropic"]:
    sys.modules[fake_mod] = MagicMock()

from stark_qa import load_qa, load_skb
qa = load_qa("prime")
skb = load_skb("prime", download_processed=True, root=None)

# ===== 2. Build R (PostgreSQL) =====
subprocess.run(["apt-get", "-y", "install", "postgresql", "postgresql-contrib"], capture_output=True)
subprocess.run(["service", "postgresql", "start"], capture_output=True)

import psycopg2, json, re

def extract_state(raw):
    if raw is None:
        return None
    match = re.search(r"is an? (\w+)\.?$", raw.strip())
    return match.group(1).lower() if match else None

subprocess.run(["sudo", "-u", "postgres", "psql", "-c", "ALTER USER postgres PASSWORD \'postgres\';"], capture_output=True)
subprocess.run(["sudo", "-u", "postgres", "createdb", "drugdb"], capture_output=True)

conn = psycopg2.connect(host="localhost", database="drugdb", user="postgres", password="postgres")
cur = conn.cursor()
cur.execute("""
    DROP TABLE IF EXISTS drug;
    CREATE TABLE drug (
        id TEXT PRIMARY KEY,
        name TEXT,
        state TEXT,
        attributes JSONB
    );
""")
conn.commit()

drug_ids = skb.get_node_ids_by_type("drug")
rows = []
for did in drug_ids:
    info = skb.node_info[did]
    name = info.get("name", None)
    raw_state = info.get("details", {}).get("state", None)
    state = extract_state(raw_state)
    rows.append((str(did), name, state, json.dumps({})))

cur.executemany("INSERT INTO drug (id, name, state, attributes) VALUES (%s, %s, %s, %s)", rows)
conn.commit()
cur.close()
conn.close()
print(f"R store rebuilt: {len(rows)} drug rows.")

# ===== 3. Load G (NetworkX) from saved pickle =====
import pickle
with open("/content/drive/MyDrive/graph_store.pkl", "rb") as f:
    G = pickle.load(f)
print(f"G store loaded: {G.number_of_edges()} edges, {G.number_of_nodes()} nodes.")

# ===== 4. Load V (FAISS) from saved index =====
import faiss
from sentence_transformers import SentenceTransformer

index = faiss.read_index("/content/drive/MyDrive/vector_store.faiss")
with open("/content/drive/MyDrive/vector_store_ids.pkl", "rb") as f:
    all_node_ids = pickle.load(f)

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
print(f"V store loaded: {index.ntotal} vectors (device: {device}).")

print("All stores (R, G, V) ready.")
