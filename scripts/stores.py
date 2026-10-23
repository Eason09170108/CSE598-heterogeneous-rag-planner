"""
stores.py

Thin query interfaces over the three heterogeneous stores built in
setup_stark_env.py:
  - V (vector store):     FAISS index over node text descriptions
  - R (relational store): PostgreSQL table `drug(id, name, state, attributes)`
  - G (graph store):      NetworkX MultiDiGraph over the STaRK-Prime KG

This module does NOT build or load the stores itself. It expects the
objects produced by setup_stark_env.py (skb, G, index, all_node_ids, model,
psycopg2 connection params) to already exist in the caller's namespace, and
wraps them in a single StoreManager object so the executor (executor.py)
and the labeling script can call query_v / query_r / query_g through one
consistent interface instead of depending on notebook globals directly.

Usage (after running setup_stark_env.py in Colab):

    from scripts.stores import StoreManager
    stores = StoreManager(skb=skb, G=G, index=index, all_node_ids=all_node_ids,
                           model=model)
    stores.query_v("drug used to treat high blood pressure")
    stores.query_r(state_filter="solid")
    stores.query_g(node_id=12345, relation="target")
"""

import faiss
import psycopg2


class StoreManager:
    """
    Bundles the three store handles and exposes one query_* method per
    store. Keeping this as a class (instead of free functions over
    notebook globals) lets the executor call all three stores through a
    single object, and makes it easy to swap in mocked stores for testing.
    """

    def __init__(self, skb, G, index, all_node_ids, model,
                 pg_host="localhost", pg_database="drugdb",
                 pg_user="postgres", pg_password="postgres"):
        self.skb = skb
        self.G = G
        self.index = index
        self.all_node_ids = all_node_ids
        self.model = model
        self.pg_host = pg_host
        self.pg_database = pg_database
        self.pg_user = pg_user
        self.pg_password = pg_password

    # ===== V: semantic search over node descriptions =====
    def query_v(self, query_text, k=10):
        """
        Returns the top-k most relevant nodes as (node_id, similarity_score).
        Uses cosine similarity via L2-normalized embeddings + IndexFlatIP.
        """
        query_emb = self.model.encode([query_text])
        faiss.normalize_L2(query_emb)
        distances, indices = self.index.search(query_emb, k)
        results = [
            (self.all_node_ids[idx], float(dist))
            for idx, dist in zip(indices[0], distances[0])
        ]
        return results

    # ===== R: structured attribute filter on the drug table =====
    def query_r(self, state_filter=None, k=10):
        """
        Filters drugs by state (solid/liquid/gas), returns (node_id, name).
        If state_filter is None, returns any k rows.
        """
        conn = psycopg2.connect(
            host=self.pg_host, database=self.pg_database,
            user=self.pg_user, password=self.pg_password,
        )
        cur = conn.cursor()
        if state_filter:
            cur.execute(
                "SELECT id, name FROM drug WHERE state = %s LIMIT %s;",
                (state_filter, k),
            )
        else:
            cur.execute("SELECT id, name FROM drug LIMIT %s;", (k,))
        results = cur.fetchall()
        cur.close()
        conn.close()
        return results

    # ===== G: single-hop relation lookup =====
    def query_g(self, node_id, relation=None, k=10):
        """
        Returns neighbor nodes of node_id. If relation is None, returns all
        relation types. Output format:
        [(neighbor_id, neighbor_name, neighbor_type, relation_type)].
        """
        results = []
        for src, dst, data in self.G.out_edges(node_id, data=True):
            if relation is None or data["relation"] == relation:
                info = self.skb.node_info[dst]
                results.append((dst, info["name"], info["type"], data["relation"]))
                if len(results) >= k:
                    break
        return results
