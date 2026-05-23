"""
memory/memory_config.py
=======================
Configuración de memoria para el agente StarCraft II v2.

Memoria de largo plazo: índice vectorial FAISS con similitud coseno.
Genera documentos textuales desde el dataset y los indexa para
recuperación semántica durante el pipeline RAG.
"""

import numpy as np
import pandas as pd
import faiss
from typing import List, Tuple

LEAGUE_NAMES = {
    1: "Bronze",  2: "Silver",   3: "Gold",    4: "Platinum",
    5: "Diamond", 6: "Master",   7: "GrandMaster", 8: "Professional"
}


# ─────────────────────────────────────────────────────────────────────────────
# GENERACIÓN DE DOCUMENTOS
# ─────────────────────────────────────────────────────────────────────────────
def _generate_documents(df: pd.DataFrame) -> List[dict]:
    """Convierte el DataFrame en documentos textuales para indexar."""
    docs = []

    # Documento comparativo de todas las ligas (maximiza cobertura en retrieval)
    comparacion = []
    for league_idx, group in df.groupby("LeagueIndex"):
        league = LEAGUE_NAMES.get(league_idx, "Unknown")
        comparacion.append(
            f"{league} (nivel {league_idx}/8): "
            f"APM={group['APM'].mean():.1f}, "
            f"Latencia={group['ActionLatency'].mean():.1f}ms, "
            f"Hotkeys={group['SelectByHotkeys'].mean():.4f}, "
            f"PACs={group['NumberOfPACs'].mean():.4f}, "
            f"Horas/semana={group['HoursPerWeek'].mean():.1f}h"
        )
    docs.append({
        "text": "Comparación directa de todas las ligas: " + " | ".join(comparacion),
        "meta": {"type": "comparison"}
    })

    # Estadísticas por liga
    for league_idx, group in df.groupby("LeagueIndex"):
        league = LEAGUE_NAMES.get(league_idx, "Unknown")
        docs.append({
            "text": (
                f"Estadísticas de liga {league} (nivel {league_idx}/8): "
                f"{len(group)} partidas. "
                f"APM promedio {group['APM'].mean():.1f} "
                f"(min {group['APM'].min():.1f}, max {group['APM'].max():.1f}). "
                f"ActionLatency {group['ActionLatency'].mean():.1f}ms. "
                f"SelectByHotkeys {group['SelectByHotkeys'].mean():.4f}. "
                f"NumberOfPACs {group['NumberOfPACs'].mean():.4f}. "
                f"HoursPerWeek {group['HoursPerWeek'].mean():.1f}h. "
                f"TotalHours {group['TotalHours'].mean():.0f}h. "
                f"Edad {group['Age'].mean():.1f} años."
            ),
            "meta": {"type": "league_stats", "league": league}
        })

    # Documentos individuales (1 de cada 10 jugadores)
    for _, row in df.iloc[::10].iterrows():
        league = LEAGUE_NAMES.get(int(row["LeagueIndex"]), "Unknown")
        docs.append({
            "text": (
                f"Jugador en liga {league} (nivel {int(row['LeagueIndex'])}/8). "
                f"APM: {row['APM']:.1f}. "
                f"Latencia: {row['ActionLatency']:.1f}ms. "
                f"Hotkeys: {row['SelectByHotkeys']:.4f}. "
                f"PACs: {row['NumberOfPACs']:.4f}. "
                f"Horas/semana: {row['HoursPerWeek']:.0f}h. "
                f"Edad: {row['Age']:.0f} años."
            ),
            "meta": {"type": "player", "league": league}
        })

    # Contexto general del dominio
    docs.append({
        "text": (
            "StarCraft II es un RTS de Blizzard. El dataset SkillCraft contiene 3395 partidas "
            "de jugadores de 8 ligas: Bronze, Silver, Gold, Platinum, Diamond, Master, "
            "GrandMaster y Professional. Variables clave: APM (acciones por minuto), "
            "ActionLatency (ms, menor=mejor), SelectByHotkeys (eficiencia teclado), "
            "NumberOfPACs (ciclos percepción-acción), GapBetweenPACs, TotalHours."
        ),
        "meta": {"type": "context"}
    })

    return docs


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DEL ÍNDICE FAISS
# ─────────────────────────────────────────────────────────────────────────────
def build_semantic_index(df: pd.DataFrame, embed_model) -> Tuple[List[dict], faiss.Index]:
    """
    Construye el índice vectorial FAISS para memoria de largo plazo.

    - Genera documentos textuales desde el DataFrame.
    - Codifica con SentenceTransformer (384 dims).
    - Normaliza vectores (faiss.normalize_L2) para similitud coseno exacta.
    - Retorna (docs, IndexFlatIP).
    """
    docs = _generate_documents(df)
    texts = [d["text"] for d in docs]

    embeddings = embed_model.encode(texts, show_progress_bar=False)
    matrix = np.array(embeddings, dtype="float32")

    # Normalización → IndexFlatIP = similitud coseno exacta
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    return docs, index


# ─────────────────────────────────────────────────────────────────────────────
# BÚSQUEDA SEMÁNTICA
# ─────────────────────────────────────────────────────────────────────────────
def semantic_search(
    embed_model,
    index: faiss.Index,
    docs: List[dict],
    query: str,
    k: int = 5
) -> List[dict]:
    """
    Recupera los k documentos más similares a la query usando similitud coseno.
    Implementa la recuperación de contexto semántico (IL2.2 — IE4).
    """
    q_vec = embed_model.encode([query])[0].astype("float32").reshape(1, -1)
    faiss.normalize_L2(q_vec)
    scores, indices = index.search(q_vec, k)
    return [docs[i] for i in indices[0] if i < len(docs)]
