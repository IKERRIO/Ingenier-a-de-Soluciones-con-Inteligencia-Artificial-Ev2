"""
demo.py — Script de demostración EP2
=====================================
Ejecuta 5 consultas que cubren todos los tipos de tarea del planificador
y guarda los resultados en query_results_v2.json.

Uso:
    python demo.py
"""

import json
import os
from starcraft_agent_v2 import StarCraftAgentV2

QUERIES = [
    # COMPARISON — IE6: decisión adaptativa con compare_leagues
    "Compara el APM promedio y la latencia de acción entre la liga Professional y Bronze.",

    # STATS_QUERY — IE1: herramienta query_league_stats
    "¿Cuántas horas por semana practican en promedio los jugadores de liga Master?",

    # RECOMMENDATION — IE5: planificación jerárquica + recommend_training
    "¿Qué métricas debo mejorar para subir de Gold a Platinum y cómo lo hago?",

    # MEMORY TEST — IE3: consulta encadenada que requiere memoria de corto plazo
    "En base a lo anterior, ¿cuánto tiempo estimado tomaría llegar a Platinum practicando 20 horas semanales?",

    # EXTERNAL — IE4: fuente externa arXiv + recuperación semántica FAISS
    "¿Qué dice la literatura científica sobre skillcraft starcraft APM rendimiento competitivo?",
]


if __name__ == "__main__":
    agent = StarCraftAgentV2()

    results = []
    for i, q in enumerate(QUERIES, 1):
        print(f"\n{'#'*60}")
        print(f"# CONSULTA {i}/5")
        print(f"{'#'*60}")
        r = agent.query(q)
        results.append(r)

    output_path = os.getenv("OUTPUT_PATH", "query_results_v2.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Resultados guardados en {output_path}")
