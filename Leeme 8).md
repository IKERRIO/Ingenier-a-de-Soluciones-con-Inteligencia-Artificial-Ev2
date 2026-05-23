# StarCraft II Performance Intelligence Agent v2
## ISY0101 – Ingeniería de Soluciones con IA – EP2

Extensión del agente EP1 con framework LangChain, sistemas de memoria de corto y largo plazo, planificación jerárquica y 4 herramientas especializadas.

---

## Arquitectura del Sistema

```
┌──────────────────────────────────────────────────────────────────┐
│  ENTRADA: Pregunta del usuario                                   │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│  HierarchicalPlanner                                             │
│  Clasifica → Descompone → Asigna herramienta sugerida            │
│  Tipos: COMPARISON | STATS_QUERY | RECOMMENDATION | EXTERNAL     │
└───────────────────────────┬──────────────────────────────────────┘
                            │ Plan (lista de pasos priorizados)
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│  Memoria de Largo Plazo — FAISS IndexFlatIP (similitud coseno)   │
│  SentenceTransformer all-MiniLM-L6-v2 (384 dims, local)          │
│  ~360 documentos indexados (stats por liga + individuales)       │
│  semantic_search(query, k=3) → contexto enriquecido              │
└───────────────────────────┬──────────────────────────────────────┘
                            │ Pregunta enriquecida con contexto
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│  AgentExecutor (LangChain)                                       │
│  Ciclo: Thought → Tool Call → Observation → Final Answer         │
│                                                                  │
│  Herramientas disponibles:                                       │
│  ├── query_league_stats(league_name)  → stats CSV               │
│  ├── compare_leagues(league_a, b)     → diferencias %           │
│  ├── recommend_training(league)       → plan de mejora          │
│  └── fetch_arxiv_papers(topic)        → papers científicos      │
└──────────┬─────────────────────────────────────────┬────────────┘
           │                                         │
           ▼                                         ▼
┌─────────────────────┐              ┌───────────────────────────┐
│  GPT-4o (LLM)       │              │  Memoria de Corto Plazo   │
│  GitHub Models API  │              │  ConversationBufferWindow │
│  temperature=0.3    │              │  Memory (k=3 turnos)      │
│  max_iterations=6   │              │  Gestionada automático    │
└──────────┬──────────┘              └───────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────┐
│  SALIDA: Respuesta fundamentada en datos reales + plan           │
└──────────────────────────────────────────────────────────────────┘
```

---

## Instalación

```bash
# Clonar repositorio
git clone https://github.com/IKERRIO/Ingenier-a-de-Soluciones-con-Inteligencia-Artificial.git
cd Ingenier-a-de-Soluciones-con-Inteligencia-Artificial

# Instalar dependencias
pip install langchain langchain-openai openai faiss-cpu sentence-transformers \
            pandas numpy python-dotenv requests

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tu GITHUB_TOKEN
```

## Variables de entorno (.env)

```
GITHUB_TOKEN=tu_token_aqui
CSV_PATH=starcraft_limpio.csv    # opcional, por defecto busca en directorio actual
OUTPUT_PATH=query_results_v2.json  # opcional
```

## Estructura del proyecto

```
starcraft_agent_v2/
├── starcraft_agent_v2.py   ← Agente principal (LangChain + orquestación)
├── demo.py                 ← Script de demostración (5 consultas)
├── memory/
│   ├── __init__.py
│   └── memory_config.py    ← FAISS IndexFlatIP (memoria largo plazo)
├── planning/
│   ├── __init__.py
│   └── planner.py          ← HierarchicalPlanner (planificación)
├── starcraft_limpio.csv    ← Dataset SkillCraft (3395 partidas)
├── query_results_v2.json   ← Resultados del demo
├── .env.example
└── README.md
```

## Ejecución

```bash
# Ejecutar demo completo (5 consultas)
python demo.py

# Usar el agente interactivamente
python -c "
from starcraft_agent_v2 import StarCraftAgentV2
agent = StarCraftAgentV2()
agent.query('Compara Professional vs Bronze en APM')
agent.query('¿Y en latencia?')  # usa memoria de corto plazo
"
```

## Herramientas implementadas

| Herramienta | Función | IL cubierto |
|---|---|---|
| `query_league_stats` | Estadísticas de una liga desde el CSV | IL2.1 |
| `compare_leagues` | Diferencias % entre dos ligas | IL2.1 |
| `recommend_training` | Plan de mejora para subir de liga | IL2.3 |
| `fetch_arxiv_papers` | Papers científicos desde arXiv API | IL2.1 |

## Sistemas de memoria

| Memoria | Tecnología | Plazo | IL cubierto |
|---|---|---|---|
| Corto plazo | `ConversationBufferWindowMemory(k=3)` | Sesión actual | IL2.2 |
| Largo plazo | FAISS `IndexFlatIP` coseno + `all-MiniLM-L6-v2` | Dataset completo | IL2.2 |

## Tecnologías

| Componente | Tecnología |
|---|---|
| LLM | GPT-4o via GitHub Models |
| Framework | LangChain (`AgentExecutor`, `@tool`, `ChatOpenAI`) |
| Embeddings | SentenceTransformer all-MiniLM-L6-v2 (384 dims, local) |
| Vector Store | FAISS IndexFlatIP (similitud coseno) |
| Memoria CP | ConversationBufferWindowMemory (k=3) |
| Planificador | HierarchicalPlanner (custom) |
| Fuente externa | arXiv REST API |
| Datos | pandas + NumPy |

## Dataset

- **Fuente**: SkillCraft1 (Thompson et al., 2013)
- **Registros**: 3,395 partidas competitivas
- **Variables**: 20 métricas de rendimiento
- **Ligas**: Bronze → Silver → Gold → Platinum → Diamond → Master → GrandMaster → Professional
