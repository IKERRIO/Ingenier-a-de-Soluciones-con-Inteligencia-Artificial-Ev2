"""
StarCraft II Performance Intelligence Agent — v2
================================================
EP2 ISY0101 — Ingeniería de Soluciones con IA
Extiende el EP1 con:
  - Framework LangChain + LangGraph (create_react_agent)
  - Memoria de corto plazo  : WindowMemory manual (k=3 turnos)
  - Memoria de largo plazo  : FAISS IndexFlatIP (similitud coseno)
  - Planificación jerárquica: HierarchicalPlanner
  - Herramientas especializadas: query, compare, recommend, external

Instalación:
  pip install langchain langchain-openai langgraph openai faiss-cpu
              sentence-transformers pandas numpy python-dotenv requests

Variables de entorno (.env):
  GITHUB_TOKEN=<tu token>
  CSV_PATH=starcraft_limpio.csv   (opcional)
"""
import warnings
import os
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os
import sys
import json
import time
import threading
import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv


# ─────────────────────────────────────────────────────────────────────────────
# SPINNER ANIMADO
# ─────────────────────────────────────────────────────────────────────────────
class Spinner:
    """Spinner de consola para operaciones pesadas."""
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, mensaje: str, color: str = "\033[96m"):
        self.mensaje = mensaje
        self.color   = color
        self.reset   = "\033[0m"
        self.verde   = "\033[92m"
        self.rojo    = "\033[91m"
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self):
        i = 0
        while not self._stop.is_set():
            frame = self.FRAMES[i % len(self.FRAMES)]
            sys.stdout.write(f"\r  {self.color}{frame}{self.reset} {self.mensaje}...")
            sys.stdout.flush()
            time.sleep(0.08)
            i += 1

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, *_):
        self._stop.set()
        self._thread.join()
        if exc_type is None:
            sys.stdout.write(f"\r  {self.verde}✅{self.reset} {self.mensaje}        \n")
        else:
            sys.stdout.write(f"\r  {self.rojo}❌{self.reset} {self.mensaje}        \n")
        sys.stdout.flush()

# ── LangChain / LangGraph ───────────────────────────────────────────────────
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

# ── Embeddings y FAISS ──────────────────────────────────────────────────────
import faiss
from sentence_transformers import SentenceTransformer

# ── Módulos del proyecto ─────────────────────────────────────────────────────
from memory_config import build_semantic_index, semantic_search
from planner       import HierarchicalPlanner

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH     = os.getenv("CSV_PATH", "starcraft_limpio.csv")
EMBED_MODEL  = "all-MiniLM-L6-v2"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Si no hay .env, pega tu token aquí directamente:
GITHUB_TOKEN = ""

LEAGUE_NAMES = {
    1: "Bronze",  2: "Silver",   3: "Gold",    4: "Platinum",
    5: "Diamond", 6: "Master",   7: "GrandMaster", 8: "Professional"
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", decimal=",")
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col].str.replace(",", "."), errors="coerce")
            except Exception:
                pass
    df = df.dropna(subset=["APM", "LeagueIndex", "ActionLatency"])
    df["LeagueName"] = df["LeagueIndex"].map(LEAGUE_NAMES)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. HERRAMIENTAS  (IL2.1 — IE1, IE2)
# ─────────────────────────────────────────────────────────────────────────────
_df: pd.DataFrame = None


@tool
def query_league_stats(league_name: str) -> str:
    """
    Consulta estadísticas agregadas de una liga específica de StarCraft II.
    Acepta: Bronze, Silver, Gold, Platinum, Diamond, Master, GrandMaster, Professional.
    Devuelve APM, ActionLatency, SelectByHotkeys, NumberOfPACs y HoursPerWeek promedio.
    """
    global _df
    name_map = {v.lower(): k for k, v in LEAGUE_NAMES.items()}
    idx = name_map.get(league_name.strip().lower())
    if idx is None:
        return f"Liga '{league_name}' no reconocida. Usa: {list(LEAGUE_NAMES.values())}"
    group = _df[_df["LeagueIndex"] == idx]
    if group.empty:
        return f"No hay datos para la liga {league_name}."
    return (
        f"Estadísticas de liga {league_name} (nivel {idx}/8) — {len(group)} partidas:\n"
        f"  APM promedio       : {group['APM'].mean():.1f}\n"
        f"  ActionLatency (ms) : {group['ActionLatency'].mean():.1f}\n"
        f"  SelectByHotkeys    : {group['SelectByHotkeys'].mean():.4f}\n"
        f"  NumberOfPACs       : {group['NumberOfPACs'].mean():.4f}\n"
        f"  HoursPerWeek       : {group['HoursPerWeek'].mean():.1f}\n"
        f"  TotalHours         : {group['TotalHours'].mean():.0f}\n"
        f"  Edad promedio      : {group['Age'].mean():.1f} años"
    )


@tool
def compare_leagues(league_a: str, league_b: str) -> str:
    """
    Compara dos ligas de StarCraft II en todas las métricas clave.
    Calcula diferencias porcentuales entre APM, latencia, hotkeys y PACs.
    Ejemplo: compare_leagues('Bronze', 'Professional')
    """
    global _df
    name_map = {v.lower(): k for k, v in LEAGUE_NAMES.items()}
    idx_a = name_map.get(league_a.strip().lower())
    idx_b = name_map.get(league_b.strip().lower())
    if idx_a is None or idx_b is None:
        return "Una o ambas ligas no reconocidas."
    ga = _df[_df["LeagueIndex"] == idx_a]
    gb = _df[_df["LeagueIndex"] == idx_b]
    metrics = ["APM", "ActionLatency", "SelectByHotkeys", "NumberOfPACs", "HoursPerWeek"]
    lines = [f"Comparación {league_a} vs {league_b}:"]
    for m in metrics:
        va, vb = ga[m].mean(), gb[m].mean()
        diff = ((vb - va) / va * 100) if va != 0 else 0
        lines.append(f"  {m:<22}: {league_a}={va:.2f}  {league_b}={vb:.2f}  (Δ {diff:+.1f}%)")
    return "\n".join(lines)


@tool
def recommend_training(current_league: str) -> str:
    """
    Genera recomendaciones de entrenamiento para subir de liga en StarCraft II.
    Calcula la brecha entre la liga actual y la siguiente y sugiere qué mejorar.
    Ejemplo: recommend_training('Gold')
    """
    global _df
    name_map = {v.lower(): k for k, v in LEAGUE_NAMES.items()}
    idx = name_map.get(current_league.strip().lower())
    if idx is None:
        return f"Liga '{current_league}' no reconocida."
    next_idx = idx + 1
    if next_idx > 8:
        return "Ya estás en la liga máxima (Professional)."
    current_name = LEAGUE_NAMES[idx]
    next_name    = LEAGUE_NAMES[next_idx]
    gc = _df[_df["LeagueIndex"] == idx]
    gn = _df[_df["LeagueIndex"] == next_idx]
    metrics = {
        "APM"             : ("aumentar", "practica micro y macro constantemente"),
        "ActionLatency"   : ("reducir",  "entrena reflejos con micro challenges"),
        "SelectByHotkeys" : ("aumentar", "memoriza control groups y atajos"),
        "NumberOfPACs"    : ("aumentar", "cicla la cámara y el minimapa cada 2-3s"),
    }
    lines = [f"Para subir de {current_name} a {next_name}:"]
    for m, (direction, tip) in metrics.items():
        vc, vn = gc[m].mean(), gn[m].mean()
        gap = abs(vn - vc)
        lines.append(f"  • {m}: debes {direction} {gap:.2f} ({vc:.2f} → {vn:.2f}). Tip: {tip}.")
    return "\n".join(lines)


@tool
def fetch_arxiv_papers(topic: str) -> str:
    """
    Fuente externa: consulta la API de arXiv para papers científicos sobre
    StarCraft II, SkillCraft o rendimiento en e-sports.
    Devuelve títulos y resúmenes de los 2 papers más relevantes.
    """
    import xml.etree.ElementTree as ET
    ns = "http://www.w3.org/2005/Atom"

    # Enriquecer query con términos del dominio
    enriched = topic.replace(" ", "+") + "+skillcraft+starcraft+esports+APM"
    url = (f"https://export.arxiv.org/api/query"
           f"?search_query=all:{enriched}&max_results=2&sortBy=relevance")
    try:
        resp = requests.get(url, timeout=10,
                            headers={"User-Agent": "StarCraftAgent-EP2/1.0"})
        root    = ET.fromstring(resp.text)
        results = []
        for entry in root.findall(f"{{{ns}}}entry"):
            title   = entry.findtext(f"{{{ns}}}title",   "").strip()
            summary = entry.findtext(f"{{{ns}}}summary", "").strip()[:300]
            if title:
                results.append(f"Título: {title}\nResumen: {summary}")
        return "\n\n".join(results) if results else "No se encontraron papers."
    except Exception as e:
        return f"Error consultando arXiv: {e}"

TOOLS = [query_league_stats, compare_leagues, recommend_training, fetch_arxiv_papers]
# ─────────────────────────────────────────────────────────────────────────────
# 3. AGENTE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
class StarCraftAgentV2:
    """
    Agente conversacional StarCraft II — EP2.

    Componentes:
        LLM        : GPT-4o via GitHub Models (ChatOpenAI)
        Framework  : LangGraph create_react_agent
        Herramientas: 4 @tool LangChain
        Memoria CP : Lista manual (k=3 turnos, 6 mensajes)
        Memoria LP : FAISS IndexFlatIP coseno + SentenceTransformer
        Planificador: HierarchicalPlanner (clasifica y descompone tareas)
    """

    def __init__(self):
        global _df

        CYAN  = "\033[96m"
        BOLD  = "\033[1m"
        RESET = "\033[0m"
        AMARILLO = "\033[93m"

        print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
        print(f"{BOLD}{CYAN}  🎮  StarCraft II Intelligence Agent  v2{RESET}")
        print(f"{BOLD}{CYAN}{'─'*60}{RESET}\n")

        # ── LLM ────────────────────────────────────────────────────────────
        with Spinner("Configurando LLM GPT-4o (GitHub Models)"):
            self.llm = ChatOpenAI(
                model="gpt-4o",
                base_url="https://models.github.ai/inference",
                api_key=GITHUB_TOKEN,
                temperature=0.3,
            )

        # ── Dataset ────────────────────────────────────────────────────────
        with Spinner(f"Cargando dataset SkillCraft ({CSV_PATH})"):
            _df     = load_data(CSV_PATH)
            self.df = _df
        print(f"     {AMARILLO}↳ {len(_df):,} partidas · {_df['LeagueIndex'].nunique()} ligas{RESET}")

        # ── Memoria de corto plazo (IL2.2 — IE3) ───────────────────────────
        self.chat_history = []
        self.k            = 3
        print(f"  \033[92m✅\033[0m Memoria CP: WindowMemory k=3 (6 mensajes)")

        # ── SentenceTransformer ─────────────────────────────────────────────
        with Spinner("Cargando modelo de embeddings (all-MiniLM-L6-v2)"):
            self.embed_model = SentenceTransformer(EMBED_MODEL)

        # ── Índice FAISS ────────────────────────────────────────────────────
        with Spinner("Construyendo índice FAISS (memoria largo plazo)"):
            self.docs, self.faiss_index = build_semantic_index(self.df, self.embed_model)
        print(f"     {AMARILLO}↳ {self.faiss_index.ntotal} documentos indexados · similitud coseno{RESET}")

        # ── Planificador (IL2.3 — IE5, IE6) ────────────────────────────────
        with Spinner("Inicializando HierarchicalPlanner"):
            self.planner = HierarchicalPlanner()
            time.sleep(0.3)   # visual pause

        # ── Agente LangGraph (IL2.1 — IE1, IE2) ────────────────────────────
        with Spinner("Montando agente LangGraph (create_react_agent)"):
            self.agent = create_react_agent(self.llm, TOOLS)
            time.sleep(0.2)

        print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
        print(f"{BOLD}  ✅  Agente listo — escribe tu consulta{RESET}")
        print(f"{BOLD}{CYAN}{'─'*60}{RESET}\n")

    # ── Consulta ─────────────────────────────────────────────────────────────
    def query(self, question: str) -> dict:
        """
        Pipeline:
        1. HierarchicalPlanner clasifica y descompone la pregunta.
        2. FAISS recupera contexto semántico (memoria LP).
        3. create_react_agent ejecuta herramientas y genera respuesta.
        4. WindowMemory actualiza historial (memoria CP).
        """
        print(f"\n{'='*60}")
        print(f"❓ Pregunta: {question}")
        print(f"{'='*60}")

        # Paso 1: Planificación
        plan = self.planner.create_plan(question)
        print(f"\n📋 Plan ({plan.task_type.value}, {len(plan.steps)} pasos):")
        for i, step in enumerate(plan.steps, 1):
            print(f"   {i}. {step.description}")

        # Paso 2: Recuperación semántica FAISS
        semantic_ctx  = semantic_search(
            self.embed_model, self.faiss_index, self.docs, question, k=3
        )
        context_text  = "\n".join([d["text"][:200] for d in semantic_ctx])
        enriched_q    = (
            f"{question}\n\n"
            f"[Contexto semántico del dataset]:\n{context_text}"
        )

        # Paso 3: Invocar agente con historial (memoria CP)
        messages = self.chat_history[-(self.k * 2):] + [
            HumanMessage(content=enriched_q)
        ]
        try:
            result = self.agent.invoke({"messages": messages})
            answer = result["messages"][-1].content
        except Exception as e:
            answer = f"Error en el agente: {e}"

        # Paso 4: Actualizar memoria CP
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))

        print(f"\n💬 Respuesta:\n{answer}")
        return {
            "question" : question,
            "plan"     : [s.description for s in plan.steps],
            "answer"   : answer
        }


# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = StarCraftAgentV2()

    queries = [
        "Compara el APM promedio y la latencia entre Professional y Bronze.",
        "¿Cuántas horas por semana practican los jugadores de liga Master?",
        "¿Qué métricas debo mejorar para subir de Gold a Platinum?",
        "En base a lo anterior, ¿cuánto tiempo tomaría con 20 horas semanales?",
        "¿Qué dice la literatura científica sobre APM y nivel competitivo en StarCraft II?",
    ]

    results = []
    for q in queries:
        r = agent.query(q)
        results.append(r)

    with open("query_results_v2.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n✅ Resultados guardados en query_results_v2.json")
