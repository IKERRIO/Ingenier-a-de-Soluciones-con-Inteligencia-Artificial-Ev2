"""
planning/planner.py
===================
Planificador jerárquico para el agente StarCraft II v2.
Implementa IL2.3 — IE5 (esquemas de planificación) e IE6 (toma de decisiones).

El planificador descompone preguntas complejas en sub-tareas ordenadas
antes de invocar al AgentExecutor, guiando el razonamiento del LLM.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


# ─────────────────────────────────────────────────────────────────────────────
# TIPOS DE TAREA
# ─────────────────────────────────────────────────────────────────────────────
class TaskType(Enum):
    COMPARISON   = "comparison"    # Comparar ligas o métricas
    STATS_QUERY  = "stats_query"   # Consultar estadísticas de una liga
    RECOMMENDATION = "recommendation"  # Recomendaciones de entrenamiento
    EXTERNAL     = "external"      # Fuente externa (arXiv)
    GENERAL      = "general"       # Pregunta general


@dataclass
class PlanStep:
    """Un paso dentro del plan jerárquico."""
    action: str
    description: str
    priority: int = 1
    tool_hint: Optional[str] = None   # Herramienta LangChain sugerida
    status: str = "pending"


@dataclass
class Plan:
    """Plan completo para resolver una consulta."""
    goal: str
    task_type: TaskType
    steps: List[PlanStep] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    status: str = "created"

    @property
    def estimated_steps(self) -> int:
        return len(self.steps)


# ─────────────────────────────────────────────────────────────────────────────
# PLANIFICADOR JERÁRQUICO
# ─────────────────────────────────────────────────────────────────────────────
class HierarchicalPlanner:
    """
    Planificador de tres niveles:
      1. Clasificación  : identifica el tipo de tarea (IE6 — toma de decisiones)
      2. Descomposición : genera pasos ordenados por prioridad (IE5)
      3. Asignación     : sugiere qué herramienta usar en cada paso

    Ejemplos de comportamiento adaptativo:
      - "APM de Professional vs Bronze"   → COMPARISON  → 3 pasos
      - "horas de Master"                 → STATS_QUERY → 2 pasos
      - "qué hacer para subir de Gold"    → RECOMMENDATION → 3 pasos
      - "papers sobre StarCraft"          → EXTERNAL → 2 pasos
    """

    # Palabras clave para clasificar la intención
    KEYWORDS = {
        TaskType.COMPARISON:     ["vs", "comparar", "diferencia", "compara", "versus"],
        TaskType.RECOMMENDATION: ["recomienda", "subir", "mejorar", "consejo", "cómo pasar",
                                  "qué hacer", "recomendación", "para llegar"],
        TaskType.EXTERNAL:       ["paper", "investigación", "artículo", "literatura",
                                  "científico", "arxiv", "estudio"],
        TaskType.STATS_QUERY:    ["apm", "latencia", "hotkeys", "pacs", "horas",
                                  "estadística", "promedio", "cuánto", "cuántas"],
    }

    def classify(self, question: str) -> TaskType:
        """Clasifica el tipo de tarea basado en palabras clave."""
        q_lower = question.lower()
        for task_type, keywords in self.KEYWORDS.items():
            if any(kw in q_lower for kw in keywords):
                return task_type
        return TaskType.GENERAL

    def create_plan(self, question: str) -> Plan:
        """
        Crea un plan jerárquico para la pregunta recibida.
        Implementa IE5 (esquemas de planificación) e IE6 (decisión adaptativa).
        """
        task_type = self.classify(question)
        steps = self._build_steps(task_type, question)
        plan = Plan(goal=question, task_type=task_type, steps=steps)
        return plan

    def _build_steps(self, task_type: TaskType, question: str) -> List[PlanStep]:
        """Genera pasos según el tipo de tarea clasificado."""

        if task_type == TaskType.COMPARISON:
            return [
                PlanStep(1, "Recuperar estadísticas de la primera liga mencionada",
                         priority=1, tool_hint="query_league_stats"),
                PlanStep(2, "Recuperar estadísticas de la segunda liga mencionada",
                         priority=2, tool_hint="query_league_stats"),
                PlanStep(3, "Comparar métricas y calcular diferencias porcentuales",
                         priority=3, tool_hint="compare_leagues"),
                PlanStep(4, "Sintetizar conclusiones sobre las diferencias encontradas",
                         priority=4, tool_hint=None),
            ]

        elif task_type == TaskType.STATS_QUERY:
            return [
                PlanStep(1, "Identificar la liga de interés en la pregunta",
                         priority=1, tool_hint=None),
                PlanStep(2, "Consultar estadísticas agregadas de esa liga",
                         priority=2, tool_hint="query_league_stats"),
                PlanStep(3, "Interpretar los datos en contexto competitivo",
                         priority=3, tool_hint=None),
            ]

        elif task_type == TaskType.RECOMMENDATION:
            return [
                PlanStep(1, "Identificar la liga actual del jugador",
                         priority=1, tool_hint=None),
                PlanStep(2, "Calcular brecha de métricas con la liga siguiente",
                         priority=2, tool_hint="recommend_training"),
                PlanStep(3, "Generar recomendaciones priorizadas por impacto",
                         priority=3, tool_hint="recommend_training"),
                PlanStep(4, "Proporcionar plan de acción concreto y medible",
                         priority=4, tool_hint=None),
            ]

        elif task_type == TaskType.EXTERNAL:
            return [
                PlanStep(1, "Formular términos de búsqueda para arXiv",
                         priority=1, tool_hint=None),
                PlanStep(2, "Consultar API arXiv para papers relevantes",
                         priority=2, tool_hint="fetch_arxiv_papers"),
                PlanStep(3, "Sintetizar hallazgos científicos con datos del dataset",
                         priority=3, tool_hint=None),
            ]

        else:  # GENERAL
            return [
                PlanStep(1, "Analizar la pregunta y determinar información necesaria",
                         priority=1, tool_hint=None),
                PlanStep(2, "Consultar datos relevantes del dataset StarCraft II",
                         priority=2, tool_hint="query_league_stats"),
                PlanStep(3, "Formular respuesta en lenguaje natural",
                         priority=3, tool_hint=None),
            ]
