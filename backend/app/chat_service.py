from __future__ import annotations

import duckdb
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.config import get_settings
from app.ai import llm_json
from app.models import Dataset, SemanticCatalog
from app.retrieval import PostgresDatasetRetriever

settings = get_settings()


def normalize_text(value: str) -> str:
    clean = unicodedata.normalize("NFKD", value.lower())
    clean = "".join(char for char in clean if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9_ ]+", " ", clean)


def ready_datasets(db: Session) -> list[Dataset]:
    return db.query(Dataset).filter(Dataset.status == "ready").order_by(Dataset.created_at.desc()).all()


def pick_primary_dataset(datasets: list[Dataset], dataset_id: str) -> Dataset:
    return next((dataset for dataset in datasets if dataset.id == dataset_id), datasets[0])


def dataset_context(dataset: Dataset) -> dict[str, Any]:
    profile = dataset.profile_json or {}
    return {
        "dataset_id": dataset.id,
        "filename": dataset.filename,
        "description": dataset.description,
        "summary": dataset.summary,
        "analysis": profile.get("analysis_json"),
        "sheets": profile.get("sheets", []),
        "executive_overview": profile.get("executive_overview", []),
    }


def catalog_rows_for_datasets(db: Session, dataset_ids: list[str]) -> list[SemanticCatalog]:
    if not dataset_ids:
        return []
    return (
        db.query(SemanticCatalog)
        .filter(SemanticCatalog.dataset_id.in_(dataset_ids))
        .order_by(SemanticCatalog.dataset_id, SemanticCatalog.sheet_name, SemanticCatalog.id)
        .all()
    )


def build_schema_context(db: Session, datasets: list[Dataset]) -> list[dict[str, Any]]:
    catalog_rows = catalog_rows_for_datasets(db, [dataset.id for dataset in datasets[:4]])
    grouped: dict[str, dict[str, Any]] = {}
    for index, dataset in enumerate(datasets[:4]):
        grouped[dataset.id] = {
            "dataset_id": dataset.id,
            "filename": dataset.filename,
            "db_alias": f"ds{index}",
            "duckdb_path": str(settings.storage_dir / "duckdb" / f"{dataset.id}.duckdb"),
            "tables": defaultdict(lambda: {"sheet_name": "", "duckdb_table": "", "columns": []}),
        }

    for row in catalog_rows:
        dataset_entry = grouped.get(row.dataset_id)
        if not dataset_entry:
            continue
        table = dataset_entry["tables"][row.duckdb_table]
        table["sheet_name"] = row.sheet_name
        table["duckdb_table"] = row.duckdb_table
        table["columns"].append(
            {
                "column_name": row.column_name,
                "duckdb_column": row.duckdb_column,
                "data_type": row.data_type,
                "semantic_type": row.semantic_type,
                "sample_values": list(row.sample_values or [])[:4],
                "metrics": row.metrics_json or {},
            }
        )

    result: list[dict[str, Any]] = []
    for dataset in datasets[:4]:
        entry = grouped[dataset.id]
        result.append(
            {
                "dataset_id": entry["dataset_id"],
                "filename": entry["filename"],
                "db_alias": entry["db_alias"],
                "duckdb_path": entry["duckdb_path"],
                "tables": [
                    {
                        "sheet_name": table["sheet_name"],
                        "duckdb_table": table["duckdb_table"],
                        "columns": table["columns"],
                    }
                    for table in entry["tables"].values()
                ],
            }
        )
    return result


def sample_rows_for_schema(schema_context: list[dict[str, Any]], limit_per_table: int = 3) -> list[dict[str, Any]]:
    previews: list[dict[str, Any]] = []
    for dataset in schema_context:
        duckdb_path = Path(dataset["duckdb_path"])
        if not duckdb_path.exists():
            continue
        con = duckdb.connect(str(duckdb_path), read_only=True)
        try:
            for table in dataset["tables"][:3]:
                column_names = [column["duckdb_column"] for column in table["columns"][:8]]
                if not column_names:
                    continue
                projection = ", ".join(f'"{column}"' for column in column_names)
                rows = con.execute(
                    f'SELECT {projection} FROM "{table["duckdb_table"]}" LIMIT {limit_per_table}'
                ).fetchdf()
                previews.append(
                    {
                        "dataset_id": dataset["dataset_id"],
                        "filename": dataset["filename"],
                        "db_alias": dataset["db_alias"],
                        "duckdb_table": table["duckdb_table"],
                        "sheet_name": table["sheet_name"],
                        "sample_rows": rows.to_dict(orient="records"),
                    }
                )
        finally:
            con.close()
    return previews


def retrieve_context(db: Session, datasets: list[Dataset], question: str) -> dict[str, Any]:
    schema_context = build_schema_context(db, datasets)
    payload: list[dict[str, Any]] = []
    for dataset in datasets[:4]:
        try:
            retriever = PostgresDatasetRetriever(db, dataset.id, similarity_top_k=4)
            nodes = retriever.retrieve(question)
            chunks = [node.node.get_content() for node in nodes[:3]]
        except Exception:
            chunks = []
        payload.append(
            {
                "dataset_id": dataset.id,
                "filename": dataset.filename,
                "chunks": chunks,
            }
        )
    return {
        "retrieved_chunks": payload,
        "schema_context": schema_context,
        "sample_rows": sample_rows_for_schema(schema_context),
    }


def sql_capable_question(question: str) -> bool:
    normalized = normalize_text(question)
    signals = [
        "por año",
        "por ano",
        "por mes",
        "por categoria",
        "separado por",
        "desglos",
        "top ",
        "ranking",
        "total",
        "suma",
        "ingres",
        "monto",
        "venta",
        "compar",
    ]
    return any(signal in normalized for signal in signals)


def generate_sql_plan(question: str, schema_context: list[dict[str, Any]], retrieved_context: dict[str, Any]) -> dict[str, Any]:
    return llm_json(
        """
Genera solo JSON válido.
Tu tarea es decidir si una pregunta debe resolverse ejecutando SQL sobre DuckDB y, si aplica, escribir una consulta segura.

Devuelve exactamente:
{
  "should_query": boolean,
  "sql": string | null,
  "title": string | null,
  "reason": string
}

Reglas:
- Solo genera consultas SELECT o WITH ... SELECT para DuckDB.
- No uses INSERT, UPDATE, DELETE, DROP, ALTER, COPY, ATTACH, DETACH, PRAGMA ni CREATE.
- Usa los alias de base provistos (ds0, ds1, etc.) y referencia tablas como ds0.nombre_tabla.
- Si la pregunta requiere agregación, cruces, ranking, series temporales o derivar año/mes desde fecha, responde con should_query=true.
- Si existe columna de fecha pero no existe columna año, deriva el año desde la fecha.
- Si el tipo de fecha no es claro, apóyate en sample_values. Puedes usar TRY_CAST(col AS DATE). Si el valor parece texto, puedes usar TRY_STRPTIME.
- Prefiere nombres de columnas DuckDB exactos.
- Limita el resultado a un tamaño razonable.
- No inventes tablas ni columnas.
""",
        f"""Pregunta:
{question}

Esquema disponible:
{json.dumps(schema_context, ensure_ascii=False, default=str)}

Contexto recuperado:
{json.dumps(retrieved_context.get("retrieved_chunks", []), ensure_ascii=False, default=str)}

Muestras:
{json.dumps(retrieved_context.get("sample_rows", []), ensure_ascii=False, default=str)}
""",
    )


def is_safe_select_sql(sql: str) -> bool:
    cleaned = (sql or "").strip().rstrip(";").strip()
    lowered = cleaned.lower()
    if not cleaned or not (lowered.startswith("select") or lowered.startswith("with")):
        return False
    forbidden = [" insert ", " update ", " delete ", " drop ", " alter ", " create ", " attach ", " detach ", " copy ", " pragma "]
    wrapped = f" {lowered} "
    return not any(token in wrapped for token in forbidden)


def execute_duckdb_sql(schema_context: list[dict[str, Any]], sql: str) -> dict[str, Any]:
    if not is_safe_select_sql(sql):
        raise ValueError("La consulta generada no es segura o no es un SELECT.")
    con = duckdb.connect()
    try:
        for dataset in schema_context:
            duckdb_path = Path(dataset["duckdb_path"])
            if not duckdb_path.exists():
                continue
            alias = dataset["db_alias"]
            safe_path = str(duckdb_path).replace("'", "''")
            con.execute(f"ATTACH '{safe_path}' AS {alias}")
        wrapped_sql = f"SELECT * FROM ({sql.rstrip(';')}) AS q LIMIT 200"
        df = con.execute(wrapped_sql).fetchdf()
        return {
            "sql": sql.rstrip(";"),
            "columns": list(df.columns),
            "rows": df.to_dict(orient="records"),
            "row_count": int(len(df)),
        }
    finally:
        con.close()


def maybe_run_sql(question: str, schema_context: list[dict[str, Any]], retrieved_context: dict[str, Any]) -> dict[str, Any] | None:
    if not schema_context or not sql_capable_question(question):
        return None
    try:
        plan = generate_sql_plan(question, schema_context, retrieved_context)
    except Exception:
        return None
    if not plan.get("should_query") or not plan.get("sql"):
        return None
    try:
        result = execute_duckdb_sql(schema_context, str(plan["sql"]))
        result["title"] = plan.get("title") or "Resultado"
        result["reason"] = plan.get("reason") or ""
        return result
    except Exception as exc:
        return {
            "sql": str(plan.get("sql") or ""),
            "columns": [],
            "rows": [],
            "row_count": 0,
            "title": plan.get("title") or "Resultado",
            "reason": f"{plan.get('reason') or ''} Error al ejecutar SQL: {exc}".strip(),
            "error": str(exc),
        }


def fallback_artifacts_from_sql_result(sql_result: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not sql_result or not sql_result.get("rows"):
        return []
    artifacts: list[dict[str, Any]] = [
        {
            "type": "table",
            "title": sql_result.get("title") or "Detalle",
            "html": None,
            "chart": None,
            "rows": list(sql_result.get("rows") or [])[:20],
        }
    ]
    columns = list(sql_result.get("columns") or [])
    rows = list(sql_result.get("rows") or [])
    if len(columns) >= 2 and len(rows) <= 12:
        numeric_candidates = [
            column for column in columns
            if all(isinstance(row.get(column), (int, float)) or row.get(column) is None for row in rows)
        ]
        if numeric_candidates:
            value_column = numeric_candidates[-1]
            label_columns = [column for column in columns if column != value_column]
            label_column = label_columns[0] if label_columns else None
            if label_column:
                artifacts.append(
                    {
                        "type": "chart",
                        "title": sql_result.get("title") or "Visualización",
                        "html": None,
                        "chart": {
                            "title": sql_result.get("title") or "Visualización",
                            "kind": "bar",
                            "data": [
                                {"label": " | ".join(str(row.get(column) or "") for column in label_columns).strip(" |"), "value": row.get(value_column)}
                                for row in rows[:12]
                            ],
                        },
                        "rows": None,
                    }
                )
    return artifacts


def fallback_response(primary: Dataset, question: str) -> dict[str, Any]:
    return {
        "response_text": f"No pude estructurar una respuesta completa para: {question}",
        "response_html": None,
        "artifacts": [],
        "focus_files": [primary.filename],
    }


def answer_question(db: Session, dataset_id: str, question: str) -> dict:
    datasets = ready_datasets(db)
    if not datasets:
        return {
            "type": "status",
            "response_text": "Todavía no hay archivos listos. Adjunta un CSV o Excel y espera el análisis.",
            "response_html": None,
            "artifacts": [],
            "focus_files": [],
        }

    primary = pick_primary_dataset(datasets, dataset_id)
    context_payload = [dataset_context(item) for item in datasets[:4]]
    retrieved_payload = retrieve_context(db, datasets, question)
    sql_result = maybe_run_sql(question, retrieved_payload.get("schema_context", []), retrieved_payload)

    try:
        response = llm_json(
            """
Responde solo con JSON válido en español.
Tu objetivo es responder preguntas generales o específicas usando el contexto de uno o varios archivos ya analizados.
Cuando exista un resultado SQL ejecutado, úsalo como fuente principal de verdad. No reutilices el formato del resumen por archivo. Si no hay precisión suficiente, dilo con claridad y responde con el mejor análisis posible.

Devuelve exactamente esta estructura:
{
  "response_text": string | null,
  "response_html": string | null,
  "artifacts": [
    {
      "type": "html" | "chart" | "table",
      "title": string | null,
      "html": string | null,
      "chart": {
        "title": string,
        "kind": "bar"|"line"|null,
        "data": [{"label": string, "value": number}]
      } | null,
      "rows": [object] | null
    }
  ],
  "focus_files": [string]
}

Reglas:
- "response_text" puede ser null si el HTML ya responde completamente.
- "response_html" es opcional. Úsalo solo cuando agregue valor real: tablas, cards, comparativos, listas bien compuestas o visualizaciones compactas con HTML simple.
- "artifacts" es opcional. Úsalo solo cuando realmente aporte valor a la respuesta.
- Si la pregunta se responde bien con texto, devuelve solo "response_text" y deja "response_html" en null y "artifacts" vacío.
- No devuelvas artefactos por defecto.
- Para visualizaciones, usa artifact.type = "chart".
- Para tablas de datos, usa artifact.type = "table".
- Para bloques compuestos adicionales, usa artifact.type = "html".
- Clasifica la intención de la pregunta antes de responder:
  1. Pregunta general, conceptual, interpretativa o de recomendación: prioriza solo "response_text".
  2. Pregunta comparativa entre archivos, periodos, categorías, países o segmentos: puedes usar "response_html" para presentar comparativos, listas, cards o secciones compactas.
  3. Pregunta con datos numéricos claros, rankings, aperturas, top N, totales o cruces cuantitativos: puedes usar "artifacts" de tipo "table" o "chart" si realmente ayudan.
- Si la pregunta es general, evita "response_html" y evita "artifacts" salvo que sean indispensables.
- Si la comparación puede entenderse bien en texto corto, mantén solo "response_text". Usa "response_html" solo cuando ordene mejor la respuesta.
- Usa gráficos de barras solo cuando el número de categorías sea pequeño y legible. Evita gráficos si no agregan claridad.
- No uses gráficos para rellenar la respuesta. Si una tabla o texto basta, no generes chart.
- No dupliques la misma información entre "response_text", "response_html" y "artifacts".
- Si generas "response_html", úsalo como complemento o como bloque principal, pero con contenido realmente distinto del texto.
- El HTML debe ser un fragmento simple y autónomo, sin <script>, sin <style>, sin markdown, sin html/body. Usa inline styles sobrios si hace falta.
- No inventes cifras ni columnas.
- Si la pregunta se enfoca en un archivo concreto, usa el contexto más cercano.
- Si recibes un resultado SQL válido, responde prioritariamente con ese resultado y no digas que falta información si la tabla ya la resuelve.
""",
            f"""
Pregunta:
{question}

Archivo principal:
{primary.filename}

Datasets disponibles:
{json.dumps(context_payload, ensure_ascii=False, default=str)}

Contexto recuperado:
{json.dumps(retrieved_payload.get("retrieved_chunks", []), ensure_ascii=False, default=str)}

Esquema y muestras:
{json.dumps({"schema_context": retrieved_payload.get("schema_context", []), "sample_rows": retrieved_payload.get("sample_rows", [])}, ensure_ascii=False, default=str)}

Resultado SQL ejecutado:
{json.dumps(sql_result, ensure_ascii=False, default=str)}
""",
        )
    except Exception:
        response = fallback_response(primary, question)

    response["response_text"] = (
        response.get("response_text")
        if isinstance(response.get("response_text"), str) and response.get("response_text", "").strip()
        else (primary.summary or "Respuesta no disponible.")
    )
    response["response_html"] = (
        response.get("response_html")
        if isinstance(response.get("response_html"), str) and response.get("response_html", "").strip()
        else None
    )
    response["focus_files"] = [str(item) for item in (response.get("focus_files") or [primary.filename])][:4]

    normalized_artifacts: list[dict[str, Any]] = []
    for artifact in list(response.get("artifacts") or [])[:6]:
        if not isinstance(artifact, dict):
            continue
        artifact_type = str(artifact.get("type") or "").strip().lower()
        if artifact_type not in {"html", "chart", "table"}:
            continue
        normalized: dict[str, Any] = {
            "type": artifact_type,
            "title": str(artifact.get("title")) if artifact.get("title") is not None else None,
            "html": None,
            "chart": None,
            "rows": None,
        }
        if artifact_type == "html":
            html = artifact.get("html")
            if isinstance(html, str) and html.strip():
                normalized["html"] = html
            else:
                continue
        elif artifact_type == "chart":
            chart = artifact.get("chart")
            if not isinstance(chart, dict):
                continue
            chart_kind = chart.get("kind")
            if chart_kind not in {"bar", "line"}:
                continue
            normalized["chart"] = {
                "title": str(chart.get("title") or normalized["title"] or "Visualización"),
                "kind": chart_kind,
                "data": list(chart.get("data") or [])[:12],
            }
        elif artifact_type == "table":
            rows = artifact.get("rows")
            if not isinstance(rows, list) or not rows:
                continue
            normalized["rows"] = rows[:12]
        normalized_artifacts.append(normalized)

    # Compatibilidad con respuestas antiguas si llegaran a existir.
    legacy_analysis = [str(item) for item in (response.get("analysis") or [])][:4]
    if legacy_analysis:
        normalized_artifacts.append(
            {
                "type": "html",
                "title": "Análisis",
                "html": "<div><h3 style='margin:0 0 10px;font-size:15px;color:#111027;'>Análisis</h3><ul style='margin:0;padding-left:18px;color:#3d394f;font-size:13px;line-height:1.5;'>"
                + "".join(f"<li>{item}</li>" for item in legacy_analysis)
                + "</ul></div>",
                "chart": None,
                "rows": None,
            }
        )
    legacy_chart = response.get("chart")
    if isinstance(legacy_chart, dict) and legacy_chart.get("kind") in {"bar", "line"}:
        normalized_artifacts.append(
            {
                "type": "chart",
                "title": str(legacy_chart.get("title") or "Visualización"),
                "html": None,
                "chart": {
                    "title": str(legacy_chart.get("title") or "Visualización"),
                    "kind": legacy_chart.get("kind"),
                    "data": list(legacy_chart.get("data") or [])[:12],
                },
                "rows": None,
            }
        )
    legacy_table = response.get("table")
    if isinstance(legacy_table, list) and legacy_table:
        normalized_artifacts.append(
            {
                "type": "table",
                "title": "Detalle",
                "html": None,
                "chart": None,
                "rows": legacy_table[:12],
            }
        )

    if not normalized_artifacts:
        normalized_artifacts = fallback_artifacts_from_sql_result(sql_result)

    response["artifacts"] = normalized_artifacts[:6]
    response["type"] = "answer"
    return response
