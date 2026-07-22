# DataAI4 MVP

MVP controlado para cargar un Excel, procesarlo de forma asincrona y conversar con los datos desde una interfaz web propia.

## Stack

- Frontend: React + Vite.
- Backend: FastAPI.
- Base de datos: PostgreSQL + pgvector.
- Procesamiento analitico: DuckDB.
- Excel: `openpyxl` + `pandas`.
- Recuperacion semantica: LlamaIndex + embeddings OpenAI.
- LLM: OpenAI, por defecto `gpt-5`.

## Ejecutar

```bash
cp .env.example .env
```

Configura `OPENAI_API_KEY` en `.env` y luego:

```bash
docker compose up --build
```

Servicios:

- Frontend: http://localhost:5173
- Backend: http://localhost:8000
- Healthcheck: http://localhost:8000/health

## Flujo

1. El usuario carga un `.xlsx` y una descripcion corta.
2. El backend guarda el archivo y lanza procesamiento asincrono.
3. Se leen todas las hojas del Excel.
4. Se crea una base DuckDB por dataset.
5. Se genera catalogo semantico en PostgreSQL.
6. Se crean embeddings en PostgreSQL con pgvector.
7. El chat responde con texto, tabla, grafico o aclaracion previa segun convenga.

El frontend esta construido como componente propio en React, no con Streamlit.
