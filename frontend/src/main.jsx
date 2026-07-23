import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  BarChart3,
  Bot,
  FileText,
  Gauge,
  MessageSquarePlus,
  Paperclip,
  RotateCcw,
  Send,
  Sparkles,
} from 'lucide-react';
import './styles.css';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

const suggestions = [
  'Resume los archivos cargados',
  '¿Qué insights iniciales encuentras?',
  'Compara patrones entre los archivos',
];

function uploadDatasetRequest(file, description, onProgress, chatSessionId) {
  return new Promise((resolve, reject) => {
    const body = new FormData();
    body.append('file', file);
    body.append('description', description);
    if (chatSessionId) body.append('chat_session_id', chatSessionId);

    const request = new XMLHttpRequest();
    request.open('POST', `${API_BASE_URL}/datasets`);
    request.responseType = 'json';

    request.upload.onprogress = (event) => {
      if (!event.lengthComputable) return;
      onProgress?.(Math.round((event.loaded / event.total) * 100));
    };

    request.onload = () => {
      if (request.status >= 200 && request.status < 300) {
        resolve(request.response);
        return;
      }
      const message = request.response?.detail || request.statusText || `Error ${request.status}`;
      reject(new Error(message));
    };

    request.onerror = () => reject(new Error('No se pudo cargar el archivo.'));
    request.send(body);
  });
}

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Error ${response.status}`);
  }
  return response.json();
}

function prettyValue(value) {
  if (typeof value === 'number') {
    return value.toLocaleString('en-US', { maximumFractionDigits: 2 });
  }
  return String(value ?? '');
}

function formatMetric(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return String(value ?? '—');
  const abs = Math.abs(number);
  if (abs >= 10_000_000) return `${(number / 1_000_000).toFixed(2)}MM`;
  if (abs >= 1_000_000) return `${(number / 1_000_000).toFixed(2)}M`;
  if (abs >= 1_000) return `${(number / 1_000).toFixed(2)}k`;
  return number.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function formatExecutiveValue(value) {
  const raw = typeof value === 'string' ? value.replace(/[,$\s]/g, '') : value;
  const number = Number(raw);
  if (!Number.isFinite(number)) return String(value ?? '—');
  return formatMetric(number);
}

function normalizeRichText(value) {
  const rawText = String(value ?? '');
  const withoutMarkdown = rawText
    .replace(/\*\*(.*?)\*\*/g, '$1')
    .replace(/__(.*?)__/g, '$1')
    .replace(/`([^`]+)`/g, '$1');

  return withoutMarkdown.replace(
    /(?<![A-Za-z0-9])(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d{4,}(?:\.\d+)?)(?![A-Za-z0-9])/g,
    (match) => {
      const normalized = match.replace(/,/g, '');
      const number = Number(normalized);
      if (!Number.isFinite(number)) return match;
      if (Math.abs(number) < 1000 && !String(match).includes('.')) return match;
      return formatMetric(number);
    },
  );
}

function progressPercent(dataset) {
  if (!dataset?.progress_total) return 0;
  return Math.min(100, Math.round(((dataset.progress_current || 0) / dataset.progress_total) * 100));
}

function datasetStatusLabel(dataset) {
  if (dataset.status === 'ready') return 'Listo';
  if (dataset.status === 'processing') return 'Analizando';
  if (dataset.status === 'error') return 'Error';
  return 'Cargando';
}

function chatMetaLabel(session) {
  const datasets = session?.datasets || [];
  const readyCount = datasets.filter((item) => item.status === 'ready').length;
  if (!datasets.length) return 'Sin archivos';
  return `${datasets.length} archivo${datasets.length > 1 ? 's' : ''} · ${readyCount} listo${readyCount > 1 ? 's' : ''}`;
}

function Sidebar({ sessions, activeSessionId, onSelect, onCreate, onReset, resetting }) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brandIcon"><BarChart3 size={18} /></div>
        <div>
          <div className="brandTitle">Metric IA</div>
          <div className="brandSubtitle">Chat analítico multiarchivo</div>
        </div>
      </div>

      <div className="sidebarSection">
        <div className="sidebarHeaderRow">
          <div className="sidebarLabel">Chats</div>
          <button type="button" className="newChatBtn" onClick={onCreate}>
            <MessageSquarePlus size={14} />
            <span>Nuevo</span>
          </button>
        </div>
        {sessions.length === 0 ? (
          <div className="sidebarEmpty">
            Aún no hay chats. Crea uno nuevo o adjunta un CSV o Excel desde el área principal.
          </div>
        ) : (
          <div className="datasetList">
            {sessions.map((session) => (
              <button
                key={session.id}
                className={`datasetItem ${session.id === activeSessionId ? 'active' : ''}`}
                onClick={() => onSelect(session.id)}
                type="button"
              >
                <div className="datasetItemIcon"><FileText size={15} /></div>
                <div className="datasetItemBody">
                  <strong>{session.title || 'Nuevo chat'}</strong>
                  <span>{chatMetaLabel(session)}</span>
                  {session.last_message_preview ? <span>{session.last_message_preview}</span> : null}
                </div>
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="sidebarBottom">
        <div className="adminCard">
          <div className="avatar">AD</div>
          <div className="adminInfo">
            <div className="adminName">Admin Retail</div>
            <div className="adminMeta">Memoria activa de archivos y chat</div>
          </div>
          <button
            type="button"
            className="resetMemoryBtn"
            onClick={onReset}
            disabled={resetting}
            aria-label="Reiniciar memoria"
            title="Reiniciar memoria"
          >
            <RotateCcw size={15} />
          </button>
        </div>
      </div>
    </aside>
  );
}

function Topbar({ readyCount, title }) {
  return (
    <header className="topbar">
      <div>
        <h1><Sparkles size={19} /> Asistente analítico</h1>
        <p>
          {title
            ? `${title}${readyCount > 0 ? ` · ${readyCount} archivo${readyCount > 1 ? 's' : ''} listo${readyCount > 1 ? 's' : ''}` : ''}.`
            : readyCount > 0
            ? `Contexto activo con ${readyCount} archivo${readyCount > 1 ? 's' : ''} listo${readyCount > 1 ? 's' : ''}.`
            : 'Crea un chat, adjunta uno o más archivos CSV o Excel y conversa sobre ellos.'}
        </p>
      </div>
    </header>
  );
}

function AssistantHeader() {
  return (
    <div className="assistantHeader">
      <div className="assistantLogo"><Bot size={16} /></div>
      <strong>Metric IA</strong>
    </div>
  );
}

function sanitizeAssistantHtml(rawHtml) {
  if (!rawHtml || typeof rawHtml !== 'string') return '';
  const parser = new DOMParser();
  const doc = parser.parseFromString(rawHtml, 'text/html');
  const blockedTags = new Set(['script', 'style', 'iframe', 'object', 'embed', 'link', 'meta']);
  const walker = doc.createTreeWalker(doc.body, NodeFilter.SHOW_ELEMENT);
  const nodes = [];

  while (walker.nextNode()) nodes.push(walker.currentNode);

  nodes.forEach((node) => {
    const tag = node.tagName.toLowerCase();
    if (blockedTags.has(tag)) {
      node.remove();
      return;
    }

    [...node.attributes].forEach((attribute) => {
      const name = attribute.name.toLowerCase();
      const value = attribute.value || '';
      if (name.startsWith('on')) {
        node.removeAttribute(attribute.name);
        return;
      }
      if ((name === 'href' || name === 'src') && /^\s*javascript:/i.test(value)) {
        node.removeAttribute(attribute.name);
      }
    });
  });

  return doc.body.innerHTML;
}

function UserBubble({ text }) {
  return (
    <div className="userBubble">
      <strong>Tú</strong>
      <span>{text}</span>
    </div>
  );
}

function ChartBlock({ chart, compact = false }) {
  const values = chart?.data || [];
  if (!values.length) return null;
  const max = Math.max(...values.map((item) => Math.abs(Number(item.value) || 0)), 1);
  return (
    <div className={`sparkCard ${compact ? 'compactChartCard' : ''}`}>
      <h3>{chart.title || 'Visualización'}</h3>
      <div className="chartKindTag">{chart?.kind === 'pie' ? 'bar' : (chart.kind || 'chart')}</div>
      <div className={`barViz ${compact ? 'compactBarViz' : ''}`}>
        {values.map((item, index) => {
          const numericValue = Number(item.value) || 0;
          const width = Math.max(10, Math.abs(numericValue) / max * 100);
          return (
            <div className={`barRow ${compact ? 'compactBarRow' : ''}`} key={`${item.label}-${index}`}>
              <div className={`barTrack barTrackLabeled ${compact ? 'compactBarTrack' : ''}`}>
                <div className="barFill" style={{ width: `${width}%` }} />
                <span className="barOverlayLabel" title={prettyValue(item.label)}>{prettyValue(item.label)}</span>
              </div>
              <strong>{formatMetric(numericValue)}</strong>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function detectSortableValue(value) {
  if (value === null || value === undefined || value === '') return { type: 'empty', value: null };
  if (typeof value === 'number') return { type: 'number', value };

  const text = String(value).trim();
  const numericCandidate = text.replace(/[, ]/g, '');
  const numericValue = Number(numericCandidate);
  if (Number.isFinite(numericValue) && /^-?\d+(\.\d+)?$/.test(numericCandidate)) {
    return { type: 'number', value: numericValue };
  }

  const dateValue = Date.parse(text);
  if (Number.isFinite(dateValue)) {
    return { type: 'date', value: dateValue };
  }

  return { type: 'text', value: text.toLowerCase() };
}

function ResultTable({ rows }) {
  if (!rows?.length) return null;
  const baseColumns = Object.keys(rows[0]);
  const [sortConfig, setSortConfig] = useState({ column: null, direction: 'asc' });
  const [columnOrder, setColumnOrder] = useState(baseColumns);

  useEffect(() => {
    setColumnOrder((current) => {
      const currentSet = new Set(current);
      const next = [...current.filter((column) => baseColumns.includes(column))];
      baseColumns.forEach((column) => {
        if (!currentSet.has(column)) next.push(column);
      });
      return next;
    });
  }, [rows]);

  const columns = useMemo(() => (
    columnOrder.filter((column) => baseColumns.includes(column))
  ), [columnOrder, baseColumns]);

  const sortedRows = useMemo(() => {
    if (!sortConfig.column) return rows;
    const sorted = [...rows].sort((left, right) => {
      const leftValue = detectSortableValue(left?.[sortConfig.column]);
      const rightValue = detectSortableValue(right?.[sortConfig.column]);

      if (leftValue.type === 'empty' && rightValue.type === 'empty') return 0;
      if (leftValue.type === 'empty') return 1;
      if (rightValue.type === 'empty') return -1;

      if (leftValue.type === rightValue.type) {
        if (leftValue.value < rightValue.value) return -1;
        if (leftValue.value > rightValue.value) return 1;
        return 0;
      }

      const leftText = String(left?.[sortConfig.column] ?? '').toLowerCase();
      const rightText = String(right?.[sortConfig.column] ?? '').toLowerCase();
      if (leftText < rightText) return -1;
      if (leftText > rightText) return 1;
      return 0;
    });
    return sortConfig.direction === 'desc' ? sorted.reverse() : sorted;
  }, [rows, sortConfig]);

  function toggleSort(column) {
    setSortConfig((current) => (
      current.column === column
        ? { column, direction: current.direction === 'asc' ? 'desc' : 'asc' }
        : { column, direction: 'asc' }
    ));
  }

  function moveColumn(column, direction) {
    setColumnOrder((current) => {
      const index = current.indexOf(column);
      if (index === -1) return current;
      const targetIndex = direction === 'left' ? index - 1 : index + 1;
      if (targetIndex < 0 || targetIndex >= current.length) return current;
      const next = [...current];
      [next[index], next[targetIndex]] = [next[targetIndex], next[index]];
      return next;
    });
  }

  return (
    <div className="tableCard">
      <div className="tableTitle">Detalle</div>
      <table>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}>
                <div className="tableHeaderCell">
                  <button
                    type="button"
                    className={`tableSortBtn ${sortConfig.column === column ? 'active' : ''}`}
                    onClick={() => toggleSort(column)}
                  >
                    <span>{column}</span>
                    <span className="tableSortIndicator">
                      {sortConfig.column === column ? (sortConfig.direction === 'asc' ? '↓' : '↑') : '↕'}
                    </span>
                  </button>
                  <div className="tableMoveActions">
                    <button
                      type="button"
                      className="tableMoveBtn"
                      onClick={() => moveColumn(column, 'left')}
                      disabled={columns[0] === column}
                      aria-label={`Mover ${column} a la izquierda`}
                      title="Mover a la izquierda"
                    >
                      ←
                    </button>
                    <button
                      type="button"
                      className="tableMoveBtn"
                      onClick={() => moveColumn(column, 'right')}
                      disabled={columns[columns.length - 1] === column}
                      aria-label={`Mover ${column} a la derecha`}
                      title="Mover a la derecha"
                    >
                      →
                    </button>
                  </div>
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sortedRows.slice(0, 50).map((row, index) => (
            <tr key={index}>
              {columns.map((column) => <td key={column}>{prettyValue(row[column])}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function HtmlBlock({ html }) {
  const safeHtml = sanitizeAssistantHtml(html);
  if (!safeHtml) return null;
  return <div className="assistantHtmlBlock" dangerouslySetInnerHTML={{ __html: safeHtml }} />;
}

function ArtifactBlock({ artifact, index }) {
  if (!artifact || !artifact.type) return null;

  if (artifact.type === 'html') {
    return <HtmlBlock key={`artifact-html-${index}`} html={artifact.html} />;
  }

  if (artifact.type === 'chart') {
    return <ChartBlock key={`artifact-chart-${index}`} chart={artifact.chart} />;
  }

  if (artifact.type === 'table') {
    return <ResultTable key={`artifact-table-${index}`} rows={artifact.rows} />;
  }

  return null;
}

function AssistantMessage({ payload }) {
  const responseText = payload?.response_text ?? payload?.response ?? '';
  const responseHtml = payload?.response_html ?? payload?.html ?? null;
  const artifacts = Array.isArray(payload?.artifacts) ? payload.artifacts : [];
  return (
    <div className="assistantMessage">
      <AssistantHeader />
      {responseText && <p className="answerText">{responseText}</p>}
      <HtmlBlock html={responseHtml} />
      {artifacts.map((artifact, index) => <ArtifactBlock key={`artifact-${index}`} artifact={artifact} index={index} />)}
    </div>
  );
}

function EmptyChat({ hasFiles }) {
  return (
    <div className="emptyChatHero">
      <div className="emptyChatCenter">
        <div className="emptyChatBadge"><Sparkles size={18} /></div>
        <h2>Empieza por preguntar o adjuntar archivos</h2>
        <p>
          {hasFiles
            ? 'Puedes hacer preguntas generales, comparativas o enfocadas en cualquiera de los archivos cargados.'
            : 'Adjunta CSV o Excel con el clip o arrástralos al área del chat. El análisis se ejecuta en background.'}
        </p>
        <div className="chips">
          {suggestions.map((item) => <span key={item}>{item}</span>)}
        </div>
      </div>
    </div>
  );
}

function UploadQueue({ uploads }) {
  if (!uploads.length) return null;
  return (
    <div className="uploadQueue">
      {uploads.map((item) => (
        <div key={item.name} className="uploadQueueItem">
          <FileText size={14} />
          <span>{item.name}</span>
          {typeof item.progress === 'number' && <em>{item.progress}%</em>}
        </div>
      ))}
    </div>
  );
}

function FileProgress({ title, detail, progress, tone = 'default' }) {
  return (
    <div className={`progressCard ${tone}`}>
      <div className="progressCardHeader">
        <strong>{title}</strong>
        <span>{progress}%</span>
      </div>
      <p>{detail}</p>
      <div className="progressTrack"><div style={{ width: `${progress}%` }} /></div>
    </div>
  );
}

function ChatComposer({ disabled, uploads, onPickFiles, onSubmit }) {
  const [question, setQuestion] = useState('');
  const fileInputRef = useRef(null);

  function submit(event) {
    event.preventDefault();
    if (!question.trim() || disabled) return;
    onSubmit(question);
    setQuestion('');
  }

  return (
    <form className="chatComposer" onSubmit={submit}>
      <UploadQueue uploads={uploads} />
      <div className="composerRow">
        <button
          className="iconBtn"
          type="button"
          onClick={() => fileInputRef.current?.click()}
          aria-label="Adjuntar archivo"
        >
          <Paperclip size={18} />
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,.xlsx,text/csv,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
          multiple
          hidden
          onChange={(event) => {
            onPickFiles(event.target.files);
            event.target.value = '';
          }}
        />
        <input
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="Pregunta por el contenido, compáralo entre archivos o pide insights específicos."
          disabled={disabled}
        />
        <button className="sendBtn" disabled={disabled || !question.trim()} type="submit">
          <Send size={17} />
        </button>
      </div>
    </form>
  );
}

function RightPanel({ datasets, activeDatasetId, onSelect }) {
  const activeDataset = datasets.find((item) => item.id === activeDatasetId) || datasets[0] || null;
  const analysis = activeDataset?.profile?.analysis_json;

  return (
    <aside className="rightPanel">
      <div className="rightPanelHeader">
        <h2>Resumen por archivo</h2>
        {datasets.length > 1 && (
          <div className="tabBar">
            {datasets.map((dataset) => (
              <button
                key={dataset.id}
                type="button"
                className={`tabBtn ${dataset.id === activeDataset?.id ? 'active' : ''}`}
                onClick={() => onSelect(dataset.id)}
              >
                {dataset.filename}
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="rightPanelBody">
        {!activeDataset && (
          <div className="emptyInsightsState">
            <div className="emptyInsightsIcon"><Gauge size={18} /></div>
            <h3>No hay análisis aún</h3>
            <p>El panel derecho se completará cuando subas uno o más archivos y termine su análisis.</p>
          </div>
        )}

        {activeDataset?.status === 'error' && (
          <div className="errorBox">
            {activeDataset.error_message || 'No se pudo analizar el archivo.'}
          </div>
        )}

        {activeDataset?.status === 'ready' && analysis && (
          <>
            <div className="analysisSummaryCard">
              <span className="analysisSummaryType">{analysis.input_kind?.toUpperCase() || 'ARCHIVO'}</span>
              <h3>{analysis.title || activeDataset.filename}</h3>
              <p>{normalizeRichText(analysis.summary || activeDataset.summary)}</p>
            </div>

            {(analysis.metrics || []).length > 0 && (
              <div className="miniGrid">
                {analysis.metrics.slice(0, 4).map((metric, index) => (
                  <div key={`${metric.label}-${index}`}>
                    <span>{metric.label}</span>
                    <strong>{formatExecutiveValue(metric.value)}</strong>
                    {metric.context ? <em>{metric.context}</em> : null}
                  </div>
                ))}
              </div>
            )}

            {(analysis.insights || []).length > 0 && (
              <div className="insightCard">
                <h3><Sparkles size={15} /> Insights clave</h3>
                <ul className="insightList">
                  {analysis.insights.slice(0, 3).map((item, index) => (
                    <li key={`${item}-${index}`}>{normalizeRichText(item)}</li>
                  ))}
                </ul>
              </div>
            )}

            {(analysis.charts || []).slice(0, 1).map((chart, index) => (
              <ChartBlock chart={chart} compact key={`${chart.title}-${index}`} />
            ))}
          </>
        )}

        {activeDataset?.status === 'ready' && !analysis && (
          <div className="insightPlaceholderCard">
            <span className="insightPlaceholderLabel">Resumen no disponible</span>
            <strong>{activeDataset.filename}</strong>
            <p>Este archivo está listo para chat, pero no tiene análisis estructurado para el panel lateral. Vuelve a cargarlo para regenerarlo.</p>
          </div>
        )}
      </div>
    </aside>
  );
}

function App() {
  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [activeDatasetId, setActiveDatasetId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [pendingJobId, setPendingJobId] = useState(null);
  const [pendingDetail, setPendingDetail] = useState('');
  const [error, setError] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const [uploadQueue, setUploadQueue] = useState([]);
  const [resetting, setResetting] = useState(false);
  const pollRef = useRef(null);

  const activeSession = useMemo(
    () => sessions.find((item) => item.id === activeSessionId) || sessions[0] || null,
    [sessions, activeSessionId],
  );
  const datasets = activeSession?.datasets || [];
  const readyDatasets = useMemo(() => datasets.filter((item) => item.status === 'ready'), [datasets]);
  const processingDatasets = useMemo(
    () => datasets.filter((item) => item.status === 'processing' || item.status === 'uploaded'),
    [datasets],
  );

  useEffect(() => {
    let timer;
    async function loadSessions() {
      try {
        const items = await api('/chat/sessions');
        setSessions(items);
        setActiveSessionId((current) => {
          if (current && items.some((item) => item.id === current)) return current;
          return items[0]?.id || null;
        });
      } catch (err) {
        setError(err.message);
      }
    }
    loadSessions();
    timer = setInterval(loadSessions, 2500);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!activeSession) {
      setMessages([]);
      setActiveDatasetId(null);
      return;
    }
    setMessages((activeSession.messages || []).map((item) => ({ role: item.role, content: item.content })));
    setActiveDatasetId((current) => {
      if (current && datasets.some((item) => item.id === current)) return current;
      return datasets[0]?.id || null;
    });
  }, [activeSession, datasets]);

  useEffect(() => {
    if (!pendingJobId) return undefined;
    pollRef.current = setInterval(async () => {
      try {
        const job = await api(`/chat/jobs/${pendingJobId}`);
        setPendingDetail(job.detail || 'Analizando.');
        if (job.status === 'complete') {
          await refreshSessions(job.chat_session_id || activeSessionId);
          setPendingJobId(null);
          setPendingDetail('');
        }
        if (job.status === 'error') {
          setMessages((items) => [
            ...items,
            { role: 'assistant', content: { response_text: job.error_message || 'No pude completar la respuesta.', response_html: null, artifacts: [] } },
          ]);
          setPendingJobId(null);
          setPendingDetail('');
        }
      } catch (err) {
        setError(err.message);
      }
    }, 1400);
    return () => clearInterval(pollRef.current);
  }, [pendingJobId]);

  async function refreshSessions(preferredSessionId = null) {
    const items = await api('/chat/sessions');
    setSessions(items);
    setActiveSessionId((current) => {
      if (preferredSessionId && items.some((item) => item.id === preferredSessionId)) return preferredSessionId;
      if (current && items.some((item) => item.id === current)) return current;
      return items[0]?.id || null;
    });
  }

  async function createSession(title = 'Nuevo chat') {
    const session = await api('/chat/sessions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title }),
    });
    setSessions((items) => [session, ...items]);
    setActiveSessionId(session.id);
    return session.id;
  }

  async function ensureActiveSession(title = 'Nuevo chat') {
    if (activeSession) return activeSession.id;
    return createSession(title);
  }

  async function uploadFiles(fileList) {
    const files = Array.from(fileList || []).filter((file) => /\.(csv|xlsx)$/i.test(file.name));
    if (!files.length) {
      setError('Solo se permiten archivos CSV o Excel (.xlsx).');
      return;
    }
    setError('');
    setUploadQueue(files.map((file) => ({ name: file.name, progress: 0 })));
    try {
      const sessionId = await ensureActiveSession('Nuevo chat');
      for (let index = 0; index < files.length; index += 1) {
        const file = files[index];
        const result = await uploadDatasetRequest(
          file,
          `Archivo ${file.name} cargado el 22 de julio de 2026 para análisis conversacional.`,
          (progress) => {
            setUploadQueue((items) => items.map((item, itemIndex) => (
              itemIndex === index ? { ...item, progress } : item
            )));
          },
          sessionId,
        );
        setActiveSessionId(result.chat_session_id || sessionId);
      }
      await refreshSessions(sessionId);
    } catch (err) {
      setError(err.message);
    } finally {
      setUploadQueue([]);
    }
  }

  async function ask(question) {
    const cleaned = question.trim();
    if (!cleaned) return;

    const sessionId = await ensureActiveSession(cleaned);
    setMessages((items) => [...items, { role: 'user', content: cleaned }]);
    setError('');

    if (!readyDatasets.length) {
      setMessages((items) => [
        ...items,
        {
          role: 'assistant',
          content: {
            response_text: 'Todavía no hay archivos listos. Adjunta un CSV o Excel con el clip o arrástralo al chat.',
            response_html: null,
            artifacts: [],
          },
        },
      ]);
      return;
    }

    try {
      const datasetId = activeDatasetId || readyDatasets[0].id;
      const job = await api('/chat/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ chat_session_id: sessionId, dataset_id: datasetId, question: cleaned }),
      });
      await refreshSessions(sessionId);
      setPendingJobId(job.job_id);
      setPendingDetail(job.detail || 'Analizando.');
    } catch (err) {
      setError(err.message);
    }
  }

  function onDrop(event) {
    event.preventDefault();
    setIsDragging(false);
    if (event.dataTransfer?.files?.length) uploadFiles(event.dataTransfer.files);
  }

  async function resetMemory() {
    setResetting(true);
    setError('');
    try {
      await api('/admin/reset-memory', { method: 'POST' });
      setSessions([]);
      setActiveSessionId(null);
      setActiveDatasetId(null);
      setMessages([]);
      setPendingJobId(null);
      setPendingDetail('');
      setUploadQueue([]);
    } catch (err) {
      setError(err.message);
    } finally {
      setResetting(false);
    }
  }

  return (
    <div className="appShell">
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSession?.id}
        onSelect={setActiveSessionId}
        onCreate={() => createSession('Nuevo chat')}
        onReset={resetMemory}
        resetting={resetting}
      />
      <main className="workspace">
        <Topbar readyCount={readyDatasets.length} title={activeSession?.title} />
        <div className="contentGrid">
          <section
            className={`chatPanel ${isDragging ? 'dragging' : ''}`}
            onDragOver={(event) => {
              event.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={(event) => {
              if (event.currentTarget.contains(event.relatedTarget)) return;
              setIsDragging(false);
            }}
            onDrop={onDrop}
          >
            {isDragging && (
              <div className="dropOverlay">
                <div className="dropCard">
                  <Paperclip size={20} />
                  <strong>Suelta los archivos aquí</strong>
                  <span>Solo CSV o Excel (.xlsx)</span>
                </div>
              </div>
            )}

            <div className="chatBody">
              {error && <div className="errorBox">{error}</div>}

              {uploadQueue.map((item) => (
                <FileProgress
                  key={`upload-${item.name}`}
                  title={`Cargando ${item.name}`}
                  detail="Subiendo archivo al backend."
                  progress={Math.max(4, item.progress || 0)}
                  tone="upload"
                />
              ))}

              {processingDatasets.map((dataset) => (
                <FileProgress
                  key={`processing-${dataset.id}`}
                  title={`Analizando ${dataset.filename}`}
                  detail={dataset.progress_detail || 'Procesando contenido del archivo.'}
                  progress={Math.max(8, progressPercent(dataset))}
                  tone="analysis"
                />
              ))}

              {messages.length === 0 && <EmptyChat hasFiles={datasets.length > 0} />}

              {messages.map((item, index) => (
                item.role === 'user'
                  ? <UserBubble key={index} text={item.content} />
                  : <AssistantMessage key={index} payload={item.content} />
              ))}

              {pendingJobId && (
                <div className="pendingBox">
                  <AssistantHeader />
                  <p>{pendingDetail || 'Analizando.'}</p>
                </div>
              )}
            </div>

            <ChatComposer
              disabled={Boolean(pendingJobId)}
              uploads={uploadQueue}
              onPickFiles={uploadFiles}
              onSubmit={ask}
            />
          </section>

          <RightPanel datasets={datasets} activeDatasetId={activeDatasetId} onSelect={setActiveDatasetId} />
        </div>
      </main>
    </div>
  );
}

createRoot(document.getElementById('root')).render(<App />);
