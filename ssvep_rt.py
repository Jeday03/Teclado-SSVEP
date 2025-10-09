"""

Modos:
- python ssvep_rt.py                    # Modo dataset padrão
- python ssvep_rt.py PALAVRA            # Modo simulação com palavra
- python ssvep_rt.py brainflow          # Modo EEG real
- python ssvep_rt.py brainflow --serial COM4 --board-id 2
- python ssvep_rt.py offline            # Modo offline
- python ssvep_rt.py record PALAVRA     # Gravar sinais sintéticos
- python ssvep_rt.py play ARQUIVO.pkl   # Reproduzir gravação
- python ssvep_rt.py list               # Listar gravações disponíveis
- python ssvep_rt.py offline --subjects 19-30 --depths high,low 
    --targets-window 16 --time-windows 500,1000,1500,2000 --alg cca

Configurações BrainFlow:
- --serial PORT: Porta serial (COM3, COM4, /dev/ttyUSB0, etc.)
- --board-id ID: Tipo de placa (-1=Synthetic, 0=Cyton, 2=Cyton+Daisy, 1=Ganglion)
- --other-params "key1=value1,key2=value2": Parâmetros extras do BrainFlow


Exemplos completos:
- EEG Cyton no Linux: python ssvep_rt.py brainflow --serial /dev/ttyUSB0 --board-id 0
- EEG Cyton Daisy: python ssvep_rt.py brainflow --serial COM4 --board-id 2
- EEG Ganglion: python ssvep_rt.py brainflow --board-id 1
- Simulador: python ssvep_rt.py brainflow --board-id -1
- Análise offline completa: python ssvep_rt.py offline --subjects 19-25 --depths high,low --targets-window 16 --time-windows 500,1000,1500 --alg cca --save-dir ./resultados
- Gravar palavra: python ssvep_rt.py record HELLO
- Reproduzir: python ssvep_rt.py play word_HELLO_20241205_143022.pkl
"""

import sys
import time
import os
import json
import numpy as np
import asyncio
import sklearn.cross_decomposition as skcd
from aiohttp import web
import pickle
from datetime import datetime

# --- datasets ---
HAVE_DATASET = True
try:
    sys.path.append("bciflow/")
    from bciflow.datasets import mengu
except Exception:
    HAVE_DATASET = False
    mengu = None

# === EEG ===
try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    HAVE_BRAINFLOW = True
except Exception:
    HAVE_BRAINFLOW = False

# ========= CONFIGURAÇÕES DE GRAVAÇÃO =========
RECORDINGS_DIR = "ssvep_recordings"
os.makedirs(RECORDINGS_DIR, exist_ok=True)

# ========= config =========
HOST = "127.0.0.1"
PORT = 8081
PORT_EVENTS = 8082
CHUNK_MS = 100
FALLBACK_SFREQ = 1000
SUBJECT = 19
DEPTH = "high"
CHANNELS_SELECTED = ["PZ","PO3","PO4","PO5","PO6","POZ","O1","O2","OZ"]
RAW_TARGET_FS = 512
ALG = "cca"
NUM_HARMONICS = 10

# Configurações originais
TARGET_FREQS = None
LABEL_FROM_FREQ = None
MIN_HZ, MAX_HZ = 5.0, 40.0
FREQ_GRID_STEP = 0.25
WINDOW_SEC = 2.0
WINDOW_STEP_SEC = 0.1
ONSET_IGNORE_SEC = 0.0
SOURCE_DEFAULT = "dataset"

if HAVE_BRAINFLOW:
    _BF_DEFAULT_ID = getattr(BoardIds, "CYTON_BOARD", BoardIds.CYTON_BOARD).value
else:
    _BF_DEFAULT_ID = 0
BRAIN_BOARD_ID = _BF_DEFAULT_ID
BRAIN_SERIAL_PORT = "COM3"
BRAIN_OTHER_PARAMS = {}  # Parâmetros adicionais do BrainFlow

# Configurações do teclado
ROWS = [
    ["1","2","3","4","5","6","7","8","9","0"],
    ["Q","W","E","R","T","Y","U","I","O","P"],
    ["A","S","D","F","G","H","J","K","L","\u2190"],
    ["Z","X","C","V","B","N","M",",","."]
]

FREQ_TECLAS = [30.5, 20.33, 15.25, 12.20, 10.17, 8.71, 7.63, 6.78, 6.10, 5.55]
FREQ_LINHAS = [5.08, 7.63, 10.17, 15.25]

# ========= WS state =========
stream_clients = set()
raw_clients = set()
dash_clients = set()
queue = asyncio.Queue()

async def _broadcast(group, msg: dict):
    if not group:
        return
    payload = json.dumps(msg)
    await asyncio.gather(*[w.send_str(payload) for w in list(group)], return_exceptions=True)

# ========= WS handlers =========
async def ws_stream_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    stream_clients.add(ws)
    print("[Stream] cliente conectado")
    try:
        async for _ in ws:
            pass
    finally:
        stream_clients.discard(ws)
        print("[Stream] cliente desconectado")
    return ws

async def ws_raw_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    raw_clients.add(ws)
    print("[Raw] cliente conectado")
    try:
        async for _ in ws:
            pass
    finally:
        raw_clients.discard(ws)
        print("[Raw] cliente desconectado")
    return ws

async def ws_events_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    dash_clients.add(ws)
    print("[Events] dashboard conectado")
    try:
        async for _ in ws:
            pass
    finally:
        dash_clients.discard(ws)
        print("[Events] dashboard desconectado")
    return ws

# ========= Páginas web =========
RAW_HTML = """
<!doctype html>
<meta charset="utf-8" />
<title>SSVEP Data</title>
<style>
 body { font-family: system-ui, sans-serif; margin: 20px; }
 pre { background: #f6f6f6; padding: 12px; border-radius: 8px; max-width: 900px; }
</style>
<h1>SSVEP Data</h1>
<p id="status">conectando...</p>
<div id="root"></div>
<script>
const statusEl = document.getElementById('status');
const root = document.getElementById('root');
const ws = new WebSocket("ws://%HOST%:%PORT%/raw_stream");
ws.onopen = () => statusEl.textContent = "conectado";
ws.onclose = () => statusEl.textContent = "desconectado";
ws.onmessage = (ev) => {
  const msg = JSON.parse(ev.data);
  const pre = document.createElement('pre');
  pre.textContent = JSON.stringify(msg, null, 2);
  root.innerHTML = '';
  root.appendChild(pre);
};
</script>
"""

DASHBOARD_HTML = """
<!doctype html>
<meta charset="utf-8" />
<title>SSVEP – Classificação em tempo real</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 20px; }
  .ok { color: green; }
  .bad { color: red; }
  pre { background: #f6f6f6; padding: 12px; border-radius: 8px; }
</style>

<h1>SSVEP – Classificação em tempo real</h1>
<p>Status: <span id="status">conectando…</span></p>
<div id="log"></div>

<script>
const log = (html) => {
  const div = document.getElementById('log');
  const el = document.createElement('div');
  el.innerHTML = html;
  div.prepend(el);
};

const ws = new WebSocket("ws://127.0.0.1:8082/events");
ws.onopen = () => document.getElementById('status').textContent = "conectado";
ws.onclose = () => document.getElementById('status').textContent = "desconectado";
ws.onmessage = (ev) => {
  const msg = JSON.parse(ev.data);

  if (msg.type === "prediction") {
    const a0 = (msg.window && typeof msg.window.start === "number") ? msg.window.start : 0;
    const a1 = (msg.window && typeof msg.window.end   === "number") ? msg.window.end   : msg.samples_used;
    const n  = (typeof msg.samples_used === "number") ? msg.samples_used : (a1 - a0);

    const ms0 = Math.round(a0 / msg.sfreq * 1000);
    const ms1 = Math.round(a1 / msg.sfreq * 1000);

    const ok = (msg.pred === msg.label_true);

    log(`<pre>
trial: ${msg.trial}
amostras usadas: ${a0} a ${a1} (n=${n})
predição: <b class="${ok ? 'ok' : 'bad'}">${msg.pred}</b>  |  verdadeiro: ${msg.label_true}
alg: ${msg.alg} | harmônicos: ${msg.num_harmonics} | latência: ${Number(msg.latency_s).toFixed(4)}s
</pre>`);
  } else if (msg.type === "end_trial") {
    log(`<b>Fim do trial ${msg.trial}</b>`);
  } else if (msg.type === "end_dataset") {
    log(`<b>Fim do dataset</b>`);
  }
};
</script>
"""

async def raw_page(request):
    html = RAW_HTML.replace("%HOST%", HOST).replace("%PORT%", str(PORT))
    return web.Response(text=html, content_type="text/html")

async def dashboard_page(request):
    html = DASHBOARD_HTML.replace("127.0.0.1:8082", f"{HOST}:{PORT_EVENTS}")
    return web.Response(text=html, content_type="text/html")

# ========= Funções ORIGINAIS =========
def _normalize_to_4d(dataset):
    X = dataset["X"]
    ch_names = np.array(dataset.get("ch_names", []), dtype=object)

    if X.ndim == 4:
        return dataset
    if X.ndim != 3:
        raise ValueError(f"X com ndim inesperado: {X.ndim}, shape={X.shape}")

    C_expected = len(ch_names) if ch_names.size else None
    if C_expected is not None and C_expected == X.shape[1]:
        X_ = X
    elif C_expected is not None and C_expected == X.shape[2]:
        X_ = np.moveaxis(X, 2, 1)
    else:
        X_ = X if X.shape[2] > X.shape[1] else np.moveaxis(X, 2, 1)

    X_ = X_[:, None, :, :]
    new_ds = dict(dataset)
    new_ds["X"] = X_
    return new_ds

def channel_selection(dataset, channels_selected):
    ds = _normalize_to_4d(dataset)
    ch_names = np.array(ds["ch_names"], dtype=object)

    lut = {str(n).upper(): i for i, n in enumerate(ch_names)}
    try:
        ch_idx = np.array([lut[str(n).upper()] for n in channels_selected], dtype=int)
    except KeyError as e:
        missing = [str(e.args[0])]
        raise ValueError(f"Canais não encontrados: {missing}. Disponíveis: {ch_names.tolist()}")

    new_ds = dict(ds)
    new_ds["X"] = ds["X"][:, :, ch_idx, :]
    new_ds["ch_names"] = ch_names[ch_idx].tolist()
    new_ds["y"] = ds.get("y")
    return new_ds

def build_target(target_freq, sfreq, total_time, num_harmonics=3, weight_scheme="1/k"):
    nyq = sfreq / 2.0
    ks = np.arange(1, num_harmonics + 1, dtype=float)
    ks = ks[target_freq * ks <= nyq]
    if ks.size == 0:
        ks = np.array([1.0])

    if weight_scheme == "1/k":
        w = 1.0 / ks
    elif weight_scheme == "1/sqrtk":
        w = 1.0 / np.sqrt(ks)
    else:
        w = np.ones_like(ks)

    y = np.zeros((ks.size * 2, total_time))
    t = np.arange(total_time) / sfreq
    for i, k in enumerate(ks):
        y[2*i,   :] = np.sin(2 * np.pi * target_freq * k * t) * w[i]
        y[2*i+1, :] = np.cos(2 * np.pi * target_freq * k * t) * w[i]
    return y

def _decimate_if_needed(chunk_2d, fs_orig, fs_target):
    if fs_target is None or fs_target == fs_orig:
        return chunk_2d, fs_orig
    ratio = fs_orig / float(fs_target)
    k = int(round(ratio))
    if k >= 1 and abs(fs_orig / k - fs_target) < 1e-6:
        return chunk_2d[:, ::k], int(round(fs_orig / k))
    return chunk_2d, fs_orig

def _looks_like_hz(unique_vals, min_hz=2.0, max_hz=60.0):
    if unique_vals.size == 0:
        return False
    uv = unique_vals.astype(float)
    if np.any(np.abs(uv - np.round(uv)) > 1e-9):
        return True
    inside = (uv >= min_hz) & (uv <= max_hz)
    return inside.mean() > 0.8 and uv.size >= 8

def infer_target_freqs_from_y(y, min_hz=2.0, max_hz=60.0, grid_step=1.0):
    y_arr = np.asarray(y)
    y_valid = y_arr[~np.isnan(y_arr)] if y_arr.size else y_arr
    uniq = np.unique(y_valid.astype(float)) if y_valid.size else np.array([], dtype=float)

    if _looks_like_hz(uniq, min_hz, max_hz):
        freqs = uniq[(uniq >= min_hz) & (uniq <= max_hz)]
        freqs = np.sort(freqs.astype(float))
        def label_mapper(freq_hz: float): return float(freq_hz)
        return freqs, label_mapper

    freqs = np.arange(min_hz, max_hz + 1e-12, grid_step, dtype=float)
    uniq_lbls = np.unique(y_arr) if y_arr.size else np.array([], dtype=float)

    if uniq_lbls.size:
        uniq_lbls = uniq_lbls.astype(float)
        def label_mapper(freq_hz: float):
            idx = int(np.argmin(np.abs(uniq_lbls - freq_hz)))
            if np.all(np.abs(uniq_lbls - np.round(uniq_lbls)) < 1e-9):
                return int(round(uniq_lbls[idx]))
            return float(uniq_lbls[idx])
    else:
        def label_mapper(freq_hz: float): return float(freq_hz)
    return freqs, label_mapper

def _classify_epoch_with_main_alg(X_epoch, sfreq, total_time):
    freqs = TARGET_FREQS if TARGET_FREQS is not None else np.arange(MIN_HZ, MAX_HZ + 1e-12, FREQ_GRID_STEP)

    X0 = X_epoch - X_epoch.mean(axis=1, keepdims=True)

    start = time.time()
    best_corr = -np.inf
    best_f = float(freqs[0]) if len(freqs) else 0.0

    for f in freqs:
        y_ = build_target(float(f), sfreq, total_time, num_harmonics=NUM_HARMONICS, weight_scheme="1/k")
        y_ = y_ - y_.mean(axis=1, keepdims=True)

        model = skcd.CCA(n_components=1) if ALG == "cca" else skcd.PLSCanonical(n_components=1)
        model.fit(X0.T, y_.T)
        X_c, y_c = model.transform(X0.T, y_.T)
        corr = np.corrcoef(X_c.T, y_c.T)[0, 1]
        if corr > best_corr:
            best_corr = corr
            best_f = float(f)

    dt = time.time() - start
    pred_hz = best_f
    pred_label = LABEL_FROM_FREQ(pred_hz) if callable(LABEL_FROM_FREQ) else pred_hz
    return pred_label, pred_hz, dt

# ========= FUNÇÕES DE GRAVAÇÃO/REPRODUÇÃO =========
def save_signal_recording(signal_data, metadata, filename=None):
    """Salva um sinal gravado em arquivo."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.pkl"
    
    filepath = os.path.join(RECORDINGS_DIR, filename)
    
    recording = {
        'signal_data': signal_data,
        'metadata': metadata,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(recording, f)
    
    print(f" Sinal salvo: {filepath}")
    return filepath

def load_signal_recording(filename):
    """Carrega um sinal gravado."""
    filepath = os.path.join(RECORDINGS_DIR, filename)
    
    try:
        with open(filepath, 'rb') as f:
            recording = pickle.load(f)
        
        print(f"Sinal carregado: {filepath}")
        return recording
    except Exception as e:
        print(f"Erro ao carregar gravação {filename}: {e}")
        raise

def list_recordings():
    """Lista todas as gravações disponíveis."""
    recordings = []
    for file in os.listdir(RECORDINGS_DIR):
        if file.endswith('.pkl'):
            filepath = os.path.join(RECORDINGS_DIR, file)
            try:
                with open(filepath, 'rb') as f:
                    recording = pickle.load(f)
                recordings.append({
                    'filename': file,
                    'metadata': recording.get('metadata', {}),
                    'timestamp': recording.get('timestamp', '')
                })
            except Exception as e:
                print(f"Erro ao carregar {file}: {e}")
                continue
    return recordings

# ========= GERADOR DE SINAIS SINTÉTICOS =========
def generate_synthetic_signal(frequency_hz, duration_sec=5, sfreq=1000, n_channels=8):
    """Gera um sinal SSVEP sintético para uma frequência específica."""
    t = np.arange(0, duration_sec, 1/sfreq)
    signal = np.zeros((n_channels, len(t)))
    
    # Canal principal tem o sinal SSVEP
    signal[0, :] = np.sin(2 * np.pi * frequency_hz * t)
    
    # Adicionar harmônicos
    signal[0, :] += 0.3 * np.sin(2 * np.pi * 2 * frequency_hz * t)
    signal[0, :] += 0.2 * np.sin(2 * np.pi * 3 * frequency_hz * t)
    
    # Outros canais têm ruído correlacionado
    for i in range(1, n_channels):
        signal[i, :] = 0.1 * np.random.randn(len(t)) + 0.1 * signal[0, :]
    
    return signal

# ========= Produtores ORIGINAIS =========
async def producer_task_dataset(app):
    """ORIGINAL - Modo dataset padrão"""
    if not HAVE_DATASET:
        print("[dataset] ERRO: 'bciflow' ausente. Use --source brainflow ou instale o dataset.")
        return

    try:
        dataset = mengu(subject=SUBJECT, path="data/ssvep/", depth=[DEPTH])
    except Exception as e:
        print(f"[dataset] ERRO ao carregar dados: {e}\nColoque os arquivos em data/ssvep/ ou rode com --source brainflow.")
        return

    dataset = _normalize_to_4d(dataset)
    dataset = channel_selection(dataset, CHANNELS_SELECTED)

    global TARGET_FREQS, LABEL_FROM_FREQ
    TARGET_FREQS, LABEL_FROM_FREQ = infer_target_freqs_from_y(
        dataset.get("y", []), min_hz=MIN_HZ, max_hz=MAX_HZ, grid_step=FREQ_GRID_STEP
    )
    print(f"[Freqs] TARGET_FREQS ({len(TARGET_FREQS)}): {TARGET_FREQS}")

    X = dataset["X"]
    sfreq = int(dataset.get("sfreq", FALLBACK_SFREQ))
    T, B, C, S = X.shape
    assert B == 1

    chunk_len = max(1, int(sfreq * (CHUNK_MS/1000)))

    for t in range(T):
        X_trial_full = X[t, 0, :, :]
        S_full = X_trial_full.shape[1]
        try:
            true_label, true_hz, _ = _classify_epoch_with_main_alg(X_trial_full, sfreq, S_full)
        except Exception as e:
            print(f"[Truth] erro trial {t}: {e}")
            true_label, true_hz = None, None

        s0 = 0
        while s0 < S:
            s1 = min(s0 + chunk_len, S)
            data_chunk = X[t, 0, :, s0:s1]

            payload_stream = {
                "type": "chunk",
                "trial": int(t),
                "sfreq": int(sfreq),
                "label": true_label,
                "label_hz": (float(true_hz) if true_hz is not None else None),
                "t0": int(s0),
                "t1": int(s1),
                "data": data_chunk.tolist(),
            }
            await _broadcast(stream_clients, payload_stream)
            await queue.put(payload_stream)

            raw_chunk, raw_fs = _decimate_if_needed(data_chunk, sfreq, RAW_TARGET_FS)
            payload_raw = {
                "fs": int(raw_fs),
                "channels": int(C),
                "trial": int(t),
                "start": int(s0),
                "end": int(s1),
                "matrix": raw_chunk.tolist(),
            }
            await _broadcast(raw_clients, payload_raw)

            await asyncio.sleep(CHUNK_MS/1000)
            s0 = s1

        end_msg = {"type": "end_trial", "trial": int(t)}
        await _broadcast(stream_clients, end_msg)
        await _broadcast(raw_clients, end_msg)
        await queue.put(end_msg)

    print("[Producer] dataset encerrado")

async def producer_task_brainflow(app):
    """ORIGINAL - Modo EEG real"""
    if not HAVE_BRAINFLOW:
        print("[BrainFlow] ERRO: pacote 'brainflow' não encontrado. pip install brainflow")
        return

    params = BrainFlowInputParams()
    params.serial_port = BRAIN_SERIAL_PORT
    
    # Adiciona parâmetros customizados do BrainFlow
    for k, v in BRAIN_OTHER_PARAMS.items():
        setattr(params, k, v)

    board = BoardShim(BRAIN_BOARD_ID, params)
    print(f"[BrainFlow] preparando sessão (board_id={BRAIN_BOARD_ID}, serial={BRAIN_SERIAL_PORT})")
    board.prepare_session()
    board.start_stream()

    sfreq = int(BoardShim.get_sampling_rate(BRAIN_BOARD_ID))
    eeg_idx = BoardShim.get_eeg_channels(BRAIN_BOARD_ID)
    ch_count = len(eeg_idx)
    print(f"[BrainFlow] conectado: sfreq={sfreq} Hz, EEG_channels={eeg_idx}")

    global TARGET_FREQS, LABEL_FROM_FREQ
    TARGET_FREQS = np.arange(MIN_HZ, MAX_HZ + 1e-12, FREQ_GRID_STEP, dtype=float)
    LABEL_FROM_FREQ = lambda f: float(f)
    print(f"[BrainFlow] grade de frequências ({len(TARGET_FREQS)}): {TARGET_FREQS}")

    chunk_len = max(1, int(round(sfreq * (CHUNK_MS/1000))))
    buf = None
    total = 0
    trial_id = 0

    try:
        while True:
            data = board.get_board_data()
            if data.size > 0:
                arr = np.asarray(data, dtype=float)
                if arr.ndim == 1:
                    arr = arr[:, None]
                arr = arr[eeg_idx, :]
                buf = arr if buf is None else np.concatenate([buf, arr], axis=1)

                while buf is not None and buf.shape[1] >= chunk_len:
                    out = buf[:, :chunk_len]
                    buf = buf[:, chunk_len:] if buf.shape[1] > chunk_len else None

                    t0 = total
                    t1 = total + out.shape[1]
                    total = t1

                    payload_stream = {
                        "type": "chunk",
                        "trial": int(trial_id),
                        "sfreq": int(sfreq),
                        "label": None,
                        "label_hz": None,
                        "t0": int(t0),
                        "t1": int(t1),
                        "data": out.tolist(),
                    }
                    await _broadcast(stream_clients, payload_stream)
                    await queue.put(payload_stream)

                    raw_chunk, raw_fs = _decimate_if_needed(out, sfreq, RAW_TARGET_FS)
                    payload_raw = {
                        "fs": int(raw_fs),
                        "channels": int(ch_count),
                        "trial": int(trial_id),
                        "start": int(t0),
                        "end": int(t1),
                        "matrix": raw_chunk.tolist(),
                    }
                    await _broadcast(raw_clients, payload_raw)

            await asyncio.sleep(CHUNK_MS/1000)
    finally:
        try:
            board.stop_stream()
        except Exception:
            pass
        try:
            board.release_session()
        except Exception:
            pass
        print("[BrainFlow] sessão encerrada")

# ========= Consumidor ORIGINAL =========
async def consumer_task(app):
    buffers = {}
    while True:
        data = await queue.get()
        typ = data.get("type")

        if typ == "chunk":
            trial = int(data["trial"])
            sfreq = int(data["sfreq"])
            label = data.get("label")
            label_hz = data.get("label_hz", None)
            chunk = np.asarray(data["data"], dtype=float)

            st = buffers.get(trial)
            if st is None:
                wlen = max(1, int(round(sfreq * WINDOW_SEC)))
                step = max(1, int(round(sfreq * WINDOW_STEP_SEC)))
                onset_ignore = int(round(sfreq * ONSET_IGNORE_SEC))
                st = {
                    "sfreq": sfreq,
                    "label": label,
                    "label_hz": label_hz,
                    "data": None,
                    "window_len": wlen,
                    "step_len": step,
                    "next_eval_at": wlen + onset_ignore
                }
            else:
                st["label"] = label
                st["label_hz"] = label_hz

            st["data"] = chunk if st["data"] is None else np.concatenate([st["data"], chunk], axis=1)
            buffers[trial] = st

            S_used = st["data"].shape[1]

            while S_used >= st["next_eval_at"]:
                end = st["next_eval_at"]
                start = max(0, end - st["window_len"])
                X_win = st["data"][:, start:end]

                pred_label, pred_hz, dt = _classify_epoch_with_main_alg(X_win, st["sfreq"], end - start)

                await _broadcast(dash_clients, {
                    "type": "prediction",
                    "trial": trial,
                    "label_true": st["label"],
                    "label_true_hz": st.get("label_hz"),
                    "samples_used": int(end - start),
                    "sfreq": st["sfreq"],
                    "alg": ALG,
                    "num_harmonics": NUM_HARMONICS,
                    "pred": pred_label,
                    "pred_label": pred_label,
                    "pred_hz": pred_hz,
                    "latency_s": dt,
                    "window": {"start": int(start), "end": int(end)}
                })

                st["next_eval_at"] += st["step_len"]

        elif typ == "end_trial":
            trial = int(data["trial"])
            buffers.pop(trial, None)
            await _broadcast(dash_clients, {"type": "end_trial", "trial": trial})

# ========= MODOS DE GRAVAÇÃO/REPRODUÇÃO =========
async def send_stable_prediction(hz, duration_s, trial_id=0, label_true=None):
    """Envia predição estável para o teclado."""
    start_time = time.time()
    
    while time.time() - start_time < duration_s:
        pred_msg = {
            "type": "prediction", 
            "pred_hz": hz,
            "pred_label": hz,
            "trial": trial_id,
            "label_true": label_true,
            "label_true_hz": label_true,
            "samples_used": int(duration_s * 1000),
            "sfreq": 1000,
            "alg": ALG,
            "num_harmonics": NUM_HARMONICS,
            "pred": hz,
            "latency_s": 0.1,
            "window": {"start": 0, "end": int(duration_s * 1000)}
        }
        await _broadcast(dash_clients, pred_msg)
        await asyncio.sleep(0.1)

async def producer_task_simulation(app, target_word="TEST"):
    """Modo simulação de palavras"""
    print(f" Modo Simulação | Palavra: '{target_word}'")
    
    # Criar mapeamentos
    letter_to_freq = {}
    letter_to_row = {}
    for row_idx, row in enumerate(ROWS):
        for col_idx, letter in enumerate(row):
            if col_idx < len(FREQ_TECLAS):
                letter_to_freq[letter] = FREQ_TECLAS[col_idx]
                letter_to_row[letter] = row_idx

    print("Aguardando teclado...")
    while not dash_clients:
        await asyncio.sleep(1)

    print("Iniciando simulação em 3 segundos...")
    await asyncio.sleep(3)

    for i, letter in enumerate(target_word):
        if letter not in letter_to_freq:
            print(f"Letra '{letter}' não encontrada")
            continue

        tecla_freq = letter_to_freq[letter]
        linha_num = letter_to_row[letter]
        linha_freq = FREQ_LINHAS[linha_num]

        print(f"\nDigitando: '{letter}' (Frequência: {tecla_freq:.2f} Hz, Linha: {linha_num+1})")

        # Linha
        await send_stable_prediction(linha_freq, 1.5, i, f"Linha{linha_num+1}")
        await asyncio.sleep(1.0)
        
        # Tecla
        await send_stable_prediction(tecla_freq, 1.5, i, letter)
        await asyncio.sleep(1.5)

    print(f"\nSimulação de '{target_word}' concluída!")

async def producer_task_record_signals(app, target_word="TEST"):
    """Grava sinais sintéticos para uma palavra específica."""
    print(f"Gravando sinais sintéticos para: '{target_word}'")
    
    # Criar mapeamentos
    letter_to_freq = {}
    letter_to_row = {}
    for row_idx, row in enumerate(ROWS):
        for col_idx, letter in enumerate(row):
            if col_idx < len(FREQ_TECLAS):
                letter_to_freq[letter] = FREQ_TECLAS[col_idx]
                letter_to_row[letter] = row_idx

    recorded_signals = []
    
    # Para cada letra na palavra, gerar sinal sintético
    for letter_idx, letter in enumerate(target_word):
        if letter not in letter_to_freq:
            print(f"Letra '{letter}' não encontrada no teclado")
            continue
        
        target_freq = letter_to_freq[letter]
        target_row = letter_to_row[letter]
        
        print(f"Gerando sinal para '{letter}' (Frequência: {target_freq:.2f} Hz, Linha: {target_row+1})")
        
        # Gerar sinal sintético
        signal_data = generate_synthetic_signal(target_freq, duration_sec=5)
        
        # Salvar metadados e sinal
        signal_info = {
            'letter': letter,
            'target_freq': target_freq,
            'row': target_row,
            'letter_position': letter_idx,
            'signal_shape': signal_data.shape,
            'type': 'synthetic'
        }
        
        recorded_signals.append({
            'metadata': signal_info,
            'signal_data': signal_data
        })
    
    # Salvar gravação completa
    if recorded_signals:
        metadata = {
            'word': target_word,
            'total_letters': len(recorded_signals),
            'recording_date': datetime.now().isoformat(),
            'signal_type': 'synthetic',
            'description': 'Sinais SSVEP sintéticos gerados automaticamente'
        }
        
        filename = f"word_{target_word}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        save_signal_recording(recorded_signals, metadata, filename)
        
        print(f"\nGravação concluída! Arquivo: {filename}")
        print(f"Palavra: '{target_word}'")
        print(f"Letras gravadas: {len(recorded_signals)}")
        print("Use: python ssvep_rt.py play " + filename)
    else:
        print("Nenhum sinal foi gravado")

async def producer_task_play_recording(app, recording_file):
    """Reproduz uma gravação salva."""
    try:
        recording = load_signal_recording(recording_file)
    except Exception as e:
        print(f"Erro ao carregar gravação: {e}")
        return
    
    signals = recording['signal_data']
    metadata = recording['metadata']
    
    word = metadata.get('word', 'DESCONHECIDA')
    print(f"Reproduzindo gravação: '{word}'")
    print(f"Arquivo: {recording_file}")
    
    # Esperar teclado conectar
    print(" Aguardando teclado...")
    while not dash_clients:
        await asyncio.sleep(1)
    
    print("Iniciando reprodução em 3 segundos...")
    await asyncio.sleep(3)
    
    # Reproduzir cada letra
    for signal_info in signals:
        metadata = signal_info['metadata']
        letter = metadata['letter']
        target_freq = metadata['target_freq']
        row = metadata['row']
        
        print(f"\nReproduzindo: '{letter}' (Frequência: {target_freq:.2f} Hz, Linha: {row+1})")
        
        linha_freq = FREQ_LINHAS[row]
        
        # Simular linha
        await send_stable_prediction(linha_freq, 1.5, metadata['letter_position'], f"Linha{row+1}")
        await asyncio.sleep(1.0)
        
        # Simular tecla
        await send_stable_prediction(target_freq, 1.5, metadata['letter_position'], letter)
        await asyncio.sleep(1.5)
    
    print(f"\nReprodução de '{word}' concluída!")

async def producer_task_list_recordings(app):
    """Lista todas as gravações disponíveis."""
    recordings = list_recordings()
    
    if not recordings:
        print("Nenhuma gravação encontrada")
        return
    
    print("\nGRAVAÇÕES DISPONÍVEIS:")
    print("=" * 50)
    for i, rec in enumerate(recordings, 1):
        metadata = rec['metadata']
        print(f"{i}. {rec['filename']}")
        print(f"   Palavra: {metadata.get('word', 'N/A')}")
        print(f"   Letras: {metadata.get('total_letters', 'N/A')}")
        print(f"   Tipo: {metadata.get('signal_type', 'N/A')}")
        print(f"   Data: {rec.get('timestamp', 'N/A')[:16]}")
        print()

# ========= Modo OFFLINE =========
def _filter_by_targets_and_window(dataset, targets_range, t1_samples):
    X = dataset["X"]
    y = np.asarray(dataset.get("y", []))
    T, B, C, S = X.shape
    assert B == 1

    tmin, tmax = int(targets_range[0]), int(targets_range[1])
    keep = (y >= tmin) & (y <= tmax)
    X_f = X[keep, :, :, :min(int(t1_samples), S)]
    y_f = y[keep]
    return X_f, y_f

def _classify_block_pls_or_cca(X_block, sfreq, total_time, targets, num_harmonics=NUM_HARMONICS, alg=ALG):
    if isinstance(targets, tuple):
        targets = np.arange(targets[0], targets[1] + 1, dtype=float)
    else:
        targets = np.asarray(targets, dtype=float)

    Y_list = []
    for f in targets:
        Y = build_target(float(f), sfreq, total_time, num_harmonics=num_harmonics, weight_scheme="flat")
        Y = Y - Y.mean(axis=1, keepdims=True)
        Y_list.append(Y)

    preds = []
    times = []
    T, B, C, S = X_block.shape
    assert B == 1
    for t in range(T):
        X_ = X_block[t, 0, :, :]
        X0 = X_ - X_.mean(axis=1, keepdims=True)
        start = time.time()

        best_corr = -np.inf
        best_idx  = 0
        for i, Y in enumerate(Y_list):
            if alg == "cca":
                model = skcd.CCA(n_components=1)
            else:
                model = skcd.PLSCanonical(n_components=1)
            model.fit(X0.T, Y.T)
            X_c, Y_c = model.transform(X0.T, Y.T)
            corr = np.corrcoef(X_c.T, Y_c.T)[0, 1]
            if corr > best_corr:
                best_corr = corr
                best_idx = i

        times.append(time.time() - start)
        preds.append(targets[best_idx])

    return np.asarray(preds), float(np.mean(times)) if times else float("nan")

def offline_benchmark(
    subjects=None,
    depths=None,
    channels_sel=CHANNELS_SELECTED,
    targets_window=16,
    time_windows_samples=(500, 1000, 1500, 2000),
    save_dir=".",
    alg=ALG,
    verbose=True,
):
    try:
        import pandas as pd
    except Exception:
        pd = None

    if not HAVE_DATASET:
        print("[offline] ERRO: 'bciflow' ausente. Baixe os dados ou rode em modo BrainFlow (--source brainflow).")
        return

    if subjects is None:
        subjects = list(range(19, 31))
    if depths is None:
        depths = ["high", "low"]

    os.makedirs(save_dir, exist_ok=True)

    fmin_off, fmax_off = 1, 60
    targets_ranges = [(i, i + targets_window - 1) for i in range(fmin_off, fmax_off - targets_window + 2)]

    for subj in subjects:
        for depth in depths:
            if verbose:
                print(f"\n[offline] Subject {subj} | depth={depth} | alg={alg}")

            try:
                dataset = mengu(subject=subj, path="data/ssvep/", depth=[depth])
            except Exception as e:
                print(f"[offline] PULANDO subject={subj}, depth={depth}: {e}")
                continue

            dataset = _normalize_to_4d(dataset)
            dataset = channel_selection(dataset, channels_sel)

            sfreq = int(dataset.get("sfreq", FALLBACK_SFREQ))
            rows = []

            for t_range in targets_ranges:
                if verbose:
                    print(f"  - targets: {t_range}")
                for tw in time_windows_samples:
                    X_blk, y_blk = _filter_by_targets_and_window(dataset, t_range, t1_samples=tw)
                    if X_blk.shape[0] == 0:
                        continue

                    total_time = X_blk.shape[3]
                    preds, mean_t = _classify_block_pls_or_cca(
                        X_blk, sfreq, total_time, targets=t_range, num_harmonics=NUM_HARMONICS, alg=alg
                    )

                    y_true = y_blk.astype(float)
                    acc = float((preds == y_true).sum()) / max(1, len(y_true))

                    if verbose:
                        print(f"    tempo [0..{tw}] -> trials={len(y_true)}  acc={acc:.3f}  mean_time={mean_t:.4f}s")

                    rows.append({
                        "subject": int(subj),
                        "depth": str(depth),
                        "targets_range": f"{int(t_range[0])}-{int(t_range[1])}",
                        "time_window_samples": int(tw),
                        "sfreq": int(sfreq),
                        "n_trials": int(len(y_true)),
                        "accuracy": float(acc),
                        "mean_time_s": float(mean_t),
                        "alg": str(alg),
                        "num_harmonics": int(NUM_HARMONICS),
                        "channels": ",".join(channels_sel),
                    })

            if rows:
                if pd is not None:
                    df = pd.DataFrame(rows)
                    out_path = os.path.join(save_dir, f"subject_{subj}_depth_{depth}_targets_{targets_window}.csv")
                    df.to_csv(out_path, index=False)
                    print(f"[offline] CSV salvo: {out_path}")
                else:
                    out_path = os.path.join(save_dir, f"subject_{subj}_depth_{depth}_targets_{targets_window}.jsonl")
                    with open(out_path, "w", encoding="utf-8") as f:
                        for r in rows:
                            f.write(json.dumps(r) + "\n")
                    print(f"[offline] JSONL salvo: {out_path}")

# ========= Setup Principal =========
def parse_arguments():
    """Determina o modo baseado nos argumentos"""
    if len(sys.argv) == 1:
        return "dataset", None
    
    first_arg = sys.argv[1].lower()
    
    # Modos existentes
    if first_arg in ["brainflow", "offline", "stream"]:
        return first_arg, None
    elif first_arg == "realtime":
        return "dataset", None
    
    # Novos modos de gravação/reprodução
    elif first_arg == "record":
        if len(sys.argv) > 2:
            return "record", sys.argv[2].upper()
        else:
            return "record", "TEST"
    
    elif first_arg == "play":
        if len(sys.argv) > 2:
            return "play", sys.argv[2]
        else:
            return "play", None
    
    elif first_arg == "list":
        return "list", None
    
    else:
        # Assume que é uma palavra para simulação
        return "simulation", sys.argv[1].upper()

def make_apps(mode="dataset", target=None):
    app_stream = web.Application()
    app_stream.add_routes([
        web.get("/stream", ws_stream_handler),
        web.get("/raw_stream", ws_raw_handler),
        web.get("/raw", raw_page),
        web.get("/dashboard", dashboard_page),
    ])
    
    app_events = web.Application()
    app_events.add_routes([web.get("/events", ws_events_handler)])

    async def on_startup(app):
        if mode == "brainflow":
            # Parse de argumentos customizados para BrainFlow
            args = sys.argv[2:]
            i = 0
            while i < len(args):
                if args[i] == "--serial" and i+1 < len(args):
                    global BRAIN_SERIAL_PORT
                    BRAIN_SERIAL_PORT = args[i+1]
                    i += 2
                elif args[i] == "--board-id" and i+1 < len(args):
                    global BRAIN_BOARD_ID
                    BRAIN_BOARD_ID = int(args[i+1])
                    i += 2
                elif args[i] == "--other-params" and i+1 < len(args):
                    # Formato: key1=value1,key2=value2
                    global BRAIN_OTHER_PARAMS
                    params_str = args[i+1]
                    for param in params_str.split(','):
                        if '=' in param:
                            key, value = param.split('=', 1)
                            BRAIN_OTHER_PARAMS[key.strip()] = value.strip()
                    i += 2
                else:
                    i += 1
            
            app["producer"] = asyncio.create_task(producer_task_brainflow(app))
            app["consumer"] = asyncio.create_task(consumer_task(app))
            print("Modo: EEG Real (BrainFlow)")
            print(f"   Porta: {BRAIN_SERIAL_PORT}")
            print(f"   Board ID: {BRAIN_BOARD_ID}")
            if BRAIN_OTHER_PARAMS:
                print(f"   Parâmetros extras: {BRAIN_OTHER_PARAMS}")
        elif mode == "simulation":
            app["producer"] = asyncio.create_task(producer_task_simulation(app, target))
            print(f"Modo: Simulação | Palavra: '{target}'")
        elif mode == "record":
            app["producer"] = asyncio.create_task(producer_task_record_signals(app, target))
            print(f"Modo: Gravação | Palavra: '{target}'")
        elif mode == "play":
            if target:
                app["producer"] = asyncio.create_task(producer_task_play_recording(app, target))
                print(f"Modo: Reprodução | Arquivo: '{target}'")
            else:
                print("Especifique um arquivo para reproduzir")
                app["producer"] = asyncio.create_task(producer_task_list_recordings(app))
        elif mode == "list":
            app["producer"] = asyncio.create_task(producer_task_list_recordings(app))
            print("Modo: Lista de Gravações")
        else:  # dataset/stream/realtime
            app["producer"] = asyncio.create_task(producer_task_dataset(app))
            app["consumer"] = asyncio.create_task(consumer_task(app))
            print(" Modo: Dataset Tempo Real")
        
        print(f" Dashboard: http://{HOST}:{PORT}/dashboard")
        print(f" Raw Data: http://{HOST}:{PORT}/raw")
        if mode in ["simulation", "play"]:
            print(f"  Teclado: speller_qwerty.html")

    async def on_cleanup(app):
        for k in ["producer", "consumer"]:
            task = app.get(k)
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    app_stream.on_startup.append(on_startup)
    app_stream.on_cleanup.append(on_cleanup)
    return app_stream, app_events

if __name__ == "__main__":
    mode, target = parse_arguments()
    
    if mode == "offline":
        # Modo offline especial
        subjects = None
        depths = None
        targets_window = 16
        time_windows_samples = [500, 1000, 1500, 2000]
        save_dir = "."
        offline_alg = ALG

        args = sys.argv[2:]
        i = 0
        while i < len(args):
            if args[i] == "--subjects" and i+1 < len(args):
                val = args[i+1]; i += 2
                if "-" in val:
                    a, b = val.split("-")
                    subjects = list(range(int(a), int(b)+1))
                else:
                    subjects = [int(x) for x in val.split(",")]
            elif args[i] == "--depths" and i+1 < len(args):
                depths = [x.strip() for x in args[i+1].split(",")]; i += 2
            elif args[i] == "--targets-window" and i+1 < len(args):
                targets_window = int(args[i+1]); i += 2
            elif args[i] == "--time-windows" and i+1 < len(args):
                time_windows_samples = [int(x) for x in args[i+1].split(",")]; i += 2
            elif args[i] == "--save-dir" and i+1 < len(args):
                save_dir = args[i+1]; i += 2
            elif args[i] == "--alg" and i+1 < len(args):
                offline_alg = args[i+1].lower(); i += 2
            else:
                i += 1
        
        if not HAVE_DATASET:
            print("[offline] ERRO: 'bciflow' ausente. Baixe os dados ou rode em modo BrainFlow (--source brainflow).")
            sys.exit(1)
        
        offline_benchmark(
            subjects=subjects, depths=depths,
            channels_sel=CHANNELS_SELECTED,
            targets_window=targets_window,
            time_windows_samples=time_windows_samples,
            save_dir=save_dir,
            alg=offline_alg,
            verbose=True,
        )
    else:
        print(f"=== SSVEP Teclado - Sistema Completo ===")
        print(f" Modo: {mode}")
        if target:
            print(f" Alvo: {target}")
        
        app_stream, app_events = make_apps(mode=mode, target=target)
        
        async def main():
            runner1 = web.AppRunner(app_stream)
            runner2 = web.AppRunner(app_events)
            
            await runner1.setup()
            await runner2.setup()
            
            site1 = web.TCPSite(runner1, HOST, PORT)
            site2 = web.TCPSite(runner2, HOST, PORT_EVENTS)
            
            await site1.start()
            await site2.start()
            
            print(f" Servidor: http://{HOST}:{PORT}")
            print(f" WebSocket: ws://{HOST}:{PORT_EVENTS}/events")
            print("  Pressione Ctrl+C para parar")
            
            await asyncio.Future()

        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("\n Servidor parado")