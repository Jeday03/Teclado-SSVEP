"""
SSVEP

Uso:
- Streaming (tempo real) – dataset (padrão):
    python ssvep_rt.py
    ou
    python ssvep_rt.py stream --subject 19 --depth high

- Streaming EEG ao vivo:
    python ssvep_rt.py stream --source brainflow --serial COM3 --board-id 0
  Observações:
    - Windows: --serial COM3    |   Linux/macOS: --serial /dev/ttyUSB0
    - --board-id:
        -1 = Synthetic (simulador)
         0 = Cyton (8 canais)
         2 = Cyton Daisy (16 canais)
         1 = Ganglion

- OFFLINE (o que o codigo do stephan fazia):
    python ssvep_rt.py offline
    # opções:
    python ssvep_rt.py offline --subjects 19-30 --depths high,low --targets-window 16 \
                                --time-windows 500,1000,1500,2000 --save-dir . --alg pls
      --subjects: intervalo "a-b" ou lista "a,b,c"
      --depths:   lista "high,low"
      --targets-window: largura da janela de alvos contíguos (ex.: 16)
      --time-windows:   endpoints em AMOSTRAS (ex.: 500,1000,1500,2000)
      --alg:            pls (padrão) ou cca
"""

import sys
import time
import os
import json
import numpy as np
import asyncio
import sklearn.cross_decomposition as skcd
from aiohttp import web

# --- datasets ---
sys.path.append("bciflow/")
from bciflow.datasets import mengu

# === EEG (BrainFlow opcional) ===
try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    HAVE_BRAINFLOW = True
except Exception:
    HAVE_BRAINFLOW = False


# ========= utils de dados =========

def _normalize_to_4d(dataset):
    """
    Garante X com shape (T, B, C, S).
    Aceita X em 4D (retorna igual) ou 3D (T,C,S ou T,S,C) e adiciona B=1.
    """
    X = dataset["X"]
    ch_names = np.array(dataset.get("ch_names", []), dtype=object)

    if X.ndim == 4:
        return dataset
    if X.ndim != 3:
        raise ValueError(f"X com ndim inesperado: {X.ndim}, shape={X.shape}")

    C_expected = len(ch_names) if ch_names.size else None
    if C_expected is not None and C_expected == X.shape[1]:
        X_ = X                       # (T,C,S)
    elif C_expected is not None and C_expected == X.shape[2]:
        X_ = np.moveaxis(X, 2, 1)    # (T,S,C) -> (T,C,S)
    else:
        X_ = X if X.shape[2] > X.shape[1] else np.moveaxis(X, 2, 1)

    X_ = X_[:, None, :, :]           # adiciona bands=1 -> (T,1,C,S)
    new_ds = dict(dataset)
    new_ds["X"] = X_
    return new_ds


def channel_selection(dataset, channels_selected):
    """
    Seleciona canais por NOME (case-insensitive) e mantém X em (T,B,C,S).
    """
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


# ========= geração de templates =========

def build_target(target_freq, sfreq, total_time, num_harmonics=3, weight_scheme="1/k"):
    """
    Gera senos/cossenos até o maior harmônico <= Nyquist e aplica pesos.
    weight_scheme: "1/k", "1/sqrtk" ou qualquer outro valor => pesos iguais (flat).
    Retorna shape (2 * n_harm_usados, total_time).
    """
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


# ========= config =========

HOST = "127.0.0.1"
PORT = 8081
PORT_EVENTS = 8082
CHUNK_MS = 100
FALLBACK_SFREQ = 1000
SUBJECT = 19
DEPTH = "high"
CHANNELS_SELECTED = ["PZ","PO3","PO4","PO5","PO6","POZ","O1","O2","OZ"]
RAW_TARGET_FS = 512              # None para não reamostrar
ALG = "pls"                      # "pls" ou "cca"
NUM_HARMONICS = 10

# alvo dinâmico (para RT)
TARGET_FREQS = None              # np.ndarray[float]
LABEL_FROM_FREQ = None           # callable: float Hz -> label (comparável ao "verdadeiro")
MIN_HZ, MAX_HZ = 5.0, 60.0
FREQ_GRID_STEP = 1.0

# ==== CONFIG (janela do classificador) ====
WINDOW_SEC = 2.0        # tamanho da janela (em segundos)
WINDOW_STEP_SEC = 0.1   # passo entre janelas (0.1s => janelas bastante sobrepostas)
ONSET_IGNORE_SEC = 0.0  # ignorar segundos iniciais do trial (dataset)

# ==== CONFIG (fonte de dados) ====
SOURCE_DEFAULT = "dataset"      # "dataset" (padrão) ou "brainflow"

# ==== CONFIG (BrainFlow / OpenBCI) ====
if HAVE_BRAINFLOW:
    _BF_DEFAULT_ID = getattr(BoardIds, "CYTON_BOARD", BoardIds.CYTON_BOARD).value
else:
    _BF_DEFAULT_ID = 0
BRAIN_BOARD_ID = _BF_DEFAULT_ID
BRAIN_SERIAL_PORT = "COM3"       # mude via --serial
BRAIN_OTHER_PARAMS = {}          # ex.: {"serial_number": "XYZ"}


# ========= helpers de streaming / alvo =========

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


# ========= WS state =========

stream_clients = set()   # /stream
raw_clients = set()      # /raw_stream (visualização)
dash_clients = set()     # /events (dashboard predições)
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


# ========= página simples /raw =========

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
async def raw_page(request):
    html = RAW_HTML.replace("%HOST%", HOST).replace("%PORT%", str(PORT))
    return web.Response(text=html, content_type="text/html")


# ========= classificador núcleo (RT) =========

def _classify_epoch_with_main_alg(X_epoch, sfreq, total_time):
    """
    Retorna (pred_label, pred_hz, dt) para a janela atual (RT).
    - De-mean por canal
    - Gera template y_ por frequência
    """
    freqs = TARGET_FREQS if TARGET_FREQS is not None else np.arange(MIN_HZ, MAX_HZ + 1e-12, FREQ_GRID_STEP)

    X0 = X_epoch - X_epoch.mean(axis=1, keepdims=True)  # de-mean por canal

    start = time.time()
    best_corr = -np.inf
    best_f = float(freqs[0]) if len(freqs) else 0.0

    for f in freqs:
        y_ = build_target(float(f), sfreq, total_time, num_harmonics=NUM_HARMONICS, weight_scheme="1/k")
        y_ = y_ - y_.mean(axis=1, keepdims=True)  # de-mean também no template

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


# ========= OFFLINE (benchmark estilo do amigo) =========

def _filter_by_targets_and_window(dataset, targets_range, t1_samples):
    """
    Filtro 'à la amigo':
      - Mantém apenas trials cujo y ∈ [targets_range[0], targets_range[1]] (inclusive)
      - Corta o sinal no intervalo [0:t1_samples] (amostras desde o início)
    Retorna X_filt (T, B, C, S’), y_filt
    """
    X = dataset["X"]                     # (T,1,C,S)
    y = np.asarray(dataset.get("y", []))
    T, B, C, S = X.shape
    assert B == 1

    tmin, tmax = int(targets_range[0]), int(targets_range[1])
    keep = (y >= tmin) & (y <= tmax)
    X_f = X[keep, :, :, :min(int(t1_samples), S)]
    y_f = y[keep]
    return X_f, y_f


def _classify_block_pls_or_cca(X_block, sfreq, total_time, targets, num_harmonics=NUM_HARMONICS, alg=ALG):
    """
    Para cada trial, calcula correlação com templates (sen/cos + harmônicos)
    para todos os alvos em 'targets' e escolhe o de maior correlação.
    Retorna: (preds, mean_time_s)
    """
    # alvos contíguos como inteiros
    if isinstance(targets, tuple):
        targets = np.arange(targets[0], targets[1] + 1, dtype=float)
    else:
        targets = np.asarray(targets, dtype=float)

    # Pré-computa templates com pesos 'flat' (como no código do amigo)
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
        X_ = X_block[t, 0, :, :]                 # (C, S)
        X0 = X_ - X_.mean(axis=1, keepdims=True) # de-mean por canal
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
    """
    Executa o mesmo experimento do seu amigo:
      - Para cada (subject, depth):
          - Para cada janela de alvos contíguos [i, i+targets_window-1] dentro de 1..60
          - Para cada janela de tempo [0..N] amostras (N em 'time_windows_samples')
        => roda PLS/CCA, mede acurácia e tempo médio, salva CSV por par (subject, depth).
    """
    try:
        import pandas as pd
    except Exception:
        pd = None

    if subjects is None:
        subjects = list(range(19, 31))
    if depths is None:
        depths = ["high", "low"]

    os.makedirs(save_dir, exist_ok=True)

    # janelas de alvo contíguas EXACTAS como no amigo: 1..60
    fmin_off, fmax_off = 1, 60
    targets_ranges = [(i, i + targets_window - 1) for i in range(fmin_off, fmax_off - targets_window + 2)]

    for subj in subjects:
        for depth in depths:
            if verbose:
                print(f"\n[offline] Subject {subj} | depth={depth} | alg={alg}")
            dataset = mengu(subject=subj, path="data/ssvep/", depth=[depth])
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

                    total_time = X_blk.shape[3]  # nº de amostras usados
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

            # salva um CSV por (subject, depth), como o do amigo
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


# ========= produtores (stream) =========

async def producer_task_dataset(app):
    """
    Lê o dataset Mengu e emite chunks por WS, trial por trial.
    """
    dataset = mengu(subject=SUBJECT, path="data/ssvep/", depth=[DEPTH])
    dataset = _normalize_to_4d(dataset)
    dataset = channel_selection(dataset, CHANNELS_SELECTED)

    # inferência global de frequências e mapeador (RT)
    global TARGET_FREQS, LABEL_FROM_FREQ
    TARGET_FREQS, LABEL_FROM_FREQ = infer_target_freqs_from_y(
        dataset.get("y", []), min_hz=MIN_HZ, max_hz=MAX_HZ, grid_step=FREQ_GRID_STEP
   )
    print(f"[Freqs] TARGET_FREQS ({len(TARGET_FREQS)}): {TARGET_FREQS}")
    print(f"[Freqs] LABEL_FROM_FREQ pronto")

    X = dataset["X"]                      # (T,1,C,S)
    sfreq = int(dataset.get("sfreq", FALLBACK_SFREQ))
    T, B, C, S = X.shape
    assert B == 1

    chunk_len = max(1, int(sfreq * (CHUNK_MS/1000)))

    for t in range(T):
        # ground truth JIT do trial inteiro
        X_trial_full = X[t, 0, :, :]      # (C,S)
        S_full = X_trial_full.shape[1]
        try:
            true_label, true_hz, _ = _classify_epoch_with_main_alg(X_trial_full, sfreq, S_full)
        except Exception as e:
            print(f"[Truth] erro trial {t}: {e}")
            true_label, true_hz = None, None

        # stream em chunks
        s0 = 0
        while s0 < S:
            s1 = min(s0 + chunk_len, S)
            data_chunk = X[t, 0, :, s0:s1]    # (C,n)

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
    """
    Lê EEG em tempo real via BrainFlow/OpenBCI e envia chunks contínuos (trial=0).
    """
    if not HAVE_BRAINFLOW:
        print("[BrainFlow] ERRO: pacote 'brainflow' não encontrado. pip install brainflow")
        return

    params = BrainFlowInputParams()
    params.serial_port = BRAIN_SERIAL_PORT
    # Se precisar, habilite campos adicionais em BRAIN_OTHER_PARAMS:
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

    # alvo dinâmico: sem y, usa grade uniforme + identidade
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
            # pega tudo que chegou desde a última chamada (não bloqueia)
            data = board.get_board_data()  # (n_channels_total, n_samples)
            if data.size > 0:
                arr = np.asarray(data, dtype=float)
                if arr.ndim == 1:
                    arr = arr[:, None]
                arr = arr[eeg_idx, :]   # só EEG -> (C, n)
                buf = arr if buf is None else np.concatenate([buf, arr], axis=1)

                # emite em múltiplos de chunk_len
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


# ========= consumidor (classificação RT) =========

async def consumer_task(app):
    # trial -> {"sfreq":..., "label":..., "label_hz":..., "data": np.ndarray (C,S_acum),
    #           "window_len": int, "step_len": int, "next_eval_at": int}
    buffers = {}
    while True:
        data = await queue.get()
        typ = data.get("type")

        if typ == "chunk":
            trial = int(data["trial"])
            sfreq = int(data["sfreq"])
            label = data.get("label")
            label_hz = data.get("label_hz", None)
            chunk = np.asarray(data["data"], dtype=float)  # (C, n)

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

            # acumula o sinal do trial
            st["data"] = chunk if st["data"] is None else np.concatenate([st["data"], chunk], axis=1)
            buffers[trial] = st

            S_used = st["data"].shape[1]

            # dispara predições sempre que cruzar o próximo múltiplo do passo
            while S_used >= st["next_eval_at"]:
                end = st["next_eval_at"]
                start = max(0, end - st["window_len"])
                X_win = st["data"][:, start:end]  # (C, janela)

                pred_label, pred_hz, dt = _classify_epoch_with_main_alg(
                    X_win, st["sfreq"], end - start
                )

                await _broadcast(dash_clients, {
                    "type": "prediction",
                    "trial": trial,
                    "label_true": st["label"],
                    "label_true_hz": st.get("label_hz"),
                    "samples_used": int(end - start),   # tamanho da janela (amostras)
                    "sfreq": st["sfreq"],
                    "alg": ALG,
                    "num_harmonics": NUM_HARMONICS,
                    "pred": pred_label,                  # compat com UI antigo
                    "pred_label": pred_label,            # redundante / UI novo
                    "pred_hz": pred_hz,
                    "latency_s": dt,
                    "window": {"start": int(start), "end": int(end)}
                })

                # agenda a próxima avaliação (janelas deslizantes)
                st["next_eval_at"] += st["step_len"]

        elif typ == "end_trial":
            trial = int(data["trial"])
            buffers.pop(trial, None)
            await _broadcast(dash_clients, {"type": "end_trial", "trial": trial})


# ========= setup / main =========

def make_apps(source="dataset"):
    app_stream = web.Application()
    app_stream.add_routes([
        web.get("/stream", ws_stream_handler),
        web.get("/raw_stream", ws_raw_handler),
        web.get("/raw", raw_page),
    ])
    app_events = web.Application()
    app_events.add_routes([web.get("/events", ws_events_handler)])

    async def on_startup_stream(app, source=source):
        if source == "brainflow":
            app["producer"] = asyncio.create_task(producer_task_brainflow(app))
        else:
            app["producer"] = asyncio.create_task(producer_task_dataset(app))
        app["consumer"] = asyncio.create_task(consumer_task(app))
        print(f"[Stream] ws://{HOST}:{PORT}/stream")
        print(f"[Raw]    ws://{HOST}:{PORT}/raw_stream")

    async def on_cleanup_stream(app):
        for k in ("producer", "consumer"):
            task = app.get(k)
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    app_stream.on_startup.append(on_startup_stream)
    app_stream.on_cleanup.append(on_cleanup_stream)
    return app_stream, app_events


if __name__ == "__main__":
    # subcomando: stream | offline (default: stream)
    mode = sys.argv[1] if (len(sys.argv) > 1 and sys.argv[1] in ("stream", "offline")) else "stream"

    if mode == "offline":
        # Novo OFFLINE "igual ao do amigo"
        subjects = None           # ex.: "19-30" ou "19,20,21"
        depths = None             # ex.: "high,low"
        targets_window = 16
        time_windows_samples = [500, 1000, 1500, 2000]  # em AMOSTRAS
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
        # modo padrão: streaming RT
        # parse extras do stream
        source = SOURCE_DEFAULT
        subj = SUBJECT
        dp = DEPTH

        # se chamaram explicitamente "stream", argumentos começam em 2; se não, em 1
        arg_start = 2 if (len(sys.argv) > 1 and sys.argv[1] == "stream") else 1
        args = sys.argv[arg_start:]

        i = 0
        while i < len(args):
            if args[i] == "--source" and i+1 < len(args):
                source = args[i+1].lower(); i += 2
            elif args[i] == "--subject" and i+1 < len(args):
                subj = int(args[i+1]); i += 2
            elif args[i] == "--depth" and i+1 < len(args):
                dp = args[i+1]; i += 2
            elif args[i] == "--serial" and i+1 < len(args):
                BRAIN_SERIAL_PORT = args[i+1]; i += 2
            elif args[i] == "--board-id" and i+1 < len(args):
                BRAIN_BOARD_ID = int(args[i+1]); i += 2
            else:
                i += 1

        # aplica subject/depth na config global (dataset)
        SUBJECT = subj
        DEPTH = dp

        if source == "brainflow" and not HAVE_BRAINFLOW:
            print("[BrainFlow] ERRO: modo --source brainflow requer 'brainflow'. pip install brainflow")
            sys.exit(1)

        app_stream, app_events = make_apps(source=source)
        loop = asyncio.get_event_loop()
        runner1 = web.AppRunner(app_stream)
        runner2 = web.AppRunner(app_events)
        loop.run_until_complete(runner1.setup())
        loop.run_until_complete(runner2.setup())
        site1 = web.TCPSite(runner1, HOST, PORT)
        site2 = web.TCPSite(runner2, HOST, PORT_EVENTS)
        loop.run_until_complete(site1.start())
        loop.run_until_complete(site2.start())
        print(f"[Events] ws://{HOST}:{PORT_EVENTS}/events (dashboard)")
        print("======== Running RT monolith ========")
        try:
            loop.run_forever()
        except KeyboardInterrupt:
            pass
