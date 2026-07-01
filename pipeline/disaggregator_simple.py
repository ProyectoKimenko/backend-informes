"""
Desagregador NILM de agua — versión SIMPLE, robusta y alineada con el estado del arte.

Base metodológica (investigación 2021-2026, verificada):
  - Bethke, Cohen & Stillwell 2021 (Environ. Sci.: Water Res. Technol.): segmentar
    eventos por la derivada/flancos del caudal + k-means NO-supervisado en 2D
    (caudal medio + DURACIÓN). ~92% de usos clasificados a fixture, 1 medidor, 1 Hz.
  - WEUSEDTO 2022 / PyNIWM 2024: caracterizar eventos por (duración, volumen, caudal),
    NO por caudal solo. El caudal solo no separa inodoro vs grifo (corto, distinto
    volumen) ni ducha vs lavadora (caudal similar, distinta duración).
  - Validación de reparto: cuotas de volumen por categoría de REU2016 (~24k hogares).

Frente a la versión 7D anterior (StandardScaler en disco + inferencia ND con fallback
silencioso, 718 líneas) y a una v1 1D (solo caudal): esta versión clusteriza en 2D
(caudal + duración), asigna por EVENTO, y NO guarda estado en disco — el "modelo" son
N centroides (caudal, duración) + tolerancia, que caben en disaggregation_profiles
(columnas mean_flow, mean_duration, centroid_nd=[caudal,duración], tol_nd). Determinista.

Interfaz compatible con worker/tasks.py:
  train_disaggregator(df) -> dict[int, profile]
  run_disaggregation(df, profiles) -> (df_events, df_result)
"""
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

# Preprocesado y segmentación de eventos (módulo compartido).
from pipeline.segmentation import (
    preprocess_signal,
    segment_events,
    is_valid_event,
    integrate_volume,
)
from pipeline.signatures import label_by_signature, VOLUME_RANGE_BY_LABEL, COMPOSITE

# Escalas fijas para hacer comparables caudal (L/min) y duración (s) en la distancia
# 2D, sin necesidad de un StandardScaler persistido. ~45 s de duración "pesa" como
# ~1 L/min de caudal. Stateless: no se guarda nada en disco.
FLOW_SCALE = 1.0    # L/min por unidad
DUR_SCALE = 45.0    # s por unidad
DUR_CAP = 600.0     # s — recortar duraciones extremas (fugas/riego) para que un
                    # evento atípico no distorsione el espacio de clustering 2D.


# -----------------------------------------------------------------------------
# Auto-etiquetado por (caudal, duración).
# Heurística de PARTIDA informada por EPA WaterSense + REU2016. ADVERTENCIA: esos
# valores son US/Canadá y techos regulatorios; los L/min reales de cada instalación
# difieren -> el operador debe calibrar/renombrar desde la UI. Usa caudal Y duración.
# -----------------------------------------------------------------------------
def auto_label(mean_flow: float, mean_duration: float) -> str:
    f, d = mean_flow, mean_duration
    if f < 2.5:
        return "Lavamanos / grifo" if d < 120 else "Goteo / llave prolongada"
    if f < 6.0:
        if d < 90:
            return "Inodoro / descarga"
        if d < 420:
            return "Ducha"
        return "Lavadora / lavavajillas"
    # caudal alto (>= 6 L/min): la duración decide
    if d < 45:
        return "Inodoro / grifo rápido"
    if d < 600:
        return "Ducha"
    return "Riego / manguera"


def device_name(mean_flow: float, mean_duration: float) -> str:
    # La CATEGORÍA es la clase del fixture (ducha/inodoro/lavamanos...), NO lleva el
    # caudal embebido. Antes devolvía "Ducha (8.0 L/min)", lo que fragmentaba cada
    # artefacto en tantas categorías como modos GMM (8.0 vs 8.3 = dos series) y se
    # multiplicaba en cada reentrenamiento. El caudal queda como metadato (mean_flow).
    return auto_label(mean_flow, mean_duration)


# -----------------------------------------------------------------------------
# Descubrimiento de perfiles: GMM 1D + BIC sobre el espacio 2D normalizado.
# BIC (no silhouette, que colapsa a k=2 con datos grandes) elige el nº de modos.
# -----------------------------------------------------------------------------
def _normalize(flow: np.ndarray, dur: np.ndarray) -> np.ndarray:
    dur = np.minimum(dur, DUR_CAP)   # recorte de duraciones extremas
    return np.column_stack([flow / FLOW_SCALE, dur / DUR_SCALE])


def find_profiles(
    flow: np.ndarray,
    dur: np.ndarray,
    k_min: int = 2,
    k_max: int = 8,
    min_sep: float = 0.9,
) -> np.ndarray:
    """Devuelve centroides RAW [[mean_flow, mean_duration], ...] ordenados por caudal."""
    Xn = _normalize(flow, dur)
    n = len(Xn)
    upper = max(k_min, min(k_max, len(np.unique(np.round(flow, 1)))))

    best_bic, best_means = None, None
    for k in range(k_min, upper + 1):
        if n < k * 10:
            break
        g = GaussianMixture(n_components=k, covariance_type="full",
                            random_state=42, n_init=3).fit(Xn)
        bic = g.bic(Xn)
        if best_bic is None or bic < best_bic:
            best_bic, best_means = bic, g.means_.copy()

    if best_means is None:
        best_means = np.array([[float(np.median(flow)) / FLOW_SCALE,
                                float(np.median(dur)) / DUR_SCALE]])

    # Ordenar por caudal y fusionar centroides demasiado cercanos en el espacio norm.
    best_means = best_means[np.argsort(best_means[:, 0])]
    merged: List[np.ndarray] = []
    for m in best_means:
        if merged and np.linalg.norm(m - merged[-1]) < min_sep:
            merged[-1] = (merged[-1] + m) / 2.0
        else:
            merged.append(m.copy())
    merged = np.array(merged)

    # De-normalizar a unidades reales (caudal L/min, duración s).
    return np.column_stack([merged[:, 0] * FLOW_SCALE, merged[:, 1] * DUR_SCALE])


def _tolerances(centroids_norm: np.ndarray, frac: float = 8.0, floor: float = 12.0) -> np.ndarray:
    """Radio de rechazo de cada perfil en el espacio 2D normalizado.

    tol = max(floor, frac · distancia al centroide más cercano). Un evento cuyo
    punto (caudal, duración) cae MÁS LEJOS que este radio del perfil asignado se
    manda a "No Detectado" (el modelo dice "no sé") en vez de forzar la atribución.
    Calibrado sobre los 1.1M de muestras reales (place 1): el piso irreducible de
    No Detectado (flujo basal que nunca forma evento) es ~2.7%; frac=8/floor=12 da
    ~3.2% (rechaza solo ~0.5% genuinamente anómalo) manteniendo 96.8% de cobertura.
    Valores agresivos (frac=3) rechazaban duchas/usos largos por la varianza natural
    de duración dentro de un fixture y disparaban No Detectado a ~18%.
    """
    tols = []
    for i, c in enumerate(centroids_norm):
        others = np.delete(centroids_norm, i, axis=0)
        nn = np.min(np.linalg.norm(others - c, axis=1)) if len(others) else 2.0
        tols.append(max(floor, nn * frac))
    return np.array(tols)


# -----------------------------------------------------------------------------
# ENTRENAMIENTO
# -----------------------------------------------------------------------------
def train_disaggregator(df: pd.DataFrame, min_events: int = 20) -> Dict[int, Dict]:
    df_proc = preprocess_signal(df)
    df_events = segment_events(df_proc)
    if len(df_events) < min_events:
        print(f"[TrainSimple] Solo {len(df_events)} eventos — insuficiente.")
        return {}

    train_mask = df_events["mean_flow"] >= 0.5
    # Excluir eventos CONCURRENTES (superposición de fixtures): su caudal/volumen están
    # inflados por la suma de >=2 usos y desplazan los centroides GMM y las firmas
    # físicas. No deben definir ningún perfil; se tratan aparte en inferencia.
    n_comp = 0
    if "is_composite" in df_events.columns:
        comp = df_events["is_composite"].fillna(False).astype(bool)
        n_comp = int(comp.sum())
        train_mask &= ~comp
    ev = df_events[train_mask]
    if n_comp:
        print(f"[TrainSimple] {n_comp} eventos concurrentes excluidos del entrenamiento.")
    if len(ev) < min_events:
        print("[TrainSimple] Pocos eventos con caudal >= 0.5.")
        return {}

    flow = ev["mean_flow"].values.astype(float)
    dur = ev["duration_s"].values.astype(float)
    vol = ev["volume_liters"].values.astype(float)
    # Caudal MODAL (nominal) por evento — más robusto que la media para la firma
    # (Autoflow/CIWS). Fallback a mean_flow si la columna no está (compat).
    modal = (ev["modal_flow"].values.astype(float)
             if "modal_flow" in ev.columns else flow)

    centroids = find_profiles(flow, dur)                 # raw [flow, dur]
    total = max(len(ev), 1)
    ev_norm = _normalize(flow, dur)

    # Asignación preliminar a los centroides crudos (solo para ponderar el merge).
    cn0 = _normalize(centroids[:, 0], centroids[:, 1])
    assign0 = np.argmin(
        np.linalg.norm(ev_norm[:, None, :] - cn0[None, :, :], axis=2), axis=1
    )

    def _sig(mask: np.ndarray) -> Tuple[float, float, float, float]:
        """Firma física del cluster: (caudal MODAL med, dur med, VOLUMEN med, CV del volumen).
        Usa el caudal modal (nominal) en vez de la media: más robusto para etiquetar."""
        if not mask.any():
            return 0.0, 0.0, 0.0, 0.0
        vv = vol[mask]
        mvol = float(np.median(vv))
        cvvol = float(np.std(vv) / np.mean(vv)) if np.mean(vv) > 0 else 0.0
        return float(np.median(modal[mask])), float(np.median(dur[mask])), mvol, cvvol

    # FUSIONAR por FIRMA FÍSICA: cada centroide crudo se etiqueta por su firma
    # (caudal, duración, VOLUMEN mediano, CV del volumen entre sus eventos) y los que
    # comparten firma se colapsan en un perfil. El VOLUMEN es el discriminador #1
    # (inodoro = volumen ~fijo); antes se agrupaba por rangos US de caudal+duración
    # que IGNORABAN el volumen y producían etiquetas físicamente imposibles.
    groups: Dict[str, List[int]] = {}
    for i in range(len(centroids)):
        lab = label_by_signature(*_sig(assign0 == i))
        groups.setdefault(lab, []).append(i)

    m_cent, m_label = [], []
    for lab, idxs in groups.items():
        w = np.array([max(int((assign0 == i).sum()), 1) for i in idxs], dtype=float)
        m_cent.append([
            float(np.average(centroids[idxs, 0], weights=w)),
            float(np.average(centroids[idxs, 1], weights=w)),
        ])
        m_label.append(lab)
    m_cent = np.array(m_cent, dtype=float)

    # Ordenar por caudal (salida estable) y recomputar tolerancias + asignación
    # final sobre los centroides YA fusionados.
    order = np.argsort(m_cent[:, 0])
    m_cent = m_cent[order]
    m_label = [m_label[i] for i in order]

    cent_norm = _normalize(m_cent[:, 0], m_cent[:, 1])
    tols = _tolerances(cent_norm)
    assign = np.argmin(
        np.linalg.norm(ev_norm[:, None, :] - cent_norm[None, :, :], axis=2), axis=1
    )

    profiles: Dict[int, Dict] = {}
    for i in range(len(m_cent)):
        fl = round(float(m_cent[i, 0]), 2)
        du = round(float(m_cent[i, 1]), 1)
        mask = assign == i
        n = int(mask.sum())
        _, _, mvol, cvvol = _sig(mask)            # firma del cluster fusionado
        std = float(np.std(flow[mask])) if n > 1 else 0.1
        profiles[i] = {
            "name": m_label[i],
            "label": m_label[i],                  # fixture (editable en la UI)
            "mean_flow": fl,
            "mean_duration": du,
            "median_volume_l": round(mvol, 2),    # firma física: volumen típico por uso
            "cv_volume": round(cvvol, 3),         # consistencia del volumen (bajo=cisterna)
            "st_deviation": round(max(std, 0.05), 2),
            "weight": round(n / total, 6),
            "tol": round(float(tols[i]) * FLOW_SCALE, 3),     # tol aproximado en L/min (fallback)
            "tol_nd": round(float(tols[i]), 4),               # tolerancia en espacio 2D normalizado
            "centroid_nd": [fl, du],                          # centroide 2D RAW [caudal, duración]
            "scaler_path": "",
        }

    print(f"[TrainSimple] {len(ev)} eventos -> {len(profiles)} perfiles (FIRMA física):")
    for p in profiles.values():
        print(f"   {p['label']:24s} | Q={p['mean_flow']:5.2f} L/min  dur~{p['mean_duration']:4.0f}s  "
              f"vol~{p['median_volume_l']:5.1f}L  cv={p['cv_volume']:.2f}  w={p['weight']:.3f}")
    return profiles


def _sustained_excess(excess: np.ndarray, index, min_s: float = 15.0,
                      min_level: float = 1.5, noise: float = 0.3) -> np.ndarray:
    """Del exceso (caudal - base) de un evento concurrente, conserva SOLO las jorobas
    SOSTENIDAS (run contiguo >= min_s con pico >= min_level) — un segundo fixture real.
    El resto (ruido puntual de la variación normal del fixture base sobre su mediana)
    se anula → vuelve al fixture dominante. Sin esto, atribuir el exceso timestep a
    timestep fragmentaba una ducha en cientos de micro-eventos de "Uso simultáneo"."""
    out = np.zeros_like(excess)
    active = excess > noise
    n = len(excess)
    i = 0
    while i < n:
        if not active[i]:
            i += 1
            continue
        j = i
        while j < n and active[j]:
            j += 1
        dur = (index[j - 1] - index[i]).total_seconds() if (j - 1) > i else 0.0
        if dur >= min_s and float(excess[i:j].max()) >= min_level:
            out[i:j] = excess[i:j]
        i = j
    return out


def _feat3(f: float, d: float, v: float) -> np.ndarray:
    """Vector normalizado (caudal, duración, volumen) para distancias de calibración."""
    return np.array([
        f / FLOW_SCALE,
        min(d, DUR_CAP) / DUR_SCALE,
        np.log1p(max(v, 0.0)) * 1.5,   # log: el volumen tiene cola larga (duchas)
    ])


def apply_confirmations(profiles: Dict, confirmations: List[Dict]) -> Dict:
    """Aplica las confirmaciones del operador (semi-supervisado, operador-in-the-loop).

    Cada confirmación es un evento típico que el operador etiquetó con el fixture real
    ({mean_flow, duration_s, volume_liters, confirmed_label}). Se asigna al perfil más
    cercano en (caudal, duración, volumen) y el `label` del perfil pasa a ser el fixture
    confirmado por MAYORÍA — el conocimiento del operador le gana a la heurística de
    firma. Se mantiene `name` (la firma física, clave estable); solo cambia el `label`
    visible (consistente con la edición manual de labels, que ya persiste).
    """
    if not confirmations or not profiles:
        return profiles
    plist = list(profiles.values())
    P = np.array([
        _feat3(float(p.get("mean_flow") or 0.0), float(p.get("mean_duration") or 0.0),
               float(p.get("median_volume_l") or 0.0))
        for p in plist
    ])
    from collections import Counter
    votes: Dict[int, Counter] = {}
    for c in confirmations:
        lab = c.get("confirmed_label")
        if not lab:
            continue
        cf = _feat3(float(c.get("mean_flow") or 0.0), float(c.get("duration_s") or 0.0),
                    float(c.get("volume_liters") or 0.0))
        j = int(np.argmin(np.linalg.norm(P - cf, axis=1)))
        votes.setdefault(j, Counter())[lab] += 1
    for j, cnt in votes.items():
        plist[j]["label"] = cnt.most_common(1)[0][0]
        print(f"[Confirm] perfil '{plist[j]['name']}' -> label operador '{plist[j]['label']}' ({sum(cnt.values())} confirmaciones)")
    return profiles


# -----------------------------------------------------------------------------
# INFERENCIA — asignación por EVENTO en 2D (caudal, duración).
# Cada evento segmentado se asigna al perfil 2D más cercano dentro de tolerancia;
# su caudal se reparte timestep a timestep (min(caudal, centroide)) con residual.
# -----------------------------------------------------------------------------
def run_disaggregation(df: pd.DataFrame, profiles: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not profiles:
        raise ValueError("No trained profiles provided.")

    df_proc = preprocess_signal(df)
    flow_s = df_proc["flow_smooth"]

    plist = list(profiles.values())
    # Una etiqueta por perfil (ya son fixtures distintos tras la fusión en train).
    # out_cols = etiquetas distintas preservando orden: si dos perfiles compartieran
    # label, sus litros SE SUMAN en una columna (no se crean "X #1").
    labels = [(p.get("label") or p["name"]) for p in plist]
    out_cols = list(dict.fromkeys(labels))

    # Centroides 2D (raw), su norma y la tolerancia de rechazo por perfil.
    cent = np.array([
        p.get("centroid_nd") or [float(p.get("mean_flow", 0.0)), float(p.get("mean_duration", 0.0))]
        for p in plist
    ], dtype=float)
    cent_norm = _normalize(cent[:, 0], cent[:, 1])
    tol_nd = np.array([float(p.get("tol_nd") or 0.0) for p in plist], dtype=float)

    df_result = pd.DataFrame(0.0, index=flow_s.index, columns=out_cols + [COMPOSITE, "No Detectado"])
    residual = flow_s.copy()

    df_seg = segment_events(df_proc)
    n_ok = n_rej = n_comp = 0
    if not df_seg.empty:
        for _, ev in df_seg.iterrows():
            s, e = ev["start_time"], ev["end_time"]
            seg = residual[s:e]
            # EVENTO CONCURRENTE (superposición de >=2 fixtures). ATRIBUCIÓN PARCIAL:
            # el caudal BASE continuo (el fixture dominante, p.ej. una ducha) se atribuye
            # a su categoría, y solo el EXCESO por encima de la base (la joroba del
            # segundo uso) va a "Uso simultáneo". Antes se mandaba el evento ENTERO a
            # "Uso simultáneo", así una ducha con una descarga breve encima volcaba TODA
            # su agua a simultáneo (inflaba esa categoría a ~47%). Conserva masa
            # (base+exceso=caudal) y hace que un falso positivo del detector solo filtre
            # una porción chica, no la ducha completa.
            if bool(ev.get("is_composite", False)):
                vals = np.asarray(seg.values, dtype=float)
                pos = vals[vals > 0]
                # base = MEDIANA (robusta a la joroba breve del segundo fixture), no p20
                # (que fugaba toda la variación normal del fixture base a "simultáneo").
                base_level = float(np.median(pos)) if pos.size else 0.0
                enb = _normalize(np.array([base_level]), np.array([ev["duration_s"]]))[0]
                db = np.linalg.norm(cent_norm - enb, axis=1)
                jb = int(np.argmin(db))
                base_ok = base_level > 0 and (tol_nd[jb] <= 0 or db[jb] <= tol_nd[jb])
                if base_ok:
                    excess = np.maximum(vals - base_level, 0.0)
                    # solo el exceso SOSTENIDO (joroba real) es "Uso simultáneo"; el
                    # ruido sobre la base vuelve al dominante -> sin fragmentación.
                    hump = _sustained_excess(excess, seg.index)
                    df_result.loc[s:e, labels[jb]] += vals - hump
                    df_result.loc[s:e, COMPOSITE] += hump
                else:
                    df_result.loc[s:e, COMPOSITE] += vals
                residual[s:e] = 0.0
                n_comp += 1
                continue
            # Clasificar el evento por su perfil 2D (caudal, duración) más cercano.
            en = _normalize(np.array([ev["mean_flow"]]), np.array([ev["duration_s"]]))[0]
            d = np.linalg.norm(cent_norm - en, axis=1)
            j = int(np.argmin(d))
            # RECHAZO por tolerancia 2D (tol_nd): si el evento cae fuera del radio
            # del perfil más cercano, el modelo dice "no sé" y el caudal queda en
            # No Detectado en vez de forzar una atribución. Antes tol_nd se
            # calculaba y persistía pero NO se usaba (todo evento se atribuía).
            if tol_nd[j] > 0 and d[j] > tol_nd[j]:
                n_rej += 1
                continue
            # RECHAZO por RANGO FÍSICO de volumen: un evento cuyo volumen es
            # implausible para la etiqueta asignada (p.ej. un "Inodoro" de 0.05 L) va
            # a No Detectado en vez de contaminar el cluster con física imposible.
            ev_vol = float(ev.get("volume_liters", 0.0) or 0.0)
            vmin, vmax = VOLUME_RANGE_BY_LABEL.get(labels[j], (0.0, float("inf")))
            if not (vmin <= ev_vol <= vmax):
                n_rej += 1
                continue
            df_result.loc[s:e, labels[j]] += seg.values
            residual[s:e] = 0.0
            n_ok += 1

    df_result["No Detectado"] = residual
    print(f"[RunSimple] eventos asignados={n_ok}/{len(df_seg)} "
          f"(concurrentes→Uso simultáneo={n_comp}, rechazados→No Detectado={n_rej})")
    # COMPOSITE entra como device para que sus eventos se emitan; si no hubo
    # concurrencia la columna es 0 y _build_events no genera nada para ella.
    return _build_events(df_result, out_cols + [COMPOSITE]), df_result


def _build_events(df_result: pd.DataFrame, device_names: List[str]) -> pd.DataFrame:
    rows = []
    for col in device_names:
        is_active = df_result[col] > 0
        changes = is_active.astype(int).diff()
        starts = df_result.index[changes == 1]
        ends = df_result.index[changes == -1]
        if len(is_active) > 0 and bool(is_active.iloc[0]):
            starts = starts.insert(0, df_result.index[0])
        if len(ends) < len(starts):
            ends = ends.append(pd.Index([df_result.index[-1]]))
        for s, e in zip(starts, ends):
            seg = df_result.loc[s:e, col]
            duration = (e - s).total_seconds()
            avg = float(seg.mean())
            vol = integrate_volume(seg)   # litros con Δt real (antes seg.sum()/60)
            if not is_valid_event(avg, duration, vol):
                continue
            rows.append({
                "device": col,
                "start_time": s,
                "end_time": e,
                "duration_seconds": float(duration),
                "flow_rate": avg,
                "volume_liters": vol,
            })
    return pd.DataFrame(rows)
