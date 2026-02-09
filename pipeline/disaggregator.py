import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import matplotlib.dates as mdates
from io import BytesIO

def preprocess_signal(df):
    df["flow_smooth"] = df["flow"].rolling(window=5, center=True).median().fillna(df["flow"])
    df["delta"] = df["flow_smooth"].diff().fillna(0)
    return df

def detect_k_optimal(data, max_k=10):
    bic_scores = []
    k_range = range(1, min(max_k + 1, len(data)))

    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(data)
        bic_scores.append(gmm.bic(data))

    best_k = k_range[np.argmin(bic_scores)]
    return best_k

def learn_fixtures(df):
    on_devices = df[df["delta"] > 0.3]["delta"].values.reshape(-1, 1)

    if len(on_devices) < 5:
        return {}

    n_components = detect_k_optimal(on_devices)
    print(f"K óptimo detectado: {n_components}")

    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(on_devices)

    means = gmm.means_.flatten()
    sorted_means = np.sort(means)

    profiles = {}
    for i, m in enumerate(sorted_means):
        name = f"Aparato_{i+1}"
        # Todo: Cambiar valores dependiendo de datos reales
        profiles[i] = {
            "name": name,
            "mean": m,
            "tol": max(0.2, m * 0.25)
            }
    return profiles

def disaggregate_events(df, profiles):
    events_log = []
    active_devices = {k: [] for k in profiles}

    if not profiles:
        return pd.DataFrame()

    min_threshold = min([p['mean'] for p in profiles.values()]) * 0.5
    significant_changes = df[df['delta'].abs() > min_threshold]

    for time, row in significant_changes.iterrows():
        delta = row["delta"]

        for dev_id, start_times in active_devices.items():
            if not start_times:
                continue
            
            # CAMBIO: Determinar duración máxima basada en el flujo promedio (física)
            # en lugar del nombre (etiqueta).
            mean_flow = profiles[dev_id]["mean"]
            
            # Reglas heurísticas (puedes ajustarlas):
            if mean_flow < 2.0:
                max_duration = 600   # ~10 min (ej. grifo lavamanos)
            elif mean_flow < 5.0:
                max_duration = 7200  # ~2 horas (ej. lavadora ciclo largo)
            else:
                max_duration = 3600  # ~1 hora (ej. ducha larga)

            last_start = start_times[-1]
            current_duration = (time - last_start).total_seconds()

            if current_duration > max_duration:
                start_times.pop()
                print(f"Evento zombie eliminado: {profiles[dev_id]['name']} excedió {current_duration}s")

        matched_id = None
        for dev_id, prof in profiles.items():
            if abs(abs(delta) - prof["mean"]) < prof["tol"]:
                matched_id = dev_id
                break

        if matched_id is None:
            continue

        if delta > 0:
            active_devices[matched_id].append(time)

        elif delta < 0:
            if active_devices[matched_id]:
                start_time = active_devices[matched_id].pop()
                duration = (time - start_time).total_seconds()
                
                # Usar la misma lógica de max_duration para validar el evento final
                mean_flow = profiles[matched_id]["mean"]
                if mean_flow < 2.0:
                    max_valid_duration = 600
                else:
                    max_valid_duration = 7200 # Ser más permisivo al cerrar eventos

                if duration < max_valid_duration:
                    events_log.append({
                        "Device": profiles[matched_id]["name"],
                        "Start": start_time,
                        "End": time,
                        "Duration_s": duration,
                        "Flow_L_min": profiles[matched_id]["mean"]
                    })
    
    # Procesar eventos que quedaron abiertos al final de los datos
    end_of_data = df.index[-1]
    for dev_id, start_times in active_devices.items():
        while start_times:
            start_time = start_times.pop()
            duration = (end_of_data - start_time).total_seconds()
            
            # Validación simple para eventos finales
            if duration < 3600: 
                events_log.append({
                    "Device": profiles[dev_id]["name"],
                    "Start": start_time,
                    "End": end_of_data,
                    "Duration_s": duration,
                    "Flow_L_min": profiles[dev_id]["mean"]
                })
                
    return pd.DataFrame(events_log)

def rebuild_consumption(df_original, df_events, profiles):
    df_result = df_original[["flow"]].copy()

    for prof in profiles.values():
        df_result[prof["name"]] = 0.0

    if df_events.empty:
        return df_result

    for _, event in df_events.iterrows():
        mask = (df_result.index >= event["Start"]) & (df_result.index < event["End"])
        df_result.loc[mask, event["Device"]] += event["Flow_L_min"]
    return df_result

def generate_stackplot(df, start_time=None, end_time=None):
    if start_time is not None and end_time is not None:
        try:
            df_plot = df.loc[start_time:end_time].copy()
            extra_title = f" ({start_time} - {end_time})"
        except KeyError:
            return
    else:
        df_plot = df.copy()
        extra_title = f" (Total)"

    if df_plot.empty:
        return
    tec_cols = ['flow', 'delta', 'flow_smooth', 'Total_Real', 'Total_Medido']
    fixture_cols = [c for c in df_plot.columns if c not in tec_cols]

    if not fixture_cols:
        return
    
    plt.figure(figsize=(15, 8))

    y_stack = [df_plot[col] for col in fixture_cols]
    colors = plt.cm.Spectral(np.linspace(0, 1, len(fixture_cols)))

    plt.stackplot(df_plot.index, y_stack,
                  labels=fixture_cols,
                  colors=colors,
                  alpha=0.85,
                  edgecolor='white',
                  linewidth=0.5)

    if 'flow' in df_plot.columns:
        plt.plot(df_plot.index, df_plot['flow'], color='black', linewidth=2,
                 linestyle='--', label='Sensor Total')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.xticks(rotation=0, fontsize=10, fontweight='bold')

    plt.title(f"Desagregación de Consumo{extra_title}", fontsize=16)
    plt.ylabel("Caudal (L/min)", fontsize=12)
    plt.xlabel("Tiempo", fontsize=12)
    plt.legend(loc='upper left', framealpha=0.9, shadow=True)
    plt.grid(True, alpha=0.3, linestyle=':')
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    return buf

def run_disaggregation(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    df = df.copy()

    if "flow" not in df.columns:
        raise ValueError("Input dataframe must contain 'flow' column")

    df = preprocess_signal(df)
    profiles = learn_fixtures(df)

    if not profiles:
        return pd.DataFrame(), pd.DataFrame(), {}

    df_events = disaggregate_events(df, profiles)
    df_result = rebuild_consumption(df, df_events, profiles)

    return df_events, df_result, profiles