import io
import math
from typing import Tuple
import numpy as np
import pandas as pd
import streamlit as st
import gpxpy
import gpxpy.gpx
import matplotlib.pyplot as plt
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

def plot_segment_overlay(df_full: pd.DataFrame, km0: float, km1: float, title: str, filename: str):
    """Trace le profil complet en gris + le tronçon [km0, km1] en orange gras."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_full["dist_km_cal"], df_full["ele_smooth"], linewidth=1.0, color="0.7")
    mask = (df_full["dist_km_cal"] >= km0) & (df_full["dist_km_cal"] <= km1)
    ax.plot(df_full.loc[mask, "dist_km_cal"], df_full.loc[mask, "ele_smooth"], linewidth=3.0, color="orange")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Altitude (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, use_container_width=True)
    fig_to_png_download_button(fig, "⬇️ Télécharger ce segment (PNG)", filename)




# ---------- Utils géodésiques ----------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

# 15 avril----
def distance_3d(lat1, lon1, ele1, lat2, lon2, ele2):
    """Distance 3D = distance horizontale + delta altitude."""
    d2d = haversine(lat1, lon1, lat2, lon2)
    if pd.isna(ele1) or pd.isna(ele2):
        return d2d
    dz = float(ele2) - float(ele1)
    return math.sqrt(d2d**2 + dz**2)

# 15 avril----

def smooth_series(y: np.ndarray, fenetre_m: int, x_m: np.ndarray) -> np.ndarray:
    if fenetre_m <= 1:
        return y.copy()
    n = len(y)
    out = np.zeros(n, dtype=float)
    for i in range(n):
        x0 = x_m[i] - fenetre_m/2
        x1 = x_m[i] + fenetre_m/2
        mask = (x_m >= x0) & (x_m <= x1)
        out[i] = np.nanmean(y[mask]) if np.any(mask) else y[i]
    out[np.isnan(out)] = y[np.isnan(out)]
    return out

# 15 avril----
def recompute_distances_with_smoothed_elevation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recalcule :
    - dist_m_horiz / dist_km_horiz : distance horizontale pure
    - dist_m / dist_km             : distance 3D basée sur ele_smooth
    """
    if df.empty or len(df) < 2:
        df["dist_m_horiz"] = df.get("dist_m", 0.0)
        df["dist_km_horiz"] = df["dist_m_horiz"] / 1000.0
        return df

    # Sauvegarde de la distance horizontale d'origine
    df["dist_m_horiz"] = df["dist_m"].copy()
    df["dist_km_horiz"] = df["dist_m_horiz"] / 1000.0

    dist3d_steps = [0.0]
    for i in range(1, len(df)):
        d = distance_3d(
            df.lat.iloc[i-1], df.lon.iloc[i-1], df.ele_smooth.iloc[i-1],
            df.lat.iloc[i],   df.lon.iloc[i],   df.ele_smooth.iloc[i]
        )
        dist3d_steps.append(d)

    df["dist_m"] = np.cumsum(dist3d_steps)
    df["dist_km"] = df["dist_m"] / 1000.0
    return df

def calibrate_distance_to_target(df: pd.DataFrame, target_km: float) -> pd.DataFrame:
    """
    Conserve la distance GPS brute et crée une distance recalée qui finit exactement à target_km.
    """
    if df.empty:
        return df

    measured_km = float(df["dist_km"].iloc[-1])
    if measured_km <= 0:
        df["dist_m_gps"] = df["dist_m"].copy()
        df["dist_km_gps"] = df["dist_km"].copy()
        df["dist_m_cal"] = df["dist_m"].copy()
        df["dist_km_cal"] = df["dist_km"].copy()
        df["dist_coef"] = 1.0
        return df

    coef = target_km / measured_km

    df["dist_m_gps"] = df["dist_m"].copy()
    df["dist_km_gps"] = df["dist_km"].copy()

    df["dist_m_cal"] = df["dist_m_gps"] * coef
    df["dist_km_cal"] = df["dist_m_cal"] / 1000.0
    df["dist_coef"] = coef

    return df

# 15 avril----


# ---------- Parsing GPX -> DataFrame ----------
def gpx_to_df(gpx_obj) -> pd.DataFrame:
    rows = []
    for track in gpx_obj.tracks:
        for seg in track.segments:
            for p in seg.points:
                rows.append(dict(lat=p.latitude, lon=p.longitude, ele=p.elevation, time=p.time))
    for route in gpx_obj.routes:
        for p in route.points:
            rows.append(dict(lat=p.latitude, lon=p.longitude, ele=p.elevation, time=p.time))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # time en naïf
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)

    # distance cumulée
    dist = [0.0]
    for i in range(1, len(df)):
        dist.append(haversine(df.lat.iloc[i-1], df.lon.iloc[i-1], df.lat.iloc[i], df.lon.iloc[i]))
    df["dist_m"] = np.cumsum(dist)
    df["dist_km"] = df["dist_m"] / 1000.0

    # vitesse si temps dispo
    if df["time"].notna().all():
        dt_s = [0.0]
        for i in range(1, len(df)):
            dt_s.append((df.time.iloc[i] - df.time.iloc[i-1]).total_seconds())
        df["dt_s"] = dt_s
        v_ms = np.divide(np.r_[np.nan, np.diff(df["dist_m"])], df["dt_s"], out=np.zeros_like(df["dist_m"]), where=np.array(df["dt_s"])!=0)
        df["v_kmh"] = v_ms * 3.6
    else:
        df["dt_s"] = np.nan
        df["v_kmh"] = np.nan

    return df


# ---------- Segments remarquables (avec tolérance) ----------
def detect_segments_remarquables(
    df: pd.DataFrame,
    pente_min_pct: float,
    deniv_min_m: float,
    dist_min_m: float,
    type_segment: str = "montee",
    tolerance_pct_opposite: float = 10.0,
) -> pd.DataFrame:
    """
    Détecte des segments remarquables en autorisant un % de dénivelé opposé.
    - type_segment: "montee" ou "descente"
    - tolerance_pct_opposite: ex. en montée, % de D- autorisé par rapport au D+ du segment
    Retourne : km_debut, km_fin, longueur_m, denivele_m (net), pente_moy_pct,
               dplus_brut, dminus_brut, oppose_observe_pct
    """
    if df.empty or df["ele"].isna().all():
        return pd.DataFrame(columns=[
            "km_debut","km_fin","longueur_m","denivele_m","pente_moy_pct",
            "dplus_brut","dminus_brut","oppose_observe_pct"
        ])

    # Vecteurs
    dist_m = df["dist_m_cal"].to_numpy()
    dist_km = df["dist_km_cal"].to_numpy()
    ele = df["ele"].to_numpy().astype(float)

    dd = np.r_[np.nan, np.diff(dist_m)]        # m
    de = np.r_[np.nan, np.diff(ele)]           # m
    pente_pct = np.where(dd > 0, (de/dd)*100.0, np.nan)

    segments = []
    n = len(df)
    i = 1  # commence au point 1 (diff dispo)

    def start_condition(idx):
        if np.isnan(pente_pct[idx]):
            return False
        if type_segment == "montee":
            return pente_pct[idx] >= pente_min_pct
        else:
            return pente_pct[idx] <= -pente_min_pct

    while i < n:
        while i < n and not start_condition(i):
            i += 1
        if i >= n:
            break

        start = i - 1 if i > 0 else i
        j = i
        cum_d = 0.0
        cum_pos = 0.0  # D+ brut cumulé
        cum_neg = 0.0  # D- brut cumulé (positivé)

        while j < n:
            if np.isnan(dd[j]) or dd[j] <= 0:
                j += 1
                continue

            step_d = dd[j]
            step_de = de[j]

            # cumuls
            cum_d += step_d
            if step_de >= 0:
                cum_pos += step_de
            else:
                cum_neg += -step_de

            # évalue cohérence + tolérance
            if type_segment == "montee":
                main = cum_pos
                oppo = cum_neg
                net = cum_pos - cum_neg
            else:
                main = cum_neg
                oppo = cum_pos
                net = cum_neg - cum_pos

            tol_ok = (main <= 0) or (oppo <= (tolerance_pct_opposite / 100.0) * main + 1e-9)
            still_valid = (net > 0) and tol_ok

            if not still_valid:
                # annule le dernier pas qui fait sortir de la tolérance
                cum_d -= step_d
                if step_de >= 0:
                    cum_pos -= step_de
                else:
                    cum_neg -= -step_de
                j -= 1
                break

            j += 1

        if j >= n:
            j = n - 1

        # Minimas
        if cum_d >= dist_min_m:
            if type_segment == "montee":
                net = cum_pos - cum_neg
            else:
                net = cum_neg - cum_pos

            if net >= deniv_min_m:
                pente_moy = (net / cum_d) * 100.0 if cum_d > 0 else 0.0
                oppose_pct_obs = (oppo / main * 100.0) if main > 0 else 0.0
                segments.append({
                    "km_debut": dist_km[start],
                    "km_fin": dist_km[j],
                    "longueur_m": cum_d,
                    "denivele_m": net,
                    "pente_moy_pct": pente_moy,
                    "dplus_brut": cum_pos,
                    "dminus_brut": cum_neg,
                    "oppose_observe_pct": oppose_pct_obs
                })

        i = max(j + 1, i + 1)

    return pd.DataFrame(segments)


# ---------- Exports ----------
def fig_to_png_download_button(fig, label: str, filename: str):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    st.download_button(label=label, data=buf.getvalue(), file_name=filename, mime="image/png")


def df_to_csv_download_button(df: pd.DataFrame, label: str, filename: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")


# ---------- Aide tronçon ----------
def compute_troncon_metrics(df_seg: pd.DataFrame) -> Tuple[float, float, float]:
    if df_seg.empty:
        return 0.0, 0.0, 0.0
    dist_km = float(df_seg["dist_km_cal"].iloc[-1] - df_seg["dist_km_cal"].iloc[0])
    de = np.r_[0.0, np.diff(df_seg["ele_smooth"])]
    dplus = float(np.sum(de[de > 0]))
    dminus = float(-np.sum(de[de < 0]))
    return dist_km, dplus, dminus


def slope_distribution_by_5pct(df_seg: pd.DataFrame) -> pd.DataFrame:
    """Calcule la distance (m) par classes de pente de 5% dans le tronçon."""
    if len(df_seg) < 2:
        return pd.DataFrame(columns=["Classe pente (%)","Distance (m)"])
    dd = np.r_[np.nan, np.diff(df_seg["dist_m"])]
    de = np.r_[np.nan, np.diff(df_seg["ele_smooth"])]
    pente = np.where(dd > 0, (de/dd) * 100.0, np.nan)

    # On associe la distance "entre i-1 et i" à la pente[i]
    dist_step = dd
    mask = ~np.isnan(pente) & ~np.isnan(dist_step) & (dist_step > 0)
    pente = pente[mask]; dist_step = dist_step[mask]

    # bornes de classes : de -40 à +40 par pas de 5, avec bords "moins de" et "plus de"
    edges = np.arange(-40, 45, 5)  # [-40,-35,...,40]
    labels = [f"{int(a)} à {int(b)}" for a, b in zip(edges[:-1], edges[1:])]
    labels = [f"< -40"] + labels + [f"> 40"]

    bins = np.r_[-np.inf, edges, np.inf]
    idx = np.digitize(pente, bins) - 1
    dist_by_bin = pd.Series(0.0, index=range(len(labels)))
    for i_bin, d in zip(idx, dist_step):
        if 0 <= i_bin < len(labels):
            dist_by_bin.iloc[i_bin] += d

    out = pd.DataFrame({
        "Classe pente (%)": labels,
        "Distance (m)": dist_by_bin.values
    })
    out["Distance (km)"] = out["Distance (m)"] / 1000.0
    out["Part (%)"] = 100.0 * out["Distance (m)"] / out["Distance (m)"].sum() if out["Distance (m)"].sum() > 0 else 0.0
    return out

    
def slope_distribution_by_5pct_filtered(df_seg: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    Répartition distance par classes de pente de 5%,
    filtrée sur 'up' (pentes >0) ou 'down' (pentes <0).
    """
    if len(df_seg) < 2:
        return pd.DataFrame(columns=["Classe pente (%)", "Distance (km)", "Part (%)"])

    # Calcul des différences de distance et d'altitude
    dd = np.r_[np.nan, np.diff(df_seg["dist_m"])]
    de = np.r_[np.nan, np.diff(df_seg["ele_smooth"])]
    pente = np.where(dd > 0, (de / dd) * 100.0, np.nan)
    dist_step = dd

    # Nettoyage des NaN
    mask = ~np.isnan(pente) & ~np.isnan(dist_step) & (dist_step > 0)
    pente = pente[mask]
    dist_step = dist_step[mask]

    # Définition des bornes de classes
    edges = np.arange(-40, 45, 5)
    bins = np.r_[-np.inf, edges, np.inf]

    # Filtrage selon le mode
    if mode == "up":
        sel = pente > 0
    elif mode == "down":
        sel = pente < 0
    else:
        sel = np.ones_like(pente, dtype=bool)

    pente = pente[sel]
    dist_step = dist_step[sel]

    if pente.size == 0:
        return pd.DataFrame(columns=["Classe pente (%)", "Distance (km)", "Part (%)"])


def filter_distribution_by_sign(dist_df: pd.DataFrame, sign: str) -> pd.DataFrame:
    """
    Filtre un tableau de répartition (issu de slope_distribution_by_5pct) en gardant
    uniquement les classes de pente positives ('up') ou négatives ('down').
    On parse les labels 'Classe pente (%)' du style 'a à b', '< -40', '> 40'.
    """
    if dist_df is None or dist_df.empty:
        return pd.DataFrame(columns=["Classe pente (%)", "Distance (km)", "Part (%)"])

    def lower_bound(label: str) -> float:
        label = str(label).strip()
        if label.startswith("<"):
            # ex: "< -40" -> borne basse très négative
            return -1e9
        if label.startswith(">"):
            # ex: "> 40" -> borne basse strictement positive
            # on retourne la borne basse du dernier intervalle (ici 40)
            try:
                val = float(label.split(">")[1].strip())
            except Exception:
                val = 0.0
            return val
        # cas "a à b"
        try:
            a = label.split("à")[0].strip()
            return float(a)
        except Exception:
            return 0.0

    lb = dist_df["Classe pente (%)"].map(lower_bound)

    if sign == "up":
        mask = lb >= 0  # on garde 0–5, 5–10, ..., et "> 40"
    elif sign == "down":
        mask = lb < 0   # on garde "< -40", -40–-35, ..., -5–0
    else:
        mask = np.ones(len(dist_df), dtype=bool)

    out = dist_df.loc[mask].copy()
    # Mise en forme demandée : Distance (km) à 1 décimale, Part (%) avec '%'
    if "Distance (km)" in out.columns:
        out["Distance (km)"] = out["Distance (km)"].round(1)
    if "Part (%)" in out.columns and out["Part (%)"].dtype != object:
        out["Part (%)"] = out["Part (%)"].round(2).astype(str) + "%"

    # Retire les lignes vides éventuelles
    if "Distance (km)" in out.columns:
        out = out[out["Distance (km)"] > 0]
    return out.reset_index(drop=True)


def plot_distribution_bar(dist_df: pd.DataFrame, title: str, filename: str):
    """Bar chart de la répartition Distance (km) par classes de pente."""
    if dist_df is None or dist_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(dist_df["Classe pente (%)"], dist_df["Distance (km)"])
    ax.set_xlabel("Classe de pente (%)")
    ax.set_ylabel("Distance (km)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    st.pyplot(fig, use_container_width=True)
    fig_to_png_download_button(fig, "⬇️ Télécharger l'histogramme (PNG)", filename)

# ---------- HELPER POUR DEFINIR UN KML -------------
def df_segment_to_kml_bytes(
    df_seg: pd.DataFrame,
    name: str,
    use_smoothed_elev: bool = True,
    altitude_bias_m: float = 0.0,
    altitude_mode: str = "absolute"  # 👈 nouveau paramètre: "absolute" | "relativeToGround"
) -> bytes:
    """
    KML simple (ligne jaune) avec biais d'altitude et mode absolute/relativeToGround.
    """
    if df_seg is None or df_seg.empty:
        return b""

    elev_col = "ele_smooth" if (use_smoothed_elev and "ele_smooth" in df_seg.columns) else "ele"

    coords = []
    for _, row in df_seg.iterrows():
        # altitude selon le mode retenu
        if altitude_mode == "relativeToGround":
            # au-dessus du terrain de +biais (pas d'altitude GPX utilisée)
            alt = float(altitude_bias_m)
        else:
            ele_val = None
            if elev_col in df_seg.columns and not pd.isna(row[elev_col]):
                ele_val = float(row[elev_col])
            alt = (ele_val if ele_val is not None else 0.0) + float(altitude_bias_m)

        coords.append(f"{float(row['lon']):.6f},{float(row['lat']):.6f},{alt:.2f}")

    coord_str = " ".join(coords)
    alt_mode_tag = "relativeToGround" if altitude_mode == "relativeToGround" else "absolute"

    # Jaune en ABGR (KML): ff00ffff
    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{name}</name>
    <Placemark>
      <name>{name}</name>
      <Style>
        <LineStyle><color>ff00ffff</color><width>10</width></LineStyle>
      </Style>
      <LineString>
        <tessellate>1</tessellate>
        <altitudeMode>{alt_mode_tag}</altitudeMode>
        <coordinates>{coord_str}</coordinates>
      </LineString>
    </Placemark>
  </Document>
</kml>"""
    return kml.encode("utf-8")


def kml_export_block(
    df_seg: pd.DataFrame,
    name: str,
    key_prefix: str,
    altitude_bias_m: float,
    color_by_slope: bool,
    palette_abs_max: float,
    add_km_waypoints: bool,
    df_full_for_km: pd.DataFrame = None,
    km_start: float = None,
    km_end: float = None,
    altitude_mode: str = "absolute",   # "absolute" | "relativeToGround"
    min_run_m: float = 500.0           # longueur mini d’un tronçon coloré
):
    """Bloc UI export KML. key_prefix doit être unique ('troncon', 'montee', 'descente')."""
    if df_seg is None or df_seg.empty:
        return

    use_smooth = st.checkbox(
        "Utiliser l'altitude lissée pour le KML",
        value=True,
        key=f"kml_use_smooth_{key_prefix}"
    )

    if color_by_slope:
        # 👉 KML coloré par pente
        kml_bytes = df_segment_to_kml_colore_par_pente_bucketise_bytes(
            df_seg,
            name=name,
            use_smoothed_elev=use_smooth,
            altitude_bias_m=altitude_bias_m,
            palette_abs_max=palette_abs_max,
            altitude_mode=altitude_mode,
            bucket_m=min_run_m
        )
        label = "⬇️ Exporter KML coloré par pente"
        fname = f"{key_prefix}_{name.replace(' ', '_').replace('–','-')}_pente.kml"
    else:
        # 👉 KML jaune simple
        kml_bytes = df_segment_to_kml_bytes(
            df_seg,
            name=name,
            use_smoothed_elev=use_smooth,
            altitude_bias_m=altitude_bias_m,
            altitude_mode=altitude_mode
        )
        label = f"⬇️ Exporter KML (jaune, +{altitude_bias_m:.1f} m)"
        fname = f"{key_prefix}_{name.replace(' ', '_').replace('–','-')}_jaune.kml"

    st.download_button(
        label=label,
        data=kml_bytes,
        file_name=fname,
        mime="application/vnd.google-earth.kml+xml",
        key=f"kml_download_{key_prefix}"
    )

    # Waypoints km optionnels
    if add_km_waypoints and (df_full_for_km is not None) and (km_start is not None) and (km_end is not None):
        km_bytes = kml_km_markers_bytes(
            df_full=df_full_for_km,
            start_km=km_start, end_km=km_end,
            use_smoothed_elev=True,                     # on met les KM sur le profil lissé
            altitude_bias_m=altitude_bias_m,
            name=f"KM {name}"
        )
        st.download_button(
            "⬇️ Exporter KML des KM",
            data=km_bytes,
            file_name=f"{key_prefix}_KM_{name.replace(' ', '_').replace('–','-')}.kml",
            mime="application/vnd.google-earth.kml+xml",
            key=f"kml_km_download_{key_prefix}"
        )


def _rgb_to_abgr_hex(r: int, g: int, b: int, alpha: int = 255) -> str:
    r = max(0, min(255, r)); g = max(0, min(255, g)); b = max(0, min(255, b))
    a = max(0, min(255, alpha))
    return f"{a:02x}{b:02x}{g:02x}{r:02x}"  # KML = ABGR

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def kml_color_from_slope_continuous(pct: float, smax: float) -> str:
    """
    Palette continue Rouge → Jaune → Vert centrée sur 0% :
      p<=-smax → Vert (#00FF00)
      p=0 → Jaune (#FFFF00)
      p>=+smax → Rouge (#FF0000)
    """
    if smax <= 0: smax = 30.0
    p = max(-smax, min(smax, pct))
    if p >= 0:
        t = p / smax
        r, g, b = 255, int(_lerp(255, 0, t)), 0        # Jaune -> Rouge
    else:
        t = (-p) / smax
        r, g, b = int(_lerp(255, 0, t)), 255, 0        # Jaune -> Vert
    return _rgb_to_abgr_hex(r, g, b, 255)

#---------------------------------------------------------------

def df_segment_to_kml_colore_par_pente_bucketise_bytes(
    df_seg: pd.DataFrame,
    name: str,
    use_smoothed_elev: bool = True,
    altitude_bias_m: float = 0.0,
    palette_abs_max: float = 30.0,
    altitude_mode: str = "absolute",   # "absolute" | "relativeToGround"
    bucket_m: float = 500.0            # taille de bloc (m)
) -> bytes:
    """
    KML coloré par pente moyenne, découpé en blocs réguliers de 'bucket_m' mètres.
    Chaque bloc a une couleur selon sa pente moyenne:
      Rouge (>= +palette_abs_max) ↔ Jaune (0) ↔ Vert (<= -palette_abs_max)
    """
    if df_seg is None or df_seg.empty or len(df_seg) < 2:
        return b""

    elev_col = "ele_smooth" if (use_smoothed_elev and "ele_smooth" in df_seg.columns) else "ele"
    x_m = df_seg["dist_m"].to_numpy(float)
    lat = df_seg["lat"].to_numpy(float)
    lon = df_seg["lon"].to_numpy(float)
    ele = df_seg[elev_col].to_numpy(float)

    # bornes
    start_m = float(x_m[0]); end_m = float(x_m[-1])
    if bucket_m <= 0: bucket_m = 500.0

    # utilitaires
    def color_for_pcent(pct: float) -> str:
        smax = max(1e-6, float(palette_abs_max))
        p = max(-smax, min(smax, float(pct)))
        if p >= 0:
            t = p / smax
            r, g, b = 255, int(255*(1-t)), 0      # Jaune -> Rouge
        else:
            t = (-p) / smax
            r, g, b = int(255*(1-t)), 255, 0      # Jaune -> Vert
        return f"ff{b:02x}{g:02x}{r:02x}"         # ABGR KML

    def interp_at(dist_m_target: float):
        # Interpole lat/lon/ele aux abscisses métriques demandées
        lat_i = float(np.interp(dist_m_target, x_m, lat))
        lon_i = float(np.interp(dist_m_target, x_m, lon))
        ele_i = float(np.interp(dist_m_target, x_m, ele)) if not np.all(np.isnan(ele)) else 0.0
        return lon_i, lat_i, ele_i

    # Découpage en pas réguliers : [b0->b1], [b1->b2], ...
    edges = np.arange(start_m, end_m + bucket_m, bucket_m)
    if edges[-1] < end_m + 1e-6:
        edges = np.r_[edges, end_m]

    runs = []  # (color, coords list, km_a, km_b)
    for i in range(len(edges)-1):
        a = float(edges[i]); b = float(edges[i+1])
        if b - a < 1e-6:
            continue
        # Interpole bornes
        lon_a, lat_a, ele_a = interp_at(a)
        lon_b, lat_b, ele_b = interp_at(b)

        # Pente moyenne du bloc
        delta_m = b - a
        pcent = ((ele_b - ele_a) / delta_m) * 100.0 if delta_m > 0 else 0.0
        col = color_for_pcent(pcent)

        # Extraire coordonnées internes du bloc pour un tracé plus fidèle
        mask = (x_m >= a) & (x_m <= b)
        lons = lon[mask]; lats = lat[mask]; eles = ele[mask]

        coords = []
        # début
        if altitude_mode == "relativeToGround":
            alt_a = altitude_bias_m
        else:
            alt_a = (ele_a if not np.isnan(ele_a) else 0.0) + altitude_bias_m
        coords.append(f"{lon_a:.6f},{lat_a:.6f},{alt_a:.2f}")

        # points internes
        for j in range(len(lons)):
            if altitude_mode == "relativeToGround":
                alt_j = altitude_bias_m
            else:
                ej = eles[j] if not np.isnan(eles[j]) else 0.0
                alt_j = ej + altitude_bias_m
            coords.append(f"{float(lons[j]):.6f},{float(lats[j]):.6f},{alt_j:.2f}")

        # fin
        if altitude_mode == "relativeToGround":
            alt_b = altitude_bias_m
        else:
            alt_b = (ele_b if not np.isnan(ele_b) else 0.0) + altitude_bias_m
        coords.append(f"{lon_b:.6f},{lat_b:.6f},{alt_b:.2f}")

        km_a = a/1000.0; km_b = b/1000.0
        runs.append((col, coords, km_a, km_b))

    if not runs:
        return b""

    # Styles (un par couleur)
    unique_colors = sorted(set(col for col, _, _, _ in runs))
    styles_xml = "\n".join(
        f"""    <Style id="line_{idx}">
      <LineStyle><color>{col}</color><width>9</width></LineStyle>
    </Style>""" for idx, col in enumerate(unique_colors)
    )
    color_to_style = {c: f"line_{i}" for i, c in enumerate(unique_colors)}
    alt_mode_tag = "relativeToGround" if altitude_mode == "relativeToGround" else "absolute"

    placemarks = []
    for idx, (col, coords, km_a, km_b) in enumerate(runs, start=1):
        if len(coords) < 2:
            continue
        placemarks.append(f"""    <Placemark>
      <name>{name} — {km_a:.1f}–{km_b:.1f} km</name>
      <styleUrl>#{color_to_style[col]}</styleUrl>
      <LineString>
        <tessellate>1</tessellate>
        <altitudeMode>{alt_mode_tag}</altitudeMode>
        <coordinates>{' '.join(coords)}</coordinates>
      </LineString>
    </Placemark>""")

    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{name} (pente colorée, blocs {int(bucket_m)} m)</name>
{styles_xml}
{chr(10).join(placemarks)}
  </Document>
</kml>"""
    return kml.encode("utf-8")



#---------------------------------------------

def kml_km_markers_bytes(
    df_full: pd.DataFrame,
    start_km: float,
    end_km: float,
    use_smoothed_elev: bool = True,
    altitude_bias_m: float = 0.0,
    name: str = "Marqueurs km"
) -> bytes:
    if df_full is None or df_full.empty:
        return b""
    elev_col = "ele_smooth" if (use_smoothed_elev and "ele_smooth" in df_full.columns) else "ele"

    x = df_full["dist_km_cal"].to_numpy(float)
    lat = df_full["lat"].to_numpy(float)
    lon = df_full["lon"].to_numpy(float)
    ele = df_full[elev_col].to_numpy(float)

    k0 = int(np.ceil(start_km))
    k1 = int(np.floor(end_km))
    if k1 < k0:
        return b""

    placemarks = []
    for k in range(k0, k1 + 1):
        lat_k = float(np.interp(k, x, lat))
        lon_k = float(np.interp(k, x, lon))
        ele_k = float(np.interp(k, x, ele)) if not np.all(np.isnan(ele)) else 0.0
        alt = ele_k + altitude_bias_m
        placemarks.append(f"""    <Placemark>
      <name>KM {k}</name>
      <Style>
        <IconStyle><color>ff00ffff</color><scale>1.1</scale></IconStyle>
        <LabelStyle><scale>0.9</scale></LabelStyle>
      </Style>
      <Point>
        <altitudeMode>absolute</altitudeMode>
        <coordinates>{lon_k:.6f},{lat_k:.6f},{alt:.2f}</coordinates>
      </Point>
    </Placemark>""")

    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{name}</name>
{chr(10).join(placemarks)}
  </Document>
</kml>"""
    return kml.encode("utf-8")



#----------------------------------------------------------
# ---------- App ------------------------------------------
#----------------------------------------------------------
st.set_page_config(page_title="Analyse GPX", layout="wide")

st.title("📈 Analyse de trace GPX (FR)")
st.caption("Slider de tronçon, métriques locales, répartition de pente, segments remarquables (tolérance), export PNG/CSV.")

# Sidebar
st.sidebar.header("⚙️ Paramètres")
uploaded = st.sidebar.file_uploader("📂 Dépose ta trace GPX", type=["gpx"])

lissage_m = st.sidebar.number_input("Lissage altitude (m) — influe sur D+/D-", 0, 200, 15, 1)
distance_cible_km = st.sidebar.number_input(
    "Distance cible officielle (km)",
    min_value=0.1,
    max_value=500.0,
    value=42.195,
    step=0.001,
    help="La trace sera recalée pour finir exactement à cette distance."
)
seuil_pente_pct = st.sidebar.number_input("Seuil de pente (%)", 0.0, 100.0, 5.0, 0.5)
deniv_min_m = st.sidebar.number_input("Dénivelé min. d’un segment remarquable (m)", 10.0, 5000.0, 80.0, 10.0)
dist_min_m = st.sidebar.number_input("Distance min. d’un segment (m)", 10.0, 50000.0, 400.0, 10.0)
nb_segments = st.sidebar.number_input("Nombre max. de segments remarquables", 1, 50, 10, 1)
altitude_bias_m = st.sidebar.number_input(
    "Décalage altitude export KML (m)",
    min_value=-50.0, max_value=50.0, value=12.0, step=0.5
)
# Nouveaux paramètres de tolérance
tol_dn_in_up_pct = st.sidebar.number_input(
    "Tolérance de D- dans une montée (%)", 0.0, 100.0, 10.0, 1.0,
    help="Pourcentage de D- autorisé à l’intérieur d’une montée, relatif au D+ du segment."
)
tol_up_in_dn_pct = st.sidebar.number_input(
    "Tolérance de D+ dans une descente (%)", 0.0, 100.0, 10.0, 1.0,
    help="Pourcentage de D+ autorisé à l’intérieur d’une descente, relatif au D- du segment."
)
st.sidebar.markdown("### 🗺️ Options KML")
enable_kml_color = st.sidebar.checkbox("Colorer le KML selon la pente", value=True)
enable_km_waypoints = st.sidebar.checkbox("Ajouter des waypoints chaque km", value=False)
palette_abs_max = st.sidebar.number_input(
    "Seuil max de la palette (|pente|, %)",
    min_value=5.0, max_value=60.0, value=30.0, step=1.0,
    help="Rouge ≥ +seuil, Jaune ≈ 0, Vert ≤ -seuil"
)
# Longueur minimale d’un tronçon coloré (pour éviter les micro-segments)
min_color_run_m = st.sidebar.number_input(
    "Longueur min. d’un tronçon coloré (m)",
    min_value=50, max_value=2000, value=500, step=50,
    help="Les segments plus courts seront fusionnés avec leurs voisins."
)

# Mode d'altitude pour le KML
alt_mode_label = st.sidebar.selectbox(
    "Mode altitude KML",
    ["Altitudes GPX + biais (absolu)", "Au-dessus du terrain (+ biais)"],
    index=0
)
altitude_mode = "absolute" if "absolu" in alt_mode_label else "relativeToGround"


st.sidebar.markdown("---")
st.sidebar.info("💡 Astuce : baisse le **Seuil de pente** et/ou le **Dénivelé min.** pour détecter plus de segments ; ajuste la **Tolérance** pour accepter des contre-pentes réalistes.")

if uploaded is None:
    st.warning("👉 Charge un fichier GPX pour commencer.")
    st.stop()

# Lecture GPX
gpx = gpxpy.parse(uploaded.getvalue().decode("utf-8", errors="ignore"))
df = gpx_to_df(gpx)
if df.empty:
    st.error("Le fichier ne contient aucun point exploitable.")
    st.stop()

# Lissage altitude
#df["ele"] = df["ele"].astype(float)
#df["ele_smooth"] = smooth_series(df["ele"].to_numpy(), lissage_m, df["dist_m"].to_numpy())
#
# D+ / D- global à partir de la série lissée
#de_all = np.r_[0.0, np.diff(df["ele_smooth"])]
#Dplus = float(np.sum(de_all[de_all > 0]))
#Dminus = float(-np.sum(de_all[de_all < 0]))
#dist_tot_km = df["dist_km"].iloc[-1]
#alt_min = float(np.nanmin(df["ele_smooth"]))
#alt_max = float(np.nanmax(df["ele_smooth"]))

# 15 avril-----
# Lissage altitude
# Lissage altitude
df["ele"] = df["ele"].astype(float)
df["ele_smooth"] = smooth_series(df["ele"].to_numpy(), lissage_m, df["dist_m"].to_numpy())

# Recalage distance sur la cible officielle
df = calibrate_distance_to_target(df, distance_cible_km)

# D+ / D- global à partir de la série lissée
de_all = np.r_[0.0, np.diff(df["ele_smooth"])]
Dplus = float(np.sum(de_all[de_all > 0]))
Dminus = float(-np.sum(de_all[de_all < 0]))

dist_tot_km_gps = df["dist_km_gps"].iloc[-1]
dist_tot_km_cal = df["dist_km_cal"].iloc[-1]
alt_min = float(np.nanmin(df["ele_smooth"]))
alt_max = float(np.nanmax(df["ele_smooth"]))

# 15 avril---

# Bandeau résumé
coef_dist = float(df["dist_coef"].iloc[0]) if "dist_coef" in df.columns else 1.0

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Distance GPX", f"{dist_tot_km_gps:.3f} km")
c2.metric("Distance cible", f"{dist_tot_km_cal:.3f} km")
c3.metric("Coef distance", f"{coef_dist:.5f}")
c4.metric("D+", f"{int(round(Dplus))} m")
c5.metric("D-", f"{int(round(Dminus))} m")
c6.metric("Alt. max", f"{int(round(alt_max))} m")

st.markdown("---")

# === SLIDER TRONÇON ===
st.subheader("🎚️ Sélection du tronçon à analyser")
min_km = float(df["dist_km_cal"].min())
max_km = float(df["dist_km_cal"].max())
troncon_km = st.slider(
    "Choisis la portion (en km) :",
    min_value=min_km, max_value=max_km,
    value=(min_km, max_km),
    step=0.01
)
km_start, km_end = troncon_km
mask_seg = (df["dist_km_cal"] >= km_start) & (df["dist_km_cal"] <= km_end)
df_seg = df.loc[mask_seg].copy()
if df_seg.empty:
    st.info("Le tronçon sélectionné est trop court. Étends un peu la plage.")
    st.stop()

# === GRAPHIQUE PROFIL (responsive) ===
st.subheader("🗻 Profil altimétrique (tronçon)")
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(df["dist_km_cal"], df["ele_smooth"], linewidth=1.0, alpha=0.25)
ax1.plot(df_seg["dist_km_cal"], df_seg["ele_smooth"], linewidth=1.8)
ax1.set_xlabel("Distance (km)")
ax1.set_ylabel("Altitude (m)")
ax1.grid(True, alpha=0.3)
st.pyplot(fig1, use_container_width=True)
fig_to_png_download_button(fig1, "⬇️ Télécharger le profil (PNG)", "profil_troncon.png")

# === MÉTRIQUES DU TRONÇON ===
dist_km_tr, dplus_tr, dminus_tr = compute_troncon_metrics(df_seg)
m1, m2, m3 = st.columns(3)
m1.metric("Longueur du tronçon", f"{dist_km_tr:.2f} km")
m2.metric("D+ du tronçon", f"{int(round(dplus_tr))} m")
m3.metric("D- du tronçon", f"{int(round(dminus_tr))} m")

st.markdown("---")

# === EXPORT DU TRONÇON (KML) ===
st.subheader("📤 Export du tronçon sélectionné")
kml_export_block(
    df_seg=df_seg,
    name=f"Tronçon {km_start:.1f}–{km_end:.1f} km",
    key_prefix="troncon",
    altitude_bias_m=altitude_bias_m,
    color_by_slope=enable_kml_color,
    palette_abs_max=palette_abs_max,
    add_km_waypoints=enable_km_waypoints,
    df_full_for_km=df, km_start=km_start, km_end=km_end,
    altitude_mode=altitude_mode,          # 👈 si tu l’utilises
    min_run_m=min_color_run_m             # 👈 NOUVEAU
)

# === DÉTAIL PAR KILOMÈTRE DU TRONÇON ===
st.subheader("📏 Détail du tronçon")

pas_affichage_km = st.selectbox(
    "Afficher les données par tranche de :",
    options=[1, 2, 5, 10],
    index=0,
    format_func=lambda x: f"{x} km"
)

# === DÉTAIL DU TRONÇON PAR PAS CHOISI ===
def _slice_km_segment(df_full: pd.DataFrame, km_a: float, km_b: float) -> pd.DataFrame:
    """
    Sous-échantillonne df_full sur [km_a, km_b] selon la distance recalée,
    en garantissant des points aux bornes par interpolation.
    """
    x = df_full["dist_km_cal"].to_numpy()
    y = df_full["ele_smooth"].to_numpy()
    m = df_full["dist_m_cal"].to_numpy()

    mask = (x >= km_a) & (x <= km_b)
    xs = x[mask]
    ys = y[mask]
    ms = m[mask]

    if xs.size == 0 or xs[0] > km_a:
        ele_a = float(np.interp(km_a, x, y))
        m_a = float(np.interp(km_a, x, m))
        xs = np.r_[km_a, xs]
        ys = np.r_[ele_a, ys]
        ms = np.r_[m_a, ms]

    if xs.size == 0 or xs[-1] < km_b:
        ele_b = float(np.interp(km_b, x, y))
        m_b = float(np.interp(km_b, x, m))
        xs = np.r_[xs, km_b]
        ys = np.r_[ys, ele_b]
        ms = np.r_[ms, m_b]

    return pd.DataFrame({
        "dist_km": xs,
        "ele_smooth": ys,
        "dist_m": ms
    })

def _metrics_on_slice(df_slice: pd.DataFrame):
    """Retourne (longueur_m, dplus_m, dminus_m, pente_moy_pct)."""
    if len(df_slice) < 2:
        return 0.0, 0.0, 0.0, 0.0
    de = np.diff(df_slice["ele_smooth"].to_numpy())
    dd = np.diff(df_slice["dist_m"].to_numpy())
    dplus = float(np.sum(de[de > 0]))
    dminus = float(-np.sum(de[de < 0]))
    longueur_m = float(df_slice["dist_m"].iloc[-1] - df_slice["dist_m"].iloc[0])
    pente_moy = ((dplus - dminus) / longueur_m * 100.0) if longueur_m > 0 else 0.0
    return longueur_m, dplus, dminus, pente_moy

total_km_trace = float(df["dist_km_cal"].iloc[-1])

# bornes alignées sur le pas choisi
pas = float(pas_affichage_km)
start_bin = math.floor(km_start / pas) * pas
end_bin = math.ceil(km_end / pas) * pas

rows = []
current = start_bin

while current < end_bin:
    a = max(current, km_start)
    b = min(current + pas, km_end)

    if b <= a:
        current += pas
        continue

    df_k = _slice_km_segment(df, a, b)
    longueur_m, dplus_m, dminus_m, pente_moy_pct = _metrics_on_slice(df_k)

    rows.append({
        "Km départ": round(a, 1),
        "Km arrivée": round(b, 1),
        "Longueur (km)": round(b - a, 1),
        "D+ (m)": int(round(dplus_m)),
        "D- (m)": int(round(dminus_m)),
        "Pente moy. (%)": f"{pente_moy_pct:+.0f}%",
        "Km restant (trace)": round(max(total_km_trace - b, 0.0), 1),
    })

    current += pas

table_km = pd.DataFrame(rows)

if table_km.empty:
    st.info("Aucune tranche disponible dans la plage sélectionnée.")
else:
    st.dataframe(table_km, use_container_width=True, hide_index=True)
    df_to_csv_download_button(
        table_km,
        f"⬇️ Exporter le détail par tranche de {int(pas_affichage_km)} km (CSV)",
        f"detail_par_{int(pas_affichage_km)}km_{km_start:.1f}-{km_end:.1f}.csv"
    )


# === RÉPARTITION DISTANCE PAR CLASSES DE PENTE (5%) ===
st.subheader("📊 Répartition de la distance par classes de pente (pas de 5%) — tronçon")

dist_by_slope = slope_distribution_by_5pct(df_seg)

if not dist_by_slope.empty:
    # Nettoyage et mise en forme
    dist_by_slope_display = pd.DataFrame({
        "Classe pente (%)": dist_by_slope["Classe pente (%)"],
        "Distance (km)": dist_by_slope["Distance (km)"].round(1),
        "Part (%)": dist_by_slope["Part (%)"].round(2).astype(str) + "%"
    })

    st.dataframe(dist_by_slope_display, use_container_width=True, hide_index=True)
    df_to_csv_download_button(
        dist_by_slope_display,
        "⬇️ Exporter répartition pente (CSV)",
        "repartition_pente_troncon.csv"
    )
else:
    st.info("Aucune donnée de pente disponible pour ce tronçon.")


# Bar plot (responsive)
fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.bar(dist_by_slope["Classe pente (%)"], dist_by_slope["Distance (km)"])
ax3.set_xlabel("Classe de pente (%)")
ax3.set_ylabel("Distance (km)")
ax3.grid(True, axis="y", alpha=0.3)
plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")
st.pyplot(fig3, use_container_width=True)
fig_to_png_download_button(fig3, "⬇️ Télécharger histogramme pentes (PNG)", "histogramme_pentes_troncon.png")

st.markdown("---")

# === DÉTECTION SEGMENTS REMARQUABLES (global) ===
st.subheader("🏔️ Montées remarquables (global)")
tbl_montee = detect_segments_remarquables(
    df.assign(ele=df["ele_smooth"]),
    pente_min_pct=seuil_pente_pct,
    deniv_min_m=deniv_min_m,
    dist_min_m=dist_min_m,
    type_segment="montee",
    tolerance_pct_opposite=tol_dn_in_up_pct,  # tolérance D- dans montée
)

if tbl_montee.empty:
    st.info("Aucune montée remarquable selon les paramètres actuels.")
else:
    tbl_montee = tbl_montee.sort_values("denivele_m", ascending=False).head(nb_segments).copy()

    # Tableau “joli” (formats demandés)
    disp_up = pd.DataFrame({
        "Km début": (tbl_montee["km_debut"].round(1)),
        "Km fin": (tbl_montee["km_fin"].round(1)),
        "Longueur (km)": (tbl_montee["longueur_m"] / 1000).round(1),
        "D+ net (m)": np.round(tbl_montee["denivele_m"]).astype(int),
        "D+ (m)": np.round(tbl_montee["dplus_brut"]).astype(int),
        "D- (m)": np.round(tbl_montee["dminus_brut"]).astype(int),
        "Pente moy. (%)": (tbl_montee["pente_moy_pct"].round(0).astype(int).astype(str) + "%"),
    })
    # Colonnes brutes cachées pour le tracé exact
    disp_up["km_debut_raw"] = tbl_montee["km_debut"]
    disp_up["km_fin_raw"] = tbl_montee["km_fin"]

    gb = GridOptionsBuilder.from_dataframe(disp_up)
    gb.configure_selection("single", use_checkbox=True)
    gb.configure_column("km_debut_raw", hide=True)
    gb.configure_column("km_fin_raw", hide=True)
    grid_up = AgGrid(
        disp_up,
        gridOptions=gb.build(),
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        fit_columns_on_grid_load=True,
        height=260,
        allow_unsafe_jscode=True
    )
    # Export CSV (sans colonnes cachées)
    df_to_csv_download_button(
        disp_up.drop(columns=["km_debut_raw", "km_fin_raw"]),
        "⬇️ Exporter montées remarquables (CSV)",
        "montees_remarquables.csv"
    )

    # --- Sélection et affichage du segment + répartition pentes (montée uniquement) ---
    sel_up = grid_up.get("selected_rows", None)
    row_up = None
    if isinstance(sel_up, list) and len(sel_up) > 0:
        row_up = sel_up[0]
    elif isinstance(sel_up, pd.DataFrame) and not sel_up.empty:
        row_up = sel_up.iloc[0].to_dict()

    km0 = km1 = None
    if row_up is not None:
        km0 = float(row_up.get("km_debut_raw", row_up.get("Km début")))
        km1 = float(row_up.get("km_fin_raw", row_up.get("Km fin")))
        plot_segment_overlay(
            df, km0, km1,
            f"Montée sélectionnée — du km {km0:.1f} au km {km1:.1f}",
            "montee_segment.png"
        )

        # Répartition pentes (montée) pour ce segment — via la répartition standard, puis filtre positif
        st.markdown("**Répartition des pentes (montée) — segment sélectionné**")
        df_seg_sel = df[(df["dist_km_cal"] >= km0) & (df["dist_km_cal"] <= km1)].copy()
        dist_all = slope_distribution_by_5pct(df_seg_sel)  # distribution complète (±)
        dist_up = filter_distribution_by_sign(dist_all, sign="up")

        if dist_up is not None and not dist_up.empty:
            st.dataframe(dist_up, use_container_width=True, hide_index=True)
            df_to_csv_download_button(
                dist_up,
                "⬇️ Exporter répartition pentes de la montée (CSV)",
                "repartition_pentes_montee_segment.csv"
            )
        else:
            st.info("Aucune classe de pente positive à afficher pour ce segment (après filtrage).")

        # Histogramme de la répartition (montée)
        plot_distribution_bar(
            dist_up,
            title="Répartition des pentes (montée) — histogramme",
            filename="histogramme_pentes_montee_segment.png"
        )

        # — Résumé du segment (montée) en une ligne + export CSV —
        if row_up is not None:
            dist_km_sel, dplus_sel, dminus_sel = compute_troncon_metrics(df_seg_sel)
            net_up = max(dplus_sel - dminus_sel, 0.0)
            dist_m_sel = float(df_seg_sel["dist_m"].iloc[-1] - df_seg_sel["dist_m"].iloc[0]) if len(df_seg_sel) >= 2 else 0.0
            pente_moy_up = (net_up / dist_m_sel * 100.0) if dist_m_sel > 0 else 0.0

            resume_up = pd.DataFrame({
                "De (km)": [round(km0, 1)],
                "À (km)": [round(km1, 1)],
                "Longueur (km)": [round(dist_km_sel, 1)],
                "D+ (m)": [int(round(dplus_sel))],
                "D- (m)": [int(round(dminus_sel))],
                "D+ net (m)": [int(round(net_up))],
                "Pente moy. (%)": [f"{int(round(pente_moy_up))}%"]
            })

            # Affichage horizontal (transposé)
            resume_up_T = resume_up.T.rename(columns={0: "Valeur"})
            st.dataframe(resume_up_T, use_container_width=True, hide_index=False)

            # Bouton d’export CSV (vertical)
            csv_up = resume_up_T.reset_index().rename(columns={"index": "Paramètre"}).to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Exporter résumé de la montée (CSV)",
                data=csv_up,
                file_name="resume_montee_segment.csv",
                mime="text/csv"
            )

            # Export KML du segment sélectionné (montée) avec +9 m
            st.markdown("**Export KML (montée)**")
            kml_export_block(
                df_seg=df_seg_sel,
                name=f"Montee {km0:.1f}–{km1:.1f} km",
                key_prefix="montee",
                altitude_bias_m=altitude_bias_m,
                color_by_slope=enable_kml_color,
                palette_abs_max=palette_abs_max,
                add_km_waypoints=enable_km_waypoints,
                df_full_for_km=df, km_start=km0, km_end=km1,
                altitude_mode=altitude_mode,        # ✅ nouveau si pas déjà présent
                min_run_m=min_color_run_m           # ✅ idem
            )


            # === DÉTAIL PAR TRANCHE — MONTÉE SÉLECTIONNÉE ===
            st.markdown(f"**📏 Détail par tranche de {int(pas_affichage_km)} km — montée sélectionnée**")
            
            total_km_trace = float(df["dist_km_cal"].iloc[-1])
            
            pas = float(pas_affichage_km)
            start_bin = math.floor(km0 / pas) * pas
            end_bin = math.ceil(km1 / pas) * pas
            
            rows_up = []
            current = start_bin
            
            while current < end_bin:
                a = max(current, km0)
                b = min(current + pas, km1)
            
                if b <= a:
                    current += pas
                    continue
            
                df_k = _slice_km_segment(df, a, b)
                longueur_m, dplus_m, dminus_m, pente_moy_pct = _metrics_on_slice(df_k)
            
                rows_up.append({
                    "Km départ": round(a, 1),
                    "Km arrivée": round(b, 1),
                    "Longueur (km)": round((b - a), 1),
                    "D+ (m)": int(round(dplus_m)),
                    "D- (m)": int(round(dminus_m)),
                    "Pente moy. (%)": f"{pente_moy_pct:+.0f}%",
                    "Km restant (trace)": round(max(total_km_trace - b, 0.0), 1),
                })
            
                current += pas
            
            table_km_up = pd.DataFrame(rows_up)
            
            if table_km_up.empty:
                st.info("Aucune tranche disponible dans la plage sélectionnée de la montée.")
            else:
                st.dataframe(table_km_up, use_container_width=True, hide_index=True)
                df_to_csv_download_button(
                    table_km_up,
                    f"⬇️ Exporter le détail par tranche de {int(pas_affichage_km)} km (montée) — CSV",
                    f"detail_par_{int(pas_affichage_km)}km_montee_{km0:.1f}-{km1:.1f}.csv"
                )


# ===== Descentes =====
st.subheader("🧗 Descentes remarquables (global)")
tbl_desc = detect_segments_remarquables(
    df.assign(ele=df["ele_smooth"]),
    pente_min_pct=seuil_pente_pct,
    deniv_min_m=deniv_min_m,
    dist_min_m=dist_min_m,
    type_segment="descente",
    tolerance_pct_opposite=tol_up_in_dn_pct,  # tolérance D+ dans descente
)

if tbl_desc.empty:
    st.info("Aucune descente remarquable selon les paramètres actuels.")
else:
    tbl_desc = tbl_desc.sort_values("denivele_m", ascending=False).head(nb_segments).copy()

    disp_dn = pd.DataFrame({
        "Km début": (tbl_desc["km_debut"].round(1)),
        "Km fin": (tbl_desc["km_fin"].round(1)),
        "Longueur (km)": (tbl_desc["longueur_m"] / 1000).round(1),
        "D- net (m)": np.round(tbl_desc["denivele_m"]).astype(int),
        "D+ (m)": np.round(tbl_desc["dplus_brut"]).astype(int),
        "D- (m)": np.round(tbl_desc["dminus_brut"]).astype(int),
        # 🔻 pente moyenne affichée avec un signe négatif
        "Pente moy. (%)": ((-tbl_desc["pente_moy_pct"]).round(0).astype(int)).astype(str) + "%",
    })

    disp_dn["km_debut_raw"] = tbl_desc["km_debut"]
    disp_dn["km_fin_raw"] = tbl_desc["km_fin"]

    gb2 = GridOptionsBuilder.from_dataframe(disp_dn)
    gb2.configure_selection("single", use_checkbox=True)
    gb2.configure_column("km_debut_raw", hide=True)
    gb2.configure_column("km_fin_raw", hide=True)
    grid_dn = AgGrid(
        disp_dn,
        gridOptions=gb2.build(),
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        fit_columns_on_grid_load=True,
        height=260,
        allow_unsafe_jscode=True
    )
    df_to_csv_download_button(
        disp_dn.drop(columns=["km_debut_raw", "km_fin_raw"]),
        "⬇️ Exporter descentes remarquables (CSV)",
        "descentes_remarquables.csv"
    )

    # --- Sélection et affichage du segment + répartition pentes (descente uniquement) ---
    sel_dn = grid_dn.get("selected_rows", None)
    row_dn = None
    if isinstance(sel_dn, list) and len(sel_dn) > 0:
        row_dn = sel_dn[0]
    elif isinstance(sel_dn, pd.DataFrame) and not sel_dn.empty:
        row_dn = sel_dn.iloc[0].to_dict()

    km0 = km1 = None
    if row_dn is not None:
        km0 = float(row_dn.get("km_debut_raw", row_dn.get("Km début")))
        km1 = float(row_dn.get("km_fin_raw", row_dn.get("Km fin")))
        plot_segment_overlay(
            df, km0, km1,
            f"Descente sélectionnée — du km {km0:.1f} au km {km1:.1f}",
            "descente_segment.png"
        )

        # Répartition pentes (descente) pour ce segment — via la répartition standard, puis filtre négatif
        st.markdown("**Répartition des pentes (descente) — segment sélectionné**")
        df_seg_sel = df[(df["dist_km_cal"] >= km0) & (df["dist_km_cal"] <= km1)].copy()
        dist_all = slope_distribution_by_5pct(df_seg_sel)  # distribution complète (±)
        dist_dn = filter_distribution_by_sign(dist_all, sign="down")

        if dist_dn is not None and not dist_dn.empty:
            st.dataframe(dist_dn, use_container_width=True, hide_index=True)
            df_to_csv_download_button(
                dist_dn,
                "⬇️ Exporter répartition pentes de la descente (CSV)",
                "repartition_pentes_descente_segment.csv"
            )
        else:
            st.info("Aucune classe de pente négative à afficher pour ce segment (après filtrage).")

        # Histogramme de la répartition (descente)
        plot_distribution_bar(
            dist_dn,
            title="Répartition des pentes (descente) — histogramme",
            filename="histogramme_pentes_descente_segment.png"
        )

        # — Résumé du segment (descente) en une ligne + export CSV —
        if row_dn is not None:
            dist_km_sel, dplus_sel, dminus_sel = compute_troncon_metrics(df_seg_sel)
            net_down = max(dminus_sel - dplus_sel, 0.0)
            dist_m_sel = float(df_seg_sel["dist_m"].iloc[-1] - df_seg_sel["dist_m"].iloc[0]) if len(df_seg_sel) >= 2 else 0.0
            pente_moy_dn = -abs(net_down / dist_m_sel * 100.0) if dist_m_sel > 0 else 0.0  # 🔹 forcer négatif

            resume_dn = pd.DataFrame({
                "De (km)": [round(km0, 1)],
                "À (km)": [round(km1, 1)],
                "Longueur (km)": [round(dist_km_sel, 1)],
                "D+ (m)": [int(round(dplus_sel))],
                "D- (m)": [int(round(dminus_sel))],
                "D- net (m)": [int(round(net_down))],
                "Pente moy. (%)": [f"{int(round(pente_moy_dn))}%"]  # 🔹 affichage avec signe négatif
            })

            # Affichage horizontal (transposé)
            resume_dn_T = resume_dn.T.rename(columns={0: "Valeur"})
            st.dataframe(resume_dn_T, use_container_width=True, hide_index=False)

            # Bouton d’export CSV (vertical)
            csv_dn = resume_dn_T.reset_index().rename(columns={"index": "Paramètre"}).to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Exporter résumé de la descente (CSV)",
                data=csv_dn,
                file_name="resume_descente_segment.csv",
                mime="text/csv"
            )

            # Export KML du segment sélectionné (descente) avec +9 m

            st.markdown("**Export KML (descente)**")
            kml_export_block(
                df_seg=df_seg_sel,
                name=f"Descente {km0:.1f}–{km1:.1f} km",
                key_prefix="descente",
                altitude_bias_m=altitude_bias_m,
                color_by_slope=enable_kml_color,
                palette_abs_max=palette_abs_max,
                add_km_waypoints=enable_km_waypoints,
                df_full_for_km=df, km_start=km0, km_end=km1,
                altitude_mode=altitude_mode,        # ✅
                min_run_m=min_color_run_m           # ✅
            )


            # === DÉTAIL PAR TRANCHE — DESCENTE SÉLECTIONNÉE ===
            st.markdown(f"**📏 Détail par tranche de {int(pas_affichage_km)} km — descente sélectionnée**")
            
            total_km_trace = float(df["dist_km_cal"].iloc[-1])
            
            pas = float(pas_affichage_km)
            start_bin = math.floor(km0 / pas) * pas
            end_bin = math.ceil(km1 / pas) * pas
            
            rows_dn = []
            current = start_bin
            
            while current < end_bin:
                a = max(current, km0)
                b = min(current + pas, km1)
            
                if b <= a:
                    current += pas
                    continue
            
                df_k = _slice_km_segment(df, a, b)
                longueur_m, dplus_m, dminus_m, pente_moy_pct = _metrics_on_slice(df_k)
            
                rows_dn.append({
                    "Km départ": round(a, 1),
                    "Km arrivée": round(b, 1),
                    "Longueur (km)": round((b - a), 1),
                    "D+ (m)": int(round(dplus_m)),
                    "D- (m)": int(round(dminus_m)),
                    "Pente moy. (%)": f"{pente_moy_pct:+.0f}%",
                    "Km restant (trace)": round(max(total_km_trace - b, 0.0), 1),
                })
            
                current += pas
            
            table_km_dn = pd.DataFrame(rows_dn)
            
            if table_km_dn.empty:
                st.info("Aucune tranche disponible dans la plage sélectionnée de la descente.")
            else:
                st.dataframe(table_km_dn, use_container_width=True, hide_index=True)
                df_to_csv_download_button(
                    table_km_dn,
                    f"⬇️ Exporter le détail par tranche de {int(pas_affichage_km)} km (descente) — CSV",
                    f"detail_par_{int(pas_affichage_km)}km_descente_{km0:.1f}-{km1:.1f}.csv"
                )



    else:
        st.caption("💡 Astuce : sélectionne une ligne du tableau pour voir le segment et sa répartition de pentes.")

