import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
import glob
import openai
import json

from openai import OpenAI



# NEW: lecture des fichiers .FIT
import fitdecode
from fitdecode import FitReader
from fitdecode.records import FitDataMessage

st.markdown("""
    <style>
    h1 {font-size: 1.8rem !important;}
    h2 {font-size: 1.4rem !important;}
    h3 {font-size: 1.1rem !important;}
    h4 {font-size: 1.0rem !important;}
    </style>
""", unsafe_allow_html=True)


st.set_page_config(page_title="Analyse Trail - Fichier unique", layout="wide")
st.title("📈 Analyse Trail - Glycémie")

st.markdown("Choisis la source des données :")

mode = st.radio(
    "Source du fichier",
    [
        "Depuis le dossier data/ (déjà traité CSV)",
        "Importer un .csv (record brut)",
        "Importer un .fit (montre)"
    ],
    horizontal=True
)

record_file = None
source_type = None   # "treated" | "raw_csv" | "raw_fit"

# ---------- Helpers ----------
def parse_fit_to_df(uploaded_fit_file, show_preview: bool = True):
    """
    Parse un fichier .FIT (fichier uploadé) -> DataFrame
    - Conserve les champs standards + developer fields (on lit tous les champs des messages 'record')
    - Normalise les noms en lower snake_case
    - Préserve la compatibilité aval en garantissant un set minimal de colonnes
    """
    rows = []
    with FitReader(uploaded_fit_file) as fit:
        for frame in fit:
            if isinstance(frame, FitDataMessage) and frame.name == "record":
                row = {}
                for f in frame.fields:
                    if f.value is None:
                        continue
                    # normalisation immédiate du nom de champ
                    col = str(f.name).strip().lower()
                    row[col] = f.value
                rows.append(row)

    df = pd.DataFrame(rows)

    # --- Normalisations & alias légers (sans écraser tes colonnes si elles existent déjà) ---
    # Glycémie: certaines montres utilisent 'glucose_level'
    if "glucose_level" in df.columns and "x_glucose_level_0_0" not in df.columns:
        df = df.rename(columns={"glucose_level": "x_glucose_level_0_0"})

    # Altitude/vitesse "enhanced" (fallback si manquants)
    if "enhanced_altitude" not in df.columns and "altitude" in df.columns:
        df["enhanced_altitude"] = df["altitude"]
    if "enhanced_speed" not in df.columns and "speed" in df.columns:
        df["enhanced_speed"] = df["speed"]

    # Timestamp propre + tri
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)

    # --- Compatibilité aval : garantir un schéma minimum et types numériques ---
    # (1) Colonnes que ton app attend fréquemment. Si absentes -> on les crée vides.
    expected_cols = [
        "timestamp", "distance", "cadence", "heart_rate",
        "enhanced_speed", "enhanced_altitude",
        # optionnelles utiles si présentes :
        "x_glucose_level_0_0", "battery_voltage", "battery_level",
        "temperature", "power"
    ]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = pd.NA

    # (2) Coercition en numérique quand pertinent (sans planter si vide)
    num_cols = [
        "distance","cadence","heart_rate","enhanced_speed","enhanced_altitude",
        "x_glucose_level_0_0","battery_voltage","battery_level","temperature","power"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Aperçu (désactivable)
    if show_preview:
        st.caption("✅ Colonnes détectées après normalisation & compat :")
        st.write(sorted(df.columns.tolist()))
        st.dataframe(df.head())

    return df



#---------- Fonctions : Pipeline standard ----------
def standard_pipeline(df, source_type):
    """
    Aligne le traitement sur ton pipeline :
    - échantillonnage 1/3
    - nettoyage/numérique
    - correction cadence x2 (CSV brut auto ; FIT via toggle en amont)
    - pente / D+ / D-
    """
    df.columns = df.columns.str.strip().str.lower()

    if 'timestamp' not in df.columns:
        st.error("❌ Colonne 'timestamp' introuvable après parsing.")
        st.stop()

    # 🧹 Réduction : 1 ligne sur 3
    df = df.iloc[::3].reset_index(drop=True)

    # 🧩 Remplacer les virgules par des points (utile surtout pour CSV EU)
    df = df.replace(',', '.', regex=True)

    # 🧼 Conversion numérique (bugfix: pas de doublon, virgule manquante corrigée)
    colonnes_numeriques = [
        "distance",
        "enhanced_altitude",
        "heart_rate",
        "cadence",
        "enhanced_speed",
        "grade_adjusted_speed",
        "x_glucose_level_0_0"
    ]
    for col in colonnes_numeriques:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 🏔️ Calculs robustes
    df["delta_altitude"] = df["enhanced_altitude"].diff() if "enhanced_altitude" in df.columns else 0.0
    df["delta_distance"] = df["distance"].diff() if "distance" in df.columns else 0.0

    denom = df["delta_distance"].replace(0, np.nan) if "delta_distance" in df.columns else np.nan
    df["pente_%"] = ((df["delta_altitude"] / denom) * 100) if "delta_altitude" in df.columns else 0.0
    df["pente_%"] = df["pente_%"].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-100, 100)

    # ⚙️ Calcul de la vitesse corrigée de la pente (grade adjusted speed)
    if "grade_adjusted_speed" not in df.columns:
        if "enhanced_speed" not in df.columns and "speed" in df.columns:
            df["enhanced_speed"] = df["speed"]

        if "enhanced_speed" in df.columns and "pente_%" in df.columns and "delta_distance" in df.columns:
            df["enhanced_speed"] = pd.to_numeric(df["enhanced_speed"], errors="coerce")
            df["pente_%"] = pd.to_numeric(df["pente_%"], errors="coerce").fillna(0)

            # 👉 fenêtre ~50 m (calculée en nb d’échantillons)
            step_m = np.nanmedian(pd.to_numeric(df["delta_distance"], errors="coerce"))
            window_n = int(max(3, round(50.0 / step_m))) if step_m and step_m > 0 else 5
            g_pct = df["pente_%"].rolling(window=window_n, min_periods=1).mean()

            # Coeffs (en %) — n’hésite pas à monter k_up si tu veux plus d’écart
            k_up, k_down = 0.04, 0.006   # ← un peu plus “fermes” que 0.03 / 0.008

            factor = np.where(g_pct >= 0, 1 + k_up * g_pct, 1 + k_down * g_pct)
            factor = np.clip(factor, 0.1, 5.0)

            df["grade_adjusted_speed"] = df["enhanced_speed"] * factor
        else:
            df["grade_adjusted_speed"] = np.nan


    if "delta_altitude" in df.columns:
        df["D+"] = df["delta_altitude"].apply(lambda x: x if x > 0 else 0)
        df["D-"] = df["delta_altitude"].apply(lambda x: -x if x < 0 else 0)
    else:
        df["D+"] = 0
        df["D-"] = 0

    df.fillna({"delta_altitude": 0, "delta_distance": 0, "D+": 0, "D-": 0}, inplace=True)
    return df

# ---------- Sélection source ----------
if mode == "Depuis le dossier data/ (déjà traité CSV)":
    os.makedirs("data", exist_ok=True)
    existing_files = sorted(glob.glob("data/*.csv"))
    if existing_files:
        display_names = [os.path.basename(p) for p in existing_files]
        choice = st.selectbox("Sélectionne un fichier dans `data/` :", display_names, index=0)
        if choice:
            selected_path = os.path.join("data", choice)
            record_file = selected_path
            source_type = "treated"
            st.success(f"✅ Fichier sélectionné : `{choice}` (déjà traité)")
    else:
        st.info("Aucun CSV trouvé dans `data/`.")
elif mode == "Importer un .csv (record brut)":
    record_file = st.file_uploader("📄 Dépose ton fichier `record.csv` (export brut)", type="csv")
    if record_file is not None:
        source_type = "raw_csv"
else:  # "Importer un .fit (montre)"
    record_file = st.file_uploader("⌚ Dépose ton fichier `.fit`", type=["fit", "FIT"])
    if record_file is not None:
        source_type = "raw_fit"

# ---------- Lecture + pipeline ----------
if record_file is not None:
    if source_type == "treated":
        st.caption("🗂️ Source : CSV **déjà traité** — aucune correction auto.")
        df = pd.read_csv(record_file, sep=None, engine="python")
        # Harmonisation minimale si déjà traité
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = standard_pipeline(df, source_type="treated")

    elif source_type == "raw_csv":
        st.caption("🆕 Source : **record brut (CSV)** — pas de correction auto, tu choisis ci-dessous.")
        df = pd.read_csv(record_file, sep=None, engine="python")
        df = standard_pipeline(df, source_type="raw_csv")

    elif source_type == "raw_fit":
        st.caption("🆕 Source : **record brut (.FIT)** — pas de correction auto, tu choisis ci-dessous.")
        df = parse_fit_to_df(record_file)
        df = standard_pipeline(df, source_type="raw_fit")

    # === Toggle unique après pipeline : correction cadence ×2 (si pertinent) ===
    if "cadence" in df.columns:
        df["cadence"] = pd.to_numeric(df["cadence"], errors="coerce")
        mean_cad = df["cadence"].mean()
        default_x2 = bool(pd.notna(mean_cad) and mean_cad < 110)  # suggestion si cadence semble en foulées/min
        apply_cadence_x2 = st.toggle(
            "🔁 Multiplier la cadence par 2",
            value=default_x2,
            help=(f"Cadence moyenne détectée : {mean_cad:.0f} spm. "
                  "Active si la montre enregistre par foulée (ex : ~85 au lieu de ~170 pas/min).")
        )
        if apply_cadence_x2:
            df["cadence"] = df["cadence"] * 2
            st.caption("✅ Correction ×2 appliquée à la cadence.")
    else:
        st.info("ℹ️ Colonne 'cadence' absente : aucune correction possible.")

    st.success("✅ Données prêtes")

    # (Optionnel) aperçu rapide
    # st.dataframe(df.head())

    # ---------- Enregistrement ----------
    st.markdown("### 💾 Enregistrement du fichier traité")
    nom_fichier = st.text_input("Nom du fichier (sans extension) :", value="donnees_traitees")

    if st.button("Enregistrer dans le dossier 'data/'"):
        os.makedirs("data", exist_ok=True)
        nom_fichier_clean = "".join(c for c in nom_fichier if c.isalnum() or c in ('_', '-')).rstrip()
        chemin_complet = f"data/{nom_fichier_clean}.csv"
        try:
            df.to_csv(chemin_complet, index=False)
            st.success(f"✅ Fichier enregistré sous : `{chemin_complet}`")
        except Exception as e:
            st.error(f"❌ Erreur lors de l'enregistrement : {e}")

else:
    st.info("⬆️ Merci d'importer un fichier `.fit` ou `record.csv` pour commencer.")

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
# =============== NORMALISATION DES DONNÉES (shim unique) ===============
df = df.copy()

# Horodatage & tri
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.sort_values("timestamp").reset_index(drop=True)

# Distance / altitude / vitesse
df["distance_m"] = pd.to_numeric(df.get("distance", np.nan), errors="coerce")
df["distance_km"] = df["distance_m"] / 1000.0
df["alt_m"] = pd.to_numeric(df.get("enhanced_altitude", np.nan), errors="coerce")
df["speed_m_s"] = pd.to_numeric(df.get("enhanced_speed", np.nan), errors="coerce")

# Pente / cardio / glycémie
df["pente_%"] = pd.to_numeric(df.get("pente_%", np.nan), errors="coerce")
df["heart_rate"] = pd.to_numeric(df.get("heart_rate", np.nan), errors="coerce")
# glycémie si dispo (mmol/L prioritaire, sinon conversion mg/dL->mmol/L)
if "x_glucose_level_0_0" in df.columns:
    df["glucose_val"] = pd.to_numeric(df["x_glucose_level_0_0"], errors="coerce")
elif "glucose_mmol" in df.columns:
    df["glucose_val"] = pd.to_numeric(df["glucose_mmol"], errors="coerce")
elif "glucose_mgdl" in df.columns:
    df["glucose_val"] = pd.to_numeric(df["glucose_mgdl"], errors="coerce") / 18.0
else:
    df["glucose_val"] = np.nan

# Oscillation verticale (mm) & temps de contact (ms)
df["vo_mm"] = pd.to_numeric(df.get("vertical_oscillation", np.nan), errors="coerce")
df["gct_ms"] = pd.to_numeric(df.get("stance_time", np.nan), errors="coerce")

# Cadence (pas/min) = cadence + fraction si dispo
cad = pd.to_numeric(df.get("cadence", np.nan), errors="coerce")
frac = pd.to_numeric(df.get("fractional_cadence", 0.0), errors="coerce")
# La fraction est généralement <1; on l’ajoute telle quelle si réaliste
frac = frac.where((frac >= 0) & (frac < 1.5), 0.0)
df["cadence_spm"] = cad + frac

# Longueur de foulée (m) – AUTO-détection d’unité depuis step_length
sl_raw = pd.to_numeric(df.get("step_length", np.nan), errors="coerce")
med_sl = np.nanmedian(sl_raw)
if np.isfinite(med_sl):
    if med_sl > 5:               # valeurs typiques "mm" (ex: 900 → 0.90 m)
        df["step_length_m"] = sl_raw / 1000.0
    elif 0.2 <= med_sl <= 2.5:   # déjà en mètres
        df["step_length_m"] = sl_raw
    else:
        # Valeurs bizarres → essaie mm sinon NaN
        df["step_length_m"] = sl_raw / 1000.0
else:
    df["step_length_m"] = np.nan

# Nettoyage réaliste (optionnel)
df.loc[~df["step_length_m"].between(0.3, 2.5), "step_length_m"] = np.nan
df.loc[~df["cadence_spm"].between(120, 220), "cadence_spm"] = np.nan  # marche/course

# Deltas temps & distance
df["dt_s"] = df["timestamp"].diff().dt.total_seconds()
df.loc[df["dt_s"] <= 0, "dt_s"] = np.nan
df["ddist_m"] = df["distance_m"].diff()
df.loc[df["ddist_m"] < 0, "ddist_m"] = np.nan

# Vitesse lissée simple (si non fournie)
if df["speed_m_s"].isna().all():
    df["speed_m_s"] = (df["ddist_m"] / df["dt_s"]).replace([np.inf, -np.inf], np.nan)
df["speed_m_s"] = df["speed_m_s"].rolling(5, min_periods=1, center=True).median()

# Dénivelé: favorise delta_altitude si présent, sinon diff(enhanced_altitude)
if "delta_altitude" in df.columns:
    df["dalt"] = pd.to_numeric(df["delta_altitude"], errors="coerce")
else:
    df["dalt"] = df["alt_m"].diff()
df.loc[~np.isfinite(df["dalt"]), "dalt"] = np.nan

# Progression (%)
total_dist_m = float(df["distance_m"].max()) if df["distance_m"].notna().any() else np.nan
df["progress_%"] = (df["distance_m"] / total_dist_m * 100).clip(0, 100) if np.isfinite(total_dist_m) and total_dist_m > 0 else np.nan
# ======================================================================
# ======================================================================

# =========================
# === 📊 Pente (%) vs Cadence (couleur = progression) | filtres + export PNG ===
# =========================
def render_cadence_vs_pente():
    st.markdown("### 📊 Pente (%) vs Cadence (couleur = progression de la distance)")

    import io

    if record_file is None:
        # Graphique vide avant chargement
        empty_df = pd.DataFrame({"pente_%": [], "cadence": [], "progress_%": []})
        empty_chart = (
            alt.Chart(empty_df)
            .mark_circle(size=40, opacity=0.75)
            .encode(
                x=alt.X("pente_%:Q", title="Pente (%)"),
                y=alt.Y("cadence:Q", title="Cadence (pas/min)"),
                color=alt.Color(
                    "progress_%:Q",
                    title="Progression (%)",
                    scale=alt.Scale(domain=[0, 100], range=["#00c853", "#ff5252"])
                )
            )
            .properties(height=380)
        )
        st.altair_chart(empty_chart, use_container_width=True)

    else:
        # Colonnes requises
        required = ["pente_%", "cadence", "distance", "timestamp"]
        if not all(c in df.columns for c in required):
            st.info("ℹ️ Il faut les colonnes 'pente_%', 'cadence', 'distance' et 'timestamp'.")
        else:
            total_dist = df["distance"].iloc[-1]
            if pd.isna(total_dist) or total_dist <= 0:
                st.info("ℹ️ Distance totale non valide pour calculer la progression (0–100%).")
            else:
                # Progression 0–100%
                df["progress_%"] = (df["distance"] / total_dist * 100).clip(0, 100)

                # --- Bornes dynamiques + défauts intelligents
                cad_series = pd.to_numeric(df["cadence"], errors="coerce").dropna()
                pente_series = pd.to_numeric(df["pente_%"], errors="coerce").dropna()

                cad_min_data = float(np.nanmin(cad_series)) if len(cad_series) else 0.0
                cad_max_data = float(np.nanmax(cad_series)) if len(cad_series) else 250.0
                pente_min_data = float(np.nanmin(pente_series)) if len(pente_series) else -100.0
                pente_max_data = float(np.nanmax(pente_series)) if len(pente_series) else 100.0

                cad_default = (max(cad_min_data, 40.0), min(cad_max_data, 190.0)) if cad_min_data < cad_max_data else (40.0, 190.0)
                pente_default = (max(pente_min_data, -30.0), min(pente_max_data, 30.0)) if pente_min_data < pente_max_data else (-30.0, 30.0)

                c1, c2, c3 = st.columns(3)
                with c1:
                    cad_range = st.slider(
                        "Cadence (pas/min)",
                        float(cad_min_data), float(cad_max_data),
                        (float(cad_default[0]), float(cad_default[1])),
                        step=1.0,
                        key="cadence_slider_cadence"
                    )
                with c2:
                    pente_range = st.slider(
                        "Pente (%)",
                        float(pente_min_data), float(pente_max_data),
                        (float(pente_default[0]), float(pente_default[1])),
                        step=0.5,
                        key="cadence_slider_pente"
                    )
                with c3:
                    prog_range = st.slider(
                        "Progression distance (%)",
                        0.0, 100.0,
                        (0.0, 100.0),  # ou (0.0, 25.0)
                        step=1.0,
                        key="cadence_slider_progress"
                    )

                # --- Filtrage
                chart_df = df[["pente_%", "cadence", "progress_%", "timestamp"]].dropna()
                chart_df = chart_df[
                    chart_df["cadence"].between(cad_range[0], cad_range[1]) &
                    chart_df["pente_%"].between(pente_range[0], pente_range[1]) &
                    chart_df["progress_%"].between(prog_range[0], prog_range[1])
                ]

                if chart_df.empty:
                    st.warning("Aucun point dans les plages sélectionnées. Ajuste les curseurs.")
                else:
                    # --- Graphique
                    scatter = (
                        alt.Chart(chart_df)
                        .mark_circle(size=40, opacity=0.8)
                        .encode(
                            x=alt.X("pente_%:Q", title="Pente (%)",
                                    scale=alt.Scale(domain=[pente_range[0], pente_range[1]])),
                            y=alt.Y("cadence:Q", title="Cadence (pas/min)",
                                    scale=alt.Scale(domain=[cad_range[0], cad_range[1]])),
                            color=alt.Color(
                                "progress_%:Q",
                                title="Progression km (%)",
                                scale=alt.Scale(domain=[prog_range[0], prog_range[1]], range=["#00c853", "#ff5252"])
                            ),
                            tooltip=[
                                alt.Tooltip("timestamp:T", title="Temps"),
                                alt.Tooltip("pente_%:Q", title="Pente (%)", format=".1f"),
                                alt.Tooltip("cadence:Q", title="Cadence (spm)", format=".0f"),
                                alt.Tooltip("progress_%:Q", title="Progression (%)", format=".1f"),
                            ],
                        )
                        .interactive()
                        .properties(height=380, width='container')
                    )

                    st.altair_chart(scatter, use_container_width=True)

                    # --- Export PNG (16:9 HD) avec mêmes filtres
                    try:
                        import altair_saver  # noqa: F401
                        export_width = 400
                        export_height = 450
                        scale_factor = 6

                        chart_to_save = scatter.properties(width=export_width, height=export_height)

                        buf = io.BytesIO()
                        chart_to_save.save(
                            buf,
                            format="png",
                            method="vl-convert",   # nécessite 'vl-convert-python'
                            scale_factor=scale_factor
                        )
                        buf.seek(0)

                        fname = (
                            f"cadence_vs_pente_"
                            f"cad{int(cad_range[0])}-{int(cad_range[1])}_"
                            f"pente{int(pente_range[0])}-{int(pente_range[1])}_"
                            f"prog{int(prog_range[0])}-{int(prog_range[1])}_16x9.png"
                        )

                        st.download_button(
                            label="📸 Télécharger le PNG (filtres appliqués, 16:9 HD)",
                            data=buf,
                            file_name=fname,
                            mime="image/png"
                        )

                    except Exception as e:
                        st.warning(
                            "⚠️ Problème avec la génération PNG :\n"
                            "Assure-toi d'avoir installé `vl-convert-python` :\n"
                            "```bash\npip install vl-convert-python\n```\n"
                            f"\nErreur : {e}"
                        )


# =========================
# === Glycémie vs Pente (couleur = progression) avec filtres + export PNG ===
# =========================
def render_glycemie_vs_pente():
    st.markdown("### 🩸 Glycémie vs Pente (couleur = progression sur la distance)")

    import io

    if record_file is None:
        # Graphique vide avant chargement
        empty_df = pd.DataFrame({"pente_%": [], "x_glucose_level_0_0": [], "progress_%": []})
        empty_chart = (
            alt.Chart(empty_df)
            .mark_circle(size=40, opacity=0.75)
            .encode(
                x=alt.X("pente_%:Q", title="Pente (%)"),
                y=alt.Y("x_glucose_level_0_0:Q", title="Glycémie"),
                color=alt.Color(
                    "progress_%:Q",
                    title="Progression (%)",
                    scale=alt.Scale(domain=[0, 100], range=["#00c853", "#ff5252"])
                )
            )
            .properties(height=380)
        )
        st.altair_chart(empty_chart, use_container_width=True)

    else:
        required = ["pente_%", "x_glucose_level_0_0", "distance"]
        if not all(c in df.columns for c in required):
            st.info("ℹ️ Graphique non affiché : il faut les colonnes 'pente_%', 'x_glucose_level_0_0' et 'distance'.")
        else:
            # Conversion sécurité
            df["x_glucose_level_0_0"] = pd.to_numeric(df["x_glucose_level_0_0"], errors="coerce")

            total_dist = df["distance"].iloc[-1]
            if pd.isna(total_dist) or total_dist <= 0:
                st.info("ℹ️ Distance totale non valide pour calculer la progression (0–100%).")
            else:
                # Progression 0–100% basée sur la distance
                df["progress_%"] = (df["distance"] / total_dist * 100).clip(0, 100)
                df["distance_km"] = df["distance"] / 1000.0

                # ---------- Contrôles de plages ----------
                # bornes issues des données (avec garde-fous)
                gly_series = df["x_glucose_level_0_0"].dropna()
                pente_series = df["pente_%"].dropna()

                gly_min_data = float(np.nanmin(gly_series)) if len(gly_series) else 0.0
                gly_max_data = float(np.nanmax(gly_series)) if len(gly_series) else 300.0
                pente_min_data = float(np.nanmin(pente_series)) if len(pente_series) else -100.0
                pente_max_data = float(np.nanmax(pente_series)) if len(pente_series) else 100.0

                # défauts intelligents (ex : 70–180 mg/dL, -10% à +10%, 0–100%)
                gly_default = (max(gly_min_data, 70.0), min(gly_max_data, 180.0)) if gly_min_data < gly_max_data else (70.0, 180.0)
                pente_default = (max(pente_min_data, -10.0), min(pente_max_data, 10.0)) if pente_min_data < pente_max_data else (-10.0, 10.0)

                c1, c2, c3 = st.columns(3)
                with c1:
                    gly_range = st.slider(
                        "Glycémie (min–max)",
                        float(gly_min_data), float(gly_max_data),
                        (float(gly_default[0]), float(gly_default[1])),
                        step=1.0,
                        key="gly_slider_glycemie"
                    )
                with c2:
                    pente_range = st.slider(
                        "Pente (%)",
                        float(pente_min_data), float(pente_max_data),
                        (float(pente_default[0]), float(pente_default[1])),
                        step=0.5,
                        key="gly_slider_pente"
                    )
                with c3:
                    prog_range = st.slider(
                        "Progression distance (%)",
                        0.0, 100.0,
                        (0.0, 100.0),  # ou (0.0, 25.0)
                        step=1.0,
                        key="gly_slider_progress"
                    )

                # ---------- Filtrage des données ----------
                chart_df = df[["pente_%", "x_glucose_level_0_0", "progress_%", "timestamp", "distance_km"]].dropna()
                chart_df = chart_df[
                    chart_df["x_glucose_level_0_0"].between(gly_range[0], gly_range[1]) &
                    chart_df["pente_%"].between(pente_range[0], pente_range[1]) &
                    chart_df["progress_%"].between(prog_range[0], prog_range[1])
                ]

                if chart_df.empty:
                    st.warning("Aucun point dans les plages sélectionnées. Ajuste tes curseurs.")
                else:
                    # ---------- Graphique ----------
                    scatter_gly = (
                        alt.Chart(chart_df)
                        .mark_circle(size=42, opacity=0.85)
                        .encode(
                            x=alt.X("pente_%:Q", title="Pente (%)",
                                    scale=alt.Scale(domain=[pente_range[0], pente_range[1]])),
                            y=alt.Y("x_glucose_level_0_0:Q", title="Glycémie",
                                    scale=alt.Scale(domain=[gly_range[0], gly_range[1]])),
                            color=alt.Color(
                                "progress_%:Q",
                                title="Progression (%)",
                                scale=alt.Scale(domain=[prog_range[0], prog_range[1]], range=["#00c853", "#ff5252"])
                            ),
                            tooltip=[
                                alt.Tooltip("timestamp:T", title="Temps"),
                                alt.Tooltip("pente_%:Q", title="Pente (%)", format=".1f"),
                                alt.Tooltip("x_glucose_level_0_0:Q", title="Glycémie"),
                                alt.Tooltip("distance_km:Q", title="Distance (km)", format=".2f"),
                                alt.Tooltip("progress_%:Q", title="Progression (%)", format=".1f"),
                            ],
                        )
                        .interactive()
                        .properties(height=380, width='container')
                    )

                    st.altair_chart(scatter_gly, use_container_width=True)

                    # ---------- Export PNG (16:9 HD) avec les mêmes filtres ----------
                    try:
                        import altair_saver  # noqa: F401
                        # Dimensions 16:9 (HD). Tu peux passer à 1920x1080 et/ou scale_factor=3 pour ultra net.
                        export_width = 400
                        export_height = 450
                        scale_factor = 6

                        chart_to_save = scatter_gly.properties(width=export_width, height=export_height)

                        buf = io.BytesIO()
                        chart_to_save.save(
                            buf,
                            format="png",
                            method="vl-convert",   # nécessite le paquet vl-convert-python
                            scale_factor=scale_factor
                        )
                        buf.seek(0)

                        # Nom de fichier parlant, basé sur les plages choisies
                        fname = (
                            f"glycemie_vs_pente_"
                            f"gly{int(gly_range[0])}-{int(gly_range[1])}_"
                            f"pente{int(pente_range[0])}-{int(pente_range[1])}_"
                            f"prog{int(prog_range[0])}-{int(prog_range[1])}_16x9.png"
                        )

                        st.download_button(
                            label="📸 Télécharger le PNG (filtres appliqués, 16:9 HD)",
                            data=buf,
                            file_name=fname,
                            mime="image/png"
                        )

                    except Exception as e:
                        st.warning(
                            "⚠️ Problème avec la génération PNG :\n"
                            "Assure-toi d'avoir installé `vl-convert-python` :\n"
                            "```bash\npip install vl-convert-python\n```\n"
                            f"\nErreur : {e}"
                        )



# =========================                   
# === ❤️ Fréquence cardiaque vs Pente (couleur = progression) | filtres + export PNG ===
# =========================
def render_hr_vs_pente():
    st.markdown("### ❤️ Fréquence cardiaque vs Pente (couleur = progression de la distance)")

    import io

    if record_file is None:
        # Graphique vide avant chargement
        empty_df = pd.DataFrame({"pente_%": [], "heart_rate": [], "progress_%": []})
        empty_chart = (
            alt.Chart(empty_df)
            .mark_circle(size=40, opacity=0.75)
            .encode(
                x=alt.X("pente_%:Q", title="Pente (%)"),
                y=alt.Y("heart_rate:Q", title="Fréquence cardiaque (bpm)"),
                color=alt.Color(
                    "progress_%:Q",
                    title="Progression (%)",
                    scale=alt.Scale(domain=[0, 100], range=["#00c853", "#ff5252"])
                )
            )
            .properties(height=380)
        )
        st.altair_chart(empty_chart, use_container_width=True)

    else:
        required = ["pente_%", "heart_rate", "distance", "timestamp"]
        if not all(c in df.columns for c in required):
            st.info("ℹ️ Il faut les colonnes 'pente_%', 'heart_rate', 'distance' et 'timestamp'.")
        else:
            total_dist = df["distance"].iloc[-1]
            if pd.isna(total_dist) or total_dist <= 0:
                st.info("ℹ️ Distance totale non valide pour calculer la progression (0–100%).")
            else:
                # Progression 0–100%
                df["progress_%"] = (df["distance"] / total_dist * 100).clip(0, 100)

                # Bornes dynamiques avec valeurs par défaut raisonnables
                hr_series = pd.to_numeric(df["heart_rate"], errors="coerce").dropna()
                pente_series = pd.to_numeric(df["pente_%"], errors="coerce").dropna()

                hr_min_data = float(np.nanmin(hr_series)) if len(hr_series) else 80.0
                hr_max_data = float(np.nanmax(hr_series)) if len(hr_series) else 210.0
                pente_min_data = float(np.nanmin(pente_series)) if len(pente_series) else -100.0
                pente_max_data = float(np.nanmax(pente_series)) if len(pente_series) else 100.0

                hr_default = (max(hr_min_data, 100.0), min(hr_max_data, 185.0)) if hr_min_data < hr_max_data else (100.0, 185.0)
                pente_default = (max(pente_min_data, -10.0), min(pente_max_data, 10.0)) if pente_min_data < pente_max_data else (-10.0, 10.0)

                c1, c2, c3 = st.columns(3)
                with c1:
                    hr_range = st.slider(
                        "Fréquence cardiaque (bpm)",
                        float(hr_min_data), float(hr_max_data),
                        (float(hr_default[0]), float(hr_default[1])),
                        step=1.0,
                        key="hr_slider_hr"
                    )
                with c2:
                    pente_range = st.slider(
                        "Pente (%)",
                        float(pente_min_data), float(pente_max_data),
                        (float(pente_default[0]), float(pente_default[1])),
                        step=0.5,
                        key="hr_slider_pente"
                    )
                with c3:
                    prog_range = st.slider(
                        "Progression distance (%)",
                        0.0, 100.0,
                        (0.0, 100.0),  # mets (0.0, 25.0) si tu veux un défaut 0–25%
                        step=1.0,
                        key="hr_slider_progress"
                    )

                # Filtrage
                chart_df = df[["pente_%", "heart_rate", "progress_%", "timestamp"]].dropna()
                chart_df = chart_df[
                    chart_df["heart_rate"].between(hr_range[0], hr_range[1]) &
                    chart_df["pente_%"].between(pente_range[0], pente_range[1]) &
                    chart_df["progress_%"].between(prog_range[0], prog_range[1])
                ]

                if chart_df.empty:
                    st.warning("Aucun point dans les plages sélectionnées. Ajuste les curseurs.")
                else:
                    # Graphique
                    scatter_hr = (
                        alt.Chart(chart_df)
                        .mark_circle(size=40, opacity=0.8)
                        .encode(
                            x=alt.X("pente_%:Q", title="Pente (%)",
                                    scale=alt.Scale(domain=[pente_range[0], pente_range[1]])),
                            y=alt.Y("heart_rate:Q", title="Fréquence cardiaque (bpm)",
                                    scale=alt.Scale(domain=[hr_range[0], hr_range[1]])),
                            color=alt.Color(
                                "progress_%:Q",
                                title="Progression (%)",
                                scale=alt.Scale(domain=[prog_range[0], prog_range[1]], range=["#00c853", "#ff5252"])
                            ),
                            tooltip=[
                                alt.Tooltip("timestamp:T", title="Temps"),
                                alt.Tooltip("pente_%:Q", title="Pente (%)", format=".1f"),
                                alt.Tooltip("heart_rate:Q", title="BPM", format=".0f"),
                                alt.Tooltip("progress_%:Q", title="Progression (%)", format=".1f"),
                            ],
                        )
                        .interactive()
                        .properties(height=380, width='container')
                    )

                    st.altair_chart(scatter_hr, use_container_width=True)

                    # Export PNG (16:9 HD) avec filtres appliqués
                    try:
                        import altair_saver  # noqa: F401

                        export_width = 400
                        export_height = 450
                        scale_factor = 6

                        chart_to_save = scatter_hr.properties(width=export_width, height=export_height)

                        buf = io.BytesIO()
                        chart_to_save.save(
                            buf,
                            format="png",
                            method="vl-convert",   # nécessite le paquet 'vl-convert-python'
                            scale_factor=scale_factor
                        )
                        buf.seek(0)

                        fname = (
                            f"hr_vs_pente_"
                            f"hr{int(hr_range[0])}-{int(hr_range[1])}_"
                            f"pente{int(pente_range[0])}-{int(pente_range[1])}_"
                            f"prog{int(prog_range[0])}-{int(prog_range[1])}_16x9.png"
                        )

                        st.download_button(
                            label="📸 Télécharger le PNG (filtres appliqués, 16:9 HD)",
                            data=buf,
                            file_name=fname,
                            mime="image/png"
                        )

                    except Exception as e:
                        st.warning(
                            "⚠️ Problème avec la génération PNG :\n"
                            "Assure-toi d'avoir installé `vl-convert-python` :\n"
                            "```bash\npip install vl-convert-python\n```\n"
                            f"\nErreur : {e}"
                        )



# =========================
# === 🗺️ Profil altimétrique + ❤️ FC lissée (60 pts) avec filtres + export PNG ===
# =========================
def render_profil_alt_fc():
    st.markdown("### 🗺️ Profil du parcours (Altitude) + ❤️ FC lissée (60 pts) sur la distance")

    import io

    if record_file is None:
        # Graphique vide avant chargement
        empty_df = pd.DataFrame({"distance_km": [], "enhanced_altitude": [], "hr60": []})
        empty_chart = (
            alt.Chart(empty_df)
            .mark_line()
            .encode(
                x=alt.X("distance_km:Q", title="Distance (km)"),
                y=alt.Y("enhanced_altitude:Q", title="Altitude (m)")
            )
            .properties(height=380)
        )
        st.altair_chart(empty_chart, use_container_width=True)

    else:
        required = ["distance", "enhanced_altitude", "heart_rate"]
        if not all(c in df.columns for c in required):
            st.info("ℹ️ Il faut les colonnes 'distance', 'enhanced_altitude' et 'heart_rate'.")
        else:
            # Distance en km
            df["distance_km"] = pd.to_numeric(df["distance"], errors="coerce") / 1000.0

            # Lissage FC sur 60 points (centré)
            hr_series = pd.to_numeric(df["heart_rate"], errors="coerce")
            df["hr60"] = hr_series.rolling(60, min_periods=30, center=True).mean()

            # Bornes automatiques
            dist_min = float(np.nanmin(df["distance_km"])) if df["distance_km"].notna().any() else 0.0
            dist_max = float(np.nanmax(df["distance_km"])) if df["distance_km"].notna().any() else 0.0
            alt_min = float(np.nanmin(df["enhanced_altitude"])) if df["enhanced_altitude"].notna().any() else 0.0
            alt_max = float(np.nanmax(df["enhanced_altitude"])) if df["enhanced_altitude"].notna().any() else 0.0
            hr_min = float(np.nanmin(df["hr60"])) if df["hr60"].notna().any() else 80.0
            hr_max = float(np.nanmax(df["hr60"])) if df["hr60"].notna().any() else 200.0

            if dist_max <= dist_min:
                st.info("ℹ️ Distance invalide dans les données.")
            else:
                # === Sliders ===
                c1, c2, c3 = st.columns(3)
                with c1:
                    d1, d2 = st.slider(
                        "Distance (km)",
                        min_value=float(np.floor(dist_min)),
                        max_value=float(np.ceil(dist_max)),
                        value=(float(dist_min), float(dist_max)),
                        step=0.1,
                        key="slider_dist_profile"
                    )
                with c2:
                    a1, a2 = st.slider(
                        "Altitude (m)",
                        min_value=float(np.floor(alt_min)),
                        max_value=float(np.ceil(alt_max)),
                        value=(float(alt_min), float(alt_max)),
                        step=5.0,
                        key="slider_alt_profile"
                    )
                with c3:
                    h1, h2 = st.slider(
                        "Fréquence cardiaque (bpm)",
                        min_value=float(np.floor(hr_min)),
                        max_value=float(np.ceil(hr_max)),
                        value=(float(max(90, hr_min)), float(min(185, hr_max))),
                        step=1.0,
                        key="slider_hr_profile"
                    )

                # Filtrage
                chart_df = df[["distance_km", "enhanced_altitude", "hr60"]].dropna(subset=["distance_km", "enhanced_altitude"])
                chart_df = chart_df[
                    (chart_df["distance_km"] >= d1) & (chart_df["distance_km"] <= d2) &
                    (chart_df["enhanced_altitude"].between(a1, a2)) &
                    (chart_df["hr60"].between(h1, h2))
                ]

                if chart_df.empty:
                    st.warning("Aucun point dans les plages sélectionnées.")
                else:
                    base = alt.Chart(chart_df).encode(
                        x=alt.X("distance_km:Q", title="Distance (km)", scale=alt.Scale(domain=[d1, d2]))
                    )

                    # Profil altimétrique (gris clair)
                    alt_line = base.mark_line(strokeWidth=1.5).encode(
                        y=alt.Y("enhanced_altitude:Q", title="Altitude (m)", scale=alt.Scale(domain=[a1, a2])),
                        color=alt.value("#b0bec5"),
                        tooltip=[
                            alt.Tooltip("distance_km:Q", title="Distance (km)", format=".2f"),
                            alt.Tooltip("enhanced_altitude:Q", title="Altitude (m)", format=".0f"),
                        ]
                    )

                    # FC lissée (rouge)
                    hr_df = chart_df.dropna(subset=["hr60"])
                    hr_line = alt.Chart(hr_df).encode(
                        x=alt.X("distance_km:Q"),
                        y=alt.Y("hr60:Q", title="BPM", scale=alt.Scale(domain=[h1, h2])),
                        color=alt.value("#ff5252"),
                        tooltip=[
                            alt.Tooltip("distance_km:Q", title="Distance (km)", format=".2f"),
                            alt.Tooltip("hr60:Q", title="FC lissée (bpm)", format=".0f"),
                        ]
                    ).mark_line(strokeWidth=1.5)

                    chart = alt.layer(alt_line, hr_line).resolve_scale(y='independent').properties(height=380, width='container')
                    st.altair_chart(chart, use_container_width=True)

                    # === Export PNG (16:9 HD) ===
                    try:
                        import altair_saver  # noqa: F401

                        export_width = 900
                        export_height = 600
                        scale_factor = 1

                        chart_to_save = chart.properties(width=export_width, height=export_height)

                        buf = io.BytesIO()
                        chart_to_save.save(
                            buf,
                            format="png",
                            method="vl-convert",   # nécessite 'vl-convert-python'
                            scale_factor=scale_factor
                        )
                        buf.seek(0)

                        fname = (
                            f"profil_altitude_fc_"
                            f"d{d1:.1f}-{d2:.1f}_"
                            f"a{int(a1)}-{int(a2)}_"
                            f"hr{int(h1)}-{int(h2)}_16x9.png"
                        )

                        st.download_button(
                            label="📸 Télécharger le PNG (16:9 HD, filtres appliqués)",
                            data=buf,
                            file_name=fname,
                            mime="image/png"
                        )

                    except Exception as e:
                        st.warning(
                            "⚠️ Problème avec la génération PNG :\n"
                            "Assure-toi d'avoir installé `vl-convert-python` :\n"
                            "```bash\npip install vl-convert-python\n```\n"
                            f"\nErreur : {e}"
                        )



# =========================
# === 🏁 Allure ajustée (min/km) vs Pente (%) | couleur = progression | filtres + export PNG ===
# =========================
def render_gas_vs_pente():
    st.markdown("### 🏁 Allure ajustée (min/km) vs Pente (%) — couleur = progression")

    import io

    if record_file is None:
        # Graphique vide avant chargement
        empty_df = pd.DataFrame({"pente_%": [], "pace_adj_min_km": [], "progress_%": []})
        empty_chart = (
            alt.Chart(empty_df)
            .mark_circle(size=40, opacity=0.75)
            .encode(
                x=alt.X("pente_%:Q", title="Pente (%)"),
                y=alt.Y("pace_adj_min_km:Q", title="Allure ajustée (min/km)"),
                color=alt.Color(
                    "progress_%:Q",
                    title="Progression (%)",
                    scale=alt.Scale(domain=[0, 100], range=["#00c853", "#ff5252"])
                )
            )
            .properties(height=380)
        )
        st.altair_chart(empty_chart, use_container_width=True)

    else:
        required = ["pente_%", "grade_adjusted_speed", "distance", "timestamp"]
        if not all(c in df.columns for c in required):
            st.info("ℹ️ Il faut les colonnes 'pente_%', 'grade_adjusted_speed', 'distance' et 'timestamp'.")
        else:
            # Conversion sécurisée
            gas = pd.to_numeric(df["grade_adjusted_speed"], errors="coerce")  # m/s
            # Allure (min/km) = 1000 / (m/s) / 60 ; si vitesse <= 0 → NaN
            df["pace_adj_min_km"] = np.where(gas > 0, (1000.0 / gas) / 60.0, np.nan)

            # Progression 0–100%
            total_dist = pd.to_numeric(df["distance"], errors="coerce").iloc[-1]
            if pd.isna(total_dist) or total_dist <= 0:
                st.info("ℹ️ Distance totale non valide pour calculer la progression (0–100%).")
            else:
                df["progress_%"] = (df["distance"] / total_dist * 100).clip(0, 100)

                # ---- Bornes dynamiques + défauts intelligents ----
                pace_series = df["pace_adj_min_km"].dropna()
                pente_series = pd.to_numeric(df["pente_%"], errors="coerce").dropna()

                # Garde-fous pour les bornes
                #pace_min_data = float(np.nanmin(pace_series)) if len(pace_series) else 3.0
                #pace_max_data = float(np.nanmax(pace_series)) if len(pace_series) else 20.0
                pace_min_data = 2.0
                pace_max_data = 30.0
                pente_min_data = float(np.nanmin(pente_series)) if len(pente_series) else -100.0
                pente_max_data = float(np.nanmax(pente_series)) if len(pente_series) else 100.0

                # Défauts "trail" raisonnables : 4:00–12:00 min/km, -15% à +15%, 0–100%
                pace_default = (max(pace_min_data, 4.0), min(pace_max_data, 12.0)) if pace_min_data < pace_max_data else (4.0, 12.0)
                pente_default = (max(pente_min_data, -15.0), min(pente_max_data, 15.0)) if pente_min_data < pente_max_data else (-15.0, 15.0)

                c1, c2, c3 = st.columns(3)
                with c1:
                    pace_range = st.slider(
                        "Allure ajustée (min/km)",
                        float(pace_min_data), float(pace_max_data),
                        (float(pace_default[0]), float(pace_default[1])),
                        step=0.1,
                        key="gas_slider_pace"
                    )
                with c2:
                    pente_range = st.slider(
                        "Pente (%)",
                        float(pente_min_data), float(pente_max_data),
                        (float(pente_default[0]), float(pente_default[1])),
                        step=0.5,
                        key="gas_slider_pente"
                    )
                with c3:
                    prog_range = st.slider(
                        "Progression distance (%)",
                        0.0, 100.0,
                        (0.0, 100.0),  # par ex. (0.0, 25.0) si tu veux commencer là
                        step=1.0,
                        key="gas_slider_progress"
                    )

                # ---- Filtrage ----
                chart_df = df[["pente_%", "pace_adj_min_km", "progress_%", "timestamp"]].dropna()
                chart_df = chart_df[
                    chart_df["pace_adj_min_km"].between(pace_range[0], pace_range[1]) &
                    chart_df["pente_%"].between(pente_range[0], pente_range[1]) &
                    chart_df["progress_%"].between(prog_range[0], prog_range[1])
                ]

                if chart_df.empty:
                    st.warning("Aucun point dans les plages sélectionnées. Ajuste les curseurs.")
                else:
                    # Scatter
                    scatter_gas = (
                        alt.Chart(chart_df)
                        .mark_circle(size=42, opacity=0.85)
                        .encode(
                            x=alt.X("pente_%:Q", title="Pente (%)",
                                    scale=alt.Scale(domain=[pente_range[0], pente_range[1]])),
                            y=alt.Y("pace_adj_min_km:Q", title="Allure ajustée (min/km)",
                                    scale=alt.Scale(domain=[pace_range[0], pace_range[1]])),
                            color=alt.Color(
                                "progress_%:Q",
                                title="Progression (%)",
                                scale=alt.Scale(domain=[prog_range[0], prog_range[1]], range=["#00c853", "#ff5252"])
                            ),
                            tooltip=[
                                alt.Tooltip("timestamp:T", title="Temps"),
                                alt.Tooltip("pente_%:Q", title="Pente (%)", format=".1f"),
                                alt.Tooltip("pace_adj_min_km:Q", title="Allure (min/km)", format=".2f"),
                                alt.Tooltip("progress_%:Q", title="Progression (%)", format=".1f"),
                            ],
                        )
                        .interactive()
                        .properties(height=380, width='container')
                    )

                    st.altair_chart(scatter_gas, use_container_width=True)

                    # ---- Export PNG (16:9 HD) avec filtres appliqués ----
                    try:
                        import altair_saver  # noqa: F401

                        export_width = 400
                        export_height = 450
                        scale_factor = 6

                        chart_to_save = scatter_gas.properties(width=export_width, height=export_height)

                        buf = io.BytesIO()
                        chart_to_save.save(
                            buf,
                            format="png",
                            method="vl-convert",   # nécessite 'vl-convert-python'
                            scale_factor=scale_factor
                        )
                        buf.seek(0)

                        fname = (
                            f"allure_ajustee_vs_pente_"
                            f"pace{pace_range[0]:.1f}-{pace_range[1]:.1f}_"
                            f"pente{int(pente_range[0])}-{int(pente_range[1])}_"
                            f"prog{int(prog_range[0])}-{int(prog_range[1])}_16x9.png"
                        )

                        st.download_button(
                            label="📸 Télécharger le PNG (filtres appliqués, 16:9 HD)",
                            data=buf,
                            file_name=fname,
                            mime="image/png"
                        )

                    except Exception as e:
                        st.warning(
                            "⚠️ Problème avec la génération PNG :\n"
                            "Assure-toi d'avoir installé `vl-convert-python` :\n"
                            "```bash\npip install vl-convert-python\n```\n"
                            f"\nErreur : {e}"
                        )



# =========================
# === 🩸 Glycémie vs ❤️ Fréquence cardiaque | couleur = progression | filtres + export PNG ===
# =========================
def render_glycemie_vs_hr():
    st.markdown("### 🩸 Glycémie vs ❤️ Fréquence cardiaque (couleur = progression sur la distance)")

    import io

    if record_file is None:
        # Graphique vide avant chargement
        empty_df = pd.DataFrame({"heart_rate": [], "x_glucose_level_0_0": [], "progress_%": []})
        empty_chart = (
            alt.Chart(empty_df)
            .mark_circle(size=40, opacity=0.75)
            .encode(
                x=alt.X("heart_rate:Q", title="Fréquence cardiaque (bpm)"),
                y=alt.Y("x_glucose_level_0_0:Q", title="Glycémie"),
                color=alt.Color(
                    "progress_%:Q",
                    title="Progression (%)",
                    scale=alt.Scale(domain=[0, 100], range=["#00c853", "#ff5252"])
                )
            )
            .properties(height=380)
        )
        st.altair_chart(empty_chart, use_container_width=True)

    else:
        required = ["heart_rate", "x_glucose_level_0_0", "distance", "timestamp"]
        if not all(c in df.columns for c in required):
            st.info("ℹ️ Il faut les colonnes 'heart_rate', 'x_glucose_level_0_0', 'distance' et 'timestamp'.")
        else:
            # Conversions sécurisées
            df["x_glucose_level_0_0"] = pd.to_numeric(df["x_glucose_level_0_0"], errors="coerce")
            df["heart_rate"] = pd.to_numeric(df["heart_rate"], errors="coerce")
            df["distance_km"] = pd.to_numeric(df["distance"], errors="coerce") / 1000.0

            # Progression 0–100%
            total_dist = pd.to_numeric(df["distance"], errors="coerce").iloc[-1]
            if pd.isna(total_dist) or total_dist <= 0:
                st.info("ℹ️ Distance totale non valide pour calculer la progression (0–100%).")
            else:
                df["progress_%"] = (df["distance"] / total_dist * 100).clip(0, 100)

                # ---- Bornes dynamiques + défauts raisonnables ----
                gly_series = df["x_glucose_level_0_0"].dropna()
                hr_series = df["heart_rate"].dropna()

                gly_min_data = float(np.nanmin(gly_series)) if len(gly_series) else 50.0
                gly_max_data = float(np.nanmax(gly_series)) if len(gly_series) else 300.0
                hr_min_data = float(np.nanmin(hr_series)) if len(hr_series) else 80.0
                hr_max_data = float(np.nanmax(hr_series)) if len(hr_series) else 210.0

                gly_default = (max(gly_min_data, 70.0), min(gly_max_data, 180.0)) if gly_min_data < gly_max_data else (70.0, 180.0)
                hr_default = (max(hr_min_data, 100.0), min(hr_max_data, 185.0)) if hr_min_data < hr_max_data else (100.0, 185.0)

                c1, c2, c3 = st.columns(3)
                with c1:
                    gly_range = st.slider(
                        "Glycémie",
                        float(gly_min_data), float(gly_max_data),
                        (float(gly_default[0]), float(gly_default[1])),
                        step=1.0,
                        key="gly_hr_slider_glycemie"
                    )
                with c2:
                    hr_range = st.slider(
                        "Fréquence cardiaque (bpm)",
                        float(hr_min_data), float(hr_max_data),
                        (float(hr_default[0]), float(hr_default[1])),
                        step=1.0,
                        key="gly_hr_slider_hr"
                    )
                with c3:
                    prog_range = st.slider(
                        "Progression distance (%)",
                        0.0, 100.0,
                        (0.0, 100.0),  # ex. (0.0, 25.0) pour début de course
                        step=1.0,
                        key="gly_hr_slider_progress"
                    )

                # ---- Filtrage ----
                chart_df = df[["heart_rate", "x_glucose_level_0_0", "progress_%", "timestamp", "distance_km"]].dropna()
                chart_df = chart_df[
                    chart_df["x_glucose_level_0_0"].between(gly_range[0], gly_range[1]) &
                    chart_df["heart_rate"].between(hr_range[0], hr_range[1]) &
                    chart_df["progress_%"].between(prog_range[0], prog_range[1])
                ]

                if chart_df.empty:
                    st.warning("Aucun point dans les plages sélectionnées. Ajuste les curseurs.")
                else:
                    # Scatter
                    scatter_gly_hr = (
                        alt.Chart(chart_df)
                        .mark_circle(size=42, opacity=0.85)
                        .encode(
                            x=alt.X("heart_rate:Q", title="Fréquence cardiaque (bpm)",
                                    scale=alt.Scale(domain=[hr_range[0], hr_range[1]])),
                            y=alt.Y("x_glucose_level_0_0:Q", title="Glycémie",
                                    scale=alt.Scale(domain=[gly_range[0], gly_range[1]])),
                            color=alt.Color(
                                "progress_%:Q",
                                title="Progression (%)",
                                scale=alt.Scale(domain=[prog_range[0], prog_range[1]], range=["#00c853", "#ff5252"])
                            ),
                            tooltip=[
                                alt.Tooltip("timestamp:T", title="Temps"),
                                alt.Tooltip("heart_rate:Q", title="BPM", format=".0f"),
                                alt.Tooltip("x_glucose_level_0_0:Q", title="Glycémie"),
                                alt.Tooltip("distance_km:Q", title="Distance (km)", format=".2f"),
                                alt.Tooltip("progress_%:Q", title="Progression (%)", format=".1f"),
                            ],
                        )
                        .interactive()
                        .properties(height=380, width='container')
                    )

                    st.altair_chart(scatter_gly_hr, use_container_width=True)

                    # ---- Export PNG (16:9 HD) avec filtres appliqués ----
                    try:
                        import altair_saver  # noqa: F401

                        export_width = 400
                        export_height = 450
                        scale_factor = 6

                        chart_to_save = scatter_gly_hr.properties(width=export_width, height=export_height)

                        buf = io.BytesIO()
                        chart_to_save.save(
                            buf,
                            format="png",
                            method="vl-convert",   # nécessite 'vl-convert-python'
                            scale_factor=scale_factor
                        )
                        buf.seek(0)

                        fname = (
                            f"glycemie_vs_fc_"
                            f"gly{int(gly_range[0])}-{int(gly_range[1])}_"
                            f"hr{int(hr_range[0])}-{int(hr_range[1])}_"
                            f"prog{int(prog_range[0])}-{int(prog_range[1])}_16x9.png"
                        )

                        st.download_button(
                            label="📸 Télécharger le PNG (filtres appliqués, 16:9 HD)",
                            data=buf,
                            file_name=fname,
                            mime="image/png"
                        )

                    except Exception as e:
                        st.warning(
                            "⚠️ Problème avec la génération PNG :\n"
                            "Assure-toi d'avoir installé `vl-convert-python` :\n"
                            "```bash\npip install vl-convert-python\n```\n"
                            f"\nErreur : {e}"
                        )



# =========================
# === 🧱 Temps en zones glycémiques (barre empilée, sans labels) + tableau + PNG ===
# =========================
def render_temps_zones_glycemiques():
    st.markdown("### 🧱 Répartition du temps en zones glycémiques (en %)")

    import io

    if record_file is None:
        empty_df = pd.DataFrame({"bar": [], "zone": [], "pourcentage": []})
        empty_chart = (
            alt.Chart(empty_df)
            .mark_bar()
            .encode(
                x=alt.X("bar:N", title=None),
                y=alt.Y("pourcentage:Q", title="Pourcentage du temps (%)"),
                color=alt.Color("zone:N", title="Zone")
            )
            .properties(height=220, width=420)   # largeur fixe
        )
        # centre le graphique
        _l, mid, _r = st.columns([1, 0.8, 1])
        with mid:
            st.altair_chart(empty_chart, use_container_width=False)

    else:
        required = ["timestamp", "x_glucose_level_0_0"]
        if not all(c in df.columns for c in required):
            st.info("ℹ️ Il faut les colonnes 'timestamp' et 'x_glucose_level_0_0'.")
        else:
            gdf = df[["timestamp", "x_glucose_level_0_0"]].dropna().copy()
            gdf["timestamp"] = pd.to_datetime(gdf["timestamp"], errors="coerce")
            gdf["x_glucose_level_0_0"] = pd.to_numeric(gdf["x_glucose_level_0_0"], errors="coerce")
            gdf = gdf.dropna(subset=["timestamp", "x_glucose_level_0_0"]).sort_values("timestamp")

            if gdf.empty or gdf["timestamp"].nunique() < 2:
                st.info("ℹ️ Pas assez de points temporels pour calculer les durées.")
            else:
                dt = gdf["timestamp"].diff().dt.total_seconds()
                med = float(np.nanmedian(dt[dt > 0])) if (dt > 0).any() else 1.0
                dt = dt.fillna(med)
                dt = dt.where(dt > 0, med)
                gdf["dt_sec"] = dt

                z1 = gdf["x_glucose_level_0_0"] < 70
                z2 = (gdf["x_glucose_level_0_0"] >= 70) & (gdf["x_glucose_level_0_0"] < 180)
                z3 = (gdf["x_glucose_level_0_0"] >= 180) & (gdf["x_glucose_level_0_0"] <= 240)
                z4 = gdf["x_glucose_level_0_0"] > 240

                t1, t2, t3, t4 = (
                    gdf.loc[z1, "dt_sec"].sum(),
                    gdf.loc[z2, "dt_sec"].sum(),
                    gdf.loc[z3, "dt_sec"].sum(),
                    gdf.loc[z4, "dt_sec"].sum(),
                )
                total = t1 + t2 + t3 + t4

                if total <= 0:
                    st.info("ℹ️ Aucune durée exploitable pour ces zones.")
                else:
                    p1, p2, p3, p4 = (100*t1/total, 100*t2/total, 100*t3/total, 100*t4/total)

                    res = pd.DataFrame({
                        "bar": ["Temps en zone"] * 4,
                        "zone": ["<70 mg/dL", "70–180 mg/dL", "180–240 mg/dL", ">240 mg/dL"],
                        "pourcentage": [p1, p2, p3, p4]
                    })

                    color_scale = alt.Scale(
                        domain=["<70 mg/dL", "70–180 mg/dL", "180–240 mg/dL", ">240 mg/dL"],
                        range=["#ff5252", "#00c853", "#ffd600", "#ff1744"]
                    )

                    bars = (
                        alt.Chart(res)
                        .mark_bar()
                        .encode(
                            x=alt.X("bar:N", title=None, axis=alt.Axis(labels=False, ticks=False)),
                            y=alt.Y("pourcentage:Q", stack="zero", title="Temps en zone (%)",
                                    scale=alt.Scale(domain=[0, 100])),
                            color=alt.Color("zone:N", title="Zone", scale=color_scale),
                            tooltip=[
                                alt.Tooltip("zone:N", title="Zone"),
                                alt.Tooltip("pourcentage:Q", title="Temps (%)", format=".1f"),
                            ]
                        )
                        .properties(height=240, width=420)   # largeur fixe
                    )

                    # centre le graphique (évite l’étirement responsive)
                    _l, mid, _r = st.columns([1, 0.8, 1])
                    with mid:
                        st.altair_chart(bars, use_container_width=False)

                    # Tableau récap
                    def fmt_hms(sec: float) -> str:
                        sec = max(0, int(round(sec)))
                        h = sec // 3600
                        m = (sec % 3600) // 60
                        s = sec % 60
                        return f"{h:d}:{m:02d}:{s:02d}"

                    table_df = pd.DataFrame({
                        "Zone": ["<70 mg/dL", "70–180 mg/dL", "180–240 mg/dL", ">240 mg/dL"],
                        "Temps (h:mm:ss)": [fmt_hms(t1), fmt_hms(t2), fmt_hms(t3), fmt_hms(t4)],
                        "Pourcentage (%)": [round(p1, 1), round(p2, 1), round(p3, 1), round(p4, 1)]
                    })

                    st.markdown("#### 📋 Récapitulatif")
                    st.dataframe(table_df, use_container_width=True)

                    # Export CSV du tableau récapitulatif
                    csv_buf = io.StringIO()
                    table_df.to_csv(csv_buf, index=False)
                    st.download_button(
                        label="💾 Télécharger le CSV du tableau",
                        data=csv_buf.getvalue(),
                        file_name="temps_zones_glycemiques.csv",
                        mime="text/csv"
                    )

                    # === Export PNG (16:9 HD) sans labels ===
                    try:
                        import altair_saver  # noqa: F401
                        export_width = 400
                        export_height = 450
                        scale_factor = 6

                        chart_to_save = bars.properties(width=export_width, height=export_height)

                        buf = io.BytesIO()
                        chart_to_save.save(
                            buf,
                            format="png",
                            method="vl-convert",   # nécessite 'vl-convert-python'
                            scale_factor=scale_factor
                        )
                        buf.seek(0)

                        fname = "temps_en_zones_glycemiques_16x9.png"
                        st.download_button(
                            label="📸 Télécharger le PNG (16:9 HD)",
                            data=buf,
                            file_name=fname,
                            mime="image/png"
                        )
                    except Exception as e:
                        st.warning(
                            "⚠️ Problème avec la génération PNG :\n"
                            "Assure-toi d'avoir installé `vl-convert-python` :\n"
                            "```bash\npip install vl-convert-python\n```\n"
                            f"\nErreur : {e}"
                        )

#=========liste des sections disponibles=========
SECTIONS = [
    ("❤️ HR vs Pente", render_hr_vs_pente),
    ("🗺️ Profil + FC", render_profil_alt_fc),
    ("🩸 Glycémie vs Pente", render_glycemie_vs_pente),
    ("🚶 Cadence vs Pente", render_cadence_vs_pente),
    ("🏁 Allure ajustée vs Pente", render_gas_vs_pente),
    ("🩸 Glycémie vs ❤️ FC", render_glycemie_vs_hr),
    ("🧱 Temps en zones glycémiques", render_temps_zones_glycemiques),
    # ➕ plus tard : ("📊 Autre analyse", render_autre_analyse),
]
#--------- CONTROLEUR D'AFFICHAGE DES SECTIONS ---------
st.subheader("Affichage")
layout = st.radio(
    "Disposition",
    ["Empilée", "Côte à côte (2 par ligne)", "Plein écran"],
    horizontal=True,
    key="layout_choice"
)

if layout == "Empilée":
    for title, fn in SECTIONS:
        fn()
        st.divider()

elif layout == "Côte à côte (2 par ligne)":
    # 2 colonnes (qui s’empilent automatiquement sur petit écran)
    per_row = 2
    for i in range(0, len(SECTIONS), per_row):
        cols = st.columns(per_row, gap="large")
        for col, (title, fn) in zip(cols, SECTIONS[i:i+per_row]):
            with col:
                fn()

elif layout == "Plein écran":
    # Choix de la section à afficher seule
    names = [name for name, _ in SECTIONS]
    pick = st.selectbox("Choisir la section", names, index=0)
    dict(SECTIONS)[pick]()

# =========================
# === 📏 Analyse par tronçon & par kilomètre ===
# =========================
st.markdown("### 📏 Analyse par kilomètre et par tronçon (avec profil)")

# Préparations sûres : distance_km, dt_sec, vitesses, paces
wrk = df.copy()

# Distance en km
wrk["distance_km"] = pd.to_numeric(wrk.get("distance", np.nan), errors="coerce") / 1000.0

# Tri temporel + delta temps (s) robuste
wrk["timestamp"] = pd.to_datetime(wrk["timestamp"], errors="coerce")
wrk = wrk.dropna(subset=["timestamp", "distance_km"]).sort_values("timestamp")
dt = wrk["timestamp"].diff().dt.total_seconds()
med = float(np.nanmedian(dt[dt > 0])) if (dt > 0).any() else 1.0
dt = dt.fillna(med)
dt = dt.where(dt > 0, med)
wrk["dt_sec"] = dt

# Delta altitude si absent
if "delta_altitude" not in wrk.columns and "enhanced_altitude" in wrk.columns:
    wrk["delta_altitude"] = pd.to_numeric(wrk["enhanced_altitude"], errors="coerce").diff()
# D+ / D-
wrk["D+"] = wrk["delta_altitude"].apply(lambda x: x if pd.notna(x) and x > 0 else 0)
wrk["D-"] = wrk["delta_altitude"].apply(lambda x: -x if pd.notna(x) and x < 0 else 0)

# Vitesse instantanée m/s : privilégier enhanced_speed, sinon calcul
if "enhanced_speed" in wrk.columns:
    speed_ms = pd.to_numeric(wrk["enhanced_speed"], errors="coerce")
else:
    dd = pd.to_numeric(wrk["distance"], errors="coerce").diff()
    speed_ms = dd / wrk["dt_sec"]  # m/s

# Vitesse ajustée (m/s) si dispo
gas_ms = pd.to_numeric(wrk.get("grade_adjusted_speed", np.nan), errors="coerce")

# Allure (min/km) depuis m/s
def ms_to_pace_min_km(v_ms):
    # pace = 1000 / v_ms / 60
    return np.where(v_ms > 0, (1000.0 / v_ms) / 60.0, np.nan)

wrk["pace_min_km"] = ms_to_pace_min_km(speed_ms)
wrk["pace_adj_min_km"] = ms_to_pace_min_km(gas_ms)

# Nettoie métriques utiles
wrk["heart_rate"] = pd.to_numeric(wrk.get("heart_rate", np.nan), errors="coerce")
wrk["cadence"] = pd.to_numeric(wrk.get("cadence", np.nan), errors="coerce")
wrk["x_glucose_level_0_0"] = pd.to_numeric(wrk.get("x_glucose_level_0_0", np.nan), errors="coerce")
wrk["enhanced_altitude"] = pd.to_numeric(wrk.get("enhanced_altitude", np.nan), errors="coerce")

# Slider distance sur tout le parcours
if wrk["distance_km"].notna().any():
    dist_min = float(np.nanmin(wrk["distance_km"]))
    dist_max = float(np.nanmax(wrk["distance_km"]))
else:
    dist_min, dist_max = 0.0, 0.0

if dist_max <= dist_min:
    st.info("ℹ️ Distance invalide dans les données.")
else:
    d1, d2 = st.slider(
        "Sélection du tronçon (km)",
        min_value=float(np.floor(dist_min)),
        max_value=float(np.ceil(dist_max)),
        value=(float(dist_min), float(dist_max)),
        step=0.1,
        key="segment_slider_km"
    )

    # Filtre sur le tronçon
    seg = wrk[(wrk["distance_km"] >= d1) & (wrk["distance_km"] <= d2)].copy()

# --- Profil altimétrique du tronçon ---
if seg.empty or seg["enhanced_altitude"].isna().all():
    st.info("ℹ️ Pas d'altitude exploitable sur ce tronçon.")
else:
    import altair as alt

    # Bornes altitude sur le tronçon sélectionné
    alt_seg = seg["enhanced_altitude"].dropna()
    alt_min_seg = float(np.nanmin(alt_seg))
    alt_max_seg = float(np.nanmax(alt_seg))

    # Slider pour forcer l'altitude minimale d'affichage
    c_alt1, c_alt2 = st.columns([1, 3])
    with c_alt1:
        alt_min_display = st.slider(
            "Altitude min (m)",
            min_value=float(np.floor(alt_min_seg)),
            max_value=float(np.floor(max(alt_max_seg - 1, alt_min_seg + 1))),
            value=float(np.floor(alt_min_seg)),
            step=5.0,
            key="segment_slider_alt_min"
        )
    with c_alt2:
        st.caption(f"Plage alt. tronçon : {int(alt_min_seg)}–{int(alt_max_seg)} m")

    # Y-scale : [altitude min choisie ; altitude max du tronçon]
    y_scale = alt.Scale(domain=[alt_min_display, float(np.ceil(alt_max_seg))])

    prof = (
        alt.Chart(seg.dropna(subset=["distance_km", "enhanced_altitude"]))
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("distance_km:Q", title="Distance (km)", scale=alt.Scale(domain=[d1, d2])),
            y=alt.Y("enhanced_altitude:Q", title="Altitude (m)", scale=y_scale),
            color=alt.value("#00c853"),
            tooltip=[
                alt.Tooltip("distance_km:Q", title="Distance (km)", format=".2f"),
                alt.Tooltip("enhanced_altitude:Q", title="Altitude (m)", format=".0f"),
            ]
        )
        .properties(height=220, width='container')
    )

    st.altair_chart(prof, use_container_width=True)


    # ---------- Tableau par kilomètre (entiers de km dans la plage) ----------
    # km_bin = km entier courant
    seg["km_bin"] = np.floor(seg["distance_km"]).astype("Int64")

    # On garde seulement les km entiers qui croisent [d1,d2]
    km_bins = seg["km_bin"].dropna().unique()
    km_bins = np.sort(km_bins[~pd.isna(km_bins)])

    def fmt_hms(sec: float) -> str:
        s = max(0, int(round(sec)))
        h = s // 3600
        m = (s % 3600) // 60
        ss = s % 60
        return f"{h:d}:{m:02d}:{ss:02d}"

    def fmt_pace_min_km(p):
        if pd.isna(p):
            return ""
        total_sec = int(round(p * 60.0))
        mm = total_sec // 60
        ss = total_sec % 60
        return f"{mm:d}:{ss:02d}"

#    rows = []
#    for k in km_bins:
#        bin_df = seg[seg["km_bin"] == k]
#        if bin_df.empty:
#            continue
#        dist_km = float(bin_df["distance_km"].max() - bin_df["distance_km"].min())
#        t_sec = float(bin_df["dt_sec"].sum())
#        dplus = float(bin_df["D+"].sum())
#        dminus = float(bin_df["D-"].sum())
#        cad_mean = float(bin_df["cadence"].mean())
#        gly_mean = float(bin_df["x_glucose_level_0_0"].mean())
#        gly_max = float(bin_df["x_glucose_level_0_0"].max())
#        gly_min = float(bin_df["x_glucose_level_0_0"].min())
#        hr_mean = float(bin_df["heart_rate"].mean())
#        pace_mean = float(bin_df["pace_min_km"].mean())
#        pace_adj_mean = float(bin_df["pace_adj_min_km"].mean())
#
#        rows.append({
#            "Km": int(k),
#            "Distance (km)": round(dist_km, 2),
#            "D+ (m)": round(dplus, 0),
#            "D- (m)": round(dminus, 0),
#            "Temps": fmt_hms(t_sec),
#            "Cadence (spm)": round(cad_mean, 0) if not np.isnan(cad_mean) else "",
#            "Gly moyenne": round(gly_mean, 1) if not np.isnan(gly_mean) else "",
#            "Gly max": round(gly_max, 1) if not np.isnan(gly_max) else "",
#            "Gly min": round(gly_min, 1) if not np.isnan(gly_min) else "",
#            "FC moyenne (bpm)": round(hr_mean, 0) if not np.isnan(hr_mean) else "",
#            "Allure moy (min/km)": fmt_pace_min_km(pace_mean),
#            "Allure ajustée moy (min/km)": fmt_pace_min_km(pace_adj_mean),
#        })

#    st.markdown("#### 📚 Tableau par kilomètre (sur le tronçon sélectionné)")
#    if rows:
#        km_table = pd.DataFrame(rows)
#        km_table = km_table.sort_values("Km")
#        st.dataframe(km_table, use_container_width=True)
#    else:
#        st.info("ℹ️ Aucun kilomètre entier détecté dans la plage choisie (essayez d’élargir légèrement).")

    # ---------- Tableau récap du tronçon ----------
    if not seg.empty:
        dist_seg_km = float(seg["distance_km"].max() - seg["distance_km"].min())
        t_seg_sec = float(seg["dt_sec"].sum())
        dplus_seg = float(seg["D+"].sum())
        dminus_seg = float(seg["D-"].sum())
        gly_seg_mean = float(seg["x_glucose_level_0_0"].mean())

        # Vitesse ascensionnelle (m/h) = D+ / (temps_h)
        hours = t_seg_sec / 3600.0 if t_seg_sec > 0 else np.nan
        v_asc_mh = dplus_seg / hours if hours and hours > 0 else np.nan

        # Allure moyenne "réelle" = temps total / distance réelle
        pace_seg = (t_seg_sec / 60.0) / dist_seg_km if dist_seg_km > 0 else np.nan

        # Allure ajustée via distance équivalente "plat"
        seg_gas_ms = pd.to_numeric(seg.get("grade_adjusted_speed", np.nan), errors="coerce")
        equiv_km_seg = np.nansum(np.where(seg_gas_ms > 0, seg_gas_ms, np.nan) * seg["dt_sec"]) / 1000.0
        pace_adj_seg = (t_seg_sec / 60.0) / equiv_km_seg if equiv_km_seg and equiv_km_seg > 0 else np.nan

        cad_seg = float(seg["cadence"].mean())
        hr_seg = float(seg["heart_rate"].mean())

        # --- Tableau "une métrique par ligne" ---
        lignes = [
            ("Distance (km)", f"{dist_seg_km:.2f}"),
            ("Temps", fmt_hms(t_seg_sec)),
            ("Gly moyenne", "" if np.isnan(gly_seg_mean) else f"{gly_seg_mean:.1f}"),
            ("D+ (m)", f"{dplus_seg:.0f}"),
            ("D- (m)", f"{dminus_seg:.0f}"),
            ("Vitesse ascensionnelle (m/h)", "" if np.isnan(v_asc_mh) else f"{v_asc_mh:.0f}"),
            ("Allure moy (min/km)", fmt_pace_min_km(pace_seg)),
            ("Allure ajustée moy (min/km)", fmt_pace_min_km(pace_adj_seg)),
            ("Cadence moy (spm)", "" if np.isnan(cad_seg) else f"{cad_seg:.0f}"),
            ("FC moyenne (bpm)", "" if np.isnan(hr_seg) else f"{hr_seg:.0f}"),
        ]

        recap_long = pd.DataFrame(lignes, columns=["Métrique", "Valeur"])

        st.markdown("#### 🧾 Récapitulatif du tronçon sélectionné")
        # Option : largeur fixe et centrage léger
        _l, mid, _r = st.columns([1, 2, 1])
        with mid:
            st.table(recap_long)  # table statique = plus lisible pour un récap



# =========================
# === 🧮 Tableau croisé : Cadence moyenne selon Pente (%) et Distance (km) ===
# =========================
st.markdown("### 🧮 Cadence moyenne selon la pente et la distance (km)")

required_cols = {"pente_%", "cadence", "distance"}
if record_file is not None and required_cols.issubset(df.columns):
    # Distance totale
    total_dist = pd.to_numeric(df["distance"], errors="coerce").iloc[-1]
    if pd.notna(total_dist) and total_dist > 0:
        # Types numériques
        df["pente_%"] = pd.to_numeric(df["pente_%"], errors="coerce")
        df["cadence"] = pd.to_numeric(df["cadence"], errors="coerce")
        df["distance"] = pd.to_numeric(df["distance"], errors="coerce")

        # Distance en km pour l'affichage colonnes (si déjà en km, c'est ok aussi)
        # Heuristique simple : si total_dist > 1000 on suppose mètres -> convertir en km
        total_dist_km = total_dist / 1000.0 if total_dist > 1000 else total_dist
        df["dist_km"] = df["distance"] / 1000.0 if total_dist > 1000 else df["distance"]

        # --- Segmentation pente (5%) + bornes (inchangé)
        edges_pente = [-np.inf] + list(np.arange(-40, 45, 5)) + [np.inf]
        labels_pente = (["< -40%"] + [f"{b} à {b+5}%" for b in range(-40, 40, 5)] + ["> 40%"])
        df["pente_bin"] = pd.cut(df["pente_%"], bins=edges_pente, labels=labels_pente, include_lowest=True, right=False)
        df["pente_bin"] = df["pente_bin"].cat.set_categories(labels_pente, ordered=True)

        # --- Utilitaires robustes pour les bornes de découpe
        def sanitize_bins(bins: np.ndarray, min_len: int = 2) -> np.ndarray:
            """Rend les bornes strictement croissantes et sans doublons (flottants)."""
            bins = np.array(bins, dtype=float)
            # Arrondir pour réduire les artefacts binaires
            bins = np.round(bins, 9)
            # Supprimer les doublons
            bins = np.unique(bins)
            # Garantir au moins 2 bornes
            if bins.size < min_len:
                return np.array([0.0, float(total_dist_km)])
            # S'assurer strictement croissant (si des diffs nulles subsistent)
            diffs = np.diff(bins)
            if np.any(diffs <= 0):
                # Corrige en imposant une croissance minimale epsilon
                eps = 1e-6
                for i in range(1, bins.size):
                    if bins[i] <= bins[i-1]:
                        bins[i] = bins[i-1] + eps
            return bins

        # --- Choix du découpage des colonnes (par % -> libellés en km, OU par km)
        mode_cols = st.radio(
            "Découpage des colonnes",
            ["Par % du parcours (libellés en km)", "Par kilomètres (taille fixe)"],
            horizontal=True
        )

        if mode_cols == "Par % du parcours (libellés en km)":
            step_pct = st.slider("Taille d'une tranche (%)", min_value=5, max_value=50, value=10, step=5)
            # bornes en % puis conversion en km
            edges_pct = np.arange(0, 100, step_pct)
            if edges_pct[-1] != 100:
                edges_pct = np.append(edges_pct, 100)
            edges_km = (edges_pct / 100.0) * total_dist_km
            # Nettoyage/robustesse
            edges_km = sanitize_bins(edges_km)
            # Forcer exactement 0 et distance totale en extrémités
            edges_km[0] = 0.0
            edges_km[-1] = float(total_dist_km)
        else:
            # Par km
            default_step_km = max(1.0, round(total_dist_km / 16, 1))
            step_km = st.slider(
                "Taille d'une tranche (km)",
                min_value=0.5,
                max_value=max(1.0, float(total_dist_km)),
                value=float(default_step_km),
                step=0.5
            )
            edges_km = np.arange(0, total_dist_km + step_km, step_km)
            # Nettoyage/robustesse
            edges_km = sanitize_bins(edges_km)
            # Forcer exactement 0 et distance totale en extrémités
            edges_km[0] = 0.0
            edges_km[-1] = float(total_dist_km)

        # Labels en km (0–X km)
        labels_km = [f"{round(edges_km[i],1)}–{round(edges_km[i+1],1)} km" for i in range(len(edges_km)-1)]

        # Binning distance en km
        df["segment_km_bin"] = pd.cut(
            df["dist_km"],
            bins=edges_km,
            labels=labels_km,
            include_lowest=True,
            right=True
        )
        df["segment_km_bin"] = df["segment_km_bin"].cat.set_categories(labels_km, ordered=True)

        # --- Pivot (lignes = pente, colonnes = segments km)
        pivot = df.pivot_table(index="pente_bin", columns="segment_km_bin", values="cadence", aggfunc="mean")

        # --- Seuils marche/course
        c1, c2 = st.columns(2)
        with c1:
            walk_max = st.slider("Seuil marche (≤ spm)", 60, 140, 110, step=1)
        with c2:
            run_min = st.slider("Seuil course (≥ spm)", 130, 200, 155, step=1)

        # Affichage sans décimales
        pivot_display = pivot.round(0)

        # ---- Styling pandas (appliqué à l'écran)
        def style_cell(v):
            import math
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return "color:#0b63c5;"
            if v <= walk_max:
                bg = "#d9fdd3"   # vert clair
            elif v >= run_min:
                bg = "#ffd6d6"   # rouge clair
            else:
                bg = "#fff3bf"   # jaune pâle
            return f"background-color:{bg}; color:#0b63c5;"

        styled = (
            pivot_display
            .style
            .applymap(style_cell)
            .format("{:.0f}")  # 0 décimale
            .set_table_styles([
                {"selector": "th", "props": [("color", "#0b63c5"), ("font-weight", "600")]},
                {"selector": "td", "props": [("border", "1px solid #eee")]},
            ])
        )

        st.table(styled)

        st.caption(
            f"Découpage : {len(labels_km)} tranches • Distance totale ≈ {total_dist_km:.1f} km"
        )

        st.markdown(
            "<div style='font-size: 0.9rem;'>"
            "🟩 <span style='color:#0b63c5;'>marche ≤ seuil</span> &nbsp; "
            "🟨 <span style='color:#0b63c5;'>intermédiaire</span> &nbsp; "
            "🟥 <span style='color:#0b63c5;'>course ≥ seuil</span>"
            "</div>",
            unsafe_allow_html=True
        )

        # ---- Export PNG fidèle aux couleurs + 0 décimale (zéro marge, rogné pile au tableau)
        import io
        import matplotlib.pyplot as plt

        values = pivot_display.values
        row_labels = pivot_display.index.astype(str).tolist()
        col_labels = pivot_display.columns.astype(str).tolist()

        cell_h = 0.4
        cell_w = 0.6
        fig_h = max(3, cell_h * (len(row_labels) + 2))
        fig_w = max(4, cell_w * (len(col_labels) + 2))

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")

        table = ax.table(
            cellText=[[ "" if pd.isna(x) else f"{int(round(x))}" for x in row] for row in values],
            rowLabels=row_labels,
            colLabels=col_labels,
            cellLoc='center',
            loc='center'
        )

        # Appliquer couleurs identiques au style
        n_rows, n_cols = values.shape
        for r in range(n_rows):
            for c in range(n_cols):
                val = values[r, c]
                cell = table[(r + 1, c)]
                if pd.isna(val):
                    cell.get_text().set_color("#0b63c5")
                    continue
                if val <= walk_max:
                    bg = "#d9fdd3"
                elif val >= run_min:
                    bg = "#ffd6d6"
                else:
                    bg = "#fff3bf"
                cell.set_facecolor(bg)
                cell.get_text().set_color("#0b63c5")

        for c in range(n_cols):
            header_cell = table[(0, c)]
            header_cell.get_text().set_color("#0b63c5")
            header_cell.set_facecolor("#f6f8fb")

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.2)

        # Dessiner pour récupérer bbox et rogner pile au tableau
        fig.canvas.draw()
        bbox = table.get_window_extent(fig.canvas.get_renderer())
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted())

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=200, bbox_inches=bbox, pad_inches=0)
        buf.seek(0)
        plt.close(fig)

        st.download_button(
            "📸 Télécharger le tableau en PNG (sans marges)",
            data=buf,
            file_name="cadence_pente_par_tranches_km.png",
            mime="image/png"
        )

        # Export CSV valeurs
        csv_bytes = pivot_display.to_csv().encode("utf-8")
        st.download_button(
            "⬇️ Exporter le tableau (CSV valeurs)",
            data=csv_bytes,
            file_name="cadence_par_pente_et_tranches_km.csv",
            mime="text/csv"
        )

    else:
        st.info("ℹ️ Distance totale non valide.")
else:
    manquantes = required_cols - set(df.columns)
    st.info(f"ℹ️ Colonnes manquantes pour ce tableau : {', '.join(sorted(manquantes))}")



# =========================
# === ❤️ Tableau croisé : Cardio moyen selon Pente (%) et Distance (km) ===
# =========================
st.markdown("### ❤️ Cardio moyen selon la pente et la distance (km)")

required_cols_hr = {"pente_bin", "segment_km_bin", "heart_rate"}
if record_file is not None and required_cols_hr.issubset(df.columns):
    # Slider FCmax
    hr_max = st.slider("Fréquence cardiaque max (bpm)", min_value=100, max_value=240, value=190, step=1)

    # Zones FCmax
    z1_max = 0.60 * hr_max
    z2_max = 0.70 * hr_max
    z3_max = 0.80 * hr_max
    z4_max = 0.90 * hr_max

    df["heart_rate"] = pd.to_numeric(df["heart_rate"], errors="coerce")

    # Pivot HR (moyenne)
    pivot_hr = df.pivot_table(
        index="pente_bin",
        columns="segment_km_bin",
        values="heart_rate",
        aggfunc="mean"
    )

    pivot_hr_display = pivot_hr.round(0)

    # ---- Style fonctionnel par zones (texte bleu, fond coloré)
    def style_hr(v):
        import math
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "color:#0b63c5;"
        if v >= z4_max:
            bg = "#d32f2f"   # Zone 5 - rouge vif
        elif v >= z3_max:
            bg = "#ef9a9a"   # Zone 4 - rouge clair
        elif v >= z2_max:
            bg = "#fff3bf"   # Zone 3 - jaune
        elif v >= z1_max:
            bg = "#d9fdd3"   # Zone 2 - vert clair
        else:
            bg = "#d0e8ff"   # Zone 1 - bleu clair
        return f"background-color:{bg}; color:#0b63c5;"

    styled_hr = (
        pivot_hr_display
        .style
        .applymap(style_hr)
        .format("{:.0f}")
        .set_table_styles([
            {"selector": "th", "props": [("color", "#0b63c5"), ("font-weight", "600")]},
            {"selector": "td", "props": [("border", "1px solid #eee")]},
        ])
    )

    # ⚠️ st.table pour garder les couleurs
    st.table(styled_hr)

    # --- Légende zones
    st.markdown(
        f"""
        <div style='font-size:0.9rem; line-height:1.6'>
        <b>Zones %FCmax</b> — FCmax = {hr_max} bpm :
        <br>🟥 <span style='color:#0b63c5;'>Z5</span> ≥ {int(z4_max)} bpm (≥90%)
        <br>🟥 <span style='color:#0b63c5;'>Z4</span> {int(z3_max)}–{int(z4_max)-1} bpm (80–90%)
        <br>🟨 <span style='color:#0b63c5;'>Z3</span> {int(z2_max)}–{int(z3_max)-1} bpm (70–80%)
        <br>🟩 <span style='color:#0b63c5;'>Z2</span> {int(z1_max)}–{int(z2_max)-1} bpm (60–70%)
        <br>🟦 <span style='color:#0b63c5;'>Z1</span> &lt; {int(z1_max)} bpm (&lt;60%)
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---- 📸 Export PNG fidèle au style (sans marges)
    import io
    import matplotlib.pyplot as plt

    values = pivot_hr_display.values
    row_labels = pivot_hr_display.index.astype(str).tolist()
    col_labels = pivot_hr_display.columns.astype(str).tolist()

    cell_h = 0.4
    cell_w = 0.6
    fig_h = max(3, cell_h * (len(row_labels) + 2))
    fig_w = max(4, cell_w * (len(col_labels) + 2))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=[[ "" if pd.isna(x) else f"{int(round(x))}" for x in row] for row in values],
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc='center',
        loc='center'
    )

    # Appliquer les couleurs des zones
    n_rows, n_cols = values.shape
    for r in range(n_rows):
        for c in range(n_cols):
            val = values[r, c]
            cell = table[(r + 1, c)]
            if pd.isna(val):
                cell.get_text().set_color("#0b63c5")
                continue
            if val >= z4_max:
                bg = "#d32f2f"   # Zone 5 - rouge vif
            elif val >= z3_max:
                bg = "#ef9a9a"   # Zone 4 - rouge clair
            elif val >= z2_max:
                bg = "#fff3bf"   # Zone 3 - jaune
            elif val >= z1_max:
                bg = "#d9fdd3"   # Zone 2 - vert clair
            else:
                bg = "#d0e8ff"   # Zone 1 - bleu clair
            cell.set_facecolor(bg)
            cell.get_text().set_color("#0b63c5")

    for c in range(n_cols):
        header_cell = table[(0, c)]
        header_cell.get_text().set_color("#0b63c5")
        header_cell.set_facecolor("#f6f8fb")

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.8, 1.2)

    # Rogner pile au tableau
    fig.canvas.draw()
    bbox = table.get_window_extent(fig.canvas.get_renderer())
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=200, bbox_inches=bbox, pad_inches=0)
    buf.seek(0)
    plt.close(fig)

    st.download_button(
        "📸 Télécharger le tableau cardio en PNG (sans marges)",
        data=buf,
        file_name="cardio_moyen_pente_tranches_km.png",
        mime="image/png"
    )

    # ---- 💾 Export CSV (valeurs)
    csv_bytes_hr = pivot_hr_display.to_csv().encode("utf-8")
    st.download_button(
        "⬇️ Exporter le tableau cardio (CSV valeurs)",
        data=csv_bytes_hr,
        file_name="cardio_moyen_pente_tranches_km.csv",
        mime="text/csv"
    )

else:
    manquantes_hr = required_cols_hr - set(df.columns)
    st.info(
        "ℹ️ Impossible d'afficher le tableau cardio : "
        + ("colonnes manquantes : " + ", ".join(sorted(manquantes_hr)) if manquantes_hr else "données indisponibles.")
    )

# =========================
# 📈 Analyse biomécanique
# =========================

# =========================
# === 📉 Oscillation verticale (mm) vs Pente (%) | couleur = distance (km) | filtres + export PNG
# =========================
st.markdown("### 📉 Oscillation verticale (mm) vs Pente (%) — couleur = distance (km)")

import io

if record_file is None:
    empty_df = pd.DataFrame({"pente_%": [], "vertical_oscillation": [], "distance_km": []})
    empty_chart = (
        alt.Chart(empty_df)
        .mark_circle(size=40, opacity=0.75)
        .encode(
            x=alt.X("pente_%:Q", title="Pente (%)"),
            y=alt.Y("vertical_oscillation:Q", title="Oscillation verticale (mm)"),
            color=alt.Color("distance_km:Q", title="Distance (km)")
        )
        .properties(height=380)
    )
    st.altair_chart(empty_chart, use_container_width=True)

else:
    required = ["pente_%", "vertical_oscillation", "distance", "timestamp"]
    if not all(c in df.columns for c in required):
        st.info("ℹ️ Il faut les colonnes 'pente_%', 'vertical_oscillation', 'distance' et 'timestamp'.")
    else:
        # Nettoyage & conversions
        df["pente_%"] = pd.to_numeric(df["pente_%"], errors="coerce")
        df["vertical_oscillation"] = pd.to_numeric(df["vertical_oscillation"], errors="coerce")
        dist_m = pd.to_numeric(df["distance"], errors="coerce")
        df["distance_km"] = (dist_m / 1000.0).clip(lower=0)

        # Bornes pour sliders
        pente_series = df["pente_%"].dropna()
        vo_series = df["vertical_oscillation"].dropna()
        dist_series = df["distance_km"].dropna()

        pente_min = float(np.nanmin(pente_series)) if len(pente_series) else -30.0
        pente_max = float(np.nanmax(pente_series)) if len(pente_series) else 30.0
        dist_min  = float(np.nanmin(dist_series))  if len(dist_series)  else 0.0
        dist_max  = float(np.nanmax(dist_series))  if len(dist_series)  else 1.0

        # Défauts trail raisonnables
        pente_default = (max(pente_min, -15.0), min(pente_max, 15.0)) if pente_min < pente_max else (-15.0, 15.0)
        dist_default  = (dist_min, dist_max)

        c1, c2 = st.columns(2)
        with c1:
            pente_range = st.slider(
                "Pente (%)",
                float(pente_min), float(pente_max),
                (float(pente_default[0]), float(pente_default[1])),
                step=0.5,
                key="vo_slider_pente"
            )
        with c2:
            dist_range = st.slider(
                "Distance (km)",
                float(dist_min), float(dist_max),
                (float(dist_default[0]), float(dist_default[1])),
                step=max(0.1, round((dist_max - dist_min) / 200, 2)),
                key="vo_slider_distance"
            )

        # Filtrage
        chart_df = df[["pente_%", "vertical_oscillation", "distance_km", "timestamp"]].dropna()
        chart_df = chart_df[
            chart_df["pente_%"].between(pente_range[0], pente_range[1]) &
            chart_df["distance_km"].between(dist_range[0], dist_range[1])
        ]

        if chart_df.empty:
            st.warning("Aucun point dans les plages sélectionnées. Ajuste les curseurs.")
        else:
            scatter_vo = (
                alt.Chart(chart_df)
                .mark_circle(size=42, opacity=0.85)
                .encode(
                    x=alt.X(
                        "pente_%:Q",
                        title="Pente (%)",
                        scale=alt.Scale(domain=[pente_range[0], pente_range[1]])
                    ),
                    y=alt.Y(
                        "vertical_oscillation:Q",
                        title="Oscillation verticale (mm)"
                    ),
                    color=alt.Color(
                        "distance_km:Q",
                        title="Distance (km)",
                        scale=alt.Scale(domain=[dist_range[0], dist_range[1]], range=["#00c853", "#ff5252"])
                    ),
                    tooltip=[
                        alt.Tooltip("timestamp:T", title="Temps"),
                        alt.Tooltip("distance_km:Q", title="Distance (km)", format=".1f"),
                        alt.Tooltip("pente_%:Q", title="Pente (%)", format=".1f"),
                        alt.Tooltip("vertical_oscillation:Q", title="Oscillation (mm)", format=".0f"),
                    ],
                )
                .interactive()
                .properties(height=380, width='container')
            )

            st.altair_chart(scatter_vo, use_container_width=True)

            # --- 🌟 Explications sous le graphe
            st.markdown(
                """
            **Comment lire le nuage de points :**
            - **Axe X = Pente (%)** : négatif = descente, positif = montée.
            - **Axe Y = Oscillation verticale (mm)** : plus c’est **bas**, moins tu “rebondis” → souvent plus **économe**.
            - **Couleur = Distance (km)** : du début (vert) vers la fin (rouge).
            - Cherche les tendances : **en montée**, beaucoup de coureurs oscillent un peu plus ; **en descente**, l’oscillation peut baisser si tu “poses” mieux les appuis.
            """
            )

            # Export PNG HD (filtres appliqués)
            try:
                import altair_saver  # noqa: F401

                export_width = 400
                export_height = 450
                scale_factor = 6

                chart_to_save = scatter_vo.properties(width=export_width, height=export_height)

                buf = io.BytesIO()
                chart_to_save.save(
                    buf,
                    format="png",
                    method="vl-convert",   # nécessite `pip install vl-convert-python`
                    scale_factor=scale_factor
                )
                buf.seek(0)

                fname = (
                    "oscillation_verticale_vs_pente_"
                    f"pente{int(pente_range[0])}-{int(pente_range[1])}_"
                    f"dist{dist_range[0]:.1f}-{dist_range[1]:.1f}_HD.png"
                )

                st.download_button(
                    label="📸 Télécharger le PNG (filtres appliqués, HD)",
                    data=buf,
                    file_name=fname,
                    mime="image/png"
                )

            except Exception as e:
                st.warning(
                    "⚠️ Problème avec la génération du PNG.\n"
                    "Installe `vl-convert-python` si nécessaire :\n"
                    "```bash\npip install vl-convert-python\n```\n"
                    f"\nErreur : {e}"
                )


# =========================
# === 📊 Tableau croisé : Oscillation (mm) par Pente (10%) × Progression (20%) + couleurs
# =========================

# 1) Progression (%)
if "progress_%" not in df.columns:
    total_dist_m = pd.to_numeric(df.get("distance", pd.Series(dtype=float)), errors="coerce").iloc[-1] if len(df) else np.nan
    if pd.notna(total_dist_m) and total_dist_m > 0:
        df["progress_%"] = (pd.to_numeric(df["distance"], errors="coerce") / total_dist_m * 100).clip(0, 100)
    else:
        df["progress_%"] = np.nan

# 2) Bins pente (10%) : < -40, [-40,-30), ..., [30,40], > 40
edges_pente = [-np.inf] + list(np.arange(-40, 50, 10)) + [np.inf]  # -40,-30,...,40
labels_pente = (["< -40%"] +
                [f"{b} à {b+10}%" for b in range(-40, 40, 10)] +
                ["> 40%"])

df["pente_bin10"] = pd.cut(
    pd.to_numeric(df["pente_%"], errors="coerce"),
    bins=edges_pente,
    labels=labels_pente,
    include_lowest=True,
    right=False  # inclut la borne gauche
)

# 3) Bins progression (20%)
edges_prog = [0, 20, 40, 60, 80, 100]
labels_prog = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]
df["progress_bin20"] = pd.cut(
    pd.to_numeric(df["progress_%"], errors="coerce"),
    bins=edges_prog,
    labels=labels_prog,
    include_lowest=True,
    right=True
)

# 4) Pivot (moyenne oscillation)
pivot_vo = df.pivot_table(
    index="pente_bin10",
    columns="progress_bin20",
    values="vertical_oscillation",
    aggfunc="mean"
)

# --- Conversion sûre en numérique pour tout le pipeline
pivot_num = pivot_vo.apply(pd.to_numeric, errors="coerce")

# Arrondis (mm) pour l’affichage
pivot_vo_display = pivot_num.round(0)

# 5) Couleurs : vert (bas) → jaune (médian) → rouge (haut)
if pivot_num.size and np.isfinite(pivot_num.values).any():
    vmin = float(np.nanmin(pivot_num.values))
    vmax = float(np.nanmax(pivot_num.values))
    vmid = float(np.nanmedian(pivot_num.values))
else:
    vmin, vmid, vmax = 0.0, 0.5, 1.0  # valeurs par défaut si tableau vide

def _interp_color(val, vmin, vmid, vmax):
    """Renvoie une couleur hexa entre vert (#d9fdd3) → jaune (#fff3bf) → rouge (#ffd6d6)."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "background-color:#f7f7f7; color:#0b63c5;"
    if not np.isfinite(v):
        return "background-color:#f7f7f7; color:#0b63c5;"

    if vmax == vmin:
        t = 0.5
    else:
        t = (v - vmin) / (vmax - vmin)
        t = min(max(t, 0.0), 1.0)

    green = (217, 253, 211)
    yellow = (255, 243, 191)
    red   = (255, 214, 214)

    if t <= 0.5:
        a = t / 0.5
        r = green[0] + a * (yellow[0] - green[0])
        g = green[1] + a * (yellow[1] - green[1])
        b = green[2] + a * (yellow[2] - green[2])
    else:
        a = (t - 0.5) / 0.5
        r = yellow[0] + a * (red[0] - yellow[0])
        g = yellow[1] + a * (red[1] - yellow[1])
        b = yellow[2] + a * (red[2] - yellow[2])

    return f"background-color:#{int(r):02x}{int(g):02x}{int(b):02x}; color:#0b63c5;"

def _style_df(data: pd.DataFrame):
    """➡️ IMPORTANT: retourne un DataFrame de styles (mêmes dimensions)."""
    styles = pd.DataFrame("", index=data.index, columns=data.columns)
    for i in data.index:
        for j in data.columns:
            styles.loc[i, j] = _interp_color(data.loc[i, j], vmin, vmid, vmax)
    return styles

styled = (
    pivot_vo_display
    .style
    .apply(_style_df, axis=None)       # renvoie un DataFrame de styles
    .format("{:.0f}")                  # 0 décimale (mm)
    .set_table_styles([
        {"selector": "th", "props": [("color", "#0b63c5"), ("font-weight", "600"), ("white-space", "nowrap")]},
        {"selector": "td", "props": [("border", "1px solid #eee"), ("min-width", "72px")]},
    ])
)

st.markdown("#### 📊 Oscillation verticale moyenne (mm) — par pente (10%) × progression (20%)")
st.table(styled)

# 6) Export CSV
csv_bytes = pivot_vo_display.to_csv().encode("utf-8")
st.download_button(
    "⬇️ Exporter le tableau (CSV)",
    data=csv_bytes,
    file_name="oscillation_par_pente10_et_progress20.csv",
    mime="text/csv"
)

# 7) Export PNG du tableau (avec couleurs)
import io, matplotlib.pyplot as plt
from matplotlib.table import Table

def fig_from_styled_pivot(pvt: pd.DataFrame, vmin: float, vmid: float, vmax: float) -> bytes:
    # Prépare figure
    n_rows, n_cols = pvt.shape
    cell_h, cell_w = 0.36, 1.15
    fig_w = max(6.0, n_cols * cell_w)
    fig_h = max(2.5, (n_rows + 1) * cell_h)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = Table(ax, bbox=[0, 0, 1, 1])

    col_labels = list(pvt.columns)
    row_labels = list(pvt.index.astype(str))

    def bg_color(val):
        try:
            v = float(val)
        except (TypeError, ValueError):
            return "#f7f7f7"
        if not np.isfinite(v):
            return "#f7f7f7"

        if vmax == vmin:
            t = 0.5
        else:
            t = (v - vmin) / (vmax - vmin)
            t = min(max(t, 0.0), 1.0)

        green = (217, 253, 211)
        yellow= (255, 243, 191)
        red   = (255, 214, 214)
        if t <= 0.5:
            a = t / 0.5
            r = green[0] + a * (yellow[0] - green[0])
            g = green[1] + a * (yellow[1] - green[1])
            b = green[2] + a * (yellow[2] - green[2])
        else:
            a = (t - 0.5) / 0.5
            r = yellow[0] + a * (red[0] - yellow[0])
            g = yellow[1] + a * (red[1] - yellow[1])
            b = yellow[2] + a * (red[2] - yellow[2])
        return f"#{int(r):02x}{int(g):02x}{int(b):02x}"

    # Coin haut-gauche vide
    tbl.add_cell(0, 0, cell_w, cell_h, text="", loc="center", facecolor="#ffffff")

    # En-têtes colonnes
    for j, c in enumerate(col_labels, start=1):
        tbl.add_cell(0, j, cell_w, cell_h, text=str(c), loc="center",
                     facecolor="#e9f2ff", edgecolor="#dddddd")

    # Lignes + cellules
    for i, rlab in enumerate(row_labels, start=1):
        tbl.add_cell(i, 0, cell_w, cell_h, text=str(rlab), loc="center",
                     facecolor="#e9f2ff", edgecolor="#dddddd")
        for j, c in enumerate(col_labels, start=1):
            val = pvt.iloc[i-1, j-1]
            txt = "" if pd.isna(val) else f"{float(val):.0f}"
            tbl.add_cell(i, j, cell_w, cell_h, text=txt, loc="center",
                         facecolor=bg_color(val), edgecolor="#eeeeee")

    ax.add_table(tbl)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=240, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return buf.getvalue()

try:
    png_bytes = fig_from_styled_pivot(pivot_vo_display, vmin, vmid, vmax)
    st.download_button(
        "📸 Exporter le tableau en PNG",
        data=png_bytes,
        file_name="oscillation_par_pente10_et_progress20.png",
        mime="image/png"
    )
except Exception as e:
    st.warning(f"⚠️ Export PNG du tableau impossible : {e}")

# 8) 🧠 Petit commentaire automatique (lecture rapide)
try:
    def mean_safe(s):
        s = pd.to_numeric(s, errors="coerce")
        val = np.nanmean(s)
        return float(val) if np.isfinite(val) else np.nan

    desc = mean_safe(df.loc[pd.to_numeric(df["pente_%"], errors="coerce") < -5, "vertical_oscillation"])
    plat = mean_safe(df.loc[pd.to_numeric(df["pente_%"], errors="coerce").between(-5, 5), "vertical_oscillation"])
    mont = mean_safe(df.loc[pd.to_numeric(df["pente_%"], errors="coerce") > 5, "vertical_oscillation"])

    if pivot_num.size and np.isfinite(pivot_num.values).any():
        min_val = float(np.nanmin(pivot_num.values))
        max_val = float(np.nanmax(pivot_num.values))
        # positions min / max (première occurrence)
        mi, mj = np.where(pivot_num.values == np.nanmin(pivot_num.values))
        Mi, Mj = np.where(pivot_num.values == np.nanmax(pivot_num.values))
        xi = pivot_num.index[int(mi[0])] if len(mi) else "—"
        xj = pivot_num.columns[int(mj[0])] if len(mj) else "—"
        XI = pivot_num.index[int(Mi[0])] if len(Mi) else "—"
        XJ = pivot_num.columns[int(Mj[0])] if len(Mj) else "—"
    else:
        min_val = max_val = np.nan
        xi = xj = XI = XJ = "—"

    st.markdown(
        f"""
**Lecture rapide des données :**
- **Moyenne par profil** (mm) → Descente: **{(desc if np.isfinite(desc) else np.nan):.0f}**, Plat: **{(plat if np.isfinite(plat) else np.nan):.0f}**, Montée: **{(mont if np.isfinite(mont) else np.nan):.0f}**.  
- **Plus faible oscillation** observée: **{(min_val if np.isfinite(min_val) else float('nan')):.0f} mm** (pente: **{xi}**, progression: **{xj}**).  
- **Plus forte oscillation** observée: **{(max_val if np.isfinite(max_val) else float('nan')):.0f} mm** (pente: **{XI}**, progression: **{XJ}**).  

💡 *En général*, moins d’oscillation = **économie**. Si tu vois l’oscillation **augmenter en fin de course**
(colonnes 60–80% / 80–100%), ça peut indiquer de la **fatigue** (moins de contrôle postural) — à confronter avec la **cadence** et le **temps de contact**.
"""
    )
except Exception as e:
    st.info(f"Note: analyse rapide non disponible ({e}).")

# =========================
# === 🚶 Longueur de pas (m) vs Pente (%) | couleur = distance (km) | filtres + export PNG
# =========================
st.markdown("### 🚶 Longueur de pas (m) vs Pente (%) — couleur = distance (km)")

import io

# Vérification colonnes nécessaires
required = ["pente_%", "step_length", "distance", "timestamp"]
if (record_file is None) or (not all(c in df.columns for c in required)):
    # Graphique vide avant chargement
    empty_df = pd.DataFrame({"pente_%": [], "step_length_m": [], "distance_km": []})
    empty_chart = (
        alt.Chart(empty_df)
        .mark_circle(size=40, opacity=0.75)
        .encode(
            x=alt.X("pente_%:Q", title="Pente (%)"),
            y=alt.Y("step_length_m:Q", title="Longueur de pas (m)"),
            color=alt.Color("distance_km:Q", title="Distance (km)")
        )
        .properties(height=380)
    )
    st.altair_chart(empty_chart, use_container_width=True)
    if record_file is not None and not all(c in df.columns for c in required):
        st.info("ℹ️ Il faut les colonnes 'pente_%', 'step_length', 'distance' et 'timestamp'.")
else:
    # ===== Nettoyage & conversions =====
    df["pente_%"] = pd.to_numeric(df["pente_%"], errors="coerce")

    # step_length en mm -> m  ✅ (correction)
    step_mm = pd.to_numeric(df["step_length"], errors="coerce")
    df["step_length_m"] = (step_mm / 1000.0).clip(lower=0)

    # (optionnel) filtrer valeurs aberrantes de foulée
    mask_real = df["step_length_m"].between(0.3, 2.5)
    df.loc[~mask_real, "step_length_m"] = np.nan

    # distance m -> km
    df["distance_km"] = pd.to_numeric(df["distance"], errors="coerce") / 1000.0

    # ===== Bornes sliders =====
    pente_series = df["pente_%"].dropna()
    sl_series = df["step_length_m"].dropna()
    dist_series = df["distance_km"].dropna()

    pente_min = float(np.nanmin(pente_series)) if len(pente_series) else -30.0
    pente_max = float(np.nanmax(pente_series)) if len(pente_series) else  30.0
    sl_min    = float(np.nanmin(sl_series))    if len(sl_series)    else  0.5
    sl_max    = float(np.nanmax(sl_series))    if len(sl_series)    else  1.8
    dist_min  = float(np.nanmin(dist_series))  if len(dist_series)  else  0.0
    dist_max  = float(np.nanmax(dist_series))  if len(dist_series)  else  1.0

    pente_default = (max(pente_min, -15.0), min(pente_max, 15.0)) if pente_min < pente_max else (-15.0, 15.0)
    sl_default    = (max(sl_min, 0.5),       min(sl_max, 1.5))     if sl_min    < sl_max    else (0.5, 1.5)
    dist_default  = (dist_min, dist_max)

    c1, c2, c3 = st.columns(3)
    with c1:
        pente_range = st.slider(
            "Pente (%)",
            float(pente_min), float(pente_max),
            (float(pente_default[0]), float(pente_default[1])),
            step=0.5,
            key="sl_slider_pente"
        )
    with c2:
        sl_range = st.slider(
            "Longueur de pas (m)",
            float(sl_min), float(sl_max),
            (float(sl_default[0]), float(sl_default[1])),
            step=0.01,
            key="sl_slider_step"
        )
    with c3:
        dist_range = st.slider(
            "Distance (km)",
            float(dist_min), float(dist_max),
            (float(dist_default[0]), float(dist_default[1])),
            step=max(0.1, round((dist_max - dist_min) / 200, 2)),
            key="sl_slider_distance"
        )

    # ===== Filtrage =====
    chart_df = df[["pente_%", "step_length_m", "distance_km", "timestamp"]].dropna()
    chart_df = chart_df[
        chart_df["pente_%"].between(pente_range[0], pente_range[1]) &
        chart_df["step_length_m"].between(sl_range[0], sl_range[1]) &
        chart_df["distance_km"].between(dist_range[0], dist_range[1])
    ]

    if chart_df.empty:
        st.warning("Aucun point dans les plages sélectionnées. Ajuste les curseurs.")
    else:
        scatter_sl = (
            alt.Chart(chart_df)
            .mark_circle(size=42, opacity=0.85)
            .encode(
                x=alt.X("pente_%:Q", title="Pente (%)",
                        scale=alt.Scale(domain=[pente_range[0], pente_range[1]])),
                y=alt.Y("step_length_m:Q", title="Longueur de pas (m)",
                        scale=alt.Scale(domain=[sl_range[0], sl_range[1]])),
                color=alt.Color("distance_km:Q", title="Distance (km)",
                        scale=alt.Scale(domain=[dist_range[0], dist_range[1]], range=["#00c853", "#ff5252"])),
                tooltip=[
                    alt.Tooltip("timestamp:T", title="Temps"),
                    alt.Tooltip("distance_km:Q", title="Distance (km)", format=".1f"),
                    alt.Tooltip("pente_%:Q", title="Pente (%)", format=".1f"),
                    alt.Tooltip("step_length_m:Q", title="Pas (m)", format=".2f"),
                ],
            )
            .interactive()
            .properties(height=380, width='container')
        )

        st.altair_chart(scatter_sl, use_container_width=True)

        st.caption(
            "Lecture : plus la longueur de pas est courte, plus la foulée est compacte (montées, terrain technique, fatigue). "
            "Sur le plat, observe l’évolution de la couleur (distance) pour voir l’effet de la fatigue."
        )

        # Export PNG HD
        try:
            import altair_saver  # noqa: F401

            export_width = 400
            export_height = 450
            scale_factor = 6

            chart_to_save = scatter_sl.properties(width=export_width, height=export_height)
            buf = io.BytesIO()
            chart_to_save.save(
                buf,
                format="png",
                method="vl-convert",   # nécessite `pip install vl-convert-python`
                scale_factor=scale_factor
            )
            buf.seek(0)

            fname = (
                "step_length_vs_pente_"
                f"pente{int(pente_range[0])}-{int(pente_range[1])}_"
                f"step{sl_range[0]:.2f}-{sl_range[1]:.2f}_"
                f"dist{dist_range[0]:.1f}-{dist_range[1]:.1f}_HD.png"
            )

            st.download_button(
                label="📸 Télécharger le PNG (filtres appliqués, HD)",
                data=buf,
                file_name=fname,
                mime="image/png"
            )

        except Exception as e:
            st.warning(
                "⚠️ Problème avec la génération du PNG.\n"
                "Installe `vl-convert-python` si nécessaire :\n"
                "```bash\npip install vl-convert-python\n```\n"
                f"\nErreur : {e}"
            )

# =========================
# === 📊 Tableau croisé : Longueur de pas (m) par Pente (10%) × Progression (20%) + couleurs
# =========================

# 0) step_length en mm -> m  ✅ (correction)
df["step_length_m"] = pd.to_numeric(df.get("step_length", np.nan), errors="coerce") / 1000.0
df.loc[~df["step_length_m"].between(0.3, 2.5), "step_length_m"] = np.nan  # optionnel

# 1) Progression (%) (réutilise si déjà calculée)
if "progress_%" not in df.columns:
    total_dist_m = pd.to_numeric(df.get("distance", pd.Series(dtype=float)), errors="coerce").iloc[-1] if len(df) else np.nan
    if pd.notna(total_dist_m) and total_dist_m > 0:
        df["progress_%"] = (pd.to_numeric(df["distance"], errors="coerce") / total_dist_m * 100).clip(0, 100)
    else:
        df["progress_%"] = np.nan

# 2) Bins pente (10%) : < -40, [-40,-30), ..., [30,40], > 40
edges_pente_sl = [-np.inf] + list(np.arange(-40, 50, 10)) + [np.inf]
labels_pente_sl = (["< -40%"] +
                   [f"{b} à {b+10}%" for b in range(-40, 40, 10)] +
                   ["> 40%"])

df["pente_bin10_sl"] = pd.cut(
    pd.to_numeric(df["pente_%"], errors="coerce"),
    bins=edges_pente_sl,
    labels=labels_pente_sl,
    include_lowest=True,
    right=False
)

# 3) Bins progression (20%)
edges_prog_sl = [0, 20, 40, 60, 80, 100]
labels_prog_sl = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]
df["progress_bin20_sl"] = pd.cut(
    pd.to_numeric(df["progress_%"], errors="coerce"),
    bins=edges_prog_sl,
    labels=labels_prog_sl,
    include_lowest=True,
    right=True
)

# 4) Pivot (moyenne step length en m)
pivot_sl = df.pivot_table(
    index="pente_bin10_sl",
    columns="progress_bin20_sl",
    values="step_length_m",
    aggfunc="mean"
)

# --- Conversion sûre
pivot_sl_num = pivot_sl.apply(pd.to_numeric, errors="coerce")

# Arrondis (m) pour l’affichage
pivot_sl_display = pivot_sl_num.round(2)

# 5) Couleurs : vert (court) → jaune → rouge (long)
if pivot_sl_num.size and np.isfinite(pivot_sl_num.values).any():
    vmin_sl = float(np.nanmin(pivot_sl_num.values))
    vmax_sl = float(np.nanmax(pivot_sl_num.values))
    vmid_sl = float(np.nanmedian(pivot_sl_num.values))
else:
    vmin_sl, vmid_sl, vmax_sl = 0.0, 0.5, 1.0

def _interp_color_sl(val, vmin, vmid, vmax):
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "background-color:#f7f7f7; color:#0b63c5;"
    if not np.isfinite(v):
        return "background-color:#f7f7f7; color:#0b63c5;"

    if vmax == vmin:
        t = 0.5
    else:
        t = (v - vmin) / (vmax - vmin)
        t = min(max(t, 0.0), 1.0)

    green = (217, 253, 211)
    yellow = (255, 243, 191)
    red   = (255, 214, 214)

    if t <= 0.5:
        a = t / 0.5
        r = green[0] + a * (yellow[0] - green[0])
        g = green[1] + a * (yellow[1] - green[1])
        b = green[2] + a * (yellow[2] - green[2])
    else:
        a = (t - 0.5) / 0.5
        r = yellow[0] + a * (red[0] - yellow[0])
        g = yellow[1] + a * (red[1] - yellow[1])
        b = yellow[2] + a * (red[2] - yellow[2])

    return f"background-color:#{int(r):02x}{int(g):02x}{int(b):02x}; color:#0b63c5;"

def _style_df_sl(data: pd.DataFrame):
    styles = pd.DataFrame("", index=data.index, columns=data.columns)
    for i in data.index:
        for j in data.columns:
            styles.loc[i, j] = _interp_color_sl(data.loc[i, j], vmin_sl, vmid_sl, vmax_sl)
    return styles

styled_sl = (
    pivot_sl_display
    .style
    .apply(_style_df_sl, axis=None)
    .format("{:.2f}")
    .set_table_styles([
        {"selector": "th", "props": [("color", "#0b63c5"), ("font-weight", "600"), ("white-space", "nowrap")]},
        {"selector": "td", "props": [("border", "1px solid #eee"), ("min-width", "72px")]},
    ])
)

st.markdown("#### 📊 Longueur de pas moyenne (m) — par pente (10%) × progression (20%)")
st.table(styled_sl)

# 6) Export CSV
csv_bytes_sl = pivot_sl_display.to_csv().encode("utf-8")
st.download_button(
    "⬇️ Exporter le tableau (CSV)",
    data=csv_bytes_sl,
    file_name="step_length_par_pente10_et_progress20.csv",
    mime="text/csv"
)

# 7) Export PNG du tableau (avec couleurs)
import io, matplotlib.pyplot as plt
from matplotlib.table import Table

def fig_from_styled_pivot_sl(pvt: pd.DataFrame, vmin: float, vmid: float, vmax: float) -> bytes:
    n_rows, n_cols = pvt.shape
    cell_h, cell_w = 0.36, 1.15
    fig_w = max(6.0, n_cols * cell_w)
    fig_h = max(2.5, (n_rows + 1) * cell_h)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = Table(ax, bbox=[0, 0, 1, 1])

    col_labels = list(pvt.columns)
    row_labels = list(pvt.index.astype(str))

    def bg_color(val):
        try:
            v = float(val)
        except (TypeError, ValueError):
            return "#f7f7f7"
        if not np.isfinite(v):
            return "#f7f7f7"

        if vmax == vmin:
            t = 0.5
        else:
            t = (v - vmin) / (vmax - vmin)
            t = min(max(t, 0.0), 1.0)

        green = (217, 253, 211)
        yellow= (255, 243, 191)
        red   = (255, 214, 214)
        if t <= 0.5:
            a = t / 0.5
            r = green[0] + a * (yellow[0] - green[0])
            g = green[1] + a * (yellow[1] - green[1])
            b = green[2] + a * (yellow[2] - green[2])
        else:
            a = (t - 0.5) / 0.5
            r = yellow[0] + a * (red[0] - yellow[0])
            g = yellow[1] + a * (red[1] - yellow[1])
            b = yellow[2] + a * (red[2] - yellow[2])
        return f"#{int(r):02x}{int(g):02x}{int(b):02x}"

    # Coin haut-gauche vide
    tbl.add_cell(0, 0, cell_w, cell_h, text="", loc="center", facecolor="#ffffff")

    # En-têtes colonnes
    for j, c in enumerate(col_labels, start=1):
        tbl.add_cell(0, j, cell_w, cell_h, text=str(c), loc="center",
                     facecolor="#e9f2ff", edgecolor="#dddddd")

    # Lignes + cellules
    for i, rlab in enumerate(row_labels, start=1):
        tbl.add_cell(i, 0, cell_w, cell_h, text=str(rlab), loc="center",
                     facecolor="#e9f2ff", edgecolor="#dddddd")
        for j, c in enumerate(col_labels, start=1):
            val = pvt.iloc[i-1, j-1]
            txt = "" if pd.isna(val) else f"{float(val):.2f}"
            tbl.add_cell(i, j, cell_w, cell_h, text=txt, loc="center",
                         facecolor=bg_color(val), edgecolor="#eeeeee")

    ax.add_table(tbl)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=240, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return buf.getvalue()

try:
    png_bytes_sl = fig_from_styled_pivot_sl(pivot_sl_display, vmin_sl, vmid_sl, vmax_sl)
    st.download_button(
        "📸 Exporter le tableau en PNG",
        data=png_bytes_sl,
        file_name="step_length_par_pente10_et_progress20.png",
        mime="image/png"
    )
except Exception as e:
    st.warning(f"⚠️ Export PNG du tableau impossible : {e}")

# 8) 🧠 Lecture rapide
try:
    def mean_safe(s):
        s = pd.to_numeric(s, errors="coerce")
        val = np.nanmean(s)
        return float(val) if np.isfinite(val) else np.nan

    # Moyennes par profil de pente
    desc_m = mean_safe(df.loc[pd.to_numeric(df["pente_%"], errors="coerce") < -5, "step_length_m"])
    plat_m = mean_safe(df.loc[pd.to_numeric(df["pente_%"], errors="coerce").between(-5, 5), "step_length_m"])
    mont_m = mean_safe(df.loc[pd.to_numeric(df["pente_%"], errors="coerce") > 5, "step_length_m"])

    if pivot_sl_num.size and np.isfinite(pivot_sl_num.values).any():
        min_val = float(np.nanmin(pivot_sl_num.values))
        max_val = float(np.nanmax(pivot_sl_num.values))
        mi, mj = np.where(pivot_sl_num.values == np.nanmin(pivot_sl_num.values))
        Mi, Mj = np.where(pivot_sl_num.values == np.nanmax(pivot_sl_num.values))
        xi = pivot_sl_num.index[int(mi[0])] if len(mi) else "—"
        xj = pivot_sl_num.columns[int(mj[0])] if len(mj) else "—"
        XI = pivot_sl_num.index[int(Mi[0])] if len(Mi) else "—"
        XJ = pivot_sl_num.columns[int(Mj[0])] if len(Mj) else "—"
    else:
        min_val = max_val = np.nan
        xi = xj = XI = XJ = "—"

    st.markdown(
        f"""
**Lecture rapide — Longueur de pas (m) :**
- **Moyenne par profil** → Descente: **{(desc_m if np.isfinite(desc_m) else np.nan):.2f} m**, Plat: **{(plat_m if np.isfinite(plat_m) else np.nan):.2f} m**, Montée: **{(mont_m if np.isfinite(mont_m) else np.nan):.2f} m**.  
- **Plus courte** observée: **{(min_val if np.isfinite(min_val) else float('nan')):.2f} m** (pente: **{xi}**, progression: **{xj}**).  
- **Plus longue** observée: **{(max_val if np.isfinite(max_val) else float('nan')):.2f} m** (pente: **{XI}**, progression: **{XJ}**).  

💡 *Guidage rapide* : pas **plus court** attendu en montée raide et/ou en fin de course (fatigue) ;
pas **plus long** sur le plat/début si frais. Confronte avec **cadence**, **VO**, et **temps de contact**.
"""
    )
except Exception as e:
    st.info(f"Note: analyse rapide non disponible ({e}).")


# =========================
# === ⏱️ Temps de pas (s) vs Pente (%) | couleur = distance (km) | filtres + export PNG
# =========================
st.markdown("### ⏱️ Temps de pas (s) vs Pente (%) — couleur = distance (km)")

import io

# Colonnes requises (on suppose step_length = temps de pas en millisecondes)
required = ["pente_%", "step_length", "distance", "timestamp"]
if (record_file is None) or (not all(c in df.columns for c in required)):
    # Graphe vide avant chargement
    empty_df = pd.DataFrame({"pente_%": [], "step_time_s": [], "distance_km": []})
    empty_chart = (
        alt.Chart(empty_df)
        .mark_circle(size=40, opacity=0.75)
        .encode(
            x=alt.X("pente_%:Q", title="Pente (%)"),
            y=alt.Y("step_time_s:Q", title="Temps de pas (s)"),
            color=alt.Color("distance_km:Q", title="Distance (km)")
        )
        .properties(height=380)
    )
    st.altair_chart(empty_chart, use_container_width=True)
    if record_file is not None and not all(c in df.columns for c in required):
        st.info("ℹ️ Il faut les colonnes 'pente_%', 'step_length' (ms), 'distance' et 'timestamp'.")
else:
    # ===== Nettoyage & conversions =====
    df["pente_%"] = pd.to_numeric(df["pente_%"], errors="coerce")

    # step_length en millisecondes -> secondes
    step_ms = pd.to_numeric(df["step_length"], errors="coerce")
    df["step_time_s"] = (step_ms / 1000.0).clip(lower=0)

    # (optionnel) filtrer les valeurs aberrantes
    # course/trail: ~0.25–0.60 s / marche en côte raide: peut monter >0.8 s
    mask_real = df["step_time_s"].between(0.2, 1.5)
    df.loc[~mask_real, "step_time_s"] = np.nan

    # distance m -> km
    df["distance_km"] = pd.to_numeric(df["distance"], errors="coerce") / 1000.0

    # ===== Bornes sliders =====
    pente_series = df["pente_%"].dropna()
    st_series    = df["step_time_s"].dropna()
    dist_series  = df["distance_km"].dropna()

    pente_min = float(np.nanmin(pente_series)) if len(pente_series) else -30.0
    pente_max = float(np.nanmax(pente_series)) if len(pente_series) else  30.0
    st_min    = float(np.nanmin(st_series))    if len(st_series)    else  0.25
    st_max    = float(np.nanmax(st_series))    if len(st_series)    else  1.00
    dist_min  = float(np.nanmin(dist_series))  if len(dist_series)  else  0.0
    dist_max  = float(np.nanmax(dist_series))  if len(dist_series)  else  1.0

    pente_default = (max(pente_min, -15.0), min(pente_max, 15.0)) if pente_min < pente_max else (-15.0, 15.0)
    st_default    = (max(st_min, 0.28),      min(st_max, 0.90))    if st_min    < st_max    else (0.28, 0.90)
    dist_default  = (dist_min, dist_max)

    c1, c2, c3 = st.columns(3)
    with c1:
        pente_range = st.slider(
            "Pente (%)",
            float(pente_min), float(pente_max),
            (float(pente_default[0]), float(pente_default[1])),
            step=0.5,
            key="st_slider_pente"
        )
    with c2:
        st_range = st.slider(
            "Temps de pas (s)",
            float(st_min), float(st_max),
            (float(st_default[0]), float(st_default[1])),
            step=0.01,
            key="st_slider_time"
        )
    with c3:
        dist_range = st.slider(
            "Distance (km)",
            float(dist_min), float(dist_max),
            (float(dist_default[0]), float(dist_default[1])),
            step=max(0.1, round((dist_max - dist_min) / 200, 2)),
            key="st_slider_distance"
        )

    # ===== Filtrage =====
    chart_df = df[["pente_%", "step_time_s", "distance_km", "timestamp"]].dropna()
    chart_df = chart_df[
        chart_df["pente_%"].between(pente_range[0], pente_range[1]) &
        chart_df["step_time_s"].between(st_range[0], st_range[1]) &
        chart_df["distance_km"].between(dist_range[0], dist_range[1])
    ]

    if chart_df.empty:
        st.warning("Aucun point dans les plages sélectionnées. Ajuste les curseurs.")
    else:
        scatter_st = (
            alt.Chart(chart_df)
            .mark_circle(size=42, opacity=0.85)
            .encode(
                x=alt.X("pente_%:Q", title="Pente (%)",
                        scale=alt.Scale(domain=[pente_range[0], pente_range[1]])),
                y=alt.Y("step_time_s:Q", title="Temps de pas (s)",
                        scale=alt.Scale(domain=[st_range[0], st_range[1]])),
                color=alt.Color("distance_km:Q", title="Distance (km)",
                        scale=alt.Scale(domain=[dist_range[0], dist_range[1]], range=["#00c853", "#ff5252"])),
                tooltip=[
                    alt.Tooltip("timestamp:T", title="Temps"),
                    alt.Tooltip("distance_km:Q", title="Distance (km)", format=".1f"),
                    alt.Tooltip("pente_%:Q", title="Pente (%)", format=".1f"),
                    alt.Tooltip("step_time_s:Q", title="Temps de pas (s)", format=".2f"),
                ],
            )
            .interactive()
            .properties(height=380, width='container')
        )

        st.altair_chart(scatter_st, use_container_width=True)

        st.caption(
            "Lecture : temps de pas ↑ = cadence ↓ (plus de contrôle en descente raide, marche en côte, ou fatigue). "
            "Sur le plat, une hausse nette en fin de course reflète la dérive de cadence."
        )

        # Export PNG HD
        try:
            import altair_saver  # noqa: F401
            export_width = 800
            export_height = 900
            scale_factor = 1

            chart_to_save = scatter_st.properties(width=export_width, height=export_height)
            buf = io.BytesIO()
            chart_to_save.save(
                buf,
                format="png",
                method="vl-convert",
                scale_factor=scale_factor
            )
            buf.seek(0)

            fname = (
                "step_time_vs_pente_"
                f"pente{int(pente_range[0])}-{int(pente_range[1])}_"
                f"time{st_range[0]:.2f}-{st_range[1]:.2f}_"
                f"dist{dist_range[0]:.1f}-{dist_range[1]:.1f}_HD.png"
            )

            st.download_button(
                label="📸 Télécharger le PNG (filtres appliqués, HD)",
                data=buf,
                file_name=fname,
                mime="image/png"
            )
        except Exception as e:
            st.warning(
                "⚠️ Problème avec la génération du PNG.\n"
                "Installe `vl-convert-python` si nécessaire :\n"
                "```bash\npip install vl-convert-python\n```\n"
                f"\nErreur : {e}"
            )



# =========================
# === 📊 Tableau croisé : Temps de pas (s) par Pente (10%) × Progression (20%) + couleurs
# =========================

# 0) step_length (ms) -> step_time_s
df["step_time_s"] = pd.to_numeric(df.get("step_length", np.nan), errors="coerce") / 1000.0
df.loc[~df["step_time_s"].between(0.2, 1.5), "step_time_s"] = np.nan  # optionnel

# 1) Progression (%) (réutilise si déjà calculée)
if "progress_%" not in df.columns:
    total_dist_m = pd.to_numeric(df.get("distance", pd.Series(dtype=float)), errors="coerce").iloc[-1] if len(df) else np.nan
    if pd.notna(total_dist_m) and total_dist_m > 0:
        df["progress_%"] = (pd.to_numeric(df["distance"], errors="coerce") / total_dist_m * 100).clip(0, 100)
    else:
        df["progress_%"] = np.nan

# 2) Bins pente (10%)
edges_pente_st = [-np.inf] + list(np.arange(-40, 50, 10)) + [np.inf]
labels_pente_st = (["< -40%"] +
                   [f"{b} à {b+10}%" for b in range(-40, 40, 10)] +
                   ["> 40%"])
df["pente_bin10_st"] = pd.cut(
    pd.to_numeric(df["pente_%"], errors="coerce"),
    bins=edges_pente_st,
    labels=labels_pente_st,
    include_lowest=True,
    right=False
)

# 3) Bins progression (20%)
edges_prog_st = [0, 20, 40, 60, 80, 100]
labels_prog_st = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]
df["progress_bin20_st"] = pd.cut(
    pd.to_numeric(df["progress_%"], errors="coerce"),
    bins=edges_prog_st,
    labels=labels_prog_st,
    include_lowest=True,
    right=True
)

# 4) Pivot (moyenne du temps de pas)
pivot_st = df.pivot_table(
    index="pente_bin10_st",
    columns="progress_bin20_st",
    values="step_time_s",
    aggfunc="mean"
)

pivot_st_num = pivot_st.apply(pd.to_numeric, errors="coerce")
pivot_st_display = pivot_st_num.round(2)

# 5) Couleurs : vert (court) → jaune → rouge (long)
if pivot_st_num.size and np.isfinite(pivot_st_num.values).any():
    vmin_st = float(np.nanmin(pivot_st_num.values))
    vmax_st = float(np.nanmax(pivot_st_num.values))
    vmid_st = float(np.nanmedian(pivot_st_num.values))
else:
    vmin_st, vmid_st, vmax_st = 0.0, 0.5, 1.0

def _interp_color_st(val, vmin, vmid, vmax):
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "background-color:#f7f7f7; color:#0b63c5;"
    if not np.isfinite(v):
        return "background-color:#f7f7f7; color:#0b63c5;"
    if vmax == vmin:
        t = 0.5
    else:
        t = (v - vmin) / (vmax - vmin)
        t = min(max(t, 0.0), 1.0)

    green = (217, 253, 211)
    yellow = (255, 243, 191)
    red   = (255, 214, 214)

    if t <= 0.5:
        a = t / 0.5
        r = green[0] + a * (yellow[0] - green[0])
        g = green[1] + a * (yellow[1] - green[1])
        b = green[2] + a * (yellow[2] - green[2])
    else:
        a = (t - 0.5) / 0.5
        r = yellow[0] + a * (red[0] - yellow[0])
        g = yellow[1] + a * (red[1] - yellow[1])
        b = yellow[2] + a * (red[2] - yellow[2])
    return f"background-color:#{int(r):02x}{int(g):02x}{int(b):02x}; color:#0b63c5;"

def _style_df_st(data: pd.DataFrame):
    styles = pd.DataFrame("", index=data.index, columns=data.columns)
    for i in data.index:
        for j in data.columns:
            styles.loc[i, j] = _interp_color_st(data.loc[i, j], vmin_st, vmid_st, vmax_st)
    return styles

styled_st = (
    pivot_st_display
    .style
    .apply(_style_df_st, axis=None)
    .format("{:.2f}")  # secondes
    .set_table_styles([
        {"selector": "th", "props": [("color", "#0b63c5"), ("font-weight", "600"), ("white-space", "nowrap")]},
        {"selector": "td", "props": [("border", "1px solid #eee"), ("min-width", "72px")]},
    ])
)

st.markdown("#### 📊 Temps de pas moyen (s) — par pente (10%) × progression (20%)")
st.table(styled_st)

# 6) Export CSV
csv_bytes_st = pivot_st_display.to_csv().encode("utf-8")
st.download_button(
    "⬇️ Exporter le tableau (CSV)",
    data=csv_bytes_st,
    file_name="step_time_par_pente10_et_progress20.csv",
    mime="text/csv"
)

# 7) Export PNG du tableau
import io, matplotlib.pyplot as plt
from matplotlib.table import Table

def fig_from_styled_pivot_st(pvt: pd.DataFrame, vmin: float, vmid: float, vmax: float) -> bytes:
    n_rows, n_cols = pvt.shape
    cell_h, cell_w = 0.36, 1.15
    fig_w = max(6.0, n_cols * cell_w)
    fig_h = max(2.5, (n_rows + 1) * cell_h)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    tbl = Table(ax, bbox=[0, 0, 1, 1])

    col_labels = list(pvt.columns)
    row_labels = list(pvt.index.astype(str))

    def bg_color(val):
        try: v = float(val)
        except (TypeError, ValueError): return "#f7f7f7"
        if not np.isfinite(v): return "#f7f7f7"
        if vmax == vmin: t = 0.5
        else:
            t = (v - vmin) / (vmax - vmin)
            t = min(max(t, 0.0), 1.0)
        green = (217, 253, 211); yellow= (255, 243, 191); red = (255, 214, 214)
        if t <= 0.5:
            a = t / 0.5
            r = green[0] + a * (yellow[0] - green[0])
            g = green[1] + a * (yellow[1] - green[1])
            b = green[2] + a * (yellow[2] - green[2])
        else:
            a = (t - 0.5) / 0.5
            r = yellow[0] + a * (red[0] - yellow[0])
            g = yellow[1] + a * (red[1] - yellow[1])
            b = yellow[2] + a * (red[2] - yellow[2])
        return f"#{int(r):02x}{int(g):02x}{int(b):02x}"

    # Coin vide + en-têtes
    tbl.add_cell(0, 0, cell_w, cell_h, text="", loc="center", facecolor="#ffffff")
    for j, c in enumerate(col_labels, start=1):
        tbl.add_cell(0, j, cell_w, cell_h, text=str(c), loc="center",
                     facecolor="#e9f2ff", edgecolor="#dddddd")

    # Lignes + cellules
    for i, rlab in enumerate(row_labels, start=1):
        tbl.add_cell(i, 0, cell_w, cell_h, text=str(rlab), loc="center",
                     facecolor="#e9f2ff", edgecolor="#dddddd")
        for j, c in enumerate(col_labels, start=1):
            val = pvt.iloc[i-1, j-1]
            txt = "" if pd.isna(val) else f"{float(val):.2f}"
            tbl.add_cell(i, j, cell_w, cell_h, text=txt, loc="center",
                         facecolor=bg_color(val), edgecolor="#eeeeee")

    ax.add_table(tbl)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=240, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return buf.getvalue()

try:
    png_bytes_st = fig_from_styled_pivot_st(pivot_st_display, vmin_st, vmid_st, vmax_st)
    st.download_button(
        "📸 Exporter le tableau en PNG",
        data=png_bytes_st,
        file_name="step_time_par_pente10_et_progress20.png",
        mime="image/png"
    )
except Exception as e:
    st.warning(f"⚠️ Export PNG du tableau impossible : {e}")

# 8) 🧠 Lecture rapide
try:
    def mean_safe(s):
        s = pd.to_numeric(s, errors="coerce")
        val = np.nanmean(s)
        return float(val) if np.isfinite(val) else np.nan

    # Moyennes par profil de pente
    desc_t = mean_safe(df.loc[pd.to_numeric(df["pente_%"], errors="coerce") < -5, "step_time_s"])
    plat_t = mean_safe(df.loc[pd.to_numeric(df["pente_%"], errors="coerce").between(-5, 5), "step_time_s"])
    mont_t = mean_safe(df.loc[pd.to_numeric(df["pente_%"], errors="coerce") > 5, "step_time_s"])

    if pivot_st_num.size and np.isfinite(pivot_st_num.values).any():
        min_val = float(np.nanmin(pivot_st_num.values))
        max_val = float(np.nanmax(pivot_st_num.values))
        mi, mj = np.where(pivot_st_num.values == np.nanmin(pivot_st_num.values))
        Mi, Mj = np.where(pivot_st_num.values == np.nanmax(pivot_st_num.values))
        xi = pivot_st_num.index[int(mi[0])] if len(mi) else "—"
        xj = pivot_st_num.columns[int(mj[0])] if len(mj) else "—"
        XI = pivot_st_num.index[int(Mi[0])] if len(Mi) else "—"
        XJ = pivot_st_num.columns[int(Mj[0])] if len(Mj) else "—"
    else:
        min_val = max_val = np.nan
        xi = xj = XI = XJ = "—"

    st.markdown(
        f"""
**Lecture rapide — Temps de pas (s) :**
- **Moyenne par profil** → Descente: **{(desc_t if np.isfinite(desc_t) else np.nan):.2f} s**, Plat: **{(plat_t if np.isfinite(plat_t) else np.nan):.2f} s**, Montée: **{(mont_t if np.isfinite(mont_t) else np.nan):.2f} s**.  
- **Plus court** observé: **{(min_val if np.isfinite(min_val) else float('nan')):.2f} s** (pente: **{xi}**, progression: **{xj}**).  
- **Plus long** observé: **{(max_val if np.isfinite(max_val) else float('nan')):.2f} s** (pente: **{XI}**, progression: **{XJ}**).  

💡 Interprétation : un **temps de pas plus long** = **cadence plus basse**. Sur le plat en fin de course,
si le temps de pas ↑ fortement, pense **ravito**, **relâchement**, et **micro-rappels de cadence**.
"""
    )
except Exception as e:
    st.info(f"Note: analyse rapide non disponible ({e}).")



# =========================
# === 🧾 Tableau récap global + par pente × progression (slider)
# =========================
st.markdown("### 🧾 Récap entraînement — global & par pente × progression")

# --- Sélecteur de taille des colonnes de progression
bin_size = st.select_slider(
    "Taille des tranches de progression (%)",
    options=[5, 10, 20, 25],
    value=20,
    help="Ex. 20% → 5 colonnes (0–20, 20–40, 40–60, 60–80, 80–100)."
)

# --- Préparation des données de base
df_local = df.copy()
df_local["timestamp"] = pd.to_datetime(df_local["timestamp"], errors="coerce")
df_local = df_local.sort_values("timestamp").reset_index(drop=True)

# Distances
df_local["distance_m"] = pd.to_numeric(df_local.get("distance", np.nan), errors="coerce")
df_local["distance_km"] = df_local["distance_m"] / 1000.0

# Altitude et pente
df_local["alt_m"] = pd.to_numeric(df_local.get("enhanced_altitude", np.nan), errors="coerce")
df_local["pente_%"] = pd.to_numeric(df_local.get("pente_%", np.nan), errors="coerce")

# Cardio
df_local["heart_rate"] = pd.to_numeric(df_local.get("heart_rate", np.nan), errors="coerce")

# 🔎 Glycémie → forcer l'unité en mg/dL
glucose_col = None
for c in df_local.columns:
    if "glucose" in str(c).lower():
        glucose_col = c
        break

if glucose_col is not None:
    g = pd.to_numeric(df_local[glucose_col], errors="coerce")
    name_lower = str(glucose_col).lower()
    if "mmol" in name_lower:
        df_local["glucose_mgdl"] = g * 18.0
    else:
        if np.nanmedian(g) < 30:
            df_local["glucose_mgdl"] = g * 18.0
        else:
            df_local["glucose_mgdl"] = g
else:
    df_local["glucose_mgdl"] = np.nan

# Oscillation verticale & temps de contact
df_local["vertical_oscillation"] = pd.to_numeric(df_local.get("vertical_oscillation", np.nan), errors="coerce")
for possible_col in ["stance_time", "ground_contact_time", "contact_time_ms"]:
    if possible_col in df_local.columns:
        df_local["gct_ms"] = pd.to_numeric(df_local[possible_col], errors="coerce")
        break
else:
    df_local["gct_ms"] = np.nan

# Step time
df_local["step_time_s"] = pd.to_numeric(df_local.get("step_length", np.nan), errors="coerce") / 1000.0
df_local.loc[~df_local["step_time_s"].between(0.2, 1.5), "step_time_s"] = np.nan

# Différences temporelles et distances instantanées
df_local["dt_s"] = df_local["timestamp"].diff().dt.total_seconds()
df_local["ddist_m"] = df_local["distance_m"].diff().clip(lower=0)

# Vitesse et foulée
df_local["speed_m_s"] = (df_local["ddist_m"] / df_local["dt_s"]).replace([np.inf, -np.inf], np.nan)
df_local["speed_m_s"] = df_local["speed_m_s"].rolling(5, min_periods=1, center=True).median()
df_local["stride_len_m_est"] = df_local["speed_m_s"] * df_local["step_time_s"]

# Dénivelé
df_local["dalt"] = df_local["alt_m"].diff()

# Binning pente et progression
edges_pente = [-np.inf] + list(np.arange(-40, 50, 10)) + [np.inf]
labels_pente = ["< -40%"] + [f"{b} à {b+10}%" for b in range(-40, 40, 10)] + ["> 40%"]
df_local["pente_bin10"] = pd.cut(df_local["pente_%"], bins=edges_pente, labels=labels_pente, include_lowest=True, right=False)

total_dist_m = float(df_local["distance_m"].max()) if df_local["distance_m"].notna().any() else np.nan
if np.isfinite(total_dist_m) and total_dist_m > 0:
    df_local["progress_%"] = (df_local["distance_m"] / total_dist_m * 100).clip(0, 100)
else:
    df_local["progress_%"] = np.nan

prog_edges = list(np.arange(0, 100, bin_size)) + [100]
prog_labels = [f"{prog_edges[i]}–{prog_edges[i+1]}%" for i in range(len(prog_edges)-1)]
df_local["progress_bin"] = pd.cut(df_local["progress_%"], bins=prog_edges, labels=prog_labels, include_lowest=True, right=True)

# ---------- Helpers ----------
def secs_to_pace_min_km(sec, dist_km):
    if not np.isfinite(sec) or not np.isfinite(dist_km) or dist_km <= 0:
        return np.nan
    return (sec / 60.0) / dist_km

def vam_m_per_h(dplus_m, secs):
    if not np.isfinite(dplus_m) or not np.isfinite(secs) or secs <= 0:
        return np.nan
    return (dplus_m / secs) * 3600.0

def weighted_mean(series, weights):
    s = pd.to_numeric(series, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    if s.notna().any() and w.notna().any() and np.nansum(w) > 0:
        return float(np.nansum(s * w) / np.nansum(w))
    return float(s.mean(skipna=True))

def group_stats(sub):
    """Calcule les stats pondérées d’un sous-ensemble."""
    t_sec = sub["dt_s"].sum(skipna=True)
    d_km = (sub["ddist_m"].sum(skipna=True) or 0.0) / 1000.0
    dplus = sub["dalt"].clip(lower=0).sum(skipna=True) or 0.0
    dneg  = sub["dalt"].clip(upper=0).sum(skipna=True) or 0.0
    pace = secs_to_pace_min_km(t_sec, d_km)
    vam = vam_m_per_h(dplus, t_sec)
    hr = sub["heart_rate"].mean(skipna=True)
    gly_mgdl = sub["glucose_mgdl"].mean(skipna=True)
    vo  = sub["vertical_oscillation"].mean(skipna=True)
    gct = sub["gct_ms"].mean(skipna=True)
    if "cadence" in sub.columns and sub["cadence"].notna().any():
        cad = sub["cadence"].mean(skipna=True)
    elif sub["step_time_s"].notna().any():
        cad = (60.0 / sub["step_time_s"]).mean(skipna=True)
    else:
        cad = np.nan
    stride = sub["stride_len_m_est"].mean(skipna=True)
    slope_mean = weighted_mean(sub["pente_%"], sub["ddist_m"])

    # Calcul des temps par plages de glycémie (en secondes)
    gly_stats = {}
    if sub["glucose_mgdl"].notna().any():
        gly_ranges = {
            ">350": sub.loc[sub["glucose_mgdl"] > 350, "dt_s"].sum(skipna=True),
            "240–350": sub.loc[sub["glucose_mgdl"].between(240, 350, inclusive="left"), "dt_s"].sum(skipna=True),
            "180–240": sub.loc[sub["glucose_mgdl"].between(180, 240, inclusive="left"), "dt_s"].sum(skipna=True),
            "110–180": sub.loc[sub["glucose_mgdl"].between(110, 180, inclusive="left"), "dt_s"].sum(skipna=True),
            "70–110": sub.loc[sub["glucose_mgdl"].between(70, 110, inclusive="left"), "dt_s"].sum(skipna=True),
            "55–70": sub.loc[sub["glucose_mgdl"].between(55, 70, inclusive="left"), "dt_s"].sum(skipna=True),
            "<55": sub.loc[sub["glucose_mgdl"] < 55, "dt_s"].sum(skipna=True),
        }
        gly_stats = {f"time_{k}_s": v for k, v in gly_ranges.items()}

    return dict(
        temps_s=t_sec, dist_km=d_km, dplus_m=dplus, dneg_m=dneg,
        pace_min_km=pace, vam_m_h=vam, hr_mean=hr, gly_mean=gly_mgdl,
        vo_mm=vo, gct_ms=gct, cad_spm=cad, stride_m=stride,
        slope_mean_pct=slope_mean, **gly_stats
    )

def pace_to_str(x):
    if not np.isfinite(x): return ""
    m = int(np.floor(x)); s = int(round((x - m) * 60))
    if s == 60: m, s = m+1, 0
    return f"{m:02d}:{s:02d}"

def secs_to_hhmm(s):
    if not np.isfinite(s): return ""
    s = int(round(s)); h = s // 3600; m = (s % 3600) // 60
    return f"{h:02d}:{m:02d}"

# ---------- LIGNES GLOBALES ----------
global_stats_list = []
for prog_lab in prog_labels:
    sub = df_local[df_local["progress_bin"] == prog_lab]
    global_stats_list.append(group_stats(sub))

def make_global_row(label, key, round_mode=None, as_time=False, as_pace=False):
    vals = []
    for stt in global_stats_list:
        v = stt.get(key, np.nan)
        if as_time:
            v = secs_to_hhmm(v)
        elif as_pace:
            v = pace_to_str(v)
        elif np.isfinite(v):
            if round_mode == 0:
                v = round(v, 0)
            elif round_mode == 1:
                v = round(v, 1)
            elif round_mode == 2:
                v = round(v, 2)
        else:
            v = np.nan
        vals.append(v)
    df_row = pd.DataFrame([vals], index=["tout"], columns=prog_labels)
    df_row.index = pd.MultiIndex.from_product([[label], df_row.index], names=["Métrique", "Pente"])
    return df_row

global_frames = [
    make_global_row("Pente moyenne (%)", "slope_mean_pct", round_mode=1),
    make_global_row("📏 Distance (km)", "dist_km", round_mode=2),
    make_global_row("🏃 Allure (min/km)", "pace_min_km", as_pace=True),
    make_global_row("❤️ FC moy.", "hr_mean", round_mode=0),
    make_global_row("Cadence (ppm)", "cad_spm", round_mode=0),
    make_global_row("🩸 Glycémie moy. (mg/dL)", "gly_mean", round_mode=0),
    make_global_row("⬆️ D+ (m)", "dplus_m", round_mode=0),
    make_global_row("⬇️ D- (m)", "dneg_m", round_mode=0),
]

# 🔹 Ajout des lignes de temps passés dans chaque plage glycémique
if any("time_" in k for k in global_stats_list[0].keys()):
    gly_ranges = [">350", "240–350", "180–240", "110–180", "70–110", "55–70", "<55"]
    for gr in gly_ranges:
        key = f"time_{gr}_s"
        global_frames.append(make_global_row(f"⏱ Temps glycémie {gr} (hh:mm)", key, as_time=True))

# ---------- DÉTAIL PAR PENTE × PROGRESSION ----------
metrics = [
    ("🏃 Allure (min/km)", "pace_min_km"),
    ("⛰️ VAM (m/h)", "vam_m_h"),
    ("Cadence (ppm)", "cad_spm"),
    ("Oscillation verticale (mm)", "vo_mm"),
    ("Temps contact (ms)", "gct_ms"),
    ("Longueur de foulée (m)", "stride_m"),
    ("📏 Distance (km)", "dist_km"),
    ("⬆️ D+ (m)", "dplus_m"),
    ("⬇️ D- (m)", "dneg_m"),
    ("🩸 Glycémie moy. (mg/dL)", "gly_mean"),
    ("Temps passé (hh:mm)", "temps_s"),
]

frames = []
frames.extend(global_frames)

for metric_name, key in metrics:
    rows = []
    # Filtrage contextuel
    if key == "vam_m_h":
        pente_labels_filtered = [p for p in labels_pente if not p.startswith("-") and not p.startswith("< -")]
    elif key == "dplus_m":
        pente_labels_filtered = [p for p in labels_pente if not p.startswith("-") and not p.startswith("< -")]
    elif key == "dneg_m":
        pente_labels_filtered = [p for p in labels_pente if p.startswith("-") or p.startswith("< -")]
    else:
        pente_labels_filtered = labels_pente

    for pente_lab in pente_labels_filtered:
        row_vals = []
        for prog_lab in prog_labels:
            sub = df_local[(df_local["pente_bin10"] == pente_lab) & (df_local["progress_bin"] == prog_lab)]
            stats = group_stats(sub)
            val = stats.get(key, np.nan)
            # Formats
            if key == "temps_s":
                val = secs_to_hhmm(val)
            elif key == "pace_min_km":
                val = pace_to_str(val)
            elif key in ("vam_m_h", "vo_mm", "gct_ms", "dplus_m", "dneg_m"):
                val = np.nan if not np.isfinite(val) else round(val, 0)
            elif key == "cad_spm":
                val = np.nan if not np.isfinite(val) else round(val, 0)
            elif key in ("stride_m", "dist_km"):
                val = np.nan if not np.isfinite(val) else round(val, 2)
            elif key == "gly_mean":
                val = np.nan if not np.isfinite(val) else round(val, 0)
            row_vals.append(val)
        rows.append(row_vals)

    mdf = pd.DataFrame(rows, index=pente_labels_filtered, columns=prog_labels)
    mdf.index = pd.MultiIndex.from_product([[metric_name], mdf.index], names=["Métrique", "Pente"])
    frames.append(mdf)

# ---------- Assemblage final
pente_progress_df = pd.concat(frames)
st.markdown("#### 🧩 Détail complet — lignes 'tout' en tête puis par pente × progression")
st.dataframe(pente_progress_df, use_container_width=True)

# Export CSV
csv_detail = pente_progress_df.to_csv(index=True).encode("utf-8")
st.download_button(
    "⬇️ Export complet (CSV)",
    data=csv_detail,
    file_name="recap_complet_global_et_par_pente_par_progression.csv",
    mime="text/csv"
)






# =========================
# === 💬 Analyse complète du tableau (IA – CSV virtuel)
# =========================

import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI

st.markdown("### 💬 Analyse complète de la sortie (IA)")

# --- Vérification DataFrame présent
if "pente_progress_df" not in locals() and "pente_progress_df" not in st.session_state:
    st.error("❌ Le tableau 'pente_progress_df' est introuvable. Charge-le avant de lancer l'analyse.")
    st.stop()

pente_progress_df = st.session_state.get("pente_progress_df", locals().get("pente_progress_df"))

# --- Conversion CSV virtuelle
csv_bytes = pente_progress_df.to_csv(index=True).encode("utf-8")

# --- Prompt complet pour GPT
prompt_gpt5 = """
🧠 PROMPT FINAL — ANALYSE PROFESSIONNELLE DU FICHIER pente_progress_analyse.csv

Tu disposes d’un tableau CSV issu d’une analyse de course de trail.
Ce tableau regroupe, pour chaque tranche de pente et pour 5 segments successifs de l’épreuve (de 0–20 % à 80–100 % de progression), les données suivantes :

- Pente moyenne (%) : inclinaison moyenne du terrain.
- Distance (km) : distance parcourue sur ce type de pente.
- Allure (min/km) : vitesse moyenne au sol.
- FC moy. (bpm) : fréquence cardiaque moyenne.
- Cadence (ppm) : nombre moyen de pas par minute.
- Glycémie moy. : si disponible.
- D+ / D– (m) : dénivelés positifs et négatifs cumulés.
- VAM (m/h) : vitesse ascensionnelle moyenne.
- Oscillation verticale (mm) : amplitude verticale du centre de gravité à chaque foulée.
- Temps de contact au sol (ms) : durée d’appui à chaque pas.
- Longueur de foulée (m) : distance parcourue par pas.
- Temps passé (hh:mm) : durée totale passée sur chaque type de pente.

Les 5 colonnes principales correspondent aux 5 segments chronologiques de la course (progression dans le trail, pas la pente).
Les lignes correspondent aux plages de pentes (forte descente à forte montée), ainsi qu’à des moyennes globales (“tout”).

---

🧩 TA MISSION

Produis une analyse complète, professionnelle et argumentée du fichier, selon ce plan :

1) Profil de course (analyse macro)
- Distance et dénivelé total (D+ / D–).
- Structure du parcours (proportion de montées/descentes).
- Évolution des indicateurs clés (allure, FC, cadence, VAM) entre le début et la fin de la course.
- Lecture de la gestion d’effort : constance, endurance, signes de fatigue ou de rupture de rythme.
- Si la répartition du temps montre une concentration sur certaines pentes, interprète ce que cela dit du profil du terrain et de la stratégie adoptée.

2) Profil du coureur (analyse comportementale et physiologique)
- En 3 à 5 phrases, décris les caractéristiques globales du coureur : endurance, résistance, gestion de l’intensité, adaptation technique en montée/descente.
- Appuie-toi sur les données (FC, cadence, VAM, oscillation, temps de contact) pour décrire son style de course : efficient, économique, technique, rigide, relâché, etc.

3) Analyse des métriques clés et interprétation
Analyse les tendances suivantes, en combinant lecture chiffrée et interprétation sportive :

a) Allure vs Pente
→ Identifier les zones de meilleure efficience (où la vitesse relative est la plus stable malgré la pente).
→ Exemple : une forte baisse d’allure sur les pentes 20–30 % après la seconde moitié de la course indique une fatigue musculaire ou une perte de force spécifique.

b) VAM (Vitesse Ascensionnelle Moyenne)
Format le tabelau de la VAM en m/h par pente et progression avec chaque colonne représentant un segment de progression (0-20%, 20-40%, etc.) et chaque ligne une plage de pente que tu retrouves dans le tableau. 
→ Évaluer la constance par pente équivalente.
→ Baisse ≥ 15–20 % à pente égale = perte de force ou gestion prudente.
→ Expliquer ce que cela révèle : endurance musculaire, gestion énergétique, économie de mouvement.

c) Cadence (ppm)
→ Identifier les variations selon la pente et les segments de progression.
→ Chute ≥ 20 % = fatigue neuromusculaire.
→ Relier cadence et foulée : une baisse de cadence associée à une hausse du temps de contact suggère une perte de tonicité.

d) Oscillation verticale (mm)
→ Valeur élevée = rebond vertical, perte d’énergie horizontale.
→ Valeur faible et stable = course fluide, relâchée, efficace.
→ Expliquer comment cela influence la performance.

e) Temps de contact au sol (ms)
→ Donne des indices sur la réactivité musculaire.
→ Hausse progressive > 8–10 % = fatigue ou baisse de puissance.
→ Corréler avec cadence et longueur de foulée.

f) Longueur de foulée (m)
→ Lire son évolution en lien avec la cadence : une foulée plus courte mais stable peut être un choix d’économie, une chute brutale traduit plutôt la fatigue.

g) Répartition du temps par pente
→ Identifier les zones de contrainte : montée longue = endurance musculaire, descente prolongée = contraintes articulaires, etc.

4) Analyse glycémique (si les données sont présentes)
- Produis une lecture approfondie de la glycémie comme indicateur de gestion énergétique et métabolique.
- Décris la tendance globale (stable, en hausse, en baisse, en dents de scie) et compare la glycémie moyenne entre les segments.
- Interprète les temps passés dans chaque plage de glycémie (si disponibles) :
  >350 mg/dL : hyperglycémie marquée — souvent liée à une intensité élevée (zones cardio 4–5), un stress important ou une alimentation trop concentrée en glucides.  
  240–350 mg/dL : glycémie haute soutenue — peut indiquer un excès d’apport ou une réponse hormonale à un effort prolongé en montée (catécholamines).  
  180–240 mg/dL : glycémie haute mais exploitable — souvent observée lors d’efforts soutenus bien gérés.  
  110–180 mg/dL : zone optimale — gestion énergétique stable et efficiente.  
  70–110 mg/dL : glycémie basse maîtrisée — typique d’un effort long en zone 2 ou début zone 3.  
  55–70 mg/dL : seuil critique — déficit d’apport ou excès d’insuline. Une consommation de 50–80 g de glucides/h est souvent nécessaire pour stabiliser.  
  <55 mg/dL : hypoglycémie sévère — perte d’efficacité mécanique et baisse de vigilance.

- Croise ces données avec :
  → la fréquence cardiaque : si glycémie haute + FC zone 4–5 → intensité élevée. Si glycémie haute sans hausse de FC → stress ou apport excessif.  
  → le dénivelé : les montées (D+) tendent à faire monter la glycémie, alors que les descentes sont souvent neutres, sauf si raides et exigeantes mentalement.  
  → les zones cardio : en endurance (Z2–Z3), la glycémie tend à baisser naturellement car le métabolisme gère mieux l’insuline.

- Résume en identifiant la qualité de la gestion énergétique :
  → bonne stabilité glycémique = excellente efficience métabolique.  
  → dérive haute = gestion perfectible ou alimentation trop dense.  
  → dérive basse = sous-apport énergétique ou la baisse de l'intensité peut également avoir causé une meilleure utilisation de l'insuline, il aurait peut etre fallut s'alimenter encore plus.
- Ne pas évoquer le diabète, uniquement la dimension énergétique et métabolique.

5) Forces
- 4 à 6 points synthétiques, avec valeurs ou tendances claires.
- Exemples :
  - “Cadence stable malgré l’augmentation de la pente (+2 % entre segments 2 et 4)”
  - “VAM homogène sur 10–20 % → excellente endurance musculaire.”
  - “Glycémie stable sur la durée (±8 %) → gestion énergétique efficace.”

6) Faiblesses ou points à surveiller
- 4 à 6 points précis, étayés par des données chiffrées.
- Exemples :
  - “Temps de contact au sol : +12 % entre 40–60 % et 80–100 % → fatigue neuromusculaire probable.”
  - “Oscillation verticale élevée sur plat (> 95 mm) → perte d’efficacité mécanique.”
  - “Glycémie en baisse de 25 % entre segments 3 et 5 → déficit d’apport énergétique.”

7) Conclusion synthétique
En 2–3 phrases, résume le diagnostic général : quelle qualité ressort le plus ? Quelle limite apparaît ? Quelle interprétation globale faire de la dynamique de course ?

---

📏 RÈGLES D’ANALYSE À APPLIQUER

Chute de cadence ≥ 20 % : fatigue neuromusculaire  
Hausse temps de contact ≥ 8–10 % : perte de tonicité ou relâchement  
Oscillation verticale > 90 mm : gaspillage énergétique  
Baisse VAM ≥ 15–20 % à pente égale : manque de force spécifique  
Cadence < 120 ppm sur forte pente : passage en marche probable  
Glycémie stable ±10 % : gestion énergétique efficace  
Glycémie en baisse >20 % : sous-apport énergétique ou intensité excessive  
Glycémie en hausse >20 % : surcharge glucidique, stress ou effort de haute intensité (zones cardio 4–5)  
Montées (D+) → hausse glycémie possible par effet hormonal  
Descente raide → hausse ponctuelle possible liée à la concentration

---

🧭 STYLE ATTENDU

- Ton expert, clair, professionnel et fluide, comme un rapport de data analyst sportif.
- Structure rigoureuse, chiffres interprétés, vocabulaire précis mais accessible.
- Expliquer les notions techniques si elles apparaissent (oscillation, temps de contact, VAM, glycémie, etc.).
- Aucune étape de raisonnement visible : produire uniquement le rapport final, complet et lisible.

"""


# --- Bouton unique pour lancer l’analyse IA
if st.button("🧠 Lancer l’analyse complète (IA)"):
    try:
        with st.spinner("Analyse complète en cours..."):
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Tu es un coach expert en data trail running et performance sportive."},
                    {"role": "user", "content": prompt_gpt5},
                    {"role": "user", "content": "Voici le fichier CSV complet de l'analyse de course :"},
                    {"role": "user", "content": csv_bytes.decode("utf-8")},
                ],
                temperature=0.4,
                max_tokens=3000,
            )

            analysis = response.choices[0].message.content
            st.markdown("### 🧩 Rapport d’analyse complet")
            st.write(analysis)

    except Exception as e:
        st.error(f"❌ Erreur lors de l'appel à GPT : {e}")
