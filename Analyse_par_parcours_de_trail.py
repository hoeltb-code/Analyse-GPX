import io
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import re

def ui_key(*parts):
    """Construit une clé unique et sûre pour Streamlit (évite les doublons)."""
    s = "_".join(str(p) for p in parts)
    return re.sub(r"\W+", "_", s).lower()


# 🔧 Helper : convertir n'importe quel timedelta (numpy ou pandas) en secondes
def td_seconds(x):
    if pd.isna(x):
        return np.nan
    try:
        return x.total_seconds()  # pandas.Timedelta
    except AttributeError:
        return pd.Timedelta(x).total_seconds()  # numpy.timedelta64 -> pandas.Timedelta


def _sort_by_km_then_time(df_):
    t = df_.copy()
    if "km" in t.columns and t["km"].notna().any():
        by = [c for c in ["km", "temps_cumule_td"] if c in t.columns]
        t = t.sort_values(by=by, ascending=[True] * len(by))
    else:
        t = t.sort_values(by=["temps_cumule_td"], ascending=[True])
    return t


def _fmt_hms_from_seconds(sec):
    if pd.isna(sec):
        return None
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


st.set_page_config(page_title="Analyse passages UTMB", layout="wide")
st.title("Analyse des passages — UTMB")
st.markdown(
    """
**Ce que fait l’app :**
1) ATTENTION !!!!!! Upload d’un CSV (UTF-8 ; séparateur ; ; décimales , pour km). WARNING !!!!!!
2) Ajoute index_utmb_tranche25 (arrondi au multiple de 25 le plus proche).
3) Affiche les **100 premières lignes** avec cette nouvelle colonne.
4) Menu déroulant pour sélectionner une tranche d’index UTMB.
5) Tableau (km, point, heure de passage moyenne) pour la tranche choisie.
"""
)

uploaded = st.file_uploader(
    "Charge ton fichier CSV (format: UTF-8, séparateur ;, décimales , pour km)",
    type=["csv"],
)


def round_to_nearest_25(x):
    if pd.isna(x):
        return np.nan
    return int(round(float(x) / 25.0) * 25)


@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    # Valeurs à traiter comme manquantes (en plus de celles par défaut)
    extra_na = ["", " ", "-", "—", "na", "NA", "N/A", "n/a", "NULL", "Null", "null"]

    df = pd.read_csv(
        file,
        sep=";",
        decimal=",",
        encoding="utf-8",
        na_values=extra_na,  # <-- clé contre l'erreur
        keep_default_na=True,
        dtype={
            "dossard": "string",
            "ruri": "string",
            "course_code": "string",
            "course": "string",
            "nom": "string",
            "prenom": "string",
            "sexe": "string",
            "categorie": "string",
            "categorie_age": "string",
            "nationalite": "string",
            # On laisse pandas inférer les numériques; les Int64 tolèrent NA
            "clt": "Int64",
            "d_plus": "Int64",
            "point": "string",
        },
    )

    # 🔤 Normaliser les noms de colonnes : trim + lowercase
    df.columns = df.columns.str.strip().str.lower()

    # Nettoyages légers d’espaces
    for col in ["point", "course", "course_code", "nom", "prenom"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    # Forcer les numériques potentiellement sales -> nombres (coerce vers NaN si non convertible)
    for col in ["index_utmb", "km"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ajout de la colonne arrondie par pas de 25
    def round_to_nearest_25(x):
        if pd.isna(x):
            return pd.NA
        return int(round(float(x) / 25.0) * 25)

    if "index_utmb" in df.columns:
        df["index_utmb_tranche25"] = df["index_utmb"].apply(round_to_nearest_25)
    else:
        df["index_utmb_tranche25"] = pd.NA

    # Parsing des heures / durées (tolérant aux valeurs manquantes)
    df["heure_passage_td"] = pd.to_timedelta(df.get("heure_passage", pd.Series(dtype="string")), errors="coerce")
    df["temps_cumule_td"] = pd.to_timedelta(df.get("temps_cumule", pd.Series(dtype="string")), errors="coerce")

    return df


if uploaded is not None:
    # Lecture
    df = load_csv(uploaded)

    st.subheader("Aperçu — 100 premières lignes (avec index_utmb_tranche25)")
    st.dataframe(df.head(100), use_container_width=True)

    # Liste des tranches disponibles (propres, triées)
    tranches = (
        df["index_utmb_tranche25"]
        .dropna()
        .drop_duplicates()
        .astype(int)
        .sort_values()
        .tolist()
        if "index_utmb_tranche25" in df.columns
        else []
    )

    if len(tranches) == 0:
        st.info("Aucune tranche d’index UTMB détectée dans le fichier.")
    else:
        st.subheader("Filtrer par tranche d’index UTMB (pas de 25)")
        selected_tranche = st.selectbox(
            "Choisis une tranche d’index UTMB",
            options=tranches,
            index=min(
                len(tranches) - 1,
                max(0, tranches.index(tranches[len(tranches) // 2])),
            )
            if len(tranches)
            else 0,
            format_func=lambda x: f"{x}",
        )

        # Filtre tranche
        df_sel = df[df["index_utmb_tranche25"] == selected_tranche].copy()

        # Colonnes de regroupement
        group_cols = []
        if "km" in df_sel.columns:
            group_cols.append("km")
        if "point" in df_sel.columns:
            group_cols.append("point")

        if "temps_cumule_td" in df_sel.columns and len(group_cols) >= 1:
            # Agrégation sur TEMPS CUMULÉ : moyenne & médiane sur valeurs > 0, + nb valides
            agg = (
                df_sel.groupby(group_cols, dropna=True)
                .agg(
                    temps_cumule_moy=("temps_cumule_td", lambda s: s[s > pd.Timedelta(0)].mean()),
                    temps_cumule_med=("temps_cumule_td", lambda s: s[s > pd.Timedelta(0)].median()),
                    nb_donnees_valides=("temps_cumule_td", lambda s: (s > pd.Timedelta(0)).sum()),
                )
                .reset_index()
            )

            # On enlève d'emblée les points avec 0 donnée valide
            agg = agg[agg["nb_donnees_valides"] > 0].copy()

            # Exclure d’emblée les points sans donnée valide
            agg = agg[agg["nb_donnees_valides"] > 0].copy()

            # Mise en forme HH:MM:SS
            def td_to_hms(td):
                if pd.isna(td):
                    return ""
                total = int(td.total_seconds())
                h = total // 3600
                m = (total % 3600) // 60
                s = total % 60
                return f"{h:02d}:{m:02d}:{s:02d}"

            agg["temps_moyen_cumule"] = agg["temps_cumule_moy"].apply(td_to_hms)
            agg["temps_median_cumule"] = agg["temps_cumule_med"].apply(td_to_hms)
            agg = agg.drop(columns=["temps_cumule_moy", "temps_cumule_med"])

            # Tri
            if "km" in agg.columns:
                agg = agg.sort_values(by=["km", "point"] if "point" in agg.columns else ["km"])
            elif "point" in agg.columns:
                agg = agg.sort_values(by=["point"])

            # ====== Paramètres de filtre interactifs ======
            st.subheader(f"Tableau — temps cumulé (moyenne & médiane) — tranche {selected_tranche}")
            c1, c2, c3 = st.columns([1, 1, 1])

            with c1:
                filtre_on = st.checkbox("Activer le filtre par % de données valides", value=True)
            with c2:
                seuil_pct = st.number_input("Seuil (%) par rapport au 1er point", min_value=0, max_value=100, value=30, step=5)
            with c3:
                n0_min = st.number_input("N0 minimal pour appliquer le filtre", min_value=0, max_value=1000, value=5, step=1)
            # ==============================================

            # --- Filtrage selon la règle configurable ---
            seuil_info = ""
            if not agg.empty:
                n0 = int(agg["nb_donnees_valides"].iloc[0])  # premier point après tri
                if filtre_on and n0 >= n0_min:
                    seuil_count = int(np.ceil((seuil_pct / 100.0) * n0))
                    agg = agg[agg["nb_donnees_valides"] >= seuil_count].copy()
                    seuil_info = f"(N0={n0}, seuil {seuil_pct}% ⇒ ≥ {seuil_count} données valides)"
                else:
                    if not filtre_on:
                        seuil_info = "(filtre désactivé)"
                    else:
                        seuil_info = f"(N0={n0} < N0 minimal {n0_min} ⇒ pas de filtre)"
            # ---------------------------------------------

            if agg.empty:
                st.info("Aucun point n’atteint le seuil (ou pas de données valides). " + seuil_info)
            else:
                st.dataframe(agg, use_container_width=True)
                st.caption(
                    "Affichage basé sur **temps_cumule** : "
                    "**temps_moyen_cumule** (moyenne) et **temps_median_cumule** (médiane). "
                    "‘nb_donnees_valides’ = nombre d’enregistrements avec un **temps cumulé > 00:00:00** dans la tranche. "
                    "Les lignes à 0 donnée valide sont exclues. "
                    f"Si activé, seules celles avec ≥ {seuil_pct}% de N0 sont conservées. {seuil_info}"
                )
        else:
            st.warning("Colonnes ‘km’, ‘point’ et/ou ‘temps_cumule’ manquantes pour construire le tableau.")
else:
    st.info("➡️ Uploade un fichier CSV pour commencer.")

# =======================
# VUE PAR DOSSARD (fichier chargé)
# =======================
st.markdown("---")
st.header("Vue par dossard")

if uploaded is not None:
    # ---- Liste des dossards + Nom/Prénom + tri naturel ----
    if "dossard" in df.columns:
        tmp = df[["dossard", "nom", "prenom"]].copy()
        tmp["dossard"] = tmp["dossard"].astype(str).str.strip()
        tmp["nom"] = tmp.get("nom", pd.Series(dtype="string")).astype("string").fillna("").str.strip()
        tmp["prenom"] = tmp.get("prenom", pd.Series(dtype="string")).astype("string").fillna("").str.strip()

        # 1 ligne par dossard (prend le 1er nom/prénom dispo)
        bib_info = (
            tmp.groupby("dossard", as_index=False)
            .agg(nom=("nom", "first"), prenom=("prenom", "first"))
        )

        # Tri : numérique si tous digits, sinon tri "naturel" (1,2,10…)
        bibs_all = bib_info["dossard"].tolist()

        def natural_key(s: str):
            return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

        if all(b.isdigit() for b in bibs_all):
            bib_info = bib_info.sort_values(by="dossard", key=lambda s: s.astype(int))
        else:
            bib_info = bib_info.sort_values(by="dossard", key=lambda s: s.astype(str).map(natural_key))

        # Libellé "dossard - Nom - Prénom"
        def label_row(r):
            parts = [str(r["dossard"]).strip()]
            if isinstance(r["nom"], str) and r["nom"]:
                parts.append(r["nom"])
            if isinstance(r["prenom"], str) and r["prenom"]:
                parts.append(r["prenom"])
            return " - ".join(parts)

        bib_info["label"] = bib_info.apply(label_row, axis=1)
        labels = bib_info["label"].tolist()
        label_to_bib = dict(zip(bib_info["label"], bib_info["dossard"].astype(str)))
        bib_set = set(bib_info["dossard"].astype(str))
    else:
        labels, label_to_bib, bib_set = [], {}, set()

    if len(labels) == 0:
        st.info("Aucun dossard détecté dans le fichier.")
    else:
        # Champ texte + selectbox
        c1, c2 = st.columns([1, 1])
        with c1:
            typed_bib = st.text_input("Tape un dossard (optionnel)", value="").strip()
        with c2:
            selected_label = st.selectbox("…ou sélectionne dans la liste", options=labels)

        # Priorité au dossard saisi s'il existe
        if typed_bib and typed_bib in bib_set:
            selected_bib = typed_bib
        else:
            if typed_bib and typed_bib not in bib_set:
                st.warning("Dossard saisi introuvable — on utilise la sélection de la liste.")
            selected_bib = label_to_bib.get(selected_label)

        # -------- Données du dossard sélectionné --------
        df_bib = df[df["dossard"].astype(str).str.strip() == selected_bib].copy()

        # Utilitaires
        def first_nonnull(series):
            s = series.dropna()
            return s.iloc[0] if not s.empty else pd.NA

        def td_to_hms(td):
            if pd.isna(td):
                return ""
            total = int(td.total_seconds())
            h = total // 3600
            m = (total % 3600) // 60
            s = total % 60
            return f"{h:02d}:{m:02d}:{s:02d}"

        # Infos athlète
        nom = first_nonnull(df_bib.get("nom", pd.Series(dtype="string")))
        prenom = first_nonnull(df_bib.get("prenom", pd.Series(dtype="string")))
        index_utmb_val = first_nonnull(df_bib.get("index_utmb", pd.Series(dtype="float")))
        sexe_val = first_nonnull(df_bib.get("sexe", pd.Series(dtype="string")))
        if pd.notna(sexe_val):
            sexe_val = str(sexe_val).strip().upper()
        try:
            index_utmb_val = ("" if pd.isna(index_utmb_val) else int(round(float(index_utmb_val))))
        except Exception:
            index_utmb_val = ""

        # Dernier point : km max puis, à km égal, temps cumulé max (sinon temps max)
        if "km" in df_bib.columns and df_bib["km"].notna().any():
            tmp2 = df_bib.copy()
            if "temps_cumule_td" in tmp2.columns:
                tmp2 = tmp2.sort_values(by=["km", "temps_cumule_td"], ascending=[True, True])
            else:
                tmp2 = tmp2.sort_values(by=["km"], ascending=[True])
            last_row = tmp2.iloc[-1]
        elif "temps_cumule_td" in df_bib.columns and df_bib["temps_cumule_td"].notna().any():
            tmp2 = df_bib.dropna(subset=["temps_cumule_td"]).sort_values("temps_cumule_td")
            last_row = tmp2.iloc[-1]
        else:
            last_row = df_bib.iloc[-1]

        last_point = last_row.get("point", pd.NA)
        last_clt = last_row.get("clt", pd.NA)

        # En-tête récap
        c1, c2, c3 = st.columns([2, 1, 2])
        with c1:
            st.markdown(
                f"**{prenom if pd.notna(prenom) else ''} {nom if pd.notna(nom) else ''}** \n"
                f"Dossard **{selected_bib}**"
            )
        with c2:
            st.markdown(
                f"Index UTMB : **{index_utmb_val if index_utmb_val != '' else '—'}** \n"
                f"Sexe : **{sexe_val if sexe_val not in [None, '', 'nan'] else '—'}**"
            )
        with c3:
            st.markdown(
                f"Dernier point : **{last_point if pd.notna(last_point) else '—'}** \n"
                f"Classement à ce point : **{'' if pd.isna(last_clt) else int(last_clt)}**"
                if pd.notna(last_clt)
                else "Classement à ce point : **—**"
            )

        # Tableau km, point, clt, temps cumulé — 1 ligne par point
        # ---------- Passages du dossard (1 ligne pour CHAQUE point de la course) ----------
        # Parse temps si besoin
        if "temps_cumule_td" not in df_bib.columns and "temps_cumule" in df_bib.columns:
            df_bib["temps_cumule_td"] = pd.to_timedelta(df_bib["temps_cumule"], errors="coerce")

        # 1) Référence des points de la course (même épreuve que le dossard si possible)
        df_course = df.copy()
        course_mask = None
        for k in ["course_code", "course"]:
            if k in df_bib.columns:
                val = df_bib[k].dropna().astype(str).str.strip()
                if len(val) > 0 and val.iloc[0] != "":
                    course_mask = (df[k].astype(str).str.strip() == val.iloc[0])
                    break
        if course_mask is not None and course_mask.any():
            df_course = df[course_mask].copy()

        # Liste des points + km de référence (km min/1er non-null), tri par km puis point
        ref_points = df_course.loc[:, [c for c in ["point", "km"] if c in df_course.columns]].copy()
        if "point" not in ref_points.columns:
            st.warning("Impossible de construire la liste de points : colonne 'point' absente.")
            ref_points = pd.DataFrame(columns=["point", "km"])

        # km de référence : on prend le plus petit km non nul par point (ou le premier dispo)
        if "km" in ref_points.columns:
            ref_points = (
                ref_points.dropna(subset=["point"])
                .sort_values(by=["point", "km"], na_position="last")
                .groupby("point", as_index=False).first()
            )
        else:
            ref_points = ref_points.dropna(subset=["point"]).drop_duplicates(subset=["point"]).assign(km=np.nan)

        # Tri d’affichage
        if "km" in ref_points.columns:
            ref_points = ref_points.sort_values(by=["km", "point"], ascending=[True, True])
        else:
            ref_points = ref_points.sort_values(by=["point"], ascending=[True])

        # 2) Pour le dossard : 1 ligne max par point (temps cumulé max puis km max)
        rep = df_bib.copy()
        if "km" not in rep.columns:
            rep["km"] = np.nan
        if "temps_cumule_td" not in rep.columns:
            rep["temps_cumule_td"] = pd.NaT

        # On garde la ligne "la plus avancée" par point (trier puis garder la dernière)
        rep = rep.sort_values(by=["point", "temps_cumule_td", "km"], ascending=[True, True, True])
        rep = rep.groupby("point", as_index=False, sort=False).tail(1)

        # Colonnes utiles côté dossard
        keep_cols = [c for c in ["point", "km", "clt", "temps_cumule_td"] if c in rep.columns]
        rep = rep[keep_cols]

        # 3) Jointure référence (tous les points) ⟵ dossard
        tab = ref_points.merge(rep, on="point", how="left", suffixes=("_ref", ""))

        # km final : priorité au km de ref s’il existe
        if "km_ref" in tab.columns:
            tab["km_final"] = tab["km_ref"].where(tab["km_ref"].notna(), tab.get("km"))
        else:
            tab["km_final"] = tab.get("km")

        # Format HH:MM:SS
        def td_to_hms(td):
            if pd.isna(td):
                return ""
            total = int(td.total_seconds())
            h = total // 3600
            m = (total % 3600) // 60
            s = total % 60
            return f"{h:02d}:{m:02d}:{s:02d}"

        tab["temps_cumule_fmt"] = tab.get("temps_cumule_td", pd.NaT).apply(td_to_hms)

        # Colonnes finales et tri
        show_cols = [c for c in ["km_final", "point", "clt", "temps_cumule_fmt"] if c in tab.columns]
        tab = tab[show_cols].rename(columns={"km_final": "km", "temps_cumule_fmt": "temps_cumule"})

        if "km" in tab.columns:
            tab = tab.sort_values(by=["km", "point"], ascending=[True, True])
        else:
            tab = tab.sort_values(by=["point"], ascending=[True])

        st.subheader("Passages du dossard (tous les points de la course)")
        st.dataframe(tab, use_container_width=True, height=min(900, 38 * (len(tab) + 2)))
        st.caption(f"Nombre de points affichés : {len(tab)} (liste de référence de la course).")

        # -------------------------------------------------------------------------------
        # =======================
        # 📈 Nuage de points — Classement au point vs Index UTMB (Altair)
        # =======================
        st.markdown("### 📈 Classement au point vs Index UTMB (toutes tranches, mise en évidence de ta tranche)")

        import io
        import altair as alt

        # 1) Déterminer la course du dossard pour extraire la liste des points de référence
        df_course = df.copy()
        course_mask = None
        for k in ["course_code", "course"]:
            if k in df_bib.columns:
                val = df_bib[k].dropna().astype(str).str.strip()
                if len(val) > 0 and val.iloc[0] != "":
                    course_mask = (df[k].astype(str).str.strip() == val.iloc[0])
                    break
        if course_mask is not None and course_mask.any():
            df_course = df[course_mask].copy()

        # 2) Construire le sélecteur “km – point”, trié par km croissant
        points_ref = df_course.loc[:, [c for c in ["point", "km"] if c in df_course.columns]].copy()
        if "point" in points_ref.columns:
            if "km" in points_ref.columns:
                points_ref = (
                    points_ref.dropna(subset=["point"])
                    .sort_values(by=["point", "km"], na_position="last")
                    .groupby("point", as_index=False).first()
                    .sort_values(by=["km", "point"])
                )
            else:
                points_ref = (
                    points_ref.dropna(subset=["point"]).drop_duplicates(subset=["point"])
                    .assign(km=np.nan).sort_values(by=["point"])
                )

            def label_km_point(r):
                if "km" in r and pd.notna(r["km"]):
                    val = float(r["km"])
                    km_txt = f"{val:.1f}".rstrip("0").rstrip(".")
                    return f"{km_txt} km - {r['point']}"
                return f"{r['point']}"

            points_ref["label"] = points_ref.apply(label_km_point, axis=1)
            point_labels = points_ref["label"].tolist()
        else:
            point_labels = []

        if len(point_labels) == 0:
            st.info("Aucun point de passage trouvé pour cette course.")
        else:
            # Par défaut: sélectionner le DERNIER point de la liste
            default_index = max(0, len(point_labels) - 1)

            selected_point_label = st.selectbox(
                "Choisis un point de course (km - point)",
                options=point_labels,
                index=default_index,                    # 👈 dernier élément
                key="scatter_point_selector_last"
            )

            row_sel = points_ref.loc[points_ref["label"] == selected_point_label].iloc[0]
            sel_point = row_sel["point"]
            sel_km = row_sel.get("km", np.nan)

            row_sel = points_ref.loc[points_ref["label"] == selected_point_label].iloc[0]
            sel_point = row_sel["point"]
            sel_km = row_sel.get("km", np.nan)

            # 3) Préparer les données pour le scatter : 1 ligne par dossard au point choisi
            sub = df_course.copy()

            # S'assurer que index_utmb_tranche25 est présent
            if "index_utmb_tranche25" not in sub.columns and "index_utmb" in sub.columns:
                sub["index_utmb_tranche25"] = pd.to_numeric(sub["index_utmb"], errors="coerce").apply(
                    lambda x: int(round(float(x) / 25.0) * 25) if pd.notna(x) else np.nan
                )

            # Filtrer sur le point choisi (par nom + km si dispo pour précision)
            if "km" in sub.columns and pd.notna(sel_km):
                sub = sub[(sub["point"] == sel_point) & (np.isclose(sub["km"], sel_km, atol=1e-3))]
            else:
                sub = sub[(sub["point"] == sel_point)]

            # Garder les champs nécessaires et nettoyer (+ sexe)
            need_cols = ["dossard", "index_utmb", "index_utmb_tranche25", "clt", "point", "km", "sexe"]
            sub = sub[[c for c in need_cols if c in sub.columns]].copy()
            sub["index_utmb"] = pd.to_numeric(sub["index_utmb"], errors="coerce")
            sub["clt"] = pd.to_numeric(sub["clt"], errors="coerce")
            if "sexe" in sub.columns:
                sub["sexe"] = sub["sexe"].astype(str).str.strip().str.upper()
            sub = sub.dropna(subset=["dossard", "index_utmb", "clt", "index_utmb_tranche25"])

            # Si doublons par dossard au même point, garder la dernière ligne
            sub = sub.sort_values(by=["dossard"]).groupby("dossard", as_index=False, sort=False).tail(1)

            if sub.empty:
                st.warning("Aucune donnée exploitable sur ce point.")
            else:
                # 4) Déterminer la tranche du dossard sélectionné
                utmb_sel_tranche = df_bib.get("index_utmb_tranche25")
                if utmb_sel_tranche is None or utmb_sel_tranche.isna().all():
                    st.warning("Impossible d’identifier la tranche UTMB du dossard sélectionné.")
                else:
                    tranche_sel = int(utmb_sel_tranche.dropna().iloc[0])

                    # 5) Catégoriser toutes les lignes par rapport à la tranche sélectionnée
                    def bucket_tranche(t):
                        if pd.isna(t):
                            return "inconnue"
                        t = int(t)
                        if t == tranche_sel:
                            return "meme_tranche"
                        elif t < tranche_sel:
                            return "tranches_inferieures"
                        else:
                            return "tranches_superieures"

                    sub["bucket"] = sub["index_utmb_tranche25"].apply(bucket_tranche)

                    # 5bis) Filtre de sexe (optionnel) — garder le point rouge quoi qu'il arrive
                    sex_options = ["Tous"]
                    if "sexe" in sub.columns:
                        sex_options = ["Tous"] + [s for s in ["F", "M"] if s in set(sub["sexe"].dropna())]
                    sex_choice = st.radio("Sexe", sex_options, horizontal=True, key=f"scatter_sex_{sel_point}")

                    chart_df_base = sub.copy()
                    if sex_choice in ("F", "M") and "sexe" in chart_df_base.columns:
                        chart_df_base = chart_df_base[chart_df_base["sexe"] == sex_choice]

                    # 6) Sliders dynamiques pour limiter les domaines X/Y (sur le df potentiellement filtré par sexe)
                    idx_min_data = float(np.nanmin(chart_df_base["index_utmb"])) if len(chart_df_base) else 0.0
                    idx_max_data = float(np.nanmax(chart_df_base["index_utmb"])) if len(chart_df_base) else 1000.0
                    clt_min_data = float(np.nanmin(chart_df_base["clt"])) if len(chart_df_base) else 1.0
                    clt_max_data = float(np.nanmax(chart_df_base["clt"])) if len(chart_df_base) else 10000.0

                    # Valeurs par défaut raisonnables
                    idx_default = (
                        max(idx_min_data, tranche_sel - 100),
                        min(idx_max_data, tranche_sel + 100),
                    )
                    clt_default = (clt_min_data, min(clt_max_data, clt_min_data + 500))

                    c1, c2 = st.columns(2)
                    with c1:
                        idx_range = st.slider(
                            "Index UTMB affiché",
                            float(idx_min_data),
                            float(idx_max_data),
                            (float(idx_default[0]), float(idx_default[1])),
                            step=1.0,
                            key=f"scatter_idx_range_{sel_point}"
                        )
                    with c2:
                        clt_range = st.slider(
                            "Classement affiché",
                            float(clt_min_data),
                            float(clt_max_data),
                            (float(clt_default[0]), float(clt_default[1])),
                            step=1.0,
                            key=f"scatter_clt_range_{sel_point}"
                        )

                    chart_df = chart_df_base[
                        chart_df_base["index_utmb"].between(idx_range[0], idx_range[1])
                        & chart_df_base["clt"].between(clt_range[0], clt_range[1])
                    ].copy()

                    # 7) Préparer couleurs
                    color_scale = alt.Scale(
                        domain=["tranches_inferieures", "meme_tranche", "tranches_superieures"],
                        range=["#8be38b", "#ffd84d", "#ff9999"]  # vert clair, jaune, rouge clair
                    )

                    # 8) Scatter de base (tous les points, bucket en couleur)
                    base = (
                        alt.Chart(chart_df)
                        .mark_circle(opacity=0.85, size=40)
                        .encode(
                            x=alt.X("index_utmb:Q", title="Index UTMB",
                                    scale=alt.Scale(domain=[idx_range[0], idx_range[1]])),
                            y=alt.Y("clt:Q", title=f"Classement à « {selected_point_label} »",
                                    scale=alt.Scale(domain=[clt_range[0], clt_range[1]])),
                            color=alt.Color(
                                "bucket:N",
                                title="Position vs tranche sélectionnée",
                                scale=color_scale,
                                legend=alt.Legend(orient="bottom", columns=3)  # légende en bas
                            ),
                            tooltip=[
                                alt.Tooltip("dossard:N", title="Dossard"),
                                alt.Tooltip("sexe:N", title="Sexe") if "sexe" in chart_df.columns else alt.Tooltip("dossard:N", title=""),
                                alt.Tooltip("index_utmb:Q", title="Index UTMB", format=".0f"),
                                alt.Tooltip("index_utmb_tranche25:Q", title="Tranche (25)"),
                                alt.Tooltip("clt:Q", title="Classement", format=".0f"),
                            ],
                        )
                        .properties(height=420)
                        .interactive()
                    )

                    # 9) Marqueur spécial pour le dossard sélectionné (rouge vif, 2× plus gros, contour noir)
                    # -> on repart du df "sub" (non filtré par sexe), mais on applique les bornes X/Y
                    highlight_df = sub[sub["dossard"].astype(str).str.strip() == str(selected_bib)].copy()
                    highlight_df = highlight_df[
                        highlight_df["index_utmb"].between(idx_range[0], idx_range[1])
                        & highlight_df["clt"].between(clt_range[0], clt_range[1])
                    ]

                    highlight = (
                        alt.Chart(highlight_df)
                        .mark_circle(size=150, color="#ff0000", stroke="black", strokeWidth=1.0, opacity=1.0)
                        .encode(
                            x="index_utmb:Q",
                            y="clt:Q",
                            tooltip=[
                                alt.Tooltip("dossard:N", title="Dossard (sélectionné)"),
                                alt.Tooltip("sexe:N", title="Sexe") if "sexe" in highlight_df.columns else alt.Tooltip("dossard:N", title=""),
                                alt.Tooltip("index_utmb:Q", title="Index UTMB", format=".0f"),
                                alt.Tooltip("clt:Q", title="Classement", format=".0f"),
                            ],
                        )
                    )

            chart = base + highlight
            st.altair_chart(chart, use_container_width=True)

            # 10) Export PNG (mêmes filtres)
            try:
                import altair_saver  # noqa: F401
                export_w = 400
                export_h = 450
                scale_factor = 6

                chart_to_save = chart.properties(width=export_w, height=export_h)

                buf = io.BytesIO()
                chart_to_save.save(
                    buf,
                    format="png",
                    method="vl-convert",   # nécessite: pip install vl-convert-python
                    scale_factor=scale_factor
                )
                buf.seek(0)

                fname = (
                    f"utmb_vs_clt_point_{str(sel_point).replace(' ','_')}_"
                    f"idx{int(idx_range[0])}-{int(idx_range[1])}_"
                    f"clt{int(clt_range[0])}-{int(clt_range[1])}.png"
                )

                st.download_button(
                    label="📸 Télécharger le graphique en PNG",
                    data=buf,
                    file_name=fname,
                    mime="image/png",
                    key=f"dl_png_scatter_utmb_clt_{sel_point}"
                )
            except Exception as e:
                st.warning(
                    "⚠️ Export PNG indisponible. Installe d'abord :\n"
                    "```bash\npip install vl-convert-python\n```\n"
                    f"Erreur : {e}"
                )

# ==========================================================
# ⏱️📈 Nuage — Temps cumulé au point vs Index UTMB (+ axe HH:MM & export PNG) — VERSION SAFE (sans Timedelta)
# ==========================================================
st.markdown("### ⏱️📈 Temps cumulé au point vs Index UTMB (même point sélectionné)")

import io
import altair as alt

# Sécurités : réutilise df_course / sel_point / sel_km créés au-dessus, sinon fallback minimal
if 'df_course' not in locals():
    df_course = df.copy()
if 'selected_point_label' not in locals() or 'sel_point' not in locals() or 'sel_km' not in locals():
    pts_fb = df_course.loc[:, [c for c in ["point","km"] if c in df_course.columns]].dropna(subset=["point"])
    if len(pts_fb):
        pts_fb = (pts_fb.sort_values(by=["point","km"], na_position="last")
                        .groupby("point", as_index=False).first()
                        .sort_values(by=["km","point"]))
        sel_point = pts_fb.iloc[0]["point"]
        sel_km = pts_fb.iloc[0].get("km", np.nan)
        selected_point_label = f"{float(sel_km):.1f} km - {sel_point}" if pd.notna(sel_km) else str(sel_point)
    else:
        sel_point, sel_km, selected_point_label = None, np.nan, "—"

# 1) Base au point choisi
sub_tc = df_course.copy()

# Colonnes nécessaires
if "index_utmb_tranche25" not in sub_tc.columns and "index_utmb" in sub_tc.columns:
    sub_tc["index_utmb_tranche25"] = pd.to_numeric(sub_tc["index_utmb"], errors="coerce").apply(
        lambda x: int(round(float(x)/25.0)*25) if pd.notna(x) else np.nan
    )
if "temps_cumule_td" not in sub_tc.columns and "temps_cumule" in sub_tc.columns:
    sub_tc["temps_cumule_td"] = pd.to_timedelta(sub_tc["temps_cumule"], errors="coerce")

# Filtre sur le point choisi
if sel_point is not None:
    if "km" in sub_tc.columns and pd.notna(sel_km):
        sub_tc = sub_tc[(sub_tc["point"] == sel_point) & (np.isclose(sub_tc["km"], sel_km, atol=1e-3))]
    else:
        sub_tc = sub_tc[(sub_tc["point"] == sel_point)]

# Garde colonnes utiles
need_cols = ["dossard","index_utmb","index_utmb_tranche25","temps_cumule_td","sexe"]
sub_tc = sub_tc[[c for c in need_cols if c in sub_tc.columns]].copy()
sub_tc["index_utmb"] = pd.to_numeric(sub_tc.get("index_utmb"), errors="coerce")
if "sexe" in sub_tc.columns:
    sub_tc["sexe"] = sub_tc["sexe"].astype(str).str.strip().str.upper()

# 1 ligne par dossard
sub_tc = sub_tc.dropna(subset=["dossard","index_utmb","temps_cumule_td","index_utmb_tranche25"])
sub_tc = sub_tc.sort_values(by=["dossard"]).groupby("dossard", as_index=False, sort=False).tail(1)

if sub_tc.empty:
    st.info("Aucune donnée exploitable (temps cumulé vs index) sur ce point.")
else:
    # 2) minutes + hms (texte) — et suppression de toute colonne Timedelta pour Altair
    def td_to_minutes(td):
        return float(pd.Timedelta(td).total_seconds())/60.0 if pd.notna(td) else np.nan

    def minutes_to_hms(m):
        if pd.isna(m): return None
        s = int(round(float(m)*60))
        h = s // 3600
        m2 = (s % 3600) // 60
        s2 = s % 60
        return f"{h:02d}:{m2:02d}:{s2:02d}"

    chart_df_tc = sub_tc.copy()
    chart_df_tc["minutes"] = chart_df_tc["temps_cumule_td"].apply(td_to_minutes).astype(float)
    chart_df_tc["hms"] = chart_df_tc["minutes"].apply(minutes_to_hms)
    if "temps_cumule_td" in chart_df_tc.columns:
        chart_df_tc = chart_df_tc.drop(columns=["temps_cumule_td"])  # ⬅️ supprime Timedelta

    # 3) Buckets couleur par rapport à la tranche du dossard sélectionné
    utmb_sel_tranche = df_bib.get("index_utmb_tranche25") if 'df_bib' in locals() else None
    tranche_sel_time = int(utmb_sel_tranche.dropna().iloc[0]) if (utmb_sel_tranche is not None and not utmb_sel_tranche.isna().all()) else None

    def bucket_tranche_time(t):
        if pd.isna(t) or tranche_sel_time is None:
            return "inconnue"
        t = int(t)
        if t == tranche_sel_time:
            return "meme_tranche"
        elif t < tranche_sel_time:
            return "tranches_inferieures"
        else:
            return "tranches_superieures"

    chart_df_tc["bucket"] = chart_df_tc["index_utmb_tranche25"].apply(bucket_tranche_time)

    # 4) Filtre de sexe
    sex_options = ["Tous"]
    if "sexe" in chart_df_tc.columns:
        vals = [s for s in ["F","M"] if s in set(chart_df_tc["sexe"].dropna())]
        if vals:
            sex_options += vals
    sex_choice_tc = st.radio("Sexe", sex_options, horizontal=True, key=ui_key("sex_tc_scatter_safe"))

    chart_df_base_tc = chart_df_tc.copy()
    if sex_choice_tc in ("F","M") and "sexe" in chart_df_base_tc.columns:
        chart_df_base_tc = chart_df_base_tc[chart_df_base_tc["sexe"] == sex_choice_tc]

    # 5) Sliders domaines (Index & Temps en minutes)
    idx_min_data = float(np.nanmin(chart_df_base_tc["index_utmb"])) if len(chart_df_base_tc) else 0.0
    idx_max_data = float(np.nanmax(chart_df_base_tc["index_utmb"])) if len(chart_df_base_tc) else 1000.0
    min_min_data = float(np.nanmin(chart_df_base_tc["minutes"])) if len(chart_df_base_tc) else 0.0
    max_min_data = float(np.nanmax(chart_df_base_tc["minutes"])) if len(chart_df_base_tc) else 60.0

    if tranche_sel_time is not None:
        idx_default = (max(idx_min_data, tranche_sel_time - 100), min(idx_max_data, tranche_sel_time + 100))
    else:
        idx_default = (idx_min_data, idx_max_data)
    y_default = (min_min_data, min(max_min_data, min_min_data + max(120.0, (max_min_data - min_min_data))))

    s1, s2 = st.columns(2)
    with s1:
        idx_range_tc = st.slider(
            "Index UTMB affiché",
            float(idx_min_data), float(idx_max_data),
            (float(idx_default[0]), float(idx_default[1])),
            step=1.0,
            key=ui_key("tc_idx_range_safe")
        )
    with s2:
        y_range_tc = st.slider(
            "Temps cumulé affiché (minutes)",
            float(min_min_data), float(max_min_data if max_min_data > 0 else 60.0),
            (float(y_default[0]), float(y_default[1])),
            step=1.0,
            key=ui_key("tc_min_range_safe")
        )

    chart_df_tc = chart_df_base_tc[
        chart_df_base_tc["index_utmb"].between(idx_range_tc[0], idx_range_tc[1]) &
        chart_df_base_tc["minutes"].between(y_range_tc[0], y_range_tc[1])
    ].copy()

    # 6) Couleurs + axe Y format HH:MM (sans wrap 24 h)
    color_scale_tc = alt.Scale(
        domain=["tranches_inferieures","meme_tranche","tranches_superieures","inconnue"],
        range=["#8be38b","#ffd84d","#ff9999","#cfcfcf"]
    )

    y_enc_tc = alt.Y(
        "minutes:Q",
        title=f"Temps cumulé (HH:MM) — {selected_point_label}",
        axis=alt.Axis(
            labelExpr=(
                "format(floor(datum.value/60), '.0f') + ':' + "
                "((floor(datum.value)%60) < 10 ? '0' : '') + "
                "format(floor(datum.value)%60, '.0f')"
            ),
            tickCount=10
        ),
        scale=alt.Scale(domain=[y_range_tc[0], y_range_tc[1]])
    )

    base_tc = (
        alt.Chart(chart_df_tc)
        .mark_circle(opacity=0.85, size=40)
        .encode(
            x=alt.X("index_utmb:Q", title="Index UTMB",
                    scale=alt.Scale(domain=[idx_range_tc[0], idx_range_tc[1]])),
            y=y_enc_tc,
            color=alt.Color("bucket:N", title="Position vs tranche sélectionnée",
                            scale=color_scale_tc, legend=alt.Legend(orient="bottom", columns=3)),
            tooltip=[
                alt.Tooltip("dossard:N", title="Dossard"),
                alt.Tooltip("sexe:N", title="Sexe") if "sexe" in chart_df_tc.columns else alt.Tooltip("dossard:N", title=""),
                alt.Tooltip("index_utmb:Q", title="Index UTMB", format=".0f"),
                alt.Tooltip("minutes:Q", title="Temps (min)", format=".1f"),
                alt.Tooltip("hms:N", title="Temps (HH:MM:SS)"),
            ],
        )
        .properties(height=420)
        .interactive()
    )

    # 7) Highlight du dossard sélectionné (point rouge 2×)
    highlight_tc = chart_df_tc[chart_df_tc["dossard"].astype(str).str.strip() == str(selected_bib)].copy()
    highlight_layer_tc = (
        alt.Chart(highlight_tc)
        .mark_circle(size=150, color="#ff0000", stroke="black", strokeWidth=1.0, opacity=1.0)
        .encode(
            x="index_utmb:Q",
            y="minutes:Q",
            tooltip=[
                alt.Tooltip("dossard:N", title="Dossard (sélectionné)"),
                alt.Tooltip("index_utmb:Q", title="Index UTMB", format=".0f"),
                alt.Tooltip("hms:N", title="Temps (HH:MM:SS)"),
            ],
        )
    )

    chart_tc = base_tc + highlight_layer_tc
    st.altair_chart(chart_tc, use_container_width=True)

    # 8) Export PNG (mêmes filtres) — safe
    try:
        import altair_saver  # noqa: F401
        export_w, export_h, scale_factor = 400, 450, 6
        chart_to_save = chart_tc.properties(width=export_w, height=export_h)

        buf = io.BytesIO()
        chart_to_save.save(
            buf,
            format="png",
            method="vl-convert",   # nécessite: pip install vl-convert-python
            scale_factor=scale_factor
        )
        buf.seek(0)

        fname = (
            f"scatter_temps_cumule_vs_idx_{str(sel_point).replace(' ','_')}_"
            f"idx{int(idx_range_tc[0])}-{int(idx_range_tc[1])}_"
            f"min{int(y_range_tc[0])}-{int(y_range_tc[1])}.png"
        )

        st.download_button(
            label="📸 Télécharger le graphique (PNG)",
            data=buf,
            file_name=fname,
            mime="image/png",
            key=ui_key("dl_png_tc_vs_idx_safe")
        )
    except Exception as e:
        st.caption("💡 Pour l’export PNG, installe `vl-convert-python` : `pip install vl-convert-python`.")
        st.warning(f"Erreur d'export PNG : {e}")


# ==========================================================
# ⏱️🟣 Nuage — Temps de tronçon (même pas n−1/n−2 que le dossard) vs Index UTMB
# ==========================================================
st.markdown("### ⏱️🟣 Temps de tronçon au point choisi vs Index UTMB (pas identique au dossard)")

import io
import altair as alt

# Sécurités / fallbacks si nécessaires
if 'df_course' not in locals():
    df_course = df.copy()
if 'selected_point_label' not in locals() or 'sel_point' not in locals() or 'sel_km' not in locals():
    pts_fb = df_course.loc[:, [c for c in ["point","km"] if c in df_course.columns]].dropna(subset=["point"])
    if len(pts_fb):
        pts_fb = (pts_fb.sort_values(by=["point","km"], na_position="last")
                        .groupby("point", as_index=False).first()
                        .sort_values(by=["km","point"]))
        sel_point = pts_fb.iloc[0]["point"]
        sel_km = pts_fb.iloc[0].get("km", np.nan)
        selected_point_label = f"{float(sel_km):.1f} km - {sel_point}" if pd.notna(sel_km) else str(sel_point)
    else:
        sel_point, sel_km, selected_point_label = None, np.nan, "—"

# ========= 1) Déterminer le PAS (1 ou 2) AU POINT choisi pour le dossard sélectionné =========
def _sort_bib_df_local(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    if "temps_cumule_td" not in t.columns and "temps_cumule" in t.columns:
        t["temps_cumule_td"] = pd.to_timedelta(t["temps_cumule"], errors="coerce")
    if "km" in t.columns and t["km"].notna().any():
        by = [c for c in ["km","temps_cumule_td"] if c in t.columns]
        t = t.sort_values(by=by, ascending=[True]*len(by))
    else:
        t = t.sort_values(by=["temps_cumule_td"], ascending=[True])
    t = t.reset_index(drop=True)
    t["row_id"] = np.arange(len(t))
    return t

def _find_rows_for_point(df_b, p, k):
    """Retourne la liste des indices row_id où point==p (et km≈k si k non-NaN)."""
    if "point" not in df_b.columns: 
        return []
    if "km" in df_b.columns and pd.notna(k):
        mask = (df_b["point"] == p) & (np.isclose(df_b["km"], k, atol=1e-3))
        idxs = df_b.index[mask].tolist()
        if not idxs:
            idxs = df_b.index[df_b["point"] == p].tolist()
    else:
        idxs = df_b.index[df_b["point"] == p].tolist()
    return idxs

# Dossard sélectionné : déterminer le pas à ce point
if sel_point is None or selected_bib is None:
    st.info("Point ou dossard sélectionné manquant.")
else:
    bib_ref_df = df_course[df_course["dossard"].astype(str) == str(selected_bib)].copy()
    bib_ref_df = _sort_bib_df_local(bib_ref_df)
    idxs_ref = _find_rows_for_point(bib_ref_df, sel_point, sel_km)
    step_used = None

    if not idxs_ref:
        st.warning("Impossible d’identifier la ligne du point choisi pour le dossard sélectionné.")
    else:
        cur_i = max(idxs_ref)
        cur_td = bib_ref_df.loc[cur_i, "temps_cumule_td"]
        prev1 = cur_i - 1
        prev2 = cur_i - 2

        seg1 = pd.NaT
        seg2 = pd.NaT
        if prev1 >= 0:
            td_prev1 = bib_ref_df.loc[prev1, "temps_cumule_td"]
            if pd.notna(cur_td) and pd.notna(td_prev1):
                seg1 = cur_td - td_prev1
        if (pd.isna(seg1) or seg1 <= pd.Timedelta(0)) and prev2 >= 0:
            td_prev2 = bib_ref_df.loc[prev2, "temps_cumule_td"]
            if pd.notna(cur_td) and pd.notna(td_prev2):
                seg2 = cur_td - td_prev2

        if pd.notna(seg1) and seg1 > pd.Timedelta(0):
            step_used = 1
        elif pd.notna(seg2) and seg2 > pd.Timedelta(0):
            step_used = 2

        if step_used is None:
            st.warning("Pas de calcul de tronçon possible pour le dossard sélectionné à ce point (temps manquants).")

    # ========= 2) Appliquer le même pas à TOUS les coureurs et prendre la MÉDIANE si multiples =========
    if step_used is not None:
        sub_all = df_course.copy()
        if "index_utmb_tranche25" not in sub_all.columns and "index_utmb" in sub_all.columns:
            sub_all["index_utmb_tranche25"] = pd.to_numeric(sub_all["index_utmb"], errors="coerce").apply(
                lambda x: int(round(float(x)/25.0)*25) if pd.notna(x) else np.nan
            )
        if "temps_cumule_td" not in sub_all.columns and "temps_cumule" in sub_all.columns:
            sub_all["temps_cumule_td"] = pd.to_timedelta(sub_all["temps_cumule"], errors="coerce")

        keep = [c for c in ["dossard","index_utmb","index_utmb_tranche25","temps_cumule_td","point","km","sexe"] if c in sub_all.columns]
        sub_all = sub_all[keep].copy()
        sub_all["index_utmb"] = pd.to_numeric(sub_all.get("index_utmb"), errors="coerce")
        if "sexe" in sub_all.columns:
            sub_all["sexe"] = sub_all["sexe"].astype(str).str.strip().str.upper()

        rows = []
        for bib, g in sub_all.groupby(sub_all["dossard"].astype(str)):
            tb = _sort_bib_df_local(g.copy())
            idxs = _find_rows_for_point(tb, sel_point, sel_km)
            segs = []
            for cur_i in idxs:
                prev_i = cur_i - step_used
                if prev_i < 0:
                    continue
                td_cur = tb.loc[cur_i, "temps_cumule_td"]
                td_prev = tb.loc[prev_i, "temps_cumule_td"]
                if pd.notna(td_cur) and pd.notna(td_prev):
                    seg = td_cur - td_prev
                    if seg > pd.Timedelta(0):
                        segs.append(seg.total_seconds())
            if len(segs) == 0:
                continue
            seg_med_sec = float(np.median(segs))
            idx_val = pd.to_numeric(tb.get("index_utmb"), errors="coerce").dropna()
            tr_val  = pd.to_numeric(tb.get("index_utmb_tranche25"), errors="coerce").dropna()
            sex_val = tb.get("sexe")
            idx_v = float(idx_val.iloc[-1]) if len(idx_val) else np.nan
            tr_v  = int(tr_val.iloc[-1]) if len(tr_val) else np.nan
            sex_v = str(sex_val.dropna().iloc[-1]).upper() if isinstance(sex_val, pd.Series) and not sex_val.dropna().empty else None

            rows.append({"dossard": bib, "index_utmb": idx_v, "index_utmb_tranche25": tr_v,
                         "sexe": sex_v, "seg_minutes": seg_med_sec/60.0})

        seg_df = pd.DataFrame(rows).dropna(subset=["dossard","index_utmb","index_utmb_tranche25","seg_minutes"])
        if seg_df.empty:
            st.info("Aucun temps de tronçon calculable à ce point pour l’ensemble des coureurs.")
        else:
            utmb_sel_tranche = df_bib.get("index_utmb_tranche25") if 'df_bib' in locals() else None
            tranche_sel_here = int(utmb_sel_tranche.dropna().iloc[0]) if (utmb_sel_tranche is not None and not utmb_sel_tranche.isna().all()) else None

            def bucket_tranche(t):
                if pd.isna(t) or tranche_sel_here is None:
                    return "inconnue"
                t = int(t)
                if t == tranche_sel_here:
                    return "meme_tranche"
                elif t < tranche_sel_here:
                    return "tranches_inferieures"
                else:
                    return "tranches_superieures"

            seg_df["bucket"] = seg_df["index_utmb_tranche25"].apply(bucket_tranche)

            sex_options = ["Tous"]
            if "sexe" in seg_df.columns:
                vals = [s for s in ["F","M"] if s in set(seg_df["sexe"].dropna())]
                if vals:
                    sex_options += vals
            sex_choice = st.radio("Sexe", sex_options, horizontal=True, key=ui_key("seg_scatter_sex"))

            chart_base = seg_df.copy()
            if sex_choice in ("F","M") and "sexe" in chart_base.columns:
                chart_base = chart_base[chart_base["sexe"] == sex_choice]

            idx_min_data = float(np.nanmin(chart_base["index_utmb"])) if len(chart_base) else 0.0
            idx_max_data = float(np.nanmax(chart_base["index_utmb"])) if len(chart_base) else 1000.0
            min_min_data = float(np.nanmin(chart_base["seg_minutes"])) if len(chart_base) else 0.0
            max_min_data = float(np.nanmax(chart_base["seg_minutes"])) if len(chart_base) else 60.0

            if tranche_sel_here is not None:
                idx_default = (max(idx_min_data, tranche_sel_here - 100), min(idx_max_data, tranche_sel_here + 100))
            else:
                idx_default = (idx_min_data, idx_max_data)
            y_default = (min_min_data, min(max_min_data, min_min_data + max(60.0, (max_min_data - min_min_data))))

            c1, c2 = st.columns(2)
            with c1:
                idx_rng = st.slider(
                    "Index UTMB affiché",
                    float(idx_min_data), float(idx_max_data),
                    (float(idx_default[0]), float(idx_default[1])),
                    step=1.0,
                    key=ui_key("seg_idx_rng")
                )
            with c2:
                y_rng = st.slider(
                    "Temps de tronçon affiché (minutes)",
                    float(min_min_data), float(max_min_data if max_min_data > 0 else 60.0),
                    (float(y_default[0]), float(y_default[1])),
                    step=0.5,
                    key=ui_key("seg_min_rng")
                )

            chart_df = chart_base[
                chart_base["index_utmb"].between(idx_rng[0], idx_rng[1]) &
                chart_base["seg_minutes"].between(y_rng[0], y_rng[1])
            ].copy()

            color_scale = alt.Scale(
                domain=["tranches_inferieures","meme_tranche","tranches_superieures","inconnue"],
                range=["#8be38b","#ffd84d","#ff9999","#cfcfcf"]
            )

            # Axe Y HH:MM sans wrap
            y_enc_seg = alt.Y(
                "seg_minutes:Q",
                title=f"Temps de tronçon (HH:MM) — {selected_point_label}  •  pas = n−{step_used}",
                axis=alt.Axis(
                    labelExpr=(
                        "format(floor(datum.value/60), '.0f') + ':' + "
                        "((floor(datum.value)%60) < 10 ? '0' : '') + "
                        "format(floor(datum.value)%60, '.0f')"
                    ),
                    tickCount=10
                ),
                scale=alt.Scale(domain=[y_rng[0], y_rng[1]])
            )

            base = (
                alt.Chart(chart_df)
                .mark_circle(opacity=0.85, size=40)
                .encode(
                    x=alt.X("index_utmb:Q", title="Index UTMB",
                            scale=alt.Scale(domain=[idx_rng[0], idx_rng[1]])),
                    y=y_enc_seg,
                    color=alt.Color("bucket:N", title="Position vs tranche sélectionnée",
                                    scale=color_scale, legend=alt.Legend(orient="bottom", columns=3)),
                    tooltip=[
                        alt.Tooltip("dossard:N", title="Dossard"),
                        alt.Tooltip("sexe:N", title="Sexe") if "sexe" in chart_df.columns else alt.Tooltip("dossard:N", title=""),
                        alt.Tooltip("index_utmb:Q", title="Index UTMB", format=".0f"),
                        alt.Tooltip("seg_minutes:Q", title="Tronçon (min)", format=".1f"),
                    ],
                )
                .properties(height=420)
                .interactive()
            )

            # Highlight du dossard sélectionné (si présent dans la fenêtre)
            hl = chart_df[chart_df["dossard"].astype(str) == str(selected_bib)]
            highlight = (
                alt.Chart(hl)
                .mark_circle(size=150, color="#ff0000", stroke="black", strokeWidth=1.0, opacity=1.0)
                .encode(
                    x="index_utmb:Q",
                    y="seg_minutes:Q",
                    tooltip=[
                        alt.Tooltip("dossard:N", title="Dossard (sélectionné)"),
                        alt.Tooltip("index_utmb:Q", title="Index UTMB", format=".0f"),
                        alt.Tooltip("seg_minutes:Q", title="Tronçon (min)", format=".1f"),
                    ],
                )
            )

            chart = base + highlight
            st.altair_chart(chart, use_container_width=True)

            # ========= 4) Export PNG =========
            try:
                import altair_saver  # noqa: F401
                export_w, export_h, scale_factor = 400, 450, 6
                chart_to_save = chart.properties(width=export_w, height=export_h)
                buf = io.BytesIO()
                chart_to_save.save(
                    buf,
                    format="png",
                    method="vl-convert",   # nécessite: pip install vl-convert-python
                    scale_factor=scale_factor
                )
                buf.seek(0)

                fname = (
                    f"scatter_troncon_vs_idx_{str(sel_point).replace(' ','_')}_"
                    f"pas{int(step_used)}_"
                    f"idx{int(idx_rng[0])}-{int(idx_rng[1])}_"
                    f"min{int(y_rng[0])}-{int(y_rng[1])}.png"
                )

                st.download_button(
                    label="📸 Télécharger le graphique (PNG)",
                    data=buf,
                    file_name=fname,
                    mime="image/png",
                    key=ui_key("dl_png_seg_vs_idx")
                )
            except Exception as e:
                st.caption("💡 Pour l’export PNG, installe `vl-convert-python` : `pip install vl-convert-python`.")



# =======================
# 🟨 Barres — Temps cumulé au point : dossard vs médian des tranches (±N*25)
# =======================
st.markdown("### 🟨 Temps cumulé au point — Dossard vs médian des tranches")

import io
import altair as alt

# Sécurité : si on vient directement ici, retrouve la course/point sélectionné
if 'sel_point' not in locals():
    # fallback minimal : prendre le 1er point de la course du dossard
    df_course = df.copy()
    course_mask = None
    for k in ["course_code", "course"]:
        if k in df_bib.columns:
            val = df_bib[k].dropna().astype(str).str.strip()
            if len(val) > 0 and val.iloc[0] != "":
                course_mask = (df[k].astype(str).str.strip() == val.iloc[0])
                break
    if course_mask is not None and course_mask.any():
        df_course = df[course_mask].copy()
    pts = df_course.loc[:, [c for c in ["point", "km"] if c in df_course.columns]].dropna(subset=["point"])
    pts = (
        pts.sort_values(by=["point", "km"], na_position="last")
           .groupby("point", as_index=False).first()
           .sort_values(by=["km", "point"])
    )
    sel_point = pts.iloc[0]["point"]
    sel_km = pts.iloc[0].get("km", np.nan)
    selected_point_label = f"{(float(sel_km)):.1f} km - {sel_point}" if pd.notna(sel_km) else str(sel_point)

# Tranche sélectionnée (du dossard)
if "index_utmb_tranche25" in df_bib.columns and df_bib["index_utmb_tranche25"].notna().any():
    tranche_sel = int(df_bib["index_utmb_tranche25"].dropna().iloc[0])
else:
    # calcule-la si manquante
    val_idx = pd.to_numeric(df_bib.get("index_utmb", pd.Series(dtype="float")), errors="coerce").dropna()
    tranche_sel = int(round(float(val_idx.iloc[0]) / 25.0) * 25) if len(val_idx) else None

if tranche_sel is None:
    st.warning("Tranche UTMB du dossard sélectionné introuvable.")
else:
    # Slider : nombre de tranches à afficher de part et d'autre (0 à 6)
    n_side = st.slider("Nombre de tranches de part et d'autre (×25)", 0, 6, 2, step=1, key="bars_n_side")

    # --- Temps cumulé du dossard au point choisi (ligne la plus avancée à ce point)
    bib_point = df_bib.copy()
    if "temps_cumule_td" not in bib_point.columns and "temps_cumule" in bib_point.columns:
        bib_point["temps_cumule_td"] = pd.to_timedelta(bib_point["temps_cumule"], errors="coerce")
    if "km" in bib_point.columns and pd.notna(sel_km):
        bib_point = bib_point[(bib_point["point"] == sel_point) & (np.isclose(bib_point["km"], sel_km, atol=1e-3))]
    else:
        bib_point = bib_point[(bib_point["point"] == sel_point)]
    bib_point = bib_point.sort_values(by=["temps_cumule_td"]).tail(1)  # le plus grand temps (ligne la + avancée)
    td_bib = bib_point["temps_cumule_td"].iloc[0] if len(bib_point) else pd.NaT

    # --- Base données pour médianes : même course, même point
    sub = df_course.copy()
    if "index_utmb_tranche25" not in sub.columns and "index_utmb" in sub.columns:
        sub["index_utmb_tranche25"] = pd.to_numeric(sub["index_utmb"], errors="coerce").apply(
            lambda x: int(round(float(x)/25.0)*25) if pd.notna(x) else np.nan
        )
    if "temps_cumule_td" not in sub.columns and "temps_cumule" in sub.columns:
        sub["temps_cumule_td"] = pd.to_timedelta(sub["temps_cumule"], errors="coerce")

    # filtre point
    if "km" in sub.columns and pd.notna(sel_km):
        sub = sub[(sub["point"] == sel_point) & (np.isclose(sub["km"], sel_km, atol=1e-3))]
    else:
        sub = sub[(sub["point"] == sel_point)]

    # Normaliser le sexe si présent
    if "sexe" in sub.columns:
        sub["sexe"] = sub["sexe"].astype(str).str.strip().str.upper()

    # === Filtre SEXE pour le CALCUL des médianes (le dossard reste affiché tel quel)
    sex_options = ["Tous"]
    if "sexe" in sub.columns:
        sex_options = ["Tous"] + [s for s in ["F", "M"] if s in set(sub["sexe"].dropna())]
    sex_choice = st.radio("Sexe pour le calcul des médianes", sex_options, horizontal=True, key="bars_sex_filter")

    sub_med = sub.copy()
    if sex_choice in ("F", "M") and "sexe" in sub_med.columns:
        sub_med = sub_med[sub_med["sexe"] == sex_choice]

    # On garde les temps > 0 (sur l'échantillon des médianes)
    sub_med = sub_med.dropna(subset=["index_utmb_tranche25", "temps_cumule_td"])
    sub_med = sub_med[sub_med["temps_cumule_td"] > pd.Timedelta(0)]

    # Construire liste des tranches à afficher
    tr_prev = [tranche_sel - 25*i for i in range(n_side, 0, -1)]   # ex: [..., tranche_sel-50, tranche_sel-25]
    tr_next = [tranche_sel + 25*i for i in range(1, n_side+1)]     # ex: [tranche_sel+25, tranche_sel+50, ...]
    tr_show = tr_prev + [tranche_sel] + tr_next

    # Calcul médian par tranche (sur l'échantillon filtré par sexe)
    med = (
        sub_med[sub_med["index_utmb_tranche25"].isin(tr_show)]
        .groupby("index_utmb_tranche25", as_index=False)["temps_cumule_td"].median()
        .rename(columns={"temps_cumule_td": "td_median"})
    )

    # Conversion en minutes (pour axe Y linéaire) + format HH:MM:SS pour tooltips
    def td_to_minutes(td):
        return float(td.total_seconds())/60.0 if pd.notna(td) else np.nan

    def td_to_hms(td):
        if pd.isna(td):
            return ""
        tot = int(td.total_seconds())
        h = tot // 3600
        m = (tot % 3600) // 60
        s = tot % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    med["minutes"] = med["td_median"].apply(td_to_minutes)
    med["hms"] = med["td_median"].apply(td_to_hms)

    # Ligne pour le dossard (centrale)
    bib_minutes = td_to_minutes(td_bib)
    bib_hms = td_to_hms(td_bib)

    # DataFrame pour Altair dans l'ordre souhaité :
    # [prev...], "médian tranche_sel", "dossard", [next...]
    rows = []
    # dégradés verts (précédents) et rouges (suivants) — jusqu'à 6 niveaux
    greens = ["#c9f7c9", "#a8eea8", "#8be38b", "#6cd86c", "#4fcc4f", "#33c233"]
    reds   = ["#ffd1d1", "#ffb3b3", "#ff9999", "#ff7f7f", "#ff6666", "#ff4c4c"]
    # borne au nombre demandé
    greens = greens[:n_side][::-1]  # plus proche = vert plus clair (dernière position à côté du centre)
    reds   = reds[:n_side]          # plus proche = rouge plus clair

    # Tranches précédentes (du plus éloigné vers le plus proche)
    for i, t in enumerate(tr_prev):
        r = med[med["index_utmb_tranche25"] == t]
        if not r.empty:
            rows.append({
                "position": f"T-{25*(len(tr_prev)-i)}",
                "tranche": t,
                "label": f"Médiane {t}",
                "minutes": r["minutes"].iloc[0],
                "hms": r["hms"].iloc[0],
                "role": "prev",
                "color": greens[i] if i < len(greens) else "#a8eea8",
            })

    # Médiane de la même tranche
    r0 = med[med["index_utmb_tranche25"] == tranche_sel]
    if not r0.empty:
        rows.append({
            "position": "Médiane T",
            "tranche": tranche_sel,
            "label": f"Médiane {tranche_sel}",
            "minutes": r0["minutes"].iloc[0],
            "hms": r0["hms"].iloc[0],
            "role": "same_med",
            "color": "#ffe680",  # jaune clair pour distinguer de la barre dossard
        })

    # Barre centrale = dossard
    rows.append({
        "position": "Dossard",
        "tranche": tranche_sel,
        "label": f"Dossard {selected_bib}",
        "minutes": bib_minutes,
        "hms": bib_hms,
        "role": "bib",
        "color": "#1e88e5",  # jaune
    })

    # Tranches suivantes (du plus proche vers le plus éloigné)
    for i, t in enumerate(tr_next):
        r = med[med["index_utmb_tranche25"] == t]
        if not r.empty:
            rows.append({
                "position": f"T+{25*(i+1)}",
                "tranche": t,
                "label": f"Médiane {t}",
                "minutes": r["minutes"].iloc[0],
                "hms": r["hms"].iloc[0],
                "role": "next",
                "color": reds[i] if i < len(reds) else "#ffb3b3",
            })

    bars_df = pd.DataFrame(rows)
    if bars_df.empty:
        st.info("Pas assez de données pour construire les médianes autour de la tranche sélectionnée.")
    else:
        # Choisir l'ordre d'affichage des catégories (tel que construit ci-dessus)
        order = bars_df["position"].tolist()

        # Détermination des bornes Y par défaut (minutes) avec marge
        y_min = 0.0
        y_max = float(np.nanmax(bars_df["minutes"])) if bars_df["minutes"].notna().any() else 60.0
        y_max = y_max * 1.1

        # Slider pour borne supérieure (affinage visuel)
        y_max = st.slider(
            "Borne haute du temps cumulé (minutes)",
            float(max(10.0, y_max/10)),  # min slider
            float(max(30.0, y_max*2)),   # max slider
            float(y_max),
            step=5.0,
            key="bars_ymax_minutes"
        )

        # Graphique Altair (barres colorées personnalisées)
        chart_bars = (
            alt.Chart(bars_df)
            .mark_bar()
            .encode(
                x=alt.X("position:N", title=f"Point : {selected_point_label}", sort=order),
                y=alt.Y("minutes:Q", title="Temps cumulé (minutes)",
                        scale=alt.Scale(domain=[y_min, y_max])),
                color=alt.Color("color:N", title=None, scale=None),
                tooltip=[
                    alt.Tooltip("label:N", title="Série"),
                    alt.Tooltip("tranche:Q", title="Tranche (25)"),
                    alt.Tooltip("hms:N", title="Temps (HH:MM:SS)"),
                ],
            )
            .properties(height=420)
        )

        # Contour noir sur la barre du dossard pour bien la distinguer
        outline = (
            alt.Chart(bars_df[bars_df["role"] == "bib"])
            .mark_bar(stroke="black", strokeWidth=2, fillOpacity=0)
            .encode(
                x=alt.X("position:N", sort=order),
                y=alt.Y("minutes:Q", scale=alt.Scale(domain=[y_min, y_max]))
            )
        )

        st.altair_chart(chart_bars + outline, use_container_width=True)

        # Export PNG
        try:
            import altair_saver  # noqa: F401
            export_w = 400
            export_h = 450
            scale_factor = 6

            chart_to_save = (chart_bars + outline).properties(width=export_w, height=export_h)

            buf = io.BytesIO()
            chart_to_save.save(
                buf,
                format="png",
                method="vl-convert",   # nécessite: pip install vl-convert-python
                scale_factor=scale_factor
            )
            buf.seek(0)

            fname = (
                f"bars_temps_cumule_{str(sel_point).replace(' ','_')}_"
                f"T{tranche_sel}_N{n_side}_ymax{int(y_max)}"
                f"{'_'+sex_choice if sex_choice in ('F','M') else ''}.png"
            )

            st.download_button(
                label="📸 Télécharger le PNG des barres",
                data=buf,
                file_name=fname,
                mime="image/png",
                key="dl_png_bars_utmb"
            )
        except Exception as e:
            st.warning(
                "⚠️ Export PNG indisponible. Installe d'abord :\n"
                "```bash\npip install vl-convert-python\n```\n"
                f"Erreur : {e}"
            )

# =======================
# ⏱️ Tableau des temps de tronçon du dossard sélectionné (différence ARRIÈRE)
# =======================
st.subheader("Temps sur chaque tronçon (dossard sélectionné)")

# Assure-toi d'avoir la colonne temps_cumule_td (Timedelta)
df_bibt = df_bib.copy()
if "temps_cumule_td" not in df_bibt.columns and "temps_cumule" in df_bibt.columns:
    df_bibt["temps_cumule_td"] = pd.to_timedelta(df_bibt["temps_cumule"], errors="coerce")

# Utilitaire robuste pour format HH:MM:SS
def td_to_hms(td):
    if pd.isna(td):
        return ""
    total = int(pd.Timedelta(td).total_seconds())
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

# On garde uniquement les colonnes utiles et on trie par km (puis temps cumulé pour départager)
cols_needed = [c for c in ["km", "point", "temps_cumule_td"] if c in df_bibt.columns]
if len(cols_needed) < 2 or "temps_cumule_td" not in cols_needed:
    st.info("Données insuffisantes pour calculer les tronçons (colonnes requises : km, point, temps_cumule).")
else:
    tab = df_bibt[cols_needed].copy()

    # tri: si km dispo -> par km croissant, sinon par temps cumulé croissant
    if "km" in tab.columns and tab["km"].notna().any():
        by_cols = [c for c in ["km", "temps_cumule_td"] if c in tab.columns]
        tab = tab.sort_values(by=by_cols, ascending=[True]*len(by_cols))
    else:
        tab = tab.sort_values(by=["temps_cumule_td"], ascending=[True])

    # === Calcul du temps de tronçon (logique ARRIÈRE)
    cur = tab["temps_cumule_td"]     # t(n)
    prev1 = cur.shift(1)             # t(n-1)
    prev2 = cur.shift(2)             # t(n-2)

    has_cur = cur.notna()

    # priorité n-1, sinon n-2
    use_prev1 = has_cur & prev1.notna()
    use_prev2 = has_cur & (~use_prev1) & prev2.notna()

    seg_td = pd.Series(pd.NaT, index=tab.index, dtype="timedelta64[ns]")
    seg_td[use_prev1] = (cur - prev1)[use_prev1]
    seg_td[use_prev2] = (cur - prev2)[use_prev2]

    # Garder uniquement segments strictement positifs
    seg_td = seg_td.where(seg_td > pd.Timedelta(0))

    # Construire le tableau final
    out = tab.copy()
    out["temps_troncon_td"] = seg_td
    # (optionnel) voir quel pas a été utilisé: 1 si n-1, 2 si n-2
    out["pas (+1/+2)"] = np.where(use_prev1, 1, np.where(use_prev2, 2, np.nan))

    # On n'affiche pas la ligne n si t(n) manquant OU si on n'a pas pu calculer le tronçon
    out = out.dropna(subset=["temps_cumule_td", "temps_troncon_td"])

    # Format HH:MM:SS
    out["temps_troncon"] = out["temps_troncon_td"].apply(td_to_hms)

    # Colonnes d'affichage (tu peux retirer la colonne du pas si tu ne la veux pas)
    show_cols = [c for c in ["km", "point", "pas (+1/+2)", "temps_troncon"] if c in out.columns]
    out = out[show_cols]

    if out.empty:
        st.info("Aucun tronçon calculable (temps cumulés manquants en n ou en n-1/n-2).")
    else:
        st.dataframe(out, use_container_width=True)


# =======================
# ⏱️ Tronçons — 5 devant / 5 derrière (pas identique +1/+2 que le dossard sélectionné — LOGIQUE ARRIÈRE)
# =======================
st.subheader("Temps de tronçon — 5 devant / 5 derrière (même méthode que le dossard, en arrière)")

course = df_course.copy() if 'df_course' in locals() else df.copy()
if "temps_cumule_td" not in course.columns and "temps_cumule" in course.columns:
    course["temps_cumule_td"] = pd.to_timedelta(course["temps_cumule"], errors="coerce")

# === Filtre H/F (H accepte aussi M) ===
if "sexe" in course.columns:
    st.write("")  # petit espace
    sex_choice = st.radio("Filtrer par sexe", ["Tous", "H", "F"], horizontal=True, key="troncons_sex_filter")

    if sex_choice != "Tous":
        def norm_sex(x):
            s = str(x).strip().upper()
            if s.startswith("F"):
                return "F"
            if s.startswith("H") or s.startswith("M"):
                return "H"
            return None
        course["_sex_norm"] = course["sexe"].map(norm_sex)
        course = course[course["_sex_norm"] == sex_choice]
        if course.empty:
            st.info("Aucun concurrent après filtrage par sexe.")
            st.stop()

# Tri robuste pour un sous-dataframe dossard
def sort_bib_df(t):
    t = t.copy()
    if "km" in t.columns and t["km"].notna().any():
        by = [c for c in ["km","temps_cumule_td"] if c in t.columns]
        t = t.sort_values(by=by, ascending=[True]*len(by))
    else:
        t = t.sort_values(by=["temps_cumule_td"], ascending=[True])
    return t

# === 1) Classement final de la course
work = course.copy()
km_max_course = float(work["km"].dropna().max()) if "km" in work.columns and work["km"].notna().any() else None

grp = []
for bib, g in work.groupby(work["dossard"].astype(str)):
    g = g[g["temps_cumule_td"].notna()].copy()
    if g.empty:
        continue
    g = sort_bib_df(g)
    last = g.iloc[-1]
    grp.append({
        "dossard": bib,
        "final_km": float(last["km"]) if "km" in g.columns and pd.notna(last.get("km", np.nan)) else np.nan,
        "final_td": last["temps_cumule_td"],
    })
rank_df = pd.DataFrame(grp)

if rank_df.empty:
    st.info("Impossible de déterminer les arrivants (pas de temps cumulés valides).")
else:
    if km_max_course is not None and not np.isnan(km_max_course):
        tol = 1e-3
        rank_df["is_finisher"] = np.isfinite(rank_df["final_km"]) & (np.abs(rank_df["final_km"] - km_max_course) <= tol)
        rank_df = rank_df.sort_values(by=["is_finisher","final_km","final_td"], ascending=[False, False, True]).reset_index(drop=True)
    else:
        rank_df["is_finisher"] = True
        rank_df = rank_df.sort_values(by=["final_td"], ascending=[True]).reset_index(drop=True)

    # Map dossard -> rang (1-based)
    rank_df["rank"] = np.arange(1, len(rank_df)+1)
    rank_map = dict(zip(rank_df["dossard"].astype(str), rank_df["rank"]))

    # Position du dossard sélectionné
    sel_row = rank_df[rank_df["dossard"].astype(str) == str(selected_bib)]
    if sel_row.empty:
        st.info("Dossard sélectionné introuvable dans le classement général – affichage impossible des voisins.")
    else:
        pos = sel_row.index[0]
        idx_before = [i for i in range(pos-5, pos) if i >= 0]                  # 5 devant (meilleurs rangs)
        idx_after  = [i for i in range(pos+1, pos+6) if i < len(rank_df)]      # 5 derrière

        bibs_before = rank_df.iloc[idx_before]["dossard"].astype(str).tolist()
        bibs_after  = rank_df.iloc[idx_after]["dossard"].astype(str).tolist()
        ordered_bibs = bibs_before + [str(selected_bib)] + bibs_after

        # === 2) Mapping dossard -> "RANK - NOM Prénom" (fallback dossard)
        def first_nonnull(series):
            s = series.dropna()
            return s.iloc[0] if not s.empty else pd.NA

        names_map = {}
        for bib, g in course.groupby(course["dossard"].astype(str)):
            nom = first_nonnull(g.get("nom", pd.Series(dtype="string")))
            prenom = first_nonnull(g.get("prenom", pd.Series(dtype="string")))
            nom = "" if pd.isna(nom) else str(nom).strip().upper()
            prenom = "" if pd.isna(prenom) else str(prenom).strip().title()
            full = f"{nom} {prenom}".strip()
            r = rank_map.get(str(bib))
            label = f"{r} - {full}" if r is not None and full else (f"{r} - {bib}" if r is not None else (full if full else str(bib)))
            names_map[str(bib)] = label

        # === 3) Construire la base (dossard sélectionné) + pas (1 ou 2) par ligne — LOGIQUE ARRIÈRE
        sel_df = course[course["dossard"].astype(str) == str(selected_bib)].copy()
        if "temps_cumule_td" not in sel_df.columns and "temps_cumule" in sel_df.columns:
            sel_df["temps_cumule_td"] = pd.to_timedelta(sel_df["temps_cumule"], errors="coerce")
        sel_df = sel_df[["km","point","temps_cumule_td"]].copy()
        sel_df = sort_bib_df(sel_df)

        cur  = sel_df["temps_cumule_td"]      # t(n)
        prev1 = cur.shift(1)                   # t(n-1)
        prev2 = cur.shift(2)                   # t(n-2)

        has_cur = cur.notna()
        use_prev1 = has_cur & prev1.notna()
        use_prev2 = has_cur & (~use_prev1) & prev2.notna()

        seg_td_sel = pd.Series(pd.NaT, index=sel_df.index, dtype="timedelta64[ns]")
        seg_td_sel[use_prev1] = (cur - prev1)[use_prev1]
        seg_td_sel[use_prev2] = (cur - prev2)[use_prev2]
        seg_td_sel = seg_td_sel.where(seg_td_sel > pd.Timedelta(0))

        # Table des segments valides du sélectionné (seulement si le calcul est possible)
        base_sel = sel_df.copy()
        base_sel["step_used"] = np.where(use_prev1, 1, np.where(use_prev2, 2, np.nan))
        base_sel["temps_troncon_td"] = seg_td_sel
        base_sel = base_sel.dropna(subset=["temps_cumule_td","temps_troncon_td","step_used"])

        if base_sel.empty:
            st.info("Aucun tronçon calculable pour le dossard sélectionné avec la règle arrière (n−1 sinon n−2).")
        else:
            # Pour affichage
            def td_to_hms(td):
                if pd.isna(td):
                    return ""
                total = int(pd.Timedelta(td).total_seconds())
                h = total // 3600
                m = (total % 3600) // 60
                s = total % 60
                return f"{h:02d}:{m:02d}:{s:02d}"

            base = base_sel[["km","point","step_used","temps_troncon_td"]].copy()
            sel_label = names_map.get(str(selected_bib), str(selected_bib))
            base[sel_label] = base["temps_troncon_td"].apply(td_to_hms)
            base[f"{sel_label}__sec"] = base["temps_troncon_td"].apply(lambda x: pd.Timedelta(x).total_seconds())

            # === 4) Appliquer EXACTEMENT le même pas (1/2) aux autres dossards, au même point — LOGIQUE ARRIÈRE
            def troncon_for_bib_same_method_backward(bib):
                g = course[course["dossard"].astype(str) == str(bib)][["km","point","temps_cumule_td"]].copy()
                if g.empty:
                    return None, names_map.get(str(bib), str(bib))
                g = sort_bib_df(g)
                g["row_id"] = np.arange(len(g))  # index stable après tri

                # indexer par (point, km) si possible; sinon par point seul
                use_km = ("km" in g.columns) and g["km"].notna().any()
                idx_map = {}
                if use_km:
                    for _, row in g.iterrows():
                        p = str(row["point"])
                        k = float(row["km"]) if pd.notna(row["km"]) else np.nan
                        idx_map[(p, k)] = int(row["row_id"])
                else:
                    for _, row in g.iterrows():
                        p = str(row["point"])
                        idx_map[p] = int(row["row_id"])

                rows = []
                for _, r in base[["km","point","step_used"]].iterrows():
                    p = str(r["point"])
                    k = float(r["km"]) if pd.notna(r["km"]) else np.nan
                    step = int(r["step_used"])

                    # trouver la ligne courante n (point/km) dans g
                    if use_km and not np.isnan(k):
                        cur_i = idx_map.get((p, k))
                        if cur_i is None:
                            cur_i = idx_map.get(p)  # fallback: point seul
                    else:
                        cur_i = idx_map.get(p)

                    if cur_i is None:
                        rows.append(np.nan)
                        continue

                    prev_i = cur_i - step
                    if prev_i < 0:
                        rows.append(np.nan)
                        continue

                    cur_td  = g.loc[g["row_id"] == cur_i, "temps_cumule_td"].values[0]
                    prev_td = g.loc[g["row_id"] == prev_i, "temps_cumule_td"].values[0]
                    if pd.isna(cur_td) or pd.isna(prev_td):
                        rows.append(np.nan)
                        continue

                    seg = cur_td - prev_td
                    try:
                        sec = seg.total_seconds()
                    except AttributeError:
                        sec = pd.Timedelta(seg).total_seconds()
                    rows.append(sec if sec > 0 else np.nan)

                label = names_map.get(str(bib), str(bib))
                out = base[["km","point"]].copy()
                out[f"{label}__sec"] = rows
                out[label] = [td_to_hms(pd.to_timedelta(s, unit="s")) if pd.notna(s) else None for s in rows]
                return out, label

            # Ajoute 5 devant (ordre de classement croissant)
            for bib in bibs_before:
                tb, label = troncon_for_bib_same_method_backward(bib)
                if tb is not None:
                    base = base.merge(tb, on=["km","point"], how="left")

            # Ajoute 5 derrière
            for bib in bibs_after:
                tb, label = troncon_for_bib_same_method_backward(bib)
                if tb is not None:
                    base = base.merge(tb, on=["km","point"], how="left")

            # Colonnes ordonnées selon classement
            time_cols = [names_map.get(b, b) for b in ordered_bibs]
            sec_cols  = [f"{c}__sec" for c in time_cols]
            keep_cols = ["km","point"] + time_cols
            for c in sec_cols:
                if c not in base.columns:
                    base[c] = np.nan

            # Tri final par km/point
            if "km" in base.columns and base["km"].notna().any():
                base = base.sort_values(by=["km","point"], ascending=[True, True])

            # Réorganise pour le styler (temps + secondes en fin si besoin)
            base = base[["km","point","step_used"] + time_cols + sec_cols]

            # === 5) Style dégradé vert->rouge par ligne (ignore None)
            secs_df = base[[f"{c}__sec" for c in time_cols]].copy()

            def row_gradient(row):
                cols = row.index.tolist()
                styles = [""] * len(cols)
                rsecs = secs_df.loc[row.name].values.astype(float)
                valid = ~np.isnan(rsecs)
                if valid.sum() <= 1:
                    return pd.Series(styles, index=row.index)

                v = rsecs[valid]
                vmin, vmax = float(np.min(v)), float(np.max(v))
                denom = (vmax - vmin) if (vmax - vmin) > 0 else 1.0

                pos = {c: cols.index(c) for c in time_cols if c in cols}

                ghex, rhex = "#c9f7c9", "#ff6b6b"
                def hex2rgb(h): return tuple(int(h[j:j+2], 16) for j in (1,3,5))
                def rgb2hex(rgb): return '#%02x%02x%02x' % rgb
                gr, rr = hex2rgb(ghex), hex2rgb(rhex)

                for i, col in enumerate(time_cols):
                    if col not in pos:
                        continue
                    val = rsecs[i]
                    if np.isnan(val):
                        continue
                    t = (val - vmin) / denom
                    rgb = tuple(int(gr[k] + t*(rr[k]-gr[k])) for k in range(3))
                    styles[pos[col]] = f"background-color: {rgb2hex(rgb)}"

                return pd.Series(styles, index=row.index)

            show_df = base[["km","point","step_used"] + time_cols].copy()
            show_df = show_df.rename(columns={"step_used": "pas (+1/+2)"})
            styled = show_df.style.apply(row_gradient, axis=1)
            st.dataframe(styled, use_container_width=True)

            # === 6) Export PNG du tableau (conserver couleurs, garder km/point, masquer 'pas (+1/+2)', ajuster entête)
            import io
            import matplotlib.pyplot as plt

            def hex_to_rgb_tuple(h):
                h = h.lstrip("#")
                return (int(h[0:2], 16)/255.0, int(h[2:4], 16)/255.0, int(h[4:6], 16)/255.0)

            def compute_cell_colors_for_png(show_df, secs_df, time_cols):
                ghex, rhex = "#c9f7c9", "#ff6b6b"
                gr, rr = hex_to_rgb_tuple(ghex), hex_to_rgb_tuple(rhex)
                colors = []
                for i in range(len(show_df)):
                    row_colors = [(1,1,1)] * len(show_df.columns)  # blanc par défaut
                    rsecs = secs_df.iloc[i].values.astype(float)    # aligne avec time_cols
                    valid = ~np.isnan(rsecs)
                    if valid.sum() > 1:
                        v = rsecs[valid]
                        vmin, vmax = float(np.min(v)), float(np.max(v))
                        denom = (vmax - vmin) if (vmax - vmin) > 0 else 1.0
                        for j, col in enumerate(time_cols):
                            if col in show_df.columns:
                                idx_in_show = show_df.columns.get_loc(col)
                                val = rsecs[j]
                                if not np.isnan(val):
                                    t = (val - vmin) / denom
                                    rgb = (
                                        gr[0] + t*(rr[0]-gr[0]),
                                        gr[1] + t*(rr[1]-gr[1]),
                                        gr[2] + t*(rr[2]-gr[2]),
                                    )
                                    row_colors[idx_in_show] = rgb
                    colors.append(row_colors)
                return colors

            # 1) Construire DF pour export PNG : garder km & point, masquer 'pas (+1/+2)' seulement
            df_png = show_df.copy()
            df_png = df_png.drop(columns=["pas (+1/+2)"], errors="ignore")

            # 2) Mettre 'km' puis 'point' en tête si présents
            lead = [c for c in ["km", "point"] if c in df_png.columns]
            df_png = df_png[lead + [c for c in df_png.columns if c not in lead]]

            # 3) Couleurs alignées
            colors_full = compute_cell_colors_for_png(show_df, secs_df, time_cols)
            keep_idxs = [show_df.columns.get_loc(c) for c in df_png.columns]
            colors_png = [[row_colors[k] for k in keep_idxs] for row_colors in colors_full]

            # 4) Générer l'image
            ncols = len(df_png.columns)
            nrows = len(df_png)
            fig_w = max(8, 0.5 * ncols)
            fig_h = max(4, 0.35 * (nrows + 1))
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            ax.axis("off")

            cell_text = df_png.fillna("").values.tolist()
            col_labels = df_png.columns.tolist()

            table = ax.table(
                cellText=cell_text,
                colLabels=col_labels,
                cellColours=colors_png,
                loc="center"
            )

            table.auto_set_font_size(False)
            table.set_fontsize(8)  # 🔠 police plus petite
            table.scale(1, 1.3)    # espacement vertical global

            # 5) Ajuster la hauteur de la première ligne (entête)
            header_height_factor = 1.5  # 🔝 augmenter si tu veux encore plus d’espace
            for j in range(ncols):
                cell = table[0, j]
                if cell is not None:
                    cell.set_height(cell.get_height() * header_height_factor)
                    cell.set_text_props(weight='bold')  # texte entête en gras

            # 6) Élargir un peu 'km' et 'point' si présents
            for name in ["km", "point"]:
                if name in df_png.columns:
                    col_idx = df_png.columns.get_loc(name)
                    for i in range(nrows + 1):
                        cell = table[i, col_idx]
                        if cell is not None:
                            cell.set_width(0.35 if name == "point" else 0.25)

            # 🔧 Ajustement optionnel de la largeur des colonnes
            # (tu peux modifier les valeurs selon ton rendu)
            for j, col_name in enumerate(df_png.columns):
                for i in range(nrows + 1):  # +1 pour inclure la ligne d'en-tête
                    cell = table[i, j]
                    if cell is not None:
                        if col_name == "km":
                            cell.set_width(0.16)      # largeur de la colonne km
                        elif col_name == "point":
                            cell.set_width(0.35)     # largeur de la colonne point
                        else:
                            cell.set_width(0.22)      # largeur de chaque coureur


            fig.tight_layout(pad=0.5)
            png_buf = io.BytesIO()
            plt.savefig(png_buf, format="png", dpi=220, bbox_inches="tight")
            plt.close(fig)
            png_buf.seek(0)

            st.download_button(
                "📥 Télécharger ce tableau (PNG)",
                data=png_buf.getvalue(),
                file_name="troncons_5_devant_5_derriere.png",
                mime="image/png"
            )




# =======================
# 🟩 Tronçons — Médiane des tranches d'index (T-3..T+3) avec le même pas que le dossard — LOGIQUE ARRIÈRE
# =======================
st.subheader("Temps de tronçon — médian par tranches d'index (T-3 à T+3)")

# Sécurité: s'assurer que 'temps_cumule_td' et 'index_utmb_tranche25' existent
course_med = df_course.copy() if 'df_course' in locals() else df.copy()
if "temps_cumule_td" not in course_med.columns and "temps_cumule" in course_med.columns:
    course_med["temps_cumule_td"] = pd.to_timedelta(course_med["temps_cumule"], errors="coerce")
if "index_utmb_tranche25" not in course_med.columns and "index_utmb" in course_med.columns:
    course_med["index_utmb_tranche25"] = pd.to_numeric(course_med["index_utmb"], errors="coerce").apply(
        lambda x: int(round(float(x)/25.0)*25) if pd.notna(x) else np.nan
    )

# Utilitaire: tri "robuste" par km puis temps cumulé
def _sort_by_km_then_time(t):
    t = t.copy()
    if "km" in t.columns and t["km"].notna().any():
        by = [c for c in ["km","temps_cumule_td"] if c in t.columns]
        t = t.sort_values(by=by, ascending=[True]*len(by))
    else:
        t = t.sort_values(by=["temps_cumule_td"], ascending=[True])
    return t

# 1) Dossard sélectionné : calcule le pas ARRIÈRE (n-1 sinon n-2) ligne par ligne
sel_bib_df = df_bib[["km","point","temps_cumule_td"]].copy()
sel_bib_df = _sort_by_km_then_time(sel_bib_df)
cur = sel_bib_df["temps_cumule_td"]     # t(n)
p1  = cur.shift(1)                      # t(n-1)
p2  = cur.shift(2)                      # t(n-2)

use1 = cur.notna() & p1.notna()         # priorité n-1
use2 = cur.notna() & (~use1) & p2.notna()  # sinon n-2

seg_td_sel = pd.Series(pd.NaT, index=sel_bib_df.index, dtype="timedelta64[ns]")
seg_td_sel[use1] = (cur - p1)[use1]
seg_td_sel[use2] = (cur - p2)[use2]
seg_td_sel = seg_td_sel.where(seg_td_sel > pd.Timedelta(0))

base_sel = sel_bib_df.copy()
base_sel["step_used"] = np.where(use1, 1, np.where(use2, 2, np.nan))
base_sel["temps_troncon_sel_td"] = seg_td_sel
# On ne garde QUE les lignes où on sait calculer le tronçon pour le dossard
base_sel = base_sel.dropna(subset=["temps_cumule_td","temps_troncon_sel_td","step_used"])

if base_sel.empty:
    st.info("Aucun tronçon calculable pour le dossard sélectionné (règle arrière n−1 sinon n−2).")
else:
    # 2) Médians des temps cumulés par tranche d'index au niveau (point,km)
    sub = course_med.dropna(subset=["index_utmb_tranche25","temps_cumule_td"]).copy()

    # On calcule la médiane au couple (tranche, point, km)
    med = (
        sub.groupby(["index_utmb_tranche25","point","km"], as_index=False)["temps_cumule_td"]
           .median()
           .rename(columns={"temps_cumule_td":"td_med"})
    )

    # 3) Liste des tranches à afficher : T-75..T+75 par pas 25 (borné par présence des données)
    t0 = int(tranche_sel)
    tr_prev = [t0 - 25*i for i in range(3, 0, -1)]  # T-75, T-50, T-25
    tr_next = [t0 + 25*i for i in range(1, 4)]      # T+25, T+50, T+75
    tr_list = [t for t in tr_prev if (med["index_utmb_tranche25"]==t).any()] + \
              ([t0] if (med["index_utmb_tranche25"]==t0).any() else []) + \
              [t for t in tr_next if (med["index_utmb_tranche25"]==t).any()]

    if len(tr_list) == 0:
        st.info("Aucune tranche d'index avec suffisamment de données pour calculer des médianes.")
    else:
        # 4) Pour chaque tranche, créer l'ordre de ses points et un mapping (point,km) -> position
        orders = {}
        med_sorted = {}
        for tval, g in med.groupby("index_utmb_tranche25"):
            g2 = _sort_by_km_then_time(g.rename(columns={"td_med":"_dummy"}))  # tri par km puis temps
            g = g.merge(g2[["point","km"]], on=["point","km"], how="left")     # on garde l'ordre
            g = g.sort_values(by=["km","point"], na_position="last")
            g = g.reset_index(drop=True)
            g["pos"] = np.arange(len(g))
            orders[tval] = g[["point","km","pos"]].copy()
            med_sorted[tval] = g[["point","km","td_med","pos"]].copy()

        # 5) Appliquer EXACTEMENT le même pas ARRIÈRE que le dossard sur les médianes
        #    Pour chaque ligne de base_sel (point/km + step), seg = td_med[pos] - td_med[pos - step]
        def tranche_troncons(tval):
            if tval not in orders:
                return None
            ord_t = orders[tval]
            med_t = med_sorted[tval]
            # mapping clé -> pos (prio (point,km) si km dispo; sinon point seul)
            use_km = ord_t["km"].notna().any()
            if use_km:
                pos_map = {(str(p), float(k) if pd.notna(k) else np.nan): int(pos)
                           for p,k,pos in zip(ord_t["point"].astype(str), ord_t["km"], ord_t["pos"])}
            else:
                pos_map = {str(p): int(pos) for p,pos in zip(ord_t["point"].astype(str), ord_t["pos"])}

            rows_sec = []
            for _, r in base_sel[["point","km","step_used"]].iterrows():
                p = str(r["point"])
                k = float(r["km"]) if pd.notna(r["km"]) else np.nan
                step = int(r["step_used"])

                if use_km and not np.isnan(k):
                    cur_pos = pos_map.get((p, k))
                    if cur_pos is None:
                        cur_pos = pos_map.get(p)
                else:
                    cur_pos = pos_map.get(p)

                if cur_pos is None:
                    rows_sec.append(np.nan)
                    continue

                prev_pos = cur_pos - step
                cur_row  = med_t[med_t["pos"] == cur_pos]
                prev_row = med_t[med_t["pos"] == prev_pos]
                if cur_row.empty or prev_row.empty:
                    rows_sec.append(np.nan)
                    continue

                cur_td  = cur_row["td_med"].iloc[0]
                prev_td = prev_row["td_med"].iloc[0]
                if pd.isna(cur_td) or pd.isna(prev_td):
                    rows_sec.append(np.nan)
                    continue

                seg = cur_td - prev_td
                try:
                    val_sec = seg.total_seconds()
                except AttributeError:
                    val_sec = pd.Timedelta(seg).total_seconds()
                rows_sec.append(val_sec if val_sec > 0 else np.nan)

            # format HH:MM:SS
            def fmt_hms(sec):
                if pd.isna(sec):
                    return None
                sec = int(sec)
                h = sec // 3600
                m = (sec % 3600) // 60
                s = sec % 60
                return f"{h:02d}:{m:02d}:{s:02d}"

            col_sec = f"{tval}__sec"
            col_txt = f"{tval}"
            out = base_sel[["km","point"]].copy()
            out[col_sec] = rows_sec
            out[col_txt] = [fmt_hms(x) for x in rows_sec]
            return out

        # 6) Construire la table finale
        comp = base_sel[["km","point","step_used"]].copy()
        comp = comp.rename(columns={"step_used":"pas (+1/+2)"})

        for tval in tr_list:
            tb = tranche_troncons(tval)
            if tb is not None:
                comp = comp.merge(tb, on=["km","point"], how="left")

        # On ordonne les colonnes avec l'ordre T-3..T+3
        label_cols = [str(t) for t in tr_list]
        sec_cols   = [f"{c}__sec" for c in label_cols]
        # Certaines tranches peuvent manquer -> on garde celles qui existent
        label_cols = [c for c in label_cols if c in comp.columns]
        sec_cols   = [c for c in sec_cols if c in comp.columns]

        # Tri visuel par km / point
        if "km" in comp.columns and comp["km"].notna().any():
            comp = comp.sort_values(by=["km","point"], ascending=[True, True])

        # === Ajouter la colonne du dossard sélectionné (au bon emplacement) ===

        # 1) Libellé "<index> - NOM Prénom"
        idx_vals = pd.to_numeric(df_bib.get("index_utmb", pd.Series(dtype="float")), errors="coerce").dropna()
        if len(idx_vals):
            idx_disp = int(round(float(idx_vals.iloc[0])))
        else:
            idx_disp = None

        def first_nonnull(series):
            s = series.dropna()
            return s.iloc[0] if not s.empty else pd.NA

        _nom = first_nonnull(df_bib.get("nom", pd.Series(dtype="string")))
        _prenom = first_nonnull(df_bib.get("prenom", pd.Series(dtype="string")))
        _nom = "" if pd.isna(_nom) else str(_nom).strip().upper()
        _prenom = "" if pd.isna(_prenom) else str(_prenom).strip().title()

        if idx_disp is not None and (_nom or _prenom):
            bib_label = f"{idx_disp} - {_nom} {_prenom}".strip()
        elif idx_disp is not None:
            bib_label = f"{idx_disp} - {selected_bib}"
        else:
            bib_label = f"{selected_bib}"

        # 2) Construire la colonne du dossard à partir de base_sel (déjà calculé avec step_used)
        def _fmt_hms(td):
            if pd.isna(td):
                return None
            total = int(pd.Timedelta(td).total_seconds())
            h = total // 3600
            m = (total % 3600) // 60
            s = total % 60
            return f"{h:02d}:{m:02d}:{s:02d}"

        bib_sec = base_sel["temps_troncon_sel_td"].apply(lambda x: np.nan if pd.isna(x) else pd.Timedelta(x).total_seconds())
        bib_txt = base_sel["temps_troncon_sel_td"].apply(_fmt_hms)

        bib_df = base_sel[["km","point"]].copy()
        bib_df[f"{bib_label}__sec"] = bib_sec.values
        bib_df[bib_label] = bib_txt.values

        # Fusionner dans comp
        comp = comp.merge(bib_df, on=["km","point"], how="left")

        # 3) Ordonner les colonnes : T-3 .. T-1, T, <BIB>, T+1 .. T+3
        def _has_col(c): return c in comp.columns
        left_cols  = [str(t) for t in tr_prev if _has_col(str(t))]
        center_col = [str(tranche_sel)] if _has_col(str(tranche_sel)) else []
        right_cols = [str(t) for t in tr_next if _has_col(str(t))]

        ordered_labels = left_cols + center_col + [bib_label] + right_cols
        ordered_labels = [c for c in ordered_labels if c in comp.columns]

        # 4) Colonnes d’affichage finales
        show_cols = ["km","point","pas (+1/+2)"] + ordered_labels
        st.dataframe(comp[show_cols], use_container_width=True)


# =======================
# 📊 Barres des temps de tronçon au point sélectionné (tranches + dossard)
# =======================
st.markdown("### 📊 Temps de tronçon par tranche (au point sélectionné)")

import altair as alt
import io

# --- Reconstruire la liste ordonnée des colonnes visibles : T-.., T, BIB, T+..
def _has_col(c): return c in comp.columns
left_cols  = [str(t) for t in tr_prev if _has_col(str(t))]
center_col = [str(tranche_sel)] if _has_col(str(tranche_sel)) else []
right_cols = [str(t) for t in tr_next if _has_col(str(t))]
ordered_labels = left_cols + center_col + [bib_label] + right_cols
ordered_labels = [c for c in ordered_labels if c in comp.columns]

# --- Préparer le sélecteur "km - point"
pts = comp[["km","point"]].drop_duplicates().copy()
def _lab(r):
    if pd.notna(r["km"]):
        km_txt = f"{float(r['km']):.1f}".rstrip("0").rstrip(".")
        return f"{km_txt} km - {r['point']}"
    return f"{r['point']}"
pts["label"] = pts.apply(_lab, axis=1)

selected_label = st.selectbox(
    "Choisis un point (km - point)",
    options=pts["label"].tolist(),
    index=0,
    key=ui_key("bars_tranches_point_selector")
)
sel_row = pts.loc[pts["label"] == selected_label].iloc[0]
sel_km, sel_point = sel_row["km"], sel_row["point"]

# --- Extraire la ligne correspondante dans comp
row = comp[(comp["point"] == sel_point) & (comp["km"] == sel_km)]
if row.empty:
    st.info("Aucune donnée disponible pour ce point.")
else:
    row = row.iloc[0]

    # Construire le DataFrame long pour Altair: (serie, seconds, hms)
    chart_rows = []
    for lab in ordered_labels:
        sec_col = f"{lab}__sec"
        sec_val = row.get(sec_col, np.nan)
        txt_val = row.get(lab, None)
        chart_rows.append({"serie": lab, "seconds": sec_val, "hms": txt_val})

    chart_df = pd.DataFrame(chart_rows)
    chart_df = chart_df.dropna(subset=["seconds"])
    if chart_df.empty:
        st.warning("Aucun temps de tronçon disponible sur ce point (toutes séries sont vides).")
    else:
        # Couleurs cohérentes selon la position vs tranche_sel
        greens = ["#c9f7c9", "#a8eea8", "#8be38b", "#6cd86c", "#4fcc4f", "#33c233"]  # de loin -> proche
        reds   = ["#ffd1d1", "#ffb3b3", "#ff9999", "#ff7f7f", "#ff6666", "#ff4c4c"]  # de proche -> loin
        palette = []
        for lab in ordered_labels:
            if lab == bib_label:
                palette.append("#1e88e5")  # 🔵 coureur sélectionné
            elif lab == str(tranche_sel):
                palette.append("#ffe680")  # 🟡 tranche centrale
            else:
                # essayer d'interpréter comme tranche numérique
                try:
                    t = int(lab)
                except Exception:
                    t = None
                if t is None:
                    palette.append("#cccccc")  # fallback gris
                else:
                    delta = (t - int(tranche_sel)) // 25  # -3 .. +3
                    if delta < 0:
                        idx = min(len(greens)-1, abs(delta)-1)
                        palette.append(greens[idx])
                    else:
                        idx = min(len(reds)-1, delta-1)
                        palette.append(reds[idx])

        # Ordonner l'axe X comme dans le tableau
        domain_x = ordered_labels

        # Convertir secondes -> minutes pour l'axe (plus lisible)
        chart_df["minutes"] = chart_df["seconds"] / 60.0

        bars = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("serie:N", title=None, sort=domain_x),
                y=alt.Y("minutes:Q", title=f"Temps de tronçon (minutes) — {selected_label}"),
                color=alt.Color("serie:N", scale=alt.Scale(domain=domain_x, range=palette), legend=None),
                tooltip=[
                    alt.Tooltip("serie:N", title="Série"),
                    alt.Tooltip("hms:N", title="Temps (HH:MM:SS)"),
                    alt.Tooltip("minutes:Q", title="Minutes", format=".1f"),
                ],
            )
            .properties(height=420)
        )

        st.altair_chart(bars, use_container_width=True)

        # --- Export PNG (optionnel)
        try:
            import altair_saver  # noqa: F401
            buf = io.BytesIO()
            bars.properties(width=400, height=450).save(
                buf, format="png", method="vl-convert", scale_factor=2
            )
            buf.seek(0)
            fname = (
                f"troncons_bar_{str(sel_point).replace(' ','_')}_"
                f"T{tranche_sel}_with_bib.png"
            )
            st.download_button(
                "📸 Télécharger le PNG du graphique",
                data=buf,
                file_name=fname,
                mime="image/png",
                key=ui_key("dl_png_troncons_bars")
            )
        except Exception as e:
            st.caption("💡 Pour l’export PNG, installe `vl-convert-python` (`pip install vl-convert-python`).")

# =======================
# 👥 Comparateur — Sélection jusqu'à 6 dossards + tables Tronçons & Cumulés
# =======================
st.markdown("---")
st.header("Comparateur de dossards (jusqu'à 6) — même méthode de calcul que le dossard de référence")

course_cmp = df_course.copy() if 'df_course' in locals() else df.copy()
if "temps_cumule_td" not in course_cmp.columns and "temps_cumule" in course_cmp.columns:
    course_cmp["temps_cumule_td"] = pd.to_timedelta(course_cmp["temps_cumule"], errors="coerce")

# --- Helpers locaux ---
def _sort_by_km_then_time_local(t):
    t = t.copy()
    if "km" in t.columns and t["km"].notna().any():
        by = [c for c in ["km","temps_cumule_td"] if c in t.columns]
        t = t.sort_values(by=by, ascending=[True]*len(by))
    else:
        t = t.sort_values(by=["temps_cumule_td"], ascending=[True])
    return t

def _fmt_hms_from_seconds_local(sec):
    if pd.isna(sec):
        return None
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def first_nonnull(series):
    s = series.dropna()
    return s.iloc[0] if not s.empty else pd.NA

# 0) Classement général pour détecter les 6 premiers (arrivants)
work = course_cmp.copy()
km_max_cmp = float(work["km"].dropna().max()) if "km" in work.columns and work["km"].notna().any() else None

grp = []
for bib, g in work.groupby(work["dossard"].astype(str)):
    g = g[g["temps_cumule_td"].notna()].copy()
    if g.empty:
        continue
    g = _sort_by_km_then_time_local(g)
    last = g.iloc[-1]
    grp.append({
        "dossard": bib,
        "final_km": float(last["km"]) if "km" in g.columns and pd.notna(last.get("km", np.nan)) else np.nan,
        "final_td": last["temps_cumule_td"],
    })
rank_tmp = pd.DataFrame(grp)

if not rank_tmp.empty:
    if km_max_cmp is not None and not np.isnan(km_max_cmp):
        tol = 1e-3
        rank_tmp["is_finisher"] = np.isfinite(rank_tmp["final_km"]) & (np.abs(rank_tmp["final_km"] - km_max_cmp) <= tol)
        # d'abord arrivants (is_finisher True), triés par temps, puis les autres
        rank_tmp = rank_tmp.sort_values(by=["is_finisher","final_td"], ascending=[False, True]).reset_index(drop=True)
    else:
        rank_tmp["is_finisher"] = True
        rank_tmp = rank_tmp.sort_values(by=["final_td"], ascending=[True]).reset_index(drop=True)
    top6_bibs = rank_tmp["dossard"].astype(str).head(6).tolist()
else:
    top6_bibs = []

# --- Construire la liste des options "dossard - NOM Prénom index" triées par dossard numérique ---
all_bibs = []
for bib, g in course_cmp.groupby(course_cmp["dossard"].astype(str)):
    nom = first_nonnull(g.get("nom", pd.Series(dtype="string")))
    prenom = first_nonnull(g.get("prenom", pd.Series(dtype="string")))
    idx = first_nonnull(pd.to_numeric(g.get("index_utmb", pd.Series(dtype="float")), errors="coerce"))
    nom_txt = "" if pd.isna(nom) else str(nom).strip().upper()
    prenom_txt = "" if pd.isna(prenom) else str(prenom).strip().title()
    idx_txt = "" if pd.isna(idx) else str(int(round(float(idx))))
    label = f"{bib} - {nom_txt} {prenom_txt} {idx_txt}".strip()
    try:
        bib_num = int(str(bib))
    except Exception:
        bib_num = None
    all_bibs.append((str(bib), bib_num, label))

def _bib_sort_key(item):
    bib, bib_num, _ = item
    return (0, bib_num) if bib_num is not None else (1, str(bib))

all_bibs = sorted(all_bibs, key=_bib_sort_key)

options = ["— Choisir —"] + [lab for (_, _, lab) in all_bibs]
bib_from_label = {lab: bib for (bib, _, lab) in all_bibs}
label_from_bib = {bib: lab for (bib, _, lab) in all_bibs}

# Indices par défaut (6 premiers aux sélecteurs)
def _default_index_for_bib(b):
    lab = label_from_bib.get(str(b))
    return options.index(lab) if lab in options else 0

default_indices = [ _default_index_for_bib(b) for b in (top6_bibs + ["","","","",""])[:6] ]
while len(default_indices) < 6:
    default_indices.append(0)

# ⚙️ Paramétrage PNG visible dans l’UI (avec tes valeurs par défaut)
with st.expander("⚙️ Paramètres d'export PNG"):
    cA, cB, cC = st.columns(3)
    with cA:
        width_km = st.number_input("Largeur colonne km", min_value=0.05, max_value=2.0, value=0.15, step=0.05)
        header_fontsize = st.number_input("Taille police en-tête", min_value=6, max_value=16, value=8, step=1)
    with cB:
        width_point = st.number_input("Largeur colonne point", min_value=0.10, max_value=2.5, value=0.35, step=0.05)
        header_height_factor = st.number_input("Hauteur relative en-tête", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
    with cC:
        width_runner = st.number_input("Largeur colonne coureurs", min_value=0.10, max_value=2.5, value=0.20, step=0.05)
        dpi_png = st.number_input("DPI PNG", min_value=100, max_value=600, value=220, step=10)


# Sélecteurs (préremplis avec les 6 premiers si disponibles)
c1, c2, c3 = st.columns(3)
with c1:
    pick1 = st.selectbox("Concurrent 1 (référence)", options=options,
                         index=default_indices[0], key=ui_key("cmp_bib1"))
with c2:
    pick2 = st.selectbox("Concurrent 2", options=options, index=default_indices[1], key=ui_key("cmp_bib2"))
with c3:
    pick3 = st.selectbox("Concurrent 3", options=options, index=default_indices[2], key=ui_key("cmp_bib3"))
c4, c5, c6 = st.columns(3)
with c4:
    pick4 = st.selectbox("Concurrent 4", options=options, index=default_indices[3], key=ui_key("cmp_bib4"))
with c5:
    pick5 = st.selectbox("Concurrent 5", options=options, index=default_indices[4], key=ui_key("cmp_bib5"))
with c6:
    pick6 = st.selectbox("Concurrent 6", options=options, index=default_indices[5], key=ui_key("cmp_bib6"))

picks = [pick1, pick2, pick3, pick4, pick5, pick6]
picked_bibs = [bib_from_label[p] for p in picks if p != "— Choisir —"]

if pick1 == "— Choisir —":
    st.info("Sélectionne au moins le **Concurrent 1 (référence)** pour lancer la comparaison.")
else:
    bib_ref = bib_from_label[pick1]

    # --- Référence selon logique arrière (n − (n−1) sinon n − (n−2)) ---
    gref = course_cmp[course_cmp["dossard"].astype(str) == str(bib_ref)][["km","point","temps_cumule_td","nom","prenom","index_utmb"]].copy()
    if gref.empty:
        st.warning("Données indisponibles pour le dossard de référence.")
    else:
        gref = _sort_by_km_then_time_local(gref)

        cur = gref["temps_cumule_td"]
        prev1 = cur.shift(1)
        prev2 = cur.shift(2)

        has_cur = cur.notna()
        use_prev1 = has_cur & prev1.notna()
        use_prev2 = has_cur & (~use_prev1) & prev2.notna()

        seg_ref_td = pd.Series(pd.NaT, index=gref.index, dtype="timedelta64[ns]")
        seg_ref_td[use_prev1] = (cur - prev1)[use_prev1]
        seg_ref_td[use_prev2] = (cur - prev2)[use_prev2]
        seg_ref_td = seg_ref_td.where(seg_ref_td > pd.Timedelta(0))

        base_sel = gref[["km","point","temps_cumule_td"]].copy()
        base_sel["step_used"] = np.where(use_prev1, 1, np.where(use_prev2, 2, np.nan))
        base_sel["temps_troncon_ref_td"] = seg_ref_td
        base_sel = base_sel.dropna(subset=["temps_cumule_td","temps_troncon_ref_td","step_used"])

        if base_sel.empty:
            st.info("Aucun tronçon calculable pour le dossard de référence (règle arrière n−1 sinon n−2).")
        else:
            # Libellés colonnes "NOM Prénom"
            def col_label_for_bib(bib):
                rows = course_cmp[course_cmp["dossard"].astype(str) == str(bib)]
                nom = first_nonnull(rows.get("nom", pd.Series(dtype="string")))
                prenom = first_nonnull(rows.get("prenom", pd.Series(dtype="string")))
                nom_txt = "" if pd.isna(nom) else str(nom).strip().upper()
                prenom_txt = "" if pd.isna(prenom) else str(prenom).strip().title()
                full = f"{nom_txt} {prenom_txt}".strip()
                return full if full else str(bib)

            labels = [(b, col_label_for_bib(b)) for b in picked_bibs]  # conserve l'ordre des picks
            if bib_ref not in [b for b,_ in labels]:
                labels = [(bib_ref, col_label_for_bib(bib_ref))] + labels
            else:
                lbl_map = dict(labels)
                labels = [(bib_ref, lbl_map.get(bib_ref, col_label_for_bib(bib_ref)))] + [(b,l) for (b,l) in labels if b != bib_ref]

            # === Tableau 1 : Temps de tronçon ===
            troncon_df = base_sel[["km","point","step_used"]].copy()
            troncon_df = troncon_df.rename(columns={"step_used": "pas (+1/+2)"})
            troncon_df[labels[0][1]] = base_sel["temps_troncon_ref_td"].apply(
                lambda x: _fmt_hms_from_seconds_local(pd.Timedelta(x).total_seconds())
            )

            def troncon_same_step_for(bib):
                g = course_cmp[course_cmp["dossard"].astype(str) == str(bib)][["km","point","temps_cumule_td"]].copy()
                if g.empty:
                    return None
                g = _sort_by_km_then_time_local(g)
                g["row_id"] = np.arange(len(g))
                use_km = g["km"].notna().any()
                if use_km:
                    idx_map = {(str(r["point"]), float(r["km"]) if pd.notna(r["km"]) else np.nan): int(r["row_id"])
                               for _, r in g.iterrows()}
                else:
                    idx_map = {str(r["point"]): int(r["row_id"]) for _, r in g.iterrows()}
                secs = []
                for _, r in base_sel[["point","km","step_used"]].iterrows():
                    p = str(r["point"]); k = float(r["km"]) if pd.notna(r["km"]) else np.nan
                    step = int(r["step_used"])
                    if use_km and not np.isnan(k):
                        cur_i = idx_map.get((p, k)) or idx_map.get(p)
                    else:
                        cur_i = idx_map.get(p)
                    if cur_i is None:
                        secs.append(np.nan); continue
                    prev_i = cur_i - step
                    if prev_i < 0:
                        secs.append(np.nan); continue
                    cur_td  = g.loc[g["row_id"] == cur_i, "temps_cumule_td"].values[0]
                    prev_td = g.loc[g["row_id"] == prev_i, "temps_cumule_td"].values[0]
                    if pd.isna(cur_td) or pd.isna(prev_td):
                        secs.append(np.nan); continue
                    seg = cur_td - prev_td
                    try:
                        s = seg.total_seconds()
                    except AttributeError:
                        s = pd.Timedelta(seg).total_seconds()
                    secs.append(s if s > 0 else np.nan)
                return secs

            for (bib, lab) in labels[1:]:
                secs = troncon_same_step_for(bib)
                troncon_df[lab] = [_fmt_hms_from_seconds_local(s) if (secs and pd.notna(s)) else None for s in (secs or [np.nan]*len(troncon_df))]

            # Affichage + style (dégradé vert→jaune→rouge par ligne)
            st.subheader("⏱️ Temps de tronçon (même pas que la référence)")
            time_cols_tr = [lab for (_, lab) in labels]  # ref d'abord

            troncon_secs_df = pd.DataFrame(index=troncon_df.index)
            troncon_secs_df[time_cols_tr[0]] = base_sel["temps_troncon_ref_td"].apply(lambda x: pd.Timedelta(x).total_seconds() if pd.notna(x) else np.nan)
            for (bib, lab) in labels[1:]:
                secs = troncon_same_step_for(bib)
                troncon_secs_df[lab] = [float(s) if pd.notna(s) else np.nan for s in (secs or [np.nan]*len(troncon_df))]

            def gradient_gyr_row(row, secs_df, col_names):
                styles = [""] * len(row.index)
                vals = []
                idx_of = {}
                for col in col_names:
                    if col in row.index:
                        idx_of[col] = row.index.get_loc(col)
                        vals.append((col, secs_df.loc[row.name, col]))
                valid = [(c, v) for (c, v) in vals if pd.notna(v)]
                if len(valid) <= 1:
                    return pd.Series(styles, index=row.index)
                vmin = min(v for _, v in valid); vmax = max(v for _, v in valid)
                span = (vmax - vmin) if (vmax - vmin) > 0 else 1.0
                def interp_color(t):
                    if t <= 0.5:
                        g = (0xC9/255, 0xF7/255, 0xC9/255)
                        y = (0xFF/255, 0xD8/255, 0x4D/255)
                        u = t / 0.5
                        return (g[0] + u*(y[0]-g[0]), g[1] + u*(y[1]-g[1]), g[2] + u*(y[2]-g[2]))
                    else:
                        y = (0xFF/255, 0xD8/255, 0x4D/255)
                        r = (0xFF/255, 0x99/255, 0x99/255)
                        u = (t - 0.5) / 0.5
                        return (y[0] + u*(r[0]-y[0]), y[1] + u*(r[1]-y[1]), y[2] + u*(r[2]-y[2]))
                for c, v in valid:
                    t = (v - vmin) / span
                    rgb = interp_color(float(t))
                    hexcol = '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
                    styles[idx_of[c]] = f"background-color: {hexcol}"
                return pd.Series(styles, index=row.index)

            styled_tr = troncon_df.style.apply(lambda r: gradient_gyr_row(r, troncon_secs_df, time_cols_tr), axis=1)
            st.dataframe(styled_tr, use_container_width=True)

            # -------- Export PNG (Tableau Tronçons) --------
            import io, matplotlib.pyplot as plt

            def _compute_colors_for_png(df_display, secs_df, time_cols):
                colors = []
                for i in range(len(df_display)):
                    row_colors = [(1,1,1)] * len(df_display.columns)
                    vals = [secs_df.loc[df_display.index[i], c] for c in time_cols if c in secs_df.columns]
                    vals = [v for v in vals if pd.notna(v)]
                    if len(vals) > 1:
                        vmin, vmax = float(min(vals)), float(max(vals)); span = (vmax - vmin) if (vmax - vmin) > 0 else 1.0
                        def interp_rgb(t):
                            if t <= 0.5:
                                g = (0xC9/255, 0xF7/255, 0xC9/255); y = (0xFF/255, 0xD8/255, 0x4D/255); u = t / 0.5
                                return (g[0] + u*(y[0]-g[0]), g[1] + u*(y[1]-g[1]), g[2] + u*(y[2]-g[2]))
                            else:
                                y = (0xFF/255, 0xD8/255, 0x4D/255); r = (0xFF/255, 0x99/255, 0x99/255); u = (t - 0.5) / 0.5
                                return (y[0] + u*(r[0]-y[0]), y[1] + u*(r[1]-y[1]), y[2] + u*(r[2]-y[2]))
                        for c in time_cols:
                            if c in df_display.columns and c in secs_df.columns:
                                v = secs_df.loc[df_display.index[i], c]
                                if pd.notna(v):
                                    t = (float(v) - vmin) / span
                                    row_colors[df_display.columns.get_loc(c)] = interp_rgb(t)
                    colors.append(row_colors)
                return colors

            def _export_png(df_display, secs_df, time_cols, filename,
                            header_fontsize=header_fontsize, header_height_factor=header_height_factor,
                            width_km=width_km, width_point=width_point, width_runner=width_runner, dpi=dpi_png):
                lead = [c for c in ["km","point"] if c in df_display.columns]
                keep_runners = [c for c in time_cols if c in df_display.columns]
                df_png = df_display[lead + keep_runners].copy()
                colors_png = _compute_colors_for_png(df_png, secs_df, keep_runners)
                ncols, nrows = len(df_png.columns), len(df_png)
                fig_w = max(8, 0.5 * ncols); fig_h = max(4, 0.35 * (nrows + 1))
                fig, ax = plt.subplots(figsize=(fig_w, fig_h)); ax.axis("off")
                cell_text = df_png.fillna("").values.tolist(); col_labels = df_png.columns.tolist()
                table = ax.table(cellText=cell_text, colLabels=col_labels, cellColours=colors_png, loc="center")
                table.auto_set_font_size(False); table.set_fontsize(header_fontsize); table.scale(1, 1.3)
                for j in range(ncols):
                    cell = table[0, j]
                    if cell is not None:
                        cell.set_height(cell.get_height() * header_height_factor)
                        cell.set_text_props(weight='bold')
                for j, col_name in enumerate(df_png.columns):
                    for i in range(nrows + 1):
                        cell = table[i, j]
                        if cell is not None:
                            if col_name == "km":
                                cell.set_width(width_km)
                            elif col_name == "point":
                                cell.set_width(width_point)
                            else:
                                cell.set_width(width_runner)
                fig.tight_layout(pad=0.5)
                buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight"); plt.close(fig); buf.seek(0)
                st.download_button("📥 Télécharger le tableau Tronçons (PNG)", data=buf.getvalue(),
                                   file_name=filename, mime="image/png")

            troncon_df_for_png = troncon_df.drop(columns=["pas (+1/+2)"], errors="ignore")
            _export_png(troncon_df_for_png, troncon_secs_df, time_cols_tr, filename="comparateur_troncons.png")

            # ============= Tableau 2 : CUMULÉS =============
            st.subheader("⏳ Temps cumulés aux points (alignés sur la référence)")

            def _secs_from_td(x):
                if pd.isna(x): return np.nan
                try: return x.total_seconds()
                except AttributeError: return pd.Timedelta(x).total_seconds()

            def _hms_from_secs(x):
                if pd.isna(x): return None
                x = int(x); h = x // 3600; m = (x % 3600) // 60; s = x % 60
                return f"{h:02d}:{m:02d}:{s:02d}"

            def _cumul_secs_for(bib):
                g = course_cmp[course_cmp["dossard"].astype(str) == str(bib)][["km","point","temps_cumule_td"]].copy()
                if g.empty: return [np.nan]*len(base_sel)
                g = _sort_by_km_then_time_local(g); g["row_id"] = np.arange(len(g))
                use_km = g["km"].notna().any()
                if use_km:
                    idx_map = {(str(r["point"]), float(r["km"]) if pd.notna(r["km"]) else np.nan): int(r["row_id"]) for _, r in g.iterrows()}
                else:
                    idx_map = {str(r["point"]): int(r["row_id"]) for _, r in g.iterrows()}
                secs = []
                for _, r in base_sel[["point","km"]].iterrows():
                    p = str(r["point"]); k = float(r["km"]) if pd.notna(r["km"]) else np.nan
                    if use_km and not np.isnan(k):
                        cur_i = idx_map.get((p, k)) or idx_map.get(p)
                    else:
                        cur_i = idx_map.get(p)
                    if cur_i is None:
                        secs.append(np.nan); continue
                    td = g.loc[g["row_id"] == cur_i, "temps_cumule_td"].values[0]
                    secs.append(_secs_from_td(td))
                return secs

            def _cumul_hms_for(bib):
                return [_hms_from_secs(s) for s in _cumul_secs_for(bib)]

            time_cols_cu = [lab for (_, lab) in labels]
            cumul_df = base_sel[["km","point"]].copy()
            for (bib, lab) in labels:
                cumul_df[lab] = _cumul_hms_for(bib)

            cumul_secs_df = pd.DataFrame(index=cumul_df.index)
            for (bib, lab) in labels:
                cumul_secs_df[lab] = _cumul_secs_for(bib)

            styled_cu = cumul_df.style.apply(lambda r: gradient_gyr_row(r, cumul_secs_df, time_cols_cu), axis=1)
            st.dataframe(styled_cu, use_container_width=True)

            # Export PNG (Cumulés) avec les mêmes réglages
            _export_png(cumul_df, cumul_secs_df, time_cols_cu, filename="comparateur_cumules.png")


# ===============================
# 👥 Top prénoms — Hommes & Femmes (finishers = temps > 0) + Export CSV
# ===============================
st.markdown("## 👥 Top prénoms — participation & finish (par sexe)")

import numpy as np
import pandas as pd

dfn = df_course.copy()

# -- Normalisation sexe
if "sexe" in dfn.columns:
    dfn["sexe"] = (
        dfn["sexe"]
        .astype(str).str.strip().str.upper()
        .replace({"H": "M", "HOMME": "M", "HOMMES": "M", "FEMME": "F", "FEMMES": "F"})
    )

# -- Détection colonne prénom
prenom_col = None
for cand in ["prenom", "prénom", "first_name", "Prenom", "Prénom", "PRENOM"]:
    if cand in dfn.columns:
        prenom_col = cand
        break

if prenom_col is None:
    st.warning("⚠️ Aucune colonne 'prenom' trouvée. Impossible de créer les tableaux par prénom.")
else:
    # Normalise le prénom (casse/tokens)
    dfn[prenom_col] = (
        dfn[prenom_col].astype(str).str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.title()
    )

    # Colonne temps cumulés
    if "temps_cumule_td" not in dfn.columns and "temps_cumule" in dfn.columns:
        dfn["temps_cumule_td"] = pd.to_timedelta(dfn["temps_cumule"], errors="coerce")

    # -- 1er & dernier point
    if "km" in dfn.columns and dfn["km"].notna().any():
        km_min = float(dfn["km"].min())
        km_max = float(dfn["km"].max())
        is_first_cp = np.isclose(dfn["km"], km_min, atol=1e-6)
        is_last_cp  = np.isclose(dfn["km"], km_max, atol=1e-6)
    else:
        if "temps_cumule_td" not in dfn.columns:
            st.warning("⚠️ Impossible d’estimer 1er/dernier point sans 'km' ni 'temps_cumule'.")
            is_first_cp = dfn.index == -1
            is_last_cp  = dfn.index == -1
        else:
            dfn = dfn.sort_values(by=["dossard", "temps_cumule_td"])
            first_idx = dfn.groupby("dossard", as_index=False).head(1).index
            last_idx  = dfn.groupby("dossard", as_index=False).tail(1).index
            is_first_cp = dfn.index.isin(first_idx)
            is_last_cp  = dfn.index.isin(last_idx)

    base_cols = ["dossard", prenom_col, "sexe", "temps_cumule_td"]
    extra_cols = [c for c in ["km", "point"] if c in dfn.columns]
    keep_cols = list(dict.fromkeys(base_cols + extra_cols))

    df_first = (
        dfn.loc[is_first_cp, keep_cols]
        .sort_values(by=["dossard"])
        .groupby("dossard", as_index=False, sort=False)
        .tail(1)
    )

    # Finishers = dernier point avec temps > 0
    df_last = (
        dfn.loc[is_last_cp & dfn["temps_cumule_td"].notna() & (dfn["temps_cumule_td"] > pd.Timedelta(0)), keep_cols]
        .sort_values(by=["dossard"])
        .groupby("dossard", as_index=False, sort=False)
        .tail(1)
    )

    def fmt_hms(td: pd.Timedelta) -> str:
        if pd.isna(td):
            return ""
        total_sec = int(round(td.total_seconds()))
        h, rem = divmod(total_sec, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def build_table_for_sex(sex_value: str, title: str, key_suffix: str):
        f_first = df_first.copy()
        f_last  = df_last.copy()
        if "sexe" in f_first.columns:
            f_first = f_first[f_first["sexe"] == sex_value]
        if "sexe" in f_last.columns:
            f_last = f_last[f_last["sexe"] == sex_value]

        # Comptes par prénom
        cnt_first = f_first.groupby(prenom_col, as_index=False).agg(participants_p1=("dossard", "nunique"))
        cnt_last  = f_last.groupby(prenom_col,  as_index=False).agg(participants_last=("dossard", "nunique"))

        # Statistiques de temps (finishers)
        stats_time = (
            f_last.groupby(prenom_col, as_index=False)
            .agg(
                mean_td=("temps_cumule_td", "mean"),
                min_td =("temps_cumule_td", "min"),
                max_td =("temps_cumule_td", "max"),
            )
        )

        tab = (
            cnt_first
            .merge(cnt_last, on=prenom_col, how="left")
            .merge(stats_time, on=prenom_col, how="left")
        )
        tab["participants_last"] = tab["participants_last"].fillna(0).astype(int)

        # % finishers (numérique pour tri)
        tab["pct_finish_num"] = np.where(
            tab["participants_p1"] > 0,
            (tab["participants_last"] / tab["participants_p1"]) * 100.0,
            np.nan
        )

        # Colonnes formatées
        tab["Temps moyen cumulé"] = tab["mean_td"].apply(fmt_hms)
        tab["Plus rapide"]        = tab["min_td"].apply(fmt_hms)
        tab["Plus lent"]          = tab["max_td"].apply(fmt_hms)

        # Tri
        tab = tab.sort_values(
            by=["participants_p1", "pct_finish_num", prenom_col],
            ascending=[False, False, True]
        )

        # Vue finale
        tab_view = tab.loc[:, [
            prenom_col,
            "participants_p1",
            "participants_last",
            "pct_finish_num",
            "Temps moyen cumulé",
            "Plus rapide",
            "Plus lent",
        ]].rename(columns={
            prenom_col: "Prénom",
            "participants_p1": "Au 1er point",
            "participants_last": "Au dernier point (temps>0)",
        }).reset_index(drop=True)

        # Affichage % avec signe (arrondi entier -> 88%)
        tab_view["% finishers"] = tab_view["pct_finish_num"].round(0).astype(pd.Int64Dtype()).astype(str) + "%"

        # Ordonner colonnes et head(20)
        tab_view = tab_view[[
            "Prénom",
            "Au 1er point",
            "Au dernier point (temps>0)",
            "% finishers",
            "Temps moyen cumulé",
            "Plus rapide",
            "Plus lent",
        ]].head(20)

        st.markdown(f"### {title}")
        st.dataframe(tab_view, use_container_width=True, hide_index=True)

        # Export CSV
        csv_bytes = tab_view.to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 Exporter en CSV",
            data=csv_bytes,
            file_name=f"top_prenoms_{'hommes' if sex_value=='M' else 'femmes'}.csv",
            mime="text/csv",
            key=f"dl_csv_prenoms_{key_suffix}"
        )

    # Tables H & F
    build_table_for_sex("M", "👟 Hommes — Top 20 prénoms", key_suffix="M")
    build_table_for_sex("F", "🎽 Femmes — Top 20 prénoms", key_suffix="F")

# ============================================
# 🧮📊 Arrivants (finishers) par catégorie d’âge + Export PNG (uses 'categorie_age')
# ============================================
st.markdown("## 🧮📊 Nombre d’arrivants par catégorie d’âge")

def _ui_key_local(s: str) -> str:
    try:
        return ui_key(s)  # si ta fonction ui_key existe déjà
    except Exception:
        return s

df_age = df_course.copy()

# Assure la colonne de temps cumulé au format timedelta
if "temps_cumule_td" not in df_age.columns and "temps_cumule" in df_age.columns:
    df_age["temps_cumule_td"] = pd.to_timedelta(df_age["temps_cumule"], errors="coerce")

# Normalise la colonne sexe (si présente)
if "sexe" in df_age.columns:
    df_age["sexe"] = (
        df_age["sexe"].astype(str).str.strip().str.upper()
        .replace({"H": "M", "HOMME": "M", "HOMMES": "M", "FEMME": "F", "FEMMES": "F"})
    )

# On prend le DERNIER enregistrement par dossard (km max si dispo, sinon dernier temps)
if "km" in df_age.columns and df_age["km"].notna().any():
    df_age = df_age.sort_values(["dossard", "km", "temps_cumule_td"])
else:
    df_age = df_age.sort_values(["dossard", "temps_cumule_td"])

last_by_bib = df_age.groupby("dossard", as_index=False).tail(1)

# Finishers = dernier point avec un temps > 0
finishers = last_by_bib[
    last_by_bib["temps_cumule_td"].notna() & (last_by_bib["temps_cumule_td"] > pd.Timedelta(0))
].copy()

# Vérification colonne obligatoire
if "categorie_age" not in finishers.columns:
    st.warning("⚠️ La colonne 'categorie_age' est absente : impossible de tracer ce graphique.")
else:
    # Définir un ordre logique des catégories (si possible) à partir d’un nombre extrait
    def _cat_order_key(cat: str):
        if pd.isna(cat):
            return (9999, cat)
        s = str(cat)
        # Cherche un entier (borne inf. de la tranche) : ex. '20-29' -> 20, 'M40' -> 40
        m = re.search(r"(\d{1,3})", s)
        if m:
            return (int(m.group(1)), s)
        # gère les cas '≤19' / '≥70'
        if "≤" in s or "<=" in s:
            return (0, s)
        if "≥" in s or ">=" in s:
            return (9000, s)
        return (5000, s)

    # Option d’affichage : Total vs Par sexe
    mode = st.radio(
        "Affichage",
        ["Total", "Par sexe"],
        horizontal=True,
        key=_ui_key_local("arrivants_age_mode_fixedcat")
    )

    if mode == "Par sexe" and "sexe" in finishers.columns:
        grp = (
            finishers.groupby(["categorie_age", "sexe"], as_index=False)
            .agg(arrivants=("dossard", "nunique"))
        )
        grp = grp[grp["categorie_age"].notna()]
        # Ordre trié par clé logique
        cat_sorted = sorted(grp["categorie_age"].unique(), key=_cat_order_key)

        chart_age = (
            alt.Chart(grp)
            .mark_bar()
            .encode(
                x=alt.X("categorie_age:N", title="Catégorie d’âge", sort=cat_sorted),
                y=alt.Y("arrivants:Q", title="Nombre d’arrivants"),
                color=alt.Color("sexe:N", title="Sexe", legend=alt.Legend(orient="bottom")),
                tooltip=[
                    alt.Tooltip("categorie_age:N", title="Catégorie d’âge"),
                    alt.Tooltip("sexe:N", title="Sexe"),
                    alt.Tooltip("arrivants:Q", title="Arrivants", format=".0f"),
                ],
            )
            .properties(height=420)
        )
    else:
        grp = (
            finishers.groupby("categorie_age", as_index=False)
            .agg(arrivants=("dossard", "nunique"))
        )
        grp = grp[grp["categorie_age"].notna()]
        cat_sorted = sorted(grp["categorie_age"].unique(), key=_cat_order_key)

        chart_age = (
            alt.Chart(grp)
            .mark_bar()
            .encode(
                x=alt.X("categorie_age:N", title="Catégorie d’âge", sort=cat_sorted),
                y=alt.Y("arrivants:Q", title="Nombre d’arrivants"),
                tooltip=[
                    alt.Tooltip("categorie_age:N", title="Catégorie d’âge"),
                    alt.Tooltip("arrivants:Q", title="Arrivants", format=".0f"),
                ],
            )
            .properties(height=420)
        )

    st.altair_chart(chart_age, use_container_width=True)

    # ===== Export PNG =====
    try:
        import altair_saver  # noqa: F401
        export_w, export_h, scale_factor = 400, 450, 6
        chart_to_save = chart_age.properties(width=export_w, height=export_h)

        buf = io.BytesIO()
        chart_to_save.save(
            buf,
            format="png",
            method="vl-convert",   # nécessite : pip install vl-convert-python
            scale_factor=scale_factor
        )
        buf.seek(0)

        fname = (
            "arrivants_par_categorie_age_"
            + ("par_sexe" if (mode == "Par sexe" and "sexe" in finishers.columns) else "total")
            + ".png"
        )

        st.download_button(
            "📸 Télécharger le graphique (PNG)",
            data=buf,
            file_name=fname,
            mime="image/png",
            key=_ui_key_local("dl_png_arrivants_age_fixedcat")
        )
    except Exception as e:
        st.caption("💡 Pour l’export PNG, installe `vl-convert-python` : `pip install vl-convert-python`.")
        st.warning(f"Erreur d'export PNG : {e}")

# ============================================
# 🕒 Tableau des temps par catégorie d’âge (Hommes / Femmes) + Export CSV
# ============================================
st.markdown("## 🕒 Temps par catégorie d’âge (Hommes / Femmes)")

import numpy as np
import pandas as pd
import re

def fmt_hms(td):
    """Convertit un timedelta ou des secondes en HH:MM:SS"""
    if pd.isna(td):
        return ""
    if isinstance(td, (float, int)):
        td = pd.to_timedelta(td, unit="s")
    total_sec = int(round(td.total_seconds()))
    h, rem = divmod(total_sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

df_age2 = df_course.copy()

# Conversion du temps cumulé
if "temps_cumule_td" not in df_age2.columns and "temps_cumule" in df_age2.columns:
    df_age2["temps_cumule_td"] = pd.to_timedelta(df_age2["temps_cumule"], errors="coerce")

# On prend le dernier point par dossard
if "km" in df_age2.columns and df_age2["km"].notna().any():
    df_age2 = df_age2.sort_values(["dossard", "km", "temps_cumule_td"])
else:
    df_age2 = df_age2.sort_values(["dossard", "temps_cumule_td"])
last_by_bib = df_age2.groupby("dossard", as_index=False).tail(1)

# Finishers = temps cumulé > 0
finishers = last_by_bib[
    last_by_bib["temps_cumule_td"].notna() & (last_by_bib["temps_cumule_td"] > pd.Timedelta(0))
].copy()

# Normalisation du sexe
if "sexe" in finishers.columns:
    finishers["sexe"] = (
        finishers["sexe"].astype(str).str.strip().str.upper()
        .replace({"H": "M", "HOMME": "M", "HOMMES": "M", "FEMME": "F", "FEMMES": "F"})
    )

# Vérification de la colonne categorie_age
if "categorie_age" not in finishers.columns:
    st.warning("⚠️ La colonne 'categorie_age' est absente.")
else:
    # Fonction pour trier les catégories d’âge
    def _cat_order_key(cat):
        if pd.isna(cat):
            return 9999
        s = str(cat)
        m = re.search(r"(\d{1,3})", s)
        if m:
            return int(m.group(1))
        if "≤" in s or "<=" in s:
            return 0
        if "≥" in s or ">=" in s:
            return 9000
        return 5000

    # Fonction de génération du tableau + export CSV
    def build_age_table(df_in, sexe_val, titre, key_suffix):
        sub = df_in[df_in["sexe"] == sexe_val].copy()
        if sub.empty:
            st.info(f"Aucun finisher pour le sexe {sexe_val}.")
            return

        tab = (
            sub.groupby("categorie_age", as_index=False)
            .agg(
                temps_median=("temps_cumule_td", "median"),
                temps_min=("temps_cumule_td", "min"),
                temps_max=("temps_cumule_td", "max"),
                nb_finishers=("dossard", "nunique"),
            )
        )

        # Format HH:MM:SS
        tab["Temps médian"] = tab["temps_median"].apply(fmt_hms)
        tab["Temps min"] = tab["temps_min"].apply(fmt_hms)
        tab["Temps max"] = tab["temps_max"].apply(fmt_hms)

        # Tri par catégorie d’âge
        tab = tab.sort_values(by="categorie_age", key=lambda x: x.map(_cat_order_key))

        # Simplification du tableau final
        tab_view = tab.loc[
            :, ["categorie_age", "nb_finishers", "Temps médian", "Temps min", "Temps max"]
        ].rename(columns={
            "categorie_age": "Catégorie d’âge",
            "nb_finishers": "Nb finishers",
        })

        # Affichage
        st.markdown(f"### {titre}")
        st.dataframe(tab_view, use_container_width=True, hide_index=True)

        # Export CSV
        csv_bytes = tab_view.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Télécharger le tableau en CSV",
            data=csv_bytes,
            file_name=f"temps_par_categorie_age_{'hommes' if sexe_val=='M' else 'femmes'}.csv",
            mime="text/csv",
            key=f"dl_csv_age_{key_suffix}"
        )

    # --- Tables Hommes & Femmes ---
    build_age_table(finishers, "M", "👟 Hommes — Temps par catégorie d’âge", key_suffix="M")
    build_age_table(finishers, "F", "🎽 Femmes — Temps par catégorie d’âge", key_suffix="F")

# ============================================
# 🏁 Tableaux par catégorie d'index UTMB (H / F)
#    Nb finishers | Temps médian | Temps min | Temps max  + Export CSV
# ============================================

st.markdown("## 🏁 Résumé par catégorie d’index UTMB — Hommes / Femmes")

import numpy as np
import pandas as pd

# Utilitaire HH:MM:SS (défini si absent)
if "fmt_hms" not in locals():
    def fmt_hms(td):
        """Convertit un timedelta OU des secondes en HH:MM:SS (chaîne)."""
        if td is None or (isinstance(td, float) and np.isnan(td)) or pd.isna(td):
            return ""
        if isinstance(td, (float, int)):
            td = pd.to_timedelta(td, unit="s")
        total_sec = int(round(pd.to_timedelta(td).total_seconds()))
        h, rem = divmod(total_sec, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

# Base course
df_utmb = df_course.copy()

# Temps cumulé en timedelta si besoin
if "temps_cumule_td" not in df_utmb.columns and "temps_cumule" in df_utmb.columns:
    df_utmb["temps_cumule_td"] = pd.to_timedelta(df_utmb["temps_cumule"], errors="coerce")

# Tranche UTMB (25) si absente
if "index_utmb_tranche25" not in df_utmb.columns and "index_utmb" in df_utmb.columns:
    df_utmb["index_utmb"] = pd.to_numeric(df_utmb["index_utmb"], errors="coerce")
    df_utmb["index_utmb_tranche25"] = df_utmb["index_utmb"].apply(
        lambda x: int(round(float(x)/25.0)*25) if pd.notna(x) else np.nan
    )

# Normaliser sexe -> H / F
if "sexe" in df_utmb.columns:
    raw = df_utmb["sexe"].astype(str).str.strip().str.upper()
    map_dict = {
        "H": "H", "M": "H", "MALE": "H", "HOMME": "H", "MAN": "H", "GARCON": "H", "G": "H",
        "F": "F", "W": "F", "WOMAN": "F", "WOMEN": "F", "FEMALE": "F", "FEMME": "F"
    }
    df_utmb["sexe"] = raw.map(map_dict).fillna(raw)  # garde la valeur si déjà H/F
else:
    df_utmb["sexe"] = None

# Dernière ligne par dossard (finish)
if "km" in df_utmb.columns and df_utmb["km"].notna().any():
    df_utmb = df_utmb.sort_values(["dossard", "km", "temps_cumule_td"])
else:
    df_utmb = df_utmb.sort_values(["dossard", "temps_cumule_td"])
last_by_bib = df_utmb.groupby("dossard", as_index=False).tail(1)

# Finishers: temps > 0 et tranche connue
finishers_u = last_by_bib[
    last_by_bib["temps_cumule_td"].notna() &
    (last_by_bib["temps_cumule_td"] > pd.Timedelta(0)) &
    last_by_bib["index_utmb_tranche25"].notna()
].copy()

# Petit diagnostic rapide (optionnel)
st.caption("Répartition sexe (finishers) : " +
           finishers_u["sexe"].value_counts(dropna=False).to_dict().__repr__())

def make_tab_by_tranche(df_in: pd.DataFrame, sex_label: str):
    """Construit le tableau agrégé par tranche 25 pour un sexe donné ('H' ou 'F')."""
    sub = df_in.copy()
    if sex_label in ("H", "F"):
        sub = sub[sub["sexe"] == sex_label]
    else:
        sub = sub[sub["sexe"].isin(["H","F"])]

    if sub.empty:
        return pd.DataFrame(columns=[
            "Catégorie index (tranche 25)", "Nb finishers", "Temps médian", "Temps min", "Temps max"
        ])

    tab = (
        sub.groupby("index_utmb_tranche25", as_index=False)
           .agg(
               nb_finishers=("dossard", "nunique"),
               temps_median=("temps_cumule_td", "median"),
               temps_min=("temps_cumule_td", "min"),
               temps_max=("temps_cumule_td", "max"),
           )
           .sort_values("index_utmb_tranche25")
    )

    tab["Temps médian"] = tab["temps_median"].apply(fmt_hms)
    tab["Temps min"]    = tab["temps_min"].apply(fmt_hms)
    tab["Temps max"]    = tab["temps_max"].apply(fmt_hms)

    tab_view = tab.loc[:, [
        "index_utmb_tranche25", "nb_finishers", "Temps médian", "Temps min", "Temps max"
    ]].rename(columns={
        "index_utmb_tranche25": "Catégorie index (tranche 25)",
        "nb_finishers": "Nb finishers",
    })
    return tab_view

# Tableau Hommes (H)
st.subheader("👨 Hommes — par tranche d’index UTMB")
tab_h = make_tab_by_tranche(finishers_u, "H")
st.dataframe(tab_h, use_container_width=True, hide_index=True)
csv_h = tab_h.to_csv(index=False).encode("utf-8")
st.download_button(
    "💾 Télécharger (CSV) — Hommes",
    data=csv_h,
    file_name="resume_tranche_utmb_hommes.csv",
    mime="text/csv",
    key="dl_csv_resume_tranche_utmb_h"
)

# Tableau Femmes (F)
st.subheader("👩 Femmes — par tranche d’index UTMB")
tab_f = make_tab_by_tranche(finishers_u, "F")
st.dataframe(tab_f, use_container_width=True, hide_index=True)
csv_f = tab_f.to_csv(index=False).encode("utf-8")
st.download_button(
    "💾 Télécharger (CSV) — Femmes",
    data=csv_f,
    file_name="resume_tranche_utmb_femmes.csv",
    mime="text/csv",
    key="dl_csv_resume_tranche_utmb_f"
)

# ============================================
# 📋 Tableau récapitulatif global — version en lignes (clé / valeur)
# ============================================

st.markdown("## 📋 Récapitulatif global — prénoms, index, finishers & chronos (version lisible)")

import numpy as np
import pandas as pd

# --- Fonctions utilitaires ---
def fmt_hms(td_or_sec):
    """Convertit un timedelta ou un nombre de secondes en HH:MM:SS."""
    if td_or_sec is None or (isinstance(td_or_sec, float) and np.isnan(td_or_sec)) or pd.isna(td_or_sec):
        return ""
    if isinstance(td_or_sec, (int, float, np.integer, np.floating)):
        td = pd.to_timedelta(float(td_or_sec), unit="s")
    else:
        td = pd.to_timedelta(td_or_sec)
    total_sec = int(round(td.total_seconds()))
    h, rem = divmod(total_sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def pick_firstname_column(df_in):
    for cand in ["prenom", "first_name", "firstname", "Prenom", "FirstName"]:
        if cand in df_in.columns:
            return cand
    return None

def normalize_sex(s):
    """Mappe diverses valeurs vers H / F."""
    if pd.isna(s): return None
    v = str(s).strip().upper()
    map_dict = {
        "H": "H", "M": "H", "MALE": "H", "HOMME": "H", "MAN": "H", "GARCON": "H", "G": "H",
        "F": "F", "W": "F", "WOMAN": "F", "WOMEN": "F", "FEMALE": "F", "FEMME": "F"
    }
    return map_dict.get(v, v)

# --- Base ---
base = df_course.copy()
if "sexe" not in base.columns:
    base["sexe"] = None
base["sexe"] = base["sexe"].apply(normalize_sex)

if "temps_cumule_td" not in base.columns and "temps_cumule" in base.columns:
    base["temps_cumule_td"] = pd.to_timedelta(base["temps_cumule"], errors="coerce")

if "index_utmb_tranche25" not in base.columns and "index_utmb" in base.columns:
    base["index_utmb"] = pd.to_numeric(base["index_utmb"], errors="coerce")
    base["index_utmb_tranche25"] = base["index_utmb"].apply(
        lambda x: int(round(float(x)/25.0)*25) if pd.notna(x) else np.nan
    )

col_prenom = pick_firstname_column(base)
if col_prenom is None:
    col_prenom = "prenom"
    base[col_prenom] = None

# --- Détermination des starters (1er point) ---
if "km" in base.columns and base["km"].notna().any():
    km_min = base["km"].min()
    starters = base[np.isclose(base["km"], km_min, atol=1e-6)].copy()
else:
    starters = base.sort_values(["dossard", "temps_cumule_td"]).groupby("dossard", as_index=False).head(1)

# --- Derniers points = finishers ---
if "km" in base.columns and base["km"].notna().any():
    last_by_bib = base.sort_values(["dossard", "km", "temps_cumule_td"]).groupby("dossard", as_index=False).tail(1)
else:
    last_by_bib = base.sort_values(["dossard", "temps_cumule_td"]).groupby("dossard", as_index=False).tail(1)

finishers = last_by_bib[
    last_by_bib["temps_cumule_td"].notna() &
    (last_by_bib["temps_cumule_td"] > pd.Timedelta(0))
].copy()

# --- Compteurs & analyses ---
def count_unique_bibs(df_in, sex=None):
    d = df_in
    if sex in ("H", "F"):
        d = d[d["sexe"] == sex]
    return int(d["dossard"].nunique()) if "dossard" in d.columns else 0

nb_start_H = count_unique_bibs(starters, "H")
nb_start_F = count_unique_bibs(starters, "F")
nb_finish_H = count_unique_bibs(finishers, "H")
nb_finish_F = count_unique_bibs(finishers, "F")
nb_finish_total = count_unique_bibs(finishers, None)

def top_firstname_counts(starters_df, finishers_df, sex_label):
    stx = starters_df[starters_df["sexe"] == sex_label]
    if stx.empty:
        return ("", 0, 0)
    st_bib_name = stx.dropna(subset=["dossard"]).loc[:, ["dossard", col_prenom]].drop_duplicates()
    st_bib_name[col_prenom] = st_bib_name[col_prenom].fillna("").astype(str).str.strip()
    st_bib_name = st_bib_name[st_bib_name[col_prenom] != ""]
    if st_bib_name.empty:
        return ("", 0, 0)
    cnt = st_bib_name[col_prenom].value_counts()
    top_name = cnt.index[0]
    top_participants = int(cnt.iloc[0])
    fin = finishers_df[finishers_df["sexe"] == sex_label]
    fin_bib = fin.dropna(subset=["dossard"]).loc[:, ["dossard", col_prenom]].drop_duplicates()
    fin_bib[col_prenom] = fin_bib[col_prenom].fillna("").astype(str).str.strip()
    finished_with_name = int(fin_bib[fin_bib[col_prenom] == top_name]["dossard"].nunique())
    return (top_name, top_participants, finished_with_name)

top_f_name, top_f_start, top_f_finish = top_firstname_counts(starters, finishers, "F")
top_h_name, top_h_start, top_h_finish = top_firstname_counts(starters, finishers, "H")

def index_extremes(fin_df, sex_label):
    d = fin_df[(fin_df["sexe"] == sex_label) & fin_df["index_utmb"].notna()].copy()
    if d.empty:
        return (np.nan, np.nan)
    return (float(d["index_utmb"].max()), float(d["index_utmb"].min()))

idx_f_max, idx_f_min = index_extremes(finishers, "F")
idx_h_max, idx_h_min = index_extremes(finishers, "H")

def fastest_time(fin_df, sex_label):
    d = fin_df[fin_df["sexe"] == sex_label].copy()
    if d.empty or d["temps_cumule_td"].isna().all():
        return ""
    best = d.loc[d["temps_cumule_td"].idxmin(), "temps_cumule_td"]
    return fmt_hms(best)

best_time_H = fastest_time(finishers, "H")
best_time_F = fastest_time(finishers, "F")

if finishers.empty or finishers["temps_cumule_td"].isna().all():
    last_finisher_firstname = ""
else:
    last_row = finishers.loc[finishers["temps_cumule_td"].idxmax()]
    last_finisher_firstname = str(last_row.get(col_prenom, "") or "").strip()

# --- Tableau format lignes ---
recap_dict = {
    "🏁 Nombre total de finishers": nb_finish_total,
    "👩 Prénom féminin le plus utilisé": top_f_name,
    "♀ Participantes au 1er point (pour ce prénom)": top_f_start,
    "♀ Finisheuses (pour ce prénom)": top_f_finish,
    "👨 Prénom masculin le plus utilisé": top_h_name,
    "♂ Participants au 1er point (pour ce prénom)": top_h_start,
    "♂ Finishers (pour ce prénom)": top_h_finish,
    "📈 Index féminin le plus haut": f"{idx_f_max:.0f}" if not np.isnan(idx_f_max) else "",
    "📉 Index féminin le plus bas": f"{idx_f_min:.0f}" if not np.isnan(idx_f_min) else "",
    "📈 Index masculin le plus haut": f"{idx_h_max:.0f}" if not np.isnan(idx_h_max) else "",
    "📉 Index masculin le plus bas": f"{idx_h_min:.0f}" if not np.isnan(idx_h_min) else "",
    "⏱️ Temps masculin le plus rapide": best_time_H,
    "⏱️ Temps féminin le plus rapide": best_time_F,
    "🏃‍♀️ Prénom du dernier finisher": last_finisher_firstname,
}

recap_table = pd.DataFrame(list(recap_dict.items()), columns=["Élément", "Valeur"])

st.dataframe(recap_table, use_container_width=True, hide_index=True)

# Export CSV
csv_recap = recap_table.to_csv(index=False).encode("utf-8")
st.download_button(
    "💾 Télécharger le récapitulatif (CSV)",
    data=csv_recap,
    file_name="recap_global_prenoms_index_finishers_lisible.csv",
    mime="text/csv",
    key="dl_csv_recap_global_lignes"
)
