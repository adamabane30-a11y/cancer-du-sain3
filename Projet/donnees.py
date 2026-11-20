"""Lecture et validation du fichier CSV importé par l'administrateur.

Le jeu importé remplace uniquement les données de référence de
l'application (médianes, bornes, exemples, statistiques) : le modèle
entraîné dans le notebook reste inchangé.
"""

import io

import pandas as pd

TAILLE_MAX = 50 * 1024 * 1024  # 50 Mo


def _lectures_possibles(contenu):
    """Décode le fichier puis tente les lectures plausibles : séparateur
    auto-détecté, export Excel français (point-virgule + virgule décimale)…
    Renvoie une liste de DataFrames."""
    texte = None
    for encodage in ("utf-8-sig", "latin-1"):
        try:
            texte = contenu.decode(encodage)
            break
        except UnicodeDecodeError:
            continue
    if texte is None:
        raise ValueError("encodage non reconnu (ni UTF-8 ni Latin-1)")

    variantes = (
        {"sep": None},  # détection automatique (virgule, point-virgule, tabulation)
        {"sep": ";", "decimal": ","},  # export Excel français
        {"sep": ",", "decimal": ","},
    )
    lectures = []
    for options in variantes:
        try:
            lectures.append(pd.read_csv(io.StringIO(texte), engine="python", **options))
        except Exception:
            continue
    return lectures


def valider_jeu(contenu, features):
    """Valide le contenu d'un CSV importé.

    Renvoie (jeu, erreurs, avertissements) : jeu vaut None si le fichier est
    inutilisable, sinon un DataFrame limité aux colonnes utiles (mesures
    converties en nombres + CLASS éventuelle).
    """
    erreurs, avertissements = [], []

    try:
        lectures = _lectures_possibles(contenu)
    except ValueError as e:
        return None, [f"Fichier illisible : {e}."], []
    if not lectures:
        return None, ["Fichier vide ou illisible : aucun séparateur de colonnes reconnu."], []

    # On retient la lecture qui reconnaît le mieux les colonnes attendues,
    # à colonnes égales celle dont les mesures sont réellement numériques.
    def pertinence(lecture):
        presentes = [f for f in features if f in lecture.columns]
        numeriques = 0
        if presentes:
            conversion = lecture[presentes].apply(pd.to_numeric, errors="coerce")
            numeriques = int(conversion.notna().all().sum())
        return (len(presentes), numeriques)

    df = max(lectures, key=pertinence)

    if len(df) == 0:
        return None, ["Le fichier ne contient aucune ligne de données."], []

    manquantes = [f for f in features if f not in df.columns]
    if manquantes:
        liste = ", ".join(manquantes[:6]) + ("…" if len(manquantes) > 6 else "")
        erreurs.append(
            f"{len(manquantes)} colonnes manquantes sur {len(features)} attendues : {liste}. "
            "Téléchargez le modèle de fichier ci-dessus pour partir d'une base correcte."
        )
    supplementaires = [c for c in df.columns if c not in features and c != "CLASS"]
    if supplementaires:
        liste = ", ".join(supplementaires[:6]) + ("…" if len(supplementaires) > 6 else "")
        avertissements.append(f"{len(supplementaires)} colonnes seront ignorées : {liste}")

    if erreurs:
        return None, erreurs, avertissements

    # Les mesures doivent être numériques.
    conversion = df[features].apply(pd.to_numeric, errors="coerce")
    non_numeriques = [f for f in features if conversion[f].isna().all()]
    if non_numeriques:
        liste = ", ".join(non_numeriques[:6]) + ("…" if len(non_numeriques) > 6 else "")
        erreurs.append(
            f"Colonnes non numériques : {liste}. Vérifiez que le séparateur décimal "
            "est le point, ou exportez le fichier avec des points-virgules."
        )
        return None, erreurs, avertissements

    df = df[[c for c in df.columns if c in features or c == "CLASS"]].copy()
    df[features] = conversion

    nb_illisibles = int(conversion.isna().sum().sum())
    if nb_illisibles:
        avertissements.append(
            f"{nb_illisibles} valeurs manquantes ou illisibles : le modèle les imputera par la médiane."
        )

    if "CLASS" in df.columns:
        classe = pd.to_numeric(df["CLASS"], errors="coerce")
        if classe.dropna().empty or not classe.dropna().isin([0, 1]).all():
            erreurs.append(
                "La colonne CLASS doit contenir uniquement 0 (tumeur maligne) ou 1 (tumeur bénigne)."
            )
        else:
            df["CLASS"] = classe
    else:
        avertissements.append(
            "Colonne CLASS absente : le bouton d'exemple charge un cas aléatoire sans distinction de classe."
        )

    if len(df) < 10:
        avertissements.append(
            f"Jeu peu fourni ({len(df)} ligne{'s' if len(df) > 1 else ''}) : "
            "les statistiques affichées seront peu représentatives."
        )

    if erreurs:
        return None, erreurs, avertissements
    return df, erreurs, avertissements
