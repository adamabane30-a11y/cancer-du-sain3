import os
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from donnees import TAILLE_MAX, valider_jeu

DOSSIER = Path(__file__).parent

st.set_page_config(page_title="Détection du cancer du sein", page_icon="🩺")

# ---------------------------------------------------------------------------
# Chargement du modèle (entraîné dans 04_Cancer_du_sein.ipynb) et des données
# ---------------------------------------------------------------------------


@st.cache_resource
def charger_modele():
    return joblib.load(DOSSIER / "modele_cancer_sein.joblib")


@st.cache_data
def charger_donnees():
    return pd.read_csv(DOSSIER / "Breast_cancer.csv")


modele = charger_modele()
df_origine = charger_donnees()
FEATURES = list(modele.feature_names_in_)  # 30 colonnes, dans l'ordre du modèle
MEDIANES_ORIGINE = df_origine[FEATURES].median()

# Jeu de données actif : celui importé par l'administrateur s'il existe,
# sinon le jeu d'origine livré avec l'application.
df = st.session_state.get("jeu_personnalise", df_origine)
NOM_JEU = st.session_state.get("nom_jeu", "Breast_cancer.csv (jeu d'origine)")
A_CLASS = "CLASS" in df.columns

# Dans le fichier CSV : CLASS 0 = tumeur maligne, CLASS 1 = tumeur bénigne.
# Le notebook a créé la cible « maligne » = (CLASS == 0), donc dans le modèle :
# predict_proba(...)[0, 1] = probabilité que la tumeur soit MALIGNE.
SEUIL = 0.5

# ---------------------------------------------------------------------------
# Traduction des 30 mesures en français
# 10 caractéristiques × 3 statistiques (moyenne, erreur type, pire valeur)
# ---------------------------------------------------------------------------

CARACS = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave points", "symmetry", "fractal dimension",
]

FR_CARAC = {
    "radius": "Rayon",
    "texture": "Texture",
    "perimeter": "Périmètre",
    "area": "Aire",
    "smoothness": "Lissage",
    "compactness": "Compacité",
    "concavity": "Concavité",
    "concave points": "Points concaves",
    "symmetry": "Symétrie",
    "fractal dimension": "Dimension fractale",
}

AIDE_CARAC = {
    "radius": "Distance moyenne entre le centre du noyau cellulaire et son contour.",
    "texture": "Variation des niveaux de gris : rugosité de l'image du noyau.",
    "perimeter": "Longueur du contour du noyau.",
    "area": "Surface du noyau.",
    "smoothness": "Variation locale de la longueur des rayons : plus le contour est irrégulier, plus la valeur est élevée.",
    "compactness": "Périmètre² / aire − 1 : à quel point la forme du noyau est compacte.",
    "concavity": "Importance des creux présents sur le contour du noyau.",
    "concave points": "Nombre de portions concaves (creux) sur le contour.",
    "symmetry": "Degré de symétrie du noyau.",
    "fractal dimension": "Complexité du contour : plus elle est élevée, plus le bord est découpé.",
}

GROUPES = [
    {
        "nom": "Moyennes",
        "stat": "moyenne",
        "explication": "Moyenne de la caractéristique sur toutes les cellules analysées.",
        "appartient": lambda f: f.startswith("mean "),
        "caracteristique": lambda f: f[len("mean "):],
    },
    {
        "nom": "Erreurs types",
        "stat": "erreur type",
        "explication": "Erreur type : variabilité de la mesure d'une image à l'autre.",
        "appartient": lambda f: f.endswith(" error"),
        "caracteristique": lambda f: f[: -len(" error")],
    },
    {
        "nom": "Pires valeurs",
        "stat": "pire valeur",
        "explication": "Moyenne des trois valeurs les plus élevées observées (le « pire » aspect des cellules).",
        "appartient": lambda f: f.startswith("worst "),
        "caracteristique": lambda f: f[len("worst "):],
    },
]


def traduire(feature):
    """Renvoie (libellé français, texte d'aide) pour une colonne anglaise."""
    for groupe in GROUPES:
        if groupe["appartient"](feature):
            carac = groupe["caracteristique"](feature)
            libelle = f"{FR_CARAC[carac]} — {groupe['stat']}"
            explication = groupe["explication"]
            aide = f"{AIDE_CARAC[carac]} Ici : {explication[0].lower() + explication[1:]}"
            return libelle, aide
    return feature.capitalize(), f"Mesure du jeu de données : {feature}."


TRADUCTION = {f: traduire(f) for f in FEATURES}

# ---------------------------------------------------------------------------
# Valeurs de référence : médiane par défaut, bornes issues des données réelles
# ---------------------------------------------------------------------------

MEDIANES = df[FEATURES].median()
MINI = df[FEATURES].min()
MAXI = df[FEATURES].max()


def format_nombre(feature):
    """Nombre de décimales adapté à l'ordre de grandeur de la mesure."""
    vmax = float(MAXI[feature])
    if vmax < 0.01:
        return "%.6f"
    if vmax < 1:
        return "%.5f"
    if vmax < 100:
        return "%.4f"
    return "%.1f"


def en_pourcentage(x):
    return f"{100 * x:.1f}".replace(".", ",") + " %"


def mot_de_passe_admin():
    """Mot de passe de la section administration : variable d'environnement
    ADMIN_PASSWORD (utile en déploiement) ou .streamlit/secrets.toml (même
    clé), sinon « admin » par défaut."""
    valeur = os.environ.get("ADMIN_PASSWORD")
    if valeur:
        return valeur
    try:
        return st.secrets["ADMIN_PASSWORD"]
    except Exception:
        return "admin"


def charger_exemple(classe):
    """Remplit le formulaire avec un cas réel tiré au hasard (0 = malin, 1 = bénin)."""
    ligne = df[df["CLASS"] == classe].sample(1).iloc[0]
    for f in FEATURES:
        st.session_state[f] = float(ligne[f])


def charger_cas_aleatoire():
    """Remplit le formulaire avec un cas du jeu actif, sans distinction de classe."""
    ligne = df.sample(1).iloc[0]
    for f in FEATURES:
        st.session_state[f] = float(ligne[f])


def valeurs_medians():
    for f in FEATURES:
        st.session_state[f] = float(MEDIANES[f])


for f in FEATURES:
    st.session_state.setdefault(f, float(MEDIANES[f]))

# ---------------------------------------------------------------------------
# Barre latérale : présentation des données et du modèle
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("🩺 À propos")

    with st.container(border=True):
        st.subheader("Les données")
        total = len(df)
        lignes = [f"Jeu actif : **{NOM_JEU}**"]
        if A_CLASS:
            n_malignes = int((df["CLASS"] == 0).sum())
            n_benignes = int((df["CLASS"] == 1).sum())
            lignes += [
                f"**{total}** cas réels (biopsies)",
                f"**{n_malignes}** tumeurs malignes ({en_pourcentage(n_malignes / total)})",
                f"**{n_benignes}** tumeurs bénignes ({en_pourcentage(n_benignes / total)})",
            ]
        else:
            lignes += [f"**{total}** cas", "colonne CLASS absente"]
        lignes.append("**30** mesures par cas")
        st.markdown("- " + "\n- ".join(lignes))

    with st.container(border=True):
        st.subheader("Le modèle")
        st.markdown(
            """
            **Régression logistique**, entraînée dans le notebook
            `04_Cancer_du_sein.ipynb`, sélectionnée comme meilleur modèle
            d'après le ROC-AUC en validation croisée.

            Chaîne de traitement :
            1. imputation des valeurs manquantes (médiane) ;
            2. standardisation des 30 mesures ;
            3. régression logistique.

            Seuil de décision : **0,50** — au-delà, la tumeur est classée
            « maligne ».
            """
        )

    st.caption(
        "Jeu de données « Wisconsin Diagnostic Breast Cancer » "
        "(569 cas, aussi disponible dans scikit-learn)."
    )

    with st.expander("⚙️ Administration"):
        if not st.session_state.get("admin_connecte"):
            mot = st.text_input("Mot de passe administrateur", type="password", key="mdp_admin")
            if st.button("Se connecter", width="stretch"):
                if mot == mot_de_passe_admin():
                    st.session_state.admin_connecte = True
                    st.rerun()
                st.error("Mot de passe incorrect.")
        else:
            st.caption(
                "Importez un jeu de données au format du modèle (30 mesures, "
                "idéalement + CLASS) : il remplace les médianes, bornes, "
                "exemples et statistiques. Le modèle, lui, reste celui du notebook."
            )
            gabarit = (
                pd.DataFrame(columns=FEATURES + ["CLASS"]).to_csv(index=False).encode("utf-8")
            )
            st.download_button(
                "Télécharger le modèle de fichier",
                gabarit,
                file_name="modele_import_cancer_sein.csv",
                mime="text/csv",
                width="stretch",
            )
            fichier = st.file_uploader("Fichier CSV à importer", type=["csv"], key="csv_admin")
            if fichier is not None:
                if fichier.size > TAILLE_MAX:
                    st.error("Fichier trop volumineux (maximum 50 Mo).")
                else:
                    jeu, erreurs, avertissements = valider_jeu(fichier.getvalue(), FEATURES)
                    for e in erreurs:
                        st.error(e)
                    for a in avertissements:
                        st.warning(a)
                    if jeu is not None:
                        st.success(f"« {fichier.name} » est valide : {len(jeu)} lignes.")
                        if st.button(
                            "Utiliser ce jeu de données", type="primary", width="stretch"
                        ):
                            st.session_state.jeu_personnalise = jeu
                            st.session_state.nom_jeu = fichier.name
                            medianes = jeu[FEATURES].median()
                            for f in FEATURES:
                                st.session_state[f] = float(medianes[f])
                            st.rerun()
            if st.session_state.get("jeu_personnalise") is not None:
                st.divider()
                st.caption(
                    "Jeu actuellement utilisé : "
                    f"{st.session_state.get('nom_jeu', 'jeu importé')}"
                )
                if st.button("Restaurer le jeu d'origine", width="stretch"):
                    st.session_state.pop("jeu_personnalise", None)
                    st.session_state.pop("nom_jeu", None)
                    for f in FEATURES:
                        st.session_state[f] = float(MEDIANES_ORIGINE[f])
                    st.rerun()

# ---------------------------------------------------------------------------
# Page principale
# ---------------------------------------------------------------------------

st.title("🩺 Détection du cancer du sein")
#st.caption("UMEF AI Bootcamp · GalsenAI Ambassador Program")

st.warning(
    "Exercice pédagogique : cet outil ne remplace en aucun cas un diagnostic "
    "médical. Seul un professionnel de santé peut poser un diagnostic.",
    title="Avertissement",
)

st.write(
    "Ajuste les mesures des cellules observées au microscope, puis clique "
    "sur « Prédire ». Les valeurs de départ sont les médianes du jeu de données."
)

c1, c2, c3 = st.columns(3)
if A_CLASS:
    c1.button("Cas malin aléatoire", on_click=charger_exemple, args=(0,), width="stretch")
    c2.button("Cas bénin aléatoire", on_click=charger_exemple, args=(1,), width="stretch")
    c3.button("Valeurs médianes", on_click=valeurs_medians, width="stretch")
else:
    c1.button("Cas aléatoire", on_click=charger_cas_aleatoire, width="stretch")
    c2.button("Valeurs médianes", on_click=valeurs_medians, width="stretch")

# ---------------------------------------------------------------------------
# Saisie des 30 mesures, réparties en 3 onglets
# ---------------------------------------------------------------------------

onglets = st.tabs(["Moyennes", "Erreurs types", "Pires valeurs"])

for onglet, groupe in zip(onglets, GROUPES):
    with onglet:
        st.caption(groupe["explication"])
        colonnes_groupe = [f for f in FEATURES if groupe["appartient"](f)]
        gauche, droite = st.columns(2)
        for i, f in enumerate(colonnes_groupe):
            libelle, aide = TRADUCTION[f]
            (gauche if i % 2 == 0 else droite).number_input(
                libelle,
                min_value=float(MINI[f]),
                max_value=float(MAXI[f]),
                key=f,
                format=format_nombre(f),
                step=None,
                help=aide,
            )

# ---------------------------------------------------------------------------
# Prédiction, lancée par le bouton « Prédire »
# ---------------------------------------------------------------------------

st.divider()

valeurs_actuelles = tuple(float(st.session_state[f]) for f in FEATURES)

if st.button("Prédire", type="primary"):
    cas = pd.DataFrame([dict(zip(FEATURES, valeurs_actuelles))])
    st.session_state.prediction = (
        valeurs_actuelles,
        float(modele.predict_proba(cas)[0, 1]),
    )

if st.session_state.get("prediction") is None:
    st.info("Ajuste les mesures ci-dessus, puis clique sur « Prédire » pour afficher le résultat.")
else:
    valeurs_predites, p_maligne = st.session_state.prediction
    if valeurs_predites != valeurs_actuelles:
        st.info("Les mesures ont changé depuis la dernière prédiction : clique à nouveau sur « Prédire ».")
    else:
        st.metric("Probabilité de tumeur maligne", en_pourcentage(p_maligne), border=True)
        st.progress(
            min(max(p_maligne, 0.0), 1.0),
            text=en_pourcentage(p_maligne) + " de probabilité de malignité",
        )

        if p_maligne >= SEUIL:
            st.error(
                "Le modèle prédit une tumeur **MALIGNE** (probabilité ≥ 50 %). "
                "À confirmer impérativement par un médecin.",
                title="Résultat",
            )
        else:
            st.success(
                "Le modèle prédit une tumeur **BÉNIGNE** (probabilité < 50 %). "
                "À confirmer par un médecin.",
                title="Résultat",
            )

        # -------------------------------------------------------------------
        # Explication : quelles mesures poussent la prédiction ?
        # -------------------------------------------------------------------

        with st.expander("Pourquoi cette prédiction ?"):
            st.write(
                "Le modèle additionne la contribution de chaque mesure (une fois "
                "standardisée) : les barres vers la droite augmentent la probabilité "
                "de malignité, celles vers la gauche la diminuent. Voici les six "
                "mesures les plus influentes pour le cas actuel :"
            )

            scaler = modele.named_steps["prep"].named_transformers_["num"].named_steps["scaler"]
            coefficients = modele.named_steps["model"].coef_[0]
            cas_predites = pd.DataFrame([dict(zip(FEATURES, valeurs_predites))])
            valeurs = cas_predites.iloc[0].to_numpy(dtype=float)
            contributions = pd.Series(
                coefficients * (valeurs - scaler.mean_) / scaler.scale_, index=FEATURES
            )
            top = contributions.abs().nlargest(6).index

            graphique = pd.DataFrame(
                {
                    "Mesure": [TRADUCTION[f][0] for f in top],
                    "Contribution": contributions[top].round(3).to_numpy(),
                }
            )
            st.bar_chart(
                graphique,
                x="Mesure",
                y="Contribution",
                horizontal=True,
                color="#C2185B",
                alt="Contributions des six mesures les plus influentes à la prédiction",
            )
            st.caption(
                "Explication simplifiée fondée sur les coefficients de la régression "
                "logistique (mesures standardisées)."
            )

# ---------------------------------------------------------------------------
# Dictionnaire des 10 caractéristiques
# ---------------------------------------------------------------------------

with st.expander("Que mesurent ces 10 caractéristiques ?"):
    st.write(
        "Pour chaque caractéristique, le jeu de données fournit trois "
        "statistiques : la **moyenne**, l'**erreur type** (variabilité) et la "
        "**pire valeur** (moyenne des trois cellules les plus marquées)."
    )
    st.table(
        pd.DataFrame(
            {
                "Caractéristique": [FR_CARAC[c] for c in CARACS],
                "Ce qu'elle mesure": [AIDE_CARAC[c] for c in CARACS],
            }
        ),
        alt="Dictionnaire des dix caractéristiques mesurées sur les cellules",
    )
