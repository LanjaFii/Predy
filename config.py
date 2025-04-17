import os
from pathlib import Path


class Config:
    # Chemins des dossiers
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "Predy/data"
    IMPORTED_DATA_DIR = DATA_DIR / "imported_data"
    MODELS_DIR = BASE_DIR / "Predy/models"
    REPORTS_DIR = BASE_DIR / "Predy/reports"
    STYLES_DIR = BASE_DIR / "Predy/styles"
    RESOURCES_DIR = BASE_DIR / "Predy/resources"


    # Créer les dossiers s'ils n'existent pas
    for dir_path in [DATA_DIR, IMPORTED_DATA_DIR, MODELS_DIR, REPORTS_DIR, STYLES_DIR, RESOURCES_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Fichiers par défaut
    SAMPLE_DATA = DATA_DIR / "sample_data.csv"
    STYLE_FILE = STYLES_DIR / "style.qss"

    # Paramètres de l'application
    APP_NAME = "Predy - Prédiction du Bonheur"
    APP_VERSION = "1.0.0"
    COMPANY_NAME = "Lanja Fii"

    # Paramètres des graphiques
    CHART_THEME = "plotly_white"
    CHART_COLORS = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"]

    @staticmethod
    def load_stylesheet():
        """Charge la feuille de style QSS"""
        if Config.STYLE_FILE.exists():
            with open(Config.STYLE_FILE, "r") as f:
                return f.read()
        return ""

    @staticmethod
    def get_icon_path(icon_name):
        """Retourne le chemin complet vers une icône"""
        icon_path = Config.RESOURCES_DIR / 'icons' / f'{icon_name}.png'
        if not icon_path.exists():
            raise FileNotFoundError(f"Icône introuvable: {icon_path}")
        return str(icon_path)