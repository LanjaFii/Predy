import os
import pandas as pd
import numpy as np
import joblib
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
                             QFileDialog, QStatusBar, QLabel, QPushButton, QMessageBox,
                             QDialog, QFormLayout, QDoubleSpinBox, QDialogButtonBox,
                             QComboBox, QSpinBox, QProgressBar, QApplication)
from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from utils.data_processor import DataProcessor
from utils.pdf_generator import generate_report
from widgets.chart_widget import ChartWidget
from widgets.stats_widget import StatsWidget
from config import Config


class PredictionDialog(QDialog):
    def __init__(self, features, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Prédire le score de bonheur")
        self.setMinimumWidth(450)

        self.layout = QFormLayout(self)

        self.feature_inputs = {}
        for feature in features:
            spinbox = self.create_appropriate_spinbox(feature)
            self.feature_inputs[feature] = spinbox
            self.layout.addRow(feature.replace('_', ' ').title(), spinbox)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addRow(self.button_box)

    def create_appropriate_spinbox(self, feature):
        """Crée le bon type de spinbox en fonction de la caractéristique"""
        if feature == 'age':
            spinbox = QSpinBox()
            spinbox.setRange(18, 100)
            spinbox.setValue(30)
        elif any(x in feature for x in ['heures', 'semaine', 'duration', 'time']):
            spinbox = QDoubleSpinBox()
            spinbox.setRange(0, 168)  # Nombre d'heures dans une semaine
            spinbox.setValue(40 if 'travail' in feature else 8)
            spinbox.setSuffix(" heures")
        elif any(x in feature for x in ['note', 'score', 'rating']):
            spinbox = QDoubleSpinBox()
            spinbox.setRange(0, 10)
            spinbox.setValue(7)
            spinbox.setDecimals(1)
        else:
            spinbox = QDoubleSpinBox()
            spinbox.setRange(0, 100)

        return spinbox

    def get_values(self):
        return {feature: spinbox.value() for feature, spinbox in self.feature_inputs.items()}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data = None
        self.current_file = None
        self.model = None
        self.pca = None
        self.scaler = None

        self.setup_ui()
        self.setup_connections()
        self.setup_prediction_ui()

    def setup_ui(self):
        """Configure l'interface utilisateur"""
        self.setWindowTitle(f"{Config.APP_NAME} {Config.APP_VERSION}")
        self.setMinimumSize(QSize(1024, 768))

        # Barre d'outils principale
        toolbar = self.addToolBar("Outils")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)

        # Actions principales
        self.import_action = toolbar.addAction(QIcon(":/icons/import.png"), "Importer")
        self.analyze_action = toolbar.addAction(QIcon(":/icons/analyze.png"), "Analyser")
        self.predict_action = toolbar.addAction(QIcon(":/icons/predict.png"), "Prédire")
        self.report_action = toolbar.addAction(QIcon(":/icons/report.png"), "Rapport")

        # Barre d'outils secondaire pour les modèles
        model_toolbar = self.addToolBar("Modèles")
        model_toolbar.setMovable(False)
        self.save_model_action = model_toolbar.addAction(QIcon(":/icons/save.png"), "Sauvegarder modèle")
        self.load_model_action = model_toolbar.addAction(QIcon(":/icons/load.png"), "Charger modèle")

        # Zone centrale
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layout principal
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)

        # Barre d'état
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("Prêt")
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.status_label)
        self.status_bar.addPermanentWidget(self.progress_bar)

        # Onglets
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Onglet Données
        self.data_tab = QWidget()
        self.data_layout = QVBoxLayout(self.data_tab)
        self.data_preview = QLabel("Aucune donnée chargée")
        self.data_preview.setAlignment(Qt.AlignCenter)
        self.data_layout.addWidget(self.data_preview)
        self.tabs.addTab(self.data_tab, "Données")

        # Onglet Analyse
        self.analysis_tab = QWidget()
        self.analysis_layout = QVBoxLayout(self.analysis_tab)
        self.tabs.addTab(self.analysis_tab, "Analyse")

        # Onglet Prédiction
        self.prediction_tab = QWidget()
        self.prediction_layout = QVBoxLayout(self.prediction_tab)
        self.tabs.addTab(self.prediction_tab, "Prédiction")

        # Widgets personnalisés
        self.chart_widget = ChartWidget()
        self.stats_widget = StatsWidget()

        # Disposition des widgets d'analyse
        analysis_top = QHBoxLayout()
        analysis_top.addWidget(self.stats_widget)
        analysis_top.addWidget(self.chart_widget)
        self.analysis_layout.addLayout(analysis_top)

        # PCA Widget
        self.pca_canvas = FigureCanvas(Figure(figsize=(5, 4)))
        self.analysis_layout.addWidget(self.pca_canvas)

    def setup_prediction_ui(self):
        """Configure l'interface de prédiction"""
        self.prediction_form = QWidget()
        self.form_layout = QFormLayout(self.prediction_form)

        self.target_combo = QComboBox()
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Régression Linéaire", "Forêt Aléatoire"])

        self.train_button = QPushButton("Entraîner le modèle")
        self.train_button.clicked.connect(self.train_model)

        self.form_layout.addRow("Variable cible:", self.target_combo)
        self.form_layout.addRow("Modèle:", self.model_combo)
        self.form_layout.addRow(self.train_button)

        self.model_info = QLabel("Aucun modèle entraîné")
        self.model_info.setWordWrap(True)

        self.prediction_layout.addWidget(self.prediction_form)
        self.prediction_layout.addWidget(self.model_info)

        self.update_prediction_ui()

    def setup_connections(self):
        """Connecte les signaux aux slots"""
        self.import_action.triggered.connect(self.import_data)
        self.analyze_action.triggered.connect(self.analyze_data)
        self.predict_action.triggered.connect(self.show_prediction_dialog)
        self.report_action.triggered.connect(self.generate_report)
        self.save_model_action.triggered.connect(self.save_model)
        self.load_model_action.triggered.connect(self.load_model)

    def import_data(self):
        """Importe un fichier de données"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Ouvrir un fichier",
            "", "Fichiers CSV (*.csv);;Tous les fichiers (*)"
        )

        if file_path:
            try:
                self.show_progress("Chargement des données...")

                # Simuler un chargement long pour la barre de progression
                QTimer.singleShot(100, lambda: self.finish_import(file_path))

            except Exception as e:
                self.hide_progress()
                QMessageBox.critical(self, "Erreur", f"Impossible de charger le fichier:\n{str(e)}")

    def finish_import(self, file_path):
        """Finalise l'importation des données"""
        try:
            self.data = pd.read_csv(file_path)
            self.current_file = os.path.basename(file_path)
            self.update_data_preview()
            self.update_prediction_ui()
            self.status_label.setText(f"Données chargées: {self.current_file}")
            self.hide_progress()
            QMessageBox.information(self, "Succès", "Données importées avec succès!")
        except Exception as e:
            self.hide_progress()
            QMessageBox.critical(self, "Erreur", f"Erreur lors du chargement:\n{str(e)}")

    def update_data_preview(self):
        """Met à jour l'aperçu des données"""
        if self.data is not None:
            preview_text = f"<b>{self.current_file}</b><br><br>"
            preview_text += f"<i>{len(self.data)} lignes, {len(self.data.columns)} colonnes</i><br><br>"
            preview_text += "<table border='1'><tr>"

            # En-têtes (5 premières colonnes)
            for col in self.data.columns[:5]:
                preview_text += f"<th>{col}</th>"
            preview_text += "</tr>"

            # Données (5 premières lignes)
            for _, row in self.data.head().iterrows():
                preview_text += "<tr>"
                for val in row[:5]:
                    preview_text += f"<td>{val}</td>"
                preview_text += "</tr>"

            preview_text += "</table>"

            if len(self.data.columns) > 5:
                preview_text += "<br><i>... et " + str(len(self.data.columns) - 5) + " colonnes supplémentaires</i>"

            self.data_preview.setText(preview_text)

    def analyze_data(self):
        """Analyse les données avec ACP et statistiques"""
        if self.data is None:
            QMessageBox.warning(self, "Avertissement", "Veuillez d'abord importer des données!")
            return

        try:
            self.show_progress("Analyse en cours...")

            # Utiliser un timer pour simuler une analyse longue
            QTimer.singleShot(100, lambda: self.finish_analysis())

        except Exception as e:
            self.hide_progress()
            QMessageBox.critical(self, "Erreur", f"Erreur lors de l'analyse:\n{str(e)}")

    def finish_analysis(self):
        """Finalise l'analyse des données"""
        try:
            # Nettoyage des données
            processor = DataProcessor(self.data)
            cleaned_data = processor.clean_data()

            # Standardisation
            self.scaler = StandardScaler()
            scaled_data = self.scaler.fit_transform(cleaned_data)

            # ACP
            self.pca = PCA(n_components=2)
            principal_components = self.pca.fit_transform(scaled_data)

            # Visualisation ACP
            self.plot_pca(principal_components)

            # Mise à jour des statistiques
            self.stats_widget.update_stats(self.data.describe())

            # Mise à jour des graphiques
            self.chart_widget.update_charts(self.data)

            self.status_label.setText("Analyse terminée avec succès")
            self.hide_progress()

        except Exception as e:
            self.hide_progress()
            QMessageBox.critical(self, "Erreur", f"Erreur lors de l'analyse:\n{str(e)}")

    def plot_pca(self, components):
        """Affiche le graphique ACP"""
        fig = self.pca_canvas.figure
        fig.clear()

        ax = fig.add_subplot(111)
        ax.scatter(components[:, 0], components[:, 1], alpha=0.5)
        ax.set_xlabel('Composante Principale 1')
        ax.set_ylabel('Composante Principale 2')
        ax.set_title('Analyse en Composantes Principales (ACP)')

        self.pca_canvas.draw()

    def update_prediction_ui(self):
        """Met à jour l'interface de prédiction"""
        if self.data is not None:
            self.target_combo.clear()
            self.target_combo.addItems(self.data.columns)

            # Sélectionner automatiquement 'bonheur_score' s'il existe
            if 'bonheur_score' in self.data.columns:
                self.target_combo.setCurrentText('bonheur_score')

    def train_model(self):
        """Entraîne le modèle de prédiction"""
        if self.data is None:
            QMessageBox.warning(self, "Avertissement", "Veuillez d'abord importer des données!")
            return

        target = self.target_combo.currentText()
        model_type = self.model_combo.currentText()

        try:
            self.show_progress("Entraînement du modèle...")

            # Utiliser un timer pour simuler un entraînement long
            QTimer.singleShot(100, lambda: self.finish_training(target, model_type))

        except Exception as e:
            self.hide_progress()
            QMessageBox.critical(self, "Erreur", f"Erreur lors de l'entraînement:\n{str(e)}")

    def finish_training(self, target, model_type):
        """Finalise l'entraînement du modèle"""
        try:
            processor = DataProcessor(self.data)
            X, y = processor.prepare_for_prediction(target)

            # Diviser les données
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Standardiser les données
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Entraîner le modèle
            if model_type == "Régression Linéaire":
                self.model = LinearRegression()
            else:
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)

            self.model.fit(X_train_scaled, y_train)

            # Évaluer le modèle
            y_pred = self.model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Afficher les résultats
            info_text = (f"<b>Modèle {model_type} entraîné</b><br>"
                         f"Variable cible: {target}<br>"
                         f"Précision (MAE): {mae:.2f}<br>"
                         f"Score R²: {r2:.2f}<br><br>"
                         f"<i>Cliquez sur 'Prédire' pour faire une nouvelle prédiction</i>")

            self.model_info.setText(info_text)
            self.status_label.setText(f"Modèle {model_type} entraîné avec succès")
            self.hide_progress()

        except ValueError as ve:
            self.hide_progress()
            QMessageBox.warning(self, "Erreur de configuration", str(ve))
        except Exception as e:
            self.hide_progress()
            QMessageBox.critical(self, "Erreur", f"Erreur lors de l'entraînement:\n{str(e)}")
            self.model = None

    def show_prediction_dialog(self):
        """Affiche la boîte de dialogue de prédiction"""
        if self.model is None:
            QMessageBox.warning(self, "Avertissement", "Veuillez d'abord entraîner un modèle!")
            return

        # Obtenir les caractéristiques utilisées pour l'entraînement
        features = [col for col in self.data.columns if col != self.target_combo.currentText()]

        dialog = PredictionDialog(features, self)
        if dialog.exec_() == QDialog.Accepted:
            input_values = dialog.get_values()

            try:
                # Préparer les données d'entrée
                input_df = pd.DataFrame([input_values])
                input_scaled = self.scaler.transform(input_df)

                # Faire la prédiction
                prediction = self.model.predict(input_scaled)[0]

                # Afficher le résultat
                target = self.target_combo.currentText()
                result_text = (f"<b>Résultat de la prédiction:</b><br><br>"
                               f"Score de {target} prédit: <b>{prediction:.2f}</b><br><br>"
                               f"<i>Variables utilisées:</i><br>")

                for feature, value in input_values.items():
                    result_text += f"{feature.replace('_', ' ').title()}: {value}<br>"

                QMessageBox.information(self, "Résultat de la prédiction", result_text)

            except Exception as e:
                QMessageBox.critical(self, "Erreur", f"Erreur lors de la prédiction:\n{str(e)}")

    def generate_report(self):
        """Génère un rapport PDF"""
        if self.data is None:
            QMessageBox.warning(self, "Avertissement", "Veuillez d'abord importer des données!")
            return

        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Enregistrer le rapport",
                str(Config.REPORTS_DIR / "rapport_predy.pdf"),
                "Fichiers PDF (*.pdf)"
            )

            if file_path:
                self.show_progress("Génération du rapport...")
                QTimer.singleShot(100, lambda: self.finish_report_generation(file_path))

        except Exception as e:
            self.hide_progress()
            QMessageBox.critical(self, "Erreur", f"Erreur lors de la génération du rapport:\n{str(e)}")

    def finish_report_generation(self, file_path):
        """Finalise la génération du rapport"""
        try:
            generate_report(file_path, self.data, self.pca, self.model)
            self.hide_progress()
            QMessageBox.information(self, "Succès", f"Rapport généré:\n{file_path}")
        except Exception as e:
            self.hide_progress()
            QMessageBox.critical(self, "Erreur", f"Erreur lors de la génération du rapport:\n{str(e)}")

    def save_model(self):
        """Sauvegarde le modèle entraîné"""
        if self.model is None:
            QMessageBox.warning(self, "Avertissement", "Aucun modèle à sauvegarder!")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Sauvegarder le modèle",
            str(Config.MODELS_DIR / "model_predy.pkl"),
            "Fichiers Pickle (*.pkl)"
        )

        if file_path:
            try:
                self.show_progress("Sauvegarde du modèle...")
                QTimer.singleShot(100, lambda: self.finish_save_model(file_path))

            except Exception as e:
                self.hide_progress()
                QMessageBox.critical(self, "Erreur", f"Erreur lors de la sauvegarde:\n{str(e)}")

    def finish_save_model(self, file_path):
        """Finalise la sauvegarde du modèle"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'target': self.target_combo.currentText(),
                'features': [col for col in self.data.columns if col != self.target_combo.currentText()]
            }

            joblib.dump(model_data, file_path)
            self.hide_progress()
            QMessageBox.information(self, "Succès", "Modèle sauvegardé avec succès!")
        except Exception as e:
            self.hide_progress()
            QMessageBox.critical(self, "Erreur", f"Erreur lors de la sauvegarde:\n{str(e)}")

    def load_model(self):
        """Charge un modèle entraîné"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Charger un modèle",
            str(Config.MODELS_DIR),
            "Fichiers Pickle (*.pkl)"
        )

        if file_path:
            try:
                self.show_progress("Chargement du modèle...")
                QTimer.singleShot(100, lambda: self.finish_load_model(file_path))

            except Exception as e:
                self.hide_progress()
                QMessageBox.critical(self, "Erreur", f"Erreur lors du chargement:\n{str(e)}")

    def finish_load_model(self, file_path):
        """Finalise le chargement du modèle"""
        try:
            model_data = joblib.load(file_path)

            self.model = model_data['model']
            self.scaler = model_data['scaler']

            # Mettre à jour l'interface
            self.target_combo.setCurrentText(model_data['target'])
            self.model_info.setText(
                f"<b>Modèle {model_data['model'].__class__.__name__} chargé</b><br>"
                f"Variable cible: {model_data['target']}<br>"
                f"<i>Prêt à faire des prédictions</i>"
            )

            self.hide_progress()
            QMessageBox.information(self, "Succès", "Modèle chargé avec succès!")
        except Exception as e:
            self.hide_progress()
            QMessageBox.critical(self, "Erreur", f"Erreur lors du chargement:\n{str(e)}")
            self.model = None

    def show_progress(self, message):
        """Affiche la barre de progression"""
        self.status_label.setText(message)
        self.progress_bar.setRange(0, 0)  # Mode indéterminé
        self.progress_bar.setVisible(True)
        QApplication.processEvents()

    def hide_progress(self):
        """Cache la barre de progression"""
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 1)  # Réinitialiser