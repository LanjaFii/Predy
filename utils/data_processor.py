import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    def __init__(self, data):
        self.data = data.copy()

    def validate(self):
        """Valide les données avant traitement"""
        errors = []
        if self.data.empty:
            errors.append("Le DataFrame est vide")

        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            errors.append("Aucune colonne numérique trouvée")

        return errors if errors else None

    def clean_data(self):
        """Nettoie et prépare les données pour l'analyse"""
        # Validation initiale
        if (errors := self.validate()):
            raise ValueError("\n".join(errors))

        # Conversion des booléens en numériques
        cleaned_data = self.data.copy()
        for col in cleaned_data.select_dtypes(include=['bool']).columns:
            cleaned_data[col] = cleaned_data[col].astype(int)

        # Imputation des valeurs manquantes
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy='mean')
        cleaned_data[numeric_cols] = imputer.fit_transform(cleaned_data[numeric_cols])

        return cleaned_data

    def prepare_for_prediction(self, target_column):
        """Prépare les données pour la prédiction"""
        cleaned_data = self.clean_data()

        if target_column not in cleaned_data.columns:
            raise ValueError(f"Colonne cible '{target_column}' non trouvée")

        X = cleaned_data.drop(columns=[target_column])
        y = cleaned_data[target_column]

        return X, y