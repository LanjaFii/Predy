import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


class DataProcessor:
    def __init__(self, data):
        self.data = data.copy()

    def clean_data(self):
        """Nettoie et prépare les données pour l'analyse"""
        # Faire une copie pour éviter les modifications sur la donnée originale
        cleaned_data = self.data.copy()

        # Convertir les colonnes booléennes en numériques
        for col in cleaned_data.select_dtypes(include=['bool']).columns:
            cleaned_data[col] = cleaned_data[col].astype(int)

        # Remplacer les chaînes vides par NaN
        cleaned_data.replace('', np.nan, inplace=True)

        # Imputer les valeurs manquantes seulement pour les colonnes numériques
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            imputer = SimpleImputer(strategy='mean')
            cleaned_data[numeric_cols] = imputer.fit_transform(cleaned_data[numeric_cols])

        return cleaned_data

    def prepare_for_prediction(self, target_column):
        """Prépare les données pour la prédiction"""
        if target_column not in self.data.columns:
            raise ValueError(f"La colonne cible '{target_column}' n'existe pas")

        # Nettoyer les données
        cleaned_data = self.clean_data()

        # Vérifier que la cible est toujours présente après le nettoyage
        if target_column not in cleaned_data.columns:
            raise ValueError(f"La colonne cible '{target_column}' a été supprimée pendant le nettoyage")

        X = cleaned_data.drop(columns=[target_column])
        y = cleaned_data[target_column]

        return X, y