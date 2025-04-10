from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem
from PyQt5.QtCore import Qt
import pandas as pd
from config import Config


class StatsWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        """Configure l'interface du widget"""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)

        self.title_label = QLabel("Statistiques Descriptives")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2C3E50;")
        self.layout.addWidget(self.title_label)

        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #BDC3C7;
                border-radius: 3px;
            }
            QHeaderView::section {
                background-color: #3498DB;
                color: white;
                padding: 4px;
                border: 1px solid #2980B9;
            }
        """)
        self.layout.addWidget(self.table)

    def update_stats(self, stats_data):
        """Met à jour le tableau avec les statistiques descriptives"""
        if stats_data is None:
            return

        self.table.clear()

        # Configurer le tableau
        self.table.setRowCount(len(stats_data))
        self.table.setColumnCount(len(stats_data.columns))
        self.table.setHorizontalHeaderLabels(stats_data.columns.tolist())
        self.table.setVerticalHeaderLabels(stats_data.index.tolist())

        # Remplir le tableau avec les données
        for i in range(len(stats_data)):
            for j in range(len(stats_data.columns)):
                value = stats_data.iloc[i, j]
                item = QTableWidgetItem(f"{value:.2f}" if isinstance(value, (float, int)) else str(value))
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(i, j, item)

        # Ajuster la taille des colonnes
        self.table.resizeColumnsToContents()