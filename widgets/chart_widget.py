from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTabWidget
from PyQt5.QtWebEngineWidgets import QWebEngineView
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from config import Config


class ChartWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        """Configure l'interface du widget"""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        # Onglet Corrélations
        self.corr_tab = QWidget()
        self.corr_view = QWebEngineView()
        self.corr_layout = QVBoxLayout(self.corr_tab)
        self.corr_layout.addWidget(self.corr_view)
        self.tabs.addTab(self.corr_tab, "Corrélations")

        # Onglet Distribution
        self.dist_tab = QWidget()
        self.dist_view = QWebEngineView()
        self.dist_layout = QVBoxLayout(self.dist_tab)
        self.dist_layout.addWidget(self.dist_view)
        self.tabs.addTab(self.dist_tab, "Distributions")

        # Onglet Tendances
        self.trend_tab = QWidget()
        self.trend_view = QWebEngineView()
        self.trend_layout = QVBoxLayout(self.trend_tab)
        self.trend_layout.addWidget(self.trend_view)
        self.tabs.addTab(self.trend_tab, "Tendances")

    def update_charts(self, data):
        """Met à jour tous les graphiques avec les nouvelles données"""
        self.update_correlation_chart(data)
        self.update_distribution_charts(data)
        self.update_trend_charts(data)

    def update_correlation_chart(self, data):
        """Met à jour le graphique de corrélation"""
        if data is None or len(data.select_dtypes(include='number').columns) < 2:
            return

        corr_matrix = data.corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.columns.tolist(),
            colorscale='Viridis',
            zmin=-1,
            zmax=1
        ))

        fig.update_layout(
            title='Matrice de Corrélation',
            template=Config.CHART_THEME,
            height=500
        )

        self.corr_view.setHtml(fig.to_html(include_plotlyjs='cdn'))

    def update_distribution_charts(self, data):
        """Met à jour les graphiques de distribution"""
        if data is None:
            return

        numeric_data = data.select_dtypes(include='number')
        if len(numeric_data.columns) == 0:
            return

        # Créer un histogramme pour chaque colonne numérique
        figs = []
        for col in numeric_data.columns:
            fig = px.histogram(data, x=col, nbins=30,
                               title=f"Distribution de {col}",
                               color_discrete_sequence=[Config.CHART_COLORS[0]])
            fig.update_layout(template=Config.CHART_THEME, height=300)
            figs.append(fig)

        # Combiner tous les graphiques en un seul HTML
        html = "<html><body>"
        for fig in figs:
            html += fig.to_html(full_html=False, include_plotlyjs='cdn')
        html += "</body></html>"

        self.dist_view.setHtml(html)

    def update_trend_charts(self, data):
        """Met à jour les graphiques de tendances"""
        if data is None:
            return

        numeric_data = data.select_dtypes(include='number')
        if len(numeric_data.columns) < 2:
            return

        # Créer des graphiques de dispersion pour chaque paire de variables
        figs = []
        cols = numeric_data.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                x_col = cols[i]
                y_col = cols[j]

                fig = px.scatter(data, x=x_col, y=y_col,
                                 title=f"{y_col} vs {x_col}",
                                 trendline="ols",
                                 color_discrete_sequence=[Config.CHART_COLORS[1]])
                fig.update_layout(template=Config.CHART_THEME, height=300)
                figs.append(fig)

        # Limiter à 6 graphiques pour éviter les performances médiocres
        figs = figs[:6]

        # Combiner tous les graphiques en un seul HTML
        html = "<html><body>"
        for fig in figs:
            html += fig.to_html(full_html=False, include_plotlyjs='cdn')
        html += "</body></html>"

        self.trend_view.setHtml(html)