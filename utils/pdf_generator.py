from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
import io
import pandas as pd
from config import Config


def generate_report(file_path, data, pca=None, model=None):
    """Génère un rapport PDF avec les résultats d'analyse"""
    doc = SimpleDocTemplate(file_path, pagesize=letter)
    styles = getSampleStyleSheet()

    # Style personnalisé
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=18,
        alignment=1,
        spaceAfter=20,
        textColor=colors.HexColor("#2C3E50")
    )

    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Heading2'],
        fontSize=12,
        alignment=1,
        spaceAfter=15,
        textColor=colors.HexColor("#7F8C8D")
    )

    # Contenu du rapport
    content = []

    # Titre
    content.append(Paragraph(Config.APP_NAME, title_style))
    content.append(Paragraph("Rapport d'analyse du bien-être", subtitle_style))

    # Informations sur les données
    content.append(Paragraph("<b>Informations sur les données</b>", styles['Heading2']))
    content.append(Spacer(1, 12))

    data_info = [
        ["Nombre d'observations", len(data)],
        ["Nombre de variables", len(data.columns)],
        ["Variables", ", ".join(data.columns)]
    ]

    info_table = Table(data_info, colWidths=[2 * inch, 4 * inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#3498DB")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#EBF5FB")),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor("#D6EAF8"))
    ]))

    content.append(info_table)
    content.append(Spacer(1, 20))

    # Statistiques descriptives
    content.append(Paragraph("<b>Statistiques descriptives</b>", styles['Heading2']))
    content.append(Spacer(1, 12))

    # Créer un tableau avec les statistiques descriptives
    desc_stats = data.describe().reset_index()
    desc_table = Table([desc_stats.columns.tolist()] + desc_stats.values.tolist())

    desc_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#3498DB")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#EBF5FB")),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor("#D6EAF8")),
        ('FONTSIZE', (0, 1), (-1, -1), 8)
    ]))

    content.append(desc_table)
    content.append(Spacer(1, 20))

    # Graphique ACP (si disponible)
    if pca is not None:
        content.append(Paragraph("<b>Analyse en Composantes Principales</b>", styles['Heading2']))
        content.append(Spacer(1, 12))

        # Créer un graphique ACP
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(pca.components_[0], pca.components_[1], alpha=0.5)
        ax.set_xlabel('Composante Principale 1')
        ax.set_ylabel('Composante Principale 2')
        ax.set_title('Analyse en Composantes Principales')

        # Convertir en image pour le PDF
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        content.append(Image(img_data, width=5 * inch, height=3.5 * inch))
        content.append(Spacer(1, 20))

    # Informations sur le modèle (si disponible)
    if model is not None:
        content.append(Paragraph("<b>Modèle de Prédiction</b>", styles['Heading2']))
        content.append(Spacer(1, 12))

        model_info = [
            ["Type de modèle", model.__class__.__name__],
            ["Paramètres", str(model.get_params())],
        ]

        model_table = Table(model_info, colWidths=[2 * inch, 4 * inch])
        model_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#3498DB")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#EBF5FB")),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor("#D6EAF8"))
        ]))

        content.append(model_table)
        content.append(Spacer(1, 20))

    # Construire le document
    doc.build(content)