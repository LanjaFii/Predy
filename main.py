import sys
from PyQt5.QtWidgets import QApplication
from main_window import MainWindow


def main():
    # Créer l'application
    app = QApplication(sys.argv)

    # Configurer l'application
    app.setApplicationName(Config.APP_NAME)
    app.setApplicationVersion(Config.APP_VERSION)
    app.setOrganizationName(Config.COMPANY_NAME)

    # Charger le style
    stylesheet = Config.load_stylesheet()
    if stylesheet:
        app.setStyleSheet(stylesheet)

    # Créer et afficher la fenêtre principale
    window = MainWindow()
    window.showMaximized()

    # Exécuter l'application
    sys.exit(app.exec_())


if __name__ == "__main__":
    from config import Config

    main()