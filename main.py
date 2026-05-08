import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QStyleFactory
from ui.main_window import MainWindow


def main():
    # Retina / HiDPI support
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)

    # Force cross-platform Fusion style so custom QSS works identically on macOS
    app.setStyle(QStyleFactory.create("Fusion"))

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
