import sys
import os


_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import time
from PyQt5.QtWidgets import QApplication, QSplashScreen
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QTimer
from config import BASE_DIR, WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT
from ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # 启动画面
    splash_path = os.path.join(BASE_DIR, "resources", "启动.png")
    pixmap = QPixmap(splash_path)
    if not pixmap.isNull():
        pixmap = pixmap.scaled(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
    splash = QSplashScreen(pixmap)
    splash.setFixedSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
    splash.show()
    app.processEvents()

    start_time = time.time()

    def load_window():
        # 创建主窗口（在启动画面显示期间加载）
        window = MainWindow()

        # 确保启动画面至少展示 2 秒
        elapsed = time.time() - start_time
        remaining = max(0, 2.0 - elapsed)

        def show_window():
            window.show()
            splash.finish(window)

        QTimer.singleShot(int(remaining * 1000), show_window)
        return window

    # 延迟到事件循环中执行，让启动画面先绘制出来
    window = [None]
    def do_load():
        window[0] = load_window()
    QTimer.singleShot(0, do_load)

    try:
        sys.exit(app.exec_())
    finally:
        if window[0]:
            window[0].close()


if __name__ == "__main__":
    main()
