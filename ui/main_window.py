from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QStatusBar,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QColor, QFont, QPainter
import qtawesome as qta

from config import BACKGROUND_IMAGE_PATH, WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT
from ui.yolo_detect import DetectionPage
from ui.ai_assist import AIAssistantPage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("农业病虫害识别系统")
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)  # ---- 主窗口最小宽度、高度 ----
        self.setWindowIcon(qta.icon('fa5s.leaf', color='#007AFF'))
        
        self.background_pixmap = QPixmap(BACKGROUND_IMAGE_PATH)
        
        self.init_ui()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        
        if not self.background_pixmap.isNull():
            scaled_pixmap = self.background_pixmap.scaled(
                self.size(), 
                Qt.KeepAspectRatioByExpanding, 
                Qt.SmoothTransformation
            )
            x = (self.width() - scaled_pixmap.width()) // 2
            y = (self.height() - scaled_pixmap.height()) // 2
            painter.drawPixmap(x, y, scaled_pixmap)
        
        painter.fillRect(self.rect(), QColor(0, 0, 0, 100))
    
    def init_ui(self):
        self.detection_page = DetectionPage()
        self.ai_page = AIAssistantPage()
        
        self.tab_widget = QTabWidget()
        self.tab_widget.setFont(QFont("SF Pro Display", 18))
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: none;
                background: transparent;
            }
            QTabBar::tab {
                padding: 12px 28px;
                margin: 6px 4px;
                color: rgba(255, 255, 255, 0.45);
                font-size: 26px;
                font-weight: 600;
                font-family: "SF Pro Display";
                background: transparent;
                border: 1.5px solid transparent;
                border-radius: 12px;
            }
            QTabBar::tab:selected {
                color: #ffffff;
                background: rgba(0, 122, 255, 0.22);
                border: 1.5px solid rgba(0, 122, 255, 0.45);
            }
            QTabBar::tab:hover:!selected {
                color: rgba(255, 255, 255, 0.82);
                background: rgba(255, 255, 255, 0.06);
                border: 1.5px solid rgba(255, 255, 255, 0.12);
            }
        """)
        
        self.tab_widget.addTab(self.detection_page, qta.icon('fa5s.search', color='#ffffff'), " 识别 ")
        self.tab_widget.addTab(self.ai_page, qta.icon('fa5s.robot', color='#ffffff'), " AI助手 ")
        
        self.setCentralWidget(self.tab_widget)
        
        self.create_status_bar()
        
        self.detection_page.status_message.connect(self.status_bar.showMessage)
        self.ai_page.status_message.connect(self.status_bar.showMessage)
        self.detection_page.record_added.connect(self.ai_page.add_detection_result)
        
        self._pages = [self.detection_page, self.ai_page]
        self._current_tab = 0
        self.tab_widget.currentChanged.connect(self._on_tab_changed)
        self.detection_page.on_enter()
    
    def create_status_bar(self):
        self.status_bar = QStatusBar()
        self.status_bar.setFont(QFont("SF Pro Display", 13))
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background: rgba(0, 0, 0, 0.2);
                color: white;
                padding: 8px;  /* ---- 状态栏内边距 ---- */
            }
        """)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("系统就绪")
    
    def _on_tab_changed(self, index):
        if index == self._current_tab:
            return
        self._pages[self._current_tab].on_leave()
        self._current_tab = index
        self._pages[index].on_enter()
    
    def closeEvent(self, event):
        for page in self._pages:
            page.on_leave()
        self.ai_page.status_message.disconnect()
        self.detection_page.cleanup()
        from service.gps import gps_service
        gps_service.stop()
        from service.soil import soil_sensor_service
        soil_sensor_service.stop()
        from service.light import light_sensor_service
        try:
            light_sensor_service.stop()
        except Exception:
            pass
        event.accept()
