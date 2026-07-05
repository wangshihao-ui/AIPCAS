import threading
from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel,
    QPushButton, QComboBox, QTextEdit, QLineEdit,
    QScrollArea, QGraphicsDropShadowEffect, QSizePolicy,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QColor
import qtawesome as qta
from config import ENABLE_EFFECTS, SOIL_SENSOR_PORT, SOIL_SENSOR_BAUD, SOIL_SENSOR_ADDR
from service.voice import voice_service
from service.ai import ai_service
from service.gps import gps_service
from service.soil import soil_sensor_service
from service.light import light_sensor_service
from utils.record_manager import load_records as rm_load_records



# 字体大小常量 —— 修改此处可统一调整全页字体

# ---- AI助手页----
FONT_SIZE_TITLE_LARGE   = 20   # 页面大标题
FONT_SIZE_TITLE_MEDIUM  = 20   # 区域标题（如"土壤数据"）
FONT_SIZE_TITLE_SMALL   = 20   # 小标题（如"AI原因分析"、"智能问答"）
FONT_SIZE_BODY          = 13   # 正文、输入框、下拉框
FONT_SIZE_VALUE         = 16   # 土壤数据数值
FONT_SIZE_CARD_TITLE    = 16   # 土壤数据卡片指标名称
FONT_SIZE_BTN           = 14   # 按钮文字
FONT_SIZE_ICON_SM       = 18   # 小图标尺寸（如按钮图标）
FONT_SIZE_ICON_XS       = 16   # 超小图标尺寸
FONT_SIZE_TOPBAR        = 13   # 顶栏时间/地点文字


BG_CARD_LIGHT           = "rgba(255, 255, 255, 0.12)"  # 浅色卡片背景（左右栏容器）
BG_CARD_DARK            = "rgba(0, 0, 0, 0.3)"       # 深色卡片背景（AI分析、聊天区）
BG_COMBOBOX             = "rgba(255, 255, 255, 0.12)" # 下拉框背景
BG_COMBOBOX_LIST        = "rgba(40, 40, 60, 0.9)"    # 下拉框列表背景
BG_SELECTION            = "rgba(0, 122, 255, 0.35)"   # 下拉框选中项背景
BRD_CARD_LIGHT          = "rgba(255, 255, 255, 0.18)" # 浅色边框
BRD_CARD_DARK           = "rgba(255, 255, 255, 0.15)" # 深色边框
BRD_INPUT               = "rgba(255, 255, 255, 0.25)" # 输入框边框
BRD_HR                  = "rgba(255, 255, 255, 0.1)"  # 分隔线
COLOR_SOIL_TITLE        = "rgba(255, 255, 255, 0.8)"  # 土壤数据标题颜色（左侧8个汉字）


# 样式模板

def _card_style(radius):
    return f"""
    QFrame {{
        background: {BG_CARD_LIGHT};
        border: 1px solid {BRD_CARD_LIGHT};
        border-radius: {radius}px;
    }}
"""

SECTION_TITLE = """
    QLabel {{
        color: rgba(255, 255, 255, 0.9);
        font-size: {size}px;
        font-weight: bold;
        font-family: "SF Pro Display";
        padding: {pad}px;
    }}
"""


def _shadow(blur=25, y=8):
    if not ENABLE_EFFECTS:
        return None
    s = QGraphicsDropShadowEffect()
    s.setBlurRadius(blur)
    s.setXOffset(0)
    s.setYOffset(y)
    s.setColor(QColor(0, 0, 0, 60))
    return s


def _bg(c1, c2):
    """RK3588 等嵌入式设备使用纯色，其他平台使用渐变"""
    if ENABLE_EFFECTS:
        return f"qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {c1}, stop:1 {c2})"
    return c1


class SoilCard(QFrame):
    def __init__(self, icon_name, title, value, unit, color, parent=None):
        super().__init__(parent)
        self._unit = unit
        self._color = color
        self.setStyleSheet("""
            QFrame {
                background: transparent;
                border: none;
            }
        """)
        self.setGraphicsEffect(None)
        self.setMinimumHeight(40)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 4, 10, 4)
        lay.setSpacing(10)

        icon_lbl = QLabel()
        icon_lbl.setPixmap(qta.icon(icon_name, color=color).pixmap(22, 22))
        icon_lbl.setFixedWidth(30)
        lay.addWidget(icon_lbl)

        title_lbl = QLabel(title)
        title_lbl.setFont(QFont("SF Pro Display", FONT_SIZE_CARD_TITLE))
        title_lbl.setStyleSheet(f"color: {COLOR_SOIL_TITLE}; border:none;")
        title_lbl.setFixedWidth(100)
        lay.addWidget(title_lbl)

        val_layout = QHBoxLayout()
        val_layout.setSpacing(6)
        
        val_lbl = QLabel(f"{value}")
        val_lbl.setFont(QFont("SF Pro Display", FONT_SIZE_VALUE, QFont.Bold))
        val_lbl.setStyleSheet(f"color: {color}; border:none;")
        val_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.val_label = val_lbl
        val_layout.addWidget(val_lbl)
        
        unit_lbl = QLabel(unit)
        unit_lbl.setFont(QFont("SF Pro Display", FONT_SIZE_CARD_TITLE))
        unit_lbl.setStyleSheet(f"color: {COLOR_SOIL_TITLE}; border:none;")
        unit_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        val_layout.addWidget(unit_lbl)
        
        lay.addLayout(val_layout)
        lay.addStretch()

    def update_value(self, value, fmt="{:.1f}"):
        if isinstance(value, (int, float)):
            if fmt.endswith("d}"):
                self.val_label.setText(f"{int(value)}")
            else:
                self.val_label.setText(fmt.format(value))
        else:
            self.val_label.setText(str(value))


class AIAssistantPage(QWidget):
    status_message = pyqtSignal(str)
    analysis_done = pyqtSignal(str)
    chat_done = pyqtSignal(str)
    soil_data_received = pyqtSignal(dict)
    light_data_received = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.chat_history = []
        self.soil_data = {}
        self.light_data = {}
        self.location = "长春"
        self.soil_data_loaded = False
        self.light_data_loaded = False
        self._analyzing = False
        self._chatting = False
        self._soil_cards = {}
        self._soil_diag_timer = None    # 土壤传感器诊断定时器
        self._soil_was_connected = False  # 离开时记录连接状态，进入时自动恢复
        self.analysis_done.connect(self._on_analysis_done)
        self.chat_done.connect(self._on_chat_done)
        self.soil_data_received.connect(self._update_soil_ui)
        self.light_data_received.connect(self._update_light_ui)
        self.init_ui()
        self._start_clock()

        soil_sensor_service.add_callback(self._on_soil_data)
        light_sensor_service.add_callback(self._on_light_data)

        # --- Restore persisted records into combo ---
        persisted = rm_load_records()
        if persisted:
            if self.result_combo.count() == 1 and self.result_combo.itemText(0) == "暂无识别记录":
                self.result_combo.removeItem(0)
            for rec in reversed(persisted):
                display = f"{rec.get('type', '')} - {rec.get('time', '')}"
                self.result_combo.addItem(display, rec.get("path", ""))

    def init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(12)

        root.addWidget(self._build_top_bar())

        body = QHBoxLayout()
        body.setSpacing(12)
        body.addWidget(self._build_left_panel(), 2)
        body.addWidget(self._build_mid_panel(), 3)
        body.addWidget(self._build_right_panel(), 3)
        root.addLayout(body, 1)

    # ─── 顶栏 ────────────────────────────────────────────
    def _build_top_bar(self):
        bar = QFrame()
        bar.setStyleSheet("background: transparent; border: none;")
        bar.setGraphicsEffect(None)
        bar.setFixedHeight(48)

        lay = QHBoxLayout(bar)
        lay.setContentsMargins(20, 6, 20, 6)

        loc_icon = QLabel()
        loc_icon.setPixmap(qta.icon('fa5s.map-marker-alt', color='#34C759').pixmap(20, 20))
        lay.addWidget(loc_icon)

        self.loc_label = QLabel("")
        self.loc_label.setFont(QFont("SF Pro Display", FONT_SIZE_TOPBAR))
        self.loc_label.setStyleSheet("color: rgba(255,255,255,0.85); border:none;")
        lay.addWidget(self.loc_label)

        lay.addStretch()

        time_icon = QLabel()
        time_icon.setPixmap(qta.icon('fa5s.clock', color='#007AFF').pixmap(20, 20))
        lay.addWidget(time_icon)

        self.time_label = QLabel("")
        self.time_label.setFont(QFont("SF Pro Display", FONT_SIZE_TOPBAR))
        self.time_label.setStyleSheet("color: rgba(255,255,255,0.85); border:none;")
        lay.addWidget(self.time_label)

        return bar

    def _start_clock(self):
        self._tick()
        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._tick)
        self._clock_timer.start(1000)
        self._fetch_location()

    def _tick(self):
        self.time_label.setText(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def _fetch_location(self):
        # 启动GPS服务
        gps_service.start()

        # 定时更新GPS位置
        self._gps_timer = QTimer(self)
        self._gps_timer.timeout.connect(self._update_gps_location)
        self._gps_timer.start(2000)  # 每2秒更新一次

    def _update_gps_location(self):
        loc = gps_service.get_location()
        if loc:
            lat, lon, alt = loc
            text = f"纬度:{lat:.4f}   经度:{lon:.4f}   海拔:{alt}m"
            self.loc_label.setText(text)
            self.location = loc  # 保存为元组，供AI分析使用

    # ─── 左栏：土壤数据 ──────────────────────────────────
    def _build_left_panel(self):
        container = QFrame()
        container.setStyleSheet(_card_style(radius=18))
        container.setGraphicsEffect(_shadow(30, 10))

        lay = QVBoxLayout(container)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(6)

        title = QLabel("土壤数据")
        title.setStyleSheet(SECTION_TITLE.format(size=FONT_SIZE_TITLE_MEDIUM, pad=8) + f"""
            background: {_bg('#007AFF', '#5856D6')};
            border-radius: 10px;
        """)
        title.setAlignment(Qt.AlignCenter)
        lay.addWidget(title)

        # 连接控制行
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(6)

        self._soil_status_label = QLabel("未连接")
        self._soil_status_label.setFont(QFont("SF Pro Display", FONT_SIZE_BODY))
        self._soil_status_label.setStyleSheet("color: #FF9500; border:none;")
        ctrl_row.addWidget(self._soil_status_label)

        ctrl_row.addStretch()

        self._soil_connect_btn = QPushButton("  连接")
        self._soil_connect_btn.setIcon(qta.icon('fa5s.plug', color='white'))
        self._soil_connect_btn.setIconSize(QSize(FONT_SIZE_ICON_SM, FONT_SIZE_ICON_SM))
        self._soil_connect_btn.setCursor(Qt.PointingHandCursor)
        self._soil_connect_btn.setMinimumHeight(30)
        self._soil_connect_btn.setStyleSheet(f"""
            QPushButton {{
                padding: 4px 12px;
                border: none;
                border-radius: 8px;
                font-size: {FONT_SIZE_BTN}px;
                font-weight: bold;
                color: white;
                font-family: "SF Pro Display";
                background: {_bg('#34C759', '#30D158')};
            }}
            QPushButton:hover {{ opacity: 0.85; }}
        """)
        self._soil_connect_btn.clicked.connect(self._toggle_soil_sensor)
        ctrl_row.addWidget(self._soil_connect_btn)

        lay.addLayout(ctrl_row)

        soil_items = [
            ('fa5s.thermometer-half','温度',   '0',  '°C',    '#FF9500'),
            ('fa5s.tint',            '湿度',   '0',  '%',     '#5AC8FA'),
            ('fa5s.bolt',            '电导率', '0',  'μS/cm', '#FFCC00'),
            ('fa5s.cube',            '盐分',   '0',  'mg/L',  '#FF6B8A'),
            ('fa5s.seedling',        '氮含量', '0',  'mg/kg', '#5AC8FA'),
            ('fa5s.leaf',            '磷含量', '0',  'mg/kg', '#FF3B30'),
            ('fa5s.tree',            '钾含量', '0',  'mg/kg', '#30D158'),
        ]
        for icon, name, val, unit, color in soil_items:
            self.soil_data[name] = f"{val}{unit}"
            card = SoilCard(icon, name, val, unit, color)
            self._soil_cards[name] = card
            lay.addWidget(card, 1)

        # 光照强度卡片
        lay.addSpacing(8)
        light_title = QLabel("环境光照")
        light_title.setStyleSheet(SECTION_TITLE.format(size=FONT_SIZE_TITLE_MEDIUM, pad=8) + f"""
            background: {_bg('#FF9500', '#FFCC00')};
            border-radius: 10px;
        """)
        light_title.setAlignment(Qt.AlignCenter)
        lay.addWidget(light_title)

        self._light_card = SoilCard('fa5s.sun', '光照强度', '0', 'W/m²', '#FFCC00')
        self._light_voltage_card = SoilCard('fa5s.bolt', '面板电压', '0', 'V', '#FF9500')
        lay.addWidget(self._light_card, 1)
        lay.addWidget(self._light_voltage_card, 1)

        return container

    # ─── 土壤传感器控制 ─────────────────────────────────
    def _toggle_soil_sensor(self):
        if soil_sensor_service.is_connected():
            soil_sensor_service.disconnect()
            self._stop_soil_diag_timer()
            self._soil_connect_btn.setText("  连接")
            self._soil_connect_btn.setIcon(qta.icon('fa5s.plug', color='white'))
            self._soil_connect_btn.setStyleSheet(f"""
                QPushButton {{
                    padding: 4px 12px;
                    border: none;
                    border-radius: 8px;
                    font-size: {FONT_SIZE_BTN}px;
                    font-weight: bold;
                    color: white;
                    font-family: "SF Pro Display";
                    background: {_bg('#34C759', '#30D158')};
                }}
                QPushButton:hover {{ opacity: 0.85; }}
            """)
            self._soil_status_label.setText("未连接")
            self._soil_status_label.setStyleSheet("color: #FF9500; border:none;")
            self.soil_data_loaded = False
            try:
                light_sensor_service.stop()
            except Exception:
                pass
            self.status_message.emit("土壤传感器已断开，光照传感器已停止")
        else:
            try:
                soil_sensor_service.connect()
                self._soil_connect_btn.setText("  断开")
                self._soil_connect_btn.setIcon(qta.icon('fa5s.unlink', color='white'))
                self._soil_connect_btn.setStyleSheet(f"""
                    QPushButton {{
                        padding: 4px 12px;
                        border: none;
                        border-radius: 8px;
                        font-size: {FONT_SIZE_BTN}px;
                        font-weight: bold;
                        color: white;
                        font-family: "SF Pro Display";
                        background: {_bg('#FF3B30', '#FF453A')};
                    }}
                    QPushButton:hover {{ opacity: 0.85; }}
                """)
                # 显示串口连接信息
                stats = soil_sensor_service.get_stats()
                self._soil_status_label.setText(f"已连接 {stats['port']}")
                self._soil_status_label.setStyleSheet("color: #34C759; border:none;")
                self._start_soil_diag_timer()
                try:
                    light_sensor_service.start()
                    self.status_message.emit(f"土壤({stats['port']})已连接，光照传感器已启动")
                except Exception as e2:
                    self.status_message.emit(f"土壤({stats['port']})已连接，光照启动失败: {e2}")
            except Exception as e:
                self.status_message.emit(f"土壤传感器连接失败: {e}")

    def _start_soil_diag_timer(self):
        """启动诊断定时器，检测土壤数据是否正常接收"""
        if self._soil_diag_timer is not None:
            return
        self._soil_diag_timer = QTimer(self)
        self._soil_diag_timer.timeout.connect(self._check_soil_status)
        self._soil_diag_timer.start(3000)  # 每3秒检查一次

    def _stop_soil_diag_timer(self):
        if self._soil_diag_timer is not None:
            self._soil_diag_timer.stop()
            self._soil_diag_timer = None

    def _check_soil_status(self):
        """检查土壤传感器数据是否正常，给出诊断提示"""
        if not soil_sensor_service.is_connected():
            self._stop_soil_diag_timer()
            return
        stats = soil_sensor_service.get_stats()
        if stats["read_count"] > 0:
            # 有数据，恢复正常显示
            self._soil_status_label.setText(f"已连接 {stats['port']} ({stats['read_count']}次)")
            self._soil_status_label.setStyleSheet("color: #34C759; border:none;")
            self.soil_data_loaded = True
            return
        # 一直没有数据
        fail = stats["fail_count"]
        err = stats["last_error"]
        if fail >= 3:
            detail = err if err else "无响应"
            self._soil_status_label.setText(f"无数据: {detail}")
            self._soil_status_label.setStyleSheet("color: #FF3B30; border:none;")

    def _on_soil_data(self, data):
        self.soil_data_received.emit(data)

    def _on_light_data(self, data):
        self.light_data_received.emit(data)

    def _update_light_ui(self, data):
        self.light_data_loaded = True
        irradiance = data.get("irradiance", 0)
        panel_voltage = data.get("panel_voltage", 0)
        self._light_card.update_value(irradiance, fmt="{:.1f}")
        self._light_voltage_card.update_value(panel_voltage, fmt="{:.3f}")
        self.light_data["光照强度"] = f"{irradiance:.1f}W/m²"
        self.light_data["面板电压"] = f"{panel_voltage:.3f}V"

    def _update_soil_ui(self, data):
        self.soil_data_loaded = True
        from service.soil import REGISTERS
        fmt_map = {r["name"]: r["fmt"] for r in REGISTERS}
        unit_map = {r["name"]: r["unit"] for r in REGISTERS}
        for name, val in data.items():
            if name in self._soil_cards:
                card = self._soil_cards[name]
                fmt = fmt_map.get(name, ".1f")
                if fmt == "d":
                    card.update_value(val, fmt="{:d}")
                else:
                    card.update_value(val, fmt="{:" + fmt + "}")
            unit = unit_map.get(name, "")
            if isinstance(val, float):
                self.soil_data[name] = f"{val:.1f}{unit}"
            else:
                self.soil_data[name] = f"{val}{unit}"

    # ─── 中栏：识别结果 + AI 分析 ─────────────────────────
    def _build_mid_panel(self):
        container = QFrame()
        container.setStyleSheet("background: transparent; border: none;")
        container.setGraphicsEffect(None)

        lay = QVBoxLayout(container)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(10)

        sel_row = QHBoxLayout()
        sel_row.setSpacing(8)

        sel_label = QLabel("识别结果：")
        sel_label.setFont(QFont("SF Pro Display", FONT_SIZE_BODY, QFont.Bold))
        sel_label.setStyleSheet("color: rgba(255,255,255,0.85); border:none;")
        sel_row.addWidget(sel_label)

        self.result_combo = QComboBox()
        self.result_combo.setFont(QFont("SF Pro Display", FONT_SIZE_BODY))
        self.result_combo.addItem("暂无识别记录")
        self.result_combo.setStyleSheet(f"""
            QComboBox {{
                padding: 6px 12px;
                border: 1px solid {BRD_INPUT};
                border-radius: 8px;
                background: {BG_COMBOBOX};
                color: white;
                min-width: 180px;
            }}
            QComboBox::drop-down {{ border:none; width:22px; }}
            QComboBox QAbstractItemView {{
                background: {BG_COMBOBOX_LIST};
                border: 1px solid {BRD_CARD_DARK};
                border-radius: 8px;
                color: white;
                selection-background-color: {BG_SELECTION};
                padding: 4px;
            }}
        """)
        sel_row.addWidget(self.result_combo, 1)
        lay.addLayout(sel_row)

        analysis_title = QLabel("AI 原因分析及建议")
        analysis_title.setStyleSheet(SECTION_TITLE.format(size=FONT_SIZE_TITLE_SMALL, pad=6) + f"""
            background: {_bg('#5856D6', '#AF52DE')};
            border-radius: 10px;
        """)
        analysis_title.setAlignment(Qt.AlignCenter)
        lay.addWidget(analysis_title)

        self.analysis_area = QTextEdit()
        self.analysis_area.setReadOnly(True)
        self.analysis_area.setFont(QFont("SF Pro Display", FONT_SIZE_BODY))
        self.analysis_area.setPlaceholderText("选择识别结果后，点击「开始分析」获取 AI 建议...")
        self.analysis_area.setStyleSheet(f"""
            QTextEdit {{
                background: {BG_CARD_DARK};
                border: 1px solid {BRD_CARD_DARK};
                border-radius: 12px;
                color: rgba(255,255,255,0.9);
                padding: 12px;
            }}
        """)
        lay.addWidget(self.analysis_area, 1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self.btn_analyze = QPushButton("  开始分析")
        self.btn_analyze.setIcon(qta.icon('fa5s.play', color='white'))
        self.btn_analyze.setIconSize(QSize(FONT_SIZE_ICON_SM, FONT_SIZE_ICON_SM))
        self.btn_analyze.setStyleSheet(self._btn_css('#007AFF', '#5856D6'))
        self.btn_analyze.clicked.connect(self._on_analyze)
        btn_row.addWidget(self.btn_analyze)

        self.btn_voice = QPushButton("  语音播报")
        self.btn_voice.setIcon(qta.icon('fa5s.volume-up', color='white'))
        self.btn_voice.setIconSize(QSize(FONT_SIZE_ICON_SM, FONT_SIZE_ICON_SM))
        self.btn_voice.setStyleSheet(self._btn_css('#34C759', '#30D158'))
        self.btn_voice.clicked.connect(self._on_voice)
        btn_row.addWidget(self.btn_voice)

        lay.addLayout(btn_row)
        return container

    # ─── 右栏：问答 ──────────────────────────────────────
    def _build_right_panel(self):
        container = QFrame()
        container.setStyleSheet(_card_style(radius=18))
        container.setGraphicsEffect(_shadow(30, 10))

        lay = QVBoxLayout(container)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(10)

        q_title = QLabel("智能问答")
        q_title.setStyleSheet(SECTION_TITLE.format(size=FONT_SIZE_TITLE_SMALL, pad=6))
        q_title.setAlignment(Qt.AlignCenter)
        lay.addWidget(q_title)

        input_row = QHBoxLayout()
        input_row.setSpacing(8)

        self.question_input = QLineEdit()
        self.question_input.setFont(QFont("SF Pro Display", FONT_SIZE_BODY))
        self.question_input.setPlaceholderText("输入农业问题...")
        self.question_input.setStyleSheet(f"""
            QLineEdit {{
                padding: 10px 14px;
                border: 1px solid {BRD_INPUT};
                border-radius: 10px;
                background: {BG_COMBOBOX};
                color: white;
            }}
            QLineEdit::placeholder {{ color: rgba(255,255,255,0.4); }}
        """)
        self.question_input.returnPressed.connect(self._on_ask)
        input_row.addWidget(self.question_input, 1)

        btn_send = QPushButton("  发送")
        btn_send.setIcon(qta.icon('fa5s.paper-plane', color='white'))
        btn_send.setIconSize(QSize(FONT_SIZE_ICON_XS, FONT_SIZE_ICON_XS))
        btn_send.setStyleSheet(self._btn_css('#007AFF', '#5856D6'))
        btn_send.clicked.connect(self._on_ask)
        input_row.addWidget(btn_send)
        lay.addLayout(input_row)

        self.chat_area = QTextEdit()
        self.chat_area.setReadOnly(True)
        self.chat_area.setFont(QFont("SF Pro Display", FONT_SIZE_BODY))
        self.chat_area.setPlaceholderText("AI 解答将显示在这里...")
        self.chat_area.setStyleSheet(f"""
            QTextEdit {{
                background: {BG_CARD_DARK};
                border: 1px solid {BRD_CARD_DARK};
                border-radius: 12px;
                color: rgba(255,255,255,0.9);
                padding: 12px;
            }}
        """)
        lay.addWidget(self.chat_area, 1)

        return container

    # ─── 工具方法 ─────────────────────────────────────────
    @staticmethod
    def _btn_css(c1, c2):
        bg = _bg(c1, c2)
        return """
            QPushButton {{
                padding: 10px 18px;
                border: none;
                border-radius: 10px;
                font-size: {sz}px;
                font-weight: bold;
                color: white;
                font-family: "SF Pro Display";
                background: {bg};
            }}
            QPushButton:hover {{ opacity: 0.85; }}
        """.format(sz=FONT_SIZE_BTN, bg=bg)

    def _on_analyze(self):
        pest_type = self.result_combo.currentText()
        if not pest_type or pest_type == "暂无识别记录":
            self.status_message.emit("请先进行识别检测")
            return
        if self._analyzing:
            self.status_message.emit("正在分析中 请稍候")
            return

        self.analysis_area.setPlainText("正在分析中，请稍候...")
        self.btn_analyze.setEnabled(False)
        self._analyzing = True
        self.status_message.emit("AI 分析中...")

        soil_info = ", ".join([f"{k}:{v}" for k, v in self.soil_data.items()]) if self.soil_data_loaded else None
        light_info = ", ".join([f"{k}:{v}" for k, v in self.light_data.items()]) if self.light_data_loaded else None
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # GPS位置信息
        if isinstance(self.location, tuple):
            lat, lon, alt = self.location
            location = f"纬度:{lat:.6f}, 经度:{lon:.6f}, 海拔:{alt}m"
        else:
            location = None

        def do_analyze():
            result = ai_service.analyze(pest_type, soil_info, location, current_time, light_info)
            self.analysis_done.emit(result)

        threading.Thread(target=do_analyze, daemon=True).start()

    def _on_analysis_done(self, result):
        if not self._analyzing:
            return
        self._analyzing = False
        self.analysis_area.setPlainText(result)
        self.btn_analyze.setEnabled(True)
        self.status_message.emit("AI 分析完成")

    def on_leave(self):
        """离开页面时完全停止所有活动"""
        voice_service.stop()
        self._stop_soil_diag_timer()
        self._analyzing = False
        self._chatting = False
        self.btn_analyze.setEnabled(True)

        # 停止时钟
        if hasattr(self, '_clock_timer') and self._clock_timer.isActive():
            self._clock_timer.stop()

        # 停止 GPS 定时器和 GPS 服务
        if hasattr(self, '_gps_timer') and self._gps_timer.isActive():
            self._gps_timer.stop()
        gps_service.stop()

        # 记录土壤传感器当前连接状态，离开时断开
        self._soil_was_connected = soil_sensor_service.is_connected()
        if self._soil_was_connected:
            soil_sensor_service.disconnect()
            try:
                light_sensor_service.stop()
            except Exception:
                pass
            self._soil_status_label.setText("未连接")
            self._soil_status_label.setStyleSheet("color: #FF9500; border:none;")
            self._soil_connect_btn.setText("  连接")
            self._soil_connect_btn.setIcon(qta.icon('fa5s.plug', color='white'))
            self._soil_connect_btn.setStyleSheet(f"""
                QPushButton {{
                    padding: 4px 12px;
                    border: none;
                    border-radius: 8px;
                    font-size: {FONT_SIZE_BTN}px;
                    font-weight: bold;
                    color: white;
                    font-family: "SF Pro Display";
                    background: {_bg('#34C759', '#30D158')};
                }}
                QPushButton:hover {{ opacity: 0.85; }}
            """)

    def on_enter(self):
        """进入页面时恢复必要活动"""
        # 重启时钟
        self._tick()
        if not hasattr(self, '_clock_timer') or not self._clock_timer.isActive():
            self._clock_timer = QTimer(self)
            self._clock_timer.timeout.connect(self._tick)
            self._clock_timer.start(1000)

        # 重启 GPS
        gps_service.start()
        if not hasattr(self, '_gps_timer') or not self._gps_timer.isActive():
            self._gps_timer = QTimer(self)
            self._gps_timer.timeout.connect(self._update_gps_location)
            self._gps_timer.start(2000)

        # 恢复土壤传感器连接
        if self._soil_was_connected:
            try:
                soil_sensor_service.connect()
                self._soil_connect_btn.setText("  断开")
                self._soil_connect_btn.setIcon(qta.icon('fa5s.unlink', color='white'))
                self._soil_connect_btn.setStyleSheet(f"""
                    QPushButton {{
                        padding: 4px 12px;
                        border: none;
                        border-radius: 8px;
                        font-size: {FONT_SIZE_BTN}px;
                        font-weight: bold;
                        color: white;
                        font-family: "SF Pro Display";
                        background: {_bg('#FF3B30', '#FF453A')};
                    }}
                    QPushButton:hover {{ opacity: 0.85; }}
                """)
                stats = soil_sensor_service.get_stats()
                self._soil_status_label.setText(f"已连接 {stats['port']}")
                self._soil_status_label.setStyleSheet("color: #34C759; border:none;")
                self._start_soil_diag_timer()
                try:
                    light_sensor_service.start()
                except Exception:
                    pass
            except Exception as e:
                self.status_message.emit(f"土壤传感器恢复失败: {e}")
                self._soil_was_connected = False

    def _on_voice(self):
        text = self.analysis_area.toPlainText().strip()
        if not text:
            self.status_message.emit("没有可播报的内容 请先进行AI分析")
            return
        voice_service.speak(text)
        self.status_message.emit("正在语音播报")

    def _on_ask(self):
        question = self.question_input.text().strip()
        if not question:
            return
        if self._chatting:
            self.status_message.emit("AI 正在回答中 请稍候")
            return

        self.chat_area.append(
            f'<p style="color:#5AC8FA;font-weight:bold;">问：{question}</p>'
        )
        self.question_input.clear()
        self._chatting = True
        self.status_message.emit("AI 思考中...")

        def do_chat():
            answer = ai_service.chat(question)
            self.chat_done.emit(answer)

        threading.Thread(target=do_chat, daemon=True).start()

    def _on_chat_done(self, answer):
        if not self._chatting:
            return
        self._chatting = False
        self.chat_area.append(
            f'<p style="color:rgba(255,255,255,0.85);">{answer}</p>'
            f'<hr style="border:1px solid {BRD_HR};">'
        )
        self.status_message.emit("AI 回答完成")

    def add_detection_result(self, pest_type, image_path):
        if self.result_combo.count() == 1 and self.result_combo.itemText(0) == "暂无识别记录":
            self.result_combo.removeItem(0)
        display_text = f"{pest_type} - {datetime.now().strftime('%H:%M:%S')}"
        self.result_combo.addItem(display_text, image_path)
