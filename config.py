import os
import platform

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- 平台检测（RK3588 等嵌入式设备关闭特效） ----
IS_EMBEDDED = platform.machine() == 'aarch64'
ENABLE_EFFECTS = not IS_EMBEDDED

# ---- 背景图片路径 ----
BACKGROUND_IMAGE_PATH = os.path.join(BASE_DIR, "resources", "background.jpg")

# ---- 主窗口设置 ----
WINDOW_MIN_WIDTH = 1600
WINDOW_MIN_HEIGHT = 950

# ---- API 配置 (DeepSeek) ----
API_KEY = "sk-8269350225b24d2dbedd32b617dc269d"
API_BASE_URL = "https://api.deepseek.com"
API_MODEL = "deepseek-v4-flash"

# ---- 语音播报配置 (edge-tts) ----
# 语音选项: zh-CN-XiaoxiaoNeural(晓晓), zh-CN-YunxiNeural(云希), zh-CN-YunyangNeural(云扬)
TTS_VOICE = "zh-CN-XiaoxiaoNeural"
TTS_RATE = "+0%"

# ---- 摄像头配置 ----
CAMERA_DEVICE_1 = 21        # 摄像头1设备号（对应 /dev/video21）
CAMERA_DEVICE_2 = 23        # 摄像头2设备号（对应 /dev/video23）
CAMERA_WIDTH = 640          # 摄像头分辨率-宽（与模型输入一致，减少预处理开销）
CAMERA_HEIGHT = 640         # 摄像头分辨率-高
CAMERA_DEVICE = CAMERA_DEVICE_1  # 向后兼容（单路摄像头场景）

# ---- YOLO 模型配置 (RKNN) ----
YOLO_CONFIDENCE = 0.6
YOLO_INPUT_SIZE = 640   # 模型输入分辨率（正方形边长）

# ---- 多模型配置 ----
# key = 下拉框显示名称, value = {path, classes}
# classes 必须与模型训练时的类别顺序完全一致
MODEL_CONFIGS = {
    "水稻病害": {
        "path": os.path.join(BASE_DIR, "resources", "rice.rknn"),
        "classes": ["细菌性叶斑病", "褐斑病", "健康叶", "叶瘟病", "叶烧病", "窄褐斑病", "颈瘟病", "稻飞虱病"],
    },
    "玉米病害": {
        "path": os.path.join(BASE_DIR, "resources", "corn.rknn"),
        "classes": ["秋黏虫幼虫", "黄秆蛀虫", "黄秆蛀虫幼虫", "灰斑病", "玉米条斑病", "玉米锈病", "褐斑病", "霜霉病", "玉米肿瘤病", "健康", "枯萎病"],
    },
    "棉花病害": {
        "path": os.path.join(BASE_DIR, "resources", "cotton.rknn"),
        "classes": ["健康棉花叶", "病害棉花植株", "病害棉花叶", "健康棉花植株"],
    },
    "番茄病害": {
        "path": os.path.join(BASE_DIR, "resources", "tomato.rknn"),
        "classes": ["早疫病", "健康", "晚疫病", "斑潜蝇", "叶霉病", "花叶病毒", "叶斑病", "蜘蛛螨", "黄化曲叶病毒"],
    },
    "土豆病害": {
        "path": os.path.join(BASE_DIR, "resources", "potato.rknn"),
        "classes": ["早疫病", "真菌病", "健康", "晚疫病", "虫害"],
    },
    "茶叶病害": {
        "path": os.path.join(BASE_DIR, "resources", "tea.rknn"),
        "classes": ["茶叶叶枯病", "茶芽枯病", "茶轮斑病", "茶藻斑病"],
    },
}

# ---- 土壤传感器配置 ----
SOIL_SENSOR_PORT = "/dev/ttyUSB0"   # RS485 串口设备
SOIL_SENSOR_BAUD = 9600             # 波特率
SOIL_SENSOR_ADDR = 1                # Modbus 设备地址
SOIL_SENSOR_INTERVAL = 2.0          # 采样间隔(秒)

# ---- 光照传感器配置 (RK3588 ADC) ----
LIGHT_SENSOR_ADC_DEVICE = "/sys/bus/iio/devices/iio:device0"  # IIO 设备路径
LIGHT_SENSOR_ADC_CHANNEL = 6                                     # ADC 通道号（P28 引脚 5）
LIGHT_SENSOR_VREF = 1.8                                          # ADC 参考电压(V)
LIGHT_SENSOR_ADC_BITS = 12                                       # ADC 位数
LIGHT_SENSOR_SAMPLES = 8                                         # 每次采样取平均次数
LIGHT_SENSOR_SAMPLE_DELAY = 0.01                                 # 平均采样间隔(s)
LIGHT_SENSOR_INTERVAL = 2.0                                      # 采样间隔(s)
# 分压电阻：假设使用 R_top=100k, R_bottom=20k 的分压比 6:1
LIGHT_SENSOR_R_TOP = 100000.0                                    # 上拉电阻(Ω)
LIGHT_SENSOR_R_BOTTOM = 20000.0                                  # 下拉电阻(Ω)
# 光照强度校准：面板在 1000 W/m^2 时的电压（分压补偿后），暗电压
LIGHT_SENSOR_CAL_VOLTAGE = 6.0
LIGHT_SENSOR_DARK_VOLTAGE = 0.0

# ---- 识别记录 ----
MAX_RECORDS = 100
