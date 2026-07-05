import threading
import time
import struct
import serial
from config import (
    SOIL_SENSOR_PORT, SOIL_SENSOR_BAUD, SOIL_SENSOR_ADDR,
    SOIL_SENSOR_INTERVAL,
)


def modbus_crc(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc


def build_read_command(addr: int, reg_addr: int, reg_count: int) -> bytes:
    cmd = struct.pack('>B B H H', addr, 0x03, reg_addr, reg_count)
    crc = modbus_crc(cmd)
    cmd += struct.pack('<H', crc)
    return cmd


def parse_temperature(raw: int) -> float:
    if raw >= 0x8000:
        raw -= 0x10000
    return raw / 10.0


def parse_humidity(raw: int) -> float:
    return raw / 10.0


REGISTERS = [
    {"name": "温度",   "unit": "°C",    "reg": 0x0000, "parser": parse_temperature, "fmt": ".1f"},
    {"name": "湿度",   "unit": "%",     "reg": 0x0001, "parser": parse_humidity,    "fmt": ".1f"},
    {"name": "电导率", "unit": "μS/cm", "reg": 0x0002, "parser": lambda v: v,       "fmt": "d"},
    {"name": "盐分",   "unit": "mg/L",  "reg": 0x0003, "parser": lambda v: v,       "fmt": "d"},
    {"name": "氮含量", "unit": "mg/kg", "reg": 0x0004, "parser": lambda v: v,       "fmt": "d"},
    {"name": "磷含量", "unit": "mg/kg", "reg": 0x0005, "parser": lambda v: v,       "fmt": "d"},
    {"name": "钾含量", "unit": "mg/kg", "reg": 0x0006, "parser": lambda v: v,       "fmt": "d"},
]

REGISTER_NAMES = [r["name"] for r in REGISTERS]


class SoilSensorService:
    def __init__(self):
        self._ser = None
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._callbacks = []
        self._last_data = {}
        self._port = SOIL_SENSOR_PORT
        self._baud = SOIL_SENSOR_BAUD
        self._addr = SOIL_SENSOR_ADDR
        self._interval = SOIL_SENSOR_INTERVAL
        self._read_count = 0          # 成功读取次数
        self._fail_count = 0          # 失败次数
        self._last_error = ""         # 最后一次错误信息

    def add_callback(self, cb):
        with self._lock:
            self._callbacks.append(cb)

    def remove_callback(self, cb):
        with self._lock:
            if cb in self._callbacks:
                self._callbacks.remove(cb)

    def get_last_data(self):
        with self._lock:
            return dict(self._last_data)

    def is_connected(self):
        return self._ser is not None and self._ser.is_open

    def get_stats(self):
        """返回读取统计信息"""
        with self._lock:
            return {
                "read_count": self._read_count,
                "fail_count": self._fail_count,
                "last_error": self._last_error,
                "port": self._port,
                "baud": self._baud,
                "addr": self._addr,
            }

    def connect(self, port=None, baud=None, addr=None):
        if port is not None:
            self._port = port
        if baud is not None:
            self._baud = baud
        if addr is not None:
            self._addr = addr

        if self._ser and self._ser.is_open:
            self.disconnect()

        try:
            self._ser = serial.Serial(
                port=self._port,
                baudrate=self._baud,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1.0,
            )
        except Exception as e:
            self._ser = None
            raise e

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def disconnect(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._ser and self._ser.is_open:
            try:
                self._ser.close()
            except Exception:
                pass
        self._ser = None
        with self._lock:
            self._read_count = 0
            self._fail_count = 0
            self._last_error = ""

    def _read_once(self):
        """与 soil_sensor_app.py 模板一致：reset, write, sleep(50ms), read(19)"""
        if not self._ser or not self._ser.is_open:
            self._last_error = "串口未打开"
            return None
        cmd = build_read_command(self._addr, 0x0000, 7)
        try:
            self._ser.reset_input_buffer()
            self._ser.write(cmd)
            time.sleep(0.05)      # 等待响应（与 Windows 模板一致）
            # 期望返回: addr(1) + func(1) + len(1) + data(14) + crc(2) = 19 bytes
            resp = self._ser.read(19)
            if len(resp) < 5:
                msg = f"无响应({len(resp)}字节) 指令={cmd.hex()}"
                self._last_error = msg
                print(f"[Soil] {msg}")
                return None
            # 验证 CRC
            resp_crc = struct.unpack('<H', resp[-2:])[0]
            calc_crc = modbus_crc(resp[:-2])
            if resp_crc != calc_crc:
                msg = f"CRC不匹配(收到{resp_crc:04X}, 计算{calc_crc:04X})"
                self._last_error = msg
                print(f"[Soil] {msg}")
                return None
            self._last_error = ""
            # 解析 7 个寄存器（每个 2 字节，大端）
            data = {}
            for i, reg in enumerate(REGISTERS):
                raw = struct.unpack('>H', resp[3 + i*2 : 5 + i*2])[0]
                val = reg["parser"](raw)
                data[reg["name"]] = val
            return data
        except Exception as e:
            msg = f"读取异常: {e}"
            self._last_error = msg
            print(f"[Soil] {msg}")
            return None

    def _loop(self):
        print(f"[Soil] 采样线程启动, port={self._port}, baud={self._baud}, addr={self._addr}")
        while self._running and self._ser and self._ser.is_open:
            data = self._read_once()
            if data:
                print(f"[Soil] 读取成功: 温度={data.get('温度')}°C, 湿度={data.get('湿度')}%")
                with self._lock:
                    self._read_count += 1
                    self._last_data = data
                    callbacks = list(self._callbacks)
                for cb in callbacks:
                    try:
                        cb(data)
                    except Exception:
                        pass
            else:
                with self._lock:
                    self._fail_count += 1
                print(f"[Soil] 本次读取无数据")
            time.sleep(self._interval)
        print(f"[Soil] 采样线程退出")

    def stop(self):
        self.disconnect()


soil_sensor_service = SoilSensorService()
