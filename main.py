# coding:UTF-8
import serial
import serial.tools.list_ports
import numpy as np
import cv2
import struct
import time
import threading
import math
import platform
import json
import pygame
import psutil
from screeninfo import get_monitors
import requests
from PIL import Image
import io
import os

# --- 新增依赖 ---
import pygame_gui


# --- IMU 依赖 (伪造部分保持不变) ---
# 确保您的项目中存在这些lib文件
# import lib.device_model as deviceModel
# from lib.data_processor.roles.jy901s_dataProcessor import JY901SDataProcessor
# from lib.protocol_resolver.roles.wit_protocol_resolver import WitProtocolResolver
class FakeDeviceModel:
    def __init__(self, *args, **kwargs):
        self.serialConfig = type('serialConfig', (), {'portName': '', 'baud': 0})
        self.dataProcessor = type('dataProcessor', (), {'onVarChanged': []})

    def openDevice(self):
        pass

    def closeDevice(self):
        pass

    def getDeviceData(self, key):
        if key == "angleX": return (pygame.mouse.get_pos()[0] / 1920 - 0.5) * 180
        if key == "angleY": return (pygame.mouse.get_pos()[1] / 1080 - 0.5) * 180
        if key == "angleZ": return 0
        return 0


deviceModel = type('deviceModel', (), {'DeviceModel': FakeDeviceModel})


class JY901SDataProcessor: pass


class WitProtocolResolver: pass


# --- 用户配置 ---
IMAGE_RESOLUTION = (100, 100)
IMU_BAUDRATE = 9600
CALIBRATION_FILE = "calibration.json"
ZOOM_FACTOR = 4.0
HEIGHT_MULTIPLIER = 40.0
FULLSCREEN_RESOLUTION = (1920, 1080)
VIDEO_BACKGROUND_PATH = "background.mp4"  # 视频背景文件路径


# ==========================================================================================
#  模块一：MaixSenseA010 深度相机控制类 (无改动)
# ==========================================================================================
class MaixSenseA010:
    def __init__(self, port, baudrate=115200, resolution=(100, 100)):
        self.port = port
        self.baudrate = baudrate
        self.resolution = resolution
        self.image_size = resolution[0] * resolution[1]
        self.packet_header = b'\x00\xff'
        self.packet_tail = b'\xdd'
        self.packet_data_size = 2 + 16 + self.image_size + 1 + 1
        self.ser = None
        self.latest_frame = None
        self._is_running = False
        self.lock = threading.Lock()
        self.thread = None
        self.is_mock = port is None
        if self.is_mock:
            print("WARNING: MaixSense port not configured. Running in mock mode.")

    def _connect(self):
        if self.is_mock: return True
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"Successfully opened MaixSense serial port: {self.port}")
            return True
        except serial.SerialException as e:
            print(f"ERROR: Could not open MaixSense serial port {self.port}: {e}")
            return False

    def send_at_command(self, command, delay=0.2):
        if self.is_mock or not self.ser or not self.ser.is_open: return
        full_command = command + '\r\n'
        self.ser.reset_input_buffer()
        self.ser.write(full_command.encode('utf-8'))
        time.sleep(delay)
        self.ser.read(self.ser.in_waiting)

    def configure_device(self):
        if self.is_mock: return
        print("--- Configuring MaixSense Module ---")
        self.send_at_command("AT+DISP=2")
        self.send_at_command("AT+ISP=1")
        self.send_at_command("AT+UNIT=0")
        self.ser.reset_input_buffer()
        print("--- MaixSense Configuration Complete ---")

    def _read_thread(self):
        if not self._connect():
            self._is_running = False
            return
        if not self.is_mock:
            self.configure_device()
        frame_counter = 0
        while self._is_running:
            if self.is_mock:
                y, x = np.mgrid[-5:5:100j, -5:5:100j]
                frame_counter += 0.1
                mock_data = np.sin(np.sqrt(x * x + y * y) + frame_counter) * 127 + 128
                with self.lock:
                    self.latest_frame = mock_data.astype(np.uint8)
                time.sleep(0.03)
                continue
            try:
                self.ser.read_until(self.packet_header)
                packet_data = self.ser.read(self.packet_data_size)
                if len(packet_data) == self.packet_data_size and packet_data[-1:] == self.packet_tail:
                    depth_frame_bytes = packet_data[18:-2]
                    if len(depth_frame_bytes) == self.image_size:
                        depth_frame = np.frombuffer(depth_frame_bytes, dtype=np.uint8)
                        depth_frame = 255 - depth_frame
                        depth_frame = depth_frame.reshape((self.resolution[1], self.resolution[0]))
                        with self.lock:
                            self.latest_frame = depth_frame
            except serial.SerialException:
                print("MaixSense serial port disconnected. Thread exiting.")
                break
        if not self.is_mock:
            self.send_at_command("AT+ISP=0")
            if self.ser and self.ser.is_open:
                self.ser.close()
                print("MaixSense serial port closed.")

    def start(self):
        if self._is_running: return
        self._is_running = True
        self.thread = threading.Thread(target=self._read_thread)
        self.thread.daemon = True
        self.thread.start()
        print("MaixSense data reading thread started.")

    def stop(self):
        self._is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
        print("MaixSense data reading thread stopped.")

    def get_frame(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None


# ==========================================================================================
#  模块二：IMU_JY901S 控制类 (无改动)
# ==========================================================================================
class IMU_JY901S:
    def __init__(self, port, baudrate, pygame_ready_event):
        self.is_mock = port is None
        self.pygame_ready_event = pygame_ready_event
        if self.is_mock:
            print("WARNING: IMU port not configured. Running in mock mode (using mouse).")
            self.device = FakeDeviceModel()
        else:
            self.device = deviceModel.DeviceModel("JY901S_IMU", WitProtocolResolver(), JY901SDataProcessor(), "51_0")
        self.device.serialConfig.portName = port
        self.device.serialConfig.baud = baudrate
        self.latest_angles = {'X': 0.0, 'Y': 0.0, 'Z': 0.0}
        self._is_running = False
        self.lock = threading.Lock()
        self.thread = None

    def _on_update(self, device_model):
        with self.lock:
            self.latest_angles['X'] = device_model.getDeviceData("angleX")
            self.latest_angles['Y'] = device_model.getDeviceData("angleY")
            self.latest_angles['Z'] = device_model.getDeviceData("angleZ")

    def _read_thread(self):
        self.pygame_ready_event.wait()
        if self.is_mock:
            while self._is_running:
                try:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    pitch = (mouse_y / FULLSCREEN_RESOLUTION[1] - 0.5) * -180
                    yaw = (mouse_x / FULLSCREEN_RESOLUTION[0] - 0.5) * 360
                    with self.lock:
                        self.latest_angles['X'] = 0
                        self.latest_angles['Y'] = pitch
                        self.latest_angles['Z'] = yaw
                except pygame.error:
                    pass
                time.sleep(0.02)
            return
        try:
            self.device.openDevice()
            self.device.dataProcessor.onVarChanged.append(self._on_update)
            print(f"Successfully opened IMU serial port: {self.device.serialConfig.portName}")
        except Exception as e:
            print(f"ERROR: Could not open IMU serial port {self.device.serialConfig.portName}: {e}")
            self._is_running = False
            return
        while self._is_running:
            time.sleep(0.5)
        self.device.closeDevice()
        print("IMU serial port closed.")

    def start(self):
        if self._is_running: return
        self._is_running = True
        self.thread = threading.Thread(target=self._read_thread)
        self.thread.daemon = True
        self.thread.start()
        print("IMU data reading thread started.")

    def stop(self):
        self._is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        print("IMU data reading thread stopped.")

    def get_angles(self):
        with self.lock:
            angles = self.latest_angles.copy()
        return angles if all(k in angles for k in ['X', 'Y', 'Z']) else None


# ==========================================================================================
#  模块三：地图管理器 (无改动)
# ==========================================================================================
class MapManager:
    def __init__(self, cache_dir="map_cache"):
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.tile_cache = {}
        self.current_tile_surface = None
        self.surface_lock = threading.Lock()
        self.gps_coords = (0, 0, 0)
        self.gps_lock = threading.Lock()
        self._is_running = True
        self.thread = threading.Thread(target=self._tile_loader_thread)
        self.thread.daemon = True
        self.thread.start()

    def _tile_loader_thread(self):
        last_loaded_tile = None
        while self._is_running:
            with self.gps_lock:
                lon, lat, zoom = self.gps_coords
            tile_x, tile_y = self.deg2num(lat, lon, zoom)
            current_tile_key = (zoom, tile_x, tile_y)
            if current_tile_key != last_loaded_tile:
                tile_surface = self._get_tile_data(lon, lat, zoom)
                if tile_surface:
                    with self.surface_lock:
                        self.current_tile_surface = tile_surface
                    last_loaded_tile = current_tile_key
            time.sleep(1)

    def _get_tile_data(self, lon, lat, zoom):
        tile_x, tile_y = self.deg2num(lat, lon, zoom)
        tile_key = (zoom, tile_x, tile_y)
        if tile_key in self.tile_cache:
            return self.tile_cache[tile_key]
        filepath = os.path.join(self.cache_dir, f"{zoom}_{tile_x}_{tile_y}.png")
        if os.path.exists(filepath):
            try:
                img = pygame.image.load(filepath).convert()
                self.tile_cache[tile_key] = img
                return img
            except pygame.error:
                os.remove(filepath)
        url = f"https://a.tile.openstreetmap.org/{zoom}/{tile_x}/{tile_y}.png"
        headers = {'User-Agent': 'SkiGoggleUI/2.0'}
        try:
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()
            img_data = io.BytesIO(response.content)
            with open(filepath, "wb") as f:
                f.write(response.content)
            pil_img = Image.open(img_data).convert("RGB")
            pygame_img = pygame.image.frombytes(pil_img.tobytes(), pil_img.size, pil_img.mode).convert()
            self.tile_cache[tile_key] = pygame_img
            return pygame_img
        except requests.exceptions.RequestException as e:
            print(f"Map tile download failed: {e}")
            return None

    def update_gps(self, lon, lat, zoom):
        with self.gps_lock:
            self.gps_coords = (lon, lat, zoom)

    def get_surface(self):
        with self.surface_lock:
            return self.current_tile_surface

    def stop(self):
        self._is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
        print("MapManager thread stopped.")

    def deg2num(self, lat_deg, lon_deg, zoom):
        lat_rad = math.radians(lat_deg)
        n = 2.0 ** zoom
        xtile = int((lon_deg + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return (xtile, ytile)


# ==========================================================================================
#  模块四：视频背景管理器 (新)
# ==========================================================================================
class VideoManager:
    def __init__(self, video_path, screen_size):
        self.video_path = video_path
        self.screen_size = screen_size
        self.cap = None
        self.is_active = False
        if os.path.exists(self.video_path):
            self.cap = cv2.VideoCapture(self.video_path)
            if self.cap.isOpened():
                self.is_active = True
                print("Video background loaded successfully.")
            else:
                print(f"ERROR: Could not open video file: {self.video_path}")
        else:
            print(f"WARNING: Video file not found at '{self.video_path}'. Using solid color background.")

    def get_frame(self):
        if not self.is_active:
            return None
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                return None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.image.frombuffer(frame.tobytes(), frame.shape[1::-1], "RGB")
        return pygame.transform.scale(frame_surface, self.screen_size)

    def stop(self):
        if self.cap:
            self.cap.release()
        print("VideoManager stopped.")


# ==========================================================================================
#  模块五：UI与可视化 (全面升级)
# ==========================================================================================
def get_display_info():
    monitors = get_monitors()
    print("Detected displays:")
    for i, m in enumerate(monitors):
        print(f"  - Display {i}: {m.width}x{m.height} at ({m.x}, {m.y}) {'(Primary)' if m.is_primary else ''}")
    target_monitor_index = 0
    if len(monitors) > 1:
        target_monitor_index = next((i for i, m in enumerate(monitors) if not m.is_primary), 0)
    target_monitor = monitors[target_monitor_index]
    print(f"Targeting display {target_monitor_index}.")
    return target_monitor_index, (target_monitor.x, target_monitor.y)


def create_rotation_matrix(pitch, yaw, roll):
    pitch, yaw, roll = map(math.radians, [pitch, yaw, roll])
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
    return Rz @ Ry @ Rx


def load_calibration():
    try:
        with open(CALIBRATION_FILE, 'r') as f:
            offsets = json.load(f)
            print(f"Calibration loaded: {offsets}")
            return offsets
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Calibration file not found or invalid, using zero offsets. Reason: {e}")
        return {'X': 0.0, 'Y': 0.0, 'Z': 0.0}


def save_calibration(offsets):
    try:
        with open(CALIBRATION_FILE, 'w') as f:
            json.dump(offsets, f, indent=4)
        print(f"Calibration saved: {offsets}")
    except Exception as e:
        print(f"Failed to save calibration: {e}")


class UIManager:
    def __init__(self, screen, is_demo_mode):
        self.screen = screen
        self.width, self.height = screen.get_size()
        self.is_demo_mode = is_demo_mode
        self.gui_manager = pygame_gui.UIManager(FULLSCREEN_RESOLUTION, 'theme.json')
        self.map_manager = MapManager()
        self.video_manager = VideoManager(VIDEO_BACKGROUND_PATH, (self.width, self.height))

        self.font_large = pygame.font.Font(None, 80)
        self.font_medium = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 36)
        self.font_tiny = pygame.font.Font(None, 24)

        self.current_lat = 22.5330
        self.current_lon = 114.0583
        self.map_zoom = 16

        # --- 横幅内容升级 ---
        self.demo_banner_x = 0
        ch_text = "演示模式：未检测到传感器，使用鼠标/模拟数据。"
        en_text = "DEMO MODE: SENSORS NOT DETECTED. USING MOUSE/SIMULATED DATA."
        separator = " ||| "
        base_text = f"{ch_text}{separator}{en_text}{separator}"

        # 确保横幅文本足够长以实现无缝滚动
        text_surf_once = self.font_tiny.render(base_text, True, (0, 0, 0))
        repeats = math.ceil(self.width * 2 / text_surf_once.get_width()) + 1
        self.full_banner_text = base_text * repeats
        self.banner_text_surf = self.font_tiny.render(self.full_banner_text, True, (0, 0, 0))
        self.banner_scroll_limit = self.banner_text_surf.get_width() / repeats * (repeats - 1)

        self.create_ui_elements()

    def create_ui_elements(self):
        # 创建所有UI元素一次
        self.weather_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect(40, 40, 350, 80), text="",
                                                         manager=self.gui_manager, object_id='#weather_label')
        self.time_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect(self.width - 400, 40, 360, 40), text="",
                                                      manager=self.gui_manager, object_id='#time_label')
        map_size = 250
        padding = 50
        map_rect = pygame.Rect(padding, self.height - map_size - padding, map_size, map_size)
        self.map_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(map_rect.left, map_rect.top - 40, map_rect.width, 30), text="",
            manager=self.gui_manager, object_id='#map_label')

        # 新增信息面板
        self.speed_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(self.width - 250, self.height / 2 - 100, 200, 50), text="",
            manager=self.gui_manager, object_id='#data_label')
        self.altitude_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(self.width - 250, self.height / 2 - 50, 200, 50), text="",
            manager=self.gui_manager, object_id='#data_label')
        self.slope_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(self.width - 250, self.height / 2, 200, 50), text="", manager=self.gui_manager,
            object_id='#data_label')

    def draw_text(self, text, font, color, position, anchor="topleft"):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(**{anchor: position})
        self.screen.blit(text_surface, text_rect)

    def draw_demo_banner(self, dt):
        if not self.is_demo_mode:
            return
        banner_h = 30
        banner_surface = pygame.Surface((self.width, banner_h))
        banner_surface.fill((255, 193, 7))

        self.demo_banner_x -= 80 * dt
        if self.demo_banner_x < -self.banner_scroll_limit:
            self.demo_banner_x += self.banner_scroll_limit

        banner_surface.blit(self.banner_text_surf, (self.demo_banner_x, 5))
        self.screen.blit(banner_surface, (0, 0))

    def update_hud_info(self, angles):
        # 模拟天气数据 (深圳南山区)
        weather_data = {'temp': 28, 'condition': 'Cloudy', 'wind': 12, 'humidity': 85}
        weather_html = f"<b>Nanshan, Shenzhen</b><br>" \
                       f"{weather_data['temp']}°C, {weather_data['condition']} | " \
                       f"Wind: {weather_data['wind']}km/h | " \
                       f"Humidity: {weather_data['humidity']}%"
        self.weather_label.set_text(weather_html)

        # 时间与电量
        try:
            battery = psutil.sensors_battery()
            batt_percent = f"{int(battery.percent)}%" if battery else "N/A"
        except (AttributeError, NotImplementedError, TypeError):
            batt_percent = "N/A"
        current_time = time.strftime("%H:%M:%S")
        self.time_label.set_text(f"{current_time} | Batt: {batt_percent}")

        # GPS
        self.map_label.set_text(f"GPS: {self.current_lat:.4f}, {self.current_lon:.4f}")

        # 新增模拟数据
        self.speed_label.set_text(f"SPD: {abs(angles['Y'] * 0.5):.1f} km/h")
        self.altitude_label.set_text(f"ALT: {1500 - angles['Y'] * 10:.1f} m")
        self.slope_label.set_text(f"SLOPE: {abs(angles['Y'] / 5):.1f}°")

    def draw_compass(self, yaw):
        center_x, center_y = self.width / 2, 100
        radius = 60
        # 绘制罗盘背景
        pygame.draw.circle(self.screen, (0, 0, 0, 128), (center_x, center_y), radius)
        pygame.draw.circle(self.screen, (0, 150, 255), (center_x, center_y), radius, 2)

        # 绘制方向刻度
        for i in range(0, 360, 30):
            angle = math.radians(i - yaw)
            start_pos = (center_x + (radius - 10) * math.sin(angle), center_y - (radius - 10) * math.cos(angle))
            end_pos = (center_x + radius * math.sin(angle), center_y - radius * math.cos(angle))
            pygame.draw.line(self.screen, (255, 255, 255), start_pos, end_pos, 2)

        # 绘制北方指针
        n_angle = math.radians(-yaw)
        n_pos = (center_x + (radius - 5) * math.sin(n_angle), center_y - (radius - 5) * math.cos(n_angle))
        self.draw_text("N", self.font_small, (255, 0, 0), n_pos, anchor="center")

        # 绘制当前方向
        self.draw_text(f"{int(yaw % 360)}°", self.font_medium, (255, 255, 255), (center_x, center_y), anchor="center")

    def draw_minimap(self):
        map_size = 250
        padding = 50
        map_rect = pygame.Rect(padding, self.height - map_size - padding, map_size, map_size)

        # 创建圆形遮罩
        target_surf = pygame.Surface((map_size, map_size), pygame.SRCALPHA)
        map_tile = self.map_manager.get_surface()

        if map_tile:
            scaled_tile = pygame.transform.scale(map_tile, (map_size, map_size))
            pygame.draw.circle(target_surf, (255, 255, 255), (map_size // 2, map_size // 2), map_size // 2)
            target_surf.blit(scaled_tile, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        else:
            pygame.draw.circle(target_surf, (10, 10, 10), (map_size // 2, map_size // 2), map_size // 2)
            self.draw_text("Loading Map...", self.font_tiny, (255, 255, 255), (map_size // 2, map_size // 2),
                           anchor="center")

        self.screen.blit(target_surf, map_rect.topleft)
        pygame.draw.circle(self.screen, (0, 150, 255), map_rect.center, map_size // 2, 3)

        # 玩家指针
        player_pos = map_rect.center
        pygame.draw.circle(self.screen, (255, 0, 0), player_pos, 8)
        pygame.draw.circle(self.screen, (255, 255, 255), player_pos, 8, 2)

    def draw_wireframe(self, depth_frame, angles, calibration_offset):
        if depth_frame is None or angles is None:
            # --- FIX: Use a pre-loaded pygame.font.Font object ---
            self.draw_text("Waiting for sensor data...", self.font_large, (255, 255, 255),
                           (self.width / 2, self.height / 2), anchor="center")
            return

        calibrated_angles = {k: angles[k] - calibration_offset[k] for k in angles}
        roll, pitch, yaw = calibrated_angles['X'], calibrated_angles['Y'], calibrated_angles['Z']
        R = create_rotation_matrix(-pitch, yaw, -roll)
        min_depth, max_depth = np.min(depth_frame), np.max(depth_frame)
        depth_range = max_depth - min_depth if max_depth > min_depth else 1
        normalized_frame = (depth_frame - min_depth) / depth_range
        h, w = depth_frame.shape
        grid_step = 5
        colormap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_VIRIDIS)
        points_to_draw = []
        center_x, center_y = self.width / 2, self.height / 2
        for y in range(0, h, grid_step):
            for x in range(0, w, grid_step):
                z_value = normalized_frame[y, x] * HEIGHT_MULTIPLIER
                p1_3d = np.array([x - w / 2, y - h / 2, -z_value])
                p1_rotated = R @ p1_3d
                p1_screen = (int(p1_rotated[0] * ZOOM_FACTOR + center_x), int(p1_rotated[1] * ZOOM_FACTOR + center_y))
                color = tuple(int(c) for c in colormap[depth_frame[y, x]][0])[::-1]
                if x + grid_step < w:
                    p2_3d = np.array(
                        [(x + grid_step) - w / 2, y - h / 2, -normalized_frame[y, x + grid_step] * HEIGHT_MULTIPLIER])
                    p2_rotated = R @ p2_3d
                    p2_screen = (
                    int(p2_rotated[0] * ZOOM_FACTOR + center_x), int(p2_rotated[1] * ZOOM_FACTOR + center_y))
                    points_to_draw.append((p1_screen, p2_screen, color))
                if y + grid_step < h:
                    p3_3d = np.array(
                        [x - w / 2, (y + grid_step) - h / 2, -normalized_frame[y + grid_step, x] * HEIGHT_MULTIPLIER])
                    p3_rotated = R @ p3_3d
                    p3_screen = (
                    int(p3_rotated[0] * ZOOM_FACTOR + center_x), int(p3_rotated[1] * ZOOM_FACTOR + center_y))
                    points_to_draw.append((p1_screen, p3_screen, color))
        for p1, p2, color in points_to_draw:
            pygame.draw.line(self.screen, color, p1, p2, 1)
        info_text = f"Roll: {roll:.1f} Pitch: {pitch:.1f} Yaw: {yaw:.1f}"
        self.draw_text(info_text, self.font_tiny, (0, 255, 0), (self.width / 2, self.height - 40), anchor="midbottom")

    def show_calibration_message(self):
        pygame_gui.windows.UIMessageWindow(
            rect=pygame.Rect((self.width / 2 - 150, self.height / 2 - 75), (300, 150)),
            html_message="<font size=5>Pose Calibrated</font><br>New zero point has been set.",
            manager=self.gui_manager, window_title="Calibration")

    def process_events(self, event):
        self.gui_manager.process_events(event)

    def update(self, time_delta, angles):
        self.current_lon += 0.00001 * (time_delta * 60)
        self.map_manager.update_gps(self.current_lon, self.current_lat, self.map_zoom)
        if angles:
            self.update_hud_info(angles)
        self.gui_manager.update(time_delta)

    def draw_all(self, dt, depth_frame, angles, calibration_offset):
        # 绘制背景
        bg_frame = self.video_manager.get_frame()
        if bg_frame:
            self.screen.blit(bg_frame, (0, 0))
        else:
            self.screen.fill(self.gui_manager.get_theme().get_colour('dark_bg'))

        # 绘制核心功能和UI
        self.draw_wireframe(depth_frame, angles, calibration_offset)
        self.draw_minimap()
        if angles:
            self.draw_compass(angles['Z'] - calibration_offset['Z'])
        self.draw_demo_banner(dt)
        self.gui_manager.draw_ui(self.screen)

    def stop(self):
        self.map_manager.stop()
        self.video_manager.stop()


def main():
    ports = serial.tools.list_ports.comports()
    print("Available serial ports:")
    for port in ports: print(f" - {port.device}: {port.description}")
    maix_port, imu_port = None, None
    if platform.system().lower() == 'windows':
        maix_port, imu_port = "COM8", "COM9"
    else:
        maix_port, imu_port = "/dev/ttyUSB1", "/dev/ttyUSB0"
    available_ports = [p.device for p in ports]
    is_demo_mode = False
    if maix_port not in available_ports:
        print(f"WARNING: MaixSense port {maix_port} not found.")
        maix_port = None
        is_demo_mode = True
    if imu_port not in available_ports:
        print(f"WARNING: IMU port {imu_port} not found.")
        imu_port = None
        is_demo_mode = True

    print("\nInitializing UI...")
    pygame.init()
    pygame_ready_event = threading.Event()
    calibration_offset = load_calibration()
    maix_device = MaixSenseA010(maix_port, resolution=IMAGE_RESOLUTION)
    imu_device = IMU_JY901S(imu_port, baudrate=IMU_BAUDRATE, pygame_ready_event=pygame_ready_event)
    maix_device.start()
    imu_device.start()

    theme_content = {
        "defaults": {"colours": {"dark_bg": "#0A193C", "normal_text": "#FFFFFF"}},
        "#weather_label": {"colours": {"normal_text": "#FFFFFF"}, "font": {"name": "dejavusans", "size": "20"},
                           "misc": {"text_horiz_alignment": "left", "text_vert_alignment": "top"}},
        "#time_label": {"colours": {"normal_text": "#FFFFFF"}, "font": {"name": "dejavusans", "size": "28"},
                        "misc": {"text_horiz_alignment": "right"}},
        "#map_label": {"colours": {"normal_text": "#FFFFFF"}, "font": {"name": "dejavusans", "size": "18"},
                       "misc": {"text_horiz_alignment": "center"}},
        "#title_label": {"font": {"name": "dejavusans", "size": "48"}},
        "#data_label": {"colours": {"normal_text": "#FFFFFF"}, "font": {"name": "dejavusans", "size": "32"},
                        "misc": {"text_horiz_alignment": "left"}}
    }
    with open('theme.json', 'w') as f:
        json.dump(theme_content, f, indent=4)

    display_index, display_pos = get_display_info()
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{display_pos[0]},{display_pos[1]}"
    screen = pygame.display.set_mode(FULLSCREEN_RESOLUTION, pygame.NOFRAME | pygame.FULLSCREEN, display=display_index)
    pygame.display.set_caption("Ski Goggle UI v5")
    pygame.mouse.set_visible(False)
    clock = pygame.time.Clock()
    ui_manager = UIManager(screen, is_demo_mode)
    pygame_ready_event.set()

    print("\n--- Controls ---")
    print("C: Calibrate current pose as zero.")
    print("Q or ESC: Quit.")
    print("Mouse (in demo mode): Control view.")

    running = True
    try:
        while running:
            time_delta = clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE)):
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                    raw_angles = imu_device.get_angles()
                    if raw_angles:
                        calibration_offset = raw_angles.copy()
                        save_calibration(calibration_offset)
                        ui_manager.show_calibration_message()
                ui_manager.process_events(event)
            depth_frame = maix_device.get_frame()
            raw_angles = imu_device.get_angles()
            ui_manager.update(time_delta, raw_angles)
            ui_manager.draw_all(time_delta, depth_frame, raw_angles, calibration_offset)
            pygame.display.flip()
    except (KeyboardInterrupt, Exception) as e:
        print(f"\nProgram interrupted or error occurred: {e}")
    finally:
        print("\nShutting down...")
        maix_device.stop()
        imu_device.stop()
        if 'ui_manager' in locals():
            ui_manager.stop()
        pygame.quit()
        print("Program finished.")


if __name__ == "__main__":
    main()
