import os
import sys
import time
import threading
import queue
import traceback
import json
import base64
import io
import math
import uuid
import ctypes
import webbrowser
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk, ImageGrab, ImageDraw, ImageChops
import pyautogui
from pynput import keyboard
import copy
from datetime import datetime
from collections import namedtuple

# --- 1. 依赖库检查与导入 ---
try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("⚠️ 警告: 未安装 opencv-python，高级图像识别功能受限。")

try:
    import pyperclip
    HAS_PYPERCLIP = True
except ImportError:
    HAS_PYPERCLIP = False

try:
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioMeterInformation
    import comtypes 
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

# --- 2. 全局系统设置 ---
pyautogui.FAILSAFE = False # 保持开启，但在逻辑中自行处理边界

def get_scale_factor():
    try:
        if sys.platform.startswith('win'):
            try: ctypes.windll.shcore.SetProcessDpiAwareness(2) 
            except: ctypes.windll.user32.SetProcessDPIAware()
        log_w, log_h = pyautogui.size(); user32 = ctypes.windll.user32
        phy_w, phy_h = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        if log_w == 0 or phy_w == 0: return 1.0, 1.0
        return max(0.5, min(4.0, phy_w / log_w)), max(0.5, min(4.0, phy_h / log_h))
    except: return 1.0, 1.0

SCALE_X, SCALE_Y = get_scale_factor()
SCALE_FACTOR = (SCALE_X + SCALE_Y) / 2.0
Box = namedtuple('Box', 'left top width height')

# --- [安全增强] 数据验证辅助函数 ---
def safe_float(value, default=0.0):
    try: return float(value)
    except (ValueError, TypeError): return default

def safe_int(value, default=0):
    try: return int(float(value))
    except (ValueError, TypeError): return default

# --- 3. UI 配色与字体 ---
COLORS = {
    'bg_app':     '#1e1e1e', 'bg_sidebar': '#252526', 'bg_canvas':  '#1e1e1e',
    'bg_panel':   '#252526', 'bg_node':    '#333333', 'bg_header':  '#3c3c3c',
    'bg_card':    '#2d2d30', 'fg_title':   '#cccccc', 'fg_text':    '#d4d4d4',
    'fg_sub':     '#858585', 'accent':     '#007acc', 'success':    '#4ec9b0',
    'danger':     '#f48771', 'warning':    '#cca700', 'control':    '#c586c0',
    'sensor':     '#ce9178', 'var_node':   '#569cd6', 'wire':       '#6e6e6e',
    'wire_active':'#dcdcaa', 'wire_hl':    '#007acc', 'socket':     '#dcdcaa',
    'grid':       '#262626', 'shadow':     '#000000', 'hover':      '#3e3e42',
    'select_box': '#007acc', 'active_border': '#007acc', 'marker':     '#f44747',
    'btn_bg':     '#3e3e42', 'btn_hover':  '#505050', 'input_bg':   '#3c3c3c',
    'hl_running': '#dcdcaa', 'hl_ok':      '#4ec9b0', 'hl_fail':    '#f48771',
    'breakpoint': '#e51400', 'log_bg':     '#1e1e1e', 'log_fg':     '#d4d4d4',
}

FONTS = {
    'node_title': ('Segoe UI', int(10 * SCALE_FACTOR), 'bold'), 
    'node_text': ('Segoe UI', int(8 * SCALE_FACTOR)),
    'code': ('Consolas', int(10 * SCALE_FACTOR)), 
    'h2': ('Segoe UI', int(11 * SCALE_FACTOR), 'bold'), 
    'small': ('Segoe UI', int(9 * SCALE_FACTOR)),
    'log': ('Consolas', int(9 * SCALE_FACTOR))
}

LOG_LEVELS = {'info': {'color': '#569cd6', 'icon': 'ℹ️'}, 'success': {'color': '#4ec9b0', 'icon': '✅'}, 'warning': {'color': '#cca700', 'icon': '⚠️'}, 'error': {'color': '#f48771', 'icon': '❌'}, 'exec': {'color': '#808080', 'icon': '▶️'}, 'paused': {'color': '#dcdcaa', 'icon': '⏸️'}}
NODE_WIDTH = int(160 * SCALE_FACTOR)
HEADER_HEIGHT = int(28 * SCALE_FACTOR)
PORT_START_Y = int(45 * SCALE_FACTOR)
PORT_STEP_Y = int(22 * SCALE_FACTOR)
GRID_SIZE = int(20 * SCALE_FACTOR)

NODE_CONFIG = {
    'start':    {'title': '▶ 开始', 'outputs': ['out'], 'color': COLORS['success']},
    'end':      {'title': '⏹️ 结束', 'outputs': [], 'color': COLORS['danger']},
    'loop':     {'title': '🔄 循环', 'outputs': ['loop', 'exit'], 'color': COLORS['control']},
    'wait':     {'title': '⏳ 延时', 'outputs': ['out'], 'color': COLORS['control']},
    'mouse':    {'title': '👆 鼠标操作', 'outputs': ['out'], 'color': COLORS['bg_header']},
    'keyboard': {'title': '⌨️ 键盘输入', 'outputs': ['out'], 'color': COLORS['bg_header']},
    'web':      {'title': '🔗 网页操作', 'outputs': ['out'], 'color': COLORS['bg_header']},
    'image':    {'title': '🎯 找图点击', 'outputs': ['found', 'timeout'], 'color': COLORS['accent']},
    'sequence': {'title': '🔀 逻辑判断链', 'outputs': ['else'], 'color': COLORS['control']},
    'if_img':   {'title': '🔍 图像检测', 'outputs': ['yes', 'no'], 'color': COLORS['control']},
    'if_static':{'title': '⏸️ 静止检测', 'outputs': ['yes', 'no'], 'color': COLORS['control']},
    'if_sound': {'title': '🔊 声音检测', 'outputs': ['yes', 'no'], 'color': COLORS['sensor']},
    'set_var':  {'title': '[x] 设置变量', 'outputs': ['out'], 'color': COLORS['var_node']},
    'var_switch':{'title': '⎇ 变量分流', 'outputs': ['else'], 'color': COLORS['var_node']},
}

MOUSE_ACTIONS = {'click': '点击', 'move': '移动', 'drag': '拖拽', 'scroll': '滚动'}
MOUSE_BUTTONS = {'left': '左键', 'right': '右键', 'middle': '中键'}
MOUSE_CLICKS = {'1': '单击', '2': '双击'}
ACTION_MAP = {'click': '单击左键', 'double_click': '双击左键', 'right_click': '单击右键', 'none': '不执行操作'}
ACTION_MAP_REVERSE = {v: k for k, v in ACTION_MAP.items()}
MATCH_STRATEGY_MAP = {'hybrid': '智能混合', 'template': '模板匹配', 'feature': '特征匹配'}
MATCH_STRATEGY_REVERSE = {v: k for k, v in MATCH_STRATEGY_MAP.items()}
VAR_OP_MAP = {'=': '等于', '!=': '不等于', 'exists': '已定义', 'not_exists': '未定义'}
VAR_OP_REVERSE = {v: k for k, v in VAR_OP_MAP.items()}
PORT_TRANSLATION = {'out': '继续', 'yes': '是', 'no': '否', 'found': '找到', 'timeout': '超时', 'loop': '循环中', 'exit': '循环结束', 'else': '其他'}

# --- 5. 图标工厂 ---
class IconFactory:
    @staticmethod
    def create(name, color, size=(16, 16), bg=COLORS['bg_header']):
        img = Image.new('RGBA', size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        w, h = size
        if name == 'play': draw.polygon([(4, 3), (4, h-3), (w-3, h/2)], fill=color)
        elif name == 'stop': draw.rectangle((4, 4, w-4, h-4), fill=color)
        elif name == 'restore': draw.rectangle((6, 6, w-3, h-3), outline=color, width=2); draw.rectangle((3, 3, w-6, h-6), outline=color, width=2)
        elif name == 'quit': draw.line((3, 3, w-3, h-3), fill=color, width=2); draw.line((3, h-3, w-3, 3), fill=color, width=2)
        elif name == 'undo': draw.arc((3, 3, w-3, h-3), 0, 180, fill=color, width=2); draw.polygon([(3, h/2), (8, h/2-3), (8, h/2+3)], fill=color)
        elif name == 'redo': draw.arc((3, 3, w-3, h-3), 0, 180, fill=color, width=2); draw.polygon([(w-3, h/2), (w-8, h/2-3), (w-8, h/2+3)], fill=color)
        return ImageTk.PhotoImage(img)

# --- 6. 基础工具类 [增强健壮性] ---
class ImageUtils:
    @staticmethod
    def img_to_b64(image):
        try: buffered = io.BytesIO(); image.save(buffered, format="PNG"); return base64.b64encode(buffered.getvalue()).decode('utf-8')
        except: return None
    
    @staticmethod
    def b64_to_img(b64_str):
        if not b64_str or not isinstance(b64_str, str): return None
        try:
            # [安全修复] 补全 Padding
            missing_padding = len(b64_str) % 4
            if missing_padding: b64_str += '=' * (4 - missing_padding)
            return Image.open(io.BytesIO(base64.b64decode(b64_str)))
        except Exception: return None
    
    @staticmethod
    def make_thumb(image, size=(240, 135)):
        if not image: return None
        try: thumb = image.copy(); thumb.thumbnail(size); return ImageTk.PhotoImage(thumb)
        except: return None

class AudioEngine:
    @staticmethod
    def get_max_audio_peak():
        if not HAS_AUDIO: return 0.0
        try:
            try: comtypes.CoInitialize()
            except: pass
            sessions = AudioUtilities.GetAllSessions()
            max_peak = 0.0
            for session in sessions:
                if session.State == 1: 
                    meter = session._ctl.QueryInterface(IAudioMeterInformation)
                    peak = meter.GetPeakValue()
                    if peak > max_peak: max_peak = peak
            return max_peak
        except Exception: return 0.0

class VisionEngine:
    @staticmethod
    def capture_screen(bbox=None):
        try: return ImageGrab.grab(bbox=bbox)
        except OSError: return None # 处理屏幕锁定/无显示器情况

    @staticmethod
    def locate(needle, confidence=0.8, timeout=0, stop_event=None, grayscale=True, multiscale=True, scaling_ratio=1.0, strategy='hybrid', region=None):
        start_time = time.time()
        while True:
            if stop_event and stop_event.is_set(): return None
            capture_bbox = (region[0], region[1], region[0] + region[2], region[1] + region[3]) if region else None
            haystack = VisionEngine.capture_screen(bbox=capture_bbox)
            if haystack is None:
                time.sleep(0.5); 
                if timeout > 0 and time.time()-start_time>=timeout: break
                continue
            
            # [安全修复] 捕获匹配过程中的异常
            try:
                result, _ = VisionEngine._advanced_match(needle, haystack, confidence, stop_event, grayscale, multiscale, scaling_ratio, strategy)
                if result:
                    if region: return Box(result.left + region[0], result.top + region[1], result.width, result.height)
                    return result
            except Exception: pass

            if timeout > 0 and time.time()-start_time>=timeout: break
            time.sleep(0.1)
        return None

    @staticmethod
    def _advanced_match(needle, haystack, confidence, stop_event, grayscale, multiscale, scaling_ratio, strategy):
        if not needle or not haystack: return None, 0.0
        if needle.width > haystack.width or needle.height > haystack.height: return None, 0.0
        
        if HAS_OPENCV:
            try:
                if grayscale: nA, hA = cv2.cvtColor(np.array(needle), cv2.COLOR_RGB2GRAY), cv2.cvtColor(np.array(haystack), cv2.COLOR_RGB2GRAY)
                else: nA, hA = cv2.cvtColor(np.array(needle), cv2.COLOR_RGB2BGR), cv2.cvtColor(np.array(haystack), cv2.COLOR_RGB2BGR)
                if strategy == 'feature': return VisionEngine._feature_match_akaze(nA, hA)
                
                nH, nW = nA.shape[:2]; hH, hW = hA.shape[:2]; scales = [1.0]
                if multiscale: scales = np.unique(np.append(np.linspace(scaling_ratio * 0.8, scaling_ratio * 1.2, 10), [1.0, scaling_ratio]))
                
                best_max, best_rect = -1, None
                for s in scales:
                    if stop_event and stop_event.is_set(): return None, 0.0
                    tW, tH = int(nW * s), int(nH * s)
                    if tW < 5 or tH < 5 or tW > hW or tH > hH: continue
                    res = cv2.matchTemplate(hA, cv2.resize(nA, (tW, tH), interpolation=cv2.INTER_AREA), cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    if max_val > best_max: best_max, best_rect = max_val, Box(max_loc[0], max_loc[1], tW, tH)
                    if best_max > 0.99: break
                if best_rect and best_max >= confidence: return best_rect, best_max
            except Exception: pass
        
        # 降级处理
        try:
            res = pyautogui.locate(needle, haystack, confidence=confidence, grayscale=grayscale)
            if res: return Box(res.left, res.top, res.width, res.height), 1.0
        except: pass
        return None, 0.0

    @staticmethod
    def _feature_match_akaze(template, target, min_match_count=4):
        try:
            akaze = cv2.AKAZE_create()
            kp1, des1 = akaze.detectAndCompute(template, None); kp2, des2 = akaze.detectAndCompute(target, None)
            if des1 is None or des2 is None: return None, 0.0
            matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des1, des2, k=2)
            good = [m for m, n in matches if m.distance < 0.75 * n.distance]
            if len(good) >= min_match_count:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if M is not None:
                    h, w = template.shape[:2]
                    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)
                    x_min, y_min = np.min(dst[:, :, 0]), np.min(dst[:, :, 1])
                    x_max, y_max = np.max(dst[:, :, 0]), np.max(dst[:, :, 1])
                    return Box(int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)), min(1.0, len(good)/len(kp1)*2.5)
            return None, 0.0
        except: return None, 0.0

    @staticmethod
    def compare_images(img1, img2, threshold=0.99):
        if not img1 or not img2: return False
        try:
            if img1.size != img2.size: img2 = img2.resize(img1.size, Image.LANCZOS)
            diff = ImageChops.difference(img1.convert('L'), img2.convert('L'))
            return (1.0 - (sum(diff.histogram()[10:]) / (img1.size[0] * img1.size[1]))) >= threshold
        except: return False

# --- 7. 日志面板 ---
class LogPanel(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=COLORS['bg_panel'], **kwargs)
        self.expanded = False
        self.height_expanded, self.height_collapsed = 250, 30
        self.toolbar = tk.Frame(self, bg=COLORS['bg_header'], height=28); self.toolbar.pack_propagate(False)
        tk.Label(self.toolbar, text="📋 执行日志", bg=COLORS['bg_header'], fg='white', font=('Segoe UI', 9, 'bold')).pack(side='left', padx=10)
        tk.Button(self.toolbar, text="🔽", command=self.toggle, bg=COLORS['bg_header'], fg=COLORS['fg_text'], bd=0, relief='flat').pack(side='right', padx=5)
        tk.Button(self.toolbar, text="🗑️", command=self.clear, bg=COLORS['bg_header'], fg=COLORS['danger'], bd=0, relief='flat').pack(side='right', padx=5)
        self.text_frame = tk.Frame(self, bg=COLORS['log_bg'])
        self.scrollbar = ttk.Scrollbar(self.text_frame)
        self.text_area = tk.Text(self.text_frame, bg=COLORS['log_bg'], fg=COLORS['log_fg'], font=FONTS['log'], state='disabled', yscrollcommand=self.scrollbar.set, bd=0, padx=5, pady=5)
        self.scrollbar.config(command=self.text_area.yview); self.scrollbar.pack(side='right', fill='y'); self.text_area.pack(side='left', fill='both', expand=True)
        for level, style in LOG_LEVELS.items(): self.text_area.tag_config(level, foreground=style['color'])
        self.status_bar = tk.Frame(self, bg=COLORS['bg_panel'], height=30, cursor="hand2"); self.status_bar.pack_propagate(False); self.status_bar.pack(side='bottom', fill='x')
        self.status_lbl = tk.Label(self.status_bar, text="就绪", bg=COLORS['bg_panel'], fg=COLORS['fg_sub'], font=FONTS['code'], anchor='w', padx=10); self.status_lbl.pack(fill='both', expand=True)
        self.status_lbl.bind("<Button-1>", lambda e: self.toggle()); self.status_bar.bind("<Button-1>", lambda e: self.toggle())
    def toggle(self):
        self.expanded = not self.expanded
        if self.expanded: self.config(height=self.height_expanded); self.toolbar.pack(side='top', fill='x'); self.text_frame.pack(side='top', fill='both', expand=True)
        else: self.toolbar.pack_forget(); self.text_frame.pack_forget(); self.config(height=self.height_collapsed)
    def add_log(self, msg, level='info'):
        if not self.winfo_exists(): return
        icon = LOG_LEVELS.get(level, {}).get('icon', 'ℹ️')
        self.status_lbl.config(text=f"[{datetime.now().strftime('%H:%M:%S')}] {icon} {msg}")
        self.text_area.config(state='normal')
        self.text_area.insert('end', f"[{datetime.now().strftime('%H:%M:%S')}] [{level.upper()}] {msg}\n", level)
        self.text_area.see('end'); self.text_area.config(state='disabled')
    def clear(self): self.text_area.config(state='normal'); self.text_area.delete(1.0, 'end'); self.text_area.config(state='disabled')

# --- 8. 悬浮窗组件 ---
class FloatWindow(tk.Toplevel):
    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.overrideredirect(True)
        self.attributes("-topmost", True, "-alpha", 0.95)
        self.config(bg=COLORS['bg_header'])
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"250x35+{sw-270}+{sh-150}")
        self.bind('<Button-1>', self.clickwin); self.bind('<B1-Motion>', self.dragwin)
        main_frame = tk.Frame(self, bg=COLORS['bg_header']); main_frame.pack(fill='both', expand=True, padx=1, pady=1)
        self.status_label = tk.Label(main_frame, text="▶️ 就绪", bg=COLORS['bg_header'], fg=COLORS['fg_text'], font=FONTS['small'], anchor='w', padx=8)
        self.status_label.pack(side='left', fill='x', expand=True)
        self.status_label.bind('<Button-1>', self.clickwin); self.status_label.bind('<B1-Motion>', self.dragwin)
        btn_frame = tk.Frame(main_frame, bg=COLORS['bg_header']); btn_frame.pack(side='right')
        self.btn_style = {'bd': 0, 'relief': 'flat', 'bg': COLORS['bg_header'], 'fg': COLORS['fg_text'], 'activebackground': COLORS['bg_panel'], 'font': ('Segoe UI', 10)}
        self.btn_main = tk.Button(btn_frame, text="▶", command=self.toggle_main, **self.btn_style, width=3)
        self.btn_main.pack(side='left')
        tk.Button(btn_frame, text="❐", command=self.restore_app, **self.btn_style, width=3).pack(side='left')
        quit_btn_style = self.btn_style.copy(); quit_btn_style['fg'] = COLORS['danger']
        tk.Button(btn_frame, text="✕", command=self.app.quit_app, **quit_btn_style, width=3).pack(side='left')
        self.update_state()
    def toggle_main(self):
        if self.app.core.running: self.app.core.stop()
        else: self.app.toggle_run(None)
    def set_status_text(self, text):
        if not self.winfo_exists(): return
        state_icon = "▶️"
        if self.app.core.running: state_icon = "⏸️" if self.app.core.paused else "⏳"
        if len(text) > 15: text = text[:14] + "..."
        self.status_label.config(text=f"{state_icon} {text}")
    def clickwin(self, event): self._offsetx = event.x; self._offsety = event.y
    def dragwin(self, event): x = self.winfo_x() + (event.x - self._offsetx); y = self.winfo_y() + (event.y - self._offsety); self.geometry(f"+{x}+{y}")
    def restore_app(self): self.app.deiconify(); self.app.state('normal'); self.destroy(); self.app.float_window = None
    def update_state(self):
        if not self.winfo_exists(): return
        if self.app.core.running:
            self.btn_main.config(text="■", fg=COLORS['danger'])
            if self.app.core.paused: self.set_status_text("暂停中...")
        else:
            self.btn_main.config(text="▶", fg=COLORS['success'])
            self.set_status_text("就绪")

# --- 9. 自动化核心引擎 [加固版] ---
class AutomationCore:
    def __init__(self, log_callback, app_instance):
        self.running = False; self.paused = False; self.step_mode = False
        self.stop_event = threading.Event(); self.pause_event = threading.Event(); self.step_event = threading.Event()
        self.log = log_callback; self.app = app_instance; self.project = None; self.runtime_memory = {}; self.io_lock = threading.Lock()
        self.active_threads = 0; self.thread_lock = threading.Lock(); self.scaling_ratio = 1.0; self.breakpoints = set()
        # [安全增强] 最大线程限制
        self.max_threads = 50 

    def load_project(self, project_data):
        self.project = project_data; self.scaling_ratio = 1.0; self.breakpoints = set(project_data.get('breakpoints', []))
        dev_scale = self.project.get('metadata', {}).get('dev_scale_x', 1.0); runtime_scale_x, _ = get_scale_factor()
        if dev_scale > 0.1 and runtime_scale_x > 0.1: self.scaling_ratio = runtime_scale_x / dev_scale
        if self.project and 'nodes' in self.project:
            for nid, node in self.project['nodes'].items():
                data = node.get('data', {})
                # 安全加载图片资源
                try:
                    if 'b64' in data and 'image' not in data and (img := ImageUtils.b64_to_img(data['b64'])): self.project['nodes'][nid]['data']['image'] = img
                    if 'anchors' in data:
                        for anchor in data['anchors']:
                            if 'b64' in anchor and 'image' not in anchor and (img := ImageUtils.b64_to_img(anchor['b64'])): anchor['image'] = img
                    if 'images' in data:
                        for img_item in data['images']:
                            if 'b64' in img_item and 'image' not in img_item and (img := ImageUtils.b64_to_img(img_item['b64'])): img_item['image'] = img
                    if 'b64_preview' in data and 'roi_preview' not in data and (img := ImageUtils.b64_to_img(data['b64_preview'])):
                        self.project['nodes'][nid]['data']['roi_preview'] = ImageUtils.make_thumb(img)
                except Exception: pass

    def start(self, start_node_id=None):
        if self.running or not self.project: return
        self.running = True; self.paused = False; self.stop_event.clear(); self.pause_event.set(); self.step_event.clear(); self.runtime_memory = {}; self.active_threads = 0
        self.log("🚀 引擎启动", "exec"); self.app.iconify()
        self.app.btn_run.config(text="⏹ 停止运行", bg=COLORS['danger']); self.app.btn_run_menu.config(state='disabled')
        self.app.show_debug_toolbar(True)
        if self.app.float_window: self.app.float_window.update_state()
        threading.Thread(target=self._run_flow_engine, args=(start_node_id,), daemon=True).start()

    def stop(self):
        if not self.running: return
        self.stop_event.set(); self.pause_event.set(); self.step_event.set()
        self.log("🛑 正在停止...", "warning"); self.app.reset_ui_state()

    def pause(self): self.paused = True; self.pause_event.clear(); self.log("⏸️ 流程已暂停", "paused"); self.app.update_debug_btn_state(paused=True); self.app.update_float_status_safe("暂停中...")
    def resume(self): self.paused = False; self.pause_event.set(); self.log("▶️ 流程继续", "info"); self.app.update_debug_btn_state(paused=False); self.app.update_float_status_safe("运行中...")
    def step(self): self.step_event.set()
    def _smart_wait(self, seconds):
        end_time = time.time() + seconds
        while time.time() < end_time:
            if self.stop_event.is_set(): return False
            self._check_pause(); time.sleep(0.05)
        return True
    
    def _check_pause(self, node_id=None):
        if node_id and node_id in self.breakpoints:
            if not self.paused: self.log(f"🔴 命中断点: {node_id}", "paused"); self.pause(); self.app.deiconify()
        if not self.pause_event.is_set():
            if self.step_event.is_set(): self.step_event.clear(); return 
            else: self.pause_event.wait()
    
    def _get_next_links(self, node_id, port_name='out'): return [l['target'] for l in self.project['links'] if l['source'] == node_id and l.get('source_port') == port_name]
    
    def _run_flow_engine(self, start_node_id=None):
        try:
            start_nodes = [start_node_id] if start_node_id else [nid for nid, n in self.project['nodes'].items() if n['type'] == 'start']
            if not start_nodes: self.log("未找到 '开始' 节点或目标节点", "error"); return
            for start_id in start_nodes: self._fork_node(start_id)
            while not self.stop_event.is_set():
                with self.thread_lock: 
                    if self.active_threads <= 0: break
                time.sleep(0.5)
        except Exception as e: traceback.print_exc(); self.log(f"引擎异常: {str(e)}", "error")
        finally:
            self.running = False
            self.log("🏁 流程结束", "info")
            self.app.highlight_node_safe(None)
            if not self.app.float_window:
                self.app.after(0, self.app.deiconify)
            self.app.after(100, self.app.reset_ui_state)

    def _fork_node(self, node_id):
        with self.thread_lock:
            # [安全增强] 防止节点ID不存在
            if node_id not in self.project['nodes']: return
            # [安全增强] 防止线程爆炸
            if self.active_threads >= self.max_threads:
                self.log(f"🔥 达到最大线程数限制 ({self.max_threads})，丢弃分支执行", "error")
                return
            self.active_threads += 1
        threading.Thread(target=self._process_node_thread, args=(node_id,), daemon=True).start()

    def _process_node_thread(self, node_id):
        try:
            if self.stop_event.is_set(): return
            if not (node := self.project['nodes'].get(node_id)): return
            self._check_pause(node_id)
            if self.stop_event.is_set(): return
            self.app.last_executed_node = node_id
            self.app.highlight_node_safe(node_id, 'running'); self.app.select_node_safe(node_id)
            node_name = node['data'].get('_user_title', NODE_CONFIG.get(node['type'], {}).get('title', node['type']))
            self.app.update_float_status_safe(node_name)
            
            # [安全增强] 捕获单个节点执行异常
            try:
                out_port = self._execute_node(node)
            except Exception as e:
                self.log(f"💥 节点 [{node_name}] 执行崩溃: {e}", "error")
                traceback.print_exc()
                out_port = 'else'

            if out_port == '__STOP__' or self.stop_event.is_set(): return
            self.log(f"  ↳ 出口: [{PORT_TRANSLATION.get(out_port, out_port)}]", "exec")
            self.app.highlight_node_safe(node_id, 'fail' if out_port in ['timeout', 'no', 'exit', 'else'] else 'ok')
            
            # [安全增强] 避免死循环导致 CPU 100%
            time.sleep(0.001)

            for next_id in self._get_next_links(node_id, out_port):
                if self.stop_event.is_set(): break
                self._fork_node(next_id)
        finally:
            with self.thread_lock: self.active_threads -= 1

    def _replace_variables(self, text):
        # [安全增强] 类型检查
        if not isinstance(text, str): return str(text)
        if not text: return text
        try:
            for var_name, var_value in self.runtime_memory.items(): 
                val_str = str(var_value) if var_value is not None else ""
                text = text.replace(f'${{{var_name}}}', val_str)
        except: pass
        return text

    def _execute_node(self, node):
        if self.stop_event.is_set(): return '__STOP__'
        
        # [安全增强] 预处理数据
        ntype = node['type']
        raw_data = node.get('data', {})
        data = {}
        for k, v in raw_data.items():
            if isinstance(v, str) and '${' in v: data[k] = self._replace_variables(v)
            else: data[k] = v

        text = data.get('_user_title', NODE_CONFIG.get(ntype, {}).get('title', ntype))
        self.log(f"执行: {text}", "exec")
        if ntype == 'start': return 'out'
        if ntype == 'end': self.stop_event.set(); return '__STOP__'
        
        if ntype == 'wait': 
            return 'out' if self._smart_wait(safe_float(data.get('seconds', 1.0))) else '__STOP__'
        
        if ntype == 'set_var':
            if 'batch_vars' in data: [self.runtime_memory.update({i['name']:i['value']}) for i in data['batch_vars'] if i.get('name')]
            if data.get('var_name'): self.runtime_memory[data['var_name']] = data.get('var_value', '')
            return 'out'
        
        if ntype == 'var_switch':
            if data.get('var_name'):
                val = str(self.runtime_memory.get(data['var_name'], ''))
                op, target = data.get('operator', '='), data.get('var_value', '')
                match = (val == target) if op == '=' else (val != target) if op == '!=' else (bool(val)) if op == 'exists' else (not bool(val))
                return 'yes' if match else 'no'
            else:
                var_names = [v.strip() for v in data.get('var_list', '').split(',') if v.strip()]
                vals = [str(self.runtime_memory.get(vn, '')) for vn in var_names]
                for case in data.get('cases', []):
                    if all(v == case.get('value', '') for v in vals): return case.get('id', 'else')
                return 'else'

        if ntype == 'sequence':
            for i in range(1, safe_int(data.get('num_steps', 3)) + 1):
                if self.stop_event.is_set(): return '__STOP__'
                target_id = (self._get_next_links(node['id'], str(i)) or [None])[0]
                if not target_id: continue
                if target_id in self.project['nodes']:
                     self.app.highlight_node_safe(target_id, 'running'); self.app.select_node_safe(target_id); self._check_pause(target_id)
                if self._execute_node(self.project['nodes'][target_id]) in ['yes', 'found', 'out', 'loop']:
                    self.app.highlight_node_safe(target_id, 'ok'); 
                    [self._fork_node(nid) for nid in self._get_next_links(target_id, self._execute_node(self.project['nodes'][target_id]))]
                    return '__STOP__'
                else: self.app.highlight_node_safe(target_id, 'fail')
            return 'else'

        if ntype == 'if_sound':
            if not HAS_AUDIO: return 'no'
            mode, thr, timeout = data.get('detect_mode','has_sound'), safe_float(data.get('threshold',0.02)), safe_float(data.get('timeout',10.0))
            dur = safe_float(data.get('duration',3.0)); start_t, silence_start = time.time(), time.time()
            while True:
                if self.stop_event.is_set(): return '__STOP__'
                self._check_pause(); peak = AudioEngine.get_max_audio_peak()
                if time.time()-start_t>timeout: return 'no'
                if mode == 'has_sound' and peak > thr: return 'yes'
                if mode == 'is_silent':
                    if peak > thr: silence_start = time.time()
                    elif time.time() - silence_start >= dur: return 'yes'
                time.sleep(0.1)

        if ntype == 'image':
            conf, timeout = safe_float(data.get('confidence', 0.9)), max(1.0, safe_float(data.get('timeout', 10.0)))
            search_region = None
            if data.get('search_mode') == 'region':
                 try: search_region = (safe_int(data.get('target_rect_x')), safe_int(data.get('target_rect_y')), safe_int(data.get('target_rect_w')), safe_int(data.get('target_rect_h')))
                 except: pass

            if (anchors := data.get('anchors', [])):
                primary_res = None
                for i, anchor in enumerate(anchors):
                    if self.stop_event.is_set(): return '__STOP__'
                    res = VisionEngine.locate(anchor['image'], confidence=conf, timeout=(timeout if i==0 else 2.0), stop_event=self.stop_event, strategy=data.get('match_strategy','hybrid'))
                    if not res: return 'timeout'
                    if i == 0: primary_res = res
                if primary_res:
                    off_x, off_y = safe_int(data.get('target_rect_x',0))-anchors[0].get('rect_x',0), safe_int(data.get('target_rect_y',0))-anchors[0].get('rect_y',0)
                    search_region = (max(0, int(primary_res.left+off_x)-15), max(0, int(primary_res.top+off_y)-15), safe_int(data.get('target_rect_w',100))+30, safe_int(data.get('target_rect_h',100))+30)

            start_time = time.time()
            while True:
                if self.stop_event.is_set(): return '__STOP__'
                self._check_pause()
                res = VisionEngine.locate(data.get('image'), confidence=conf, timeout=0, stop_event=self.stop_event, region=search_region, strategy=data.get('match_strategy','hybrid'))
                if res:
                    with self.io_lock:
                        if (act := data.get('click_type', 'click')) != 'none':
                            rx, ry = data.get('relative_click_pos', (0.5, 0.5))
                            tx, ty = res.left + (res.width * rx) + safe_int(data.get('offset_x', 0)), res.top + (res.height * ry) + safe_int(data.get('offset_y', 0))
                            pyautogui.moveTo(tx / SCALE_X, ty / SCALE_Y)
                            getattr(pyautogui, {'click':'click','double_click':'doubleClick','right_click':'rightClick'}.get(act, 'click'))()
                    return 'found'
                if data.get('auto_scroll', False):
                    with self.io_lock: pyautogui.scroll(safe_int(data.get('scroll_amount', -500)))
                    if not self._smart_wait(0.8): return '__STOP__'
                if time.time() - start_time > timeout: break
                time.sleep(0.2)
            return 'timeout'

        if ntype == 'mouse':
            with self.io_lock:
                action = data.get('mouse_action', 'click')
                dur = safe_float(data.get('duration', 0.5))
                btn = data.get('mouse_button', 'left')
                
                target_x, target_y = None, None
                if action in ['click', 'move', 'drag']:
                    try:
                        target_x = safe_int(data.get('x', 0)) / SCALE_X
                        target_y = safe_int(data.get('y', 0)) / SCALE_Y
                    except: pass

                if action == 'click':
                    clicks = safe_int(data.get('click_count', 1))
                    if target_x is not None and target_y is not None:
                        pyautogui.click(x=target_x, y=target_y, clicks=clicks, button=btn, duration=dur)
                    else:
                        pyautogui.click(clicks=clicks, button=btn)
                elif action == 'move':
                    if target_x is not None and target_y is not None:
                        pyautogui.moveTo(target_x, target_y, duration=dur)
                elif action == 'drag':
                    if target_x is not None and target_y is not None:
                        pyautogui.dragTo(target_x, target_y, duration=dur, button=btn)
                elif action == 'scroll':
                    amount = safe_int(data.get('amount', -500))
                    pyautogui.scroll(amount)
            return 'out'

        if ntype == 'keyboard':
            self._smart_wait(0.2)
            with self.io_lock:
                if data.get('kb_mode', 'text') == 'text':
                     if not data.get('slow_type', False) and HAS_PYPERCLIP:
                        try: pyperclip.copy(data.get('text','')); pyautogui.hotkey('ctrl','v')
                        except: pyautogui.write(data.get('text',''))
                     else: pyautogui.write(data.get('text',''))
                     if data.get('press_enter', False): pyautogui.press('enter')
                else: pyautogui.hotkey(*[x.strip() for x in data.get('key_name', 'enter').lower().split('+')])
            return 'out'
        
        if ntype == 'web': webbrowser.open(data.get('url')); self._smart_wait(2); return 'out'

        if ntype == 'loop':
            if data.get('infinite', True):
                return 'loop'
            with self.io_lock:
                k = f"loop_{node['id']}"; c = self.runtime_memory.get(k, 0)
                if c < safe_int(data.get('count', 3)): self.runtime_memory[k] = c + 1; return 'loop'
                else: 
                    if k in self.runtime_memory: del self.runtime_memory[k]
                    return 'exit'

        if ntype == 'if_static':
            roi = data.get('roi')
            if not roi: return 'no'
            dur, thr, max_t, interval = safe_float(data.get('duration',5.0)), safe_float(data.get('threshold',0.98)), safe_float(data.get('timeout',20.0)), safe_float(data.get('interval',1.0))
            start_t, static_start, last_frame = time.time(), time.time(), VisionEngine.capture_screen(bbox=roi)
            while True:
                if self.stop_event.is_set(): return '__STOP__'
                self._check_pause()
                if time.time()-start_t > max_t: return 'no'
                if not self._smart_wait(interval): return '__STOP__'
                curr = VisionEngine.capture_screen(bbox=roi)
                if not last_frame or not curr: continue
                is_static = VisionEngine.compare_images(last_frame, curr, thr)
                if is_static:
                    if time.time() - static_start >= dur: return 'yes'
                else:
                    static_start = time.time(); last_frame = curr
        
        if ntype == 'if_img':
            if not (imgs := data.get('images', [])): return 'no'
            hay = VisionEngine.capture_screen()
            for img in imgs:
                res, _ = VisionEngine._advanced_match(img.get('image'), hay, safe_float(data.get('confidence',0.9)), self.stop_event, data.get('use_grayscale',True), data.get('use_multiscale',True), self.scaling_ratio, data.get('match_strategy','hybrid'))
                if not res: return 'no'
            return 'yes'

        return 'out'

# --- 10. 历史记录管理器 ---
class HistoryManager:
    def __init__(self, editor):
        self.editor = editor; self.undo_stack = []; self.redo_stack = []; self.max_history = 50
    def save_state(self):
        state = self.editor.get_data()
        if self.undo_stack and json.dumps(state, sort_keys=True) == json.dumps(self.undo_stack[-1], sort_keys=True): return
        self.undo_stack.append(state); self.redo_stack.clear()
        if len(self.undo_stack) > self.max_history: self.undo_stack.pop(0)
    def undo(self, event=None):
        if not self.undo_stack: return
        self.redo_stack.append(self.editor.get_data())
        self.editor.load_data(self.undo_stack.pop()); self.editor.app.property_panel.clear()
        self.editor.app.log("↩ 撤销", "info")
    def redo(self, event=None):
        if not self.redo_stack: return
        self.undo_stack.append(self.editor.get_data())
        self.editor.load_data(self.redo_stack.pop()); self.editor.app.property_panel.clear()
        self.editor.app.log("↪ 重做", "info")

# --- 11. UI 组件 ---
class GraphNode:
    def __init__(self, canvas, node_id, ntype, x, y, data=None):
        self.canvas, self.id, self.type, self.x, self.y = canvas, node_id, ntype, x, y
        self.data = data if data is not None else {}
        cfg = NODE_CONFIG.get(ntype, {})
        self.title_text, self.header_color = cfg.get('title', ntype), cfg.get('color', COLORS['bg_header'])
        if '_user_title' not in self.data: self.data['_user_title'] = f"{self.title_text} {sum(1 for n in self.canvas.nodes.values() if n.type == ntype) + 1}"
        
        if ntype == 'sequence': self.outputs = [str(i) for i in range(1, safe_int(self.data.get('num_steps', 3)) + 1)] + ['else']
        elif ntype == 'var_switch':
            if self.data.get('var_name'): self.outputs = ['yes', 'no']
            else: self.outputs = [c['id'] for c in self.data.get('cases', [])] + ['else']
        else: self.outputs = cfg.get('outputs', [])

        self.w, self.h = NODE_WIDTH, PORT_START_Y + max(1, len(self.outputs)) * PORT_STEP_Y if self.outputs else 50
        self.tags = (f"node_{self.id}", "node"); self.has_breakpoint = False; self.draw()

    def draw(self):
        z = self.canvas.zoom; vx, vy, vw, vh = self.x*z, self.y*z, self.w*z, self.h*z
        self.canvas.delete(f"node_{self.id}")
        self.sel_rect = self.canvas.create_rectangle(vx - 3*z, vy - 3*z, vx + vw + 3*z, vy + vh + 3*z, outline=COLORS['accent'], width=4*z, tags=self.tags+('selection',), state='hidden')
        self.canvas.create_rectangle(vx+4*z,vy+4*z,vx+vw+4*z,vy+vh+4*z,fill=COLORS['shadow'],outline="",tags=self.tags)
        self.body_item=self.canvas.create_rectangle(vx,vy,vx+vw,vy+vh,fill=COLORS['bg_node'],outline=COLORS['bg_node'],width=2*z,tags=self.tags+('body',))
        self.canvas.create_rectangle(vx,vy,vx+vw,vy+HEADER_HEIGHT*z,fill=self.header_color,outline="",tags=self.tags+('header',))
        
        truncated_title = (self.data.get('_user_title', self.title_text)[:15] + '...') if len(self.data.get('_user_title', self.title_text)) > 15 else self.data.get('_user_title', self.title_text)
        self.canvas.create_text(vx+10*z, vy+14*z, text=truncated_title, fill=COLORS['fg_title'], font=('Segoe UI', max(6, int(10*z)), 'bold'), anchor="w", tags=self.tags)
        
        if self.has_breakpoint: self.canvas.create_oval(vx+vw-12*z, vy+8*z, vx+vw-4*z, vy+16*z, fill=COLORS['breakpoint'], outline="white", width=1, tags=self.tags)
        if self.type != 'start':
            iy=self.get_input_port_y(visual=True);self.canvas.create_oval(vx-5*z,iy-5*z,vx+5*z,iy+5*z,fill=COLORS['socket'],outline=COLORS['bg_canvas'],width=2*z,tags=self.tags+('port_in',))
        
        port_labels = {'out':'继续', 'yes':'是', 'no':'否', 'found':'找到', 'timeout':'超时', 'else':'Else'}
        if self.type == 'var_switch':
             for c in self.data.get('cases', []): port_labels[c['id']] = f"={c['value']}"

        for i,name in enumerate(self.outputs):
            py=self.get_output_port_y(i,visual=True)
            self.canvas.create_oval(vx+vw-5*z,py-5*z,vx+vw+5*z,py+5*z,fill=COLORS.get(f"socket_{name}", COLORS['socket']),outline=COLORS['bg_canvas'],width=2*z,tags=self.tags+(f'port_out_{name}','port_out',name))
            self.canvas.create_text(vx+vw-12*z,py,text=port_labels.get(name,name),fill=COLORS['fg_sub'],font=('Segoe UI', max(5, int(8*z))),anchor="e",tags=self.tags)
        
        if self.id in self.canvas.selected_node_ids: self.canvas.itemconfig(self.sel_rect, state='normal'); self.canvas.tag_lower(self.sel_rect, self.body_item)
        self.hover_rect=self.canvas.create_rectangle(vx-1*z,vy-1*z,vx+vw+1*z,vy+vh+1*z,outline=COLORS['hover'],width=1*z,state='hidden',tags=self.tags+('hover',))
    
    def set_sensor_active(self,is_active): self.canvas.itemconfig(self.body_item,outline=COLORS['active_border'] if is_active else COLORS['bg_node'])
    def get_input_port_y(self,visual=False): offset=HEADER_HEIGHT+14; return (self.y+offset)*self.canvas.zoom if visual else self.y+offset
    def get_output_port_y(self,index=0,visual=False): offset=PORT_START_Y+(index*PORT_STEP_Y); return (self.y+offset)*self.canvas.zoom if visual else self.y+offset
    def get_port_y_by_name(self,port_name,visual=False):
        try: idx=self.outputs.index(port_name)
        except ValueError: idx=0
        return self.get_output_port_y(idx,visual)
    def set_pos(self,x,y): self.x,self.y=x,y; self.draw()
    def set_selected(self,selected): self.canvas.itemconfig(self.sel_rect,state='normal' if selected else 'hidden'); (selected and self.canvas.tag_lower(self.sel_rect, self.body_item))
    def contains(self,log_x,log_y): return self.x<=log_x<=self.x+self.w and self.y<=log_y<=self.y+self.h
    def update_data(self,key,value): self.canvas.history.save_state(); self.data[key]=value; (key=='_user_title' and self.draw())

class FlowEditor(tk.Canvas):
    def __init__(self,parent,app,**kwargs):
        super().__init__(parent,bg=COLORS['bg_canvas'],highlightthickness=0,**kwargs)
        self.app,self.nodes,self.links=app,{},[]
        self.selected_node_ids = set(); self.drag_data = {"type": None}; self.wire_start = None; self.temp_wire = None; self.selection_box = None
        self.history = HistoryManager(self); self.zoom=1.0; self.bind_events(); self.full_redraw()
        
    @property
    def selected_node_id(self): return next(iter(self.selected_node_ids)) if self.selected_node_ids else None
    
    def bind_events(self):
        self.bind("<ButtonPress-1>",self.on_lmb_press);self.bind("<B1-Motion>",self.on_lmb_drag);self.bind("<ButtonRelease-1>",self.on_lmb_release)
        self.bind("<ButtonPress-3>",self.on_rmb_press);self.bind("<B3-Motion>",self.on_rmb_drag);self.bind("<ButtonRelease-3>",self.on_rmb_release)
        self.bind("<ButtonPress-2>",self.on_pan_start);self.bind("<B2-Motion>",self.on_pan_drag);self.bind("<ButtonRelease-2>",self.on_pan_end)
        self.bind("<MouseWheel>",self.on_scroll)
        self.bind_all("<Delete>",self._on_delete_press,add="+");self.bind_all("<Control-z>", self.history.undo, add="+"); self.bind_all("<Control-y>", self.history.redo, add="+")
        self.bind("<Motion>",self.on_mouse_move); self.bind("<Configure>",self.full_redraw)
    
    def on_rmb_press(self, event): self._rmb_start = (event.x, event.y); self._rmb_moved = False; self.scan_mark(event.x, event.y)
    def on_rmb_drag(self, event): (abs(event.x-self._rmb_start[0])>5 or abs(event.y-self._rmb_start[1])>5) and setattr(self,'_rmb_moved',True); self._rmb_moved and (self.config(cursor="fleur"),self.scan_dragto(event.x, event.y, gain=1),self._draw_grid())
    def on_rmb_release(self, event): self.config(cursor="arrow"); (not getattr(self,'_rmb_moved',False) and self.on_right_click_menu(event))
    
    def on_pan_start(self,event): self.config(cursor="fleur");self.scan_mark(event.x,event.y)
    def on_pan_drag(self,event): self.scan_dragto(event.x,event.y,gain=1);self._draw_grid()
    def on_pan_end(self,event): self.config(cursor="arrow")

    def _on_delete_press(self,e): (self.selected_node_ids and (self.history.save_state(), [self.delete_node(nid) for nid in list(self.selected_node_ids)]))
    def on_mouse_move(self,event):
        lx,ly=self.get_logical_pos(event.x,event.y)
        [self.itemconfig(n.hover_rect, state='normal' if n.contains(lx, ly) and n.id not in self.selected_node_ids else 'hidden') for n in self.nodes.values()]
    def get_logical_pos(self,event_x,event_y): return self.canvasx(event_x)/self.zoom,self.canvasy(event_y)/self.zoom
    def full_redraw(self,event=None): self.delete("all");self._draw_grid(); [n.draw() for n in self.nodes.values()]; self.redraw_links()
    def _draw_grid(self):
        w,h=self.winfo_width(),self.winfo_height(); x1,y1,x2,y2=self.canvasx(0),self.canvasy(0),self.canvasx(w),self.canvasy(h)
        if (step:=int(GRID_SIZE*self.zoom))<5: return
        start_x,start_y=int(x1//step)*step,int(y1//step)*step
        for i in range(start_x,int(x2)+step,step): self.create_line(i,y1,i,y2,fill=COLORS['grid'],tags="grid")
        for i in range(start_y,int(y2)+step,step): self.create_line(x1,i,x2,i,fill=COLORS['grid'],tags="grid")
        self.tag_lower("grid")
    
    def add_node(self,ntype,x,y,data=None,node_id=None, save_history=True): 
        if save_history: self.history.save_state()
        node=GraphNode(self,node_id or str(uuid.uuid4()),ntype,x,y,data)
        self.nodes[node.id]=node; self.select_node(node.id); return node
    
    def delete_node(self,node_id):
        if node_id in self.nodes:
            # [安全修复] 删除连线
            self.links = [l for l in self.links if l['source'] != node_id and l['target'] != node_id]
            self.delete(f"node_{node_id}"); del self.nodes[node_id]
            if node_id in self.selected_node_ids: 
                self.selected_node_ids.remove(node_id)
                if not self.selected_node_ids:
                    self.app.property_panel.show_empty()
            self.redraw_links()
    
    def duplicate_node(self,node_id):
        if not (src:=self.nodes.get(node_id)): return
        self.history.save_state()
        new_data=copy.deepcopy({k:v for k,v in src.data.items() if k not in ['image','tk_image','images','roi_preview','anchors']})
        new_data['_user_title'] = new_data.get('_user_title', src.title_text) + " (Copy)"
        if 'b64' in src.data and (img := ImageUtils.b64_to_img(src.data['b64'])): new_data.update({'image':img,'tk_image':ImageUtils.make_thumb(img)})
        if 'b64_preview' in src.data and (img := ImageUtils.b64_to_img(src.data['b64_preview'])): new_data.update({'roi_preview':ImageUtils.make_thumb(img)})
        if 'anchors' in src.data:
            new_data['anchors'] = []
            for anc in src.data['anchors']:
                new_anc = copy.deepcopy({k:v for k,v in anc.items() if k not in ['image']})
                if 'b64' in new_anc and (img := ImageUtils.b64_to_img(new_anc['b64'])): new_anc['image'] = img
                new_data['anchors'].append(new_anc)
        if 'images' in src.data:
            new_data['images'] = []
            for item in src.data['images']:
                new_item = copy.deepcopy({k:v for k,v in item.items() if k not in ['image', 'tk_image']})
                if 'b64' in new_item and (img := ImageUtils.b64_to_img(new_item['b64'])): new_item.update({'image':img,'tk_image':ImageUtils.make_thumb(img,size=(120,67))})
                new_data['images'].append(new_item)
        self.add_node(src.type,src.x+20,src.y+20,data=new_data,save_history=False)

    def on_scroll(self, e):
        old_zoom = self.zoom; new_zoom = max(0.4, min(3.0, self.zoom * (1.1 if e.delta > 0 else 0.9)))
        if new_zoom == self.zoom: return
        self.zoom = new_zoom; self.full_redraw()
        self.scan_mark(e.x, e.y); self.scan_dragto(int(e.x - (e.x * (new_zoom/old_zoom - 1))), int(e.y - (e.y * (new_zoom/old_zoom - 1))), gain=1); self._draw_grid()

    def on_lmb_press(self,event):
        lx,ly=self.get_logical_pos(event.x,event.y); vx,vy=self.canvasx(event.x),self.canvasy(event.y)
        for item in self.find_overlapping(vx-2,vy-2,vx+2,vy+2):
            tags=self.gettags(item)
            if "port_out" in tags and (nid:=next((t[5:] for t in tags if t.startswith("node_")),None)) and nid in self.nodes:
                self.wire_start={'node':self.nodes[nid],'port':next((t for t in tags if t in self.nodes[nid].outputs),'out')};self.drag_data={"type":"wire"}; return
        
        clicked_node=next((node for node in reversed(list(self.nodes.values())) if node.contains(lx,ly)),None)
        if clicked_node:
            is_selected = clicked_node.id in self.selected_node_ids
            ctrl_pressed = (event.state & 0x0004)
            if not ctrl_pressed:
                if not is_selected: self.select_node(clicked_node.id)
            else: self.select_node(clicked_node.id, add=True)
            self.drag_data = {"type": "node", "start_lx": lx, "start_ly": ly, "start_pos": {nid: (self.nodes[nid].x, self.nodes[nid].y) for nid in self.selected_node_ids}, "dragged": False}
            self.history.save_state(); [self.tag_raise(f"node_{nid}") for nid in self.selected_node_ids]
        else:
            if not (event.state & 0x0004): self.select_node(None)
            self.drag_data = {"type": "box_select", "start_vx": vx, "start_vy": vy}; self.selection_box = self.create_rectangle(vx, vy, vx, vy, outline=COLORS['select_box'], width=2, dash=(4,4), tags="selection_box")

    def on_lmb_drag(self,event):
        lx,ly=self.get_logical_pos(event.x,event.y); vx,vy=self.canvasx(event.x),self.canvasy(event.y)
        if self.drag_data["type"]=="node":
            self.drag_data["dragged"] = True; dx, dy = lx - self.drag_data["start_lx"], ly - self.drag_data["start_ly"]
            for nid, (sx, sy) in self.drag_data.get("start_pos", {}).items():
                if nid in self.nodes: self.nodes[nid].x, self.nodes[nid].y = sx + dx, sy + dy; self.nodes[nid].draw()
            self.redraw_links()
        elif self.drag_data["type"]=="box_select":
            if self.selection_box: self.coords(self.selection_box, self.drag_data["start_vx"], self.drag_data["start_vy"], vx, vy)
        elif self.drag_data["type"]=="wire":
            if self.temp_wire: self.delete(self.temp_wire)
            n,p=self.wire_start['node'],self.wire_start['port']
            self.temp_wire=self.draw_bezier((n.x+n.w)*self.zoom,n.get_port_y_by_name(p,visual=True),vx,vy,state="active")

    def on_lmb_release(self,event):
        if self.drag_data.get("type")=="node":
            if self.drag_data.get("dragged", False):
                [self.nodes[nid].set_pos(round(self.nodes[nid].x/GRID_SIZE)*GRID_SIZE, round(self.nodes[nid].y/GRID_SIZE)*GRID_SIZE) for nid in self.selected_node_ids if nid in self.nodes]; self.redraw_links()
            else: 
                if self.history.undo_stack: self.history.undo_stack.pop()
        elif self.drag_data.get("type")=="box_select":
            if self.selection_box:
                coords = self.coords(self.selection_box); overlapping = self.find_overlapping(*coords)
                [self.select_node(t[5:], add=True) for item in overlapping for t in self.gettags(item) if t.startswith("node_") and t[5:] in self.nodes]
                self.delete(self.selection_box); self.selection_box = None
        elif self.drag_data.get("type")=="wire":
            if self.temp_wire: self.delete(self.temp_wire)
            lx,ly=self.get_logical_pos(event.x,event.y)
            for node in self.nodes.values():
                if node.type != 'start' and node.id!=self.wire_start['node'].id and math.hypot(lx-node.x,ly-node.get_input_port_y(visual=False))<(25/self.zoom):
                    self.history.save_state(); self.links.append({'id':str(uuid.uuid4()),'source':self.wire_start['node'].id,'source_port':self.wire_start['port'],'target':node.id}); self.redraw_links(); break
        self.drag_data,self.wire_start,self.temp_wire={"type":None},None,None
    
    def select_node(self, node_id, add=False):
        if not add: [self.nodes[nid].set_selected(False) for nid in self.selected_node_ids if nid in self.nodes]; self.selected_node_ids.clear()
        if node_id and node_id in self.nodes: 
            self.selected_node_ids.add(node_id); self.nodes[node_id].set_selected(True); self.app.property_panel.load_node(self.nodes[node_id])
        elif not add: 
            self.app.property_panel.show_empty()
        self.redraw_links()

    def draw_bezier(self,x1,y1,x2,y2,state="normal",link_id=None, highlighted=False):
        offset=max(50*self.zoom,abs(x1-x2)*0.5); width = 4*self.zoom if highlighted else (3*self.zoom if state=="active" else 2*self.zoom)
        color = COLORS['wire_hl'] if highlighted else COLORS['wire_active' if state=="active" else 'wire']
        return self.create_line(x1,y1,x1+offset,y1,x2-offset,y2,x2,y2,smooth=True,splinesteps=50,fill=color,width=width,arrow=tk.LAST,arrowshape=(8*self.zoom,10*self.zoom,3*self.zoom),tags=("link",)+((f"link_{link_id}",) if link_id else ()))
    
    def redraw_links(self):
        self.delete("link"); 
        for l in self.links:
            if l['source'] in self.nodes and l['target'] in self.nodes:
                n1,n2=self.nodes[l['source']],self.nodes[l['target']]
                self.draw_bezier((n1.x+n1.w)*self.zoom,n1.get_port_y_by_name(l.get('source_port','out'),visual=True),n2.x*self.zoom,n2.get_input_port_y(visual=True),link_id=l['id'], highlighted=(l['source'] in self.selected_node_ids or l['target'] in self.selected_node_ids))
        self.tag_lower("link"); self.tag_lower("grid")
    
    def on_right_click_menu(self,event):
        vx,vy=self.canvasx(event.x),self.canvasy(event.y)
        for item in self.find_overlapping(vx-3,vy-3,vx+3,vy+3):
            tags=self.gettags(item)
            if (nid:=next((t[5:] for t in tags if t.startswith("node_")),None)):
                if "port_out" in tags: 
                     self.history.save_state(); self.links=[l for l in self.links if not (l['source']==nid and l.get('source_port')==next((t for t in tags if t in self.nodes[nid].outputs),'out'))]; self.redraw_links(); return
                if "port_in" in tags: 
                     self.history.save_state(); self.links=[l for l in self.links if not l['target']==nid]; self.redraw_links(); return
        
        lx, ly = self.get_logical_pos(event.x, event.y)
        if (node := next((n for n in reversed(list(self.nodes.values())) if n.contains(lx, ly)), None)):
            m=tk.Menu(self,tearoff=0,bg=COLORS['bg_card'],fg=COLORS['fg_text'],font=FONTS['small'])
            m.add_command(label="📥 复制节点",command=lambda: self.duplicate_node(node.id))
            m.add_command(label="🔴 切换断点",command=lambda: self.toggle_breakpoint(node.id))
            m.add_separator()
            m.add_command(label="❌ 删除节点",command=lambda: (self.history.save_state(), self.delete_node(node.id)),foreground=COLORS['danger'])
            m.post(event.x_root,event.y_root)

    def toggle_breakpoint(self, node_id):
        if node_id in self.nodes:
            self.nodes[node_id].has_breakpoint = not self.nodes[node_id].has_breakpoint; self.nodes[node_id].draw()

    def get_data(self):
        nodes_d = {}
        for nid,n in self.nodes.items():
            clean_data={k:v for k,v in n.data.items() if k not in ['image','tk_image','roi_preview','anchors']}
            if 'image' in n.data: clean_data['b64']=ImageUtils.img_to_b64(n.data['image'])
            if 'roi_preview' in n.data and 'b64_preview' not in clean_data: clean_data['b64_preview'] = ImageUtils.img_to_b64(ImageUtils.b64_to_img(n.data.get('b64_preview')))
            if 'anchors' in n.data:
                clean_data['anchors'] = []
                for anc in n.data['anchors']:
                    new_anc = {k: v for k, v in anc.items() if k not in ['image']}
                    if 'image' in anc: new_anc['b64'] = ImageUtils.img_to_b64(anc['image'])
                    clean_data['anchors'].append(new_anc)
            if 'images' in n.data:
                clean_data['images'] = []
                for img_item in n.data['images']:
                    new_item = {k: v for k, v in img_item.items() if k not in ['image', 'tk_image']}
                    if 'image' in img_item: new_item['b64'] = ImageUtils.img_to_b64(img_item['image'])
                    clean_data['images'].append(new_item)
            nodes_d[nid]={'id':nid,'type':n.type,'x':int(n.x),'y':int(n.y),'data':clean_data, 'breakpoint': n.has_breakpoint}
        breakpoints = [nid for nid, n in self.nodes.items() if n.has_breakpoint]
        return {'nodes':nodes_d, 'links':self.links, 'breakpoints': breakpoints, 'metadata':{'dev_scale_x':SCALE_X,'dev_scale_y':SCALE_Y}}

    # [安全增强] 清理无效连线
    def _sanitize_project(self):
        valid_node_ids = set(self.nodes.keys())
        clean_links = []
        removed_count = 0
        for link in self.links:
            if link['source'] in valid_node_ids and link['target'] in valid_node_ids:
                clean_links.append(link)
            else:
                removed_count += 1
        self.links = clean_links
        if removed_count > 0:
            self.app.log(f"🧹 自动清理了 {removed_count} 条无效连线", "warning")

    def load_data(self,data):
        self.delete("all");self.nodes.clear();self.links.clear()
        try:
            self.app.core.load_project(data)
            breakpoints = set(data.get('breakpoints', []))
            for nid,n_data in data.get('nodes',{}).items():
                d=n_data.get('data',{})
                if 'image' in d: d['tk_image'] = ImageUtils.make_thumb(d['image'])
                if 'b64_preview' in d and (img:=ImageUtils.b64_to_img(d['b64_preview'])): d['roi_preview'] = ImageUtils.make_thumb(img)
                n = self.add_node(n_data['type'],n_data['x'],n_data['y'],data=d,node_id=nid, save_history=False)
                if n_data.get('breakpoint', False) or nid in breakpoints: n.has_breakpoint = True; n.draw()
            self.links=data.get('links',[])
            self._sanitize_project() # 清洗
            self.full_redraw()
        except Exception as e:
            self.app.log(f"❌ 项目加载失败:文件可能已损坏 ({e})", "error")
            self.app.clear()

# --- 12. 属性面板 [安全版] ---
class PropertyPanel(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=COLORS['bg_panel']); self.app, self.current_node = app, None
        self.static_monitor_active, self.is_monitoring_audio = False, False
        tk.Label(self, text="属性设置", bg=COLORS['bg_sidebar'], fg=COLORS['fg_text'], font=FONTS['h2'], pady=10).pack(fill='x')
        scroll_frame = tk.Frame(self, bg=COLORS['bg_panel'])
        scroll_frame.pack(fill='both', expand=True)
        self.scrollbar = ttk.Scrollbar(scroll_frame, orient="vertical")
        self.scrollbar.pack(side='right', fill='y')
        self.canvas = tk.Canvas(scroll_frame, bg=COLORS['bg_panel'], yscrollcommand=self.scrollbar.set, highlightthickness=0)
        self.canvas.pack(side='left', fill='both', expand=True)
        self.scrollbar.config(command=self.canvas.yview)
        self.content = tk.Frame(self.canvas, bg=COLORS['bg_panel'], padx=10, pady=10)
        self.content_id = self.canvas.create_window((0, 0), window=self.content, anchor='nw')
        self.content.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig(self.content_id, width=e.width))
        self.canvas.bind("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        self.show_empty()
    
    def clear(self): 
        [w.destroy() for w in self.content.winfo_children()]
        self.current_node = None; self.static_monitor_active = False

    def show_empty(self):
        self.clear()
        tk.Label(self.content, text="未选择节点", bg=COLORS['bg_panel'], fg=COLORS['fg_sub'], font=('Segoe UI', 9)).pack(pady=40)

    def load_node(self, node):
        self.clear(); self.current_node = node; self._input(self.content, "节点名称", '_user_title', node.data.get('_user_title', node.title_text))
        ntype, data = node.type, node.data
        
        if ntype == 'wait': self._input(self.content, "等待秒数", 'seconds', data.get('seconds', 1.0), validation_func=lambda v: safe_float(v, 1.0))
        elif ntype == 'loop': 
            self._chk(self.content, "无限循环", 'infinite', data.get('infinite', True))
            if not data.get('infinite', True):
                self._input(self.content, "循环次数", 'count', data.get('count', 5), validation_func=safe_int)
        elif ntype == 'sequence':
            sec = self._create_section("逻辑链设置")
            self._input(sec, "分支尝试数量", 'num_steps', data.get('num_steps', 3), validation_func=lambda v: safe_int(v, 3))
            
        elif ntype == 'image':
            base_sec = self._create_section("基础操作")
            if 'tk_image' in data and data['tk_image']: self._draw_image_preview(base_sec, data)
            self._btn(base_sec, "📸 截取目标", self.app.do_snip)
            self._btn(base_sec, "➕ 添加锚点", self.app.do_add_anchor)
            
            anchors = data.get('anchors', [])
            if anchors:
                anchor_sec = self._create_section(f"锚点列表 ({len(anchors)})")
                for i, anc in enumerate(anchors):
                    row = tk.Frame(anchor_sec, bg=COLORS['bg_card'], pady=2); row.pack(fill='x', pady=1)
                    tk.Label(row, text=f"锚点 {i+1}", bg=COLORS['bg_card'], fg=COLORS['success'], width=8, anchor='w').pack(side='left', padx=5)
                    self._btn_icon(row, "🗑️", lambda idx=i: self._delete_anchor(idx), color=COLORS['danger'])

            action_sec = self._create_section("执行动作")
            self._combo(action_sec, "动作",'click_type',list(ACTION_MAP.values()),ACTION_MAP.get(data.get('click_type','click')),lambda e:self._save('click_type',ACTION_MAP_REVERSE.get(e.widget.get())))
            
            off_frame = tk.Frame(action_sec, bg=action_sec.cget('bg')); off_frame.pack(fill='x', pady=5)
            self._compact_input(off_frame, "偏移 X", 'offset_x', data.get('offset_x', 0), safe_int)
            self._compact_input(off_frame, "Y", 'offset_y', data.get('offset_y', 0), safe_int)
            self._btn_icon(off_frame, "🎯", self.open_visual_offset_picker, bg=COLORS['control'], width=3)

            search_sec = self._create_section("搜索参数")
            search_modes = {'fullscreen': '🖥️ 全屏幕', 'region': '🔲 固定区域'}
            current_mode = data.get('search_mode', 'fullscreen')
            self._combo(search_sec,"搜索范围",'search_mode',list(search_modes.values()),search_modes.get(current_mode),lambda e:self._save('search_mode',{v: k for k, v in search_modes.items()}.get(e.widget.get())))
            
            if current_mode == 'region':
                 rx, ry, rw, rh = data.get('target_rect_x', 0), data.get('target_rect_y', 0), data.get('target_rect_w', 0), data.get('target_rect_h', 0)
                 tk.Label(search_sec, text=f"区域: {rx},{ry} {rw}x{rh}", bg=COLORS['bg_card'], fg=COLORS['fg_sub'], font=('Consolas', 8)).pack(fill='x', pady=(0, 5))
                 self._btn(search_sec, "📐 重新框选区域", self.app.do_set_target)

            self._combo(search_sec, "匹配策略",'match_strategy',list(MATCH_STRATEGY_MAP.values()),MATCH_STRATEGY_MAP.get(data.get('match_strategy','hybrid')),lambda e:self._save('match_strategy',MATCH_STRATEGY_REVERSE.get(e.widget.get())))
            self._input(search_sec, "相似度",'confidence',data.get('confidence',0.9), validation_func=safe_float)
            self._input(search_sec, "超时(秒)",'timeout',data.get('timeout',10.0), validation_func=safe_float)
            self._chk(search_sec, "启用自动滚动", 'auto_scroll', data.get('auto_scroll', False))
            if data.get('auto_scroll'): self._input(search_sec, "滚动量", 'scroll_amount', data.get('scroll_amount', -500), validation_func=safe_int)
            self._btn(search_sec, "🧪 测试匹配", self.start_test_match)
            self.test_result_label = tk.Label(search_sec, bg=search_sec.cget('bg'), fg=COLORS['fg_sub'], font=FONTS['small']); self.test_result_label.pack(fill='x')
            
        elif ntype == 'if_img':
            sec = self._create_section("检测条件")
            if (images := data.get('images', [])):
                for img_data in images:
                    f = tk.Frame(sec, bg=COLORS['bg_card']); f.pack(fill='x', pady=2)
                    if img_data.get('tk_image'): tk.Canvas(f, width=40, height=22, bg='black', highlightthickness=0).pack(side='left', padx=5) 
                    self._btn_icon(f, "❌", lambda i=img_data.get('id'): self._delete_image_condition(i), color=COLORS['danger'])
            self._btn(sec, "➕ 添加截图条件", self.app.do_snip); self._btn(sec, "🧪 测试所有条件", self.start_test_match)
            self.test_result_label = tk.Label(sec, bg=sec.cget('bg'), fg=COLORS['fg_sub'], font=FONTS['small']); self.test_result_label.pack(fill='x')

        elif ntype == 'if_static':
            base_sec = self._create_section("监控区域")
            if 'roi_preview' in data and data['roi_preview']: 
                c = tk.Canvas(base_sec, width=240, height=135, bg='black', highlightthickness=0); c.pack(pady=5)
                c.create_image(120, 67, image=data['roi_preview'], anchor='center')
            self._btn(base_sec, "📸 截取监控区域", self.app.do_snip)
            
            param_sec = self._create_section("检测参数")
            self._input(param_sec, "静止持续", 'duration', data.get('duration', 5.0), safe_float)
            self._input(param_sec, "最大超时", 'timeout', data.get('timeout', 20.0), safe_float)
            self._input(param_sec, "灵敏度", 'threshold', data.get('threshold', 0.98), safe_float)
            
            monitor_frame = self._create_section("实时测试")
            self.lbl_monitor_status = tk.Label(monitor_frame, text="等待启动...", bg=monitor_frame.cget('bg'), fg=COLORS['fg_sub'], font=('Consolas', 9))
            self.lbl_monitor_status.pack(fill='x', pady=5)
            self.btn_monitor = self._btn(monitor_frame, "🔴 启动监控", self._toggle_static_monitor)

        elif ntype == 'if_sound':
            sec = self._create_section("声音检测")
            self._combo(sec, "模式", 'detect_mode', ['has_sound', 'is_silent'], data.get('detect_mode', 'has_sound'), lambda e:self._save('detect_mode',e.widget.get()))
            self._input(sec, "阈值(0-1)", 'threshold', data.get('threshold', 0.02), safe_float)
            self._input(sec, "超时(秒)", 'timeout', data.get('timeout', 10.0), safe_float)
            btn_text = "⏹ 停止" if self.is_monitoring_audio else "🔊 实时监测"
            self.monitor_audio_btn = self._btn(sec, btn_text, self._toggle_audio_monitor)

        elif ntype == 'mouse':
            sec = self._create_section("鼠标操作")
            current_action = data.get('mouse_action', 'click')
            self._combo(sec, "动作类型", 'mouse_action', list(MOUSE_ACTIONS.values()), MOUSE_ACTIONS.get(current_action), lambda e:self._save('mouse_action', {v: k for k, v in MOUSE_ACTIONS.items()}.get(e.widget.get())))
            
            if current_action == 'click':
                self._combo(sec, "按键", 'mouse_button', list(MOUSE_BUTTONS.values()), MOUSE_BUTTONS.get(data.get('mouse_button', 'left')), lambda e:self._save('mouse_button', {v: k for k, v in MOUSE_BUTTONS.items()}.get(e.widget.get())))
                self._combo(sec, "点击次数", 'click_count', list(MOUSE_CLICKS.values()), MOUSE_CLICKS.get(str(data.get('click_count', '1'))), lambda e:self._save('click_count', {v: k for k, v in MOUSE_CLICKS.items()}.get(e.widget.get())))
            
            if current_action in ['move', 'drag']:
                self._input(sec, "移动耗时(秒)", 'duration', data.get('duration', 0.5), safe_float)
            
            if current_action == 'drag':
                 self._combo(sec, "按住按键", 'mouse_button', list(MOUSE_BUTTONS.values()), MOUSE_BUTTONS.get(data.get('mouse_button', 'left')), lambda e:self._save('mouse_button', {v: k for k, v in MOUSE_BUTTONS.items()}.get(e.widget.get())))

            if current_action in ['click', 'move', 'drag']:
                coord_frame = tk.Frame(sec, bg=sec.cget('bg')); coord_frame.pack(fill='x', pady=5)
                self._compact_input(coord_frame, "X", 'x', data.get('x', 0), safe_int)
                self._compact_input(coord_frame, "Y", 'y', data.get('y', 0), safe_int)
                self._btn_icon(coord_frame, "📍", self.app.pick_coordinate, width=3)
            
            if current_action == 'scroll':
                self._input(sec, "滚动量", 'amount', data.get('amount', -500), safe_int)

        elif ntype == 'keyboard':
             sec = self._create_section("键盘操作")
             self._combo(sec, "类型",'kb_mode',['text','key'],data.get('kb_mode','text'),lambda e:[self._save('kb_mode',e.widget.get()), self.load_node(self.current_node)])
             if data.get('kb_mode', 'text')=='text':
                 self._input(sec, "文本内容",'text',data.get('text',''))
                 self._chk(sec, "模拟打字 (慢速)", 'slow_type', data.get('slow_type', False))
                 self._chk(sec, "输入后按回车", 'press_enter', data.get('press_enter', False))
             else: 
                 self._input(sec, "组合键",'key_name',data.get('key_name','enter'))

        elif ntype == 'web': self._input(self.content, "网址", 'url', data.get('url', ''))

        elif ntype == 'set_var':
            sec = self._create_section("变量设置")
            tk.Label(sec, text="每行 'name=value':", bg=sec.cget('bg'), fg=COLORS['fg_text'], font=FONTS['small']).pack(anchor='w')
            txt = tk.Text(sec, height=5, bg=COLORS['input_bg'], fg=COLORS['fg_text'], bd=0, insertbackground='white'); txt.pack(fill='x', pady=(2,5))
            existing = ""
            for item in data.get('batch_vars', []): existing += f"{item.get('name')}={item.get('value')}\n"
            if not existing and data.get('var_name'): existing = f"{data.get('var_name')}={data.get('var_value')}"
            txt.insert('1.0', existing)
            def save_vars():
                res = [{'name':line.split('=')[0].strip(), 'value':line.split('=')[1].strip()} for line in txt.get('1.0', 'end').strip().split('\n') if '=' in line]
                self._save('batch_vars', res)
            self._btn(sec, "💾 保存", save_vars)

        elif ntype == 'var_switch':
            if data.get('var_name'):
                 sec = self._create_section("单变量分流")
                 self._input(sec, "变量名", 'var_name', data.get('var_name'))
                 self._combo(sec, "操作", 'operator', list(VAR_OP_MAP.values()), VAR_OP_MAP.get(data.get('operator','=')), lambda e:self._save('operator',VAR_OP_REVERSE.get(e.widget.get())))
                 self._input(sec, "对比值", 'var_value', data.get('var_value',''))
            else:
                 sec = self._create_section("多变量分流")
                 self._input(sec, "变量列表(逗号隔开)", 'var_list', data.get('var_list',''))
                 cases_frame = tk.Frame(sec, bg=sec.cget('bg')); cases_frame.pack(fill='x', pady=5)
                 def update_cases(): node.outputs = [c['id'] for c in data.get('cases',[])] + ['else']; node.h = PORT_START_Y + max(1, len(node.outputs)) * PORT_STEP_Y; node.draw(); self.app.editor.redraw_links(); self.load_node(node)
                 def add_case(): data.setdefault('cases', []).append({'value':'new','id':f'case_{len(data["cases"])}_{int(time.time())}'}); update_cases()
                 for i, c in enumerate(data.get('cases', [])):
                     r = tk.Frame(cases_frame, bg=COLORS['bg_card']); r.pack(fill='x', pady=1)
                     e = tk.Entry(r, bg=COLORS['input_bg'], width=10, fg=COLORS['fg_text'], bd=0, insertbackground=COLORS['fg_title']); e.insert(0, c.get('value')); e.pack(side='left', fill='x', expand=True, padx=5, pady=3)
                     e.bind("<KeyRelease>", lambda ev, idx=i: data['cases'][idx].update({'value':ev.widget.get()}) or node.draw())
                     self._btn_icon(r, "❌", lambda idx=i: [data['cases'].pop(idx), update_cases()], color=COLORS['danger'])
                 self._btn(sec, "➕ 添加条件", add_case)

    def _create_section(self, text):
        f = tk.Frame(self.content, bg=COLORS['bg_panel'], pady=5)
        f.pack(fill='x')
        tk.Label(f, text=text, bg=COLORS['bg_panel'], fg=COLORS['accent'], font=('Segoe UI', 9, 'bold')).pack(anchor='w')
        tk.Frame(f, height=1, bg=COLORS['bg_header']).pack(fill='x', pady=(2, 5))
        return f

    def _draw_image_preview(self, parent, data):
        c = tk.Canvas(parent, width=240, height=135, bg='black', highlightthickness=0); c.pack(pady=5)
        c.create_image(120, 67, image=data['tk_image'], anchor='center')
        w, h = data['image'].size
        ratio = min(240/w, 135/h) if w > 0 and h > 0 else 0
        dw, dh = int(w * ratio), int(h * ratio)
        off_x, off_y = (240 - dw) // 2, (135 - dh) // 2
        def on_click(e):
            rx = max(0.0, min(1.0, (e.x - off_x) / dw if dw > 0 else 0))
            ry = max(0.0, min(1.0, (e.y - off_y) / dh if dh > 0 else 0))
            self._save('relative_click_pos', (rx, ry)); self.load_node(self.current_node)
        c.bind("<Button-1>", on_click)
        rx, ry = data.get('relative_click_pos', (0.5, 0.5))
        cx, cy = off_x + (rx * dw), off_y + (ry * dh)
        c.create_oval(cx-3, cy-3, cx+3, cy+3, fill=COLORS['marker'], outline='white', width=1)

    def _delete_anchor(self, idx):
        if not self.current_node: return
        anchors = self.current_node.data.get('anchors', [])
        if 0 <= idx < len(anchors): del anchors[idx]; self._save('anchors', anchors); self.load_node(self.current_node)

    def _delete_image_condition(self, iid):
        images = self.current_node.data.get('images', [])
        self.current_node.data['images'] = [i for i in images if i.get('id') != iid]
        self.load_node(self.current_node)

    def _toggle_static_monitor(self):
        if self.static_monitor_active:
            self.static_monitor_active = False; self.btn_monitor.config(text="🔴 启动监控", bg=COLORS['btn_bg']); self.lbl_monitor_status.config(text="监控已停止")
        else:
            if not self.current_node.data.get('roi'): messagebox.showwarning("提示", "请先截取监控区域！"); return
            self.static_monitor_active = True; self.btn_monitor.config(text="⏹ 停止监控", bg=COLORS['danger']); threading.Thread(target=self._static_monitor_thread, daemon=True).start()

    def _static_monitor_thread(self):
        roi, thr, dur = self.current_node.data.get('roi'), safe_float(self.current_node.data.get('threshold', 0.98)), safe_float(self.current_node.data.get('duration', 5.0))
        last_frame = VisionEngine.capture_screen(bbox=roi); static_start = time.time()
        while self.static_monitor_active and self.current_node and self.current_node.type == 'if_static':
            curr = VisionEngine.capture_screen(bbox=roi); is_static = VisionEngine.compare_images(last_frame, curr, thr)
            elapsed = time.time() - static_start if is_static else 0
            if self.lbl_monitor_status.winfo_exists(): self.lbl_monitor_status.config(text=f"{'🟢 静止' if is_static else '🌊 运动'} | {elapsed:.1f}s / {dur}s", fg=COLORS['success'] if elapsed>=dur else COLORS['warning'])
            if not is_static: static_start = time.time(); last_frame = curr
            time.sleep(0.1)
        self.static_monitor_active = False

    def _toggle_audio_monitor(self):
        self.is_monitoring_audio = not self.is_monitoring_audio
        self.monitor_audio_btn.config(text="⏹ 停止" if self.is_monitoring_audio else "🔊 实时监测")
        if self.is_monitoring_audio: threading.Thread(target=self._audio_monitor_thread, daemon=True).start()

    def _audio_monitor_thread(self):
        while self.is_monitoring_audio and self.app.property_panel.winfo_exists():
            vol = AudioEngine.get_max_audio_peak()
            if vol > 0.001: self.app.log(f"📊 音量峰值: {vol:.4f}", "info")
            time.sleep(0.5)

    def open_visual_offset_picker(self):
        self.app.iconify(); time.sleep(0.3); full_screen = ImageGrab.grab()
        try:
            res = VisionEngine.locate(self.current_node.data.get('image'), confidence=0.8, timeout=1.0)
            if not res: self.app.deiconify(); messagebox.showerror("错误", "未在屏幕找到基准图"); return
            top = tk.Toplevel(self.app); top.attributes("-fullscreen", True, "-topmost", True); cv = tk.Canvas(top, width=full_screen.width, height=full_screen.height); cv.pack()
            tk_img = ImageTk.PhotoImage(full_screen); cv.create_image(0,0,image=tk_img,anchor='nw')
            cv.create_rectangle(res.left, res.top, res.left+res.width, res.top+res.height, outline='green', width=2)
            cx, cy = res.left+res.width/2, res.top+res.height/2
            def confirm(e): self._save('offset_x', int(e.x-cx)); self._save('offset_y', int(e.y-cy)); top.destroy(); self.app.deiconify(); self.load_node(self.current_node)
            cv.bind("<Button-1>", confirm); cv.bind("<Button-3>", lambda e: [top.destroy(), self.app.deiconify()])
            top.img_ref = tk_img; self.wait_window(top)
        except Exception as e: self.app.deiconify(); traceback.print_exc()

    def start_test_match(self):
        threading.Thread(target=self._test_match_worker, daemon=True).start()
    def _test_match_worker(self):
        self.app.iconify(); time.sleep(0.5); res_txt = "未找到"
        try:
            if self.current_node.type == 'if_img':
                imgs = self.current_node.data.get('images', [])
                if not imgs: res_txt = "无条件"
                else: 
                     passed = True; screen = VisionEngine.capture_screen()
                     for img in imgs:
                         if not VisionEngine._advanced_match(img.get('image'), screen, 0.8, None, True, True, 1.0, 'hybrid')[0]: passed = False; break
                     res_txt = "✅ 全部满足" if passed else "❌ 条件不满足"
            else:
                 res = VisionEngine.locate(self.current_node.data.get('image'), confidence=0.8)
                 res_txt = "✅ 找到" if res else "❌ 未找到"
        except: pass
        self.app.deiconify(); self.test_result_label.config(text=res_txt)

    # [安全增强] 输入控件带验证
    def _input(self,parent,label,key,val, validation_func=None): 
        f=tk.Frame(parent,bg=parent.cget('bg'));f.pack(fill='x',pady=2)
        tk.Label(f,text=label,bg=parent.cget('bg'),fg=COLORS['fg_text'],font=FONTS['small']).pack(side='left',padx=(0,5))
        e=tk.Entry(f,bg=COLORS['input_bg'],fg=COLORS['fg_text'],bd=0,insertbackground=COLORS['fg_title'],font=FONTS['code'])
        e.insert(0,str(val));e.pack(fill='x',pady=2,ipady=3,expand=True)
        
        def on_release(ev):
            raw_val = e.get()
            final_val = validation_func(raw_val) if validation_func else raw_val
            self._save(key, final_val)
            
        e.bind("<KeyRelease>", on_release)

    def _compact_input(self, parent, label, key, val, validation_func=None):
        tk.Label(parent, text=label, bg=parent.cget('bg'), fg=COLORS['fg_text'], font=FONTS['small']).pack(side='left', padx=(5,2))
        e = tk.Entry(parent, bg=COLORS['input_bg'], fg=COLORS['fg_text'], bd=0, insertbackground=COLORS['fg_title'], width=5)
        e.insert(0, str(val)); e.pack(side='left', padx=2)
        
        def on_release(ev):
            raw_val = e.get()
            final_val = validation_func(raw_val) if validation_func else raw_val
            self._save(key, final_val)
            
        e.bind("<KeyRelease>", on_release)

    def _combo(self,parent,label,key,values,val,cmd): 
        f=tk.Frame(parent,bg=parent.cget('bg'));f.pack(fill='x',pady=2)
        tk.Label(f,text=label,bg=parent.cget('bg'),fg=COLORS['fg_text'],font=FONTS['small']).pack(side='left',padx=(0,5))
        cb=ttk.Combobox(f,values=values,state="readonly", font=FONTS['code']);cb.set(val)
        cb.pack(fill='x',pady=2,expand=True);cb.bind("<<ComboboxSelected>>",cmd)

    def _btn(self,parent,txt,cmd,bg=None): 
        b=tk.Button(parent,text=txt,command=cmd,bg=bg or COLORS['btn_bg'],fg=COLORS['fg_text'],bd=0,activebackground=COLORS['btn_hover'],activeforeground='white',relief='flat',pady=2, font=FONTS['small'])
        b.pack(fill='x',pady=3, ipady=1)
        return b

    def _btn_icon(self, parent, txt, cmd, bg=None, color=None, width=None):
        b=tk.Button(parent,text=txt,command=cmd,bg=bg or COLORS['bg_card'],fg=color or COLORS['fg_text'],bd=0,activebackground=COLORS['btn_hover'],relief='flat', width=width)
        b.pack(side='right', padx=2)

    def _chk(self,parent,txt,key,val): 
        var=tk.BooleanVar(value=val)
        tk.Checkbutton(parent,text=txt,variable=var,bg=parent.cget('bg'),fg=COLORS['fg_text'],selectcolor=COLORS['bg_app'],activebackground=parent.cget('bg'),borderwidth=0,highlightthickness=0,command=lambda:[self._save(key,var.get()), self.load_node(self.current_node)]).pack(anchor='w',pady=2)

    def _save(self,key,val): (self.current_node and self.current_node.update_data(key,val))

# --- 13. 主程序入口 ---
class App(tk.Tk):
    def __init__(self):
        super().__init__(); self.title("Qflow 1.3"); self.geometry("1400x900"); self.configure(bg=COLORS['bg_app'])
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # [安全增强] 全局异常钩子
        self.report_callback_exception = self._on_ui_exception
        
        self.core = AutomationCore(self.log, self); self.log_q = queue.Queue()
        self.drag_node_type, self.drag_ghost = None, None
        self.float_window = None 
        
        self._setup_styles(); self._setup_ui()
        self.bind("<Unmap>", self._on_window_unmap)
        self.bind("<Map>", self._on_window_map)
        self.after(100, self._poll_log)

    def _on_ui_exception(self, exc, val, tb):
        err_msg = "".join(traceback.format_exception(exc, val, tb))
        print(err_msg)
        self.log(f"🔥 系统严重错误: {val}", "error")
        messagebox.showerror("应用错误", f"发生未处理的异常:\n{val}\n请查看日志或控制台")

    def _setup_styles(self): 
        s = ttk.Style()
        s.theme_use('clam')
        
        s.configure('TCombobox', 
            fieldbackground=COLORS['bg_card'], 
            background=COLORS['bg_card'], 
            foreground=COLORS['fg_text'], 
            arrowcolor=COLORS['fg_text'],
            bordercolor=COLORS['bg_header'],
            lightcolor=COLORS['bg_card'], 
            darkcolor=COLORS['bg_card'],
            insertcolor=COLORS['fg_text'],
            arrowsize=18
        )
        s.map('TCombobox',
            foreground=[('readonly', COLORS['fg_text'])],
            fieldbackground=[('readonly', COLORS['bg_card'])],
            background=[('readonly', COLORS['bg_card'])],
            selectbackground=[('readonly', COLORS['bg_app'])],
            selectforeground=[('readonly', COLORS['accent'])],
        )
        self.option_add('*TCombobox*Listbox.background', COLORS['bg_card'])
        self.option_add('*TCombobox*Listbox.foreground', COLORS['fg_text'])
        self.option_add('*TCombobox*Listbox.selectBackground', COLORS['accent'])
        self.option_add('*TCombobox*Listbox.selectForeground', '#ffffff')
        self.option_add('*TCombobox*Listbox.font', FONTS['code'])
        self.option_add('*TCombobox*Listbox.bd', 0)
        
        s.configure("Vertical.TScrollbar", gripcount=0, background=COLORS['bg_header'], darkcolor=COLORS['bg_app'], lightcolor=COLORS['bg_app'], troughcolor=COLORS['bg_app'], bordercolor=COLORS['bg_app'], arrowcolor=COLORS['fg_title'])

    def _setup_ui(self):
        title_bar = tk.Frame(self, bg=COLORS['bg_app'], height=50); title_bar.pack(fill='x', pady=5, padx=20)
        
        tk.Label(title_bar, text="Qflow 1.3", font=('Impact', 24), bg=COLORS['bg_app'], fg=COLORS['accent']).pack(side='left', padx=(0, 20))
        
        ops_frame = tk.Frame(title_bar, bg=COLORS['bg_app']); ops_frame.pack(side='left')
        self._flat_btn(ops_frame, "📂 打开", self.load)
        self._flat_btn(ops_frame, "💾 保存", self.save)
        self._flat_btn(ops_frame, "🗑️ 清空", self.clear)
        
        undo_frame = tk.Frame(ops_frame, bg=COLORS['bg_app']); undo_frame.pack(side='left', padx=10)
        self.icon_undo = IconFactory.create('undo', COLORS['fg_text'], bg=COLORS['bg_app'])
        self.icon_redo = IconFactory.create('redo', COLORS['fg_text'], bg=COLORS['bg_app'])
        tk.Button(undo_frame, image=self.icon_undo, command=lambda: self.editor.history.undo(), bg=COLORS['bg_app'], bd=0, activebackground=COLORS['bg_header']).pack(side='left', padx=2)
        tk.Button(undo_frame, image=self.icon_redo, command=lambda: self.editor.history.redo(), bg=COLORS['bg_app'], bd=0, activebackground=COLORS['bg_header']).pack(side='left', padx=2)
        
        self.debug_frame = tk.Frame(title_bar, bg=COLORS['bg_app'])
        self.btn_pause = tk.Button(self.debug_frame, text="⏸ 暂停", command=self.toggle_pause, bg=COLORS['warning'], fg='#1f1f1f', bd=0, relief='flat', padx=10, font=FONTS['small'])
        self.btn_pause.pack(side='left', padx=2, ipady=4)
        self.debug_frame.pack(side='left', padx=20)
        
        self.run_btn_frame = tk.Frame(title_bar, bg=COLORS['bg_app']); self.run_btn_frame.pack(side='right')
        self.btn_run = tk.Button(self.run_btn_frame, text="▶ 启动", command=lambda: self.toggle_run(None), bg=COLORS['success'], fg='#1f1f1f', font=('Segoe UI', 11, 'bold'), padx=15, bd=0, relief='flat'); self.btn_run.pack(side='left', fill='y')
        self.btn_run_menu = tk.Menubutton(self.run_btn_frame, text="▼", bg=COLORS['success'], fg='#1f1f1f', font=('Segoe UI', 10, 'bold'), padx=5, bd=0, relief='flat', direction='below', activebackground=COLORS['success'])
        self.last_executed_node = None 
        self.run_menu = tk.Menu(self.btn_run_menu, tearoff=0, bg=COLORS['bg_card'], fg=COLORS['fg_text'], font=FONTS['small']); 
        self.run_menu.add_command(label="▶ 从头开始运行", command=lambda: self.toggle_run(None)); 
        self.run_menu.add_command(label="▶ 从选中节点开始运行", command=lambda: self.editor.selected_node_id and self.toggle_run(self.editor.selected_node_id)); 
        self.run_menu.add_command(label="🔄 重试最后执行节点", command=lambda: self.last_executed_node and self.toggle_run(self.last_executed_node)); 
        self.btn_run_menu.config(menu=self.run_menu); self.btn_run_menu.pack(side='left', fill='y', padx=(1,0))
        
        paned=tk.PanedWindow(self,orient='horizontal',bg=COLORS['bg_app'],sashwidth=6,sashrelief='flat',bd=0);paned.pack(fill='both',expand=True,padx=10,pady=(0,5))
        toolbox=tk.Frame(paned,bg=COLORS['bg_sidebar'],width=200);toolbox.pack_propagate(False)
        for t, l in [("事件 / 逻辑",['start','end','set_var','var_switch']), ("控制流程",['sequence','if_img','if_static','if_sound','loop','wait']), ("动作操作",['mouse','keyboard','image','web'])]: self._add_group(toolbox,t,l)
        paned.add(toolbox,minsize=180)
        
        self.editor=FlowEditor(paned,self); paned.add(self.editor,minsize=400,stretch="always")
        self.property_panel=PropertyPanel(paned,self); paned.add(self.property_panel,minsize=100,width=140)
        self.log_panel=LogPanel(self); self.log_panel.pack(side='bottom',fill='x',pady=(0,10),padx=10)
        self.editor.add_node('start',100,100, save_history=False); self._setup_hotkeys()
        self.show_debug_toolbar(False)

    def _flat_btn(self,p,txt,cmd): tk.Button(p,text=txt,command=cmd,bg=COLORS['bg_header'],fg=COLORS['fg_text'],bd=0,padx=15,pady=5,activebackground=COLORS['bg_panel'],relief='flat',font=FONTS['small']).pack(side='left',padx=5)
    
    def _add_group(self, p, title, items):
        tk.Label(p, text=title, bg=COLORS['bg_sidebar'], fg=COLORS['fg_sub'], font=('Segoe UI', 9, 'bold'), pady=8).pack(anchor='w', padx=10)
        for t in items:
            lbl = tk.Label(p, text=NODE_CONFIG[t]['title'], bg=COLORS['bg_card'], fg=COLORS['fg_text'], anchor='w', padx=10, pady=6, cursor="hand2")
            lbl.pack(fill='x', pady=1, padx=10); lbl.bind("<ButtonPress-1>", lambda e, ntype=t: self.on_sidebar_drag_start(e, ntype))
            lbl.bind("<B1-Motion>", self.on_sidebar_drag_motion); lbl.bind("<ButtonRelease-1>", self.on_sidebar_drag_release)
    
    def _on_window_unmap(self, event):
        if event.widget == self and self.state() == 'iconic' and not self.float_window: self.float_window = FloatWindow(self)
    def _on_window_map(self, event):
        if event.widget == self and self.float_window: self.float_window.restore_app()

    def update_float_status_safe(self, text):
        if self.float_window and self.float_window.winfo_exists(): self.after(0, lambda: self.float_window.set_status_text(text))

    def on_sidebar_drag_start(self,event,ntype): self.drag_node_type=ntype;self.drag_ghost=tk.Toplevel(self);self.drag_ghost.overrideredirect(True);self.drag_ghost.attributes("-topmost",True,"-alpha",0.7);tk.Label(self.drag_ghost,text=NODE_CONFIG[ntype]['title'],bg=COLORS['accent'],fg='#1f1f1f',padx=10,pady=5).pack();self.drag_ghost.geometry(f"+{event.x_root+10}+{event.y_root+10}")
    def on_sidebar_drag_motion(self,event): (self.drag_ghost and self.drag_ghost.geometry(f"+{event.x_root+10}+{event.y_root+10}"))
    def on_sidebar_drag_release(self,event):
        if self.drag_ghost: self.drag_ghost.destroy();self.drag_ghost=None
        if self.editor.winfo_rootx()<=event.x_root<=self.editor.winfo_rootx()+self.editor.winfo_width() and self.editor.winfo_rooty()<=event.y_root<=self.editor.winfo_rooty()+self.editor.winfo_height():
            log_x,log_y=self.editor.canvasx(event.x_root-self.editor.winfo_rootx())/self.editor.zoom,self.editor.canvasy(event.y_root-self.editor.winfo_rooty())/self.editor.zoom
            self.editor.add_node(self.drag_node_type,round(log_x/GRID_SIZE)*GRID_SIZE,round(log_y/GRID_SIZE)*GRID_SIZE)
        self.drag_node_type=None
    
    def do_snip(self): self.iconify();self.update();self.after(400, lambda: self._start_snip_overlay(mode='normal'))
    def do_add_anchor(self): self.iconify();self.update();self.after(400, lambda: self._start_snip_overlay(mode='add_anchor'))
    def do_set_target(self): self.iconify();self.update();self.after(400, lambda: self._start_snip_overlay(mode='set_target'))

    def _start_snip_overlay(self, mode='normal'):
        top=tk.Toplevel(self);top.attributes("-fullscreen",True,"-alpha",0.3,"-topmost",True);top.configure(cursor="cross");c=tk.Canvas(top,bg="black",highlightthickness=0);c.pack(fill='both',expand=True)
        info_lbl = tk.Label(top, text="", font=('Segoe UI', 16, 'bold'), fg='white', bg='black')
        info_lbl.place(x=50, y=50)
        if mode == 'add_anchor': info_lbl.config(text="请框选一个【锚点/特征】区域 (ESC取消)", fg='#a6e3a1')
        elif mode == 'set_target': info_lbl.config(text="请框选【最终目标】区域 (ESC取消)", fg='#f38ba8')
        else: info_lbl.config(text="请框选区域 (ESC取消)")
        s, r = [0,0], [None]
        def dn(e): 
            s[0], s[1] = e.x_root, e.y_root
            color = 'green' if mode=='add_anchor' else 'red'
            r[0] = c.create_rectangle(e.x, e.y, e.x, e.y, outline=color, width=2)
        def mv(e): (r[0] and c.coords(r[0], s[0] - top.winfo_rootx(), s[1] - top.winfo_rooty(), e.x, e.y))
        def up(e):
            x1, y1, x2, y2 = min(s[0], e.x_root), min(s[1], e.y_root), max(s[0], e.x_root), max(s[1], e.y_root)
            if x2-x1 < 5 or y2-y1 < 5: 
                if r[0]: c.delete(r[0]); r[0]=None
                return
            rect = (x1, y1, x2, y2)
            top.destroy()
            if mode == 'add_anchor': self.after(200, lambda: self._internal_add_anchor(rect))
            elif mode == 'set_target': self.after(200, lambda: self._internal_set_target(rect))
            else: self.after(200, lambda: self._internal_capture(rect))
        c.bind("<ButtonPress-1>",dn);c.bind("<B1-Motion>",mv);c.bind("<ButtonRelease-1>",up);
        top.bind("<Escape>",lambda e:[top.destroy(),self.deiconify()])
        self.wait_window(top)

    def _internal_capture(self, rect):
        x1, y1, x2, y2 = rect
        try:
            img = ImageGrab.grab(bbox=(x1, y1, x2, y2)); self.deiconify(); self.lift()
            if (node := self.property_panel.current_node):
                node.update_data('_dummy_for_history', time.time())
                if node.type == 'if_img':
                    current_images = node.data.get('images', [])
                    current_images.append({'id': str(uuid.uuid4()),'image': img,'tk_image': ImageUtils.make_thumb(img, size=(120, 67)),'b64': ImageUtils.img_to_b64(img)})
                    node.update_data('images', current_images)
                elif node.type == 'if_static':
                    node.update_data('roi', (x1, y1, x2, y2)); node.update_data('roi_preview', ImageUtils.make_thumb(img)); node.update_data('b64_preview', ImageUtils.img_to_b64(img))
                else:
                    node.update_data('image', img); node.update_data('tk_image', ImageUtils.make_thumb(img)); node.update_data('b64', ImageUtils.img_to_b64(img))
                self.property_panel.load_node(node)
                self.log(f"🖼️ 截取成功 ({x2-x1}x{y2-y1})", "success")
        except Exception as e: self.deiconify(); self.log(f"截图失败: {e}", "error")

    def _internal_add_anchor(self, rect):
        try:
            img = ImageGrab.grab(bbox=rect)
            self.deiconify(); self.lift()
            if (node := self.property_panel.current_node) and node.type == 'image':
                node.update_data('_dummy_for_history', time.time())
                if 'anchors' not in node.data or not node.data['anchors']:
                     keys_to_clear = ['image', 'tk_image', 'b64', 'target_rect_x', 'target_rect_y', 'target_rect_w', 'target_rect_h']
                     for k in keys_to_clear:
                        if k in node.data: del node.data[k]
                anchors = node.data.get('anchors', [])
                anchors.append({'id': str(uuid.uuid4()),'image': img,'b64': ImageUtils.img_to_b64(img),'rect_x': rect[0], 'rect_y': rect[1]})
                node.update_data('anchors', anchors)
                self.property_panel.load_node(node)
                self.log(f"⚓ 锚点添加成功 (共 {len(anchors)} 个)", 'success')
        except Exception as e: self.deiconify(); self.log(f"锚点截图失败: {e}", "error")

    def _internal_set_target(self, rect):
        try:
            img = ImageGrab.grab(bbox=rect)
            self.deiconify(); self.lift()
            if (node := self.property_panel.current_node) and node.type == 'image':
                node.update_data('image', img); node.update_data('tk_image', ImageUtils.make_thumb(img)); node.update_data('b64', ImageUtils.img_to_b64(img))
                node.update_data('target_rect_x', rect[0]); node.update_data('target_rect_y', rect[1]); node.update_data('target_rect_w', rect[2] - rect[0]); node.update_data('target_rect_h', rect[3] - rect[1])
                self.property_panel.load_node(node)
                self.log("🎯 目标区域已设定", 'success')
        except Exception as e: self.deiconify(); self.log(f"目标截图失败: {e}", "error")

    def pick_coordinate(self): self.iconify();self.update();self.after(400, self._start_coordinate_overlay)
    
    def _start_coordinate_overlay(self):
        top=tk.Toplevel(self);top.attributes("-fullscreen",True,"-alpha",0.1,"-topmost",True);top.configure(cursor="none");c=tk.Canvas(top,bg="white",highlightthickness=0);c.pack(fill='both',expand=True)
        w,h=top.winfo_screenwidth(),top.winfo_screenheight();h_line,v_line=c.create_line(0,0,w,0,fill="red",width=1),c.create_line(0,0,0,h,fill="red",width=1);lbl_bg=c.create_rectangle(0,0,0,0,fill="#ffffdd",outline="black");lbl=c.create_text(0,0,text="",fill="black",anchor="nw",font=("Consolas",10))
        def on_move(e): 
            c.coords(h_line,0,e.y,w,e.y); c.coords(v_line,e.x,0,e.x,h)
            txt=f"X:{int(e.x_root)}, Y:{int(e.y_root)}"
            c.itemconfig(lbl,text=txt); bbox=c.bbox(lbl); c.coords(lbl,e.x+15,e.y+15); c.coords(lbl_bg,e.x+13,e.y+13,e.x+17+bbox[2]-bbox[0],e.y+17+bbox[3]-bbox[1])
        c.bind("<Motion>",on_move)
        c.bind("<Button-1>",lambda e:[top.destroy(),self.after(200,lambda:self._apply_picked_coordinate(int(e.x_root),int(e.y_root)))])
        c.bind("<Button-3>",lambda e:[top.destroy(),self.deiconify()]);self.wait_window(top)
        
    def _apply_picked_coordinate(self,x,y): 
        self.deiconify(); self.lift()
        if (node := self.property_panel.current_node):
            node.update_data('x',x); node.update_data('y',y)
            self.property_panel.load_node(node)
            self.log(f"📍 坐标: ({x},{y})", "info")

    def toggle_run(self, start_id=None): 
        if self.core.running: self.core.stop()
        else: self.core.load_project(self.editor.get_data()); self.core.start(start_id)

    def show_debug_toolbar(self, show):
        if show: self.debug_frame.pack(side='left', padx=20)
        else: self.debug_frame.pack_forget()

    def toggle_pause(self): (self.core.resume() if self.core.paused else self.core.pause())
    def update_debug_btn_state(self, paused): self.btn_pause.config(text="▶ 继续" if paused else "⏸ 暂停", bg=COLORS['success'] if paused else COLORS['warning'])

    def reset_ui_state(self): 
        self.core.running=False; self.btn_run.config(text="▶ 启动", bg=COLORS['success']); self.btn_run_menu.config(state='normal')
        self.show_debug_toolbar(False); self.update_debug_btn_state(False)
        [self.update_sensor_visual_safe(nid,False) for nid in self.editor.nodes]
        if self.log_panel and self.log_panel.winfo_exists(): self.log_panel.pack(side='bottom', fill='x', pady=(0,10), padx=10)
        if self.float_window: self.float_window.update_state()
    
    def update_sensor_visual_safe(self,nid,active): self.after(0,lambda: nid in self.editor.nodes and self.editor.nodes[nid].set_sensor_active(active))
    def log(self,msg, level='info'): self.log_q.put((msg, level))
    def _poll_log(self):
        while not self.log_q.empty(): item = self.log_q.get(); self.log_panel.add_log(item[0], item[1])
        self.after(100,self._poll_log)
    
    # [安全增强] UI线程安全操作
    def highlight_node_safe(self,nid,status=None): 
        def _task():
            if not self.editor.winfo_exists(): return
            self.editor.delete("hl")
            if nid and nid in self.editor.nodes:
                n = self.editor.nodes[nid]
                z = self.editor.zoom
                self.editor.create_rectangle(n.x*z-3*z, n.y*z-3*z, (n.x+n.w)*z+3*z, (n.y+n.h)*z+3*z, outline=COLORS.get(f"hl_{status}",COLORS['hl_ok']), width=3*z, tags="hl")
        self.after(0, _task)
    
    def select_node_safe(self, nid): self.after(0, lambda: self.editor.select_node(nid) if self.editor.winfo_exists() else None)

    def save(self):
        if (fpath:=filedialog.asksaveasfilename(defaultextension=".qflow",filetypes=[("QFlow Project","*.qflow")])):
            try:
                with open(fpath,'w',encoding='utf-8') as f: json.dump(self.editor.get_data(),f,indent=2,ensure_ascii=False)
                self.log(f"💾 保存成功: {os.path.basename(fpath)}", "success")
            except Exception as e: messagebox.showerror("保存失败",str(e))
    def load(self):
        if (fpath:=filedialog.askopenfilename(filetypes=[("QFlow Project","*.qflow")])):
            try:
                with open(fpath,'r',encoding='utf-8') as f: self.editor.load_data(json.load(f))
                self.log(f"📂 加载成功: {os.path.basename(fpath)}", "success")
            except Exception as e: messagebox.showerror("加载失败",str(e))
    def clear(self): (messagebox.askyesno("确认","清空画布？") and (self.editor.load_data({'nodes':{},'links':[]}),self.editor.add_node('start',100,100, save_history=False),self.log("🗑️ 画布已清空", "warning")))
    
    def _setup_hotkeys(self):
        def on_start(): self.after(0, lambda: self.toggle_run(None) if not self.core.running else None)
        def on_stop(): self.after(0, lambda: self.core.stop() if self.core.running else None)
        threading.Thread(target=lambda: keyboard.GlobalHotKeys({'<alt>+1': on_start, '<alt>+2': on_stop}).start(),daemon=True).start()
    
    def quit_app(self):
        if self.core.running: self.core.stop()
        self.destroy()

    def _on_closing(self): (not self.core.running or messagebox.askyesno("退出","确定要强制退出吗？")) and self.quit_app()

if __name__ == "__main__":
    app = App()
    app.mainloop()