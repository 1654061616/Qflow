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
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
from PIL import Image, ImageTk, ImageGrab, ImageChops
import pyautogui
from pynput import keyboard
import copy
from datetime import datetime
from collections import namedtuple

# --- 1. 依赖库检查 ---
try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("⚠️ 警告: 未安装 opencv-python，高级图像识别功能受限。")

try:
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioMeterInformation
    import comtypes 
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

# --- 2. 系统与配置管理 ---
pyautogui.FAILSAFE = False

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

def safe_float(value, default=0.0):
    try: return float(value)
    except (ValueError, TypeError): return default

def safe_int(value, default=0):
    try: return int(float(value))
    except (ValueError, TypeError): return default

# --- 主题定义 ---
THEMES = {
    'Dark': {
        'bg_app': '#202020', 'bg_sidebar': '#2B2B2B', 'bg_canvas': '#181818', 'bg_panel': '#2B2B2B',
        'bg_node': '#353535', 'bg_header': '#3c3c3c', 'bg_card': '#404040', 'fg_title': '#eeeeee',
        'fg_text': '#dcdcdc', 'fg_sub': '#aaaaaa', 'accent': '#64b5f6', 'grid': '#2a2a2a',
        'wire': '#777777', 'wire_active': '#dcdcaa', 'socket': '#cfcfcf', 'btn_bg': '#505050',
        'input_bg': '#222222'
    },
    'Light': {
        'bg_app': '#f0f0f0', 'bg_sidebar': '#e0e0e0', 'bg_canvas': '#ffffff', 'bg_panel': '#e0e0e0',
        'bg_node': '#f5f5f5', 'bg_header': '#d0d0d0', 'bg_card': '#ffffff', 'fg_title': '#333333',
        'fg_text': '#222222', 'fg_sub': '#555555', 'accent': '#1976d2', 'grid': '#eeeeee',
        'wire': '#a0a0a0', 'wire_active': '#ff9800', 'socket': '#888888', 'btn_bg': '#bbbbbb',
        'input_bg': '#ffffff'
    },
    'Hacker': {
        'bg_app': '#000000', 'bg_sidebar': '#0a0a0a', 'bg_canvas': '#000000', 'bg_panel': '#0a0a0a',
        'bg_node': '#111111', 'bg_header': '#003300', 'bg_card': '#111111', 'fg_title': '#00ff00',
        'fg_text': '#00cc00', 'fg_sub': '#008800', 'accent': '#00ff00', 'grid': '#002200',
        'wire': '#005500', 'wire_active': '#00ff00', 'socket': '#004400', 'btn_bg': '#002200',
        'input_bg': '#001100'
    }
}

COLORS = THEMES['Dark'].copy()
COLORS.update({
    'success': '#4caf50', 'danger': '#ef5350', 'warning': '#ffca28', 'control': '#ab47bc',
    'sensor': '#ff7043', 'var_node': '#26c6da', 'wire_hl': '#4fc3f7', 'shadow': '#101010',
    'hover': '#505050', 'select_box': '#4fc3f7', 'active_border': '#4fc3f7', 'marker': '#f44747',
    'btn_hover': '#606060', 'hl_running': '#ffeb3b', 'hl_ok': '#4caf50', 'hl_fail': '#f44747',
    'breakpoint': '#e53935', 'log_bg': '#1e1e1e', 'log_fg': '#d4d4d4'
})

FONTS = {
    'node_title': ('Segoe UI', int(10 * SCALE_FACTOR), 'bold'), 
    'node_text': ('Segoe UI', int(8 * SCALE_FACTOR)),
    'code': ('Consolas', int(9 * SCALE_FACTOR)), 
    'h2': ('Segoe UI', int(11 * SCALE_FACTOR), 'bold'), 
    'small': ('Segoe UI', int(8 * SCALE_FACTOR)),
    'log': ('Consolas', int(9 * SCALE_FACTOR))
}

SETTINGS = {
    'hotkey_start': '<alt>+1',
    'hotkey_stop': '<alt>+2',
    'theme': 'Dark'
}

LOG_LEVELS = {'info': {'color': '#64b5f6', 'icon': 'ℹ️'}, 'success': {'color': '#81c784', 'icon': '✅'}, 'warning': {'color': '#ffd54f', 'icon': '⚠️'}, 'error': {'color': '#e57373', 'icon': '❌'}, 'exec': {'color': '#9e9e9e', 'icon': '▶️'}, 'paused': {'color': '#fff176', 'icon': '⏸️'}}
NODE_WIDTH = int(200 * SCALE_FACTOR)
HEADER_HEIGHT = int(28 * SCALE_FACTOR)
PORT_START_Y = int(45 * SCALE_FACTOR)
PORT_STEP_Y = int(24 * SCALE_FACTOR)
GRID_SIZE = int(20 * SCALE_FACTOR)

NODE_CONFIG = {
    'start':    {'title': '▶ 开始', 'outputs': ['out'], 'color': '#2e7d32'},
    'end':      {'title': '⏹️ 结束', 'outputs': [], 'color': '#c62828'},
    'loop':     {'title': '🔄 循环', 'outputs': ['loop', 'exit'], 'color': '#7b1fa2'},
    'wait':     {'title': '⏳ 延时', 'outputs': ['out'], 'color': '#4527a0'},
    'mouse':    {'title': '👆 鼠标', 'outputs': ['out'], 'color': '#1565c0'},
    'keyboard': {'title': '⌨️ 键盘', 'outputs': ['out'], 'color': '#1565c0'},
    'cmd':      {'title': '💻 命令', 'outputs': ['out'], 'color': '#1565c0'},
    'web':      {'title': '🔗 网页', 'outputs': ['out'], 'color': '#0277bd'},
    'image':    {'title': '🎯 找图', 'outputs': ['found', 'timeout'], 'color': '#ef6c00'},
    'if_img':   {'title': '🔍 检测', 'outputs': ['yes', 'no'], 'color': '#ef6c00'},
    'if_static':{'title': '⏸️ 静止', 'outputs': ['yes', 'no'], 'color': '#d84315'},
    'if_sound': {'title': '🔊 声音', 'outputs': ['yes', 'no'], 'color': '#d84315'},
    'set_var':  {'title': '[x] 变量', 'outputs': ['out'], 'color': '#00838f'},
    'var_switch':{'title': '⎇ 分流', 'outputs': ['else'], 'color': '#00838f'},
    'sequence': {'title': '🔀 序列', 'outputs': ['else'], 'color': '#7b1fa2'},
    'reroute':  {'title': '●', 'outputs': ['out'], 'color': '#777777'}
}

PORT_TRANSLATION = {'out': '继续', 'yes': '是', 'no': '否', 'found': '找到', 'timeout': '超时', 'loop': '循环', 'exit': '退出', 'else': '否则'}
MOUSE_ACTIONS = {'click': '点击', 'move': '移动', 'drag': '拖拽', 'scroll': '滚动'}
MOUSE_BUTTONS = {'left': '左键', 'right': '右键', 'middle': '中键'}
ACTION_MAP = {'click': '单击左键', 'double_click': '双击左键', 'right_click': '单击右键', 'none': '不执行操作'}
MATCH_STRATEGY_MAP = {'hybrid': '智能混合', 'template': '模板匹配', 'feature': '特征匹配'}
VAR_OP_MAP = {'=': '等于', '!=': '不等于', 'exists': '已定义', 'not_exists': '未定义'}

# --- 3. 基础工具类 ---
class ImageUtils:
    @staticmethod
    def img_to_b64(image):
        try: buffered = io.BytesIO(); image.save(buffered, format="PNG"); return base64.b64encode(buffered.getvalue()).decode('utf-8')
        except: return None
    
    @staticmethod
    def b64_to_img(b64_str):
        if not b64_str or not isinstance(b64_str, str): return None
        try:
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
        except OSError: return None

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

# --- 4. 日志与核心 ---
class LogPanel(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=COLORS['bg_panel'], **kwargs)
        self.height_expanded, self.height_collapsed = 200, 30
        self.expanded = False
        self.toolbar = tk.Frame(self, bg=COLORS['bg_header'], height=28); self.toolbar.pack_propagate(False); self.toolbar.pack(fill='x')
        tk.Label(self.toolbar, text="📋 执行日志", bg=COLORS['bg_header'], fg='white', font=FONTS['node_title']).pack(side='left', padx=10)
        tk.Button(self.toolbar, text="🗑️", command=self.clear, bg=COLORS['bg_header'], fg=COLORS['danger'], bd=0).pack(side='right', padx=5)
        
        self.text_frame = tk.Frame(self, bg=COLORS['log_bg'])
        self.scrollbar = ttk.Scrollbar(self.text_frame)
        self.text_area = tk.Text(self.text_frame, bg=COLORS['log_bg'], fg=COLORS['log_fg'], font=FONTS['log'], state='disabled', yscrollcommand=self.scrollbar.set, bd=0, padx=5, pady=5)
        self.scrollbar.config(command=self.text_area.yview); self.scrollbar.pack(side='right', fill='y'); self.text_area.pack(side='left', fill='both', expand=True)
        for level, style in LOG_LEVELS.items(): self.text_area.tag_config(level, foreground=style['color'])
        self.pack(side='bottom', fill='x')
        self.config(height=self.height_expanded); self.text_frame.pack(fill='both', expand=True)

    def add_log(self, msg, level='info'):
        if not self.winfo_exists(): return
        self.text_area.config(state='normal')
        self.text_area.insert('end', f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n", level)
        self.text_area.see('end'); self.text_area.config(state='disabled')
    def clear(self): self.text_area.config(state='normal'); self.text_area.delete(1.0, 'end'); self.text_area.config(state='disabled')

class AutomationCore:
    def __init__(self, log_callback, app_instance):
        self.running = False; self.paused = False; self.stop_event = threading.Event(); self.pause_event = threading.Event()
        self.log = log_callback; self.app = app_instance; self.project = None; self.runtime_memory = {}; self.io_lock = threading.Lock()
        self.active_threads = 0; self.thread_lock = threading.Lock(); self.scaling_ratio = 1.0; self.breakpoints = set()
        self.max_threads = 50 

    def load_project(self, project_data):
        self.project = project_data; self.scaling_ratio = 1.0; self.breakpoints = set(project_data.get('breakpoints', []))
        dev_scale = self.project.get('metadata', {}).get('dev_scale_x', 1.0); runtime_scale_x, _ = get_scale_factor()
        if dev_scale > 0.1 and runtime_scale_x > 0.1: self.scaling_ratio = runtime_scale_x / dev_scale
        if self.project and 'nodes' in self.project:
            for nid, node in self.project['nodes'].items():
                data = node.get('data', {})
                try:
                    if 'b64' in data and 'image' not in data and (img := ImageUtils.b64_to_img(data['b64'])): self.project['nodes'][nid]['data']['image'] = img
                    if 'anchors' in data:
                        for anchor in data['anchors']:
                            if 'b64' in anchor and 'image' not in anchor and (img := ImageUtils.b64_to_img(anchor['b64'])): anchor['image'] = img
                    if 'images' in data:
                        for img_item in data['images']:
                            if 'b64' in img_item and 'image' not in img_item and (img := ImageUtils.b64_to_img(img_item['b64'])): img_item['image'] = img
                except Exception: pass

    def start(self, start_node_id=None):
        if self.running or not self.project: return
        self.running = True; self.paused = False; self.stop_event.clear(); self.pause_event.set(); self.runtime_memory = {}; self.active_threads = 0
        self.log("🚀 引擎启动", "exec"); self.app.iconify()
        threading.Thread(target=self._run_flow_engine, args=(start_node_id,), daemon=True).start()

    def stop(self):
        if not self.running: return
        self.stop_event.set(); self.pause_event.set(); self.log("🛑 正在停止...", "warning")
        self.app.after(0, self.app.reset_ui_state)

    def pause(self): 
        self.paused = True; self.pause_event.clear(); self.log("⏸️ 流程暂停", "paused")
        self.app.after(0, lambda: self.app.update_debug_btn_state(True))
        
    def resume(self): 
        self.paused = False; self.pause_event.set(); self.log("▶️ 流程继续", "info")
        self.app.after(0, lambda: self.app.update_debug_btn_state(False))
    
    def _smart_wait(self, seconds):
        end_time = time.time() + seconds
        while time.time() < end_time:
            if self.stop_event.is_set(): return False
            self._check_pause(); time.sleep(0.05)
        return True
    
    def _check_pause(self, node_id=None):
        if node_id and node_id in self.breakpoints:
            if not self.paused: self.log(f"🔴 命中断点: {node_id}", "paused"); self.pause(); self.app.after(0, self.app.deiconify)
        if not self.pause_event.is_set(): self.pause_event.wait()
    
    def _get_next_links(self, node_id, port_name='out'): return [l['target'] for l in self.project['links'] if l['source'] == node_id and l.get('source_port') == port_name]
    
    def _run_flow_engine(self, start_node_id=None):
        try:
            start_nodes = [start_node_id] if start_node_id else [nid for nid, n in self.project['nodes'].items() if n['type'] == 'start']
            if not start_nodes: self.log("未找到开始节点", "error"); return
            for start_id in start_nodes: self._fork_node(start_id)
            while not self.stop_event.is_set():
                with self.thread_lock: 
                    if self.active_threads <= 0: break
                time.sleep(0.5)
        except Exception as e: traceback.print_exc(); self.log(f"引擎异常: {str(e)}", "error")
        finally:
            self.running = False; self.log("🏁 流程结束", "info"); 
            self.app.highlight_node_safe(None); 
            self.app.after(0, self.app.deiconify); 
            self.app.after(100, self.app.reset_ui_state)

    def _fork_node(self, node_id):
        with self.thread_lock:
            if node_id not in self.project['nodes']: return
            if self.active_threads >= self.max_threads: return
            self.active_threads += 1
        threading.Thread(target=self._process_node_thread, args=(node_id,), daemon=True).start()

    def _process_node_thread(self, node_id):
        try:
            if self.stop_event.is_set(): return
            if not (node := self.project['nodes'].get(node_id)): return
            self._check_pause(node_id)
            if self.stop_event.is_set(): return
            self.app.highlight_node_safe(node_id, 'running'); self.app.select_node_safe(node_id)
            try: out_port = self._execute_node(node)
            except Exception as e: self.log(f"💥 节点执行错误: {e}", "error"); out_port = 'else'
            if out_port == '__STOP__' or self.stop_event.is_set(): return
            
            if node['type'] != 'reroute':
                self.log(f"↳ [{node.get('data',{}).get('_user_title','Node')}] -> {PORT_TRANSLATION.get(out_port, out_port)}", "exec")
            
            self.app.highlight_node_safe(node_id, 'fail' if out_port in ['timeout', 'no', 'exit', 'else'] else 'ok')
            time.sleep(0.01)
            for next_id in self._get_next_links(node_id, out_port):
                if self.stop_event.is_set(): break
                self._fork_node(next_id)
        finally:
            with self.thread_lock: self.active_threads -= 1

    def _replace_variables(self, text):
        if not isinstance(text, str): return str(text)
        try:
            for k, v in self.runtime_memory.items(): text = text.replace(f'${{{k}}}', str(v) if v is not None else "")
        except: pass
        return text

    def _execute_node(self, node):
        if self.stop_event.is_set(): return '__STOP__'
        ntype = node['type']; data = {k: (self._replace_variables(v) if isinstance(v, str) and '${' in v else v) for k, v in node.get('data', {}).items()}
        if ntype == 'reroute': return 'out'
        if ntype == 'start': return 'out'
        if ntype == 'end': self.stop_event.set(); return '__STOP__'
        if ntype == 'wait': return 'out' if self._smart_wait(safe_float(data.get('seconds', 1.0))) else '__STOP__'
        if ntype == 'set_var':
            if 'batch_vars' in data: [self.runtime_memory.update({i['name']:i['value']}) for i in data['batch_vars'] if i.get('name')]
            if data.get('var_name'): self.runtime_memory[data['var_name']] = data.get('var_value', '')
            return 'out'
        if ntype == 'var_switch':
            val = str(self.runtime_memory.get(data.get('var_name',''), ''))
            op, target = data.get('operator', '='), data.get('var_value', '')
            if data.get('var_name'): return 'yes' if ((val==target) if op=='=' else (val!=target)) else 'no'
            vals = [str(self.runtime_memory.get(vn.strip(), '')) for vn in data.get('var_list', '').split(',') if vn.strip()]
            for case in data.get('cases', []):
                if all(v == case.get('value', '') for v in vals): return case.get('id', 'else')
            return 'else'
        if ntype == 'sequence':
            for i in range(1, safe_int(data.get('num_steps', 3)) + 1):
                if self.stop_event.is_set(): return '__STOP__'
                target_id = (self._get_next_links(node['id'], str(i)) or [None])[0]
                if not target_id: continue
                if self._execute_node(self.project['nodes'][target_id]) in ['yes', 'found', 'out', 'loop']:
                    [self._fork_node(nid) for nid in self._get_next_links(target_id, self._execute_node(self.project['nodes'][target_id]))]
                    return '__STOP__'
            return 'else'
        if ntype == 'if_sound':
            if not HAS_AUDIO: return 'no'
            start_t = time.time()
            while time.time()-start_t < safe_float(data.get('timeout',10.0)):
                if self.stop_event.is_set(): return '__STOP__'
                if AudioEngine.get_max_audio_peak() > safe_float(data.get('threshold',0.02)): return 'yes'
                time.sleep(0.1)
            return 'no'
        
        # --- 图像搜索逻辑 ---
        if ntype == 'image':
            conf, timeout = safe_float(data.get('confidence', 0.9)), max(0.5, safe_float(data.get('timeout', 10.0)))
            
            search_region = None
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
                
                if bool(data.get('auto_scroll', False)):
                     scroll_amount = safe_int(data.get('scroll_amount', -500))
                     self.log(f"📜 未找到目标，自动滚动: {scroll_amount}", "exec")
                     with self.io_lock: 
                         pyautogui.scroll(scroll_amount)
                     if not self._smart_wait(0.8): return '__STOP__'

                if time.time() - start_time > timeout: break
                time.sleep(0.2)
            return 'timeout'

        if ntype == 'mouse':
            with self.io_lock:
                action, dur = data.get('mouse_action', 'click'), safe_float(data.get('duration', 0.5))
                tx, ty = (safe_int(data.get('x',0))/SCALE_X, safe_int(data.get('y',0))/SCALE_Y) if action in ['click','move','drag'] else (None, None)
                if action == 'click': pyautogui.click(x=tx, y=ty, clicks=safe_int(data.get('click_count', 1)), button=data.get('mouse_button', 'left'), duration=dur)
                elif action == 'move': pyautogui.moveTo(tx, ty, duration=dur)
                elif action == 'scroll': pyautogui.scroll(safe_int(data.get('amount', -500)))
            return 'out'
        if ntype == 'keyboard':
            with self.io_lock:
                if data.get('kb_mode', 'text') == 'text': pyautogui.write(data.get('text','')); (data.get('press_enter', False) and pyautogui.press('enter'))
                else: pyautogui.hotkey(*[x.strip() for x in data.get('key_name', 'enter').lower().split('+')])
            return 'out'
        if ntype == 'cmd':
            try: 
                if sys.platform == 'win32': subprocess.Popen(data.get('command', ''), shell=True)
                else: subprocess.Popen(data.get('command', ''), shell=True, executable='/bin/bash')
            except Exception as e: self.log(f"CMD错误: {e}", "error")
            return 'out'
        if ntype == 'web': webbrowser.open(data.get('url')); self._smart_wait(2); return 'out'
        if ntype == 'loop':
            if data.get('infinite', True): return 'loop'
            with self.io_lock:
                k = f"loop_{node['id']}"; c = self.runtime_memory.get(k, 0)
                if c < safe_int(data.get('count', 3)): self.runtime_memory[k] = c + 1; return 'loop'
                else: 
                    if k in self.runtime_memory: del self.runtime_memory[k]
                    return 'exit'
        if ntype == 'if_img':
            if not (imgs := data.get('images', [])): return 'no'
            hay = VisionEngine.capture_screen()
            for img in imgs:
                if not VisionEngine._advanced_match(img.get('image'), hay, safe_float(data.get('confidence',0.9)), self.stop_event, True, True, self.scaling_ratio, 'hybrid')[0]: return 'no'
            return 'yes'
        return 'out'

# --- 5. 历史记录 ---
class HistoryManager:
    def __init__(self, editor):
        self.editor = editor
        self.undo_stack = []
        self.redo_stack = []
        self.max_history = 50

    def save_state(self):
        state = self.editor.get_data()
        if self.undo_stack:
            last = json.dumps(self.undo_stack[-1], sort_keys=True)
            curr = json.dumps(state, sort_keys=True)
            if last == curr: return
        self.undo_stack.append(state)
        self.redo_stack.clear()
        if len(self.undo_stack) > self.max_history: self.undo_stack.pop(0)

    def undo(self, event=None):
        if not self.undo_stack: return
        self.redo_stack.append(self.editor.get_data())
        self.editor.load_data(self.undo_stack.pop())
        self.editor.app.property_panel.clear()

    def redo(self, event=None):
        if not self.redo_stack: return
        self.undo_stack.append(self.editor.get_data())
        self.editor.load_data(self.redo_stack.pop())
        self.editor.app.property_panel.clear()

class GraphNode:
    def __init__(self, canvas, node_id, ntype, x, y, data=None):
        self.canvas, self.id, self.type, self.x, self.y = canvas, node_id, ntype, x, y
        self.data = data if data is not None else {}
        cfg = NODE_CONFIG.get(ntype, {})
        self.title_text, self.header_color = cfg.get('title', ntype), cfg.get('color', COLORS['bg_header'])
        if '_user_title' not in self.data: self.data['_user_title'] = self.title_text
        
        self.outputs = cfg.get('outputs', [])
        if ntype == 'sequence': self.outputs = [str(i) for i in range(1, safe_int(self.data.get('num_steps', 3)) + 1)] + ['else']
        elif ntype == 'var_switch':
            if self.data.get('var_name'): self.outputs = ['yes', 'no']
            else: self.outputs = [c['id'] for c in self.data.get('cases', [])] + ['else']

        self.w = NODE_WIDTH
        self.h = 100 
        self.tags = (f"node_{self.id}", "node"); self.has_breakpoint = False
        self.widgets = [] 
        self.tk_thumbs_cache = []
        self.draw()

    def draw(self):
        z = self.canvas.zoom
        vx, vy, vw, vh = self.x*z, self.y*z, self.w*z, self.h*z
        self.canvas.delete(f"node_{self.id}")
        
        # --- 通用选中框 ---
        self.sel_rect = self.canvas.create_rectangle(vx-3*z, vy-3*z, vx+vw+3*z, vy+vh+3*z, outline=COLORS['accent'], width=4*z, tags=self.tags+('selection',), state='hidden')

        base_h = HEADER_HEIGHT + 10
        ports_h = max(1, len(self.outputs)) * PORT_STEP_Y
        widgets_h = 0
        self.has_widgets = False
        self.is_visual_node = self.type in ['image', 'if_img', 'if_static']
        
        if not self.is_visual_node and self.type not in ['reroute', 'start', 'end']:
            widgets_h = 35
            self.has_widgets = True
            
        img_display_h = 0
        toolbar_h = 0
        self.tk_thumbs_cache = [] 

        if self.is_visual_node:
            toolbar_h = 30 
            if self.type == 'if_img' and self.data.get('images'):
                img_list = self.data.get('images', [])
                count = len(img_list)
                if count > 0:
                    rows = math.ceil(count / 2.0)
                    row_h = 60 
                    img_display_h = (rows * row_h) + 10 
            else:
                target_img = None
                if self.type == 'image': target_img = self.data.get('image')
                elif self.type == 'if_static': target_img = self.data.get('roi_preview')
                
                if target_img:
                    try:
                        iw, ih = target_img.size
                        scale = (self.w - 8) / iw 
                        calc_h = int(ih * scale)
                        img_display_h = min(calc_h, 120) + 5
                    except: img_display_h = 80

        if self.type == 'reroute': 
            self.w, self.h = 30, 30
            self.canvas.create_oval(vx, vy, vx+vw, vy+vh, fill=COLORS['wire'], outline="", tags=self.tags+('body',))
            if self.id in self.canvas.selected_node_ids: self.canvas.itemconfig(self.sel_rect, state='normal')
            self.hover_rect = self.canvas.create_rectangle(vx, vy, vx+vw, vy+vh, tags=self.tags+('hover',), state='hidden')
            return
        
        self.h = PORT_START_Y + ports_h + widgets_h + toolbar_h + img_display_h + 5

        self.clear_widgets()
        
        self.canvas.create_rectangle(vx+4*z, vy+4*z, vx+vw+4*z, vy+vh+4*z, fill=COLORS['shadow'], outline="", tags=self.tags)
        self.body_item = self.canvas.create_rectangle(vx, vy, vx+vw, vy+vh, fill=COLORS['bg_node'], outline=COLORS['bg_node'], width=2*z, tags=self.tags+('body',))
        self.canvas.create_rectangle(vx, vy, vx+vw, vy+HEADER_HEIGHT*z, fill=self.header_color, outline="", tags=self.tags+('header',))
        
        self.canvas.create_text(vx+10*z, vy+14*z, text=self.data.get('_user_title', self.title_text), fill=COLORS['fg_title'], font=('Segoe UI', max(6, int(10*z)), 'bold'), anchor="w", tags=self.tags)
        if self.has_breakpoint: self.canvas.create_oval(vx+vw-12*z, vy+8*z, vx+vw-4*z, vy+16*z, fill=COLORS['breakpoint'], outline="white", width=1, tags=self.tags)

        if self.type != 'start':
            iy = self.get_input_port_y(visual=True)
            self.canvas.create_oval(vx-5*z, iy-5*z, vx+5*z, iy+5*z, fill=COLORS['socket'], outline=COLORS['bg_canvas'], width=2*z, tags=self.tags+('port_in',))
        
        port_labels = PORT_TRANSLATION.copy()
        if self.type == 'var_switch':
             for c in self.data.get('cases', []): port_labels[c['id']] = f"={c['value']}"

        for i, name in enumerate(self.outputs):
            py = self.get_output_port_y(i, visual=True)
            self.canvas.create_oval(vx+vw-5*z, py-5*z, vx+vw+5*z, py+5*z, fill=COLORS.get(f"socket_{name}", COLORS['socket']), outline=COLORS['bg_canvas'], width=2*z, tags=self.tags+(f'port_out_{name}','port_out',name))
            self.canvas.create_text(vx+vw-12*z, py, text=port_labels.get(name, name), fill=COLORS['fg_sub'], font=('Segoe UI', max(5, int(8*z))), anchor="e", tags=self.tags)

        self.widget_offset_y = PORT_START_Y + ports_h
        if self.has_widgets and z > 0.6: 
            self.render_widgets(vx, vy, vw, z)
            self.widget_offset_y += 35

        if self.is_visual_node:
            toolbar_y = vy + (self.widget_offset_y * z)
            tool_frame = tk.Frame(self.canvas, bg=COLORS['bg_node'])
            def cmd_snip(nid=self.id): self.canvas.select_node(nid); self.canvas.app.do_snip()
            def cmd_test(nid=self.id): self.canvas.select_node(nid); self.canvas.app.property_panel.start_test_match()
            btn_style = {'bg': '#404040', 'fg': '#eeeeee', 'bd': 0, 'activebackground': '#505050', 'font': FONTS['small']}
            tk.Button(tool_frame, text="📸 截取", command=cmd_snip, **btn_style).pack(side='left', fill='x', expand=True, padx=(0, 1), pady=0)
            tk.Button(tool_frame, text="⚡ 测试", command=cmd_test, **btn_style).pack(side='left', fill='x', expand=True, padx=(1, 0), pady=0)
            self.widgets.append(self.canvas.create_window(vx + vw/2, toolbar_y, window=tool_frame, width=vw-8*z, height=24*z, anchor='n', tags=self.tags))
            
            img_start_y = toolbar_y + 28*z

            if self.type == 'if_img' and self.data.get('images'):
                imgs = self.data.get('images', [])
                cell_w = (vw - 12*z) / 2
                cell_h = 55 * z
                
                for idx, item in enumerate(imgs):
                    if not item.get('image'): continue
                    col = idx % 2
                    row = idx // 2
                    ix = vx + 4*z + col * (cell_w + 4*z)
                    iy = img_start_y + row * (cell_h + 4*z)
                    
                    self.canvas.create_rectangle(ix, iy, ix+cell_w, iy+cell_h, fill='#000000', outline=COLORS['wire'], width=1, tags=self.tags)
                    thumb_img = item['image'].copy()
                    thumb_img.thumbnail((int(cell_w), int(cell_h)), Image.Resampling.LANCZOS)
                    tk_thumb = ImageTk.PhotoImage(thumb_img)
                    self.tk_thumbs_cache.append(tk_thumb)
                    
                    self.canvas.create_image(ix + cell_w/2, iy + cell_h/2, image=tk_thumb, anchor='center', tags=self.tags)
                    self.canvas.create_text(ix+3*z, iy+3*z, text=str(idx+1), fill='white', font=('Segoe UI', int(8*z), 'bold'), anchor='nw', tags=self.tags)

            elif img_display_h > 0:
                target_img = self.data.get('image') if self.type == 'image' else self.data.get('roi_preview')
                if target_img:
                    disp_w = int(vw - 8*z)
                    disp_h = int((img_display_h - 5) * z)
                    thumb = target_img.copy()
                    thumb.thumbnail((disp_w, disp_h), Image.Resampling.LANCZOS)
                    tk_thumb = ImageTk.PhotoImage(thumb)
                    self.tk_thumbs_cache.append(tk_thumb)
                    self.canvas.create_rectangle(vx+4*z, img_start_y, vx+vw-4*z, img_start_y+disp_h, fill='#000000', outline=COLORS['wire'], width=1, tags=self.tags)
                    self.canvas.create_image(vx + vw/2, img_start_y + disp_h/2, image=tk_thumb, anchor='center', tags=self.tags)

        if self.id in self.canvas.selected_node_ids: self.canvas.itemconfig(self.sel_rect, state='normal')
        self.hover_rect = self.canvas.create_rectangle(vx-1*z, vy-1*z, vx+vw+1*z, vy+vh+1*z, outline=COLORS['hover'], width=1*z, state='hidden', tags=self.tags+('hover',))

    def render_widgets(self, vx, vy, vw, z):
        y_cursor = vy + (self.widget_offset_y * z) 
        
        def create_entry(key, default, label_txt, width=8):
            val = self.data.get(key, default)
            frame = tk.Frame(self.canvas, bg=COLORS['bg_node'])
            tk.Label(frame, text=label_txt, bg=COLORS['bg_node'], fg=COLORS['fg_sub'], font=FONTS['small']).pack(side='left')
            e = tk.Entry(frame, bg=COLORS['input_bg'], fg='white', bd=0, width=width, insertbackground='white', font=FONTS['code'])
            e.insert(0, str(val))
            e.pack(side='left', padx=5)
            e.bind("<FocusOut>", lambda ev: self.update_data(key, e.get())) 
            e.bind("<Return>", lambda ev: [self.update_data(key, e.get()), self.canvas.focus_set()])
            self.widgets.append(self.canvas.create_window(vx + 10*z, y_cursor, window=frame, anchor='nw', tags=self.tags))

        def create_combo(key, options_map, default, width=8):
            if isinstance(options_map, dict):
                options = list(options_map.values())
                curr_val = self.data.get(key, default)
                disp_val = options_map.get(curr_val, curr_val)
                map_inv = {v: k for k, v in options_map.items()}
            else:
                options = options_map
                disp_val = self.data.get(key, default)
                map_inv = None

            cb = ttk.Combobox(self.canvas, values=options, state='readonly', width=width, font=FONTS['code'])
            try: cb.set(disp_val)
            except: pass
            
            def on_sel(ev):
                val = cb.get()
                final_val = map_inv.get(val, val) if map_inv else val
                self.update_data(key, final_val)
                
            cb.bind("<<ComboboxSelected>>", on_sel)
            self.widgets.append(self.canvas.create_window(vx + 10*z, y_cursor, window=cb, anchor='nw', tags=self.tags))
            
        if self.type == 'wait': create_entry('seconds', '1.0', '等待(s):')
        elif self.type == 'loop': create_entry('count', '5', '循环:')
        elif self.type == 'keyboard': create_entry('text', '', '文本:', width=10)
        elif self.type == 'cmd': create_entry('command', '', '命令:', width=12)
        elif self.type == 'set_var': 
            f = tk.Frame(self.canvas, bg=COLORS['bg_node'])
            e1 = tk.Entry(f, bg=COLORS['input_bg'], fg='white', bd=0, width=5, font=FONTS['code']); e1.insert(0, self.data.get('var_name','')); e1.pack(side='left')
            e1.bind("<FocusOut>", lambda e: self.update_data('var_name', e1.get()))
            tk.Label(f, text="=", bg=COLORS['bg_node'], fg='white').pack(side='left')
            e2 = tk.Entry(f, bg=COLORS['input_bg'], fg='white', bd=0, width=5, font=FONTS['code']); e2.insert(0, self.data.get('var_value','')); e2.pack(side='left')
            e2.bind("<FocusOut>", lambda e: self.update_data('var_value', e2.get()))
            self.widgets.append(self.canvas.create_window(vx + 10*z, y_cursor, window=f, anchor='nw', tags=self.tags))
        
        elif self.type == 'mouse': create_combo('mouse_action', MOUSE_ACTIONS, 'click', width=12)

    def clear_widgets(self):
        for w in self.widgets: self.canvas.delete(w)
        self.widgets.clear()

    def update_data(self, key, value):
        if str(self.data.get(key)) == str(value): return
        self.canvas.history.save_state()
        self.data[key] = value
        if key == 'cases' or key == 'var_name': self.draw() 
        if key in ['image', 'images', 'roi_preview']: self.draw() 
        if self.canvas.app.property_panel.current_node == self:
            self.canvas.app.property_panel.load_node(self)

    def set_sensor_active(self,is_active): self.canvas.itemconfig(self.body_item,outline=COLORS['active_border'] if is_active else COLORS['bg_node'])
    def get_input_port_y(self,visual=False): 
        if self.type == 'reroute': return (self.y + 15)*self.canvas.zoom if visual else self.y + 15
        offset=HEADER_HEIGHT+14; return (self.y+offset)*self.canvas.zoom if visual else self.y+offset
    def get_output_port_y(self,index=0,visual=False): 
        if self.type == 'reroute': return (self.y + 15)*self.canvas.zoom if visual else self.y + 15
        offset=PORT_START_Y+(index*PORT_STEP_Y); return (self.y+offset)*self.canvas.zoom if visual else self.y+offset
    def get_port_y_by_name(self, port_name, visual=False):
        try: idx = self.outputs.index(port_name)
        except ValueError: idx = 0
        return self.get_output_port_y(idx, visual)
        
    def set_pos(self,x,y): self.x,self.y=x,y; self.draw()
    def set_selected(self,selected): self.canvas.itemconfig(self.sel_rect,state='normal' if selected else 'hidden'); (selected and self.canvas.tag_lower(self.sel_rect, self.body_item))
    def contains(self,log_x,log_y): return self.x<=log_x<=self.x+self.w and self.y<=log_y<=self.y+self.h
    
    def update_position(self, dx, dy):
        self.canvas.move(self.tags[0], dx, dy)

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
        self.bind("<Configure>",self.full_redraw)
    
    def on_rmb_press(self, event): self._rmb_start = (event.x, event.y); self._rmb_moved = False; self.scan_mark(event.x, event.y)
    def on_rmb_drag(self, event): (abs(event.x-self._rmb_start[0])>5 or abs(event.y-self._rmb_start[1])>5) and setattr(self,'_rmb_moved',True); self._rmb_moved and (self.config(cursor="fleur"),self.scan_dragto(event.x, event.y, gain=1),self._draw_grid())
    def on_rmb_release(self, event): self.config(cursor="arrow"); (not getattr(self,'_rmb_moved',False) and self.on_right_click_menu(event))
    def on_pan_start(self,event): self.config(cursor="fleur");self.scan_mark(event.x,event.y)
    def on_pan_drag(self,event): self.scan_dragto(event.x,event.y,gain=1);self._draw_grid()
    def on_pan_end(self,event): self.config(cursor="arrow")
    def _on_delete_press(self,e): 
        if self.selected_node_ids:
            self.history.save_state(); to_del = list(self.selected_node_ids)
            for nid in to_del: self.delete_node(nid)
            self.select_node(None)
    def get_logical_pos(self,event_x,event_y): return self.canvasx(event_x)/self.zoom,self.canvasy(event_y)/self.zoom
    def full_redraw(self,event=None): 
        self.config(bg=COLORS['bg_canvas']) 
        self.delete("all");self._draw_grid(); [n.draw() for n in self.nodes.values()]; self.redraw_links()

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
            self.links = [l for l in self.links if l['source'] != node_id and l['target'] != node_id]
            self.nodes[node_id].clear_widgets()
            self.delete(f"node_{node_id}"); del self.nodes[node_id]
            self.redraw_links()

    def on_scroll(self, e):
        old_zoom = self.zoom; new_zoom = max(0.4, min(3.0, self.zoom * (1.1 if e.delta > 0 else 0.9)))
        if new_zoom == self.zoom: return
        self.zoom = new_zoom; self.full_redraw()

    def on_lmb_press(self,event):
        lx,ly=self.get_logical_pos(event.x,event.y); vx,vy=self.canvasx(event.x),self.canvasy(event.y)
        
        items = self.find_overlapping(vx-2,vy-2,vx+2,vy+2)
        for item in items:
            t_list = self.gettags(item)
            if "port_out" in t_list and (nid:=next((t[5:] for t in t_list if t.startswith("node_")),None)) and nid in self.nodes:
                self.wire_start={'node':self.nodes[nid],'port':next((t for t in t_list if t in self.nodes[nid].outputs),'out')};self.drag_data={"type":"wire"}; return
        
        clicked_node=next((node for node in reversed(list(self.nodes.values())) if node.contains(lx,ly)),None)
        if clicked_node:
            if not (event.state & 0x0004): 
                if clicked_node.id not in self.selected_node_ids: self.select_node(clicked_node.id)
            else: self.select_node(clicked_node.id, add=True)
            self.drag_data = {
                "type": "node", 
                "last_vx": vx, 
                "last_vy": vy,
                "dragged": False
            }
            self.history.save_state(); [self.tag_raise(f"node_{nid}") for nid in self.selected_node_ids]
        else:
            if not (event.state & 0x0004): self.select_node(None)
            self.drag_data = {"type": "box_select", "start_vx": vx, "start_vy": vy}; self.selection_box = self.create_rectangle(vx, vy, vx, vy, outline=COLORS['select_box'], width=2, dash=(4,4), tags="selection_box")

    def on_lmb_drag(self,event):
        lx,ly=self.get_logical_pos(event.x,event.y); vx,vy=self.canvasx(event.x),self.canvasy(event.y)
        
        if self.drag_data["type"]=="node":
            self.drag_data["dragged"] = True
            dx = vx - self.drag_data["last_vx"]
            dy = vy - self.drag_data["last_vy"]
            self.drag_data["last_vx"] = vx
            self.drag_data["last_vy"] = vy
            
            for nid in self.selected_node_ids:
                if nid in self.nodes:
                    node = self.nodes[nid]
                    node.x += dx / self.zoom
                    node.y += dy / self.zoom
                    node.update_position(dx, dy)
            self.redraw_links()
            
        elif self.drag_data["type"]=="box_select":
            if self.selection_box: self.coords(self.selection_box, self.drag_data["start_vx"], self.drag_data["start_vy"], vx, vy)
        elif self.drag_data["type"]=="wire":
            if self.temp_wire: self.delete(self.temp_wire)
            n,p=self.wire_start['node'],self.wire_start['port']
            x_start = (n.x+n.w)*self.zoom if n.type!='reroute' else (n.x+15)*self.zoom
            self.temp_wire=self.draw_bezier(x_start, n.get_port_y_by_name(p,visual=True),vx,vy,state="active")

    def on_lmb_release(self,event):
        lx,ly=self.get_logical_pos(event.x,event.y)
        
        if self.drag_data.get("type")=="node":
            if self.drag_data.get("dragged", False):
                for nid in self.selected_node_ids:
                    if nid in self.nodes:
                        self.nodes[nid].set_pos(round(self.nodes[nid].x/GRID_SIZE)*GRID_SIZE, round(self.nodes[nid].y/GRID_SIZE)*GRID_SIZE)
                self.redraw_links()
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
                is_reroute = node.type == 'reroute'
                dist = math.hypot(lx-(node.x+15 if is_reroute else node.x), ly-node.get_input_port_y(visual=False))
                if node.id!=self.wire_start['node'].id and dist < (30/self.zoom):
                    if node.type=='start': continue
                    self.history.save_state(); self.links.append({'id':str(uuid.uuid4()),'source':self.wire_start['node'].id,'source_port':self.wire_start['port'],'target':node.id}); self.redraw_links(); break
        self.drag_data,self.wire_start,self.temp_wire={"type":None},None,None
    
    def select_node(self, node_id, add=False):
        if not add: [self.nodes[nid].set_selected(False) for nid in self.selected_node_ids if nid in self.nodes]; self.selected_node_ids.clear()
        if node_id and node_id in self.nodes: 
            self.selected_node_ids.add(node_id); self.nodes[node_id].set_selected(True)
        if not self.selected_node_ids: self.app.property_panel.show_empty()
        elif len(self.selected_node_ids) == 1: self.app.property_panel.load_node(self.nodes[next(iter(self.selected_node_ids))])
        else: self.app.property_panel.show_multi_select(len(self.selected_node_ids))
        self.redraw_links()

    def draw_bezier(self,x1,y1,x2,y2,state="normal",link_id=None, highlighted=False):
        offset=max(50*self.zoom,abs(x1-x2)*0.5); width = 4*self.zoom if highlighted else (3*self.zoom if state=="active" else 2*self.zoom)
        color = COLORS['wire_hl'] if highlighted else COLORS['wire_active' if state=="active" else 'wire']
        return self.create_line(x1,y1,x1+offset,y1,x2-offset,y2,x2,y2,smooth=True,splinesteps=50,fill=color,width=width,arrow=tk.NONE,tags=("link",)+((f"link_{link_id}",) if link_id else ()))
    
    def redraw_links(self):
        self.delete("link"); 
        for l in self.links:
            if l['source'] in self.nodes and l['target'] in self.nodes:
                n1,n2=self.nodes[l['source']],self.nodes[l['target']]
                x1 = (n1.x + n1.w)*self.zoom if n1.type != 'reroute' else (n1.x + 15)*self.zoom
                y1 = n1.get_port_y_by_name(l.get('source_port','out'),visual=True)
                x2 = n2.x*self.zoom if n2.type != 'reroute' else (n2.x + 15)*self.zoom
                y2 = n2.get_input_port_y(visual=True)
                self.draw_bezier(x1,y1,x2,y2,link_id=l['id'], highlighted=(l['source'] in self.selected_node_ids or l['target'] in self.selected_node_ids))
        self.tag_lower("link"); self.tag_lower("grid")
    
    def align_nodes(self, mode):
        if len(self.selected_node_ids) < 2: return
        self.history.save_state()
        nodes = [self.nodes[nid] for nid in self.selected_node_ids if nid in self.nodes]
        if mode == 'left':
            target_x = min(n.x for n in nodes)
            for n in nodes: n.set_pos(target_x, n.y)
        elif mode == 'top':
            target_y = min(n.y for n in nodes)
            for n in nodes: n.set_pos(n.x, target_y)
        self.redraw_links()

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
        node = next((n for n in reversed(list(self.nodes.values())) if n.contains(lx, ly)), None)
        
        m=tk.Menu(self,tearoff=0,bg=COLORS['bg_card'],fg=COLORS['fg_text'],font=FONTS['small'])
        if node:
            m.add_command(label="📥 复制",command=lambda: (self.history.save_state(), self.add_node(node.type, node.x+20, node.y+20, data=copy.deepcopy(node.data), save_history=False)))
            m.add_command(label="🔴 断点",command=lambda: setattr(node, 'has_breakpoint', not node.has_breakpoint) or node.draw())
            m.add_separator()
            m.add_command(label="❌ 删除",command=lambda: (self.history.save_state(), self.delete_node(node.id)),foreground=COLORS['danger'])
        else:
            if len(self.selected_node_ids) > 1:
                m.add_command(label="⬅ 左对齐", command=lambda: self.align_nodes('left'))
                m.add_command(label="⬆ 顶对齐", command=lambda: self.align_nodes('top'))
        
        m.post(event.x_root,event.y_root)

    def sanitize_data_for_json(self, data):
        if isinstance(data, dict):
            new_dict = {}
            for k, v in data.items():
                if k in ['image', 'tk_image', 'roi_preview']: continue 
                if isinstance(v, (Image.Image, ImageTk.PhotoImage)): continue
                new_dict[k] = self.sanitize_data_for_json(v)
            return new_dict
        elif isinstance(data, list):
            return [self.sanitize_data_for_json(item) for item in data]
        else:
            return data

    def get_data(self):
        nodes_d = {}
        for nid, n in self.nodes.items():
            clean_data = self.sanitize_data_for_json(n.data)
            if 'image' in n.data and 'b64' not in clean_data: clean_data['b64'] = ImageUtils.img_to_b64(n.data['image'])
            nodes_d[nid]={'id':nid,'type':n.type,'x':int(n.x),'y':int(n.y),'data':clean_data, 'breakpoint': n.has_breakpoint}
        breakpoints = [nid for nid, n in self.nodes.items() if n.has_breakpoint]
        return {'nodes':nodes_d, 'links':self.links, 'breakpoints': breakpoints, 'metadata':{'dev_scale_x':SCALE_X,'dev_scale_y':SCALE_Y}}

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
            self.full_redraw()
        except Exception as e: self.app.log(f"❌ 加载失败: {e}", "error")

# --- 6. 属性面板 ---
class PropertyPanel(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=COLORS['bg_panel'])
        self.app, self.current_node = app, None
        self.static_monitor_active = False
        self.is_monitoring_audio = False
        
        header = tk.Frame(self, bg=COLORS['bg_sidebar'], height=40)
        header.pack(fill='x')
        tk.Label(header, text="属性设置", bg=COLORS['bg_sidebar'], fg=COLORS['fg_sub'], font=('Microsoft YaHei', 10, 'bold')).pack(side='left', padx=10, pady=10)
        
        self.scrollbar = ttk.Scrollbar(self, orient="vertical")
        self.scrollbar.pack(side='right', fill='y')
        self.canvas = tk.Canvas(self, bg=COLORS['bg_panel'], highlightthickness=0, yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side='left', fill='both', expand=True)
        self.scrollbar.config(command=self.canvas.yview)
        
        self.content = tk.Frame(self.canvas, bg=COLORS['bg_panel'], padx=10, pady=10)
        self.content_id = self.canvas.create_window((0, 0), window=self.content, anchor='nw')
        
        self.content.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig(self.content_id, width=e.width))
        self.canvas.bind("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        
        self.show_empty()
    
    def clear(self): 
        for w in self.content.winfo_children(): w.destroy()
        self.current_node = None
        self.static_monitor_active = False
        self.is_monitoring_audio = False

    def show_empty(self): 
        self.clear()
        tk.Label(self.content, text="未选择节点", bg=COLORS['bg_panel'], fg=COLORS['fg_sub'], font=FONTS['small']).pack(pady=40)
    
    def show_multi_select(self, count): 
        self.clear()
        tk.Label(self.content, text=f"选中 {count} 个节点", bg=COLORS['bg_panel'], fg=COLORS['accent']).pack(pady=40)

    def load_node(self, node):
        self.clear()
        self.current_node = node
        ntype, data = node.type, node.data
        
        SOUND_MODES = {'has_sound': '检测声音', 'is_silent': '检测静音'}
        KB_MODES = {'text': '输入文本', 'key': '按键组合'}
        MOUSE_CLICKS = {'1': '单击', '2': '双击'} 

        if ntype != 'reroute':
            self._input(self.content, "节点名称", '_user_title', data.get('_user_title', node.title_text))

        # --- 1. 逻辑与控制类 ---
        if ntype == 'wait':
             self._input(self.content, "等待秒数", 'seconds', data.get('seconds', 1.0), safe_float)
        
        elif ntype == 'loop':
             self._chk(self.content, "无限循环", 'infinite', data.get('infinite', True))
             if not data.get('infinite', True):
                 self._input(self.content, "循环次数", 'count', data.get('count', 5), safe_int)
        
        elif ntype == 'sequence':
             sec = self._create_section("逻辑链设置")
             self._input(sec, "分支尝试数量", 'num_steps', data.get('num_steps', 3), safe_int)
        
        elif ntype == 'set_var':
            sec = self._create_section("变量设置")
            tk.Label(sec, text="每行 'name=value':", bg=sec.cget('bg'), fg=COLORS['fg_text'], font=FONTS['small']).pack(anchor='w')
            txt = tk.Text(sec, height=5, bg=COLORS['input_bg'], fg='white', bd=0, font=FONTS['code'])
            txt.pack(fill='x', pady=(2,5))
            
            existing = ""
            for item in data.get('batch_vars', []): existing += f"{item.get('name')}={item.get('value')}\n"
            if not existing and data.get('var_name'): existing = f"{data.get('var_name')}={data.get('var_value')}"
            txt.insert('1.0', existing)
            
            def save_vars(ev=None):
                res = [{'name':line.split('=')[0].strip(), 'value':line.split('=')[1].strip()} for line in txt.get('1.0', 'end').strip().split('\n') if '=' in line]
                self._save('batch_vars', res)
            txt.bind("<FocusOut>", save_vars)
            self._btn(sec, "💾 保存变量列表", save_vars)

        elif ntype == 'var_switch':
             if data.get('var_name'):
                 sec = self._create_section("单变量分流")
                 self._input(sec, "变量名", 'var_name', data.get('var_name'))
                 self._combo(sec, "操作", 'operator', list(VAR_OP_MAP.values()), VAR_OP_MAP.get(data.get('operator','=')), lambda e:self._save('operator', {v: k for k, v in VAR_OP_MAP.items()}.get(e.widget.get())))
                 self._input(sec, "对比值", 'var_value', data.get('var_value',''))
             else:
                 sec = self._create_section("多变量分流")
                 self._input(sec, "变量列表(逗号隔开)", 'var_list', data.get('var_list',''))
                 cases_frame = tk.Frame(sec, bg=sec.cget('bg')); cases_frame.pack(fill='x', pady=5)
                 
                 def update_cases():
                     node.outputs = [c['id'] for c in data.get('cases',[])] + ['else']
                     node.h = PORT_START_Y + max(1, len(node.outputs)) * PORT_STEP_Y
                     node.draw(); self.app.editor.redraw_links(); self.load_node(node)
                 
                 def add_case(): 
                     data.setdefault('cases', []).append({'value':'new','id':uuid.uuid4().hex})
                     update_cases()
                 
                 for i, c in enumerate(data.get('cases', [])):
                     r = tk.Frame(cases_frame, bg=COLORS['bg_card']); r.pack(fill='x', pady=1)
                     e = tk.Entry(r, bg=COLORS['input_bg'], width=10, fg='white', bd=0, font=FONTS['code']); e.insert(0, c.get('value'))
                     e.pack(side='left', fill='x', expand=True, padx=5, pady=3)
                     e.bind("<FocusOut>", lambda ev, idx=i: data['cases'][idx].update({'value':ev.widget.get()}) or node.draw())
                     self._btn_icon(r, "❌", lambda idx=i: [data['cases'].pop(idx), update_cases()], color=COLORS['danger'])
                 self._btn(sec, "➕ 添加条件", add_case)

        # --- 2. 动作执行类 ---
        elif ntype == 'cmd':
             self._input(self.content, "系统命令", 'command', data.get('command', ''))
             tk.Label(self.content, text="例: calc.exe 或 python script.py", bg=COLORS['bg_panel'], fg=COLORS['fg_sub'], font=FONTS['small']).pack(anchor='w')

        elif ntype == 'web':
             self._input(self.content, "URL", 'url', data.get('url', ''))

        elif ntype == 'mouse':
            sec = self._create_section("鼠标操作")
            current_action = data.get('mouse_action', 'click')
            
            # 动作类型
            def on_action_change(e):
                val = {v: k for k, v in MOUSE_ACTIONS.items()}.get(e.widget.get())
                self._save('mouse_action', val)
                self.load_node(self.current_node) 

            self._combo(sec, "动作类型", 'mouse_action', list(MOUSE_ACTIONS.values()), MOUSE_ACTIONS.get(current_action, '点击'), on_action_change)
            
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
             curr_kb = data.get('kb_mode', 'text')
             self._combo(sec, "类型", 'kb_mode', list(KB_MODES.values()), KB_MODES.get(curr_kb), lambda e: [self._save('kb_mode', {v: k for k, v in KB_MODES.items()}.get(e.widget.get())), self.load_node(self.current_node)])
             
             if curr_kb == 'text':
                 self._input(sec, "文本内容",'text',data.get('text',''))
                 self._chk(sec, "模拟打字 (慢速)", 'slow_type', data.get('slow_type', False))
                 self._chk(sec, "输入后按回车", 'press_enter', data.get('press_enter', False))
             else: 
                 self._input(sec, "组合键",'key_name',data.get('key_name','enter'))
                 tk.Label(sec, text="例: ctrl+c, alt+tab", bg=COLORS['bg_panel'], fg=COLORS['fg_sub'], font=FONTS['small']).pack(anchor='w')

        # --- 3. 视觉与感知类 ---
        elif ntype == 'image':
            base_sec = self._create_section("基础操作")
            if 'tk_image' in data and data['tk_image']: self._draw_image_preview(base_sec, data)
            self._btn(base_sec, "📸 重新/截取目标", self.app.do_snip)
            
            anchors = data.get('anchors', [])
            if anchors:
                anchor_sec = self._create_section(f"锚点列表 ({len(anchors)})")
                for i, anc in enumerate(anchors):
                    row = tk.Frame(anchor_sec, bg=COLORS['bg_card'], pady=2); row.pack(fill='x', pady=1)
                    tk.Label(row, text=f"锚点 {i+1}", bg=COLORS['bg_card'], fg=COLORS['success'], width=8, anchor='w').pack(side='left', padx=5)
                    self._btn_icon(row, "🗑️", lambda idx=i: self._delete_anchor(idx), color=COLORS['danger'])

            search_sec = self._create_section("搜索参数")
            self._combo(search_sec, "匹配策略",'match_strategy',list(MATCH_STRATEGY_MAP.values()),MATCH_STRATEGY_MAP.get(data.get('match_strategy','hybrid')),lambda e:self._save('match_strategy',{v: k for k, v in MATCH_STRATEGY_MAP.items()}.get(e.widget.get())))
            self._input(search_sec, "相似度",'confidence',data.get('confidence',0.9), safe_float)
            self._input(search_sec, "超时(秒)",'timeout',data.get('timeout',10.0), safe_float)
            self._chk(search_sec, "启用自动滚动", 'auto_scroll', data.get('auto_scroll', False))
            if data.get('auto_scroll'): self._input(search_sec, "滚动量", 'scroll_amount', data.get('scroll_amount', -500), safe_int)
            
            action_sec = self._create_section("执行动作")
            self._combo(action_sec, "动作",'click_type',list(ACTION_MAP.values()),ACTION_MAP.get(data.get('click_type','click')),lambda e:self._save('click_type', {v: k for k, v in ACTION_MAP.items()}.get(e.widget.get())))
            
            off_frame = tk.Frame(action_sec, bg=action_sec.cget('bg')); off_frame.pack(fill='x', pady=5)
            self._compact_input(off_frame, "偏移 X", 'offset_x', data.get('offset_x', 0), safe_int)
            self._compact_input(off_frame, "Y", 'offset_y', data.get('offset_y', 0), safe_int)
            self._btn_icon(off_frame, "🎯", self.open_visual_offset_picker, bg=COLORS['control'], width=3)

            self._btn(search_sec, "🧪 测试匹配", self.start_test_match)
            self.test_result_label = tk.Label(search_sec, bg=search_sec.cget('bg'), fg=COLORS['fg_sub'], font=FONTS['small']); self.test_result_label.pack(fill='x')
        
        elif ntype == 'if_img':
            sec = self._create_section("检测条件")
            if (images := data.get('images', [])):
                for img_data in images:
                    f = tk.Frame(sec, bg=COLORS['bg_card']); f.pack(fill='x', pady=2)
                    if img_data.get('tk_image'): 
                        c = tk.Canvas(f, width=40, height=22, bg='black', highlightthickness=0)
                        c.pack(side='left', padx=5)
                        c.create_image(20, 11, image=img_data['tk_image'], anchor='center')
                    
                    self._btn_icon(f, "❌", lambda i=img_data.get('id'): self._delete_image_condition(i), color=COLORS['danger'])
            self._btn(sec, "➕ 添加截图条件", self.app.do_snip)
            
            param_sec = self._create_section("参数设置")
            self._input(param_sec, "相似度",'confidence',data.get('confidence',0.9), safe_float)
            self._input(param_sec, "超时(秒)",'timeout',data.get('timeout',10.0), safe_float)

            self._btn(sec, "🧪 测试所有条件", self.start_test_match)
            self.test_result_label = tk.Label(sec, bg=sec.cget('bg'), fg=COLORS['fg_sub'], font=FONTS['small']); self.test_result_label.pack(fill='x')

        elif ntype == 'if_static':
             base_sec = self._create_section("监控区域")
             if 'roi_preview' in data and data['roi_preview']: 
                 c = tk.Canvas(base_sec, width=240, height=135, bg='black', highlightthickness=0); c.pack(pady=5)
                 c.create_image(120, 67, image=data['roi_preview'], anchor='center')
             self._btn(base_sec, "📸 截取监控区域", self.app.do_snip)
             
             param_sec = self._create_section("检测参数")
             self._input(param_sec, "静止持续(s)", 'duration', data.get('duration', 5.0), safe_float)
             self._input(param_sec, "最大超时(s)", 'timeout', data.get('timeout', 20.0), safe_float)
             self._input(param_sec, "灵敏度(0-1)", 'threshold', data.get('threshold', 0.98), safe_float)
             
             monitor_frame = self._create_section("实时测试")
             self.lbl_monitor_status = tk.Label(monitor_frame, text="等待启动...", bg=monitor_frame.cget('bg'), fg=COLORS['fg_sub'], font=('Consolas', 9))
             self.lbl_monitor_status.pack(fill='x', pady=5)
             self.btn_monitor = self._btn(monitor_frame, "🔴 启动监控", self._toggle_static_monitor)

        elif ntype == 'if_sound':
             sec = self._create_section("声音检测")
             curr_mode = data.get('detect_mode', 'has_sound')
             self._combo(sec, "模式", 'detect_mode', list(SOUND_MODES.values()), SOUND_MODES.get(curr_mode), lambda e:self._save('detect_mode', {v: k for k, v in SOUND_MODES.items()}.get(e.widget.get())))
             self._input(sec, "阈值(0-1)", 'threshold', data.get('threshold', 0.02), safe_float)
             self._input(sec, "超时(秒)", 'timeout', data.get('timeout', 10.0), safe_float)
             btn_text = "⏹ 停止" if self.is_monitoring_audio else "🔊 实时监测"
             self.monitor_audio_btn = self._btn(sec, btn_text, self._toggle_audio_monitor)

    # --- 辅助方法 ---
    def _create_section(self, text):
        f = tk.Frame(self.content, bg=COLORS['bg_panel'], pady=5)
        f.pack(fill='x')
        tk.Label(f, text=text, bg=COLORS['bg_panel'], fg=COLORS['accent'], font=('Segoe UI', 9, 'bold')).pack(anchor='w')
        tk.Frame(f, height=1, bg=COLORS['bg_header']).pack(fill='x', pady=(2, 5))
        return f

    def _input(self, parent, label, key, val, validation_func=None):
        f = tk.Frame(parent, bg=parent.cget('bg')); f.pack(fill='x', pady=2)
        tk.Label(f, text=label, bg=parent.cget('bg'), fg=COLORS['fg_text'], font=FONTS['small']).pack(side='left', padx=(0,5))
        e = tk.Entry(f, bg=COLORS['input_bg'], fg='white', bd=0, insertbackground='white', font=FONTS['code'])
        e.insert(0, str(val))
        e.pack(fill='x', pady=2, ipady=3, expand=True)
        def on_change(ev=None):
            raw_val = e.get()
            final_val = validation_func(raw_val) if validation_func else raw_val
            self._save(key, final_val)
        e.bind("<FocusOut>", on_change)
        e.bind("<Return>", lambda ev: [on_change(), self.canvas.focus_set()])

    def _compact_input(self, parent, label, key, val, validation_func=None):
        tk.Label(parent, text=label, bg=parent.cget('bg'), fg=COLORS['fg_text'], font=FONTS['small']).pack(side='left', padx=(5,2))
        e = tk.Entry(parent, bg=COLORS['input_bg'], fg='white', bd=0, insertbackground='white', width=6)
        e.insert(0, str(val)); e.pack(side='left', padx=2)
        def on_change(ev=None):
            raw_val = e.get()
            final_val = validation_func(raw_val) if validation_func else raw_val
            self._save(key, final_val)
        e.bind("<FocusOut>", on_change)
        e.bind("<Return>", lambda ev: [on_change(), self.canvas.focus_set()])

    def _combo(self, parent, label, key, values, val, cmd):
        f = tk.Frame(parent, bg=parent.cget('bg')); f.pack(fill='x', pady=2)
        tk.Label(f, text=label, bg=parent.cget('bg'), fg=COLORS['fg_text'], font=FONTS['small']).pack(side='left', padx=(0,5))
        cb = ttk.Combobox(f, values=values, state='readonly', font=FONTS['code']); cb.set(val)
        cb.pack(fill='x', pady=2, expand=True); cb.bind("<<ComboboxSelected>>", cmd)

    def _btn(self, parent, txt, cmd, bg=None):
        b = tk.Button(parent, text=txt, command=cmd, bg=bg or COLORS['btn_bg'], fg='white', bd=0, activebackground=COLORS['btn_hover'], relief='flat', pady=2, font=FONTS['small'])
        b.pack(fill='x', pady=3, ipady=1)
        return b
    
    def _btn_icon(self, parent, txt, cmd, bg=None, color=None, width=None):
        b = tk.Button(parent, text=txt, command=cmd, bg=bg or COLORS['bg_card'], fg=color or 'white', bd=0, activebackground=COLORS['btn_hover'], relief='flat', width=width)
        b.pack(side='right', padx=2)
        return b

    def _chk(self, parent, txt, key, val):
        var = tk.BooleanVar(value=val)
        tk.Checkbutton(parent, text=txt, variable=var, bg=parent.cget('bg'), fg='white', selectcolor=COLORS['bg_app'], activebackground=parent.cget('bg'), borderwidth=0, highlightthickness=0, command=lambda: [self._save(key, var.get()), self.load_node(self.current_node)]).pack(anchor='w', pady=2)

    def _save(self, key, val):
        if self.current_node:
            self.current_node.update_data(key, val)

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
            self._save('relative_click_pos', (rx, ry))
            self.load_node(self.current_node) 
            
        c.bind("<Button-1>", on_click)
        rx, ry = data.get('relative_click_pos', (0.5, 0.5))
        cx, cy = off_x + (rx * dw), off_y + (ry * dh)
        c.create_oval(cx-3, cy-3, cx+3, cy+3, fill=COLORS['marker'], outline='white', width=1)

    def open_visual_offset_picker(self):
        self.app.iconify(); time.sleep(0.3); full_screen = ImageGrab.grab()
        try:
            res = VisionEngine.locate(self.current_node.data.get('image'), confidence=0.8, timeout=1.0)
            if not res: self.app.deiconify(); messagebox.showerror("错误", "未在屏幕找到基准图"); return
            
            top = tk.Toplevel(self.app); top.attributes("-fullscreen", True, "-topmost", True)
            top.config(cursor="crosshair")
            cv = tk.Canvas(top, width=full_screen.width, height=full_screen.height); cv.pack()
            tk_img = ImageTk.PhotoImage(full_screen)
            cv.create_image(0,0,image=tk_img,anchor='nw')
            cv.create_rectangle(res.left, res.top, res.left+res.width, res.top+res.height, outline='green', width=2)
            
            cx, cy = res.left+res.width/2, res.top+res.height/2
            
            cv.create_line(cx-10, cy, cx+10, cy, fill='red', width=2)
            cv.create_line(cx, cy-10, cx, cy+10, fill='red', width=2)
            
            line_id = cv.create_line(cx, cy, cx, cy, fill='blue', dash=(4, 4), width=1)
            text_id = cv.create_text(cx, cy, text="Offset: 0, 0", fill='blue', anchor='sw', font=('Consolas', 10, 'bold'))
            
            def on_motion(e):
                cv.coords(line_id, cx, cy, e.x, e.y)
                cv.coords(text_id, e.x + 10, e.y - 10)
                cv.itemconfig(text_id, text=f"Offset: {int(e.x-cx)}, {int(e.y-cy)}")

            def confirm(e): 
                self._save('offset_x', int(e.x-cx)); self._save('offset_y', int(e.y-cy))
                cv.create_oval(e.x-3, e.y-3, e.x+3, e.y+3, fill='red', outline='white')
                top.update()
                time.sleep(0.2)
                top.destroy(); self.app.deiconify(); self.load_node(self.current_node)
            
            cv.bind("<Motion>", on_motion)
            cv.bind("<Button-1>", confirm)
            cv.bind("<Button-3>", lambda e: [top.destroy(), self.app.deiconify()])
            top.img_ref = tk_img 
            self.wait_window(top)
        except Exception as e: 
            self.app.deiconify(); traceback.print_exc()

    def _delete_anchor(self, idx):
        anchors = self.current_node.data.get('anchors', [])
        if 0 <= idx < len(anchors): 
            del anchors[idx]
            self._save('anchors', anchors); self.load_node(self.current_node)

    def _delete_image_condition(self, iid):
        images = self.current_node.data.get('images', [])
        self.current_node.data['images'] = [i for i in images if i.get('id') != iid]
        self.load_node(self.current_node)

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
                         if not VisionEngine._advanced_match(img.get('image'), screen, 0.8, None, True, True, 1.0, 'hybrid')[0]: 
                            passed = False; break
                     res_txt = "✅ 全部满足" if passed else "❌ 条件不满足"
            else:
                 res = VisionEngine.locate(self.current_node.data.get('image'), confidence=0.8)
                 res_txt = "✅ 找到" if res else "❌ 未找到"
        except: pass
        self.app.deiconify(); self.test_result_label.config(text=res_txt)

    def _toggle_static_monitor(self):
        if self.static_monitor_active:
            self.static_monitor_active = False
            self.btn_monitor.config(text="🔴 启动监控", bg=COLORS['btn_bg'])
            self.lbl_monitor_status.config(text="监控已停止")
        else:
            if not self.current_node.data.get('roi'): messagebox.showwarning("提示", "请先截取监控区域！"); return
            self.static_monitor_active = True
            self.btn_monitor.config(text="⏹ 停止监控", bg=COLORS['danger'])
            threading.Thread(target=self._static_monitor_thread, daemon=True).start()

    def _static_monitor_thread(self):
        roi = self.current_node.data.get('roi')
        thr = safe_float(self.current_node.data.get('threshold', 0.98))
        dur = safe_float(self.current_node.data.get('duration', 5.0))
        last_frame = VisionEngine.capture_screen(bbox=roi)
        static_start = time.time()
        
        while self.static_monitor_active and self.current_node and self.current_node.type == 'if_static':
            curr = VisionEngine.capture_screen(bbox=roi)
            is_static = VisionEngine.compare_images(last_frame, curr, thr)
            elapsed = time.time() - static_start if is_static else 0
            
            if self.lbl_monitor_status.winfo_exists():
                txt = f"{'🟢 静止' if is_static else '🌊 运动'} | {elapsed:.1f}s / {dur}s"
                color = COLORS['success'] if elapsed >= dur else (COLORS['fg_text'] if is_static else COLORS['warning'])
                self.lbl_monitor_status.config(text=txt, fg=color)
            
            if not is_static: static_start = time.time(); last_frame = curr
            time.sleep(0.1)
        self.static_monitor_active = False

    def _toggle_audio_monitor(self):
        self.is_monitoring_audio = not self.is_monitoring_audio
        self.monitor_audio_btn.config(text="⏹ 停止" if self.is_monitoring_audio else "🔊 实时监测")
        if self.is_monitoring_audio: threading.Thread(target=self._audio_monitor_thread, daemon=True).start()

    def _audio_monitor_thread(self):
        while self.is_monitoring_audio and self.winfo_exists():
            vol = AudioEngine.get_max_audio_peak()
            if vol > 0.001: self.app.log(f"📊 音量峰值: {vol:.4f}", "info")
            time.sleep(0.5)

# --- 7. 设置对话框 ---
class SettingsDialog(tk.Toplevel):
    def __init__(self, parent, app):
        super().__init__(parent); self.app = app
        self.title("设置"); self.geometry("400x300"); self.config(bg=COLORS['bg_panel'])
        self.resizable(False, False); self.transient(parent); self.grab_set()
        
        self.app.stop_hotkeys()
        self.protocol("WM_DELETE_WINDOW", self.on_cancel)
        
        self.geometry("+%d+%d" % (parent.winfo_rootx()+50, parent.winfo_rooty()+50))

        f_theme = tk.Frame(self, bg=COLORS['bg_panel'], pady=10, padx=20)
        f_theme.pack(fill='x')
        tk.Label(f_theme, text="界面主题:", bg=COLORS['bg_panel'], fg=COLORS['fg_text']).pack(side='left')
        self.combo_theme = ttk.Combobox(f_theme, values=list(THEMES.keys()), state='readonly')
        self.combo_theme.set(SETTINGS.get('theme', 'Dark'))
        self.combo_theme.pack(side='right', fill='x', expand=True, padx=10)

        f_hk = tk.Frame(self, bg=COLORS['bg_panel'], pady=10, padx=20)
        f_hk.pack(fill='x')
        
        self.hk_vars = {
            'start': tk.StringVar(value=SETTINGS.get('hotkey_start', '<alt>+1')),
            'stop': tk.StringVar(value=SETTINGS.get('hotkey_stop', '<alt>+2'))
        }

        self._create_hotkey_entry(f_hk, "启动快捷键:", 'start', 0)
        self._create_hotkey_entry(f_hk, "停止快捷键:", 'stop', 1)

        f_hk.columnconfigure(1, weight=1)

        btn_frame = tk.Frame(self, bg=COLORS['bg_panel'], pady=20)
        btn_frame.pack(side='bottom', fill='x')
        tk.Button(btn_frame, text="保存并重启UI", command=self.save, bg=COLORS['accent'], fg='white', bd=0, padx=20).pack(side='right', padx=20)
        tk.Button(btn_frame, text="取消", command=self.on_cancel, bg=COLORS['btn_bg'], fg='white', bd=0, padx=20).pack(side='right')

    def _create_hotkey_entry(self, parent, label, key, row):
        tk.Label(parent, text=label, bg=COLORS['bg_panel'], fg=COLORS['fg_text']).grid(row=row, column=0, sticky='w', pady=5)
        e = tk.Entry(parent, textvariable=self.hk_vars[key], bg=COLORS['input_bg'], fg='white', insertbackground='white', readonlybackground=COLORS['input_bg'])
        e.grid(row=row, column=1, sticky='ew', padx=10)
        
        e.bind("<FocusIn>", lambda ev: self._on_focus(e, key))
        e.bind("<KeyPress>", lambda ev: self._on_key(ev, key))
        e.bind("<Button-3>", lambda ev: self._clear_hotkey(key)) 

    def _on_focus(self, entry, key):
        entry.config(state='normal', bg=COLORS['accent']) 
        
    def _clear_hotkey(self, key):
        self.hk_vars[key].set("")
        
    def _on_key(self, event, key):
        if event.keysym in ['Shift_L', 'Shift_R', 'Control_L', 'Control_R', 'Alt_L', 'Alt_R']: return 
        
        if event.keysym == 'Escape':
            self.hk_vars[key].set("")
            self.focus_set() 
            return "break"

        parts = []
        if event.state & 0x0004: parts.append("<ctrl>")
        if event.state & 0x20000 or event.state & 0x0008: parts.append("<alt>") 
        if event.state & 0x0001: parts.append("<shift>")
        
        char = event.keysym.lower()
        if char not in ['control_l', 'control_r', 'alt_l', 'alt_r', 'shift_l', 'shift_r']:
             parts.append(char)
        
        hotkey_str = "+".join(parts)
        self.hk_vars[key].set(hotkey_str)
        return "break" 
    
    def on_cancel(self):
        self.app.refresh_hotkeys()
        self.destroy()

    def save(self):
        SETTINGS['theme'] = self.combo_theme.get()
        SETTINGS['hotkey_start'] = self.hk_vars['start'].get()
        SETTINGS['hotkey_stop'] = self.hk_vars['stop'].get()
        
        new_theme = THEMES.get(SETTINGS['theme'], THEMES['Dark'])
        COLORS.update(new_theme)
        
        self.app.refresh_hotkeys()
        self.app.restart_ui()
        self.destroy()

# --- 8. 主程序 ---
class App(tk.Tk):
    def __init__(self):
        super().__init__(); self.title("Qflow 1.5 —— QwejayHuang"); self.geometry("1400x900")
        self.core = AutomationCore(self.log, self); self.log_q = queue.Queue()
        self.drag_node_type, self.drag_ghost = None, None
        self.hotkey_listener = None
        
        self._setup_ui()
        self.refresh_hotkeys()
        self.after(100, self._poll_log)

    def _setup_ui(self):
        self.configure(bg=COLORS['bg_app'])
        
        for widget in self.winfo_children(): widget.destroy()

        title_bar = tk.Frame(self, bg=COLORS['bg_app'], height=50); title_bar.pack(fill='x', pady=5, padx=20)
        tk.Label(title_bar, text="QFLOW 1.5", font=('Impact', 24), bg=COLORS['bg_app'], fg=COLORS['accent']).pack(side='left', padx=(0, 20))
        
        ops = tk.Frame(title_bar, bg=COLORS['bg_app']); ops.pack(side='left')
        for txt, cmd in [("📂 打开", self.load), ("💾 保存", self.save), ("🗑️ 清空", self.clear), ("⚙️ 设置", self.open_settings)]:
            tk.Button(ops, text=txt, command=cmd, bg=COLORS['bg_header'], fg='white', bd=0, padx=10).pack(side='left', padx=2)
            
        self.btn_run = tk.Button(title_bar, text="▶ 启动", command=lambda: self.toggle_run(None), bg=COLORS['success'], fg='#1f1f1f', font=('Segoe UI', 11, 'bold'), padx=15, bd=0)
        self.btn_run.pack(side='right')
        
        self.btn_pause = tk.Button(title_bar, text="⏸ 暂停", command=self.toggle_pause, bg=COLORS['warning'], fg='#1f1f1f', bd=0, padx=10, state='disabled')
        self.btn_pause.pack(side='right', padx=10)
        
        paned=tk.PanedWindow(self,orient='horizontal',bg=COLORS['bg_app'],sashwidth=4,bd=0);paned.pack(fill='both',expand=True,padx=10,pady=(0,5))
        
        toolbox=tk.Frame(paned,bg=COLORS['bg_sidebar']); self._build_toolbox(toolbox); paned.add(toolbox,minsize=160)
        self.editor=FlowEditor(paned,self); paned.add(self.editor,minsize=400,stretch="always")
        # [# 修改属性面板宽度]
        self.property_panel=PropertyPanel(paned,self); paned.add(self.property_panel,minsize=250)
        
        self.log_panel=LogPanel(self)
        self.editor.add_node('start',100,100, save_history=False)

    def restart_ui(self):
        data = self.editor.get_data()
        self._setup_ui()
        self.editor.load_data(data)

    def open_settings(self): SettingsDialog(self, self)

    def refresh_hotkeys(self):
        if self.hotkey_listener: self.hotkey_listener.stop()
        self.hotkey_listener = keyboard.GlobalHotKeys({
            SETTINGS['hotkey_start']: self.on_hotkey_start,
            SETTINGS['hotkey_stop']: self.on_hotkey_stop
        })
        self.hotkey_listener.start()
        
    def stop_hotkeys(self):
        if self.hotkey_listener:
            self.hotkey_listener.stop()
            self.hotkey_listener = None

    def on_hotkey_start(self):
        if not self.core.running:
            self.log("⌨️ 快捷键启动", "info")
            self.after(0, lambda: self.toggle_run(None))

    def on_hotkey_stop(self):
        if self.core.running:
            self.log("⌨️ 快捷键停止", "warning")
            self.core.stop()

    def _build_toolbox(self, p):
        tool_groups = [
            ("逻辑组件", ['start', 'end', 'loop', 'sequence', 'set_var', 'var_switch']),
            ("动作执行", ['mouse', 'keyboard', 'cmd', 'web', 'wait']),
            ("视觉/感知", ['image', 'if_img', 'if_static', 'if_sound'])
        ]
        for title, items in tool_groups:
            tk.Label(p, text=title, bg=COLORS['bg_sidebar'], fg=COLORS['fg_sub'], font=('Segoe UI', 8, 'bold'), pady=8).pack(anchor='w', padx=10)
            for t in items:
                if t not in NODE_CONFIG: continue
                f = tk.Frame(p, bg=COLORS['bg_card'], cursor="hand2", pady=2); f.pack(fill='x', pady=1, padx=8)
                tk.Frame(f, bg=NODE_CONFIG[t]['color'], width=4).pack(side='left', fill='y')
                l = tk.Label(f, text=NODE_CONFIG[t]['title'], bg=COLORS['bg_card'], fg=COLORS['fg_text'], anchor='w', padx=8, pady=6)
                l.pack(side='left', fill='both', expand=True)
                for w in [f, l]: 
                    w.bind("<ButtonPress-1>", lambda e, t=t: self.on_drag_start(e, t))
                    w.bind("<B1-Motion>", self.on_drag_move); w.bind("<ButtonRelease-1>", self.on_drag_end)

    def on_drag_start(self,e,t): self.drag_node_type=t; self.drag_ghost=tk.Toplevel(self); self.drag_ghost.overrideredirect(True); self.drag_ghost.attributes("-alpha",0.7); tk.Label(self.drag_ghost,text=NODE_CONFIG[t]['title'],bg=COLORS['accent']).pack()
    def on_drag_move(self,e): (self.drag_ghost and self.drag_ghost.geometry(f"+{e.x_root+10}+{e.y_root+10}"))
    def on_drag_end(self,e):
        if self.drag_ghost: self.drag_ghost.destroy(); self.drag_ghost=None
        if self.editor.winfo_containing(e.x_root, e.y_root) == self.editor:
            self.editor.add_node(self.drag_node_type, self.editor.canvasx(e.x_root-self.editor.winfo_rootx())/self.editor.zoom, self.editor.canvasy(e.y_root-self.editor.winfo_rooty())/self.editor.zoom)

    # --- 截图功能 ---

    def do_snip(self):
        self.iconify()
        self.update() 
        self.after(400, lambda: self._start_snip_overlay())

    def _start_snip_overlay(self):
        top = tk.Toplevel(self)
        top.attributes("-fullscreen", True, "-alpha", 0.3, "-topmost", True)
        top.configure(cursor="cross", bg="black")
        
        c = tk.Canvas(top, bg="black", highlightthickness=0)
        c.pack(fill='both', expand=True)

        info_lbl = tk.Label(top, text="请框选区域 (ESC取消)", font=('Segoe UI', 16, 'bold'), fg='white', bg='black')
        info_lbl.place(x=50, y=50)

        s, r = [0, 0], [None]

        def dn(e): 
            s[0], s[1] = e.x, e.y
            if r[0]: c.delete(r[0])
            r[0] = c.create_rectangle(e.x, e.y, e.x, e.y, outline='red', width=2)

        def mv(e): 
            if r[0]: c.coords(r[0], s[0], s[1], e.x, e.y)

        def up(e): 
            x1, y1, x2, y2 = min(s[0], e.x), min(s[1], e.y), max(s[0], e.x), max(s[1], e.y)
            top.destroy()
            self.after(200, lambda: self._capture((x1, y1, x2, y2)))

        def cancel(e):
            top.destroy()
            self.deiconify()

        c.bind("<ButtonPress-1>", dn)
        c.bind("<B1-Motion>", mv)
        c.bind("<ButtonRelease-1>", up)
        c.bind("<Button-3>", cancel)
        top.bind("<Escape>", cancel)

    def _capture(self, rect):
        x1, y1, x2, y2 = rect
        if x2 - x1 < 5 or y2 - y1 < 5: 
            self.deiconify()
            return
            
        try:
            img = ImageGrab.grab(bbox=(x1, y1, x2, y2))
            
            self.deiconify()
            
            if (n := self.property_panel.current_node): 
                n.update_data('_dummy_for_history', time.time())
                
                if n.type == 'if_img': 
                    n.data.setdefault('images', []).append({
                        'id': uuid.uuid4().hex, 
                        'image': img, 
                        'tk_image': ImageUtils.make_thumb(img), 
                        'b64': ImageUtils.img_to_b64(img)
                    })
                    self.property_panel.load_node(n)
                    
                elif n.type == 'if_static': 
                    n.update_data('roi', (x1, y1, x2-x1, y2-y1))
                    n.data['roi_preview'] = ImageUtils.make_thumb(img)
                    n.data['b64_preview'] = ImageUtils.img_to_b64(img)
                    n.draw()
                    self.property_panel.load_node(n)
                    
                else: 
                    n.update_data('image', img)
                    n.update_data('tk_image', ImageUtils.make_thumb(img))
                    n.update_data('b64', ImageUtils.img_to_b64(img))
                    n.draw()
                    self.property_panel.load_node(n)
                    
            self.log(f"🖼️ 截取成功 ({x2-x1}x{y2-y1})", "success")
            
        except Exception as e:
            self.deiconify()
            self.log(f"截图失败: {e}", "error")
    
    def pick_coordinate(self): self.iconify(); self.after(500, lambda: self._coord_overlay())
    def _coord_overlay(self):
        top=tk.Toplevel(self);top.attributes("-fullscreen",True,"-alpha",0.1,"-topmost",True);c=tk.Canvas(top,bg="white");c.pack(fill='both',expand=True)
        def clk(e): top.destroy(); self.deiconify(); (self.property_panel.current_node and (self.property_panel.current_node.update_data('x',e.x_root) or self.property_panel.current_node.update_data('y',e.y_root) or self.property_panel.load_node(self.property_panel.current_node)))
        c.bind("<Button-1>",clk)

    def toggle_run(self, start_id): 
        if self.core.running: 
            self.core.stop()
        else: 
            self.btn_run.config(text="⏹ 停止", bg=COLORS['danger'])
            self.btn_pause.config(state='normal', text="⏸ 暂停", bg=COLORS['warning'])
            self.core.load_project(self.editor.get_data())
            self.core.start(start_id)

    def toggle_pause(self): 
        if self.core.paused:
            self.core.resume()
        else:
            self.core.pause()
    
    def update_debug_btn_state(self, paused): 
        if paused:
            self.btn_pause.config(text="▶ 继续", bg=COLORS['success'])
        else:
            self.btn_pause.config(text="⏸ 暂停", bg=COLORS['warning'])
            
    def reset_ui_state(self): 
        self.core.running=False
        self.btn_run.config(text="▶ 启动", bg=COLORS['success'])
        self.btn_pause.config(text="⏸ 暂停", bg=COLORS['warning'], state='disabled')
        [self.highlight_node_safe(n, None) for n in self.editor.nodes]
    
    def log(self,msg, level='info'): self.log_q.put((msg, level))
    def _poll_log(self):
        while not self.log_q.empty(): item = self.log_q.get(); self.log_panel.add_log(item[0], item[1])
        self.after(100,self._poll_log)
    def highlight_node_safe(self, nid, status=None):
        def _task():
            if not self.editor.winfo_exists(): return
            self.editor.delete("hl")
            if nid and nid in self.editor.nodes and status:
                n = self.editor.nodes[nid]; z = self.editor.zoom; color = COLORS.get(f"hl_{status}", COLORS['hl_ok'])
                self.editor.create_rectangle(n.x * z - 3 * z, n.y * z - 3 * z, (n.x + n.w) * z + 3 * z, (n.y + n.h) * z + 3 * z, outline=color, width=3 * z, tags="hl")
        self.after(0, _task)
    def select_node_safe(self, nid): self.after(0, lambda: self.editor.select_node(nid))
    def save(self):
        if (f:=filedialog.asksaveasfilename(defaultextension=".qflow")): 
            with open(f, 'w', encoding='utf-8') as fp: json.dump(self.editor.get_data(), fp, ensure_ascii=False, indent=2)
    def load(self):
        if (f:=filedialog.askopenfilename()): 
            with open(f, 'r', encoding='utf-8') as fp: self.editor.load_data(json.load(fp))
    def clear(self): self.editor.load_data({'nodes':{},'links':[]})

if __name__ == "__main__": App().mainloop()