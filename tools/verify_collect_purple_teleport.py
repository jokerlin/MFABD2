"""地图采集「紫色传送阵」误判与次选死循环回归：真实资源与内核，内存截图，不连接游戏。

修复前 Collect_ClickTelepor_TargetClass_Purple 的 count=1：普通白金传送阵边缘只要有 1 个像素
落进紫色范围就判为紫色，转去 PurpleProcess 点第 2 个图标。次选点了不传送时，PurpleProcess 的
next 里有它自己，又没有 max_hit，任务永不结束（2026-10-12、10-19 实机卡在血骑士 tp2）。

夹具 area_map_stray_pixel.png 取自 2026-10-18 实机误判截图（施塔因之塔第 1 层），只保留传送阵
图标与右换页箭头两块，其余填平色；图标框内 (622,332) 是那个落进紫色范围的像素。

运行：python -B tools/verify_collect_purple_teleport.py -v
"""

from pathlib import Path
import tempfile
import threading
import unittest

import numpy as np
from PIL import Image
from maa.controller import CustomController
from maa.define import LoggingLevelEnum
from maa.library import Library
from maa.resource import Resource
from maa.tasker import Tasker

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).resolve().parent / "fixtures/collect_purple_teleport/area_map_stray_pixel.png"
Library.version()

ENTRY = "Collect_ClickTeleporCircleIco"
PURPLE = "Collect_ClickTelepor_TargetClass_Purple"
PURPLE_PROCESS = "Collect_ClickTelepor_TargetClass_PurpleProcess"
WHITE = "Collect_ClickTelepor_TargetClass_White"
SKILL_ENTRY = "Collect_Skill_Entry"
PURPLE_PROCESS_LIMIT = 5  # PurpleProcess.max_hit
ICON = [601, 318, 29, 28]
SECOND_ICON_DX = 200
PURPLE_PATCH = [610, 325, 10, 10]  # 10x10 紫块，代表真正的紫色传送阵
ARRIVED_MARK = [1117, 639, 53, 51]  # Rec_SandBox_Pin_Tpl 的 roi，到达箱庭后白块出现
WATCHDOG_S = 60


def load_frame():
    return np.ascontiguousarray(np.asarray(Image.open(FIXTURE).convert("RGB"))[:, :, ::-1])


def paint_purple(frame):
    x, y, w, h = PURPLE_PATCH
    frame[y:y + h, x:x + w] = (210, 115, 165)  # BGR，落在 lower/upper 之间
    return frame


def add_second_icon(frame):
    # 原样复制首选图标到右侧，作为「点了不传送」的次选
    x, y, w, h = ICON
    patch = frame[y - 10:y + h + 10, x - 10:x + w + 10].copy()
    frame[y - 10:y + h + 10, x - 10 + SECOND_ICON_DX:x + w + 10 + SECOND_ICON_DX] = patch
    return frame


class AreaMapController(CustomController):
    """区域地图画面：只有点中首选图标才会传送，次选怎么点画面都不变。"""

    def __init__(self, frame):
        super().__init__()
        self.frame = frame
        self.arrived = np.full_like(frame, 60)
        x, y, w, h = ARRIVED_MARK
        self.arrived[y:y + h, x:x + w] = 255
        self.clicks = []

    def connect(self): return True
    def request_uuid(self): return "collect-purple-teleport-offline"
    def get_features(self): return 0
    def start_app(self, *args): return True
    def stop_app(self, *args): return True
    def swipe(self, *args): return True
    def touch_down(self, *args): return True
    def touch_move(self, *args): return True
    def touch_up(self, *args): return True
    def click_key(self, *args): return True
    def input_text(self, *args): return True
    def key_down(self, *args): return True
    def key_up(self, *args): return True

    def click(self, x, y):
        self.clicks.append((x, y))
        return True

    def screencap(self):
        x, y, w, h = ICON
        if any(x <= cx < x + w and y <= cy < y + h for cx, cy in self.clicks):
            return self.arrived
        return self.frame


class PurpleTeleportTests(unittest.TestCase):
    def test_stray_pixel_is_not_purple(self):
        controller, names, timed_out = self.run_entry(load_frame())
        self.assertFalse(timed_out)
        self.assertNotIn(PURPLE, names)
        self.assertIn(WHITE, names)
        self.assertEqual(names[-1], SKILL_ENTRY)

    def test_purple_icon_still_detected(self):
        # 负向断言的配对正例：真紫色仍要走规避分支
        controller, names, timed_out = self.run_entry(paint_purple(load_frame()))
        self.assertFalse(timed_out)
        self.assertIn(PURPLE, names)

    def test_unresponsive_second_icon_falls_back_to_first(self):
        controller, names, timed_out = self.run_entry(add_second_icon(paint_purple(load_frame())))
        self.assertFalse(timed_out)
        self.assertEqual(names.count(PURPLE_PROCESS), PURPLE_PROCESS_LIMIT)
        self.assertIn(WHITE, names)
        self.assertEqual(names[-1], SKILL_ENTRY)
        x, y, w, h = ICON
        self.assertTrue(all(cx >= x + SECOND_ICON_DX for cx, _ in controller.clicks[:PURPLE_PROCESS_LIMIT]))

    def run_entry(self, frame):
        fixture = {
            PURPLE_PROCESS: {"post_delay": 0, "rate_limit": 0, "timeout": 500},
            WHITE: {"post_delay": 0, "rate_limit": 0},
            # 内存截图没有箱庭 UI，用右下角白块代表「已到达」
            SKILL_ENTRY: {"recognition": "ColorMatch", "roi": ARRIVED_MARK, "lower": [250, 250, 250],
                          "upper": [255, 255, 255], "count": ARRIVED_MARK[2] * ARRIVED_MARK[3], "next": []},
        }
        # MaaFramework keeps its log handle open until process exit on Windows.
        with tempfile.TemporaryDirectory(prefix="mfabd2-purple-", ignore_cleanup_errors=True) as directory:
            Tasker.set_log_dir(directory)
            Tasker.set_stdout_level(LoggingLevelEnum.Off)
            Tasker.set_save_draw(False)
            Tasker.set_save_on_error(False)
            resource = Resource()
            controller = AreaMapController(frame)
            tasker = Tasker()
            # 修复前次选死循环，看门狗停掉后交给断言报错
            fired = threading.Event()
            watchdog = threading.Timer(WATCHDOG_S, lambda: (fired.set(), tasker.post_stop()))
            try:
                self.assertTrue(resource.post_bundle(ROOT / "assets/resource/base").wait().succeeded)
                self.assertTrue(resource.override_pipeline(fixture))
                self.assertTrue(controller.post_connection().wait().succeeded)
                self.assertTrue(tasker.bind(resource, controller))
                watchdog.start()
                detail = tasker.post_task(ENTRY).wait().get()
                return controller, [node.name for node in detail.nodes if node], fired.is_set()
            finally:
                watchdog.cancel()
                tasker.post_stop().wait()
                del tasker
                resource.clear()
                Tasker.set_log_dir("")


if __name__ == "__main__":
    unittest.main()
