"""시각화 서비스 - BBox, 키포인트, 감정 텍스트 (v3 신규 작성)"""
import cv2
import numpy as np
import yaml
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"


class Visualizer:
    """프레임에 추론 결과를 시각화"""

    def __init__(self, pose_config_path: str):
        self.config = self._load_config(pose_config_path)
        self.kp_colors = self.config.get("keypoint_colors", [])
        self.skeleton_links = self.config.get("skeleton_links", [])
        self.kp_names = self.config.get("keypoints", {})

    @staticmethod
    def _load_config(path: str) -> dict:
        p = Path(path)
        if not p.exists():
            return {}
        with open(p) as f:
            return yaml.safe_load(f) or {}

    def draw(self, frame: np.ndarray, bbox, keypoints, species: str,
             det_conf: float, state) -> np.ndarray:
        """전체 시각화 실행"""
        vis = frame

        # BBox
        if bbox is not None:
            vis = self._draw_bbox(vis, bbox, species, det_conf, state)

        # 키포인트
        if keypoints is not None:
            vis = self._draw_keypoints(vis, keypoints)

        # 감정 텍스트
        if state and state.display_emotion:
            vis = self._draw_emotion(vis, bbox, state)

        # 디버그 정보
        if state and state.last_debug:
            vis = self._draw_debug(vis, bbox, state.last_debug)

        return vis

    def _draw_bbox(self, frame, bbox, species, conf, state):
        x1, y1, x2, y2 = map(int, bbox)

        is_neg = False
        if state and state.display_emotion:
            is_neg = state.display_emotion != "긍정" and state.display_emotion != ""

        color = (0, 0, 255) if is_neg else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        sp_text = f"{species.upper()} {int(conf * 100)}%"
        frame = self._put_korean(frame, sp_text, (x1, y1 - 30), 20, (255, 255, 255), bg=(0, 0, 0, 160))
        return frame

    def _draw_keypoints(self, frame, keypoints):
        for idx, kp in enumerate(keypoints):
            if len(kp) >= 3 and kp[2] < 0.3:
                continue
            x, y = int(kp[0]), int(kp[1])
            if x <= 1 and y <= 1:
                continue
            color = tuple(self.kp_colors[idx]) if idx < len(self.kp_colors) else (0, 255, 255)
            cv2.circle(frame, (x, y), 4, color, -1, cv2.LINE_AA)
        return frame

    def _draw_emotion(self, frame, bbox, state):
        if not bbox is not None:
            return frame
        x1, y1, x2, y2 = map(int, bbox)
        emo = state.display_emotion
        conf = state.display_conf

        text = f"{emo} ({conf * 100:.0f}%)"
        is_neg = emo != "긍정"
        color = (0, 0, 255) if is_neg else (0, 255, 0)

        frame = self._put_korean(frame, text, (x1, y2 + 10), 24, color, bg=(0, 0, 0, 160))
        return frame

    def _draw_debug(self, frame, bbox, debug_text):
        if bbox is None:
            return frame
        x1, _, _, y2 = map(int, bbox)
        frame = self._put_korean(frame, debug_text, (x1, y2 + 40), 16, (200, 200, 200), bg=(0, 0, 0, 128))
        return frame

    @staticmethod
    def _put_korean(img, text, pos, size, color, bg=None):
        """한글 텍스트 렌더링 (PIL 사용)"""
        try:
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil, "RGBA")
            try:
                font = ImageFont.truetype(FONT_PATH, int(size))
            except OSError:
                font = ImageFont.load_default()

            if bg and bg[3] > 0:
                bbox = draw.textbbox((0, 0), text, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                pad = 4
                draw.rectangle(
                    [(pos[0] - pad, pos[1] - pad), (pos[0] + tw + pad, pos[1] + th + pad)],
                    fill=bg,
                )

            draw.text(pos, text, font=font, fill=color)
            return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception:
            return img
