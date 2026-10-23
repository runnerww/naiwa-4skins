"""A bounded, smoothed pupil layer in the original idle-frame coordinates."""
from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from pet_paths import ROOT

ASSETS = ROOT / "assets" / "eyes"


class EyeFollow:
    INTERVAL = 1 / 30
    SMOOTH_SEC = 0.120
    LIMIT = (2.0, 1.5)

    def __init__(self, size, assets=ASSETS):
        metadata = json.loads((assets / "geometry.json").read_text(encoding="utf-8"))
        self.reference = tuple(metadata["referenceSize"])
        self.center = tuple(metadata["pupilCenter"])
        self.pupil_size = tuple(metadata["pupilSize"])
        self.eye_box = tuple(metadata["eyeBox"])
        self.backing = self._read(assets / "backing.png")
        self.pupil = self._read(assets / "pupil.png")
        self.size = tuple(size)
        self.position = (0.0, 0.0)
        self.last_update = None
        self._prepare()

    @staticmethod
    def _read(path):
        with Image.open(path) as image:
            rgba = image.convert("RGBA")
        box = rgba.getchannel("A").point(lambda a: 255 if a > 128 else 0).getbbox()
        if not box:
            raise ValueError(f"Empty eye asset: {path}")
        return rgba.crop(box)

    def _prepare(self):
        self.sx = self.size[0] / self.reference[0]
        self.sy = self.size[1] / self.reference[1]
        # Supersampling preserves subpixel movement even at the smallest pet size.
        self.resolution = 8
        self.roi = (math.floor(135 * self.sx), math.floor(24 * self.sy),
                    math.ceil(152 * self.sx), math.ceil(42 * self.sy))
        left, top, right, bottom = self.roi
        self.patch_size = ((right - left) * self.resolution, (bottom - top) * self.resolution)
        self.static = Image.new("RGBA", self.patch_size)
        self.mask = Image.new("L", self.patch_size)
        box = tuple((v * (self.sx if i % 2 == 0 else self.sy) - (left if i % 2 == 0 else top))
                    * self.resolution for i, v in enumerate(self.eye_box))
        ImageDraw.Draw(self.mask).ellipse(box, fill=255)
        x0, y0, x1, y1 = map(round, box)
        base = self.backing.resize((max(1, x1 - x0), max(1, y1 - y0)), Image.Resampling.LANCZOS)
        # The generated alpha only defines the texture; the calibrated mask keeps the original rim fixed.
        base.putalpha(255)
        self.static.paste(base, (x0, y0))
        self.pupil_sprite = self.pupil.resize(
            (max(1, round(self.pupil_size[0] * self.sx * self.resolution)),
             max(1, round(self.pupil_size[1] * self.sy * self.resolution))), Image.Resampling.LANCZOS)

    def reset(self):
        self.position = (0.0, 0.0)
        self.last_update = None

    def advance(self, point, now):
        """point is in untransformed, displayed content coordinates, not screen pixels."""
        dx = point[0] / self.sx - self.center[0]
        dy = point[1] / self.sy - self.center[1]
        distance = math.hypot(dx, dy)
        factor = math.tanh(distance / 200.0) / distance if distance > 1e-9 else 0.0
        target = (self.LIMIT[0] * dx * factor, self.LIMIT[1] * dy * factor)
        dt = self.INTERVAL if self.last_update is None else max(0, min(0.25, now - self.last_update))
        self.last_update = now
        blend = 1 - math.exp(-dt / self.SMOOTH_SEC)
        before = self.position
        self.position = tuple(b if abs(b - a) < 0.005 else a + (b - a) * blend
                              for a, b in zip(before, target))
        return max(abs(a - b) for a, b in zip(before, self.position)) > 1e-6

    def render_rgba(self, rgba):
        patch = self.static.copy()
        x = ((self.center[0] + self.position[0]) * self.sx - self.roi[0]) * self.resolution
        y = ((self.center[1] + self.position[1]) * self.sy - self.roi[1]) * self.resolution
        pupil = self.pupil_sprite.transform(
            self.patch_size, Image.Transform.AFFINE,
            (1, 0, self.pupil_sprite.width / 2 - x, 0, 1, self.pupil_sprite.height / 2 - y),
            Image.Resampling.BICUBIC)
        patch.alpha_composite(pupil)
        patch.putalpha(self.mask)
        patch = patch.resize((self.roi[2] - self.roi[0], self.roi[3] - self.roi[1]), Image.Resampling.LANCZOS)
        result = rgba.copy()
        result.alpha_composite(patch, (self.roi[0], self.roi[1]))
        result.putalpha(rgba.getchannel("A"))
        return result


class EyeShift:
    """整块眼睛贴片的轻微平移，供没有独立瞳孔素材的新皮肤使用。"""

    INTERVAL = 1 / 30
    SMOOTH_SEC = 0.120
    REFERENCE = (320, 305)

    def __init__(self, size, box, limit=1.8):
        self.size = tuple(size)
        self.sx = self.size[0] / self.REFERENCE[0]
        self.sy = self.size[1] / self.REFERENCE[1]
        self.roi = (round(box[0] * self.sx), round(box[1] * self.sy),
                    round((box[0] + box[2]) * self.sx), round((box[1] + box[3]) * self.sy))
        self.limit = (limit, max(1.0, limit * 0.7))
        self.center = ((self.roi[0] + self.roi[2]) / 2, (self.roi[1] + self.roi[3]) / 2)
        self.position = (0.0, 0.0)
        self.last_update = None
        self._mask = self._build_mask()

    def _build_mask(self):
        w, h = self.roi[2] - self.roi[0], self.roi[3] - self.roi[1]
        scale = 4
        mask = Image.new("L", (w * scale, h * scale))
        ImageDraw.Draw(mask).ellipse((0, 0, w * scale - 1, h * scale - 1), fill=255)
        mask = mask.resize((w, h), Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(max(1, min(w, h) / 12)))
        return mask

    def reset(self):
        self.position = (0.0, 0.0)
        self.last_update = None

    def advance(self, point, now):
        dx = point[0] - self.center[0]
        dy = point[1] - self.center[1]
        distance = math.hypot(dx, dy)
        factor = math.tanh(distance / max(40.0, self.size[0] * 0.6)) / distance if distance > 1e-9 else 0.0
        target = (self.limit[0] * dx * factor, self.limit[1] * dy * factor)
        dt = self.INTERVAL if self.last_update is None else max(0, min(0.25, now - self.last_update))
        self.last_update = now
        blend = 1 - math.exp(-dt / self.SMOOTH_SEC)
        before = self.position
        self.position = tuple(b if abs(b - a) < 0.005 else a + (b - a) * blend
                              for a, b in zip(before, target))
        return max(abs(a - b) for a, b in zip(before, self.position)) > 1e-6

    def render_rgba(self, rgba):
        patch = rgba.crop(self.roi)
        patch.putalpha(self._mask)
        result = rgba.copy()
        result.alpha_composite(patch, (round(self.roi[0] + self.position[0]),
                                       round(self.roi[1] + self.position[1])))
        result.putalpha(rgba.getchannel("A"))
        return result
