"""
图像预处理模块：自动旋转矫正 → 光照增强 → 透视矫正
在 ingest.py 和 main.py 中统一调用预处理链。

依赖: opencv-python-headless, Pillow, numpy
"""
import os
import numpy as np
from PIL import Image, ImageEnhance, ExifTags

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def auto_rotate(image: Image.Image) -> Image.Image:
    """根据 EXIF 信息自动旋转摆正图片"""
    try:
        exif = image._getexif()
        if exif is None:
            return image

        orientation_key = None
        for k, v in ExifTags.TAGS.items():
            if v == "Orientation":
                orientation_key = k
                break

        if orientation_key is None:
            return image

        orientation = exif.get(orientation_key)
        if orientation == 3:
            image = image.rotate(180, expand=True)
        elif orientation == 6:
            image = image.rotate(270, expand=True)
        elif orientation == 8:
            image = image.rotate(90, expand=True)
    except Exception:
        pass
    return image


def enhance_contrast_clahe(image: Image.Image, clip_limit: float = 2.0) -> Image.Image:
    """CLAHE 光照增强，改善过暗/过亮的面签照片"""
    if not HAS_CV2:
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(1.2)

    img_np = np.array(image.convert("RGB"))
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(result)


def face_detect_crop(image: Image.Image, margin: float = 0.3) -> Image.Image:
    """人脸检测并裁剪面部区域

    OpenCV 5+ 移除了 CascadeClassifier，此步骤仅在 OpenCV 4 上可用。
    """
    if not HAS_CV2:
        return image

    # OpenCV 5+ 不包含 cascade 模块
    cv_version = tuple(int(v) for v in cv2.__version__.split(".")[:2])
    if cv_version >= (5, 0):
        return image

    img_np = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    try:
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if cascade.empty():
            return image
    except Exception:
        return image

    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return image

    # 取最大人脸
    (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])

    h_img, w_img = img_np.shape[:2]
    # 向上扩展包含额头，向下扩展包含上半身
    y1 = max(0, y - int(h * margin))
    y2 = min(h_img, y + h + int(h * (margin * 2)))
    x1 = max(0, x - int(w * margin * 0.5))
    x2 = min(w_img, x + w + int(w * margin * 0.5))

    cropped = img_np[y1:y2, x1:x2]
    return Image.fromarray(cropped)


def detect_document_corners(image: Image.Image) -> Image.Image:
    """文档透视矫正：找最大四边形轮廓并做透视变换"""
    if not HAS_CV2:
        return image

    img_np = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    img_area = image.width * image.height
    if area < img_area * 0.3:
        return image

    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
    if len(approx) != 4:
        return image

    pts = np.float32([p[0] for p in approx])
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    w = max(int(np.linalg.norm(rect[1] - rect[0])), int(np.linalg.norm(rect[2] - rect[3])))
    h = max(int(np.linalg.norm(rect[3] - rect[0])), int(np.linalg.norm(rect[2] - rect[1])))
    if w < 50 or h < 50:
        return image

    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img_np, matrix, (w, h))
    return Image.fromarray(warped)


class PreprocessingPipeline:
    """图像预处理管道，支持配置化组装"""

    def __init__(self, config: dict = None):
        """
        Args:
            config: config.yaml 的 preprocessing section
        """
        self.config = config or {}
        # 注册算子
        self._steps = []

        if self.config.get("auto_rotate", False):
            self._steps.append(("auto_rotate", auto_rotate))

        if self.config.get("enhance_contrast", False):
            clip_limit = self.config.get("clahe_clip_limit", 2.0)
            self._steps.append(("enhance_contrast", lambda img: enhance_contrast_clahe(img, clip_limit)))

        if self.config.get("perspective_correct", False):
            self._steps.append(("correct_perspective", detect_document_corners))

        if self.config.get("face_detect", False):
            margin = self.config.get("face_margin", 0.3)
            self._steps.append(("face_detect", lambda img: face_detect_crop(img, margin)))

    def __call__(self, image: Image.Image) -> Image.Image:
        """执行预处理链"""
        for name, fn in self._steps:
            try:
                image = fn(image)
            except Exception as e:
                print(f"  预处理步骤 '{name}' 失败: {e}")
                continue
        return image

    def describe(self) -> str:
        """返回当前预处理链描述"""
        if not self._steps:
            return "无预处理"
        return " → ".join([name for name, _ in self._steps])


# 快速测试
if __name__ == "__main__":
    pipe = PreprocessingPipeline({
        "auto_rotate": True,
        "enhance_contrast": True,
        "perspective_correct": False,
    })
    img = Image.new("RGB", (224, 224), color=(200, 200, 200))
    result = pipe(img)
    print(f"预处理测试通过: {pipe.describe()}, 输出尺寸: {result.size}")
