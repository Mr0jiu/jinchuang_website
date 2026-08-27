"""
check_face_count.py
扫描 data/loan_001~loan_4481/face_signing.jpg
用 InsightFace (RetinaFace) 检测每张图中的人数
标注要求：必须是双人，单人/多人标记为异常
"""
import os
import csv
import re
import argparse
from collections import defaultdict

import cv2
import numpy as np
from insightface.app import FaceAnalysis

DATA_DIR = 'data'
CSV_PATH = 'data/annotations.csv'
OUTPUT_CSV = 'data/face_count_check.csv'


def find_image(n):
    """查找 loan_n 的面签照，兼容 3/4 位目录名"""
    for fmt in [f'loan_{n:04d}', f'loan_{n:03d}']:
        p = os.path.join(DATA_DIR, fmt, 'face_signing.jpg')
        if os.path.exists(p):
            return p
    return None


def loan_num_from_path(path):
    m = re.search(r'loan_0*(\d+)', path)
    return int(m.group(1)) if m else 0


def main():
    print("=" * 60)
    print("面签照双人检测 - InsightFace RetinaFace")
    print("=" * 60)

    # 初始化 InsightFace
    print("加载人脸检测模型...")
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("模型加载完成！")

    # 收集所有面签照文件
    face_images = []
    for n in range(1, 4482):
        path = find_image(n)
        if path:
            face_images.append((n, path))

    print(f"找到 {len(face_images)} 张面签照文件\n")

    results = []
    abnormal = []

    for idx, (n, path) in enumerate(face_images):
        img = cv2.imread(path)
        if img is None:
            print(f"[{idx+1}/{len(face_images)}] loan_{n:04d} 读取失败")
            results.append((n, -1, 'READ_ERROR'))
            continue

        # InsightFace 检测
        faces = app.get(img)
        face_count = len(faces)

        # 判断
        if face_count == 0:
            status = 'NO_PERSON'
        elif face_count == 1:
            status = 'SINGLE'
        elif face_count == 2:
            status = 'OK'
        else:
            status = f'MULTI_{face_count}'

        results.append((n, face_count, status))

        if status != 'OK':
            abnormal.append((n, face_count, status, path))
            print(f"[{idx+1}/{len(face_images)}] loan_{n:04d} 异常: {status} ({face_count}人)")
        elif idx % 300 == 0:
            print(f"[{idx+1}/{len(face_images)}] loan_{n:04d} OK ({face_count}人)")

    # 统计
    print(f"\n{'='*60}")
    print(f"检测完成！总计 {len(results)} 张面签照")
    print(f"{'='*60}")

    status_count = defaultdict(int)
    for r in results:
        status_count[r[2]] += 1

    for s, c in sorted(status_count.items()):
        print(f"  {s}: {c} 张")

    print(f"\n--- 异常图片 ({len(abnormal)} 张) ---")
    for n, fc, status, path in abnormal:
        print(f"  loan_{n:04d} {status} ({fc}人)")

    # 保存 CSV 报告
    with open(OUTPUT_CSV, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['loan_id', 'face_count', 'status', 'file_path'])
        for n, fc, st in results:
            writer.writerow([f'loan_{n:04d}', fc, st, f'loan_{n:04d}/face_signing.jpg'])

    print(f"\n报告已保存: {OUTPUT_CSV}")
    return abnormal


if __name__ == '__main__':
    abnormal = main()
