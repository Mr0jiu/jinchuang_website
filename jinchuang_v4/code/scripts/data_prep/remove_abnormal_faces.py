"""
remove_abnormal_faces.py
剔除 225 张异常面签照（单人/无人/多人），更新 CSV
"""
import csv
import os
import re
from collections import defaultdict

DATA_DIR = 'data'
CSV_PATH = 'data/annotations.csv'
CHECK_CSV = 'data/face_count_check.csv'

# 读取异常列表
abnormal_loans = set()
with open(CHECK_CSV, encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['status'] != 'OK':
            n = int(row['loan_id'].split('_')[1])
            abnormal_loans.add(n)

print(f"待剔除异常图片: {len(abnormal_loans)} 张")

# 1. 删除文件
deleted_files = 0
for n in abnormal_loans:
    for fmt in [f'loan_{n:04d}', f'loan_{n:03d}']:
        path = os.path.join(DATA_DIR, fmt, 'face_signing.jpg')
        if os.path.exists(path):
            os.remove(path)
            deleted_files += 1
            break
print(f"已删除文件: {deleted_files} 张")

# 2. 读取 CSV，过滤掉异常面签照的行
rows_kept = []
rows_removed = 0
affected_sgs = set()

with open(CSV_PATH, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        if row['image_type'] != 'face_signing':
            rows_kept.append(row)
            continue

        m = re.search(r'loan_0*(\d+)', row['file_path'])
        n = int(m.group(1)) if m else 0

        if n in abnormal_loans:
            rows_removed += 1
            if row['similar_group']:
                affected_sgs.add(row['similar_group'])
        else:
            rows_kept.append(row)

print(f"CSV 移除行: {rows_removed}")
print(f"受影响的相似组: {len(affected_sgs)}")

# 3. 写回 CSV
with open(CSV_PATH, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows_kept)

print(f"CSV 保留总行数: {len(rows_kept)}")

# 4. 检查受影响相似组的状态
sg_counts = defaultdict(int)
for row in rows_kept:
    sg = row['similar_group']
    if sg and sg.startswith('SG_') and row['is_similar_pair'] == '1':
        sg_counts[sg] += 1

print(f"\n受影响相似组的状态:")
for sg in sorted(affected_sgs):
    cnt = sg_counts.get(sg, 0)
    if cnt == 0:
        print(f"  ⚠ {sg}: 已为空组（全部被剔除）")
    elif cnt == 1:
        print(f"  ⚠ {sg}: 仅剩 1 张")
    else:
        print(f"  ✓ {sg}: 剩余 {cnt} 张")

print(f"\n完成！")
