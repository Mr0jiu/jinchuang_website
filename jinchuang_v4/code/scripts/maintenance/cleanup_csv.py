"""
cleanup_csv.py
清理 CSV 中引用已删除面签照的行，确保数据一致性
"""
import csv
import os
import re
from collections import defaultdict

DATA_DIR = 'data'
CSV_PATH = 'data/annotations.csv'

# 读取 CSV
with open(CSV_PATH, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

print(f"原始总行数: {len(rows)}")

# 分离面签照和非面签照
face_rows = [r for r in rows if r['image_type'] == 'face_signing']
other_rows = [r for r in rows if r['image_type'] != 'face_signing']

print(f"面签照行: {len(face_rows)}")
print(f"非面签照行: {len(other_rows)}")

# 检查面签照文件是否存在
kept_face = []
removed_face = []
for r in face_rows:
    fp = os.path.join(DATA_DIR, r['file_path'])
    if os.path.exists(fp):
        kept_face.append(r)
    else:
        removed_face.append(r)

print(f"保留面签照行: {len(kept_face)}")
print(f"移除面签照行: {len(removed_face)}")

# 统计移除的相似组影响
removed_sgs = defaultdict(int)
for r in removed_face:
    if r['similar_group']:
        removed_sgs[r['similar_group']] += 1

# 合并并写回
all_kept = other_rows + kept_face
all_kept.sort(key=lambda r: (
    int(re.search(r'loan_0*(\d+)', r['file_path']).group(1)) if re.search(r'loan_0*(\d+)', r['file_path']) else 0,
    {'bank_statement':0,'contract':1,'id_card_back':2,'id_card_front':3,'face_signing':4}.get(r['image_type'], 99)
))

with open(CSV_PATH, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_kept)

print(f"\nCSV 已更新: {len(all_kept)} 行 (移除 {len(removed_face)} 行)")

# 检查组完整性
sg_counts = defaultdict(int)
for r in kept_face:
    if r['similar_group'] and r['is_similar_pair'] == '1':
        sg_counts[r['similar_group']] += 1

print(f"\n相似组状态:")
problematic = 0
for sg in sorted(sg_counts.keys()):
    cnt = sg_counts[sg]
    if cnt <= 1:
        print(f"  ⚠ {sg}: 仅 {cnt} 张")
        problematic += 1

if problematic == 0:
    print(f"  全部正常！共 {len(sg_counts)} 组")

# 最终统计
sim_count = sum(1 for r in kept_face if r['is_similar_pair'] == '1')
ind_count = sum(1 for r in kept_face if r['is_similar_pair'] == '0')
print(f"\n最终面签照统计:")
print(f"  相似图: {sim_count} 张, {len(sg_counts)} 组")
print(f"  独立图: {ind_count} 张")
print(f"  合计: {sim_count + ind_count} 张")
