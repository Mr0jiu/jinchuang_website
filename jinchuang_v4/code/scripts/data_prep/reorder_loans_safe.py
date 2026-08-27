"""
reorder_loans_safe.py
安全重排 loan 目录编号（无冲突）
"""
import csv
import os
import re
import shutil

DATA_DIR = 'data'
CSV_PATH = 'data/annotations.csv'

# 1. 收集所有有 face_signing.jpg 的目录，按编号排序
valid = []
for n in range(1, 4482):
    for fmt in [f'loan_{n:04d}', f'loan_{n:03d}']:
        d = os.path.join(DATA_DIR, fmt)
        if os.path.isfile(os.path.join(d, 'face_signing.jpg')):
            valid.append((n, d))
            break

valid.sort(key=lambda x: x[0])
print(f"有面签照的目录: {len(valid)} 个")

# 2. 第一步：全部移到临时目录名
print("\n第一步：移动到临时目录...")
tmp_map = {}  # old_number → (temp_path, new_number)
for i, (old_n, old_dir) in enumerate(valid):
    new_n = i + 1
    tmp_name = f'_tmp_loan_{old_n:04d}'
    tmp_path = os.path.join(DATA_DIR, tmp_name)
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    os.rename(old_dir, tmp_path)
    tmp_map[old_n] = (tmp_path, new_n)
    if (i+1) % 500 == 0:
        print(f"  已移动 {i+1}/{len(valid)}")

print(f"  全部 {len(valid)} 个目录已移到临时名")

# 3. 第二步：从临时名重命名为最终名
print("\n第二步：重命名为最终编号...")
old_to_new = {}  # old_number → new_number
for i, (old_n, _) in enumerate(valid):
    new_n = i + 1
    tmp_path, _ = tmp_map[old_n]
    new_dir_name = f'loan_{new_n:03d}' if new_n <= 74 else f'loan_{new_n:04d}'
    new_path = os.path.join(DATA_DIR, new_dir_name)
    os.rename(tmp_path, new_path)
    old_to_new[old_n] = new_n

print(f"  全部重命名完成")

# 4. 更新 CSV
print("\n第三步：更新 CSV...")
with open(CSV_PATH, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

updated = 0
for row in rows:
    m = re.search(r'loan_0*(\d+)', row['file_path'])
    if not m:
        continue
    old_n = int(m.group(1))
    if old_n in old_to_new:
        new_n = old_to_new[old_n]
        parts = row['file_path'].split('/')
        parts[0] = f'loan_{new_n:03d}' if new_n <= 74 else f'loan_{new_n:04d}'
        row['file_path'] = '/'.join(parts)
        row['loan_id'] = f'LN2024{new_n:07d}'
        updated += 1

# 排序
type_order = {'bank_statement':0,'contract':1,'id_card_back':2,'id_card_front':3,'face_signing':4}
rows.sort(key=lambda r: (
    int(re.search(r'loan_0*(\d+)', r['file_path']).group(1)) if re.search(r'loan_0*(\d+)', r['file_path']) else 0,
    type_order.get(r['image_type'], 99)
))

# 重写 image_id
for i, row in enumerate(rows):
    row['image_id'] = f'IMG_{i+1:05d}'

with open(CSV_PATH, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"  CSV 更新 {updated} 行，总行数 {len(rows)}")

# 5. 验证
face_count = sum(1 for r in rows if r['image_type'] == 'face_signing')
disk_count = 0
for n in range(1, len(valid) + 1):
    np = f'loan_{n:03d}' if n <= 74 else f'loan_{n:04d}'
    if os.path.isfile(os.path.join(DATA_DIR, np, 'face_signing.jpg')):
        disk_count += 1

print(f"\n验证:")
print(f"  CSV 面签行: {face_count}")
print(f"  磁盘面签照: {disk_count}")
print(f"  一致: {'是' if face_count == disk_count else '否'}")

# 6. 清理残留的空目录和临时目录
print("\n第四步：清理...")
for n in range(1, 4482):
    for fmt in [f'loan_{n:04d}', f'loan_{n:03d}']:
        d = os.path.join(DATA_DIR, fmt)
        if os.path.isdir(d) and not os.listdir(d):
            shutil.rmtree(d)
            break

tmp_left = [d for d in os.listdir(DATA_DIR) if d.startswith('_tmp_')]
for t in tmp_left:
    shutil.rmtree(os.path.join(DATA_DIR, t))

print(f"  清理完成，当前 loan 目录范围: 1 ~ {len(valid)}")
print("全部完成！")
