"""
reorder_loans.py
将面签照按顺序排列，删除空目录，更新 CSV
"""
import csv
import os
import re
import shutil

DATA_DIR = 'data'
CSV_PATH = 'data/annotations.csv'

# 1. 找到所有有 face_signing.jpg 的 loan 目录
valid_loans = []
for n in range(1, 4482):
    for fmt in [f'loan_{n:04d}', f'loan_{n:03d}']:
        path = os.path.join(DATA_DIR, fmt, 'face_signing.jpg')
        if os.path.exists(path):
            valid_loans.append((n, os.path.join(DATA_DIR, fmt)))
            break

valid_loans.sort(key=lambda x: x[0])
print(f"有面签照的目录: {len(valid_loans)} 个")

# 2. 读取 CSV
with open(CSV_PATH, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

print(f"CSV 总行数: {len(rows)}")

# 3. 构建 old_loan → new_loan 映射
#    也保留 old_dir 和 new_dir
loan_map = {}  # old_number → new_number
for i, (old_n, old_dir) in enumerate(valid_loans):
    new_n = i + 1  # 1-based sequential
    loan_map[old_n] = new_n

# 4. 重命名目录（从后往前避免冲突）
#    先检查是否有目标目录名冲突
print("\n检查目录冲突...")
new_paths = {}
for old_n, old_dir in valid_loans:
    new_n = loan_map[old_n]
    new_dir = os.path.join(DATA_DIR, f'loan_{new_n:03d}' if new_n <= 74 else f'loan_{new_n:04d}')
    new_paths[old_n] = (old_dir, new_dir)

    if os.path.exists(new_dir) and old_dir != new_dir:
        # 如果新目录名已存在且不是同一个目录
        if os.path.exists(os.path.join(new_dir, 'face_signing.jpg')):
            print(f"  冲突: {new_dir} 已有面签照！")
            # 需要先移动到临时目录
            tmp = new_dir + '_tmp'
            if os.path.exists(tmp):
                shutil.rmtree(tmp)
            os.rename(new_dir, tmp)
            print(f"  已将 {new_dir} → {tmp}")

# 从后往前重命名
print("\n开始重命名...")
for old_n, old_dir in sorted(new_paths.keys(), reverse=True):
    _, new_dir = new_paths[old_n]
    if old_dir == new_dir:
        continue
    if os.path.exists(new_dir):
        # 如果是冲突后被临时移走的，忽略
        pass
    os.makedirs(os.path.dirname(new_dir), exist_ok=True)
    os.rename(old_dir, new_dir)

# 清理临时目录
for n in range(1, 4482):
    tmp = os.path.join(DATA_DIR, f'loan_{n:04d}_tmp')
    if os.path.exists(tmp):
        # 检查是否已有目标
        pass

print(f"重命名完成")

# 5. 更新 CSV 中的 file_path 和 loan_id
print("\n更新 CSV...")
updated = 0
for row in rows:
    m = re.search(r'loan_0*(\d+)', row['file_path'])
    if not m:
        continue
    old_n = int(m.group(1))
    if old_n in loan_map:
        new_n = loan_map[old_n]
        # 更新 file_path
        parts = row['file_path'].split('/')
        new_dir = f'loan_{new_n:03d}' if new_n <= 74 else f'loan_{new_n:04d}'
        parts[0] = new_dir
        row['file_path'] = '/'.join(parts)
        # 更新 loan_id
        row['loan_id'] = f'LN2024{new_n:07d}'
        updated += 1

# 排序
def sort_key(r):
    m = re.search(r'loan_0*(\d+)', r['file_path'])
    n = int(m.group(1)) if m else 0
    order = {'bank_statement':0,'contract':1,'id_card_back':2,'id_card_front':3,'face_signing':4}
    return (n, order.get(r['image_type'], 99))

rows.sort(key=sort_key)

# 重新生成 image_id
for i, row in enumerate(rows):
    row['image_id'] = f'IMG_{i+1:05d}'

with open(CSV_PATH, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"CSV 更新: {updated} 行")

# 6. 验证
face_count = sum(1 for r in rows if r['image_type'] == 'face_signing')
dir_count = 0
for n in range(1, len(valid_loans) + 1):
    path = os.path.join(DATA_DIR, f'loan_{n:03d}' if n <= 74 else f'loan_{n:04d}', 'face_signing.jpg')
    if os.path.exists(path):
        dir_count += 1

print(f"\n验证:")
print(f"  CSV 面签行: {face_count}")
print(f"  磁盘面签照: {dir_count}")
print(f"  完全一致: {'✅' if face_count == dir_count else '❌'}")

# 删除空的旧目录
print("\n清理空的旧目录...")
for n in range(1, 4482):
    for fmt in [f'loan_{n:04d}', f'loan_{n:03d}']:
        d = os.path.join(DATA_DIR, fmt)
        if os.path.isdir(d):
            files = [f for f in os.listdir(d) if not f.endswith('.tmp')]
            if not files:
                shutil.rmtree(d)

print("全部完成！")
