"""
修复 CSV v5 - 位置映射
因为重命名是保序的 (order-preserving)，第 i 个剩余的 committed 行对应第 i 个当前文件夹
"""
import csv, os, re, subprocess

DATA_DIR = 'data'
CSV_PATH = 'data/annotations.csv'
FIELDS = ['image_id', 'file_path', 'image_type', 'business_type', 'loan_id', 'similar_group', 'is_similar_pair']

def loan_path(n, f):
    return f'loan_{n:03d}/{f}' if n <= 74 else f'loan_{n:04d}/{f}'

def loan_num(p):
    m = re.search(r'loan_0*(\d+)', p)
    return int(m.group(1)) if m else 0

# 1. Committed CSV (strip BOM)
r = subprocess.run(['git', 'show', 'HEAD:data/annotations.csv'], capture_output=True, text=True, encoding='utf-8')
raw = r.stdout.lstrip('﻿')
committed = list(csv.DictReader(raw.splitlines()))
print(f'Committed CSV: {len(committed)}')

# 2. 分离面签照，排序
face = [x for x in committed if x['image_type'] == 'face_signing']
face.sort(key=lambda x: loan_num(x['file_path']))

sim = [x for x in face if x['is_similar_pair'] == '1']
ind = [x for x in face if x['is_similar_pair'] == '0']
ind_75 = sorted([x for x in ind if loan_num(x['file_path']) > 74],
                key=lambda x: loan_num(x['file_path']))
ind_1_74 = [x for x in ind if loan_num(x['file_path']) <= 74]

print(f'Committed: 面签照={len(face)} 相似={len(sim)} 独立={len(ind)} (075+独立={len(ind_75)})')

# 3. 删除最后 407 张 075+ 独立图
keep_ind_75 = ind_75[:-407]
removed_count = len(ind_75) - len(keep_ind_75)
print(f'删除独立图: {removed_count} (保留 {len(keep_ind_75)})')

# 4. 剩余面签照（按原 loan 号排序）
remaining = sim + ind_1_74 + keep_ind_75
remaining.sort(key=lambda x: loan_num(x['file_path']))
print(f'剩余面签照: {len(remaining)} (预期 4074)')

# 5. 扫描当前磁盘上 075+ 的文件夹
current_dirs = []
for e in os.listdir(DATA_DIR):
    m = re.match(r'loan_0*(\d+)$', e)
    if m and os.path.isdir(os.path.join(DATA_DIR, e)):
        n = int(m.group(1))
        if n > 74:
            current_dirs.append(n)

current_dirs.sort()
print(f'当前 075+ 文件夹: {len(current_dirs)} 个 (loan_{current_dirs[0]}~loan_{current_dirs[-1]})')

# 6. 验证数量匹配
loan_75plus = [x for x in remaining if loan_num(x['file_path']) > 74]
print(f'映射前验证: remaining 075+={len(loan_75plus)}, 磁盘 075+={len(current_dirs)}')
assert len(loan_75plus) == len(current_dirs), f'数量不匹配! {len(loan_75plus)} vs {len(current_dirs)}'

# 7. 构建行
rows = []

# 非面签照直接用 committed 的（loan_001~074）
for r in committed:
    if r['image_type'] != 'face_signing':
        n = loan_num(r['file_path'])
        if 1 <= n <= 74:
            new_path = loan_path(n, r['file_path'].split('/')[-1])
            rr = dict(r)
            rr['file_path'] = new_path
            rows.append(rr)

# 面签照用位置映射
# loan_001~074 的面签照保持不变
for i, r in enumerate(face):
    n = loan_num(r['file_path'])
    if n <= 74:
        rr = dict(r)
        rr['file_path'] = loan_path(n, 'face_signing.jpg')
        rows.append(rr)

# loan_075+ 的面签照做位置映射
for old_row, new_n in zip(loan_75plus, current_dirs):
    rr = dict(old_row)
    rr['file_path'] = loan_path(new_n, 'face_signing.jpg')
    rows.append(rr)

print(f'总行数: {len(rows)}')

# 8. 排序
rows.sort(key=lambda r: (loan_num(r['file_path']),
    {'bank_statement':0,'contract':1,'id_card_back':2,'id_card_front':3,'face_signing':4}
    .get(r['image_type'], 99)))

# 9. 写回
with open(CSV_PATH, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(rows)

# 10. 验证
vf = [r for r in rows if r['image_type'] == 'face_signing']
print(f'\n面签照:{len(vf)} 相似:{sum(1 for r in vf if r["is_similar_pair"]=="1")} 独立:{sum(1 for r in vf if r["is_similar_pair"]=="0")}')

bad = sum(1 for r in rows if not os.path.isfile(os.path.join(DATA_DIR, r['file_path'])))
print(f'无效路径:{bad}')
if bad > 0:
    for r in rows[:5]:
        if not os.path.isfile(os.path.join(DATA_DIR, r['file_path'])):
            print(f'  缺失: {r["file_path"]}')

for n in range(1, 75):
    cnt = sum(1 for r in rows if r['file_path'].startswith(f'loan_{n:03d}/'))
    if cnt != 5:
        print(f'  loan_{n:03d}: {cnt}/5 (应为 5)')

from collections import Counter
for t, c in Counter(r['image_type'] for r in rows).most_common():
    print(f'  {t}: {c}')

# 验证总数
dir_count = len([1 for e in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, e))
    and re.match(r'loan_0*\d+$', e)])
print(f'磁盘文件夹数: {dir_count}')
print(f'CSV 行数:   {len(rows)}')
print('Done!')
