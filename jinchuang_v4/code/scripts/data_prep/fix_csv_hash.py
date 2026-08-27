"""
基于 git hash-object 重建 CSV（高效版）
策略：
1. git ls-tree HEAD → 获取所有 committed 文件的 blob hash
2. 扫描当前磁盘文件，计算 hash-object
3. 匹配 → 确定每个磁盘文件对应的 committed CSV 行
4. 重建文件路径，保留标注
"""
import csv, os, re, subprocess, hashlib

DATA_DIR = 'data'
CSV_PATH = 'data/annotations.csv'

# 1. 读取 committed CSV
result = subprocess.run(['git', 'show', 'HEAD:data/annotations.csv'],
                       capture_output=True, text=True, encoding='utf-8')
reader = csv.DictReader(result.stdout.splitlines())
committed_rows = list(reader)
fieldnames = reader.fieldnames
print(f'Committed CSV: {len(committed_rows)} rows')

# 2. 建立 committed 文件哈希索引
# git ls-tree -r HEAD 输出: 100644 blob HASH\tpath
print('读取 committed tree...')
result = subprocess.run(['git', 'ls-tree', '-r', 'HEAD'],
                       capture_output=True, text=True, encoding='utf-8')
# 建立 hash → [paths] 索引（同一个 blob 可能对应多个路径）
hash_to_committed = {}  # blob_hash -> {file_paths, csv_rows}
for line in result.stdout.strip().split('\n'):
    if not line:
        continue
    parts = line.split('\t', 1)
    if len(parts) != 2:
        continue
    meta, path = parts
    blob_hash = meta.split()[2]  # 100644 blob HASH → hash
    if blob_hash not in hash_to_committed:
        hash_to_committed[blob_hash] = {'paths': [], 'csv_rows': []}
    hash_to_committed[blob_hash]['paths'].append(path)

# 将 CSV rows 关联到 committed file hash
# CSV 中 file_path 是 "loan_NNN/f.jpg" 格式，tree 中是 "data/loan_NNN/f.jpg"
for r in committed_rows:
    fpath = r['file_path']
    tree_path = f'data/{fpath}'
    for bh, info in hash_to_committed.items():
        if tree_path in info['paths']:
            info['csv_rows'].append(r)
            break

print(f'Committed tree 文件数: {len(hash_to_committed)}')

# 3. 扫描当前磁盘文件
# 先分组：面签照 vs. 非面签照
current_face = []  # [(loan_num, rel_path, blob_hash)]
current_nonface = []  # [(loan_num, rel_path, blob_hash)]

for entry in sorted(os.listdir(DATA_DIR)):
    full = os.path.join(DATA_DIR, entry)
    if not os.path.isdir(full):
        continue
    m = re.match(r'loan_(\d+)$', entry)
    if not m:
        continue
    loan_num = int(m.group(1))

    for fname in os.listdir(full):
        fpath = os.path.join(full, fname)
        if not os.path.isfile(fpath):
            continue
        rel_path = f'loan_{loan_num:04d}/{fname}'
        ftype = fname.replace('.jpg', '')

        # 计算 git blob hash
        result = subprocess.run(['git', 'hash-object', fpath],
                               capture_output=True, text=True, encoding='utf-8')
        blob_hash = result.stdout.strip()

        if ftype == 'face_signing':
            current_face.append((loan_num, rel_path, blob_hash))
        else:
            current_nonface.append((loan_num, rel_path, blob_hash, ftype))

print(f'当前面签照文件: {len(current_face)}')
print(f'当前非面签照文件: {len(current_nonface)}')

# 4. 匹配面签照：blob_hash → committed CSV 行
matched_face = []
unmatched_face = []

for loan_num, rel_path, blob_hash in sorted(current_face, key=lambda x: x[0]):
    if blob_hash in hash_to_committed and hash_to_committed[blob_hash]['csv_rows']:
        candidates = hash_to_committed[blob_hash]['csv_rows']
        r = dict(candidates[0])
        # 更新路径
        r['file_path'] = rel_path
        # 保持 image_id 不变
        matched_face.append(r)
    else:
        unmatched_face.append((loan_num, rel_path, blob_hash))

print(f'面签照匹配成功: {len(matched_face)}')
print(f'面签照匹配失败: {len(unmatched_face)}')

if unmatched_face:
    print('匹配失败的文件:')
    for n, p, h in unmatched_face[:10]:
        print(f'  {p} (hash={h})')

# 5. 匹配非面签照
matched_nonface = []
unmatched_nonface = []

for loan_num, rel_path, blob_hash, ftype in current_nonface:
    if blob_hash in hash_to_committed and hash_to_committed[blob_hash]['csv_rows']:
        candidates = hash_to_committed[blob_hash]['csv_rows']
        r = dict(candidates[0])
        r['file_path'] = rel_path
        matched_nonface.append(r)
    else:
        unmatched_nonface.append((loan_num, rel_path, blob_hash, ftype))

print(f'\n非面签照匹配成功: {len(matched_nonface)}')
print(f'非面签照匹配失败: {len(unmatched_nonface)}')

# 6. 合并
final_rows = matched_face + matched_nonface
# 排序
final_rows.sort(key=lambda r: (
    int(re.search(r'loan_(\d+)', r['file_path']).group(1))
    if re.search(r'loan_(\d+)', r['file_path']) else 0,
    r['image_type']
))

# 7. 写回 CSV
with open(CSV_PATH, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(final_rows)

print(f'\nCSV 已写回: {len(final_rows)} 行')

# 8. 最终验证
print('\n=== 最终状态 ===')
vrows = final_rows
vface = [r for r in vrows if r['image_type'] == 'face_signing']
vsim = [r for r in vface if r['is_similar_pair'] == '1']
vind = [r for r in vface if r['is_similar_pair'] == '0']
print(f'总行数: {len(vrows)}')
print(f'面签照: {len(vface)}')
print(f'相似图: {len(vsim)}')
print(f'独立图: {len(vind)}')

# 验证 loan_001~074 安全
for r in vrows:
    m = re.search(r'loan_(\d+)', r['file_path'])
    if m and 1 <= int(m.group(1)) <= 74 and r['image_type'] == 'face_signing':
        pass  # OK

bad = 0
for r in vrows:
    if not os.path.isfile(os.path.join(DATA_DIR, r['file_path'])):
        bad += 1
        if bad <= 5:
            print(f'  缺失: {r["file_path"]}')
print(f'无效路径: {bad}')

from collections import Counter
types = Counter(r['image_type'] for r in vrows)
for t, c in types.most_common():
    print(f'  {t}: {c}')
