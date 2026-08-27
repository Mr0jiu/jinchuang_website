"""
修复 CSV v3 - 处理文件夹命名不一致问题
- loan_001~074: 3-digit 格式
- loan_0075~4074: 4-digit 格式
使用 git hash 匹配保留标注数据
"""
import csv, os, re, subprocess

DATA_DIR = 'data'
CSV_PATH = 'data/annotations.csv'

def loan_path(loan_num, filename):
    """生成正确的路径格式"""
    if loan_num <= 74:
        return f'loan_{loan_num:03d}/{filename}'
    else:
        return f'loan_{loan_num:04d}/{filename}'

# 1. 读取 committed CSV
result = subprocess.run(['git', 'show', 'HEAD:data/annotations.csv'],
                       capture_output=True, text=True, encoding='utf-8')
reader = csv.DictReader(result.stdout.splitlines())
committed_rows = list(reader)
fieldnames = reader.fieldnames
print(f'Committed CSV: {len(committed_rows)} rows')

# 2. 构建 committed tree 的 hash 索引
result = subprocess.run(['git', 'ls-tree', '-r', 'HEAD'],
                       capture_output=True, text=True, encoding='utf-8')
# hash -> list of (tree_path, csv_rows)
hash_map = {}
for line in result.stdout.strip().split('\n'):
    if not line:
        continue
    parts = line.split('\t', 1)
    if len(parts) != 2:
        continue
    meta, path = parts
    blob_hash = meta.split()[2]
    if blob_hash not in hash_map:
        hash_map[blob_hash] = []
    hash_map[blob_hash].append(path)

# 关联 CSV rows (tree path = data/ + CSV file_path)
csv_by_tree_path = {}
for r in committed_rows:
    tree_path = f'data/{r["file_path"]}'
    csv_by_tree_path[tree_path] = r

for bh, tree_paths in hash_map.items():
    for tp in tree_paths:
        if tp in csv_by_tree_path:
            hash_map[bh] = hash_map.get(bh, [])
            hash_map[bh].append(csv_by_tree_path[tp])

print(f'Committed tree 文件数: {len(hash_map)}')

# 3. 扫描磁盘文件，按 hash 匹配
# 收集所有磁盘文件的 (loan_num, filename, blob_hash)
disk_files = []
for entry in os.listdir(DATA_DIR):
    full = os.path.join(DATA_DIR, entry)
    if not os.path.isdir(full):
        continue
    # 匹配 loan_NNN 或 loan_NNNN
    m = re.match(r'loan_0*(\d+)$', entry)  # loan_001 or loan_0001 -> 1
    if not m:
        continue
    loan_num = int(m.group(1))
    for fname in os.listdir(full):
        fpath = os.path.join(full, fname)
        if not os.path.isfile(fpath):
            continue
        disk_files.append((loan_num, fname, fpath))

print(f'磁盘文件总数: {len(disk_files)}')

rows_out = []
unmatched_files = []
face_matched = 0
nonface_matched = 0

for loan_num, fname, fpath in disk_files:
    ftype = fname.replace('.jpg', '')
    rel_path = loan_path(loan_num, fname)

    # 计算 hash
    result = subprocess.run(['git', 'hash-object', fpath],
                           capture_output=True, text=True, encoding='utf-8')
    blob_hash = result.stdout.strip()

    # 尝试匹配
    matched_row = None
    if blob_hash in hash_map:
        for row in hash_map[blob_hash]:
            if isinstance(row, dict) and 'image_id' in row:
                matched_row = dict(row)
                break

    if matched_row:
        matched_row['file_path'] = rel_path
        rows_out.append(matched_row)
        if ftype == 'face_signing':
            face_matched += 1
        else:
            nonface_matched += 1
    else:
        unmatched_files.append((loan_num, fname, rel_path, blob_hash))
        if ftype == 'face_signing':
            rows_out.append({
                'image_id': f'IMG_{loan_num:05d}',
                'file_path': rel_path,
                'image_type': 'face_signing',
                'business_type': '',
                'loan_id': f'LN{2024000000 + loan_num:010d}',
                'similar_group': '',
                'is_similar_pair': '0'
            })
            face_matched += 1

print(f'匹配的面签照: {face_matched}')
print(f'匹配的非面签照: {nonface_matched}')
print(f'未匹配文件: {len(unmatched_files)}')

if unmatched_files:
    print('未匹配文件样本:')
    for n, fn, rp, h in unmatched_files[:5]:
        print(f'  {rp} (hash={h[:12]}...)')

# 4. 排序
def sort_key(r):
    m = re.search(r'loan_0*(\d+)', r['file_path'])
    n = int(m.group(1)) if m else 0
    # face_signing 排在最后（每种 loan 类型内部排序）
    ftype_order = {
        'bank_statement': 0, 'contract': 1,
        'id_card_back': 2, 'id_card_front': 3, 'face_signing': 4
    }
    return (n, ftype_order.get(r['image_type'], 99))

rows_out.sort(key=sort_key)

# 5. 写回 CSV
with open(CSV_PATH, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows_out)

print(f'\nCSV 已写回: {len(rows_out)} 行')

# 6. 最终验证
print('\n=== 最终状态 ===')
vface = [r for r in rows_out if r['image_type'] == 'face_signing']
vsim = [r for r in vface if r['is_similar_pair'] == '1']
vind = [r for r in vface if r['is_similar_pair'] == '0']
print(f'总行数: {len(rows_out)}')
print(f'面签照: {len(vface)}')
print(f'相似图: {len(vsim)}')
print(f'独立图: {len(vind)}')

# 验证所有文件存在
bad = 0
for r in rows_out:
    if not os.path.isfile(os.path.join(DATA_DIR, r['file_path'])):
        bad += 1
        if bad <= 5:
            print(f'  缺失: {r["file_path"]}')
print(f'无效路径: {bad}')

from collections import Counter
types = Counter(r['image_type'] for r in rows_out)
for t, c in types.most_common():
    print(f'  {t}: {c}')

# 验证 loan_001~074 完整性
for n in range(1, 75):
    prefix = f'loan_{n:03d}/'
    face_rows = [r for r in rows_out if r['file_path'].startswith(prefix)]
    if len(face_rows) != 5:
        print(f'  ⚠️  loan_{n:03d}: {len(face_rows)}/5 files')

# 验证磁盘与 CSV 文件数一致
csv_count = sum(1 for _ in rows_out)
disk_count = len(disk_files)
print(f'\nCSV 文件数: {csv_count}, 磁盘文件数: {disk_count}')
if csv_count == disk_count:
    print('✅ CSV 文件数与磁盘一致')
else:
    print(f'⚠️  不一致 (差 {abs(csv_count - disk_count)})')
