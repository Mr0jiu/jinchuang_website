"""
基于 MD5 哈希重建 CSV
- 计算当前磁盘上每个 face_signing.jpg 的 MD5
- 匹配 committed CSV 中同一文件（MD5 相同）
- 保留备注数据，重建 file_path
"""
import csv, os, re, subprocess, hashlib

DATA_DIR = 'data'
CSV_PATH = 'data/annotations.csv'

def md5_file(path):
    """计算文件的 MD5"""
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

# 1. 读取 committed CSV
result = subprocess.run(['git', 'show', 'HEAD:data/annotations.csv'],
                       capture_output=True, text=True, encoding='utf-8')
reader = csv.DictReader(result.stdout.splitlines())
committed_rows = list(reader)
fieldnames = reader.fieldnames
print(f'Committed CSV: {len(committed_rows)} rows')

# 2. 计算 committed CSV 中所有 face_signing.jpg 的 MD5
# 需要从 git 中检出 committed 文件内容来计算 MD5
print('计算 committed 文件 MD5...')
committed_face_md5 = {}  # md5 -> row
for r in committed_rows:
    if r['image_type'] == 'face_signing':
        # 从 git HEAD 读取文件内容并计算 MD5
        try:
            result = subprocess.run(
                ['git', 'show', f'HEAD:{r["file_path"]}'],
                capture_output=True, timeout=30
            )
            if result.returncode == 0:
                md5 = hashlib.md5(result.stdout).hexdigest()
                if md5 not in committed_face_md5:
                    committed_face_md5[md5] = []
                committed_face_md5[md5].append(r)
        except:
            pass

print(f'Committed 面签照 MD5 数: {len(committed_face_md5)}')

# 3. 扫描当前磁盘上的所有文件
print('\n扫描当前磁盘...')
nonface_rows = []  # 非面签照：直接从 committed CSV 的 loan_001~074 取
current_face_files = []  # (loan_num, file_path, md5)

for entry in sorted(os.listdir(DATA_DIR)):
    full = os.path.join(DATA_DIR, entry)
    if not os.path.isdir(full):
        continue
    m = re.match(r'loan_(\d+)$', entry)
    if not m:
        continue
    loan_num = int(m.group(1))

    # 扫描文件夹内所有文件
    for fname in os.listdir(full):
        fpath = os.path.join(full, fname)
        if not os.path.isfile(fpath):
            continue

        rel_path = f'loan_{loan_num:04d}/{fname}'
        ftype = fname.replace('.jpg', '')

        if ftype == 'face_signing':
            md5 = md5_file(fpath)
            current_face_files.append((loan_num, rel_path, md5))
        else:
            # 非面签照：在 committed CSV 中找到对应行
            match = [r for r in committed_rows
                     if r['file_path'].endswith(fname)
                     and r['image_type'] == ftype
                     and re.search(r'loan_\d+', r['file_path'])]
            # 简化：使用该 loan 在 committed 中的数据
            committed_match = [r for r in committed_rows
                               if r['file_path'] == f'loan_{loan_num:04d}/{fname}'
                               and r['image_type'] == ftype
                               and r['business_type'] != '']
            if committed_match:
                r = dict(committed_match[0])
                r['file_path'] = rel_path
                nonface_rows.append(r)
            else:
                # 从 committed CSV 中按类型匹配
                found = False
                for cr in committed_rows:
                    if cr['image_type'] == ftype and cr['business_type']:
                        r = dict(cr)
                        r['file_path'] = rel_path
                        r['image_id'] = f"IMG_{loan_num:05d}"
                        r['loan_id'] = f"LN{2024000000 + loan_num:010d}"
                        nonface_rows.append(r)
                        found = True
                        break
                if not found:
                    print(f'  未匹配: {rel_path}')

print(f'当前面签照文件: {len(current_face_files)}')
print(f'当前非面签照行: {len(nonface_rows)}')

# 4. MD5 匹配
matched = 0
unmatched = 0
new_face_rows = []

for loan_num, rel_path, md5 in sorted(current_face_files, key=lambda x: x[0]):
    if md5 in committed_face_md5:
        candidates = committed_face_md5[md5]
        # 如果多个行有相同 MD5（完全相同的内容），取第一个
        r = dict(candidates[0])
        if len(candidates) > 1:
            # 多行有相同 MD5，尝试按匹配 loan_num 或 image_id 细化
            better = [c for c in candidates
                      if f'loan_{loan_num:04d}' in c['file_path']]
            if better:
                r = dict(better[0])

        r['file_path'] = rel_path
        # 更新 image_id 反映新编号（可选）
        new_face_rows.append(r)
        matched += 1
    else:
        # 无法匹配 - 可能是生成的图像
        print(f'  未匹配: {rel_path} (md5={md5})')
        unmatched += 1

print(f'\n匹配成功: {matched}')
print(f'匹配失败: {unmatched}')

if unmatched > 0:
    print('⚠️  有未能匹配的文件，CSV 不完整！')
    # 询问是否继续
    response = input('是否继续？(y/N): ')
    if response.lower() != 'y':
        print('已取消')
        exit()

# 5. 合并所有行
final_rows = new_face_rows + nonface_rows
# 排序
final_rows.sort(key=lambda r: (
    int(re.search(r'loan_(\d+)', r['file_path']).group(1))
    if re.search(r'loan_(\d+)', r['file_path']) else 0,
    r['image_type']
))

# 6. 写回 CSV
with open(CSV_PATH, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(final_rows)

print(f'\nCSV 已写回: {len(final_rows)} 行')

# 7. 最终验证
print('\n=== 最终状态 ===')
vface = [r for r in final_rows if r['image_type'] == 'face_signing']
vsim = [r for r in vface if r['is_similar_pair'] == '1']
vind = [r for r in vface if r['is_similar_pair'] == '0']
print(f'总行数: {len(final_rows)}')
print(f'面签照: {len(vface)}')
print(f'相似图: {len(vsim)}')
print(f'独立图: {len(vind)}')

# 验证所有文件存在
bad = 0
for r in final_rows:
    if not os.path.isfile(os.path.join(DATA_DIR, r['file_path'])):
        bad += 1
        if bad <= 5:
            print(f'  文件缺失: {r["file_path"]}')
print(f'无效路径: {bad}')

# 类型汇总
from collections import Counter
types = Counter(r['image_type'] for r in final_rows)
for t, c in types.most_common():
    print(f'  {t}: {c}')
