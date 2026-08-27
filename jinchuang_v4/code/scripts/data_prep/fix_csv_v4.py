"""
修复 CSV v4 - 混合策略：hash 匹配 + 位置匹配
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

# 1. Committed CSV
r = subprocess.run(['git', 'show', 'HEAD:data/annotations.csv'], capture_output=True, text=True, encoding='utf-8')
committed = list(csv.DictReader(r.stdout.splitlines()))
print(f'Committed CSV: {len(committed)}')

face_c = [x for x in committed if x['image_type'] == 'face_signing']
nonface_c = [x for x in committed if x['image_type'] != 'face_signing']
face_c.sort(key=lambda x: loan_num(x['file_path']))

sim = [x for x in face_c if x['is_similar_pair'] == '1']
ind = [x for x in face_c if x['is_similar_pair'] == '0']
ind_75 = [x for x in ind if loan_num(x['file_path']) > 74]
ind_74 = [x for x in ind if loan_num(x['file_path']) <= 74]
ind_75.sort(key=lambda x: loan_num(x['file_path']))
print(f'  面签照:{len(face_c)} 相似:{len(sim)} 独立:{len(ind)} (loan_075+:{len(ind_75)})')

# 2. 移除 407 张独立图
killed = set(id(x) for x in ind_75[-407:])
remaining = sim + ind_74 + [x for x in ind_75 if id(x) not in killed]
remaining.sort(key=lambda x: loan_num(x['file_path']))
print(f'  剩余面签照: {len(remaining)} (预期 4074)')

# 3. 建立 hash 索引
r2 = subprocess.run(['git', 'ls-tree', '-r', 'HEAD'], capture_output=True, text=True, encoding='utf-8')
tree_of = {}
for x in committed:
    tree_of[f'data/{x["file_path"]}'] = x
hash_idx = {}
for line in r2.stdout.strip().split('\n'):
    if not line: continue
    p = line.split('\t', 1)
    if len(p) != 2: continue
    h = p[0].split()[2]
    if p[1] in tree_of:
        hash_idx.setdefault(h, []).append(tree_of[p[1]])
print(f'Hash 索引: {len(hash_idx)}')

# 4. 扫描磁盘
disk = []
for e in os.listdir(DATA_DIR):
    f = os.path.join(DATA_DIR, e)
    if not os.path.isdir(f): continue
    m = re.match(r'loan_0*(\d+)$', e)
    if not m: continue
    n = int(m.group(1))
    for fn in os.listdir(f):
        fp = os.path.join(f, fn)
        if not os.path.isfile(fp): continue
        disk.append((n, fn, fp))
print(f'磁盘文件: {len(disk)}')

# 5. Hash 匹配
matched = {}
unmatched = []
for n, fn, fp in disk:
    rp = loan_path(n, fn)
    hr = subprocess.run(['git', 'hash-object', fp], capture_output=True, text=True, encoding='utf-8')
    h = hr.stdout.strip()
    if h in hash_idx:
        row = dict(hash_idx[h][0])
        row['file_path'] = rp
        matched[(n, fn)] = row
    else:
        unmatched.append((n, fn, rp))

print(f'Hash 匹配: {len(matched)}, 未匹配: {len(unmatched)}')
for n, fn, rp in unmatched[:5]:
    print(f'  {rp}')

# 6. 未匹配面签照用位置匹配
um_face = sorted([(n, fn, rp) for n, fn, rp in unmatched if fn == 'face_signing.jpg'])
matched_loans = set(n for n, fn in matched if fn == 'face_signing.jpg')
pool = [x for x in remaining if loan_num(x['file_path']) not in matched_loans]
pool.sort(key=lambda x: loan_num(x['file_path']))
print(f'未匹配面签照:{len(um_face)} 可用池:{len(pool)}')

for (n, fn, rp), cr in zip(um_face, pool):
    row = dict(cr)
    row['file_path'] = rp
    matched[(n, fn)] = row

still = [(n, fn, rp) for n, fn, rp in unmatched if (n, fn) not in matched]
print(f'位匹配后剩余未匹配:{len(still)}')
for n, fn, rp in still:
    ft = fn.replace('.jpg', '')
    if ft == 'face_signing':
        matched[(n, fn)] = {'image_id': f'IMG_{n:05d}', 'file_path': rp,
            'image_type': 'face_signing', 'business_type': '',
            'loan_id': f'LN{2024000000+n:010d}', 'similar_group': '', 'is_similar_pair': '0'}

# 7. 排序输出
rows = list(matched.values())
rows.sort(key=lambda r: (loan_num(r['file_path']),
    {'bank_statement':0,'contract':1,'id_card_back':2,'id_card_front':3,'face_signing':4}.get(r['image_type'],99)))

with open(CSV_PATH, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(rows)

print(f'\nCSV 写回: {len(rows)} 行')

# 8. 验证
v = rows
vf = [r for r in v if r['image_type'] == 'face_signing']
vs = [r for r in vf if r['is_similar_pair'] == '1']
vi = [r for r in vf if r['is_similar_pair'] == '0']
print(f'面签照:{len(vf)} 相似:{len(vs)} 独立:{len(vi)}')

bad = sum(1 for r in v if not os.path.isfile(os.path.join(DATA_DIR, r['file_path'])))
print(f'无效路径:{bad}')

for n in range(1, 75):
    cnt = sum(1 for r in v if r['file_path'].startswith(f'loan_{n:03d}/'))
    if cnt != 5: print(f'  loan_{n:03d}: {cnt}/5')

from collections import Counter
for t, c in Counter(r['image_type'] for r in v).most_common():
    print(f'  {t}: {c}')

print('Done!')
