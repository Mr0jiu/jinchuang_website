"""
gen_407_ai_similar.py
从 SG_016~SG_423 中选 407 组，每组以第一张 face_signing.jpg 为底图，
通过 qwen-image-2.0 修改发型/衣服/背景，生成新图保存到 loan_4075~loan_4481
支持断点续传（已生成的组自动跳过）
"""
import base64, csv, json, os, sys, time, re, io
from PIL import Image
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ── 配置 ──
API_KEY = "sk-ws-H.EHHIYYH.WUBg.MEYCIQClyMTBNJBNI0GawJWkxNEcd-9eDTLl-4gnJ7cJf2BwywIhAJIkLc8u0yTUV7ctmdM1_4W61S9Kex2QDnpM4Yd3cKrK"
HEADERS = {'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'}
API_URL = 'https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation'
CSV_PATH = 'data/annotations.csv'
DATA_DIR = 'data'
PROGRESS_FILE = 'gen_407_progress.txt'
START_LOAN = 4075

# ── 提示词 ──
PROMPTS = {
    'shirt_bg': (
        '图中是两个人：一位客户和一位银行柜员，两人同框。'
        '请对两人都做以下两处修改：'
        '1. 将两个人穿的衣服都改成白色衬衫；'
        '2. 将背景改成浅灰色银行柜台环境。'
        '发型完全保持不变。'
        '保留两个人的面部五官特征完全不变，两人都要在画面中，'
        '双人同框半身构图不变，画质影调和原图一致，高清写实。'
    ),
    'hair': (
        '图中是两个人：一位客户和一位银行柜员，两人同框。'
        '请只修改发型：将两个人的发型都改成板寸短发。'
        '衣服和背景完全保持不变。'
        '保留两个人的面部五官特征完全不变，两人都要在画面中，'
        '双人同框半身构图不变，画质影调和原图一致，高清写实。'
    ),
    'shirt': (
        '图中是两个人：一位客户和一位银行柜员，两人同框。'
        '请只修改衣服：将两个人穿的衣服都改成白色衬衫。'
        '发型和背景完全保持不变。'
        '保留两个人的面部五官特征完全不变，两人都要在画面中，'
        '双人同框半身构图不变，画质影调和原图一致，高清写实。'
    ),
    'bg': (
        '图中是两个人：一位客户和一位银行柜员，两人同框。'
        '请只修改背景：将背景改成浅灰色银行柜台环境。'
        '两个人的发型和衣服完全保持不变。'
        '保留两个人的面部五官特征完全不变，两人都要在画面中，'
        '双人同框半身构图不变，画质影调和原图一致，高清写实。'
    ),
}


def resolve_path(rel_path):
    """兼容 CSV 中两种路径格式：先试原路径，再试零补齐格式"""
    full = os.path.join(DATA_DIR, rel_path)
    if os.path.exists(full):
        return full
    # 尝试修正目录名：loan_075 → loan_0075
    parts = rel_path.split('/')
    if len(parts) >= 2:
        m = re.match(r'^loan_(\d+)$', parts[0])
        if m:
            n = int(m.group(1))
            fixed_dir = f'loan_{n:04d}'
            fixed = os.path.join(DATA_DIR, fixed_dir, *parts[1:])
            if os.path.exists(fixed):
                return fixed
    return full  # 返回原路径，让调用方处理 FileNotFoundError


def build_sg_mapping():
    """从 CSV 构建 SG→(rel_path, loan_id) 映射，仅取首次出现"""
    mapping = {}
    with open(CSV_PATH, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sg = row['similar_group']
            if sg and sg.startswith('SG_') and row['is_similar_pair'] == '1':
                if sg not in mapping:
                    mapping[sg] = (row['file_path'], row['loan_id'])
    return mapping


def get_prompt_key(rank):
    if rank < 100:
        return 'shirt_bg'
    elif rank < 200:
        return 'hair'
    elif rank < 300:
        return 'shirt'
    else:
        return 'bg'


def _session():
    s = requests.Session()
    retries = Retry(total=5, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
    s.mount('https://', HTTPAdapter(max_retries=retries))
    s.mount('http://', HTTPAdapter(max_retries=retries))
    return s


def download_img(url, timeout=120):
    """带重试的图片下载"""
    with _session() as s:
        r = s.get(url, timeout=timeout)
        return r.content


def generate_image(source_path, prompt_key):
    """调用 API 生成编辑图，返回原始图片 bytes（PNG）"""
    prompt = PROMPTS[prompt_key]
    with open(source_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')

    payload = {
        'model': 'qwen-image-2.0',
        'input': {
            'messages': [{
                'role': 'user',
                'content': [
                    {'image': f'data:image/jpeg;base64,{img_b64}'},
                    {'text': prompt}
                ]
            }]
        }
    }

    r = requests.post(API_URL, headers=HEADERS, json=payload, timeout=300)
    if r.status_code != 200:
        raise RuntimeError(f'API error {r.status_code}: {r.text[:300]}')

    data = r.json()
    content = data['output']['choices'][0]['message']['content']
    for item in content:
        if item.get('image'):
            url = item['image']
            if url.startswith('data:'):
                return base64.b64decode(url.split(',', 1)[1])
            else:
                return download_img(url)
    raise RuntimeError('No image in API response: ' + json.dumps(content, ensure_ascii=False))


def load_progress():
    if not os.path.exists(PROGRESS_FILE):
        return set()
    with open(PROGRESS_FILE, 'r') as f:
        return set(line.strip() for line in f if line.strip())


def save_progress(done_set):
    with open(PROGRESS_FILE, 'w') as f:
        for sg in sorted(done_set):
            f.write(sg + '\n')


def main():
    print("=" * 60)
    print("AI 相似图批量生成 v1")
    print("=" * 60)

    mapping = build_sg_mapping()

    # 从 SG_017 开始选，跳过 SG_016（不符合要求）、SG_181（已删除），延伸至 SG_424 凑满 407 组
    selected = []
    skip_sgs = {16, 181}
    for sg in sorted(mapping.keys(), key=lambda x: int(x.split('_')[1])):
        num = int(sg.split('_')[1])
        if num in skip_sgs:
            continue
        if num >= 17:
            selected.append(sg)
        if len(selected) >= 407:
            break

    print(f"选中 {len(selected)} 组: {selected[0]} ~ {selected[-1]}")

    done_set = load_progress()
    print(f"进度: 已完成 {len(done_set)} / 407 组")

    for rank, sg in enumerate(selected):
        new_loan_num = START_LOAN + rank
        new_dir = os.path.join(DATA_DIR, f'loan_{new_loan_num:04d}')
        dest_path = os.path.join(new_dir, 'face_signing.jpg')

        if sg in done_set and os.path.exists(dest_path):
            print(f"  [{rank+1}/407] {sg} → loan_{new_loan_num:04d}/  [skip]")
            continue

        source_rel_path, _ = mapping[sg]
        source_path = resolve_path(source_rel_path)
        prompt_key = get_prompt_key(rank)
        label = {'shirt_bg': '衬衫+背景', 'hair': '仅发型', 'shirt': '仅衣服', 'bg': '仅背景'}[prompt_key]

        if not os.path.exists(source_path):
            print(f"  [{rank+1}/407] {sg} 源图不存在: {source_path} [FAIL]")
            continue

        print(f"  [{rank+1}/407] {sg} → loan_{new_loan_num:04d} [{label}]", end=' ', flush=True)

        try:
            png_data = generate_image(source_path, prompt_key)
            os.makedirs(new_dir, exist_ok=True)

            tmp = dest_path + '.tmp'
            with open(tmp, 'wb') as f:
                f.write(png_data)
            img = Image.open(tmp).convert('RGB')
            img.save(dest_path, quality=87, subsampling=2)
            os.remove(tmp)

            kb = os.path.getsize(dest_path) / 1024
            print(f"[OK] ({kb:.0f}KB)")
            done_set.add(sg)
            save_progress(done_set)

        except Exception as e:
            print(f"[FAIL] {e}")
            # 不标记为已完成，下次续跑会重试
            time.sleep(5)

        time.sleep(1.2)

    print(f"\n{'='*60}")
    print(f"生成完成！成功: {len(done_set)} / 407")
    print(f"{'='*60}")

    # CSV 更新
    print("\n更新 CSV...")
    update_csv(selected, done_set)
    print("全部完成！")


def update_csv(selected, done_set):
    """追加新行到 annotations.csv"""
    with open(CSV_PATH, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    max_img_num = 0
    for row in rows:
        m = re.match(r'IMG_(\d+)', row['image_id'])
        if m:
            max_img_num = max(max_img_num, int(m.group(1)))

    new_rows = []
    for rank, sg in enumerate(selected):
        if sg not in done_set:
            continue
        n = START_LOAN + rank
        max_img_num += 1
        new_rows.append({
            'image_id': f'IMG_{max_img_num:05d}',
            'file_path': f'loan_{n:04d}/face_signing.jpg',
            'image_type': 'face_signing',
            'business_type': '',
            'loan_id': f'LN2024{n:07d}',
            'similar_group': sg,
            'is_similar_pair': '1',
        })

    all_rows = rows + new_rows
    with open(CSV_PATH, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"CSV 已更新 +{len(new_rows)} 行，总计 {len(all_rows)} 行")


if __name__ == '__main__':
    main()
