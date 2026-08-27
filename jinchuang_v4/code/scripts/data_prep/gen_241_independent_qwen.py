"""
gen_241_independent_qwen.py
用 qwen-image-2.0 生成 241 张独立面签照
"""
import base64, csv, json, os, re, sys, time
from PIL import Image
import requests

API_KEY = "sk-ws-H.EHHIYYH.WUBg.MEYCIQClyMTBNJBNI0GawJWkxNEcd-9eDTLl-4gnJ7cJf2BwywIhAJIkLc8u0yTUV7ctmdM1_4W61S9Kex2QDnpM4Yd3cKrK"
HEADERS = {'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'}
API_URL = 'https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation'
DATA_DIR = 'data'
CSV_PATH = 'data/annotations.csv'
PROGRESS_FILE = 'gen_241_independent_progress.txt'
START_LOAN = 3975  # 接在3974之后
COUNT = 241

# 多样的面签提示词（轮换使用）- 强调双人正脸面向镜头
PROMPTS = [
    "生成一张银行面签照片，一位亚洲男性客户和一位银行女柜员在柜台前双人同框，两人都正面面向镜头，五官清晰可见，客户穿深色西装，柜员穿职业装，暖色灯光，写实摄影风格",
    "生成一张银行面签照片，一位亚洲女性客户和一位银行男柜员在柜台前办理业务，两人都正脸看向镜头，面部清晰无遮挡，客户穿浅色外套，柜员穿白衬衫，自然光线，高清写实",
    "生成一张银行面签照片，中年男性客户和女柜员签署贷款文件，两人都面向镜头露出正脸，柜台摆放合同与签字笔，暖黄色室内灯，半身双人构图，真实照片质感",
    "生成一张银行面签照片，年轻客户与银行工作人员在柜台前核对资料，两人都正脸面对镜头，面部完整清晰，客户穿商务休闲装，柜员穿制服，柔和顶光，写实影像",
    "生成一张银行面签照片，男性客户和银行柜员双人合影，两人都正脸面向镜头无遮挡，客户手持证件，背景为银行窗口，暖调光线，高清相机实拍",
    "生成一张银行面签照片，女性客户与男柜员办理业务，两人都正脸朝向镜头，客户穿深色正装，柜台放置文件，室内自然光，双人半身肖像，真实感强",
    "生成一张银行面签照片，男性客户和女柜员在银行服务窗口前，两人都露出正脸看向镜头，五官清晰，客户穿格纹衬衫，暖色灯光，写实照片风格，双人清晰同框",
    "生成一张银行面签照片，中年女客户和男柜员签署协议，两人都正面面对镜头，面部清晰可见，客户戴眼镜，浅灰色柜台背景，柔和光线，双人半身照，高清写实",
]

neg_prompt = "双人客户，卡通，水印，文字，遮挡人脸，畸形手指，白底证件照"


def generate_one(prompt_text, idx):
    """调用 qwen-image-2.0 生成一张图片，返回 JPEG bytes"""
    payload = {
        'model': 'qwen-image-2.0',
        'input': {
            'messages': [{
                'role': 'user',
                'content': [{'text': prompt_text}]
            }]
        }
    }
    r = requests.post(API_URL, headers=HEADERS, json=payload, timeout=300)
    if r.status_code != 200:
        raise RuntimeError(f'API error {r.status_code}: {r.text[:200]}')

    data = r.json()
    content = data['output']['choices'][0]['message']['content']
    for item in content:
        if item.get('image'):
            url = item['image']
            if url.startswith('data:'):
                raw = base64.b64decode(url.split(',', 1)[1])
            else:
                r2 = requests.get(url, timeout=120)
                raw = r2.content
            img = Image.open(io_load(raw)).convert('RGB')
            buf = io_save(img)
            return buf
    raise RuntimeError('No image in response')


import io as _io
def io_load(data):
    return _io.BytesIO(data)
def io_save(img):
    if img.size != (1024, 1024):
        img = img.resize((1024, 1024), Image.LANCZOS)
    # 动态 quality 使文件大小接近 148KB
    for q in range(90, 70, -1):
        buf = _io.BytesIO()
        img.save(buf, format='JPEG', quality=q, subsampling=2)
        if buf.tell() <= 155000:
            return buf.getvalue()
    buf = _io.BytesIO()
    img.save(buf, format='JPEG', quality=75, subsampling=2)
    return buf.getvalue()


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # 已有进度
    done_set = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            done_set = set(line.strip() for line in f if line.strip())

    print(f"目标: {COUNT} 张, 已完成: {len(done_set)}")

    new_rows = []
    with open(CSV_PATH, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        existing = list(reader)

    max_img_id = max(int(re.match(r'IMG_(\d+)', r['image_id']).group(1)) for r in existing if re.match(r'IMG_(\d+)', r['image_id']))

    for i in range(COUNT):
        if str(i) in done_set:
            continue

        prompt = PROMPTS[i % len(PROMPTS)]
        loan_n = START_LOAN + i
        new_dir = os.path.join(DATA_DIR, f'loan_{loan_n:04d}')
        dest = os.path.join(new_dir, 'face_signing.jpg')

        print(f"[{i+1}/{COUNT}] loan_{loan_n:04d} 生成中...", end=' ', flush=True)

        try:
            jpg_data = generate_one(prompt, i)
            os.makedirs(new_dir, exist_ok=True)
            with open(dest, 'wb') as f:
                f.write(jpg_data)
            kb = os.path.getsize(dest) / 1024
            print(f"OK ({kb:.0f}KB)")

            done_set.add(str(i))
            with open(PROGRESS_FILE, 'w') as f:
                for s in sorted(done_set):
                    f.write(s + '\n')

            max_img_id += 1
            new_rows.append({
                'image_id': f'IMG_{max_img_id:05d}',
                'file_path': f'loan_{loan_n:04d}/face_signing.jpg',
                'image_type': 'face_signing',
                'business_type': '',
                'loan_id': f'LN2024{loan_n:07d}',
                'similar_group': '',
                'is_similar_pair': '0',
                'edit_type': '',
            })

        except Exception as e:
            print(f"FAIL: {e}")
            time.sleep(5)

        time.sleep(1.2)

    if new_rows:
        all_rows = existing + new_rows
        type_order = {'bank_statement':0,'contract':1,'id_card_back':2,'id_card_front':3,'face_signing':4}
        all_rows.sort(key=lambda r: (
            int(re.search(r'loan_0*(\d+)', r['file_path']).group(1)) if re.search(r'loan_0*(\d+)', r['file_path']) else 0,
            type_order.get(r['image_type'], 99)
        ))
        for i, r in enumerate(all_rows):
            r['image_id'] = f'IMG_{i+1:05d}'

        with open(CSV_PATH, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

    print(f"\n完成! 新增 {len(new_rows)} 张独立图")
    face = [r for r in all_rows if r['image_type'] == 'face_signing']
    print(f"面签照总计: {len(face)} 张")


if __name__ == '__main__':
    main()
