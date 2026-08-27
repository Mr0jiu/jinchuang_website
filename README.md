# 金创 · 金融影像智能相似度检测系统

> **当前网站（2026-08 起）：`dashboard_ui/`** —— `modern_server.py`（FastAPI 后端）+ `index.html`（前端单页）。启动 `python dashboard_ui/modern_server.py` 后访问 http://127.0.0.1:5173 ，依赖与说明见 `dashboard_ui/README.md`。

本仓库包含网站运行所需的全部文件：`dashboard_ui/`（当前网站）、`web1/`（仅保留 `model_bridge.py` 模型桥接层 + `data.db` 数据库，由 modern_server 共用）、`siglip2/`（SigLIP2 模型配置/tokenizer）、`jinchuang_v4/`（模型代码 + FAISS 索引 + 图片数据）、`outputs/same_customer_experiment/`（Tab5 同客户区分实验数据与报告）。

> ⚠️ **Git 仓库不含模型权重**：`siglip2/model.safetensors`（1.5G）超出 GitHub 单文件 100MB 限制，已写进 `.gitignore` 不入库。仓库内只保留模型配置/tokenizer（`siglip2/` 下其余 7 个小文件）。获取模型权重见下方「让 SigLIP2 模型本地可加载」。其余（代码、数据库、FAISS 索引、3254 条虚构图片数据）均在仓库内。

## ✅ 下载校验（2026-08-10）

| 项 | 结果 |
|---|---|
| `siglip2/model.safetensors` | size=1500800904 ✓，sha256=`612923381...6a0b` 与云端逐字节一致 ✓ |
| `web1/data.db` | size=10059776 ✓ |
| `outputs/mvp/face_signing.faiss` | size=9996333 ✓ |
| `extracted/data/` | 3256 条 = 3254 个 `loan_*` + 2 个 annotations CSV ✓（与云端条数一致）；loan_001 五张 jpg 字节数与云端逐一相符 ✓ |
| 路径自解析 | `web1/` 与 `jinchuang_v4/` 同级 → `_find_sibling` 能定位 `code/` 与 `extracted/` ✓ |
| 本地总大小 | ~3.3G（模型 1.5G + 图片 1.9G + 代码/DB/索引） |

## 目录结构

```
jinchuang_website/
├── dashboard_ui/                  # ★ 当前网站（FastAPI 后端 + 前端单页）
│   ├── modern_server.py           # 后端服务（python dashboard_ui/modern_server.py 启动）
│   ├── index.html                 # 前端单页（7 个 Tab：总览/检测/查询/策略/同客户/记录/报告）
│   ├── evidence_bridge.py         # 26 维证据计算
│   ├── refresh_stats.py           # 首页缓存表重算
│   ├── backfill_loans.py / backfill_business_type.py   # 数据回填工具
│   └── README.md                  # 启动与依赖说明
├── web1/                          # 运行时共用件（modern_server 自动从这里定位）
│   ├── model_bridge.py            # 模型桥接层（懒加载 SigLIP2 + 分类器 + FAISS）
│   └── data.db                    # ★ 唯一数据库（loans/customers/reviews/users…）
├── siglip2/                       # SigLIP2 模型（扁平目录副本，权重不入库）
│   ├── config.json / preprocessor_config.json
│   ├── tokenizer.json / tokenizer.model / tokenizer_config.json / special_tokens_map.json
│   └── README.md
├── outputs/same_customer_experiment/   # Tab5 同客户区分实验（results.json + Word 报告）
└── jinchuang_v4/
    ├── code/                      # = MODEL_CODE_DIR（model_bridge 通过 _find_sibling 定位）
    │   ├── config.yaml            # 模型/检索/分类器 prompt 配置
    │   ├── api.py / ingest.py / main.py / requirements.txt
    │   ├── src/                   # model.py / classifier.py / retrieval.py / ...
    │   ├── mvp/pipeline.py        # FAISS 建库脚本
    │   ├── scripts/               # 两阶段管线脚本
    │   └── outputs/mvp/           # ★ 运行时产物（仅下载了网站实际读取的文件）
    │       ├── face_signing.faiss        # FAISS 索引（3254 条）
    │       ├── face_manifest.csv         # 索引元数据
    │       ├── classifier.pt             # 分类头权重
    │       ├── classification_metrics.json / two_stage_summary.json / fraud_monitoring_summary.json / run_summary.json
    └── extracted/                 # = DATA_DIR（图片数据）
        └── data/                  # 3254 个 loan_* 目录 + annotations*.csv，每个 loan 含 5 张 jpg
```

## 路径自解析（无需改代码即可本地跑）

`model_bridge.py` 用 `_find_sibling` 从 `web1/` 向上找 `jinchuang_v4` 兄弟目录。本布局里
`web1/` 与 `jinchuang_v4/` 同属 `jinchuang_website/`，故：
- `MODEL_CODE_DIR` = `jinchuang_website/jinchuang_v4/code` ✓
- `DATA_DIR`       = `jinchuang_website/jinchuang_v4/extracted` ✓
- FAISS 索引       = `MODEL_CODE_DIR/outputs/mvp/face_signing.faiss` ✓

也可用环境变量覆盖：`MODEL_CODE_DIR` / `DATA_DIR` / `FAISS_INDEX`。

## 让 SigLIP2 模型本地可加载

`src/model.py` 里 `DEFAULT_MODEL_NAME = "google/siglip2-base-patch16-224"`，`AutoModel.from_pretrained(model_name)`
在离线模式下从 HF 缓存读取。本地拿到的是扁平 `siglip2/` 目录，三种接法任选其一：

1. **改模型名指向本地目录**（最简单）：把 `src/model.py` 的 `DEFAULT_MODEL_NAME` 与 `jinchuang_v4/code/config.yaml` 的 `model.name` 改成本地 `siglip2/` 的绝对路径，例如 `C:/研1/金创/jinchuang_website/siglip2`。
2. **重建 HF 缓存**：在 `~/.cache/huggingface/hub/models--google--siglip2-base-patch16-224/snapshots/<任意hash>/` 下放好上述 7 个文件（用真实文件，不要软链），并写 `refs/main` 内容为该 hash。
3. 设 `HF_HOME` 指向自建缓存目录。

> 注意：云端 `model.safetensors` 经核对**天生是 siglip 架构**（patch embedding 为卷积），只能用 `SiglipModel` 加载；名字里的 "siglip2" 指训练方案，非新架构。详见 `src/model.py`。

## 启动命令

```bash
python dashboard_ui/modern_server.py     # 浏览器访问 http://127.0.0.1:5173
```

需要 SigLIP2 实时推理（Tab2 智能检测）时带上离线环境变量（云端四环境变量，本地同样需要）：

```bash
ALLOW_LOCAL_MODEL=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python dashboard_ui/modern_server.py
```

- `ALLOW_LOCAL_MODEL=1`：开启 SigLIP2 实时推理（懒加载，首张上传图才载入模型）。
- `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`：绕开 protobuf 新版与 TF `_pb2` 冲突。
- `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1`：离线读缓存，避免对 huggingface.co 反复重试。

## 数据库

- `data.db` schema：`loans` 表用 `verify_status`（N/R/F/B/C）一个字段承载自动验证 + 人工复核，`auto_group` 存网页自算相似组号。

## 未下载的内容（可按需补取）

- `outputs/mvp/` 的大体积分析产物：`image_embeddings.npy`(49M)、`face_embeddings.npy`、`fraud_monitoring.csv`(18M)、`stage1_similarity_report.csv`(15M) 等 —— 网站运行时不读，仅训练/分析用。
- `jinchuang_v4/code/experiments/`、各 `*.bak.*` 备份、`run.log*`、`__pycache__/`。
- 云端 `/workspace/website/`（老版）、`/workspace/siglip2/siglip2-so400m-patch16-384/`（不完整）。
