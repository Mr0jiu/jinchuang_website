# 蓝白色业务展示端（含智能检测后端）

## 启动（完整功能，推荐）

```bash
python dashboard_ui/modern_server.py
```

浏览器访问 `http://127.0.0.1:5173`。Tab2「智能检测」的批量检测由 `/api/detect` 提供：
零样本五类识别 → SigLIP2 特征提取（自动使用本地 CUDA GPU）→ FAISS Top-5 检索 → 差异化阈值判定。

依赖（本机已装好，Python 3.11 + CUDA）：

```bash
pip install torch --find-links https://mirrors.aliyun.com/pytorch-wheels/cu128/   # RTX 50 系必须 cu128
pip install transformers pillow pyyaml fastapi uvicorn python-multipart faiss-cpu opencv-python-headless -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

## 路径自适应（本地/云端两栖）

| 配置 | 优先级 |
|---|---|
| 模型权重 | 本地快照 `../siglip2/`（或环境变量 `SIGLIP2_LOCAL_DIR`）→ HF 在线 |
| 数据库 | `dashboard_ui/data.db` → 复用 `web1/data.db` |
| 影像目录 | 环境变量 `DATA_DIR` → 向上搜索 `jinchuang_v4/extracted/data` → 云端 `/workspace/...` 兜底 |
| model_bridge | 同目录 → 自动挂载 `../web1/`（单一源，勿复制两份） |

## 注意

- `serve.py` 是纯静态服务器，**没有 `/api/*` 接口**：用它启动时智能检测、相似组、
  复核等功能全部不可用，仅适合离线预览页面样式。
- FAISS 清单 `face_manifest.csv` 的 loan_id 是影像目录名（loan_001），loans 表主键是
  业务号（LN2024xxxx），`/api/detect` 已做映射统一返回业务号。

## Tab1 指标缓存表（dashboard_*）

Tab1（业务总览）的展示指标由库内面签特征两两相似计算而来，计算量大，因此预计算后
落在 SQLite 三张缓存表中，页面只读缓存，不再使用硬编码兜底值：

| 表 | 内容 |
|---|---|
| `dashboard_overview` | KPI 与面签概况快照（单行）：客户数、贷款数、总影像数（贷款×5）、面签影像数、待复核、涉及订单数、总对数、高相似对/率（≥97%）、同/跨客户对数 |
| `dashboard_similarity_dist` | 相似度分布柱状图：95% / 96% / 97% 三档阈值下的相似对组数（含同/跨客户拆分） |
| `dashboard_loan_behavior` | 贷款行为饼图：同客户复用 / 跨客户复用 / 正常客户三分类贷款数（阈值≥97%，任一同客户对优先归类同客户复用） |

刷新时机：库内 customers / loans / face_feature 计数与缓存不一致时 `/api/stats`
自动重算；「刷新数据」按钮调用 `POST /api/dashboard/refresh`；`/api/rebuild`
重建完成后自动刷新。「最新可疑交易」等明细表仍直接查询 loans + customers 实时返回。
