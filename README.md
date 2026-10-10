# AutoAiAd Strategy

AutoAiAd Strategy 是 AutoAiAd 素材质量评分体系的独立训练仓库，包含从样本构造、多模态特征抽取、
时间外验证、模型训练到正式模型制品校验的完整代码。质量分用于素材初筛和排序，不直接
触发自动投放或停投。

## 仓库内容

- `material_quality/sample/`：7日GMV标签、样本范围与样本落库逻辑。
- `material_quality/feature/`：视频、画面、OCR、ASR、脚本及语义特征抽取。
- `material_quality/training/`：数据集冻结、滚动时间切分、消融实验、分类器和正式模型训练。
- `material_quality/inference/`：特征Schema、模型加载、Smoke Test和评分入口。
- `material_quality/sql/`：训练样本及特征表结构。
- `material_quality/tests/`：样本、特征、训练和推理测试。
- `material_quality/models/`：模型卡和版本清单；权重与固定测试特征由本地训练生成。

当前随仓库保存说明的模型版本为 `mq-quality-v1-20260824-py38`，训练范围、指标、依赖和限制
见对应目录中的 `MODEL_CARD.md` 与 `manifest.json`。

## 安全边界

本仓库公开保存代码、表结构、测试和模型说明；不上传模型权重及固定测试特征。以下内容禁止提交：

- 用户视频、关键帧、OCR/ASR原文和训练样本。
- 数据库快照、特征向量、预测结果和业务日志。
- API Key、数据库密码、OAuth Token、SSH配置和隧道脚本。
- Chinese-CLIP、Whisper、OCR等第三方大模型权重及下载缓存。
- 本机虚拟环境、运行状态、临时文件和LaunchAgent配置。

这些内容已由根目录 `.gitignore` 排除，但提交前仍需执行敏感信息扫描。

## 环境准备

基础训练环境兼容 Python 3.8：

```bash
python3.8 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

需要完整多模态特征抽取时，再安装对应环境：

```bash
.venv/bin/pip install -r material_quality/feature/requirements-semantic-py38.txt
```

复制脱敏配置模板，并只在本机填写真实配置：

```bash
cp material_quality/.env.example material_quality/.env.local
```

数据库连接只接受 `MATERIAL_QUALITY_DATABASE_URL`。未配置时默认使用内存SQLite，便于运行
纯计算测试，但不能执行样本构造或数据库评估任务。

## 测试

```bash
PYTHONPATH=. .venv/bin/python -m unittest discover -s material_quality/tests -t .
```

## 训练流程

1. 执行 `material_quality/sql/` 中的迁移并构造7日标签样本。
2. 抽取画面、OCR、ASR、音频和脚本特征，产物写入被忽略的运行目录。
3. 导出不可变数据集快照，按时间滚动验证并完成消融实验。
4. 使用全部成熟样本训练正式双塔融合模型。
5. 生成 `model.joblib`、`manifest.json`、`smoke_test.npz` 和 `MODEL_CARD.md`。
6. 在锁定的 Python 3.8 推理环境执行摘要校验和 Smoke Test 后发布。

正式模型训练示例：

```bash
PYTHONPATH=. python -m material_quality.training.train_serving_model \
  --base-snapshot-dir runtime/snapshots/base \
  --script-feature-root runtime/script_features \
  --script-feature-version codex_agent_script_cnclip_v1 \
  --output-dir runtime/models/mq-quality-next \
  --model-version mq-quality-next
```

更完整的历史训练口径和命令见 `material_quality/training/README.md`。

## 独立运行说明

纯训练可直接使用本地冻结快照，不需要业务后端。真实训练样本和特征须自行在本地准备。
本地抽取使用 `feature/run_local_worker.sh`，其中 `local_readonly` 模式不依赖主项目的千川模块。
历史服务器模式的 `feature/worker.py`、`feature/audit_urls.py` 和服务器 URL 解析仍依赖
主项目 `chuan_api`；如需使用，须额外提供主项目模块和授权配置。

直接运行 Python 命令前，需要自行导出 `.env.local` 中的配置；只有本地 Worker 启动脚本
会加载该文件。完整多模态抽取另需 FFmpeg 和对应第三方模型权重。

公开仓库不带 `model.joblib` 和 `smoke_test.npz`，对应正式制品测试会明确跳过；
将本地制品放回模型版本目录后，该测试会自动执行摘要校验和预测重放。
