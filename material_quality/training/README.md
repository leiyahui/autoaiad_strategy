# 素材质量样本流水线

## 当前样本定义

- 当前训练范围：全行业视频素材，范围版本为 `all_industries_v1`；`food_v1` 仅保留为早期食品范围对照。
- 样本粒度：一个 `material_id` 对应一条样本，同一素材跨计划数据合并。
- 起始日期：素材在食品计划中首次产生非零投放指标的自然日。
- 标签窗口：起始日期起前 7 个自然日，左闭右开。
- 原始标签：`label_gmv_7d`，即窗口内累计 GMV。
- 主训练标签：`label_gmv_per_delivery_day`，即窗口内累计GMV除以实际投放自然日数。
- 实际投放不足2天：`label_status=insufficient_delivery_days`，不进入训练。
- 辅助字段：消耗、订单、展现、点击、覆盖计划数，仅用于审计和后续归一化。
- 完整窗口：`label_status=ready`；未满 7 天为 `window_open`。
- 异常标签：完整窗口累计 GMV 小于 0 时标记为 `invalid_negative_gmv`。

样本构造阶段不下载视频、不抽取特征，也不使用商品、账户、预算等字段作为训练特征。

## 数据表

- `material_quality_daily_stat`：按计划、素材、自然日汇总的可重建中间表。
- `material_quality_asset`：素材媒体信息和后续下载、特征处理状态。
- `material_quality_sample`：一素材一条记录的 7 日 GMV 样本。
- `material_quality_sample_context`：全行业样本的主行业和跨行业数量。
- `material_quality_semantic_feature`：中文画面语义向量、OCR文本及版式统计。

## 目录边界

- `sample/`：样本范围筛选、7 日标签聚合和样本落库任务。
- `feature/`：URL刷新、临时下载、媒体探测和多模态特征构造。
- `training/`：数据集快照、滚动时间切分、归一化、训练和评估。
- `tests/`：素材质量流水线测试。
- `../sql/add_material_quality_sample_tables.sql`：样本相关数据库迁移。

## 执行方式

首次使用前执行：

```bash
mysql qianchuan_db < sql/add_material_quality_sample_tables.sql
```

预览食品计划范围：

```bash
python -m material_quality.sample.build_samples --dry-run
```

小批验证：

```bash
python -m material_quality.sample.build_samples --plan-limit 5
```

全量刷新并生成样本：

```bash
nice -n 10 ionice -c 2 -n 7 \
  python -m material_quality.sample.build_samples --continue-on-error
```

构造独立的全行业样本范围：

```bash
nice -n 15 ionice -c 2 -n 7 \
  python -m material_quality.sample.build_samples \
  --cohort all_industries_v1 \
  --all-industries \
  --continue-on-error
```

全行业任务不会写入 `food_v1`。每条样本会在
`material_quality_sample_context` 中保存主行业和跨行业数量，主行业按素材历史GMV、
消耗和覆盖计划贡献依次确定。原始7日GMV保持不变，行业和月份归一化只在训练快照中计算。

不扫描小时表，只根据现有日汇总重新生成标签：

```bash
python -m material_quality.sample.build_samples --skip-daily-refresh
```

每个计划的日汇总删除和重建位于同一事务，任务中断后可以直接重新执行。样本表通过唯一键幂等更新，不会因重复执行产生重复样本。

## 特征演示任务

首次使用前执行：

```bash
mysql qianchuan_db < sql/add_material_quality_feature_table.sql
```

跑通一条素材的URL刷新、临时下载、抽帧、特征落库和临时文件清理：

```bash
python -m material_quality.feature.worker \
  --feature-version media_visual_demo_v1 \
  --limit 1 \
  --scan-limit 20
```

演示版本生成媒体属性和80维低阶视觉统计向量，特征版本为
`media_visual_demo_v1`。它用于验证完整流水线，不等同于后续正式的多模态模型向量。

生产试跑可以指定独立特征版本和成功数量：

```bash
python -m material_quality.feature.worker \
  --feature-version media_visual_stats_v1 \
  --limit 100 \
  --scan-limit 2000
```

同一任务内会缓存用户和广告主的授权检查结果，已过期授权只检查一次，
不会对其关联的每条素材重复尝试Token刷新或素材接口。
当所有候选广告主都无法获取素材时会标记为 `unavailable`，后续批次默认跳过；
客户重新授权后可以使用 `--retry-unavailable` 显式重新尝试。
默认使用素材ID哈希做确定性随机排序，避免优先处理高GMV造成样本偏差；
可以使用 `--label-group positive` 或 `--label-group zero` 分层补充样本。
瞬时限流和接口异常默认冷却30分钟后再试。

按首次统计日期优先补充近期训练、验证或测试样本：

```bash
python -m material_quality.feature.worker \
  --feature-version media_visual_stats_v1 \
  --first-date-from 2026-07-01 \
  --first-date-to 2026-08-11 \
  --limit 300 \
  --scan-limit 8000
```

日期范围只作用于已经完成7日标签窗口的样本，尚未闭合窗口的素材不会提前进入训练集。

随机抽样审计实时URL可获取率，不下载或保存视频：

```bash
python -m material_quality.feature.audit_urls --sample-size 200
```

## 全行业一次性多模态特征

正式版本每个视频均匀抽取最多8个高分辨率关键帧，保存官方Chinese-CLIP RN50
逐帧向量的均值和标准差、画面变化、OCR去重文本及版式、分块文本语义、中文ASR
逐段文本及语义，以及响度、停顿和语速等音频统计。
视频和关键帧都位于独立临时目录，单条素材成功或失败后都会自动清理。
服务器历史任务默认从ModelScope下载官方权重；新的批量抽取不再放到训练数据服务器执行。
进程被强制中断时可能来不及执行上下文清理，后续任务会自动删除超过6小时的遗留目录。

先创建独立数据表并安装Python 3.8兼容的隔离环境：

```bash
mysql qianchuan_db < sql/add_material_quality_semantic_feature_table.sql
python -m venv --system-site-packages .venv-material-quality-semantic
.venv-material-quality-semantic/bin/pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.1.2 torchvision==0.16.2
.venv-material-quality-semantic/bin/pip install \
  -r material_quality/feature/requirements-semantic-py38.txt
```

先用不同主行业的小批素材验证耗时、向量维度、OCR内容和临时目录清理：

```bash
nice -n 15 ionice -c 2 -n 7 \
  .venv-material-quality-semantic/bin/python \
  -m material_quality.feature.semantic_worker \
  --cohort all_industries_v1 \
  --feature-version multimodal_cnclip_whisper_small_v2 \
  --include-window-open \
  --min-delivery-days 2 \
  --retry-unavailable \
  --limit 10 \
  --scan-limit 500
```

正式批量前需要根据小批平均耗时确定并发数。训练数据服务器没有GPU，默认单进程只使用
4个CPU线程，避免和数据库、调度任务争抢资源。模型环境与生产API的Python环境隔离。

## 基线模型训练与评估

训练代码只位于 `training/`，模型和数据快照保存在代码仓库外。第一版使用素材
80维低阶视觉统计向量和媒体属性，不使用消耗、展现、点击、订单或其他投放结果字段。

训练环境必须和生产API环境隔离，并安装Python 3.8兼容依赖：

```bash
python -m venv --system-site-packages .venv-material-quality
.venv-material-quality/bin/pip install \
  -r material_quality/training/requirements-py38.txt
```

执行滚动时间验证和八月锁定测试：

```bash
nice -n 15 ionice -c 2 -n 7 \
  .venv-material-quality/bin/python \
  -m material_quality.training.train_baseline \
  --run-id mq_food_v1_media_stats_v1_20260818
```

任务会冻结当前数据库样本，使用“是否产生GMV分类 + 正GMV对数回归”的两阶段模型，
同时训练直接泊松回归对照模型。模型选择仅使用六月和七月滚动验证，八月测试集不参与调参。

### 多模态V2快速验证

多模态V2使用 `label_gmv_per_delivery_day` 作为标签，只读取投放至少两天、标签窗口
已经闭合的成熟样本。画面语义、画面时序、OCR语义及统计、ASR语义、音频统计和
媒体属性统一在 `training/semantic_dataset.py` 中组装，缺少OCR或ASR的素材使用零向量
并保留模态存在标记。

为避免训练占用业务服务资源，先在训练数据服务器以最低优先级冻结数据快照：

```bash
nice -n 19 ionice -c 2 -n 7 \
  .venv-material-quality-semantic/bin/python \
  -m material_quality.training.train_semantic_baseline \
  --export-snapshot runtime/artifacts/mq_multimodal_v2_snapshot_20260819
```

将快照复制到训练机器后，使用七月数据训练和选择模型，八月样本只做锁定测试：

```bash
python -m material_quality.training.train_semantic_baseline \
  --snapshot-dir /path/to/mq_multimodal_v2_snapshot_20260819 \
  --output-dir /path/to/mq_multimodal_v2_quick_20260819 \
  --test-from 2026-08-01 \
  --validation-days 7 \
  --bootstrap-iterations 200
```

快速验证会比较仅媒体属性、画面语义、画面+OCR、画面+ASR和全模态五组特征，
主要观察时间外测试集的ROC-AUC、PR-AUC、日均GMV秩相关和Top20%提升。产物包括
冻结数据摘要、模型、完整指标、中文报告和逐素材预测。

## 本地离线抽取

生产服务器只保留MySQL和业务服务。视频下载、FFmpeg、OCR、ASR、中文CLIP和训练
都在本地Mac执行。MySQL不开放公网端口，本地通过SSH隧道访问；本地URL解析只读取
数据库中尚未过期的Token，不刷新或修改OAuth授权记录。任务状态、聚合向量和原始
Embedding均写入本地目录，不再更新训练数据服务器的素材质量任务表和特征表。

本地使用独立Python 3.9环境，不修改前后端依赖：

```bash
/opt/homebrew/anaconda3/envs/python39/bin/python -m venv \
  --system-site-packages material_quality/.venv-local
material_quality/.venv-local/bin/pip install \
  -r material_quality/feature/requirements-local-py39.txt
```

本地运行目录为 `material_quality/.runtime/`，其中虚拟环境、模型、SQLite任务库、
原始特征、素材档案、临时视频和停止文件均已忽略，不会进入Git。素材档案按内容
SHA256去重，不永久保留原始MP4或代理视频；最多保留48张最长边640像素的WebP帧、
40kbps单声道Opus音频、逐帧视觉Embedding、OCR/ASR逐块Embedding、聚合向量、
时间轴和关键帧拼图。抽帧优先覆盖开场前5秒的0.5秒间隔画面，再加入明显场景切换
前后帧，并用2秒候选点补齐完整时间线；临时原视频在完整档案原子落盘后删除。数据库
连接通过 `MATERIAL_QUALITY_DATABASE_URL` 配置；如需SSH隧道，由部署环境单独管理，
仓库不保存主机地址、密钥或隧道脚本。

本地V4特征完成后，只使用标签已经闭合且实际投放至少两天的素材冻结训练快照：

```bash
PYTHONPATH=. material_quality/.venv-local/bin/python \
  -m material_quality.training.train_semantic_baseline \
  --cohort all_industries_v1 \
  --feature-version multimodal_cnclip_whisper_replayable_v4 \
  --raw-feature-root material_quality/.runtime/raw_features \
  --export-local-snapshot \
  material_quality/.runtime/training/mq_multimodal_v4/snapshot
```

训练会比较原始高维强正则模型和分模态PCA模型。PCA只在训练时间窗内拟合，画面、
OCR和ASR分别降到64维，低阶统计特征只做标准化，避免验证集或测试集信息泄漏：

```bash
PYTHONPATH=. material_quality/.venv-local/bin/python \
  -m material_quality.training.train_semantic_baseline \
  --snapshot-dir material_quality/.runtime/training/mq_multimodal_v4/snapshot \
  --output-dir material_quality/.runtime/training/mq_multimodal_v4/output \
  --test-from 2026-08-01 \
  --validation-days 7 \
  --bootstrap-iterations 2000
```

## GPT-5.6 Luna脚本理解特征

脚本理解任务只读取内容哈希目录中的关键帧、拼图、OCR、ASR和时间轴，不读取
`assets/*.json`，因此模型输入中不包含GMV、行业、投放日期或素材标签。结果作为
独立Sidecar保存在 `.runtime/script_features/`，不会修改V4原始档案。

当前本地批量任务优先复用Codex CLI现有登录态，不需要另存API密钥。守护程序只从
冻结快照读取1,334个内容哈希，再按哈希读取无标签内容档案；三路并发、每批两条，
已有完整Sidecar自动跳过。首次失败会拆成单条并按指数退避重试，全部完成后自动退出：

```bash
material_quality/.venv-local/bin/python \
  material_quality/feature/codex_script_supervisor.py \
  --daemonize
```

查询进度或安全停止：

```bash
material_quality/.venv-local/bin/python \
  material_quality/feature/codex_script_supervisor.py --status
material_quality/.venv-local/bin/python \
  material_quality/feature/codex_script_supervisor.py --stop
```

守护程序发送给Luna的字段固定为内容哈希、媒体属性、帧时间轴、场景切换时间、OCR、
ASR和关键帧拼图。GMV、训练标签、投放天数和冻结训练矩阵不会进入提示词；Codex子进程
使用只读沙箱和临时会话。实时进度写入 `.runtime/agent_script_auto/status.json`，结构化
原始结果、客观摘要、故事线、还原脚本、钩子、卖点、转场和证据局限完整保存在Sidecar
JSON中，并以本地Chinese-CLIP生成独立2,048维向量。

以下API密钥方式保留作独立备用通道，不与Codex登录态守护程序同时启动。

先在当前终端安全设置 `OPENAI_API_KEY`，不要把密钥写入仓库。使用3条成熟样本验证：

```bash
PYTHONPATH=. material_quality/.venv-local/bin/python \
  -m material_quality.feature.script_agent_worker \
  --script-model gpt-5.6-luna \
  --embedding-model text-embedding-3-large \
  --reasoning-effort low \
  --workers 1 \
  --limit 3
```

小批结构和用量审计通过后，可以提高到4并发。任务按内容哈希幂等保存，完整结果会
自动跳过，接口瞬时失败可直接重跑：

```bash
PYTHONPATH=. material_quality/.venv-local/bin/python \
  -m material_quality.feature.script_agent_worker \
  --workers 4
```

全部1,334条成熟样本完成后，使用同一冻结标签、同一时间切分比较是否增加脚本特征：

```bash
PYTHONPATH=. material_quality/.venv-local/bin/python \
  -m material_quality.training.train_script_ablation \
  --base-snapshot-dir \
  material_quality/.runtime/training/mq_multimodal_v4_20260820_1334/snapshot \
  --script-feature-root material_quality/.runtime/script_features \
  --output-dir \
  material_quality/.runtime/training/mq_multimodal_v4_script_ablation \
  --bootstrap-iterations 2000
```

评估同时比较画面、画面加脚本、现有全模态、现有全模态加脚本和仅脚本五组特征。
脚本向量独立PCA，所有降维和标准化只在训练时间窗拟合；报告还会计算脚本增量的
配对Bootstrap区间。

运行单条演示：

```bash
material_quality/feature/run_local_worker.sh \
  --feature-version multimodal_cnclip_whisper_replayable_v4 \
  --include-window-open \
  --min-delivery-days 2 \
  --first-date-from 2026-07-01 \
  --keyframe-count 48 \
  --limit 1 \
  --scan-limit 100
```

本地任务启动时会先按 `(user_id, advertiser_id, app_key)` 批量读取最新授权。
Refresh Token已过期的账户直接前置跳过；Access Token暂不可用但授权仍有效的账户
只在本地延后，线上刷新Token后可以自动恢复。同一素材关联多个账户时，只要一个账户
可用就继续处理。多个进程通过共享文件按应用统一限流，不会把单进程请求频率叠加。

以本地低优先级启动一致分片；不同终端分别设置分片序号即可并行：

```bash
material_quality/feature/run_local_worker.sh --shard-count 3 --shard-index 0
material_quality/feature/run_local_worker.sh --shard-count 3 --shard-index 1
material_quality/feature/run_local_worker.sh --shard-count 3 --shard-index 2
```

三路任务分别处理哈希分片0、1、2，每路最多成功处理600条、扫描6000条候选，
每进程使用3个PyTorch和ASR线程。三路共享SQLite任务库、跨进程限流文件和安全停止
文件，执行 `stop` 会让全部任务都在当前素材边界退出。首次启用新特征版本时应先使用
命令行运行1条演示，确认原始向量、拼图和临时目录清理均正确后再启动三路任务。

本地SQLite记录待处理、授权延后、URL不可用、特征重试和完成状态。任务始终只保留
当前素材的临时视频，成功或失败后立即清理。可用磁盘低于20GB时停止领取新素材。
需要安全停止时创建 `.runtime/STOP`，任务会在当前素材处理完成后退出。确认退出后删除
该文件才能重新启动。数据库隧道如由部署环境建立，也应由同一部署流程关闭。

## 正式模型训练与本地评分

正式版固定使用画面和Luna脚本两个强正则子模型，各占50%。训练和推理必须使用
`training/requirements-py38.txt` 中锁定的依赖版本：

```bash
PYTHONPATH=. material_quality/.venv-local/bin/python \
  -m material_quality.training.train_serving_model \
  --base-snapshot-dir material_quality/.runtime/training/mq_multimodal_v4_20260821_2524/snapshot \
  --script-feature-root material_quality/.runtime/script_features_luna_consistent \
  --script-feature-version codex_agent_script_cnclip_v1 \
  --model-version mq-quality-v1-20260824-py38 \
  --output-dir material_quality/.runtime/models/mq-quality-v1-20260824-py38
```

对已完成特征抽取的内容哈希进行评分：

```bash
PYTHONPATH=. material_quality/.venv-local/bin/python \
  -m material_quality.inference.predict \
  --model-dir material_quality/.runtime/models/mq-quality-v1-20260824-py38 \
  --raw-feature-root material_quality/.runtime/raw_features \
  --script-feature-root material_quality/.runtime/script_features_luna_consistent \
  --content-sha256 需要评分的视频内容SHA256
```

`quality_score` 是素材相对训练样本的0至100百分位分数，不代表预计GMV金额。
当前只允许用于素材初筛和排序，不直接触发自动投放或停投。

## 与业务后端的边界

用户上传素材的在线任务、API、数据库业务模型和Worker仍保留在 AutoAiAd 主代码库。
本仓库只负责样本、特征、训练、离线评估、正式模型制品和可复现推理逻辑。
