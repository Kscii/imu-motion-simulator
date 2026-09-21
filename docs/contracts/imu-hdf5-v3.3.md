# IMU HDF5 3.3.0 交付合同

HDF5 3.3 有两种互不混用的合成数据制品：正式 `imu_dataset` 是人工审核与正式标签后的 `training_only` 快照；`imu_dataset_provisional` 是仅机器 QA 通过的 `unverified_synthetic` 核心数据。仅前者是本项目的正式接受输出。

## 核心

必需根对象：`/samples`、`/sequences`、`/annotations`、`/labels`。可选：`/media`、`/replay`、`/assets`、`/provenance`。未知根对象拒绝。

必需属性包括 `imu_schema_version=3.3.0`、`artifact_profile=imu_dataset`、UUID `artifact_id`、`dataset_id`、`sampling_rate_hz=25.0`、`axis_frame=sensor_local`、`evaluation_role=training_only`、计数和 `logical_content_sha256`。

### `/samples`

float32 `[N,6]`，列顺序固定：

```text
acceleration_x_mps2 acceleration_y_mps2 acceleration_z_mps2
angular_velocity_x_rad_s angular_velocity_y_rad_s angular_velocity_z_rad_s
```

加速度是含重力响应的 sensor-local specific force。所有值有限。

### `/sequences`

字段顺序保持与 3.2 相同：

```text
sample_start int64, sample_stop int64,
source_file UTF-8, participant_id UTF-8, recording_id UTF-8,
body_location UTF-8, activity_code UTF-8, is_fall bool,
supervision_kind UTF-8, source_sampling_rate_hz float64
```

所有半开样本区间非空、连续、不重叠，并恰好覆盖 `/samples`。多安装点各自成为一条 sequence；它们可共享同一个 motion/selection 和 replay record。

### `/annotations` 与 `/labels`

annotation 字段为 `sequence_index, kind, start_sample, stop_sample, code`。索引相对该 sequence；activity/exclude 为半开区间，onset/impact 为点。

`/labels/catalog` 冻结 taxonomy/version/code/name/is_fall/active；`/labels/sequence_versions` 恰好覆盖全部序列。BABEL 候选不能直接导出；最新人工 review revision 必须 accepted 且标签可以由 taxonomy 解析。

## 可选 MP4

`/media/index` 可只覆盖部分 sequence。同一个多传感器动作默认只在第一条 sequence 嵌入一份 MP4。原视频字节是连续、无 HDF5 压缩的 uint8；SHA-256、物理 offset、长度、容器和时长必须闭合。

`/media/timing/<sequence_index>` 为严格递增的 `[recording_time_ns, media_time_ns]` 映射。查看器只在映射范围内插值，不外推。

## 可选回放

3.3 支持两种 record：

- 旧物理回放：根位姿和标量 `joint_position`，引用冻结 URDF 模型包；
- kinematic replay-v2：原生 DMPL 来源；根位姿、`joint_local_quaternion_wxyz [F,52,4]`、`betas [16]`、`dmpls [F,8]`，model family 固定为 SMPL+H；
- kinematic replay-v3：Stage-II 无 DMPL 来源；数组形状不变，metadata 明确 `disabled-zero` 且验证器强制 DMPL 全零。

回放保留 motion 原时钟，不复制成 25 Hz。一个 record 可被多个安装点 sequence 引用，且必须覆盖其所有 25 Hz 样本。带 replay 必带 `/assets`；v2 自包含 SMPL+H 和 DMPL 原件，v3 只包含 SMPL+H，均按 hash 去重并禁止网络或外部 HDF5 引用。

回放 float32 只用于展示；motion-v2 的 float64 原件仍是 IMU 重算事实。

## provenance 与不可变性

合成文件必须记录每个 sequence 的 motion、sensors、review artifact UUID/SHA-256、route、model hash、来源分组键和 review 中的来源质量标记。文件不能把来源受试者冒充合成人体 participant。

逻辑核心 hash 沿用 3.2 算法，只覆盖规范化 sequence 元数据、局部 annotation 和 float32 samples；labels 文本、media、replay 与 assets 不参与核心 hash。附件或包装变化仍产生新的完整文件身份与外部完整 SHA-256。

写入唯一临时 H5，关闭重开后执行全结构、时钟、语义、hash、附件和资产验证，再不可覆盖地发布。旧 3.2 文件只通过显式迁移产生新 3.3 副本，不原地修改。

## 接受门槛

1. motion、selection、sensors、review 和可选 MP4 hashes 属于同一血缘；
2. 最新 review revision 为 accepted，含 reviewer、reason 和可解析标签；
3. 六轴有限且 valid，全序列 25 Hz 网格闭合；
4. replay/MP4 的时间覆盖和可见传感器与同一 layout 一致；
5. 输出通过 `imu-sim contracts validate FILE.h5` 的完整验证。

## 平台候选与快照

机器候选清单不是交付快照。它只列出通过硬门槛的 motion/sensors/selection 对象 hash、`synthetic-motion/v1/objects/...` key、候选标签和非阻塞 warning。人工质量 revision 使用独立的 `pass|reject` 决定和前一 revision hash；质量 pass 可以暂时没有正式标签，reject 必须提供理由。进入快照时必须再核验最新质量 pass 与已解析的正式标签。

快照只收录最新 human-pass 候选。未审核和 reject 项具名排除，旧快照不回写。快照 manifest 固定 `hdf5_version=3.3.0`，每个 shard 不超过 4 GiB，必须含 replay、不得含 MP4；完整验证会重新核对 shard 的字节数、SHA-256 和 HDF5 合同。`imu-sim handoff fixtures` 生成两条候选（一条 pass、一条 reject）的可运行本地样例，`handoff snapshot` 从候选、revision 与现成 HDF5 3.3 shard 构造新快照。它们不执行云上传。

## 未审核核心数据导出

`imu-sim production export-core JOB.yaml [OTHER_JOB.yaml ...]` 从一份或多份共享发布前缀的生产账本中冻结当时**完整发布且机器 QA 通过**的候选，生成独立的不可变部分或完整导出。它不读取人工 pass/reject，不进入训练快照页；新到达的候选只能产生新版本。正式 prod 的 22 来源完整版本必须同时提供原生与 Stage-II 两份配置及 `--audit-report`，两份任务和最终云端审计全部完成后才标记 `coverage=complete`；单份 prod 账本即使结束也只是部分版。导出冻结来源任务配置哈希、云端审计哈希与候选提交集合哈希。对象位于当前 dev/prod 命名空间的 `datasets/provisional/exports/<export_id>/`，H5 与含大小、SHA-256、规则版本和覆盖状态的 manifest 分开存储，manifest 最后写入。

这类 H5 仍保留六轴 `/samples`、`/sequences` 和空 `/annotations`，但**没有正式 `/labels`、replay、模型资产或 MP4**。`/sequences.activity_code` 固定为 `unverified`，布尔 `is_fall=false` 只是原有核心字段的占位值，不能作为非跌倒真值。每条序列的 `candidate_id`、`version_id` 和提交哈希写在 `/candidate_index`，可据此在平台打开精确回放。规则产生的建议另存于 `/weak_labels/index`，状态只能为 `weak` 或 `unresolved`；`/weak_labels/source_candidates` 原样保留来源候选。`/provenance/metadata` 冻结规则正文、规则 SHA-256、提交集合哈希及局限。同一来源标签同时命中全局规则和具名来源规则时，具名来源规则优先；未知、冲突或时间标签不一致时保持未解析，任何跌倒都不自动确认为真。

3.2 真实 IMU 文件及现有正式 3.3 `imu_dataset` 合同不变；旧版本读取器会拒绝新的 provisional profile。使用者必须同时检查 `artifact_profile` 与 `evaluation_role`，不可把弱标签当成人工确认标签。正式可验证数据仍须在平台完成质量审核、正式标签和训练快照。

### 本地临时弱标签预设

`configs/production/local-activity-presets-v1.json` 是人工维护的**本地试用**词表，不覆盖平台受控规则，也不是人工审核结果。`tools/compile_local_activity_presets.py` 只读已发布候选的本地 outbox，把 BABEL 类别组合、明确的 Stage-II 来源动作名和少量文件名动作词编译为版本固定的精确匹配规则，并先报告全库命中数。文件名只在没有可用来源类别、动作词唯一且不含跌倒/恢复风险词时使用；`contains_*` 表示片段中有来源时间段证据，不声称全片只做该动作。未匹配或冲突的片段仍为 `unresolved`，但六轴、来源候选和回放定位索引完整保留。

`production export-core --rules-file` 可以用冻结规则直接生成另一个不可变 provisional 版本；此路径会写入原目标发布存储。若只需要本机文件，`tools/relabel_local_core_after_benchmark.py` 从**已完成全量审计的既有核心 H5**复制完全相同的六轴与候选身份，只重算独立弱标签并写入本地新 HDF5 3.3，输出清单明确 `cloud_published=false`。两种方式都不把 `unverified`、`is_fall=false` 占位值变成 ADL 或已核验非跌倒真值。
