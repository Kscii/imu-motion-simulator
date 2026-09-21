# 运动与传感器合同

## canonical motion-v2

内部 HDF5 根沿用 `ims_schema_version=1.0.0`、`artifact_kind=motion` 和 UUID `artifact_id`。`kind_metadata.motion_contract_version=2` 固定 SMPL+H 语义。

| 数组 | dtype / shape | 含义 |
|---|---|---|
| `time_ns` | int64 `[T]` | 来源帧零起点的有理时钟 |
| `root_position_m` | float64 `[T,3]` | canonical world 中 pelvis 位置 |
| `root_quaternion_wxyz` | float64 `[T,4]` | 根主动四元数 |
| `joint_local_quaternion_wxyz` | float64 `[T,52,4]` | 固定 SMPL+H 次序的局部旋转；第 0 列与 root quaternion 相同 |
| `betas` | float64 `[16]` | 来源人物形状 |
| `dmpls` | float64 `[T,8]` | 原生来源的逐帧动态形变；缺失来源为全零占位 |
| `valid` | bool `[T]` | 该帧是否可消费 |

坐标为右手 X 前、Y 左、Z 上，长度 m，时间 ns。所有浮点有限、四元数单位化、时间严格按 metadata 中的整数 numerator/denominator 计算。source dataset/member、gender、原帧率、模型 member hash 和原 archive hash 必须保存。

`resolved_config.dynamic_shape` 在新产物中固定 `{source_available,effective_policy,components}`。原生 AMASS 为 `source/8`；GRAB/SOMA Stage-II 为 `disabled-zero/8`，此时合同强制 `dmpls` 全零。motion 不保存全帧顶点。消费者只在 `source_available=true` 时加载 DMPL 基；bone-rigid 消费者只需形状化静止关节与 FK。

下载的拟合参数保持原样。可见但不破坏结构、有限值、时钟或 IMU 收敛的局部时间不连续写入 review revision 的 `quality_flags`，scope 为 `downloaded-amass-fit`、disposition 为 `advisory`；它不改写 motion，也不自动阻塞导出。

## selection-v1

selection 是不可变 JSON：`selection_id`、父 `motion_id`/SHA-256、`[start_frame,stop_frame)` 和候选标签。它不能跨出父范围，也不复制运动数组。BABEL 标签保存为 candidate；人工 revision 可以修改。

## sensors-v2

`artifact_kind=sensors`，`kind_metadata.sensor_contract_version=2`。

| 数组 | dtype / shape |
|---|---|
| `time_ns` | int64 `[N]` |
| `specific_force_m_s2` | float64 `[N,M,3]` |
| `angular_velocity_rad_s` | float64 `[N,M,3]` |
| `valid` | bool `[N,M]` |

metadata 必须包含父 motion、可选 selection、layout ID、有序 mount IDs、profile ID 与 `ideal/calibrated` variant。layout 和 profile 内容及 SHA-256 都写入血缘，不存在隐藏默认。

理想流保留重力响应，使用刚性安装、连续轨迹导数、显式低通与固定 25 Hz 输出网格。profile v3 将内部工作网格解析为不低于配置下限的来源帧率整数倍，保存实际解析频率，并与两倍频率做收敛检查；它也显式记录片段边界保护区。短到无法在保护区之间留下样本的片段仍保存 motion 与 IMU，但必须记录 `boundary_guard_complete=false` 并自动 QA 失败，不能升级为收敛证据。边界估计仍保留在理想流中，其延拓假设写入 provenance。校准流只能从理想 sensors 派生，必须记录 profile、父 hash 和随机 seed；不得改写理想文件。

## 表面传感器候选

surface-DMPL 会把安装点绑定到表面三角形/重心坐标，并用包含 pose correctives 和 DMPL 的顶点轨迹求导。它只有在至少两个现实配对数据集上满足下列门槛才成为默认：加速度中位误差改善不低于 5%，陀螺中位误差恶化不超过 2%，有效率不下降。否则保持 bone-rigid。
