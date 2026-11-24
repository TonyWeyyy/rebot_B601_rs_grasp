# reBot B601-RS 视觉抓取操作指南（重构版）

> 工程：`~/wey/rebot_developer/rebot_grasp` ｜ conda 环境：`rebotarm`（Python 3.10）
> 机型：RS（`sdk/reBotArm_control_py/config/rebotarm.yaml` → `hardware_yaml: rebotarm_rs.yaml`）
> 当前主力路线：**YOLOE 实例分割 + AnyGrasp 6D 抓取 / 抓取放置 + 链式 IK 预检**

---

## 一、项目架构速览

```text
rebot_grasp/
├── config/
│   ├── default.yaml                  # 全局统一配置（相对路径，含相机/标定/夹爪/AnyGrasp/放置参数）
│   └── calibration/<camera_type>/    # intrinsics.npz、hand_eye.npz
├── core/                             # 公共运行核心层（各脚本共享）
│   ├── robot_session.py              # 机械臂生命周期、安全退出回零、抓取/放置执行状态机
│   ├── ik_planner.py                 # 链式 IK 预检（pre→grasp→retreat / pre_place→place→pre_place）+ 综合评分
│   └── preview_app.py                # 预览帧率统计
├── drivers/
│   ├── camera/                       # base.py（含多帧深度中值融合）、orbbec_gemini2.py、realsense.py（均支持 serial）
│   └── robot/grasp_driver.py         # 夹爪力控状态机、二次收紧、滑落检测 is_holding()、防坠落 stop()
├── calibration/
│   ├── aruco_pose.py                 # ArUco 单次检测+绘制（无重复 PnP）
│   └── hand_eye.py                   # 手眼标定 + 平移/旋转 RMSE 残差评估 + 离群剔除 + 多求解器对比
├── utils/
│   ├── yolo_utils.py                 # YOLO/YOLOE 加载与解析
│   ├── target_selector.py            # 统一目标/容器/关键词筛选
│   ├── pointcloud.py                 # 缓存像素网格的点云构建 + 分割 Mask/BBox 区域引导
│   ├── visualization.py              # 独立 2D/Open3D 可视化（不依赖 GraspNet/pointnet2）
│   ├── grasp_types.py                # 统一 GraspFrameResult 与基座坐标转换
│   ├── anygrasp_utils.py             # AnyGrasp 推理与基座几何安全检查（支持 Mask 引导）
│   ├── graspnet_utils.py             # GraspNet 推理（SDK 按需延迟导入）
│   ├── ordinary_grasp.py             # 2.5D OBB/Mask 短轴抓取（支持 camera_z / ray 进给）
│   └── place_utils.py                # 容器定位（支持 Mask 质心）与放置安全校验
├── scripts/                          # 全部可执行入口（见第三节）
└── tests/                            # 离线单元测试套件（unittest / pytest）
```

---

## 二、每次新开终端 / 重启后必做（3 步）

```bash
# 1. 激活环境并进入项目
conda activate rebotarm
cd ~/wey/rebot_developer/rebot_grasp

# 2. 拉起 CAN 总线（每次插拔或重启后必做）
sudo ip link set can0 down 2>/dev/null || true
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
ip -details link show can0

# 3. 开放 USB 设备权限（相机 + 串口）
sudo chmod a+rw /dev/bus/usb/*/*
sudo chmod 666 /dev/ttyUSB0 2>/dev/null || true
```

**说明**：
- **无需再手动 `export PYTHONPATH`**：代码已内置 `prioritize_env_pinocchio()`，导入机械臂 SDK 前会自动优先使用当前 conda 环境的 `pinocchio`，避免与 `~/.local` 中的 `cmeel` 混库；同时仍能正常加载 `~/.local` 里的 `pyorbbecsdk`。
- 运行抓取脚本的终端**不要 source ROS 环境**（避免 `/opt/ros/...` 的 Python/NumPy 冲突）。
- 运行 AnyGrasp 前可先做一次环境与授权检查（必须同时看到 `License passed: True, state: PASSED` 与 `failures=0`）：
  ```bash
  python scripts/anygrasp_env_check.py --check-license
  ```

---

## 三、常用操作流程与命令速查

### 1. 离线测试与环境检查（不连相机/机械臂）

每次修改代码或配置后，建议先跑一遍离线验证：

```bash
python -m unittest discover -s tests -v     # 核心模块单元测试（25 项）
python scripts/selftest_gripper_tightening.py  # 夹爪力控与收紧回归（9 项）
python scripts/selftest_anygrasp_place.py      # 放置几何与执行安全回归（10 项）
python scripts/anygrasp_selfcheck.py           # AnyGrasp 几何与投影自检（需 graspnetAPI）
```

---

### 2. 手眼标定（Eye-in-Hand，拆装相机/夹爪后重做）

```bash
# 自动模式：遍历 50 个预设位姿，自动输出残差统计、剔除离群帧、对比 5 种求解器并保存原始样本
python scripts/collect_handeye_eih.py --reject-outliers --compare-methods --save-samples

# 手动模式：重力补偿拖动手臂，按 Enter 采集，按 c/q 结束求解
python scripts/collect_handeye_eih.py --manual --reject-outliers --compare-methods
```

- 开始前核对 `config/default.yaml` 中 `calibration.aruco.marker_length_m` 是否等于 ArUco 实际边长（米）。
- 建议采集 ≥15 个不同姿态样本；求解后会打印 `Consistency RMSE: trans=...mm, rot=...deg`，平移 RMSE 越小标定越准。

---

### 3. AnyGrasp 抓取放置（主力推荐：`scripts/anygrasp_place.py`）

**第一步：先跑 Dry-Run 验证检测、Mask 引导、容器定位与全链路 IK**

```bash
# 启动后只移动到 ready 观察位，按 G 规划并检查 IK，不执行抓取/放置/夹爪动作
python scripts/anygrasp_place.py \
  --target-class toy \
  --container-class box \
  --min-base-z 0.05 \
  --dry-run --move-ready
```

**第二步：带二次人工确认运行真机**

```bash
python scripts/anygrasp_place.py \
  --target-class toy \
  --container-class box \
  --min-base-z 0.05 \
  --confirm-motion
```

**常用可选参数**（命令行参数优先级高于 `config/default.yaml`，且 `--target-class` / `--container-class` 会自动注入 YOLOE 开放词汇列表，无需手改 YAML）：

- `--place-height 0.12`：覆盖基座系释放高度（米）
- `--fusion-frames 3`：触发规划时的多帧深度中值融合帧数（`1` 为关闭融合）
- `--extra-yolo-class "red cup"`：追加额外开放词汇检测类别（可重复传入）
- `--yolo-conf 0.20`：覆盖 YOLO 置信度阈值
- `--target-margin-px 12` / `--target-expand-ratio 1.35`：覆盖目标区域扩展参数
- `--camera-type orbbec_gemini2` / `--width` / `--height` / `--fps`：覆盖相机配置
- `--no-open3d`：关闭 Open3D 3D 点云窗口（跳过点云渲染对象构建，更快）

**真机执行完整流程与交互**：
1. 启动后机械臂先移动到 `ready_pose`。
2. 画面中确认目标（绿色框）和容器（蓝色框）稳定后，按 `G` 或 `Space`：
   - 冻结当前机械臂 TCP 位姿，连续采集 `fusion_frames` 帧做深度中值融合。
   - 同帧完成目标与容器定位（容器默认使用分割掩膜质心计算 XY），先做 `pre-place → place → post-release-retract` 三段链式 IK 预检。
   - 用膨胀后的目标分割 Mask 引导 AnyGrasp 推理（点数不足自动回退 bbox），过滤碰撞、宽度、倾角、工作区与最低高度后，做 `pregrasp → grasp → retreat` 三段链式 IK 并综合打分选出最优抓取。
3. 开启 `--confirm-motion` 时：
   - 核对终端打印的 5 个位姿后，输入全大写 **`YES`** 才开始张爪并执行抓取。
   - 若空抓：自动先**垂直回撤到 `retreat`**，再回 `ready`，严禁进入放置。
   - 抓起回撤并回 `ready` 后，系统自动检查 `is_holding()` 确认物体未滑落；确认路径无碰撞后输入全大写 **`PLACE`** 才前往容器释放。
   - 释放物体并等待 `release_wait_s` 后，先**垂直抬升回 `pre-place`** 避开箱壁，再回 `ready` 并合拢夹爪复位。
4. 按 `R` 恢复实时预览；按 `Q` / `Esc` 退出。

---

### 4. AnyGrasp / GraspNet 单物体抓取（`scripts/grasp.py`）

```bash
# 纯相机预览（不连机械臂）
python scripts/anygrasp_camera_demo.py --target-class toy
python scripts/graspnet_camera_demo.py --target-class toy

# AnyGrasp 真机（先 dry-run，再真抓）
python scripts/grasp.py --estimator anygrasp --target-class toy --min-base-z 0.05 --dry-run --move-ready
python scripts/grasp.py --estimator anygrasp --target-class toy --min-base-z 0.05 --confirm-motion

# 临时回退 GraspNet 路线
python scripts/grasp.py --estimator graspnet --target-class toy --dry-run --move-ready
python scripts/grasp.py --estimator graspnet --target-class toy --confirm-motion
```

- CLI 覆盖项（均严格优先于 YAML）：`--min-depth`、`--max-depth`、`--target-expand-ratio`、`--target-margin-px`、`--top-k`、`--fusion-frames`、`--pregrasp-offset`、`--retreat-offset`、`--min-base-z`、`--extra-yolo-class`。

---

### 5. OBB 2.5D 短轴抓取与放置路线（轻量、无需 AnyGrasp/GraspNet）

```bash
python scripts/object_detection.py           # 仅验证相机 + YOLO + 像素点击测距
python scripts/ordinary_grasp_pipeline.py    # 仅验证 OBB/Mask 短轴姿态估计（不连机械臂）

# OBB 单物体抓取
python scripts/main.py --dry-run --move-ready
python scripts/main.py --confirm-motion

# OBB 参数化抓取放置（set.py 统一实现，set2.py 为红色玩具→盒子预设封装）
python scripts/set.py --target-keywords banana --container-keywords box --place-height 0.08 --dry-run --move-ready
python scripts/set.py --target-keywords toy,red --container-keywords box,container,tray,bowl --place-height 0.15 --confirm-motion
python scripts/set2.py --confirm-motion
```

---

## 四、重要安全规则（务必了解）

1. **退出必先平滑回零再断电（防坠落保护）**
   - B601-RS **无机械抱闸**，电机失能会立即自由落体。
   - 因此**无论是否加 `--dry-run`**，只要机械臂电机已使能且不在零位（例如加了 `--move-ready` 抬升到观察位），按 `Q` 或 `Ctrl+C` 退出时程序都会先以平滑轨迹 `safe_home()` 回到零位，再失能断电；驱动层 `GraspDriver.stop()` 也会在关节偏离零位 `> 0.05 rad` 时强制拦截直接断电。
   - `--dry-run` 仅跳过启动移动（未传 `--move-ready` 时）、抓取/放置运动和退出时的夹爪松爪动作；若机械臂本来就停在零位，`safe_home()` 会立即返回不产生运动。
2. **持物取消放置时的处理**
   - 在 `--confirm-motion` 第二步输入非 `PLACE` 取消放置时，物体会继续保持夹持在 `ready` 位，**不会自动松爪**。
   - 此时不要直接按 `Q`（真机退出会执行松爪+回零）；请先人工接住物体，再按 `Q` 退出。
3. **碰撞检测与几何底线不可关闭**
   - AnyGrasp 代码强制 `collision_detection=True`，且所有候选必须同时通过开口宽度、倾角、指尖/抓取点最低 Z、基座工作区和三段链式 IK 检查。

---

## 五、常用调参速查（均在 `config/default.yaml`）

| 症状 / 需求 | 配置路径 | 说明 |
|---|---|---|
| 指定多相机中的某一台 | `camera.serial` | 填相机 SN 字符串；`null` 表示默认第一台 |
| 系统性抓取偏移 | `calibration.hand_eye_compensation_m` | 基座系 `[x, y, z]` 手动微调（米），建议先看标定 RMSE |
| 深度噪点/空洞导致点云抖动 | `grasp_pipeline.grasp.depth_fusion_frames` | 触发规划时中值融合帧数（默认 `3`，设 `1` 关闭） |
| 邻近多物体干扰 AnyGrasp | `anygrasp.use_mask_steering` / `mask_dilate_px` | 默认开启分割 Mask 引导并膨胀 `10 px`；点数不足自动回退 bbox |
| 视野边缘 OBB 抓取倾斜 | `grasp_pipeline.grasp.ordinary_approach_mode` | `camera_z`（默认，垂直光轴）或 `ray`（沿射线） |
| 候选太多导致 IK 检查慢 | `grasp_pipeline.grasp.max_ik_candidates` | 单帧最多送入链式 IK 的候选数（默认 `20`） |
| 抓取点偏高/偏低 | `grasp_pipeline.grasp.insertion_depth_m` / `depth_quantile` | 沿进给方向额外插入深度（默认 `0.02`）/ 深度分位数 |
| 防碰桌面底线 | `grasp_pipeline.grasp.min_base_z_m` | 基座系最低允许 Z（同时约束 pre/grasp/retreat/tip） |
| 容器斜视放置 XY 偏心 | `place.use_mask_centroid` | 默认 `true`，使用有效深度掩膜质心替代矩形框中心 |
| 释放后刮蹭容器壁 | `place.retract_to_pre_place` / `pre_place_offset_m` | 默认释放后先垂直抬升 `8 cm` 再回 ready |
| 调整抓取/放置速度 | `grasp_pipeline.grasp.*_duration_s` / `place.*_duration_s` | 分别控制单抓与抓取放置各段轨迹时长 |
| 柔性夹爪空抓误报/夹不紧 | `robot.gripper.rs.*` | 支持配置 `close_torque`、`default_force`、`post_contact_close_angle`、`hard_stop_angle`、`stall_vel`、`contact_torque_ratio`、`debug` |
| 校准夹爪阈值需看实时数据 | `robot.gripper.rs.debug` | 默认 `false`（避免 500 Hz 循环打印抖动）；校准时临时改 `true` |

---

## 六、常见问题排查

| 现象 | 排查方式 |
|---|---|
| `AnyGrasp create_detector() returned None` / `feature id mismatch` | 运行 `python scripts/anygrasp_env_check.py --feature-id --check-license`，参考 [AnyGrasp通行证问题解决.md](file:///home/tony/wey/rebot_developer/rebot_grasp/AnyGrasp通行证问题解决.md) 核对本机 Feature ID 与 license |
| `anygrasp_env_check.py` 报错或失败 | 确认 `License passed: True, state: PASSED` 且 `failures=0`（已修复返回值漏检问题） |
| `Timestamp anomaly detected` 偶发日志 | Gemini 2 USB 带宽/供电瞬时抖动；偶发且能正常出图不影响运行，频繁掉帧请换主板直连 USB 3.0 |
| 检测不到目标或容器 | 确认画面清晰且目标完整；命令行直接传 `--target-class` / `--container-class` 即可自动加入开放词汇列表 |
| Place IK 失败 | 检查 `place.orientation_rpy`（已验证 `[0.0, 0.7, 0.0]`）与容器摆放位置是否在工作区内，不要放宽安全限制 |
| `can0` 打不开 | 执行 `ip link show` 确认 USB2CAN 接入后重跑第二节 CAN 初始化命令 |
