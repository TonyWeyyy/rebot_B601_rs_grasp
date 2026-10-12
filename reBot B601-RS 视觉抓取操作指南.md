# reBot B601-RS 视觉抓取（方式一）操作指南

> 工程：`~/wey/rebot_developer/rebot_grasp`，conda 环境：`rebotarm`（Python 3.10）
> 机型：RS（`sdk/reBotArm_control_py/config/rebotarm.yaml` → `hardware_yaml: rebotarm_rs.yaml`）

---

## 一、每次新开终端必做（3 步）

```bash
# 1. 激活环境
conda activate rebotarm
cd ~/wey/rebot_developer/rebot_grasp

# 2. 拉起 CAN 总线（每次插拔/重启后都要做）
sudo ip link set can0 down 2>/dev/null
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up

# 3. USB 设备权限（相机 + 机械臂串口）
sudo chmod a+rw /dev/bus/usb/*/*
sudo chmod 666 /dev/ttyUSB0
```

**注意**：
- 运行抓取脚本的终端**不要 source ROS 环境**（`/opt/ros/humble` 的 numpy 可能和 conda 环境冲突）。如果报 numpy 相关错误，换一个没 source 过 ROS 的干净终端。
- 编译 pointnet2/knn 用的 `CPATH`/`LD_LIBRARY_PATH` 等 export 只在**编译时**需要，日常运行不需要。如果重开终端后要重新编译，先执行：
  ```bash
  export NVCUDA=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cu13
  export CPATH=$NVCUDA/include:$CPATH
  export CPLUS_INCLUDE_PATH=$NVCUDA/include:$CPLUS_INCLUDE_PATH
  export LIBRARY_PATH=$NVCUDA/lib:$LIBRARY_PATH
  export LD_LIBRARY_PATH=$NVCUDA/lib:$LD_LIBRARY_PATH
  ```

---

## 二、一次性配置清单（已完成，不用重做）

| 项目 | 状态 | 位置 |
|---|---|---|
| conda 环境 `rebotarm` + 依赖 | ✅ | — |
| 机械臂 SDK | ✅ | `sdk/reBotArm_control_py`（`pip install -e .`） |
| YOLOE 权重 | ✅ | `models/yoloe-26l-seg.pt` |
| pointnet2 / knn CUDA 扩展 | ✅ | site-packages（含手动补拷的 3 个 .py + `__init__.py`） |
| GraspNet 权重 | ✅ | `sdk/graspnet-baseline/checkpoints/checkpoint-rs.tar` |
| 手眼标定 | ✅ | `config/calibration/<camera_type>/hand_eye.npz` |
| grasp_driver.py 柔性夹爪修复 | ✅ | `drivers/robot/grasp_driver.py`（见附录 C） |

**只有以下情况才需要重做**：
- 手眼标定：相机/夹爪拆装过、抓取系统性偏移变大 → 重跑 `scripts/collect_handeye_eih.py`
- CUDA 扩展：torch 升级后 → 重新 `pip install . --no-build-isolation`（先配好上面的 export）

---

## 三、常用命令速查

### 手眼标定（Eye-in-Hand）
```bash
python scripts/collect_handeye_eih.py          # 自动模式：遍历 50 个预设位姿
python scripts/collect_handeye_eih.py --manual # 手动模式：推末端采样，Enter 采，c/q 结束
```
- 开始前确认 `config/default.yaml` 的 `calibration.aruco.marker_length_m` = ArUco 实际边长
- 最少 5 个样本，建议 ≥15 个

### OBB 路线（默认，自上而下抓取）
```bash
python scripts/object_detection.py           # 只验证相机 + YOLO
python scripts/ordinary_grasp_pipeline.py    # 只验证抓取姿态估计（不连机械臂）
python scripts/main.py --dry-run             # 只打印位姿，不动机械臂
python scripts/main.py                       # 完整抓取
python scripts/set.py                        # 抓取并放置（按 Q 退出自动回零）
```

### GraspNet 路线（6D 姿态，可斜夹/侧夹）
```bash
python scripts/graspnet_camera_demo.py                   # 纯相机 Demo，先跑这个
python scripts/grasp.py --dry-run --target-class "toy"   # dry-run 验证位姿
python scripts/grasp.py --target-class "toy"             # 真抓
```
按键：`G`/`Space` 推理当前帧，`R` 恢复预览，`Q`/`Esc` 退出。

---

## 四、调参速查（都在 `config/default.yaml`，除非另注）

| 症状 | 参数 | 说明 |
|---|---|---|
| 抓取系统性偏高/偏低/偏左/偏右 | `calibration.hand_eye_compensation_m: [x, y, z]` | 手眼标定手动补偿，X 前后、Y 左右、Z 高低，1cm 量级微调 |
| 抓取点落在物体太靠上/下沿 | `grasp_pipeline.grasp.depth_quantile` | 抓取深度取点云的分位数 |
| 预抓取接近高度 | `grasp_pipeline.grasp.pregrasp_offset_m` | **只影响接近轨迹，不影响最终抓取高度** |
| 防碰桌面底线 | `grasp_pipeline.grasp.min_base_z_m` | 基座系最低抓取 Z，设桌面高度 +1cm |
| GraspNet 夹不实/插得不够深 | `grasp_pipeline.grasp.insertion_depth_m` | 沿进给方向额外插入深度（默认 0.015） |
| 夹不紧留缝 | `robot.gripper.rs.close_torque` / `default_force` | 闭合力矩 / 保持力矩；柔性夹爪建议 close_torque 0.5~0.6、default_force 0.4 |
| 换检测目标 | `yolo.custom_classes` | 开放词汇类别名列表 |
| 预备位 | `robot.ready_pose` | 机械臂观察/准备位姿 |

---

## 五、常见问题

| 现象 | 排查 |
|---|---|
| 按 G 不抓 | 手眼标定文件不存在 / IK 不可达 → 先 `--dry-run` |
| 误报 Empty grasp | 看附录 C；柔性夹爪阈值已校准，换新物体后复发就把 `[gripper]` 调试打印发给助手重新定阈值 |
| 相机丢帧 / Timestamp anomaly | USB 供电或带宽不足，换主板直连 USB 3.0 口，别过 Hub |
| `Invalid k1=42.37` 警告 | SDK 畸变系数兼容问题，按零畸变处理，不致命 |
| numpy / ROS 相关报错 | 终端 source 过 ROS，换干净终端 |
| pointnet2 import 报错 | torch 升级后需重编译扩展（先 export 路径，见第一节） |
| 容器 (box) 检测置信度低 | 改善光照/背景对比；把 `custom_classes` 里容器描述写具体（如 `cardboard box`） |

---

## 附录 A：GraspNet CUDA 扩展修复记录（2026-10-12）

- nvcc 必须和 torch 的 CUDA 版本一致（13.0）：`conda install -c nvidia cuda-nvcc=13.0`
- 头文件不在 conda targets 目录，在 pip wheel 自带的 `$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cu13/include`，通过 `CPATH` 等 export 喂给编译器
- 该仓库 pointnet2 的 setup.py 只打包 `_ext.so`，需手动补拷 `pointnet2_utils.py` / `pytorch_utils.py` / `pointnet2_modules.py` 到 `site-packages/pointnet2/` 并 `touch __init__.py`
- 包内顶层导入需改相对导入（`import pytorch_utils as pt_utils` → `from . import pytorch_utils as pt_utils` 等）

## 附录 B：夹爪参数（实测校准值，RS + 柔性夹爪）

- 张开位 ≈ +4.9 rad，闭合方向为负
- **空夹机械限位 ≈ -0.75 rad**（指尖互碰，可重复）
- 自由闭合反馈扭矩峰值 ≈ 0.24 N·m；接触物体后快速爬升
- 柔性夹爪握物时电机可能走出几乎全部行程 → 空抓判定不能依赖"停得早"，要靠"停稳后的最终位置 vs 限位阈值"

## 附录 C：grasp_driver.py 修改记录（柔性夹爪空抓误报修复）

文件：`drivers/robot/grasp_driver.py`，相对原版共 4 处改动：

1. **空抓判定重构**（`gripper_tick` CLOSING 分支）：三路判定——
   - `torque_loaded and not at_hard_stop` → 立即判成功（早期接触，位置还高）
   - `stopped and at_hard_stop` → 空夹
   - `stopped and not at_hard_stop` → 成功
   （原版在高速穿过硬限位阈值时会抢跑误判空夹；柔性夹爪握物会压穿旧阈值 0.05）
2. **`_hard_stop_angle` = `open_sign × -0.65`**（实测限位 -0.75 + 0.1 余量；原版 0.05 对柔性夹爪完全不对）
3. **接触扭矩阈值 = 0.35 × close_torque**（若 close_torque 降到 0.6，把比例改回 0.45）
4. **HOLDING 分支死代码修复**：原来嵌在 CLOSING 内永不执行，已挪到外层，夹持期间持续发"位置+保持力矩"
5. **`get_tcp_pose()` 参数修复**：`compute_fk(model, q)` 第三个参数是 frame 名字符串，不是关节数 `self._n`（传整数会被 Pinocchio 拒绝）
6. **调试开关**：`__init__` 里 `self._debug = True` 时每 0.1s 打印 `[gripper] pos/vel/torq`，校准用；日常运行改回 `False`
