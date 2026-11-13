# reBot B601-RS：AnyGrasp 完整适配操作手册

项目：`~/wey/rebot_developer/rebot_grasp`  
硬件：B601-RS、RS 柔性夹爪、腕部 Orbbec Gemini 2  
环境：conda `rebotarm`、Python 3.10、torch CUDA 13  
适配版本：AnyGrasp SDK 2026-07 之后的接口（`create_detector`、region/approach steering）

## 1. 这次代码已经改了什么

| 文件 | 内容 |
|---|---|
| `utils/anygrasp_utils.py` | 新版 SDK 加载、RGB-D 点云、YOLO 区域引导、接近方向引导、base 系安全检查 |
| `scripts/grasp.py` | 新增 `--estimator anygrasp/graspnet`，复用原 TCP 转换、IK、抓取和夹爪驱动 |
| `scripts/anygrasp_camera_demo.py` | 使用项目真实相机/YOLO/Open3D 接口的相机-only 验证 |
| `scripts/anygrasp_env_check.py` | 环境、权重、gsnet.so、license 只读检查 |
| `scripts/anygrasp_selfcheck.py` | 无权重、无相机、无机械臂的离线测试 |
| `config/default.yaml` | 默认估计器切换为 AnyGrasp，新增推理和安全配置 |
| 原 GraspNet 文件 | 全部保留，可用 CLI 随时回退 |

重要坐标事实：AnyGrasp/GraspNet 抓取坐标系中，**X 轴是接近方向，Y 轴是夹爪开合方向**；`translation` 是抓取中心，指尖位置为 `translation + depth * R[:,0]`。项目原有 `graspnet_rotation_to_rebot_tcp_rotation()` 继续负责到 B601 TCP 的转换，不要重复转换。

## 2. 合并代码前的注意事项

你上传的包不包含 `models/` 和 `sdk/`。本次交付也沿用这一边界，不包含权重、机械臂 SDK、GraspNet SDK 和 AnyGrasp SDK。

如果使用覆盖式合并：
1. 将整个交付目录解压到临时目录。
2. 备份本机原项目。
3. 用交付中的源码覆盖本机同名文件，**不要删除本机原有 `models/`、`sdk/`、标定 npz**。
4. 按下面步骤新增 `sdk/anygrasp_sdk`。

备份：

```bash
cd ~/wey/rebot_developer
cp -a rebot_grasp rebot_grasp_before_anygrasp_$(date +%Y%m%d_%H%M%S)
cd rebot_grasp
```

## 3. 先验证原路线和备份环境

```bash
conda activate rebotarm
cd ~/wey/rebot_developer/rebot_grasp

python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import graspnetAPI; print('graspnetAPI OK')"
python -c "from pointnet2 import pointnet2_utils; print('pointnet2 OK')"
python scripts/graspnet_camera_demo.py
```

原相机 demo 正常才继续。如果原路线已损坏，先恢复原路线，避免同时排查两套问题。

## 4. 下载 AnyGrasp SDK 并申请 license

### 4.1 下载和选择 Python 3.10 二进制

```bash
cd ~/wey/rebot_developer/rebot_grasp/sdk
git clone https://github.com/graspnet/anygrasp_sdk.git
cd anygrasp_sdk
ls grasp_detection/gsnet_versions/
cp grasp_detection/gsnet_versions/gsnet.cpython-310-x86_64-linux-gnu.so \
   grasp_detection/gsnet.so
```

如果实际文件名不同，选择含 `cpython-310`、`x86_64` 的版本，不能使用其他 Python 版本的 `.so`。

### 4.2 获取 feature id 并申请

```bash
cd ~/wey/rebot_developer/rebot_grasp/sdk/anygrasp_sdk/license_registration
cp ../grasp_detection/gsnet.so .
python -c "from gsnet import get_feature_id; print(get_feature_id())"
```

复制完整 feature id，格式类似 `N12345678900987654321`。若末尾有 `%`，填表时去掉。打开官方 SDK README 中的申请表并提交。官方说明通常 5 个工作日回复，注意垃圾邮件。新版 SDK 必须使用新工具生成的 feature id。

### 4.3 安装 license 和权重

收到 zip 后解压，应得到 `license/`，其中包含 `licenseCfg.json`、公钥、签名和 `.lic`：

```bash
cd ~/wey/rebot_developer/rebot_grasp/sdk/anygrasp_sdk
cp -a /你的实际下载路径/license grasp_detection/
cd grasp_detection
python -c "from gsnet import check_license; check_license('license')"
```

通过官方项目页提供的权重下载方式取得 `checkpoint_detection.tar`：

```bash
mkdir -p ~/wey/rebot_developer/rebot_grasp/sdk/anygrasp_sdk/grasp_detection/log
# 最终路径必须是：
# sdk/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar
```

## 5. 编译 CUDA 13 版 MinkowskiEngine

必须使用 AnyGrasp 作者修改版的 `cuda-13` 分支，不要使用旧 NVIDIA 主分支代替。

```bash
conda activate rebotarm
conda install -c nvidia cuda-nvcc=13.0 cuda-cudart-dev=13.0 -y
conda install -c anaconda openblas-devel -y
pip install ninja

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
hash -r
which nvcc
nvcc --version       # 必须为 13.x，并与 torch 编译 CUDA 匹配

NVCUDA=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cu13
export CPATH=$NVCUDA/include:$CPATH
export CPLUS_INCLUDE_PATH=$NVCUDA/include:$CPLUS_INCLUDE_PATH
export LIBRARY_PATH=$NVCUDA/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$NVCUDA/lib:$LD_LIBRARY_PATH

cd ~/wey/rebot_developer/rebot_grasp/sdk/anygrasp_sdk
mkdir -p dependencies
cd dependencies
git clone https://github.com/chenxi-wang/MinkowskiEngine.git
cd MinkowskiEngine
git checkout cuda-13
python setup.py install \
  --blas_include_dirs=${CONDA_PREFIX}/include \
  --blas_library_dirs=${CONDA_PREFIX}/lib \
  --blas=openblas
```

安装其余依赖：

```bash
cd ~/wey/rebot_developer/rebot_grasp/sdk/anygrasp_sdk
pip install -r requirements.txt
```

`graspnetAPI` 和 `pointnet2` 已在原项目中可用时不要重装。只有导入失败才处理：
- `graspnetAPI` 缺失：按原项目 GraspNet 安装方式从源码安装，避免旧依赖约束改动现有 numpy。
- `pointnet2` 缺失：优先恢复原扩展；确需重编时使用 `pip install . --no-build-isolation` 和原项目的包修复流程。

日常运行不需要这些 CUDA export；不要在 grasp 终端 source ROS。

## 6. 环境检查和官方 demo

```bash
conda activate rebotarm
cd ~/wey/rebot_developer/rebot_grasp
python scripts/anygrasp_env_check.py --check-license
```

必须显示 `failures=0`。需要重新查看 feature id 时：

```bash
python scripts/anygrasp_env_check.py --feature-id --check-license
```

官方样例：

```bash
cd ~/wey/rebot_developer/rebot_grasp/sdk/anygrasp_sdk/grasp_detection
python demo.py --checkpoint_path log/checkpoint_detection.tar --vis
```

验收：样例点云与夹爪正常显示，无 license、gsnet、MinkowskiEngine 或 CUDA 错误。

## 7. 本项目离线自检

```bash
cd ~/wey/rebot_developer/rebot_grasp
python scripts/anygrasp_selfcheck.py
```

必须全部 PASS。它验证：
- AnyGrasp X 轴接近方向和 depth 指尖修正；
- 原 B601 TCP 转换、插入深度、预抓取与回撤不变；
- 指尖过低、水平接近、开口超限、工作区越界会被拒绝；
- 新版 SDK 的 region/approach steering 与碰撞检测参数；
- Gemini 深度毫米值到相机系米制点云的投影。

自检通过仍不代表现场标定、IK 或轨迹安全。

## 8. 核对配置和安全边界

配置位于 `config/default.yaml`：

```yaml
grasp_pipeline:
  grasp:
    estimator: anygrasp
```

CLI 优先级更高：

```bash
python scripts/grasp.py --estimator graspnet --dry-run  # 临时回原路线
```

逐项核对：

| 配置 | 当前初值 | 真机前要求 |
|---|---:|---|
| `anygrasp.max_gripper_width` | 0.09 m | 已与原驱动 `GRIPPER_MAX_DISTANCE_M=0.09` 对齐 |
| `anygrasp.gripper_height` | 0.03 m | 按参与碰撞模型的手指高度复核，不是夹爪总长 |
| `safety.width_margin_m` | 0.003 m | 开口不能顶到机械极限 |
| `approach_thresh_deg` | 45° | 推理阶段只保留近 top-down 候选 |
| `safety.max_tilt_deg` | 45° | base 系最终再检查一次 |
| `collision_detection` | true | 代码强制，禁止真机关闭 |
| `dense_grasp` | false | 稳定后再用于堆叠场景 |
| `min_base_z_m` | 原项目 0.01 m | 必须为桌面高度 + 1 cm；若桌面定义变化需重核 |
| `base_workspace_min/max` | 保守示例 | 按实际底座、桌面、臂展复核，不是通用标定值 |
| `lims` | 相机系示例 | 排除机械臂本体、支架和桌外背景 |

安全过滤会检查 pregrasp、grasp、retreat、最终指尖和抓取中心；随后才进入原有 IK 检查。**点云碰撞检测不等于机械臂本体无碰撞，也不等于轨迹安全。**

## 9. 相机-only 验证（不需要 CAN）

机械臂保持静止，人员不进入工作区。相机使用主板直连 USB 3.0，按原教程设置 USB 权限。

```bash
conda activate rebotarm
cd ~/wey/rebot_developer/rebot_grasp
python scripts/anygrasp_camera_demo.py --target-class "toy"
```

关闭 YOLO 做全场景测试：

```bash
python scripts/anygrasp_camera_demo.py --no-yolo
```

按键：G/空格推理，R 恢复预览，Q/Esc 退出。

至少测试单个方块、圆柱、瓶子、小物体、两个相邻物体，各换 3 个位置。验收：
- 候选位于合理夹持区域，不悬空、不扎桌面；
- 开合轴横跨可夹持方向；
- 终端 center/tip 均为合理米制坐标；
- 静止场景重复拍照无厘米级随机跳动；
- YOLO 模式下候选落在目标区域，而不是背景；
- Open3D 中夹爪方向与物体几何一致。

此阶段不连接机械臂，因此无法做 base 系最终安全判定；不要跳过下面的 dry-run。

## 10. dry-run 的真实行为与操作

本版已修正原脚本：`--dry-run` **不再运动到 ready，也不在退出时开合夹爪**。它仍初始化 SDK、读取关节状态并做 IK 检查，所以必须在整洁工作区进行。

先让机械臂处于原观察位形，然后新终端执行：

```bash
conda activate rebotarm
cd ~/wey/rebot_developer/rebot_grasp
sudo ip link set can0 down 2>/dev/null
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
sudo chmod a+rw /dev/bus/usb/*/*
sudo chmod 666 /dev/ttyUSB0

python scripts/grasp.py --estimator anygrasp --dry-run --target-class "toy"
```

按 G 后逐项确认：
- 机械臂没有 ready 或抓取运动；
- 使用拍照时刻的 TCP 位姿计算 `T_cam2base`，推理期间机械臂不能移动；
- 日志出现 `[AnyGrasp safety] pass` 和原 IK 可执行信息；
- pregrasp 高于 grasp，retreat 沿接近反方向退开；
- grasp/tip 均高于 z 地板，所有位置在工作区；
- TCP RPY 和夹爪方向与原 GraspNet 同类 top-down 位姿一致；
- 被拒绝的候选有明确原因，不会强行执行最高分候选。

## 11. 首次真机和逐级开放

### 阶段 A：近 top-down

1. 桌面只放一个轻质、防滑、不易碎物体，机械臂从原 ready/观察位开始。
2. 保持原低速轨迹时长，不修改夹爪力控和空抓阈值。
3. 急停可触达，最好另一人在场，手不进入工作区。
4. 执行：

```bash
python scripts/grasp.py --estimator anygrasp --target-class "toy"
```

5. 手放急停旁，按 G 后观察完整过程。位姿突变、碰撞趋势、异响或失控立即急停。
6. 前 5 次只验证抓起和保持，不接放置。
7. 至少 4 种物体、累计 20 次，无碰桌、无异常空抓误判后再放宽。

### 阶段 B：侧抓

每次只把 `approach_thresh_deg` 和 `safety.max_tilt_deg` 从 45 同时调到一个新值，例如 60。每次重新执行：自检 → 相机-only → dry-run → 低速真机。不要直接允许任意方向，不要同时改插入深度或夹爪力控。

侧抓稳定后再试 `dense_grasp: true`。若空抓误判复现，保存 `[gripper]` 遥测并按原 CLOSING/HOLDING 阈值标定流程处理，不得关闭空抓检测。

## 12. A/B 对比

使用同一相机、同一标定、同一 ready 位姿和相同物体位置。每种网络、每类物体各 5 次：

```bash
python scripts/grasp.py --estimator graspnet --target-class "toy"
python scripts/grasp.py --estimator anygrasp  --target-class "toy"
```

记录抓取成功率、安全筛选通过率、IK 通过率、碰桌次数、空抓误判和推理时间。碰撞次数必须为 0，不能用成功率换安全。

AnyGrasp 更可能在新物体、复杂外形和堆叠场景体现优势；单视角遮挡、标定误差、深度噪声和夹爪问题不会因换网络自动消失。

## 13. 故障处理

- `create_detector() returned None`：查 license、本机 feature id、权重与 license 配套关系。
- `gsnet` 导入失败：确认 Python 3.10 对应 `.so` 已复制为 `grasp_detection/gsnet.so`。
- `MinkowskiEngine` 编译失败：确认作者修改版、`cuda-13` 分支、nvcc 13.x 和 CUDA 头文件 export。
- 区域点不足：检查 YOLO 框、深度质量、`lims` 和 `target_expand_ratio`，不要直接降到极少点。
- 全部候选被拒绝：逐条看 rank 原因，分别处理倾角、z、工作区、开口，不要总开关式关闭安全过滤。
- 系统性偏低/偏高：沿用 `hand_eye_compensation_m` 小步修正；`pregrasp_offset_m` 不改变最终抓取高度。
- 相机掉帧：换主板直连 USB 3.0，排除 Hub/前面板带宽问题。

## 14. 回退

临时回退，不改 YAML：

```bash
python scripts/grasp.py --estimator graspnet --dry-run --target-class "toy"
```

永久回退：把 `grasp_pipeline.grasp.estimator` 改为 `graspnet`，或恢复升级前备份。AnyGrasp 文件和 license 可保留，不影响原路线。
