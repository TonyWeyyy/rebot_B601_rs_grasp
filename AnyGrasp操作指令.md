# reBot B601-RS AnyGrasp 项目：关机重启后运行操作指令

适用项目：`~/wey/rebot_developer/rebot_grasp`  
硬件：reBot Arm B601-RS、RS 柔性夹爪、腕部 Orbbec Gemini 2  
环境：conda `rebotarm`、Python 3.10、PyTorch CUDA 13

## 1. 每次重启后的最简启动流程

完成物理检查：机械臂供电和急停正常，Gemini 2 接主板直连 USB 3.0，USB2CAN 已连接，工作区无人员和障碍物。

打开一个新终端，依次执行：

```bash
# 1. 激活项目环境
conda activate rebotarm

# 2. 进入项目
cd ~/wey/rebot_developer/rebot_grasp

# 3. 重新配置 CAN；先 down 可避免接口已 up 时修改 bitrate 报 busy
sudo ip link set can0 down 2>/dev/null || true
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up

# 4. 临时开放 Gemini 2 USB 权限；重启或重新插拔后可能需要重做
sudo chmod a+rw /dev/bus/usb/*/*

# 5. 检查 CAN 状态，应看到 can0、UP 和 bitrate 1000000
ip -details link show can0

# 6. 当前终端优先使用 conda 的 Pinocchio，避免用户目录 cmeel 混库
export PYTHONPATH="$CONDA_PREFIX/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"

# 7. 检查 AnyGrasp 权重、gsnet.so、license 和依赖
python scripts/anygrasp_env_check.py --check-license
```

最后一项必须显示：

```text
failures=0
```

然后按本文第 4 节启动需要的功能。

`PYTHONPATH` 只影响当前终端及其中启动的程序，不修改全局环境。新终端必须重新执行；不要在抓取终端执行 `source /opt/ros/...`。

## 2. 哪些内容不需要每次重做

只要没有重装环境、重拉项目或更换硬件，以下均持久保存：

- conda `rebotarm` 环境和 Python 包；
- AnyGrasp 的 `gsnet.so`、`checkpoint_detection.tar` 和 license；
- MinkowskiEngine、pointnet2、knn 等已编译扩展；
- YOLOE 模型 `models/yoloe-26l-seg.pt`；
- 手眼标定、相机内参和 `hand_eye_compensation_m`；
- `config/default.yaml` 中的工作区、放置姿态、夹爪力控和运动时长；
- 原 GraspNet、OBB 和 AnyGrasp 项目代码。

日常运行不需要重新编译，不需要重新申请 license，不需要重新手眼标定，也不需要执行 CUDA 编译用的 `NVCUDA/CPATH/LIBRARY_PATH` 等 export。

## 3. 重启后首次运行前的安全检查

- 机械臂从安全的 ready/观察位形开始；不要在弯曲肘部的非零位形直接禁用电机。
- 目标为已验证的轻质物体，容器为大开口、矮容器，两者完整进入相机画面。
- 当前放置配置应保持已经验证成功的姿态：

```yaml
place:
  orientation_rpy: [0.0, 0.7, 0.0]
  height_m: 0.15
  pre_place_offset_m: 0.08
```

- 保持碰撞检测开启，保持夹爪宽度和高度门槛，不为提速或提高成功率关闭安全过滤。
- 点云碰撞检测和端点 IK 不等于全臂或轨迹安全；确认真机路径、容器壁间隙和动力停止手段。

## 4. 启动项目

### 4.1 推荐：AnyGrasp 抓取放置，带两次人工确认

完成第 1 节后，在同一终端执行：

```bash
python scripts/anygrasp_place.py \
  --target-class toy \
  --container-class box \
  --min-base-z 0.05 \
  --confirm-motion
```

启动后：

1. 非 dry-run 模式会先运动到 ready。
2. 按 `G` 或空格，拍照并规划目标和容器。
3. 必须看到目标和容器检测成功、`pre-place/place` IK 均为 `success=True`，以及 AnyGrasp 安全检查和目标 IK 通过。
4. 核对五个规划位姿后，输入大写 `YES`，允许抓取。
5. 抓起、回撤并回 ready 后，确认前往容器的路径，输入大写 `PLACE`，允许放置和释放。
6. 其他输入均取消。空抓不会继续放置。

按键：

| 按键/输入 | 作用 |
|---|---|
| `G` / 空格 | 拍照并规划一次抓取放置 |
| `R` | 恢复实时预览 |
| `Q` / `Esc` | 退出 |
| `YES` | 抓取前确认，必须全大写 |
| `PLACE` | 抓起并回 ready 后的放置确认，必须全大写 |

第二次确认取消时，物体保持夹持在 ready，程序不会自动松爪。持有物体时不要直接按 `Q`，因为正常退出清理会释放夹爪；先人工接管物体和机械臂。

### 4.2 只规划不抓取：AnyGrasp 抓取放置 dry-run

机械臂需要先到 ready，但后续全部禁止运动：

```bash
python scripts/anygrasp_place.py \
  --target-class toy \
  --container-class box \
  --min-base-z 0.05 \
  --dry-run \
  --move-ready
```

`--dry-run --move-ready` 唯一允许的运动是启动时到 ready。若机械臂已经处于 ready，去掉 `--move-ready`：

```bash
python scripts/anygrasp_place.py \
  --target-class toy \
  --container-class box \
  --min-base-z 0.05 \
  --dry-run
```

此模式不执行抓取、放置、回撤或夹爪动作，退出也不松爪。

### 4.3 只做 AnyGrasp 抓取

带人工确认：

```bash
python scripts/grasp.py \
  --estimator anygrasp \
  --target-class toy \
  --min-base-z 0.05 \
  --confirm-motion
```

抓取 dry-run，只允许启动时到 ready：

```bash
python scripts/grasp.py \
  --estimator anygrasp \
  --target-class toy \
  --min-base-z 0.05 \
  --dry-run \
  --move-ready
```

### 4.4 临时回退原 GraspNet 抓取

```bash
python scripts/grasp.py \
  --estimator graspnet \
  --target-class toy \
  --dry-run \
  --move-ready
```

确认真机规划正确后，再去掉 `--dry-run --move-ready` 并按原项目安全流程运行。回退不需要删除 AnyGrasp 文件。

## 5. 可选自检

以下自检不连接相机、CAN 或机械臂。日常重启且代码未变时可跳过；修改代码、配置或环境后建议重新执行：

```bash
cd ~/wey/rebot_developer/rebot_grasp
python scripts/anygrasp_selfcheck.py
python scripts/selftest_anygrasp_place.py
```

放置自检应显示 `Ran 9 tests` 和 `OK`。自检通过不代表现场轨迹已经安全。

## 6. 常见启动故障

### 6.1 `can0` 不存在或 CAN 无法启动

```bash
ip link show
```

确认 USB2CAN 已插好。如果接口名不是 `can0`，不要直接套用命令，先按实际接口名调整。重新插拔后再次执行第 1 节 CAN 配置。

### 6.2 相机打不开或 USB 权限不足

```bash
sudo chmod a+rw /dev/bus/usb/*/*
```

重新插拔 Gemini 2 到主板直连 USB 3.0，避免 Hub 和机箱前面板。单次 `Timestamp anomaly` 后仍能取帧通常不是本次失败原因；若连续掉帧并最终出现连续帧失败，优先换 USB 口。

### 6.3 Pinocchio `undefined symbol` 或 cmeel 混库

确认当前终端已执行：

```bash
conda activate rebotarm
export PYTHONPATH="$CONDA_PREFIX/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
```

然后重新运行脚本。不要全局卸载 cmeel，不要为该项目修改 ROS 环境。

### 6.4 AnyGrasp license 检查失败

先执行：

```bash
cd ~/wey/rebot_developer/rebot_grasp
python scripts/anygrasp_env_check.py --feature-id --check-license
```

若提示 Feature ID mismatch，不要直接重新申请 license。此前本机 license 对应的 `docker0` MAC 为：

```text
5a:fd:d3:ae:41:ac
```

检查当前值：

```bash
ifconfig docker0 | grep ether
```

如果重启后该值发生变化，且当前没有依赖 Docker 网络的任务，可恢复：

```bash
sudo ip link set dev docker0 down
sudo ip link set dev docker0 address 5a:fd:d3:ae:41:ac
sudo ip link set dev docker0 up
ifconfig docker0 | grep ether
```

回到项目目录重新检查：

```bash
cd ~/wey/rebot_developer/rebot_grasp
python scripts/anygrasp_env_check.py --check-license
```

只有显示 `failures=0` 才继续。该 MAC 恢复属于本机 license 配套操作，不是每次重启都必须执行；仅在检查失败且确认 Docker MAC 变化时使用。

### 6.5 检测不到目标或容器

- 确认目标和容器完整进入画面，容器口可见，不只是侧面或一个角。
- 查看启动时打印的 YOLO open-vocabulary 类别列表；`box/container/tray/bowl` 是不同类别词。
- 改善光照和背景对比，不先降低置信度门槛。
- 若实际检测类别是 `container`，启动参数改为 `--container-class container`；若是 `tray`，改为 `--container-class tray`。

### 6.6 放置 IK 失败

确认 YAML 中仍是已验证的：

```yaml
place:
  orientation_rpy: [0.0, 0.7, 0.0]
```

优先调整容器位置，不扩大工作区来绕过 IK，不删除 `pre-place/place` 预检。

### 6.7 想恢复保守慢速运动

运动时长保存在 `config/default.yaml` 的 `place:` 段，重启后仍保留。若提速后出现抖动、物体滑动或容器壁间隙不足，可恢复：

```yaml
place:
  pregrasp_duration_s: 4.0
  grasp_duration_s: 3.0
  retreat_duration_s: 3.0
  ready_duration_s: 4.0
  pre_place_duration_s: 4.0
  place_duration_s: 3.0
  release_wait_s: 1.0
```

只恢复时长，不改 MIT 模式、电机限速、夹爪力矩或安全过滤。

## 7. 每次重启的勾选清单

- [ ] 机械臂、相机、CAN 和急停物理连接正常。
- [ ] `conda activate rebotarm`。
- [ ] 进入 `~/wey/rebot_developer/rebot_grasp`。
- [ ] `can0` 已设置为 1 Mbps 并 up。
- [ ] Gemini 2 USB 权限已设置。
- [ ] 当前终端已导出 conda 优先的 `PYTHONPATH`。
- [ ] `anygrasp_env_check.py --check-license` 显示 `failures=0`。
- [ ] 目标、容器、工作区和急停现场检查完成。
- [ ] 代码或环境变化后，两项离线自检通过。
- [ ] 先 dry-run 验证新场景，再带 `--confirm-motion` 运行真机。
