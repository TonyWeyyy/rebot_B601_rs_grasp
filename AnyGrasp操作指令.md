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

# 6. 检查 AnyGrasp 权重、gsnet.so、license 和依赖
python scripts/anygrasp_env_check.py --check-license
```

最后一项必须同时看到：

```text
License passed: True, state: PASSED
AnyGrasp environment failures=0
```

然后按本文第 4 节启动需要的功能。

> 说明：代码已内置 `prioritize_env_pinocchio()` 自动规避 `~/.local` 下 `cmeel` 与 conda `pinocchio` 混库问题，一般不再需要手动 `export PYTHONPATH`；仍请不要在抓取终端执行 `source /opt/ros/...`。

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

- 机械臂从安全的零位或 ready 位开始；**B601-RS 无机械抱闸，退出时程序会强制先平滑回零 `safe_home()` 再失能电机**（即使在 `--dry-run` 下也是如此；若已在零位则不会额外移动）。
- 目标为已验证的轻质物体，容器为大开口、矮容器，两者完整进入相机画面。
- 核对 `config/default.yaml` 中 `place:` 段配置：

```yaml
place:
  orientation_rpy: [0.0, 0.7, 0.0]
  height_m: 0.10              # 可按容器高度在 YAML 或命令行 --place-height 调整
  pre_place_offset_m: 0.08
  retract_to_pre_place: true  # 释放后先垂直抬回 pre-place，再回 ready
  use_mask_centroid: true     # 用分割掩膜质心计算容器 XY
```

- 保持碰撞检测开启，保持夹爪宽度、倾角、工作区和高度门槛，不为提速或提高成功率关闭安全过滤。
- 点云碰撞检测和链式 IK 不等于全臂无碰撞；首次运行新场景务必确认真机路径、容器壁间隙和动力停止手段。

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
2. 按 `G` 或空格，连续采集多帧做深度中值融合，并同帧规划目标和容器。
3. 必须看到目标和容器检测成功、`pre-place / place / post-release-retract` 三段链式 IK 均为 `success=True`，以及 AnyGrasp 安全检查和目标三段链式 IK 通过。
4. 核对五个规划位姿后，输入大写 `YES`，允许抓取。
5. 空抓时会自动先垂直回撤到 `retreat` 再回 ready，禁止继续放置；抓起回撤并回 ready 后会自动校验夹爪未滑落（`is_holding()`），确认前往容器路径安全后输入大写 `PLACE`，允许放置和释放。
6. 释放后先垂直抬升回 `pre-place` 避开容器壁，再回 ready 并闭合夹爪。

按键与输入：

| 按键/输入 | 作用 |
|---|---|
| `G` / 空格 | 拍照（多帧深度融合）并规划一次抓取放置 |
| `R` | 恢复实时预览 |
| `Q` / `Esc` | 退出（先平滑回零再失能；非 dry-run 会先松爪） |
| `YES` | 抓取前确认，必须全大写 |
| `PLACE` | 抓起并回 ready 后的放置确认，必须全大写 |

> 第二次确认取消时，物体保持夹持在 ready，程序不会自动松爪。持有物体时不要直接按 `Q`，因为非 dry-run 退出会松爪；先人工接住物体再按 `Q`。

### 4.2 只规划不抓取：AnyGrasp 抓取放置 dry-run

机械臂需要先抬升到 ready 观察位，后续全部禁止抓取/放置/夹爪运动：

```bash
python scripts/anygrasp_place.py \
  --target-class toy \
  --container-class box \
  --min-base-z 0.05 \
  --dry-run \
  --move-ready
```

若机械臂已经处于观察位，去掉 `--move-ready`：

```bash
python scripts/anygrasp_place.py \
  --target-class toy \
  --container-class box \
  --min-base-z 0.05 \
  --dry-run
```

> 安全说明：`--dry-run` 不执行任何抓取、放置或夹爪开合动作，但退出时只要机械臂不在零位，**仍会先执行平滑回零 `safe_home()` 再关闭电机**，避免断电坠落。

### 4.3 只做 AnyGrasp 抓取

带人工确认：

```bash
python scripts/grasp.py \
  --estimator anygrasp \
  --target-class toy \
  --min-base-z 0.05 \
  --confirm-motion
```

抓取 dry-run（启动时到 ready）：

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

确认真机规划正确后，再去掉 `--dry-run --move-ready` 并带上 `--confirm-motion` 运行。回退不需要删除 AnyGrasp 文件。

## 5. 离线自检与单元测试

以下测试不连接相机、CAN 或机械臂；修改代码、配置或环境后建议全部执行：

```bash
cd ~/wey/rebot_developer/rebot_grasp
python -m unittest discover -s tests -v
python scripts/selftest_gripper_tightening.py
python scripts/selftest_anygrasp_place.py
python scripts/anygrasp_selfcheck.py
```

- `tests/` 应通过全部 25 项，`selftest_gripper_tightening.py` 应通过 9 项，`selftest_anygrasp_place.py` 应显示 `Ran 10 tests` 和 `OK`，`anygrasp_selfcheck.py` 应通过 4 项 PASS。
- 离线自检通过不代表现场轨迹无碰撞，真机前仍需现场观察确认。

## 6. 常见启动故障

### 6.1 `can0` 不存在或 CAN 无法启动

```bash
ip link show
```

确认 USB2CAN 已插好。如果接口名不是 `can0`，按实际接口名调整。重新插拔后再次执行第 1 节 CAN 配置。

### 6.2 相机打不开或 USB 权限不足

```bash
sudo chmod a+rw /dev/bus/usb/*/*
```

重新插拔 Gemini 2 到主板直连 USB 3.0，避免 Hub 和机箱前面板。单次 `Timestamp anomaly` 后仍能取帧通常不影响运行；若连续掉帧请优先换 USB 口。

### 6.3 Pinocchio `undefined symbol` 或 cmeel 混库

代码已自动优先使用当前 conda 环境的 `pinocchio`。如在自定义独立脚本中遇到该错误，可在终端执行：

```bash
conda activate rebotarm
export PYTHONPATH="$CONDA_PREFIX/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
```

### 6.4 AnyGrasp license 检查失败

先执行：

```bash
cd ~/wey/rebot_developer/rebot_grasp
python scripts/anygrasp_env_check.py --feature-id --check-license
```

如提示 `feature id mismatch`，参考根目录 [AnyGrasp通行证问题解决.md](file:///home/tony/wey/rebot_developer/rebot_grasp/AnyGrasp通行证问题解决.md) 核对本机网卡/Feature ID，或按官方流程更新对应 license。必须同时看到 `License passed: True, state: PASSED` 和 `failures=0` 才继续。

### 6.5 检测不到目标或容器

- 确认目标和容器完整进入画面，容器口可见，不只是侧面或一个角。
- `--target-class` / `--container-class` 传入的新类名会自动追加到 YOLOE 开放词汇列表，也可用 `--extra-yolo-class` 补充同义词。
- 改善光照和背景对比，再考虑微调 `--yolo-conf`。

### 6.6 放置 IK 失败

确认 `config/default.yaml` 中仍是已验证的 `place.orientation_rpy: [0.0, 0.7, 0.0]`。优先调整容器位置，不盲目扩大工作区，不跳过 `pre-place / place / post-release-retract` 链式 IK 预检。

### 6.7 想恢复保守慢速运动

若提速后出现抖动、物体滑动或容器壁间隙不足，可将 `config/default.yaml` 中 `place:` 时长调回：

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

## 7. 每次重启的勾选清单

- [ ] 机械臂、相机、CAN 和急停物理连接正常。
- [ ] `conda activate rebotarm` 并进入 `~/wey/rebot_developer/rebot_grasp`。
- [ ] `can0` 已设置为 1 Mbps 并 UP，Gemini 2 USB 权限已设置。
- [ ] `anygrasp_env_check.py --check-license` 显示 `License passed: True, state: PASSED` 且 `failures=0`。
- [ ] 目标、容器、工作区和急停现场检查完成。
- [ ] 代码或配置变化后，离线单元测试与自检全部通过。
- [ ] 先 `--dry-run --move-ready` 验证新场景，再带 `--confirm-motion` 运行真机。
