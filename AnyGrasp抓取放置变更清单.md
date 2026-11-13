# AnyGrasp 抓取放置变更清单

## 本次目标

在已验证的 AnyGrasp 抓取基础上，新增“目标与容器同帧检测 → 抓取 → 回撤 → ready → 预放置 → 放置 → 释放 → ready”流程。不修改原 OBB 放置入口，也不改变已验证的 RS 夹爪力控、空抓判断、TCP 转换和手眼标定。

## 本机新增文件

1. `scripts/anygrasp_place.py`：AnyGrasp 抓取放置主入口。
2. `scripts/selftest_anygrasp_place.py`：无相机、CAN、机械臂和 GPU 的离线回归。
3. `utils/place_utils.py`：容器选择、容器中心投影、放置位姿生成、工作区与目标/容器分离检查。

## 本机替换文件

1. `utils/anygrasp_utils.py`：`infer_frame()` 新增可选 `detections`，放置流程复用同一帧 YOLO 结果，避免目标和容器来自不同帧。
2. `config/default.yaml`：新增 `place:` 配置；保留已受控扩展的 base X 上限 `0.50 m`。
3. `README_zh.md`、`AnyGrasp升级操作手册.md`：加入自检、dry-run、首次真机和回退步骤。

## 保持不变

- `drivers/robot/grasp_driver.py`
- `scripts/grasp.py` 的 AnyGrasp/GraspNet 抓取路线
- `scripts/set.py`、`scripts/set2.py` 的原 OBB 放置路线
- `sdk/`、`models/`、手眼标定和相机标定文件
- 夹爪开口、力控、空抓判断和碰撞过滤

覆盖合并时不要删除本机 `sdk/`、`models/` 和标定文件。

## 安全行为

- `--dry-run`：不执行抓取、放置、回撤或夹爪动作，退出也不松爪。
- `--dry-run --move-ready`：唯一允许的运动是启动时到 ready。
- `--confirm-motion`：抓取前必须输入大写 `YES`；抓起并回 ready 后必须输入大写 `PLACE`。
- 第二次确认取消：物体保持夹持，不自动释放。
- 空抓：直接结束，禁止继续放置。
- 在任何抓取动作前，同时通过目标和 pre-place/place 的工作区与 IK 预检。
- 最终候选宽度仍不超过 `0.087 m`；碰撞检测强制开启；高度、倾角和工作区过滤不关闭。

端点 IK 与点云碰撞检测不证明全臂或轨迹无碰撞。首次真机必须现场确认整条路径、容器壁/桌面间隙和动力停止手段。

## 当前验证边界

已完成源码语法检查、配置断言和放置离线回归。真实 Gemini 2 同帧目标/容器检测、真实 B601-RS 放置轨迹和容器内释放仍需按手册分阶段验证；离线通过不等于真机已验收。
