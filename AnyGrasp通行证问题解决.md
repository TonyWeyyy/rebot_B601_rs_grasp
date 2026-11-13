这是 **重启后 Docker 网卡 MAC 变化，导致 AnyGrasp 本机 Feature ID 改变**：

```text
当前本机：local=N88422209594732680315
license 对应：license=N48653159963195872945
```

之前的处理方法是：**把 `docker0` 的 MAC 恢复为申请 license 时的值**，不需要重新安装 AnyGrasp，也不需要重新申请 license。

## 1. 检查当前 Docker MAC

```bash
ifconfig docker0 | grep ether
```

原 license 对应的 MAC 是：

```text
5a:fd:d3:ae:41:ac
```

如果显示的不是这个值，继续下面步骤。

## 2. 恢复原 MAC

以下操作会短暂断开 Docker 桥接网络，先确保没有正在使用 Docker 网络的任务：

```bash
sudo ip link set dev docker0 down
sudo ip link set dev docker0 address 5a:fd:d3:ae:41:ac
sudo ip link set dev docker0 up
```

确认恢复结果：

```bash
ifconfig docker0 | grep ether
```

应该显示：

```text
ether 5a:fd:d3:ae:41:ac
```

## 3. 检查 Feature ID 是否恢复

```bash
cd ~/wey/rebot_developer/rebot_grasp/sdk/anygrasp_sdk/grasp_detection

python -c "from gsnet import get_feature_id; print(get_feature_id())"
```

期望恢复为：

```text
N48653159963195872945
```

## 4. 重新检查 license

```bash
cd ~/wey/rebot_developer/rebot_grasp
python scripts/anygrasp_env_check.py --check-license
```

必须看到：

```text
License passed: True, state: PASSED
```

然后才能继续运行 AnyGrasp。

## 特别注意：这次日志的 `failures=0` 是误导

你这次同时出现了：

```text
License passed: False
[AnyGrasp] environment failures=0
```

说明当前检查脚本的汇总逻辑有缺陷，**不能只看最后的 `failures=0`**。正确验收必须同时满足：

```text
License passed: True, state: PASSED
AnyGrasp environment failures=0
```

如果 `docker0` 已经是原 MAC，但 Feature ID 仍然不匹配，就不要继续改其他网卡，保留 `ifconfig -a` 和 Feature ID 输出进一步核对。

我也已经把这个注意事项补进了重启操作文档。