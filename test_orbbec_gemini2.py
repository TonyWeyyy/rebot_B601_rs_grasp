# import cv2
# import numpy as np
# import os
# import time
# from datetime import datetime
# from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBError

# # ==================== 配置区域 ====================
# SAVE_DIR = "gemini2_test_output"          # 输出目录
# NUM_FRAMES = 10                           # 保存多少组图像
# MIN_DEPTH_MM = 20                         # 深度有效范围下限 (mm)
# MAX_DEPTH_MM = 5000                       # 深度有效范围上限 (mm)
# # =================================================

# def setup_output_dir():
#     """创建输出目录"""
#     os.makedirs(SAVE_DIR, exist_ok=True)
#     print(f"[INFO] 输出目录: {os.path.abspath(SAVE_DIR)}")


# def test_device_enumeration():
#     """测试 1：枚举已连接的 Orbbec 设备"""
#     print("\n" + "=" * 50)
#     print("[测试 1] 设备枚举")
#     print("=" * 50)

#     try:
#         from pyorbbecsdk import Context
#         ctx = Context()
#         device_list = ctx.query_devices()
#         count = device_list.get_count()
#         print(f"[结果] 检测到 {count} 个 Orbbec 设备")

#         if count == 0:
#             print("[失败] 未找到任何设备。请检查 USB 连接和 udev 规则。")
#             return False

#         for i in range(count):
#             device = device_list.get_device_by_index(i)
#             info = device.get_device_info()
#             print(f"  设备 {i}: {info.get_name()}  "
#                   f"(PID: {info.get_pid():04x}, SN: {info.get_serial_number()})")
#         return True

#     except Exception as e:
#         print(f"[失败] 设备枚举出错: {e}")
#         return False


# def test_pipeline_start():
#     """测试 2：启动 Pipeline（零配置模式）"""
#     print("\n" + "=" * 50)
#     print("[测试 2] Pipeline 启动")
#     print("=" * 50)

#     try:
#         pipeline = Pipeline()
#         pipeline.start()  # 零配置模式，自动加载默认流配置
#         print("[结果] Pipeline 启动成功")

#         # 获取相机参数，确认深度/彩色内参可用
#         try:
#             camera_param = pipeline.get_camera_param()
#             print(f"  深度内参: fx={camera_param.depth_intrinsic.fx:.1f}, "
#                   f"fy={camera_param.depth_intrinsic.fy:.1f}")
#             print(f"  彩色内参: fx={camera_param.rgb_intrinsic.fx:.1f}, "
#                   f"fy={camera_param.rgb_intrinsic.fy:.1f}")
#         except Exception as e:
#             print(f"  [警告] 无法获取相机内参: {e}")

#         return pipeline

#     except Exception as e:
#         print(f"[失败] Pipeline 启动失败: {e}")
#         return None


# def test_frame_capture(pipeline):
#     """测试 3：获取并保存彩色 + 深度帧"""
#     print("\n" + "=" * 50)
#     print("[测试 3] 帧捕获与保存")
#     print("=" * 50)

#     saved_count = 0
#     depth_stats = []

#     for i in range(NUM_FRAMES * 3):  # 多尝试几次，跳过预热帧
#         try:
#             frames = pipeline.wait_for_frames(1000)  # 超时 1000ms
#             if frames is None:
#                 continue

#             color_frame = frames.get_color_frame()
#             depth_frame = frames.get_depth_frame()

#             if color_frame is None or depth_frame is None:
#                 continue

#             # ---- 彩色帧处理 ----
#             color_width = color_frame.get_width()
#             color_height = color_frame.get_height()
#             color_format = color_frame.get_format()

#             # 转换为 OpenCV BGR 格式
#             if color_format == OBFormat.RGB:
#                 color_image = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
#                 color_image = color_image.reshape((color_height, color_width, 3))
#                 color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
#             elif color_format == OBFormat.BGR:
#                 color_image = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
#                 color_image = color_image.reshape((color_height, color_width, 3))
#             elif color_format == OBFormat.MJPG:
#                 color_image = cv2.imdecode(
#                     np.frombuffer(color_frame.get_data(), dtype=np.uint8),
#                     cv2.IMREAD_COLOR)
#             elif color_format == OBFormat.YUYV:
#                 yuyv = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
#                 yuyv = yuyv.reshape((color_height, color_width, 2))
#                 color_image = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUYV)
#             else:
#                 print(f"  [警告] 不支持的彩色格式: {color_format}")
#                 continue

#             # ---- 深度帧处理 ----
#             depth_width = depth_frame.get_width()
#             depth_height = depth_frame.get_height()
#             depth_scale = depth_frame.get_depth_scale()

#             depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
#             depth_data = depth_data.reshape((depth_height, depth_width))
#             depth_mm = depth_data.astype(np.float32) * depth_scale

#             # 计算深度统计（只统计有效范围）
#             valid_mask = (depth_mm >= MIN_DEPTH_MM) & (depth_mm <= MAX_DEPTH_MM)
#             if np.any(valid_mask):
#                 valid_depths = depth_mm[valid_mask]
#                 depth_stats.append({
#                     "min": float(np.min(valid_depths)),
#                     "max": float(np.max(valid_depths)),
#                     "mean": float(np.mean(valid_depths)),
#                     "valid_ratio": float(np.sum(valid_mask)) / depth_mm.size * 100
#                 })

#             # ---- 保存图像 ----
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
#             color_path = os.path.join(SAVE_DIR, f"color_{timestamp}.png")
#             depth_path = os.path.join(SAVE_DIR, f"depth_{timestamp}.png")

#             cv2.imwrite(color_path, color_image)

#             # 深度图伪彩色可视化后保存
#             depth_clipped = np.clip(depth_mm, MIN_DEPTH_MM, MAX_DEPTH_MM)
#             depth_norm = ((depth_clipped - MIN_DEPTH_MM) /
#                           (MAX_DEPTH_MM - MIN_DEPTH_MM) * 255).astype(np.uint8)
#             depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
#             cv2.imwrite(depth_path, depth_colormap)

#             saved_count += 1
#             print(f"  [{saved_count}/{NUM_FRAMES}] 已保存: "
#                   f"彩色 {color_width}x{color_height}, "
#                   f"深度 {depth_width}x{depth_height}")

#             if saved_count >= NUM_FRAMES:
#                 break

#         except OBError as e:
#             print(f"  [错误] SDK 错误: {e}")
#             continue
#         except Exception as e:
#             print(f"  [错误] 处理帧时出错: {e}")
#             continue

#     # ---- 深度数据统计 ----
#     if depth_stats:
#         avg = lambda key: np.mean([s[key] for s in depth_stats])
#         print(f"\n[深度数据统计] (基于 {len(depth_stats)} 帧)")
#         print(f"  平均最近距离: {avg('min'):.1f} mm")
#         print(f"  平均最远距离: {avg('max'):.1f} mm")
#         print(f"  平均距离:     {avg('mean'):.1f} mm")
#         print(f"  有效像素占比: {avg('valid_ratio'):.1f}%")

#     return saved_count


# def test_live_preview(pipeline):
#     """测试 4：实时预览（按 Q 或 ESC 退出）"""
#     print("\n" + "=" * 50)
#     print("[测试 4] 实时预览 (按 Q 或 ESC 退出)")
#     print("=" * 50)

#     try:
#         while True:
#             frames = pipeline.wait_for_frames(1000)
#             if frames is None:
#                 continue

#             color_frame = frames.get_color_frame()
#             depth_frame = frames.get_depth_frame()
#             if color_frame is None or depth_frame is None:
#                 continue

#             # 彩色图
#             color_width = color_frame.get_width()
#             color_height = color_frame.get_height()
#             color_image = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
#             color_image = color_image.reshape((color_height, color_width, 3))
#             color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

#             # 深度图
#             depth_width = depth_frame.get_width()
#             depth_height = depth_frame.get_height()
#             depth_scale = depth_frame.get_depth_scale()
#             depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
#             depth_data = depth_data.reshape((depth_height, depth_width))
#             depth_mm = depth_data.astype(np.float32) * depth_scale

#             depth_clipped = np.clip(depth_mm, MIN_DEPTH_MM, MAX_DEPTH_MM)
#             depth_norm = ((depth_clipped - MIN_DEPTH_MM) /
#                           (MAX_DEPTH_MM - MIN_DEPTH_MM) * 255).astype(np.uint8)
#             depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

#             # 并排显示
#             combined = np.hstack((color_image, depth_colormap))
#             cv2.imshow("Gemini 2 Test - Color | Depth (Q/ESC to quit)", combined)

#             key = cv2.waitKey(1)
#             if key == ord('q') or key == 27:  # Q 或 ESC
#                 break

#     except Exception as e:
#         print(f"[错误] 预览出错: {e}")
#     finally:
#         cv2.destroyAllWindows()


# def main():
#     print("╔" + "═" * 48 + "╗")
#     print("║      Orbbec Gemini 2 相机测试脚本 (pyorbbecsdk2)      ║")
#     print("╚" + "═" * 48 + "╝")

#     setup_output_dir()

#     # 测试 1：设备枚举
#     if not test_device_enumeration():
#         print("\n[终止] 设备未连接，请检查硬件后重试。")
#         return

#     # 测试 2：启动 Pipeline
#     pipeline = test_pipeline_start()
#     if pipeline is None:
#         print("\n[终止] Pipeline 启动失败。")
#         return

#     try:
#         # 测试 3：捕获并保存帧
#         saved = test_frame_capture(pipeline)

#         if saved == 0:
#             print("\n[失败] 未能捕获任何有效帧。")
#             return

#         print(f"\n[成功] 共保存 {saved} 组图像到 {os.path.abspath(SAVE_DIR)}")

#         # 测试 4：实时预览
#         test_live_preview(pipeline)

#     finally:
#         # 清理资源
#         try:
#             pipeline.stop()
#             print("\n[INFO] Pipeline 已停止。")
#         except Exception:
#             pass


# if __name__ == "__main__":
#     main()

import cv2
import numpy as np
from pyorbbecsdk import Pipeline, OBFormat

def main():
    # 启动相机（零配置模式，自动开启默认彩色+深度流）
    pipeline = Pipeline()
    pipeline.start()
    print("相机已启动。按 'q' 或 ESC 退出。")

    try:
        while True:
            frames = pipeline.wait_for_frames(1000)  # 超时 1000ms
            if frames is None:
                continue

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame is None or depth_frame is None:
                continue

            # ---------- 处理彩色帧 ----------
            w = color_frame.get_width()
            h = color_frame.get_height()
            fmt = color_frame.get_format()
            data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)

            if fmt == OBFormat.RGB:
                color = data.reshape((h, w, 3))
                color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            elif fmt == OBFormat.BGR:
                color = data.reshape((h, w, 3))
            elif fmt == OBFormat.MJPG:
                color = cv2.imdecode(data, cv2.IMREAD_COLOR)
            elif fmt == OBFormat.YUYV:
                yuyv = data.reshape((h, w, 2))
                color = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUYV)
            else:
                print(f"不支持的彩色格式: {fmt}")
                continue

            # ---------- 处理深度帧 ----------
            dw = depth_frame.get_width()
            dh = depth_frame.get_height()
            depth_scale = depth_frame.get_depth_scale()
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data = depth_data.reshape((dh, dw))
            depth_mm = depth_data.astype(np.float32) * depth_scale

            # 限制显示范围 20mm ~ 5000mm，并转为伪彩色
            depth_clipped = np.clip(depth_mm, 20, 5000)
            depth_norm = ((depth_clipped - 20) / (5000 - 20) * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

            # ---------- 并排显示（统一高度） ----------
            if depth_color.shape[0] != color.shape[0]:
                scale = color.shape[0] / depth_color.shape[0]
                new_w = int(depth_color.shape[1] * scale)
                depth_color = cv2.resize(depth_color, (new_w, color.shape[0]))

            combined = np.hstack((color, depth_color))
            cv2.imshow("Gemini 2 - Color | Depth  (press q/ESC to quit)", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q 或 ESC
                break

    except Exception as e:
        print(f"运行出错: {e}")
    finally:
        # 释放资源
        pipeline.stop()
        cv2.destroyAllWindows()
        print("相机已关闭。")

if __name__ == "__main__":
    main()