"""生成物用途常量（图片 / 视频）。

本模块只放纯常量，**不 import 任何项目内模块**，供 file_store、清理任务、
API 层（images / videos / media）共用，避免 api 与 services 之间出现环依赖。
"""
from __future__ import annotations

# 生图回源转存结果，对外通过 /v1/images/content/{id} 访问
GENERATED_IMAGE_PURPOSE = "generated_image"
# 视频回源转存结果，对外通过 /v1/videos/content/{id} 访问
GENERATED_VIDEO_PURPOSE = "generated_video"
# 全部「生成物」用途：媒体库聚合展示 + 生成物 TTL 统一清理的适用范围
MEDIA_PURPOSES = (GENERATED_IMAGE_PURPOSE, GENERATED_VIDEO_PURPOSE)
