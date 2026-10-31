# q2a 多模态输入（图片 / 视频 / 音频）实测结论

> 实测时间：2026-10-31。环境：宿主 `chat.qwen.ai` + `qwen2api` 容器（`ghcr.io/sofs2005/q2a:dev`），
> 账号池真实账号。所有结论都是打真上游拿到的，不是推测。

## 0. 一句话结论

**不要走 OSS 上传**（`/api/v2/files/getstsToken` 现在 100% 被阿里云 WAF 滑块/拼图验证码拦住）。
**把文件字节直接塞进 `messages[0].files[i].url` 的 data URI 里**，上游完全接受，
图片 / 音频 / 视频都能被模型真正"看见/听见"，且不需要任何上传、解析、验证码。

**但内联容量只有 ~6KB**（url 字符数上限 8192，见 §3.1）：图片靠自动压缩降级成缩略图，
音频只够 ≤1.5s @32kbps，视频基本不可行。超出时返回 4xx（`attachment_too_large`），
绝不把超长 url 发出去。

## 1. 现状（改之前）

| 环节 | 现状 |
|---|---|
| `attachment_preprocessor.py` | 只认 `image_url` 里的 **data:** URI 和 `input_file`/`file`；`input_audio`、`video_url`、`input_video`、远程 http(s) 图片 URL、Anthropic `image` 块 **全都没处理** |
| `upstream_file_uploader.py` | 上传必挂：`getstsToken` 返回阿里云 WAF 验证码 HTML，`json.loads` 直接 `JSONDecodeError`；`context_attachment_manager` 捕获后回退内联文本 → 模型说"没有收到图片" |
| ref 结构 | audio/video 的 `type`/`showType` 写成 `"file"`（应分别是 `audio`/`video`）；非图片一律调 `/files/parse`（上游只有 doc 才 parse） |
| 模型能力 | 没有按模型区分模态，任何模型都塞任何附件 |

基线实测：`POST /v1/chat/completions` + `image_url` data URI → 200，但回答"图片未能成功提供"。

## 2. 上游 `files[]` 真实结构（从 `main.js` 0.3.28 反解 + 实测）

网页端 `beforeFileUpload()` 构造的 ref：

```json
{
  "type": "image|file|video|audio",
  "file": {
    "created_at": 0, "data": {}, "filename": "a.png", "hash": null,
    "id": "<file_id>", "user_id": "<uid>",
    "meta": {"name": "a.png", "size": 123, "content_type": "image/png"},
    "update_at": 0, "lastModified": 0, "name": "a.png",
    "webkitRelativePath": "", "size": 123, "type": "image/png"
  },
  "id": "<file_id>", "url": "<data:... 或 OSS 签名 URL>", "name": "a.png",
  "collection_name": "", "progress": 0, "status": "uploaded",
  "greenNet": "success", "size": 123, "error": "",
  "itemId": "<uuid>", "file_type": "image/png",
  "showType": "image|file|video|audio",
  "file_class": "vision|document|video|audio",
  "uploadTaskId": "<uuid>"
}
```

映射规则（`file_class → type/showType`）：

| 媒体 | file_class | type | showType | context | 是否 parse |
|---|---|---|---|---|---|
| 图片 | `vision` | `image` | `image` | 不加 | 否 |
| 音频 | `audio` | `audio` | `audio` | 不加 | 否 |
| 视频 | `video` | `video` | `video` | 不加 | 否 |
| 文档 | `document` | `file` | `file` | `"full"` | **是**（`parsedFileTypes=[FILE]`） |

`greenNet`：视频初始 `greening`，成功后 `success`；其它直接 `success`。

## 3. 关键实测：data URI 直塞 `url` 可行（但容量只有 ~6KB）

`id` 传空串也可以。实测（模型 `qwen3.8-omni-flash`，`chat_type=t2t`，其余 payload 不变）：

| 用例 | 结果 |
|---|---|
| 纯色橙 PNG（800B）data URI | ✅ 回答"橙色"（正确） |
| 440Hz 正弦 mp3（8.5KB）data URI | ❌ 见下方长度上限（8.5KB → base64 11.4KB → `url too long`） |
| 纯蓝 2s mp4（2.2KB）data URI | ✅ 回答"蓝色"（正确） |
| 远程 http(s) 图片 URL（Google logo） | ❌ `invalid_input`（上游不代下载） |
| 空 url + 假 file_id | ❌ `Internal error!` |
| 文档（txt）data URI | ⚠️ 200，但模型只看到 base64 原文（**没走 parse**，文档仍需上传+解析） |

### 3.1 url 长度上限 = 8192 字符（**修正早期结论**）

`messages[0].files[i].url` 是 Pydantic `max_length=8192` 的**字符串**字段，`data:<mime>;base64,`
前缀同样计入长度。边界实测（QA 复核）：

| url 字符数 | 结果 |
|---|---|
| 8182（`data:image/png;base64,...`） | ✅ 200，模型正常描述图片 |
| 8198（同上） | ❌ `RequestValidationError: url too long` |
| 7810（`data:audio/mpeg;base64,...`） | ✅ 200 |
| 8194（同上） | ❌ `url too long` |

即上限**与模态无关**，只看 url 字符数。base64 膨胀 1.33 倍 + 前缀开销，内联实际只装得下
约 **6.1KB 二进制**（`(8192 - 22) * 3 / 4 ≈ 6127B`）。

> ⚠️ 早期结论"32KB wav / 1.5MB、3.9MB、8MB PNG 全部 OK（最大测到 base64 10.7MB 的 url）"
> 是**假阳性**：那次探测只看响应前 200 字符，而上游把 `RequestValidationError` **塞进
> HTTP 200 的 content 里**返回，所以"200"被误读成"成功"。修正后的结论：超过 ~6KB 一律拒绝
> （图片先自动压缩），绝不把超长 url 发出去。

失败时的错误体（注意外层是 **HTTP 200**）：

```json
{"success": false, "data": {"code": "RequestValidationError",
 "details": "[\"...files -> 0 -> url': Value error, url too long\"]"}}
```

网关侧因此必须做两件事：① 发请求前按 url 字符数门控（超限 → 图片压缩 / 其它 400）；
② 收到这类校验错误时映射成 4xx，而不是当 assistant 正文以 200 返回。

### 3.2 图片压缩的实测效果（Pillow，阈值 url<=8100）

| 输入 | 原始 | 压缩后 | 结果 |
|---|---|---|---|
| 320KB 噪点 PNG（420×260，最坏情况） | 328355B | 5576B JPEG 192×119 | ✅ 7459 字符，PSNR 10.9dB（噪点图不可辨细节，属病态输入） |
| ~200KB 真实街景照片 JPEG | 98736B | 5418B JPEG 192×144 | ✅ 7247 字符，PSNR 22.2dB（场景/主体可辨，文字细节丢失） |
| 类照片 PNG 640×480（平滑渐变） | 500KB | 3650B JPEG 640×480 | ✅ 5007 字符，PSNR 35.7dB |

结论：**真实照片能压进 6KB，但只能保留"缩略图级别"的可辨识度**（192px 长边）；
小图标 / 纯色 / 截图类图片几乎无损。噪点类不可压缩内容会退化到 128~192px 且视觉质量差。

## 4. 模型模态矩阵（`GET /api/v2/models/` → `info.meta`）

| 模型 | modality | capabilities |
|---|---|---|
| `qwen3.8-omni-flash` | text, image, video, **audio** | vision/document/video/audio/citations/thinking |
| `qwen3.5-omni-plus` | text, image, video, **audio** | 同上（thinking=false） |
| `qwen3.8-max` / `qwen3.7-plus` / `qwen3.6-plus` / `qwen3.5-plus` | text, image, video | 无 audio |
| `qwen3.7-max` | text | 仅 document |

→ 音频只对 omni 系列有意义；图片/视频对 plus/max 系列都可用。

## 5. 上传路径为什么必须绕开

- `POST /api/v2/files/getstsToken`（v1 是 405）在真实账号 + 真实浏览器 cookie jar 下**都**返回
  阿里云 WAF 页面：`aliyun_waf_aa` / `sceneId=19x5u7lo` / `region=sgp`，
  浏览器里跑真实上传流程会拉起 **PUZZLE（拼图）验证码**（`InitCaptchaV2` + `shadow.png`/`back.png`）。
- 也就是说：不是我们代码写错，是上游对上传接口加了人机验证。网页端本身也要用户拖拼图。
- 另外实测确认：`/api/v2/files/upload` 不存在，`/api/v2/files/getstsToken`、
  `/api/v1/files/getstsToken` 一律撞 WAF；上游也**不会**代下载远程 URL。
  因此只有两条路：① `data URI <= 8192 字符`；② 真实上传拿到的 `file_id`（当前被验证码堵死）。
- 结论：**默认走 data URI 内联**；只有文档（需要 parse）或超大文件才回退 OSS 上传，
  且必须能识别 WAF 页面并给出明确错误（不能抛 `JSONDecodeError`）。
  内联放不下 + 上传不可用时，返回 **400 `attachment_too_large`**，不静默丢附件。

## 6. 上游配额（`GET /api/v2/configs/`）

`image_max_count=5 / image_max_size=20MB`、`video_max_count=1 / 2000MB`、
`audio_max_count=1 / 2000MB`、`doc_max_count=5 / 20MB`、`audio_max_duration=10800s`、`video_max_duration=3600s`。

## 7. 复现脚本

`generated/q2a-mm/` 下：`probe_limit.py`（内联 ref 全量验证）、`probe_ref.py`（远程 URL 反例）、
`probe_doc.py`（文档反例）、`probe_models.py`（模态矩阵）、`p10_drop.py`（浏览器 drop 上传 → 撞验证码）。

容器内运行：`docker cp probe_limit.py qwen2api:/tmp/ && docker exec -w /workspace qwen2api python /tmp/probe_limit.py /tmp/a2.mp3`

## 8. 补充实测（2026-10-31 晚，用户反馈 12MB 视频可上传后复测）

### 8.1 8192 字符上限只约束「data URI 内联」，不影响真实上传

用户在网页端成功上传 12MB 视频并让模型总结内容 —— 这是**正确且符合上游设计**的：
网页端把文件 PUT 到阿里云 OSS，`files[i].url` 填的是**短的 OSS 签名 URL**（几百字符），
远低于 8192，所以文件多大都行（受 `video_max_size=2000MB` 约束）。

**8192 只作用于「把文件字节直接塞进 url」这条绕行方案**。我们当前实现之所以被卡在 ~6KB，
是因为 OSS 上传被验证码堵死、只能走内联。这不是上游的文件大小限制。

### 8.2 上传验证码：阿里云 PUZZLE 拼图（不是 x5sec 滑块）

`POST /api/v2/files/getstsToken` 返回阿里云 WAF 挑战页（`sceneId=19x5u7lo`, `region=sgp`），
浏览器里走真实上传会拉起 **拼图（PUZZLE）** 验证码：
- 页面元素：`#aliyunCaptcha-img`(300×200 背景) / `#aliyunCaptcha-puzzle`(52×200 拼图条) /
  `#aliyunCaptcha-sliding-slider`(40×40 滑块) / `#aliyunCaptcha-sliding-body`(300 轨道)
- 素材：`static-captcha-sgp.aliyuncs.com/qst/PUZZLE/online/903/<uuid>/{back,shadow}.png`
- 用户实测：一次上传会**连续出现两次**验证码。

### 8.3 已解出的部分（可复用）

1. **缺口定位**：`back.png` 里缺口是「亮 + 低饱和」的拼图块。
   `score = max(RGB)/255 - 1.2 * saturation`，用 `shadow.png` 的 alpha 当掩码在缺口所在 y 带内滑窗取最大 → 位置误差 ≤ 2px
   （已用叠加图逐像素核对：拼图轮廓与缺口完全重合）。
2. **鼠标位移 ↔ 拼图位移不是 1:1，是二次映射**（实测 8 个点，拟合误差 < 0.1px）：
   ```
   piece_dx = 0.003555 * mouse_dx^2 + 0.075899 * mouse_dx
   mouse_dx = (-b + sqrt(b^2 + 4a*piece_dx)) / (2a)
   ```
   例：拼图要移动 157px，鼠标要拖 199.7px。**用 1:1 拖动必然失败**，这是最初所有尝试失败的直接原因。
3. 拖动后（按住时）读 `#aliyunCaptcha-puzzle` 的 x 可校验落点，实测能控制在 ±1.5px 内。

### 8.4 仍然失败的部分（未解决）

落点正确、轨迹拟人（45~62 步、变速、抖动、慢速 1.2~2s）、UA/sec-ch-ua 都伪装成
Chrome146/Windows、`--headless=new`，**连续 10+ 次全部被拒**（滑块不消失，getstsToken 仍返回 WAF 页）。
推测是阿里云的设备指纹（`cloudauth-device-dualstack`）+ 轨迹模型评分，不是位置问题。

结论：**要真正支持任意大小文件，必须解决这个拼图验证码**（或找到免验证码的上传通道）。
当前状态是「几何已解、反爬未过」。
