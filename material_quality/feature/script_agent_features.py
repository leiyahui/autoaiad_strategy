"""构造无标签视频证据、校验客观脚本并生成独立文本向量。"""

import base64
import datetime
import hashlib
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_SCRIPT_MODEL = "gpt-5.6-luna"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_SCRIPT_FEATURE_VERSION = "gpt56_luna_script_v1"
DASHSCOPE_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_DEFAULT_SCRIPT_MODEL = "qwen-vl-max"
CN_CLIP_SCRIPT_EMBEDDING_MODEL = "cn-clip-rn50-text-aggregate"
SCRIPT_SCHEMA_VERSION = "material_quality_script_agent_v1"
SCRIPT_FUNCTION_NAME = "submit_video_script_analysis"

SYSTEM_INSTRUCTIONS = """角色：电商短视频客观脚本分析器。

目标：根据按时间排列的画面、OCR、ASR和场景切换证据，还原素材真实表达的脚本、
故事线和营销结构，供后续统计建模使用。

成功标准：
- 按时间顺序描述开场、展开、对比证明或信任背书、优惠和行动引导；
- 区分画面事实、语音内容和屏幕文字，OCR或ASR明显错误时谨慎修正；
- 卖点、痛点、证明方式和转场必须能在输入证据中找到依据；
- 信息不足时使用“无法判断”或放入证据局限，不补造人物、品牌、价格或功效。

约束：
- 不评价素材好坏，不预测任何投放或商业表现；
- 不使用行业常识替代视频证据；
- 只提交规定的结构化函数参数，不输出额外说明。
"""


SCRIPT_ANALYSIS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "objective_summary": {"type": "string"},
        "content_category": {"type": "string"},
        "presentation_style": {
            "type": "string",
            "enum": [
                "真人口播", "商品展示", "剧情演绎", "测评对比", "知识讲解",
                "用户证言", "直播切片", "混合", "其他", "无法判断",
            ],
        },
        "opening_hook": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "hook_type": {
                    "type": "string",
                    "enum": [
                        "痛点", "问题", "利益承诺", "结果展示", "冲突悬念",
                        "权威背书", "价格优惠", "人物吸引", "商品直出", "其他",
                        "无明显钩子", "无法判断",
                    ],
                },
                "visual": {"type": "string"},
                "spoken_or_text": {"type": "string"},
                "estimated_end_second": {"type": "number"},
                "evidence": {"type": "string"},
            },
            "required": [
                "hook_type", "visual", "spoken_or_text",
                "estimated_end_second", "evidence",
            ],
        },
        "storyline": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "start_second": {"type": "number"},
                    "end_second": {"type": "number"},
                    "stage": {
                        "type": "string",
                        "enum": [
                            "开场", "痛点", "商品引入", "卖点讲解", "使用演示",
                            "对比证明", "信任背书", "价格优惠", "行动引导", "收尾",
                            "其他", "无法判断",
                        ],
                    },
                    "visual_action": {"type": "string"},
                    "narration": {"type": "string"},
                    "onscreen_text": {"type": "string"},
                    "communication_purpose": {"type": "string"},
                },
                "required": [
                    "start_second", "end_second", "stage", "visual_action",
                    "narration", "onscreen_text", "communication_purpose",
                ],
            },
        },
        "reconstructed_script": {"type": "string"},
        "selling_points": {"type": "array", "items": {"type": "string"}},
        "pain_points": {"type": "array", "items": {"type": "string"}},
        "proof_elements": {"type": "array", "items": {"type": "string"}},
        "promotion_offers": {"type": "array", "items": {"type": "string"}},
        "calls_to_action": {"type": "array", "items": {"type": "string"}},
        "characters": {"type": "array", "items": {"type": "string"}},
        "settings": {"type": "array", "items": {"type": "string"}},
        "transitions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "timestamp": {"type": "number"},
                    "from_scene": {"type": "string"},
                    "to_scene": {"type": "string"},
                    "method": {
                        "type": "string",
                        "enum": [
                            "硬切", "镜头运动", "遮挡转场", "特效转场", "字幕转场",
                            "同场景切换", "其他", "无法判断",
                        ],
                    },
                },
                "required": ["timestamp", "from_scene", "to_scene", "method"],
            },
        },
        "pacing": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "tempo": {
                    "type": "string",
                    "enum": ["慢", "中等", "快", "变化明显", "无法判断"],
                },
                "scene_count": {"type": "integer"},
                "opening_seconds": {"type": "number"},
                "average_scene_seconds": {"type": "number"},
                "density_notes": {"type": "string"},
            },
            "required": [
                "tempo", "scene_count", "opening_seconds",
                "average_scene_seconds", "density_notes",
            ],
        },
        "evidence_limitations": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "objective_summary", "content_category", "presentation_style",
        "opening_hook", "storyline", "reconstructed_script", "selling_points",
        "pain_points", "proof_elements", "promotion_offers", "calls_to_action",
        "characters", "settings", "transitions", "pacing",
        "evidence_limitations",
    ],
}


def _file_sha256(path: str) -> str:
    """计算本地输入文件摘要，保证特征可追溯。"""
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: str) -> dict:
    """读取本地JSON对象。"""
    try:
        with open(path, "r", encoding="utf-8") as source:
            payload = json.load(source)
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError("JSON读取失败：{}，原因：{}".format(path, exc))
    if not isinstance(payload, dict):
        raise ValueError("JSON根节点必须是对象：{}".format(path))
    return payload


def _image_data_url(path: str) -> str:
    """把本地关键帧编码为Responses API可读取的数据地址。"""
    extension = os.path.splitext(path)[1].lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    media_type = media_types.get(extension)
    if not media_type:
        raise ValueError("不支持的关键帧格式：{}".format(extension))
    with open(path, "rb") as source:
        encoded = base64.b64encode(source.read()).decode("ascii")
    return "data:{};base64,{}".format(media_type, encoded)


def select_evidence_frames(
    keyframe_files: Sequence[dict],
    max_frames: int = 8,
) -> List[dict]:
    """优先保留开场，再均匀覆盖完整故事线。"""
    records = [item for item in keyframe_files if isinstance(item, dict)]
    records.sort(key=lambda item: float(item.get("timestamp") or 0.0))
    if len(records) <= max_frames:
        return records
    opening_count = min(4, max_frames, len(records))
    selected_indices = list(range(opening_count))
    remaining_count = max_frames - opening_count
    if remaining_count > 0:
        start = opening_count
        span = len(records) - start
        for position in range(remaining_count):
            relative = int(round(
                float(position + 1) * float(span - 1) / float(remaining_count)
            ))
            selected_indices.append(min(start + relative, len(records) - 1))
    return [records[index] for index in sorted(set(selected_indices))]


def _truncate_text(value: object, max_characters: int) -> str:
    """限制单条证据文本长度，避免异常素材无限放大请求。"""
    text_value = str(value or "").strip()
    if len(text_value) <= max_characters:
        return text_value
    return text_value[:max_characters] + "\n[后续内容因长度限制省略]"


def build_script_request(
    content_manifest_path: str,
    model: str = DEFAULT_SCRIPT_MODEL,
    reasoning_effort: str = "low",
    max_frames: int = 8,
) -> Tuple[dict, dict]:
    """从本地档案构造不含标签的Luna结构化分析请求。"""
    manifest = _read_json(content_manifest_path)
    content_directory = os.path.dirname(content_manifest_path)
    selected_frames = select_evidence_frames(
        manifest.get("keyframe_files") or [],
        max_frames=max_frames,
    )
    if not selected_frames:
        raise ValueError("内容档案没有可用于脚本理解的关键帧")

    evidence_summary = {
        "duration_seconds": manifest.get("duration"),
        "frame_timestamps": manifest.get("frame_timestamps") or [],
        "scene_change_timestamps": manifest.get("scene_change_timestamps") or [],
        "selected_frames": [
            {
                "timestamp": item.get("timestamp"),
                "reasons": item.get("reasons") or [],
                "file": item.get("file"),
            }
            for item in selected_frames
        ],
        "ocr_text": _truncate_text(manifest.get("ocr_text"), 10000),
        "asr_text": _truncate_text(manifest.get("asr_text"), 14000),
        "asr_segments": [
            {
                "start": item.get("start"),
                "end": item.get("end"),
                "text": item.get("text"),
            }
            for item in (manifest.get("asr_segments") or [])[:160]
            if isinstance(item, dict)
        ],
    }
    content = [{
        "type": "input_text",
        "text": (
            "以下证据按视频时间顺序提供。第一张是全局关键帧拼图，后续为带时间戳的"
            "代表帧。请结合OCR、ASR和场景切换还原客观脚本。\n\n{}"
        ).format(json.dumps(evidence_summary, ensure_ascii=False)),
    }]
    contact_sheet_file = manifest.get("contact_sheet_file")
    if contact_sheet_file:
        contact_sheet_path = os.path.join(content_directory, str(contact_sheet_file))
        if os.path.isfile(contact_sheet_path):
            content.append({
                "type": "input_image",
                "image_url": _image_data_url(contact_sheet_path),
                "detail": "high",
            })
    keyframe_directory = os.path.join(
        content_directory,
        str(manifest.get("keyframe_directory") or ""),
    )
    selected_evidence = []
    for item in selected_frames:
        frame_path = os.path.join(keyframe_directory, str(item.get("file") or ""))
        if not os.path.isfile(frame_path):
            raise ValueError("代表帧文件不存在：{}".format(frame_path))
        content.append({
            "type": "input_text",
            "text": "代表帧时间戳：{:.3f}秒；抽取原因：{}".format(
                float(item.get("timestamp") or 0.0),
                ",".join(str(value) for value in (item.get("reasons") or [])),
            ),
        })
        content.append({
            "type": "input_image",
            "image_url": _image_data_url(frame_path),
            "detail": "high",
        })
        selected_evidence.append({
            "file": os.path.basename(frame_path),
            "timestamp": float(item.get("timestamp") or 0.0),
            "sha256": _file_sha256(frame_path),
        })

    request_payload = {
        "model": model,
        "instructions": SYSTEM_INSTRUCTIONS,
        "input": [{"role": "user", "content": content}],
        "reasoning": {"effort": reasoning_effort},
        "max_output_tokens": 5000,
        "store": False,
        "prompt_cache_key": "material-quality-script-agent-v1",
        "tools": [{
            "type": "function",
            "name": SCRIPT_FUNCTION_NAME,
            "description": "提交基于视频证据还原的客观脚本结构",
            "parameters": SCRIPT_ANALYSIS_SCHEMA,
            "strict": True,
        }],
        "tool_choice": {"type": "function", "name": SCRIPT_FUNCTION_NAME},
    }
    audit_info = {
        "source_manifest_sha256": _file_sha256(content_manifest_path),
        "selected_frames": selected_evidence,
        "prompt_sha256": hashlib.sha256(
            SYSTEM_INSTRUCTIONS.encode("utf-8")
        ).hexdigest(),
    }
    return request_payload, audit_info


class OpenAIJSONClient:
    """使用标准库调用OpenAI JSON接口，避免污染现有模型环境。"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: int = 180,
        max_retries: int = 3,
    ):
        if not api_key:
            raise ValueError("未设置OpenAI API密钥")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = int(timeout_seconds)
        self.max_retries = int(max_retries)

    def post(self, path: str, payload: dict) -> dict:
        """发送JSON请求，并仅对瞬时错误进行有限重试。"""
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                self.base_url + path,
                data=body,
                headers={
                    "Authorization": "Bearer {}".format(self.api_key),
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=self.timeout_seconds,
                ) as response:
                    result = json.loads(response.read().decode("utf-8"))
                if not isinstance(result, dict):
                    raise RuntimeError("OpenAI接口返回的JSON根节点不是对象")
                return result
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[-1200:]
                last_error = RuntimeError(
                    "OpenAI接口返回HTTP {}：{}".format(exc.code, detail)
                )
                if exc.code not in (408, 409, 429) and exc.code < 500:
                    break
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                last_error = exc
            if attempt < self.max_retries:
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError("OpenAI接口调用失败：{}".format(last_error))


class DashScopeJSONClient:
    """使用标准库调用百炼OpenAI兼容接口，保持评估环境依赖最小化。"""

    def __init__(
        self,
        api_key: str,
        base_url: str = DASHSCOPE_DEFAULT_BASE_URL,
        timeout_seconds: int = 180,
        max_retries: int = 3,
    ):
        if not api_key:
            raise ValueError("未设置百炼API密钥")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = int(timeout_seconds)
        self.max_retries = int(max_retries)

    def post(self, path: str, payload: dict) -> dict:
        """发送JSON请求，并仅对限流、超时和服务端错误退避重试。"""
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                self.base_url + path,
                data=body,
                headers={
                    "Authorization": "Bearer {}".format(self.api_key),
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=self.timeout_seconds,
                ) as response:
                    result = json.loads(response.read().decode("utf-8"))
                if not isinstance(result, dict):
                    raise RuntimeError("百炼接口返回的JSON根节点不是对象")
                return result
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[-1200:]
                last_error = RuntimeError(
                    "百炼接口返回HTTP {}：{}".format(exc.code, detail)
                )
                if exc.code not in (408, 409, 429) and exc.code < 500:
                    break
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                last_error = exc
            if attempt < self.max_retries:
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError("百炼接口调用失败：{}".format(last_error))


def build_dashscope_script_request(
    content_manifest_path: str,
    model: str = DASHSCOPE_DEFAULT_SCRIPT_MODEL,
    max_frames: int = 8,
) -> Tuple[dict, dict]:
    """把同一份无标签证据转换为百炼多模态Chat Completions请求。"""
    source_request, audit_info = build_script_request(
        content_manifest_path,
        model=model,
        reasoning_effort="low",
        max_frames=max_frames,
    )
    source_content = source_request["input"][0]["content"]
    user_content = []
    for item in source_content:
        if item.get("type") == "input_text":
            user_content.append({
                "type": "text",
                "text": str(item.get("text") or ""),
            })
        elif item.get("type") == "input_image":
            user_content.append({
                "type": "image_url",
                "image_url": {"url": str(item.get("image_url") or "")},
            })
    system_prompt = (
        SYSTEM_INSTRUCTIONS
        + "\n必须只输出一个符合下列JSON Schema的JSON对象，不要输出Markdown代码块或额外说明：\n"
        + json.dumps(SCRIPT_ANALYSIS_SCHEMA, ensure_ascii=False)
    )
    audit_info = dict(audit_info)
    audit_info["provider"] = "dashscope"
    audit_info["request_schema_version"] = "dashscope_chat_json_object_v1"
    audit_info["prompt_sha256"] = hashlib.sha256(
        system_prompt.encode("utf-8")
    ).hexdigest()
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.0,
        "max_completion_tokens": 5000,
        "response_format": {"type": "json_object"},
        "enable_thinking": False,
    }, audit_info


def parse_dashscope_script_analysis(response: dict) -> dict:
    """读取百炼Chat Completions结果，并执行与离线Luna完全相同的严格校验。"""
    choices = response.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise RuntimeError("百炼脚本分析响应没有候选结果")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            str(item.get("text") or "")
            for item in content
            if isinstance(item, dict)
        )
    text_value = str(content or "").strip()
    if text_value.startswith("```"):
        lines = text_value.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text_value = "\n".join(lines).strip()
    try:
        analysis = json.loads(text_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("百炼脚本分析结果不是有效JSON：{}".format(exc))
    if isinstance(analysis, dict):
        for storyline_item in analysis.get("storyline") or []:
            if isinstance(storyline_item, dict) and storyline_item.get("stage") == "证明":
                # 百炼会把提示词中的通用“证明”原样返回，统一归入已有训练枚举。
                storyline_item["stage"] = "对比证明"
    _validate_script_analysis(analysis)
    return analysis


def parse_script_analysis(response: dict) -> dict:
    """从Responses函数调用中读取严格结构化脚本。"""
    if response.get("status") != "completed":
        raise RuntimeError("脚本分析响应未完成：{}".format(response.get("status")))
    for item in response.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "function_call":
            continue
        if item.get("name") != SCRIPT_FUNCTION_NAME:
            continue
        try:
            analysis = json.loads(str(item.get("arguments") or ""))
        except (TypeError, ValueError) as exc:
            raise ValueError("脚本分析函数参数不是有效JSON：{}".format(exc))
        _validate_script_analysis(analysis)
        return analysis
    raise RuntimeError("脚本分析响应没有返回规定的函数调用")


def _validate_schema_value(value: object, schema: dict, path: str) -> None:
    """递归校验受控脚本结构，避免宽松检查掩盖Agent字段漂移。"""
    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(value, dict):
            raise ValueError("{}必须是对象".format(path))
        properties = schema.get("properties") or {}
        missing = [
            key for key in schema.get("required") or []
            if key not in value
        ]
        if missing:
            raise ValueError("{}缺少字段：{}".format(path, ",".join(missing)))
        if schema.get("additionalProperties") is False:
            unexpected = sorted(set(value) - set(properties))
            if unexpected:
                raise ValueError("{}包含未定义字段：{}".format(
                    path,
                    ",".join(unexpected),
                ))
        for key, child_value in value.items():
            child_schema = properties.get(key)
            if child_schema is not None:
                _validate_schema_value(
                    child_value,
                    child_schema,
                    "{}.{}".format(path, key),
                )
    elif expected_type == "array":
        if not isinstance(value, list):
            raise ValueError("{}必须是数组".format(path))
        item_schema = schema.get("items") or {}
        for index, child_value in enumerate(value):
            _validate_schema_value(
                child_value,
                item_schema,
                "{}[{}]".format(path, index),
            )
    elif expected_type == "string":
        if not isinstance(value, str):
            raise ValueError("{}必须是字符串".format(path))
    elif expected_type == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("{}必须是数值".format(path))
    elif expected_type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("{}必须是整数".format(path))
    enum_values = schema.get("enum")
    if enum_values is not None and value not in enum_values:
        raise ValueError("{}不在允许枚举中：{}".format(path, value))


def _validate_script_analysis(analysis: object) -> None:
    """在本地严格复核全部字段，防止异常响应进入训练。"""
    _validate_schema_value(analysis, SCRIPT_ANALYSIS_SCHEMA, "脚本分析")
    for item in analysis["storyline"]:
        start = float(item["start_second"])
        end = float(item["end_second"])
        if start < 0 or end < start:
            raise ValueError("故事线节点时间范围无效")


def canonical_script_text(analysis: dict) -> str:
    """生成稳定文本表示，供独立Embedding模型编码。"""
    sections = [
        "客观摘要：{}".format(analysis.get("objective_summary") or ""),
        "内容品类：{}".format(analysis.get("content_category") or ""),
        "呈现方式：{}".format(analysis.get("presentation_style") or ""),
        "开场钩子：{}".format(json.dumps(
            analysis.get("opening_hook") or {}, ensure_ascii=False, sort_keys=True
        )),
        "还原脚本：{}".format(analysis.get("reconstructed_script") or ""),
        "故事线：{}".format(json.dumps(
            analysis.get("storyline") or [], ensure_ascii=False, sort_keys=True
        )),
        "卖点：{}".format("；".join(analysis.get("selling_points") or [])),
        "痛点：{}".format("；".join(analysis.get("pain_points") or [])),
        "证明：{}".format("；".join(analysis.get("proof_elements") or [])),
        "优惠：{}".format("；".join(analysis.get("promotion_offers") or [])),
        "行动引导：{}".format("；".join(analysis.get("calls_to_action") or [])),
        "转场：{}".format(json.dumps(
            analysis.get("transitions") or [], ensure_ascii=False, sort_keys=True
        )),
        "节奏：{}".format(json.dumps(
            analysis.get("pacing") or {}, ensure_ascii=False, sort_keys=True
        )),
    ]
    return "\n".join(sections)


def request_embedding(
    client: OpenAIJSONClient,
    text_value: str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> Tuple[np.ndarray, dict]:
    """使用独立文本Embedding模型编码脚本结果。"""
    response = client.post("/embeddings", {
        "model": embedding_model,
        "input": text_value,
        "encoding_format": "float",
    })
    data = response.get("data") or []
    if not data or not isinstance(data[0], dict):
        raise RuntimeError("Embedding接口没有返回向量")
    embedding = np.asarray(data[0].get("embedding") or [], dtype=np.float32)
    if embedding.ndim != 1 or embedding.size == 0:
        raise RuntimeError("脚本Embedding维度无效")
    if not np.isfinite(embedding).all():
        raise RuntimeError("脚本Embedding包含非有限数值")
    return embedding, response.get("usage") or {}


def analyze_content_archive(
    client: OpenAIJSONClient,
    content_manifest_path: str,
    script_model: str = DEFAULT_SCRIPT_MODEL,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    reasoning_effort: str = "low",
    max_frames: int = 8,
) -> Tuple[dict, np.ndarray]:
    """完成单个内容档案的脚本理解和文本向量生成。"""
    request_payload, audit_info = build_script_request(
        content_manifest_path,
        model=script_model,
        reasoning_effort=reasoning_effort,
        max_frames=max_frames,
    )
    response = client.post("/responses", request_payload)
    analysis = parse_script_analysis(response)
    canonical_text = canonical_script_text(analysis)
    embedding, embedding_usage = request_embedding(
        client,
        canonical_text,
        embedding_model=embedding_model,
    )
    source_manifest = _read_json(content_manifest_path)
    result = {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "script_model": script_model,
        "embedding_model": embedding_model,
        "reasoning_effort": reasoning_effort,
        "content_sha256": source_manifest.get("content_sha256"),
        "source_feature_version": source_manifest.get("feature_version"),
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "response_id": response.get("id"),
        "response_model": response.get("model"),
        "response_usage": response.get("usage") or {},
        "embedding_usage": embedding_usage,
        "embedding_dim": int(embedding.size),
        "canonical_text": canonical_text,
        "analysis": analysis,
        "audit": audit_info,
    }
    return result, embedding


def script_feature_paths(
    output_root: str,
    script_feature_version: str,
    content_sha256: str,
) -> Dict[str, str]:
    """返回内容级脚本特征路径。"""
    normalized_sha = str(content_sha256 or "").lower()
    if len(normalized_sha) != 64:
        raise ValueError("内容哈希长度无效")
    directory = os.path.join(
        os.path.realpath(output_root),
        script_feature_version,
        "content",
        normalized_sha[:2],
    )
    basename = os.path.join(directory, normalized_sha)
    return {"directory": directory, "manifest": basename + ".json", "vectors": basename + ".npz"}


def save_script_feature(
    output_root: str,
    script_feature_version: str,
    result: dict,
    embedding: np.ndarray,
) -> Dict[str, str]:
    """先原子保存向量，再以JSON清单作为完成标记。"""
    paths = script_feature_paths(
        output_root,
        script_feature_version,
        str(result.get("content_sha256") or ""),
    )
    os.makedirs(paths["directory"], exist_ok=True)
    file_descriptor, vector_temporary = tempfile.mkstemp(
        prefix=".script_vector_", suffix=".npz.tmp", dir=paths["directory"]
    )
    try:
        with os.fdopen(file_descriptor, "wb") as output:
            np.savez_compressed(output, script_embedding=np.asarray(embedding, dtype=np.float32))
            output.flush()
            os.fsync(output.fileno())
        os.replace(vector_temporary, paths["vectors"])
    finally:
        if os.path.exists(vector_temporary):
            os.remove(vector_temporary)

    file_descriptor, manifest_temporary = tempfile.mkstemp(
        prefix=".script_manifest_", suffix=".json.tmp", dir=paths["directory"]
    )
    try:
        payload = dict(result)
        payload["script_feature_version"] = script_feature_version
        payload["vectors_file"] = os.path.basename(paths["vectors"])
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, sort_keys=True)
            output.flush()
            os.fsync(output.fileno())
        os.replace(manifest_temporary, paths["manifest"])
    finally:
        if os.path.exists(manifest_temporary):
            os.remove(manifest_temporary)
    return paths


def has_script_feature(
    output_root: str,
    script_feature_version: str,
    content_sha256: str,
) -> bool:
    """检查脚本清单和向量是否同时完整存在。"""
    paths = script_feature_paths(output_root, script_feature_version, content_sha256)
    if not os.path.isfile(paths["manifest"]) or not os.path.isfile(paths["vectors"]):
        return False
    try:
        manifest = _read_json(paths["manifest"])
        with np.load(paths["vectors"], allow_pickle=False) as vectors:
            embedding = np.asarray(vectors["script_embedding"], dtype=np.float32)
        return (
            manifest.get("content_sha256") == content_sha256
            and int(manifest.get("embedding_dim") or 0) == int(embedding.size)
            and embedding.ndim == 1
            and embedding.size > 0
            and bool(np.isfinite(embedding).all())
        )
    except (OSError, ValueError, TypeError, KeyError):
        return False
