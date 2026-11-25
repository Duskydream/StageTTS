"""大模型 Function Calling 多轮调度器、时长约束反思闭环与自主质检循环。"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from openai import OpenAI

from backend.agent.prompts import AUTO_QA_AGENT_PROMPT, EDITING_AGENT_SYSTEM_PROMPT
from backend.agent.tools import (
    TOOL_REGISTRY,
    TOOLS_SCHEMA,
    inspect_project_quality,
    load_project_script,
    synthesize_segments,
)
from backend.schemas import (
    AgentChatResponse,
    SegmentDurationCheck,
    ToolCallLog,
)
from backend.services.translation import (
    client_model,
    create_director_client,
    create_openai_client,
)
from backend.services.tts_client import TTSServiceUnavailableError

_TOOL_ALLOWED_ARGS: Dict[str, set[str]] = {
    item["function"]["name"]: set(item["function"]["parameters"].get("properties", {}).keys())
    for item in TOOLS_SCHEMA
}


def _resolve_agent_client() -> OpenAI:
    primary = create_openai_client()
    # Keep unit tests patching `orchestrator.create_openai_client` with mocks isolated.
    if isinstance(primary, OpenAI):
        return create_director_client(primary)
    return primary


def _summarize_tool_result(tool_name: str, result: Dict[str, Any]) -> str:
    if tool_name == "inspect_project_quality":
        return f"全片体检完成：共发现 {result.get('issue_count', 0)} 条待优化片段"
    if tool_name == "query_script":
        return f"查询到 {result.get('count', 0)} 条片段"
    if tool_name == "batch_update_segments":
        updated = result.get("updated_ids", [])
        pending = result.get("marked_pending_ids", [])
        return f"更新片段 {updated}（待重合成: {pending}）"
    if tool_name == "split_segment":
        return (
            f"已拆分片段 #{result.get('left_id')} -> "
            f"#{result.get('left_id')} & #{result.get('right_id')} "
            f"(@ {result.get('split_time_sec')}s)"
        )
    if tool_name == "merge_segments":
        return f"已合并片段 {result.get('removed_ids')} -> #{result.get('kept_id')}"
    if tool_name == "synthesize_segments":
        synth = result.get("synthesized_ids", [])
        durations = result.get("overflow_segments", [])
        short = sum(1 for item in durations if item.get("duration_state") == "too_short") if durations else 0
        long = len(durations) - short
        return f"合成片段 {synth}（时长偏差: 超长 {long} 条 / 过短 {short} 条）"
    if tool_name == "update_mix_settings":
        return (
            f"已更新混音设置（总音量 {result.get('bgm_volume')}，"
            f"自动闪避 {'开' if result.get('auto_ducking', True) else '关'}）"
        )
    if tool_name == "render_final_video":
        return f"已导出成片 {result.get('output_video')}（字幕={result.get('burn_subtitles', 'none')}）"
    return "执行完成"


def _run_tool_loop(
    project_id: str,
    *,
    system_prompt: str,
    user_content: str,
    auto_synthesize: bool,
    max_steps: int = 6,
    client: OpenAI | None = None,
) -> AgentChatResponse:
    active_client = client or _resolve_agent_client()
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    tool_logs: List[ToolCallLog] = []
    final_reply = "已完成指令处理。"
    pending_modified: set[int] = set()
    synthesized_ids: set[int] = set()
    overflow: dict[int, SegmentDurationCheck] = {}
    synthesis_errors: list[str] = []

    def record_synthesis(result: Dict[str, Any], attempted_ids: List[int] | None = None) -> None:
        completed = result["synthesized_ids"]
        synthesized_ids.update(completed)
        if attempted_ids is None:
            pending_modified.clear()
        else:
            pending_modified.difference_update(attempted_ids)
        pending_modified.difference_update(completed)
        for seg_id in completed:
            overflow.pop(seg_id, None)
        for item in result["overflow_segments"]:
            check = SegmentDurationCheck.model_validate(item)
            overflow[check.segment_id] = check
        synthesis_errors.extend(result["errors"])

    for _ in range(max_steps):
        completion = active_client.chat.completions.create(
            model=client_model(active_client),
            messages=messages,  # type: ignore[arg-type]
            tools=TOOLS_SCHEMA,  # type: ignore[arg-type]
            tool_choice="auto",
            temperature=0.2,
        )
        assistant_msg = completion.choices[0].message
        tool_calls = getattr(assistant_msg, "tool_calls", None) or []

        if not tool_calls:
            final_reply = assistant_msg.content or final_reply
            break

        assistant_record: Dict[str, Any] = {
            "role": "assistant",
            "content": assistant_msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ],
        }
        messages.append(assistant_record)

        for tc in tool_calls:
            tool_name = tc.function.name
            func = TOOL_REGISTRY.get(tool_name)
            try:
                args = json.loads(tc.function.arguments or "{}")
                if not isinstance(args, dict):
                    raise ValueError("工具参数必须为 JSON 对象")
                # 强制锁定 project_id，禁止越权读写其他项目
                args["project_id"] = project_id

                if func is None:
                    raise ValueError(f"禁止调用未授权工具: {tool_name}")
                allowed = _TOOL_ALLOWED_ARGS.get(tool_name, set())
                extra = set(args) - allowed
                if extra:
                    raise ValueError(f"工具 {tool_name} 包含未声明参数: {sorted(extra)}")

                if tool_name == "synthesize_segments":
                    raw_ids = args.get("segment_ids")
                    attempted = [int(i) for i in raw_ids] if isinstance(raw_ids, list) else None
                    if attempted is not None:
                        args["segment_ids"] = attempted
                    try:
                        result = synthesize_segments(**(args | {"auto_reflect": auto_synthesize}))
                    except TTSServiceUnavailableError:
                        if attempted is None:
                            pending_modified.clear()
                        else:
                            pending_modified.difference_update(attempted)
                        raise
                    record_synthesis(result, attempted)
                else:
                    result = func(**args)
                summary = _summarize_tool_result(tool_name, result)
                tool_logs.append(
                    ToolCallLog(
                        tool_name=tool_name,
                        arguments=args,
                        result_summary=summary,
                    )
                )

                if tool_name in {"batch_update_segments", "split_segment", "merge_segments"}:
                    pending_modified.update(result.get("marked_pending_ids") or [])
            except Exception as exc:
                result = {"error": str(exc)}
                tool_logs.append(
                    ToolCallLog(
                        tool_name=tool_name,
                        arguments={"project_id": project_id},
                        result_summary=f"调用失败: {exc}",
                    )
                )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

    if auto_synthesize and pending_modified:
        synth_args = {"project_id": project_id, "segment_ids": sorted(pending_modified)}
        synth_res = synthesize_segments(**synth_args)
        record_synthesis(synth_res)
        tool_logs.append(
            ToolCallLog(
                tool_name="synthesize_segments",
                arguments=synth_args,
                result_summary=f"自动收尾合成: {_summarize_tool_result('synthesize_segments', synth_res)}",
            )
        )

    if overflow:
        details = "、".join(f"#{item.segment_id} 超 {item.overflow_seconds:.3f}s" for item in overflow.values())
        final_reply += f"\n配音仍超出时间槽：{details}。"
    if synthesis_errors:
        final_reply += "\n" + "\n".join(dict.fromkeys(synthesis_errors))
    if final_reply == "已完成指令处理。" and tool_logs:
        final_reply = "已执行：" + "；".join(log.result_summary for log in tool_logs)

    latest_script = load_project_script(project_id)
    return AgentChatResponse(
        project_id=project_id,
        reply=final_reply,
        tool_logs=tool_logs,
        synthesized_ids=sorted(synthesized_ids),
        script=latest_script,
    )


def run_agent_chat(
    project_id: str,
    user_message: str,
    auto_synthesize: bool = True,
) -> AgentChatResponse:
    """自然语言批量剪辑 Agent 主调度入口（严格通过确定性工具白名单执行）。"""
    script = load_project_script(project_id)
    project_context = {
        "project_id": script.project_id,
        "mode": script.mode,
        "duration": script.media.duration,
        "bgm_volume": script.media.bgm_volume,
        "has_cinematic_stems": script.media.cinematic_stems is not None,
        "stem_volumes": script.media.stem_volumes.model_dump() if script.media.stem_volumes else None,
        "auto_ducking": script.media.auto_ducking,
        "characters": {
            key: char.display_name for key, char in script.characters.items()
        },
        "segment_count": len(script.segments),
        "auto_synthesize": auto_synthesize,
    }
    return _run_tool_loop(
        project_id,
        system_prompt=EDITING_AGENT_SYSTEM_PROMPT,
        user_content=(
            f"当前项目上下文：\n{json.dumps(project_context, ensure_ascii=False)}\n\n"
            f"用户指令：{user_message}"
        ),
        auto_synthesize=auto_synthesize,
        max_steps=6,
    )


def run_auto_qa_loop(
    project_id: str,
    *,
    max_rounds: int = 2,
    auto_synthesize: bool = True,
) -> AgentChatResponse:
    """自动执行全片质检（翻译密度/硬伤/切分/音频异常），必要时调度导演 Agent 局部修复并重配。"""
    report = inspect_project_quality(project_id)
    script = load_project_script(project_id)
    if report["issue_count"] == 0:
        return AgentChatResponse(
            project_id=project_id,
            reply="全片自动体检通过：未发现译文过密、斜杠硬伤、切分碎片或音频时长异常。",
            tool_logs=[
                ToolCallLog(
                    tool_name="inspect_project_quality",
                    arguments={"project_id": project_id},
                    result_summary="全片体检完成：0 个待优化项",
                )
            ],
            synthesized_ids=[],
            script=script,
        )

    compact_bible = {
        k: v for k, v in (script.show_bible or {}).items()
        if k != "line_notes" and v
    }
    qa_context = {
        "project_id": project_id,
        "source_lang": script.source_lang,
        "glossary": script.glossary,
        "characters": {
            spk: {"display_name": ch.display_name, "speaking_style": ch.speaking_style}
            for spk, ch in script.characters.items()
        },
        "show_bible": compact_bible,
        "initial_diagnostics": {
            "issue_count": report["issue_count"],
            "code_counts": report["code_counts"],
            "issues": report["issues"][:18],
        },
    }
    return _run_tool_loop(
        project_id,
        system_prompt=AUTO_QA_AGENT_PROMPT,
        user_content=(
            "请基于以下自动体检报告和设定表执行精准局部修复（不要改动无问题片段）：\n"
            f"{json.dumps(qa_context, ensure_ascii=False)}"
        ),
        auto_synthesize=auto_synthesize,
        max_steps=max(4, max_rounds * 3),
    )
