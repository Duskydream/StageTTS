import { useEffect, useState } from "react";
import { useProjectStore } from "../../store/useProjectStore";
import { useWorkflowStore } from "../../store/useWorkflowStore";
import type { ProjectScript } from "../../types/script";

const field = "w-full rounded border border-white/10 bg-[#181818] px-2 py-1 text-[11px] text-primary";
const button =
  "rounded bg-[#333333] px-2 py-1 text-[11px] text-primary hover:bg-[#404040] disabled:opacity-40";

export function TranslationSettings({ script }: { script: ProjectScript }) {
  const updateSettings = useProjectStore((s) => s.updateTranslationSettings);
  const retranslate = useProjectStore((s) => s.retranslate);
  const offline = useProjectStore((s) => s.backendStatus !== "online");
  const busy = useProjectStore((s) => s.isLoading || s.isSynthesizing);
  const pipelineRunning = useWorkflowStore((s) => s.isPipelineRunning);
  const [sourceLang, setSourceLang] = useState("");
  const [terms, setTerms] = useState("");
  const [message, setMessage] = useState("");

  useEffect(() => {
    setSourceLang(script.source_lang ?? "");
    setTerms(
      Object.entries(script.glossary ?? {})
        .map(([source, target]) => `${source} = ${target}`)
        .join("\n"),
    );
  }, [script.source_lang, script.glossary]);

  function applySettings() {
    const entries: Array<[string, string]> = [];
    for (const [index, line] of terms.split("\n").entries()) {
      if (!line.trim()) continue;
      const separator = line.indexOf("=");
      const source = line.slice(0, separator).trim();
      const target = line.slice(separator + 1).trim();
      if (separator < 0 || !source || !target) {
        setMessage(`第 ${index + 1} 行格式应为 原词 = 译词`);
        return false;
      }
      entries.push([source, target]);
    }
    updateSettings(sourceLang.trim().toLowerCase() || null, Object.fromEntries(entries));
    return true;
  }

  return (
    <details className="border-b border-dashed border-white/[0.08] px-3.5 py-2.5 text-[11px] text-secondary">
      <summary className="flex cursor-pointer select-none items-center justify-between text-primary">
        <span>翻译</span>
        <span className="font-mono text-[10px] text-muted">{sourceLang || "待定"}</span>
      </summary>
      <div className="mt-2 space-y-2">
        <label className="flex items-center gap-2">
          <span className="w-8 shrink-0 text-muted">原声</span>
          <select
            className={field}
            value={sourceLang}
            onChange={(e) => setSourceLang(e.target.value)}
            disabled={busy || pipelineRunning}
          >
            <option value="">待确认</option>
            <option value="ja">日语</option>
            <option value="en">英语</option>
            <option value="zh">普通话</option>
            <option value="ko">韩语</option>
            <option value="yue">粤语</option>
            {sourceLang && !["ja", "en", "zh", "ko", "yue"].includes(sourceLang) && (
              <option value={sourceLang}>{sourceLang}</option>
            )}
          </select>
        </label>
        <textarea
          className={field}
          rows={3}
          aria-label="术语表"
          placeholder="术语表：每行 原词 = 译词"
          value={terms}
          onChange={(e) => setTerms(e.target.value)}
        />
        <div className="flex gap-1.5">
          <button
            className={button}
            type="button"
            disabled={busy || pipelineRunning}
            onClick={() => {
              if (applySettings()) setMessage("已应用");
            }}
          >
            应用
          </button>
          <button
            className={button}
            type="button"
            disabled={
              offline ||
              busy ||
              pipelineRunning ||
              !script.segments.some((seg) => seg.orig_text.trim())
            }
            onClick={async () => {
              if (!applySettings()) return;
              setMessage("重译中…");
              setMessage(await retranslate());
            }}
          >
            重译
          </button>
        </div>
        {message && (
          <p role="status" className="text-[10px]">
            {message}
          </p>
        )}
      </div>
    </details>
  );
}
