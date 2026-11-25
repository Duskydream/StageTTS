import { ChevronDown, ChevronRight, Settings, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { fetchLLMSettings, saveLLMSettings, type LLMSettings } from "../services/api";
import { TrainingSettingsPanel } from "./TrainingSettings";

export function SettingsPage({ onClose }: { onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [settings, setSettings] = useState<LLMSettings | null>(null);
  const [baseUrl, setBaseUrl] = useState("");
  const [model, setModel] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [clearKey, setClearKey] = useState(false);
  const [directorModel, setDirectorModel] = useState("");
  const [directorBaseUrl, setDirectorBaseUrl] = useState("");
  const [directorApiKey, setDirectorApiKey] = useState("");
  const [clearDirectorKey, setClearDirectorKey] = useState(false);
  const [showDirectorSection, setShowDirectorSection] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [reload, setReload] = useState(0);

  useEffect(() => {
    dialog.current?.showModal();
    return () => dialog.current?.close();
  }, []);

  function syncFromResponse(result: LLMSettings) {
    setSettings(result);
    setBaseUrl(result.base_url);
    setModel(result.model);
    setDirectorModel(result.director_model ?? "");
    setDirectorBaseUrl(result.director_base_url ?? "");
    if (
      result.director_configured ||
      Boolean(result.director_model?.trim()) ||
      Boolean(result.director_base_url?.trim()) ||
      Boolean(result.director_api_key_configured)
    ) {
      setShowDirectorSection(true);
    }
  }

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError("");
    void fetchLLMSettings()
      .then((result) => {
        if (!active) return;
        syncFromResponse(result);
      })
      .catch(() => {
        if (active) setError("读取设置失败");
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [reload]);

  async function save(event: React.FormEvent) {
    event.preventDefault();
    if (saving || loading || !settings) return;
    setSaving(true);
    setError("");
    setMessage("");
    try {
      const result = await saveLLMSettings({
        base_url: baseUrl.trim(),
        model: model.trim(),
        ...(apiKey.trim() ? { api_key: apiKey.trim() } : {}),
        clear_api_key: clearKey,
        director_model: directorModel.trim(),
        director_base_url: directorBaseUrl.trim(),
        ...(directorApiKey.trim() ? { director_api_key: directorApiKey.trim() } : {}),
        clear_director_api_key: clearDirectorKey,
      });
      syncFromResponse(result);
      setApiKey("");
      setClearKey(false);
      setDirectorApiKey("");
      setClearDirectorKey(false);
      setMessage(result.configured ? "已保存" : "已保存（请补充 API Key）");
    } catch {
      setError("保存失败");
    } finally {
      setSaving(false);
    }
  }

  const inputClass =
    "h-8 w-full rounded border border-white/10 bg-[#181818] px-2.5 text-[12px] text-primary focus:border-[#6094c6] focus:outline-none disabled:opacity-50";
  return createPortal(
    <dialog
      ref={dialog}
      onCancel={(event) => {
        if (saving) event.preventDefault();
        else onClose();
      }}
      aria-labelledby="settings-title"
      className="m-auto w-[min(560px,calc(100vw-32px))] max-h-[calc(100vh-32px)] overflow-y-auto rounded-lg border border-white/10 bg-[#222222] p-0 text-primary shadow-2xl backdrop:bg-black/60"
    >
      <div className="flex items-center justify-between border-b border-white/10 px-5 py-3">
        <h1 id="settings-title" className="flex items-center gap-2 text-[14px] font-medium">
          <Settings className="h-4 w-4" />
          设置
        </h1>
        <button
          type="button"
          aria-label="关闭设置"
          disabled={saving}
          onClick={onClose}
          className="rounded p-1 text-secondary hover:bg-white/10 disabled:opacity-50"
        >
          <X className="h-4 w-4" />
        </button>
      </div>
      <form onSubmit={(event) => void save(event)} className="space-y-4 p-5 text-[12px]">
        <div className="flex items-center justify-between">
          <h2 className="font-medium text-primary">LLM 接口（主模型 / 翻译执行）</h2>
          <details className="text-[11px] text-muted">
            <summary className="cursor-pointer select-none hover:text-primary">说明</summary>
            <p className="mt-1 text-secondary">
              用于分批口语翻译、时长改写与 AI 助手；Key 仅存本地后端，不回显浏览器。
            </p>
          </details>
        </div>

        {loading ? (
          <p role="status" className="text-secondary">
            读取中…
          </p>
        ) : settings ? (
          <fieldset disabled={saving} className="space-y-3 disabled:opacity-60">
            <label className="block space-y-1">
              <span className="text-secondary">Base URL</span>
              <input
                type="url"
                required
                maxLength={2048}
                value={baseUrl}
                onChange={(event) => {
                  setBaseUrl(event.target.value);
                  setMessage("");
                }}
                placeholder="https://api.example.com/v1"
                autoComplete="off"
                spellCheck={false}
                className={inputClass}
              />
            </label>
            <label className="block space-y-1">
              <span className="text-secondary">模型</span>
              <input
                required
                maxLength={256}
                value={model}
                onChange={(event) => {
                  setModel(event.target.value);
                  setMessage("");
                }}
                placeholder="模型 ID"
                autoComplete="off"
                spellCheck={false}
                className={inputClass}
              />
            </label>
            <label className="block space-y-1">
              <span className="text-secondary">
                API Key · {settings.api_key_configured ? "已配置" : "未配置"}
              </span>
              <input
                type="password"
                maxLength={8192}
                value={apiKey}
                disabled={clearKey}
                onChange={(event) => {
                  setApiKey(event.target.value);
                  setMessage("");
                }}
                placeholder="留空保留现有 Key"
                autoComplete="new-password"
                spellCheck={false}
                className={inputClass}
              />
            </label>
            <label className="flex items-center gap-2 text-[11px] text-secondary">
              <input
                type="checkbox"
                checked={clearKey}
                onChange={(event) => {
                  setClearKey(event.target.checked);
                  setApiKey("");
                  setMessage("");
                }}
              />
              清除 Key
            </label>

            <div className="rounded-md border border-white/10 bg-[#1a1a1a]">
              <button
                type="button"
                aria-expanded={showDirectorSection}
                onClick={() => setShowDirectorSection((open) => !open)}
                className="flex w-full items-center justify-between px-3 py-2 text-left hover:bg-white/5"
              >
                <span className="flex items-center gap-1.5 font-medium text-primary">
                  {showDirectorSection ? (
                    <ChevronDown className="h-3.5 w-3.5 text-secondary" />
                  ) : (
                    <ChevronRight className="h-3.5 w-3.5 text-secondary" />
                  )}
                  导演模型（Director LLM · 可选）
                </span>
                <span
                  className={
                    settings.director_configured
                      ? "rounded bg-[#3A6EA5]/25 px-1.5 py-0.5 text-[10px] text-[#8bb5de]"
                      : "rounded bg-white/5 px-1.5 py-0.5 text-[10px] text-muted"
                  }
                >
                  {settings.director_configured ? "独立模型" : "默认复用主模型"}
                </span>
              </button>
              {showDirectorSection && (
                <div className="space-y-2.5 border-t border-white/10 px-3 py-3">
                  <p className="text-[11px] leading-relaxed text-muted">
                    负责 Pass 0 全片剧本统筹与 Pass 2 导演审校润色；导演模型名称留空时自动复用上方主模型。接口地址或 Key 留空则继承主模型配置。
                  </p>
                  <label className="block space-y-1">
                    <span className="text-secondary">导演模型名称</span>
                    <input
                      maxLength={256}
                      value={directorModel}
                      onChange={(event) => {
                        setDirectorModel(event.target.value);
                        setMessage("");
                      }}
                      placeholder="留空则复用主模型，如 claude-sonnet-4-5 / gpt-4o"
                      autoComplete="off"
                      spellCheck={false}
                      className={inputClass}
                    />
                  </label>
                  <label className="block space-y-1">
                    <span className="text-secondary">导演接口地址（可选）</span>
                    <input
                      type="url"
                      maxLength={2048}
                      value={directorBaseUrl}
                      onChange={(event) => {
                        setDirectorBaseUrl(event.target.value);
                        setMessage("");
                      }}
                      placeholder="留空则继承主模型接口地址"
                      autoComplete="off"
                      spellCheck={false}
                      className={inputClass}
                    />
                  </label>
                  <label className="block space-y-1">
                    <span className="text-secondary">
                      导演 API Key（可选） ·{" "}
                      {settings.director_api_key_configured ? "已配置独立 Key" : "继承主模型 Key"}
                    </span>
                    <input
                      type="password"
                      maxLength={8192}
                      value={directorApiKey}
                      disabled={clearDirectorKey}
                      onChange={(event) => {
                        setDirectorApiKey(event.target.value);
                        setMessage("");
                      }}
                      placeholder={
                        settings.director_api_key_configured
                          ? "留空保留现有独立 Key"
                          : "留空则继承主模型 API Key"
                      }
                      autoComplete="new-password"
                      spellCheck={false}
                      className={inputClass}
                    />
                  </label>
                  {settings.director_api_key_configured && (
                    <label className="flex items-center gap-2 text-[11px] text-secondary">
                      <input
                        type="checkbox"
                        checked={clearDirectorKey}
                        onChange={(event) => {
                          setClearDirectorKey(event.target.checked);
                          setDirectorApiKey("");
                          setMessage("");
                        }}
                      />
                      清除独立 Key
                    </label>
                  )}
                </div>
              )}
            </div>
          </fieldset>
        ) : (
          <button
            type="button"
            onClick={() => setReload((value) => value + 1)}
            className="text-[#8bb5de]"
          >
            重试
          </button>
        )}

        {error && (
          <p role="alert" className="text-status-error">
            {error}
          </p>
        )}
        {message && (
          <p role="status" className="text-status-ready">
            {message}
          </p>
        )}

        <TrainingSettingsPanel />

        <div className="flex justify-end gap-2 border-t border-white/10 pt-3">
          <button
            type="button"
            onClick={onClose}
            disabled={saving}
            className="h-8 rounded bg-white/5 px-4 hover:bg-white/10 disabled:opacity-50"
          >
            关闭
          </button>
          <button
            type="submit"
            disabled={loading || saving || !settings || !model.trim() || !baseUrl.trim()}
            className="h-8 rounded bg-[#3A6EA5] px-4 text-white hover:bg-[#467DB8] disabled:opacity-50"
          >
            {saving ? "保存中…" : "保存"}
          </button>
        </div>
      </form>
    </dialog>,
    document.body,
  );
}
