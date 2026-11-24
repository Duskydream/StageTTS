import { useEffect, useState } from "react";
import {
  applyInferenceBaseModel,
  fetchTrainingSettings,
  fetchTtsSettings,
  saveTrainingSettings,
  saveTtsEngine,
  type TrainingSettingsResponse,
  type TtsEngineId,
  type TtsSettingsResponse,
} from "../services/api";

export function TrainingSettingsPanel() {
  const [data, setData] = useState<TrainingSettingsResponse | null>(null);
  const [ttsData, setTtsData] = useState<TtsSettingsResponse | null>(null);
  const [model, setModel] = useState<"v2" | "v2Pro" | "v2ProPlus">("v2Pro");
  const [gpt, setGpt] = useState("15");
  const [sovits, setSovits] = useState("10");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [reload, setReload] = useState(0);

  useEffect(() => {
    let active = true;
    setError("");
    void Promise.all([fetchTrainingSettings(), fetchTtsSettings()])
      .then(([trainingRes, ttsRes]) => {
        if (!active) return;
        setData(trainingRes);
        setTtsData(ttsRes);
        setModel(trainingRes.settings.base_model);
        setGpt(String(trainingRes.settings.gpt_epochs));
        setSovits(String(trainingRes.settings.sovits_epochs));
      })
      .catch(() => {
        if (active) setError("读取失败");
      });
    return () => {
      active = false;
    };
  }, [reload]);

  const validEpoch = (value: string) => /^\d+$/.test(value) && Number(value) >= 1 && Number(value) <= 100;
  const valid = validEpoch(gpt) && validEpoch(sovits);

  async function selectEngine(nextEngine: TtsEngineId) {
    if (!ttsData || busy || ttsData.engine === nextEngine) return;
    setBusy(true);
    setError("");
    setMessage("");
    try {
      const result = await saveTtsEngine(nextEngine);
      setTtsData(result);
      setMessage(nextEngine === "cosyvoice3" ? "配音引擎已切至 CosyVoice 3" : "配音引擎已切至 GPT-SoVITS");
    } catch (err) {
      setError(err instanceof Error ? err.message : "引擎切换失败");
    } finally {
      setBusy(false);
    }
  }

  async function save() {
    if (!data || !valid || busy) return;
    setBusy(true);
    setError("");
    setMessage("");
    try {
      const result = await saveTrainingSettings({
        base_model: model,
        gpt_epochs: Number(gpt),
        sovits_epochs: Number(sovits),
      });
      setData(result);
      setMessage("微调配置已保存");
    } catch {
      setError("保存失败");
    } finally {
      setBusy(false);
    }
  }

  async function applyInference() {
    if (!data || busy || !option?.available) return;
    setBusy(true);
    setError("");
    setMessage("");
    try {
      const result = await applyInferenceBaseModel(model);
      setData(result);
      setMessage(`默认推理已切至 ${model}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "切换失败");
    } finally {
      setBusy(false);
    }
  }

  const option = data?.models.find((entry) => entry.id === model);
  const selectedEngineOption = ttsData?.engines.find((entry) => entry.id === ttsData.engine);
  const field =
    "h-8 w-full rounded border border-white/10 bg-[#181818] px-2.5 text-[12px] text-primary disabled:opacity-50";

  return (
    <section
      className="space-y-3 border-t border-white/10 pt-4 text-[12px]"
      aria-labelledby="training-settings-title"
    >
      <div className="flex items-center justify-between gap-2">
        <h2 id="training-settings-title" className="font-medium text-primary">
          配音引擎 · {ttsData?.engine === "cosyvoice3" ? "CosyVoice 3" : `GPT-SoVITS (${data?.inference_version ?? "—"})`}
        </h2>
        <details className="text-[11px] text-muted">
          <summary className="cursor-pointer select-none hover:text-primary">说明</summary>
          <p className="mt-1 text-secondary">
            CosyVoice 3 无需微调即可直接零样本合成；GPT-SoVITS 支持角色音色微调，切换引擎会自动释放旧服务显存。
          </p>
        </details>
      </div>

      {ttsData && (
        <div className="space-y-1.5">
          <div className="grid grid-cols-2 gap-1.5 rounded bg-[#181818] p-1">
            {ttsData.engines.map((entry) => {
              const active = ttsData.engine === entry.id;
              return (
                <button
                  key={entry.id}
                  type="button"
                  disabled={busy || !entry.available}
                  onClick={() => void selectEngine(entry.id)}
                  title={
                    entry.available
                      ? entry.label
                      : `未就绪，缺少：${entry.missing_files.join("、")}`
                  }
                  className={`flex items-center justify-between rounded px-2.5 py-1.5 text-left text-[12px] transition-colors disabled:opacity-45 ${
                    active
                      ? "bg-[#3A6EA5] text-white"
                      : "text-secondary hover:bg-[#262626] hover:text-primary"
                  }`}
                >
                  <span className="truncate font-medium">{entry.label}</span>
                  <span className={`ml-1.5 shrink-0 text-[10px] ${active ? "text-white/80" : "text-muted"}`}>
                    {entry.available ? "就绪" : "未部署"}
                  </span>
                </button>
              );
            })}
          </div>
          {selectedEngineOption && !selectedEngineOption.available && (
            <p className="break-all text-[11px] text-status-warning">
              缺少：{selectedEngineOption.missing_files.join("、")}
            </p>
          )}
        </div>
      )}

      {data ? (
        <details open={ttsData?.engine !== "cosyvoice3"} className="rounded border border-white/10 bg-[#1B1B1B] px-3 py-2">
          <summary className="cursor-pointer select-none text-[11px] font-medium text-secondary hover:text-primary">
            GPT-SoVITS 底模与微调参数 · {data.inference_version ?? "—"}
          </summary>
          <fieldset
            disabled={busy}
            className="mt-3 space-y-3"
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                void save();
              }
            }}
          >
            <label className="block space-y-1">
              <span className="text-secondary">底模</span>
              <select
                className={field}
                value={model}
                onChange={(event) => {
                  setModel(event.target.value as typeof model);
                  setMessage("");
                }}
              >
                {data.models.map((entry) => (
                  <option key={entry.id} value={entry.id}>
                    {entry.id}
                    {entry.id === "v2Pro" ? "（推荐）" : ""}
                    {entry.available ? " · 就绪" : " · 缺权重"}
                  </option>
                ))}
              </select>
            </label>
            <div className="grid grid-cols-2 gap-3">
              <label className="block space-y-1">
                <span className="text-secondary">GPT 轮数</span>
                <input
                  aria-label="GPT 微调轮数"
                  type="number"
                  min={1}
                  max={100}
                  step={1}
                  value={gpt}
                  onChange={(event) => {
                    setGpt(event.target.value);
                    setMessage("");
                  }}
                  className={field}
                />
              </label>
              <label className="block space-y-1">
                <span className="text-secondary">SoVITS 轮数</span>
                <input
                  aria-label="SoVITS 微调轮数"
                  type="number"
                  min={1}
                  max={100}
                  step={1}
                  value={sovits}
                  onChange={(event) => {
                    setSovits(event.target.value);
                    setMessage("");
                  }}
                  className={field}
                />
              </label>
            </div>
            {!valid && <p className="text-[11px] text-status-warning">轮数范围 1–100</p>}
            {option && !option.available && (
              <p className="break-all text-[11px] text-status-warning">
                缺少：{option.missing_files.join("、")}
              </p>
            )}
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                disabled={busy || !valid}
                onClick={() => void save()}
                className="h-8 rounded bg-[#3A6EA5] px-3.5 text-white disabled:opacity-50"
              >
                {busy ? "处理中…" : "保存微调"}
              </button>
              <button
                type="button"
                disabled={busy || !option?.available || data.inference_version === model}
                onClick={() => void applyInference()}
                className="h-8 rounded border border-white/15 bg-[#242424] px-3.5 text-primary hover:bg-[#303030] disabled:opacity-50"
              >
                {data.inference_version === model ? `当前推理：${model}` : "设为默认推理"}
              </button>
            </div>
          </fieldset>
        </details>
      ) : (
        !error && <p className="text-secondary">读取中…</p>
      )}
      {error && (
        <div role="alert" className="text-status-error">
          {error}{" "}
          <button
            type="button"
            onClick={() => setReload((value) => value + 1)}
            className="underline"
          >
            重试
          </button>
        </div>
      )}
      {message && (
        <p role="status" className="text-status-ready">
          {message}
        </p>
      )}
    </section>
  );
}
