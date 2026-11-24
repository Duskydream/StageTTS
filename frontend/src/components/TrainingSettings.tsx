import { useEffect, useState } from "react";
import {
  applyInferenceBaseModel,
  fetchTrainingSettings,
  saveTrainingSettings,
  type TrainingSettingsResponse,
} from "../services/api";

export function TrainingSettingsPanel() {
  const [data, setData] = useState<TrainingSettingsResponse | null>(null);
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
    void fetchTrainingSettings()
      .then((result) => {
        if (!active) return;
        setData(result);
        setModel(result.settings.base_model);
        setGpt(String(result.settings.gpt_epochs));
        setSovits(String(result.settings.sovits_epochs));
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
  const field =
    "h-8 w-full rounded border border-white/10 bg-[#181818] px-2.5 text-[12px] text-primary disabled:opacity-50";

  return (
    <section
      className="space-y-3 border-t border-white/10 pt-4 text-[12px]"
      aria-labelledby="training-settings-title"
    >
      <div className="flex items-center justify-between gap-2">
        <h2 id="training-settings-title" className="font-medium text-primary">
          GPT-SoVITS 模型 · {data?.inference_version ?? "—"}
        </h2>
        <details className="text-[11px] text-muted">
          <summary className="cursor-pointer select-none hover:text-primary">说明</summary>
          <p className="mt-1 text-secondary">
            建议先校对说话人归属再微调；默认 15/10 轮为起点，可通过角色面板一键启动。
          </p>
        </details>
      </div>

      {data ? (
        <fieldset
          disabled={busy}
          className="space-y-3"
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
