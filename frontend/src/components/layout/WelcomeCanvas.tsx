import { ChevronRight, Clapperboard, FolderOpen, Play, Upload } from "lucide-react";
import type React from "react";
import { useRef, useState } from "react";
import { useProjectStore } from "../../store/useProjectStore";

/**
 * Premiere 风格极简项目启动器（去口号、图标+操作直达）。
 */
export const WelcomeCanvas: React.FC = () => {
  const projectList = useProjectStore((s) => s.projectList);
  const backendStatus = useProjectStore((s) => s.backendStatus);
  const isLoading = useProjectStore((s) => s.isLoading);
  const importMediaFile = useProjectStore((s) => s.importMediaFile);
  const loadProjectScript = useProjectStore((s) => s.loadProjectScript);
  const loadDemoProject = useProjectStore((s) => s.loadDemoProject);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFiles = (files: FileList | null) => {
    const file = files?.[0];
    if (file) {
      void importMediaFile(file);
    }
  };

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setIsDragging(false);
        handleFiles(e.dataTransfer.files);
      }}
      className="flex h-full w-full items-center justify-center bg-[#161616] p-6 text-[12px]"
    >
      <div
        className={`w-full max-w-xl overflow-hidden rounded-[4px] border bg-[#222222] transition-colors ${
          isDragging ? "border-[#3A6EA5]" : "border-white/[0.06]"
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*,audio/*"
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />

        <div className="flex h-9 items-center justify-between border-b border-white/[0.06] bg-[#1C1C1C] px-3.5">
          <div className="flex items-center gap-2">
            <Clapperboard className="h-4 w-4 text-secondary" strokeWidth={1.5} />
            <span className="text-[13px] font-medium text-primary">StageTTS</span>
          </div>
          <span
            title={
              backendStatus === "online"
                ? "后端已连接"
                : backendStatus === "connecting"
                  ? "连接中..."
                  : "离线模式"
            }
            className={`h-2 w-2 rounded-full ${
              backendStatus === "online"
                ? "bg-status-ready"
                : backendStatus === "connecting"
                  ? "bg-secondary animate-pulse"
                  : "bg-status-warning"
            }`}
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-[200px_1fr]">
          <div className="space-y-1 border-b border-white/[0.06] p-2.5 md:border-b-0 md:border-r">
            <button
              type="button"
              disabled={isLoading}
              onClick={() => fileInputRef.current?.click()}
              title="选择或拖拽 MP4 / WAV"
              className="group flex w-full items-center justify-between rounded-[4px] px-2.5 py-2 text-left transition-colors hover:bg-[#2C2C2C] disabled:opacity-50"
            >
              <div className="flex items-center gap-2">
                <Upload className="h-4 w-4 text-secondary group-hover:text-primary" strokeWidth={1.5} />
                <span className="font-medium text-primary">导入媒体</span>
              </div>
              <ChevronRight className="h-3.5 w-3.5 text-muted group-hover:text-primary" strokeWidth={1.5} />
            </button>

            <button
              type="button"
              disabled={isLoading}
              onClick={() => void loadDemoProject()}
              className="group flex w-full items-center justify-between rounded-[4px] px-2.5 py-2 text-left transition-colors hover:bg-[#2C2C2C] disabled:opacity-50"
            >
              <div className="flex items-center gap-2">
                <Play className="h-4 w-4 text-secondary group-hover:text-primary" strokeWidth={1.5} />
                <span className="font-medium text-primary">演示样例</span>
              </div>
              <ChevronRight className="h-3.5 w-3.5 text-muted group-hover:text-primary" strokeWidth={1.5} />
            </button>
          </div>

          <div className="flex min-h-[200px] flex-col">
            <div className="flex h-8 items-center justify-between border-b border-white/[0.06] bg-[#1D1D1D] px-3 text-[11px] text-secondary">
              <span className="flex items-center gap-1.5 font-medium">
                <FolderOpen className="h-3.5 w-3.5 text-secondary" strokeWidth={1.5} />
                <span>最近项目</span>
              </span>
              <span className="font-mono tabular-nums text-muted">{projectList.length}</span>
            </div>

            {projectList.length > 0 ? (
              <div className="flex-1 overflow-y-auto">
                <table className="w-full border-collapse text-left text-[12px]">
                  <tbody>
                    {projectList.map((projId) => (
                      <tr
                        key={projId}
                        className="border-b border-white/[0.05] transition-colors hover:bg-[#2C2C2C]"
                      >
                        <td className="px-3 py-1.5 font-mono tabular-nums text-primary">{projId}</td>
                        <td className="w-16 px-3 py-1.5 text-right">
                          <button
                            type="button"
                            disabled={isLoading}
                            onClick={() => void loadProjectScript(projId)}
                            className="h-6 rounded-[4px] border border-white/[0.08] bg-[#2C2C2C] px-2 text-[11px] text-primary transition-colors hover:bg-[#363636] disabled:opacity-50"
                          >
                            打开
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="flex flex-1 items-center justify-center px-6 py-8 text-[11px] text-muted">
                {backendStatus === "online" ? "暂无项目" : "未连接后端"}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
