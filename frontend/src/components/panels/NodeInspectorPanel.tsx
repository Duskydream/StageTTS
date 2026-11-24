import { Terminal } from "lucide-react";
import type React from "react";
import { useWorkflowStore } from "../../store/useWorkflowStore";

/**
 * 节点流右侧常驻运行日志面板。
 */
export const PipelineLogsPanel: React.FC = () => {
  const pipelineLogs = useWorkflowStore((s) => s.pipelineLogs);

  return (
    <div className="flex h-full flex-col bg-[#1E1E1E] py-2.5 pl-3 pr-3.5 text-[12px]">
      <div className="mb-2 flex items-center text-[11px] font-medium text-secondary">
        <span className="flex items-center gap-1.5">
          <Terminal className="h-3.5 w-3.5 text-secondary" strokeWidth={1.5} />
          <span>日志</span>
        </span>
      </div>
      <div className="flex-1 space-y-1.5 overflow-y-auto rounded-[4px] bg-[#141414] p-2.5 font-mono tabular-nums text-[11px]">
        {pipelineLogs.map((log) => (
          <div key={log.id} className="leading-relaxed">
            <span className="text-muted">[{log.timestamp}]</span>{" "}
            <span
              className={
                log.level === "success"
                  ? "text-status-ready"
                  : log.level === "warn"
                    ? "text-status-warning"
                    : log.level === "error"
                      ? "text-status-error"
                      : "text-primary"
              }
            >
              {log.message}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};
