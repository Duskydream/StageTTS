import {
  Background,
  BackgroundVariant,
  Controls,
  type NodeTypes,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
} from "@xyflow/react";
import type React from "react";
import { useMemo } from "react";
import "@xyflow/react/dist/style.css";
import { Maximize2, Play, RotateCcw } from "lucide-react";
import { type StageFlowNode, useWorkflowStore } from "../../store/useWorkflowStore";
import { pipelineNodeHint, pipelineStartForNode } from "../../types/workflow";
import { PipelineRunControls } from "./PipelineRunControls";
import { StageNode } from "./StageNode";

const PipelineGraphInner: React.FC = () => {
  const nodes = useWorkflowStore((s) => s.nodes);
  const edges = useWorkflowStore((s) => s.edges);
  const onNodesChange = useWorkflowStore((s) => s.onNodesChange);
  const onEdgesChange = useWorkflowStore((s) => s.onEdgesChange);
  const selectNode = useWorkflowStore((s) => s.selectNode);
  const resetGraphLayout = useWorkflowStore((s) => s.resetGraphLayout);
  const isPipelineRunning = useWorkflowStore((s) => s.isPipelineRunning);
  const runPipeline = useWorkflowStore((s) => s.runPipeline);
  const selectedNodeId = useWorkflowStore((s) => s.selectedNodeId);
  const startStep = pipelineStartForNode(selectedNodeId);
  const selectedTitle = nodes.find((node) => node.id === selectedNodeId)?.data.title;
  const canGenerateScript = startStep === "parse" || startStep === "translate";

  const { fitView } = useReactFlow();

  const nodeTypes = useMemo<NodeTypes>(
    () => ({
      stageNode: StageNode,
    }),
    [],
  );

  const handleReset = () => {
    resetGraphLayout();
    window.setTimeout(() => {
      void fitView({ padding: 0.08, duration: 250 });
    }, 30);
  };

  return (
    <div className="relative h-full w-full overflow-hidden bg-[#141414]">
      {/* 极简左上悬浮控制条（参数全部内嵌节点卡片） */}
      <div className="absolute left-3 top-3 z-10 flex w-fit max-w-[calc(100%-1.5rem)] flex-wrap items-center gap-1 rounded-[4px] border border-white/[0.08] bg-[#1E1E1E]/95 p-1 text-[12px]">
        <span
          className="flex h-7 items-center gap-1.5 rounded-[3px] bg-white/[0.04] px-2 text-[11px] text-secondary"
          title={startStep ? `从「${selectedTitle}」开始` : pipelineNodeHint(selectedNodeId)}
        >
          <span
            className={`h-1.5 w-1.5 rounded-full ${startStep ? "bg-status-ready" : selectedNodeId ? "bg-status-warning" : "bg-[#666666]"}`}
          />
          <span>{selectedTitle ?? "选择起点"}</span>
        </span>

        {canGenerateScript && (
          <button
            type="button"
            onClick={() => void runPipeline({ parseOnly: true })}
            disabled={isPipelineRunning}
            title={startStep === "translate" ? "仅翻译并保存剧本" : "仅解析并生成剧本"}
            className="flex h-7 items-center gap-1.5 rounded-[4px] border border-white/[0.08] bg-[#282828] px-2.5 font-medium text-primary transition-colors hover:bg-[#333333] disabled:opacity-45"
          >
            <Play className="h-3.5 w-3.5 text-secondary" strokeWidth={1.6} />
            <span>{startStep === "translate" ? "仅翻译" : "仅解析"}</span>
          </button>
        )}

        <button
          type="button"
          onClick={() => void runPipeline({ parseOnly: false })}
          disabled={isPipelineRunning || !startStep}
          title={startStep ? `从「${selectedTitle}」运行后续流程` : pipelineNodeHint(selectedNodeId)}
          className="flex h-7 items-center gap-1.5 rounded-[4px] bg-[#3A6EA5] px-2.5 font-medium text-white transition-colors hover:bg-[#467DB8] disabled:opacity-45"
        >
          <Play className="h-3.5 w-3.5" strokeWidth={1.6} />
          <span>{isPipelineRunning ? "运行中" : "运行"}</span>
        </button>

        <PipelineRunControls />

        <button
          type="button"
          title="适配画布"
          aria-label="适配画布"
          onClick={() => void fitView({ padding: 0.08, duration: 250 })}
          className="flex h-7 w-7 items-center justify-center rounded-[4px] text-secondary transition-colors hover:bg-[#2E2E2E] hover:text-primary"
        >
          <Maximize2 className="h-3.5 w-3.5" strokeWidth={1.6} />
        </button>

        <button
          type="button"
          title="重置布局"
          aria-label="重置布局"
          onClick={handleReset}
          disabled={isPipelineRunning}
          className="flex h-7 w-7 items-center justify-center rounded-[4px] text-secondary transition-colors hover:bg-[#2E2E2E] hover:text-primary disabled:opacity-45"
        >
          <RotateCcw className="h-3.5 w-3.5" strokeWidth={1.6} />
        </button>


      </div>

      <ReactFlow<StageFlowNode>
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={(_, node) => selectNode(node.id)}
        onPaneClick={() => selectNode(null)}
        multiSelectionKeyCode={null}
        fitView
        fitViewOptions={{ padding: 0.08 }}
        minZoom={0.4}
        maxZoom={1.8}
        proOptions={{ hideAttribution: true }}
      >
        <Background
          variant={BackgroundVariant.Lines}
          gap={24}
          lineWidth={1}
          color="rgba(255, 255, 255, 0.03)"
        />
        <Controls position="bottom-left" />
      </ReactFlow>
    </div>
  );
};

export const PipelineGraph: React.FC = () => {
  return (
    <ReactFlowProvider>
      <PipelineGraphInner />
    </ReactFlowProvider>
  );
};
