import { useModels } from '../../hooks/useModels'
import { useRunEvaluation } from '../../hooks/useEvaluation'

function formatSize(bytes: number): string {
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(0)} KB`
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function formatDate(modified: string): string {
  return new Date(modified).toLocaleDateString()
}

export function ModelsSection() {
  const { data: modelsData, isLoading } = useModels()
  const evalMutation = useRunEvaluation()

  if (isLoading) {
    return <div className="text-sm text-hud-text-dim text-center py-4">Loading models...</div>
  }

  if (!modelsData?.models.length) {
    return <div className="text-sm text-hud-text-dim text-center py-4">No models found</div>
  }

  return (
    <div className="flex flex-col">
      {modelsData.models.map(model => {
        // Active model heuristic (per D-10): check if filename contains "research_cnn"
        const isActive = model.filename.includes('research_cnn')

        return (
          <div
            key={model.path}
            className="flex items-center justify-between py-2 border-b border-hud-border last:border-b-0"
          >
            <div className="flex flex-col gap-0.5 min-w-0 flex-1">
              <div className="flex items-center gap-1.5">
                <span className="font-mono text-sm text-hud-text truncate">{model.filename}</span>
                {isActive && (
                  <span className="text-xs bg-hud-success/20 text-hud-success px-1.5 py-0.5 rounded">
                    ACTIVE
                  </span>
                )}
              </div>
              <div className="flex gap-2 text-xs text-hud-text-dim">
                <span>{formatSize(model.size_bytes)}</span>
                <span>{formatDate(model.modified)}</span>
              </div>
            </div>

            <button
              onClick={() => evalMutation.mutate({ model_path: model.path })}
              disabled={evalMutation.isPending}
              className="text-xs px-2 py-1 border border-hud-border rounded hover:border-hud-accent text-hud-text-dim hover:text-hud-text flex-shrink-0 ml-2"
            >
              Evaluate
            </button>
          </div>
        )
      })}
    </div>
  )
}
