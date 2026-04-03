import { useState } from 'react'
import { useModels } from '../../hooks/useModels'
import { useActivateModel } from '../../hooks/usePipeline'

function formatSize(bytes: number): string {
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(0)} KB`
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function formatDate(modified: string): string {
  return new Date(modified).toLocaleDateString()
}

interface ModelsSectionProps {
  onEvaluateModel: (modelPath: string) => void
}

export function ModelsSection({ onEvaluateModel }: ModelsSectionProps) {
  const { data: modelsData, isLoading } = useModels()
  const activateMutation = useActivateModel()
  const [activatedPath, setActivatedPath] = useState<string | null>(null)

  if (isLoading) {
    return <div className="text-sm text-hud-text-dim text-center py-4">Loading models...</div>
  }

  if (!modelsData?.models.length) {
    return <div className="text-sm text-hud-text-dim text-center py-4">No models found</div>
  }

  function handleActivate(modelPath: string) {
    activateMutation.mutate(
      { model_path: modelPath },
      { onSuccess: () => setActivatedPath(modelPath) },
    )
  }

  return (
    <div className="flex flex-col">
      {/* Activation status feedback */}
      {activateMutation.isError && (
        <div className="text-hud-danger text-xs py-1">
          {activateMutation.error?.message ?? 'Activation failed'}
        </div>
      )}
      {activateMutation.isSuccess && (
        <div className="text-hud-success text-xs py-1">
          {activateMutation.data?.message}
        </div>
      )}

      {modelsData.models.map(model => {
        const isActive = activatedPath
          ? model.path === activatedPath
          : model.filename.includes('research_cnn')

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

            <div className="flex gap-1 shrink-0 ml-2">
              <button
                onClick={() => onEvaluateModel(model.path)}
                className="text-xs px-2 py-1 border border-hud-border rounded hover:border-hud-accent text-hud-text-dim hover:text-hud-text"
              >
                Eval
              </button>
              <button
                onClick={() => handleActivate(model.path)}
                disabled={activateMutation.isPending || isActive}
                className={`text-xs px-2 py-1 rounded ${
                  isActive
                    ? 'bg-hud-success/20 text-hud-success cursor-default'
                    : activateMutation.isPending
                      ? 'bg-hud-border text-hud-text-dim cursor-not-allowed'
                      : 'bg-hud-accent/20 text-hud-accent border border-hud-accent/50 hover:bg-hud-accent/30'
                }`}
              >
                {activateMutation.isPending ? '...' : isActive ? 'Active' : 'Activate'}
              </button>
            </div>
          </div>
        )
      })}
    </div>
  )
}
