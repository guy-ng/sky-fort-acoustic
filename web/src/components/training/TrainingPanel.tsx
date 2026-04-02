import { useState } from 'react'
import { TrainSection } from './TrainSection'
import { EvalSection } from './EvalSection'
import { ModelsSection } from './ModelsSection'

type Section = 'train' | 'evaluate' | 'models'

function AccordionHeader({
  title,
  open,
  onClick,
}: {
  title: string
  open: boolean
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      className="text-xs uppercase tracking-wider font-semibold text-hud-text-dim hover:text-hud-text w-full flex items-center justify-between py-2 border-b border-hud-border"
    >
      <span>{title}</span>
      <span className="text-[10px]">{open ? '\u25BC' : '\u25B6'}</span>
    </button>
  )
}

export function TrainingPanel() {
  const [openSection, setOpenSection] = useState<Section | null>('train')

  function toggle(section: Section) {
    setOpenSection(prev => (prev === section ? null : section))
  }

  return (
    <div className="flex flex-col gap-0">
      <AccordionHeader
        title="TRAIN"
        open={openSection === 'train'}
        onClick={() => toggle('train')}
      />
      {openSection === 'train' && (
        <div className="py-2">
          <TrainSection />
        </div>
      )}

      <AccordionHeader
        title="EVALUATE"
        open={openSection === 'evaluate'}
        onClick={() => toggle('evaluate')}
      />
      {openSection === 'evaluate' && (
        <div className="py-2">
          <EvalSection />
        </div>
      )}

      <AccordionHeader
        title="MODELS"
        open={openSection === 'models'}
        onClick={() => toggle('models')}
      />
      {openSection === 'models' && (
        <div className="py-2">
          <ModelsSection />
        </div>
      )}
    </div>
  )
}
