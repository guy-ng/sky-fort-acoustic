import { useMutation } from '@tanstack/react-query'
import type { EvalRunParams, EvalResultResponse } from '../utils/types'

export function useRunEvaluation() {
  return useMutation<EvalResultResponse, Error, EvalRunParams>({
    mutationFn: (params: EvalRunParams) =>
      fetch('/api/eval/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      }).then(r => r.json()),
  })
}
