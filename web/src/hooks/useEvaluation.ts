import { useMutation } from '@tanstack/react-query'
import type { EvalRunParams, EvalResultResponse } from '../utils/types'

export function useRunEvaluation() {
  return useMutation<EvalResultResponse, Error, EvalRunParams>({
    mutationFn: async (params: EvalRunParams) => {
      const res = await fetch('/api/eval/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({ message: `Evaluation failed (${res.status})` }))
        throw new Error(body.message ?? `Evaluation failed (${res.status})`)
      }
      return res.json()
    },
  })
}
