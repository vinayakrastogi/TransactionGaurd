import { useState, useEffect } from 'react'
import { api, MetricsResponse } from '../api'
import { BarChart2, TrendingUp, TrendingDown, Minus, Loader2 } from 'lucide-react'

function MetricCell({ label, value, delta }: { label: string; value: number; delta?: number }) {
  return (
    <div className="text-center">
      <p className="text-xs text-white/35 mb-1">{label}</p>
      <p className="text-base font-bold font-mono text-white">{(value * 100).toFixed(1)}%</p>
      {delta !== undefined && (
        <p className={`text-xs font-mono flex items-center justify-center gap-0.5 mt-0.5 ${
          delta > 0.001 ? 'text-emerald-400' : delta < -0.001 ? 'text-danger-400' : 'text-white/30'
        }`}>
          {delta > 0.001 ? <TrendingUp size={10} /> : delta < -0.001 ? <TrendingDown size={10} /> : <Minus size={10} />}
          {delta >= 0 ? '+' : ''}{(delta * 100).toFixed(2)}%
        </p>
      )}
    </div>
  )
}

function ConfusionMatrix({ tp, fp, tn, fn }: { tp: number; fp: number; tn: number; fn: number }) {
  const total = tp + fp + tn + fn
  const cells = [
    { label: 'True Neg', value: tn, color: 'bg-emerald-500/15 border-emerald-500/20' },
    { label: 'False Pos', value: fp, color: 'bg-orange-500/15 border-orange-500/20' },
    { label: 'False Neg', value: fn, color: 'bg-red-500/15 border-red-500/20' },
    { label: 'True Pos',  value: tp, color: 'bg-guard-500/15 border-guard-500/20' },
  ]
  return (
    <div className="grid grid-cols-2 gap-2 mt-3">
      {cells.map(c => (
        <div key={c.label} className={`rounded-lg p-3 border ${c.color}`}>
          <p className="text-xs text-white/40 mb-1">{c.label}</p>
          <p className="font-mono font-bold text-white text-sm">{c.value.toLocaleString()}</p>
          <p className="text-xs text-white/30">{(c.value / total * 100).toFixed(2)}%</p>
        </div>
      ))}
    </div>
  )
}

export default function MetricsPanel() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [expanded, setExpanded] = useState<'baseline' | 'graph' | null>(null)

  useEffect(() => {
    setLoading(true)
    api.getMetrics()
      .then(setMetrics)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="glass-card">
      <div className="flex items-center gap-3 p-5 border-b border-white/8">
        <div className="w-8 h-8 rounded-lg bg-emerald-500/20 flex items-center justify-center">
          <BarChart2 className="text-emerald-400" size={16} />
        </div>
        <div>
          <h2 className="font-semibold text-white text-sm">Model Performance</h2>
          <p className="text-xs text-white/40">Phase 2 (Tabular) vs Phase 3 (+ Graph features)</p>
        </div>
      </div>

      <div className="p-5">
        {loading && (
          <div className="flex items-center justify-center h-32 gap-3">
            <Loader2 className="animate-spin text-guard-400" size={20} />
            <span className="text-white/40 text-sm">Loading metrics...</span>
          </div>
        )}
        {error && <p className="text-danger-400 text-sm">{error}</p>}

        {metrics && (
          <>
            {/* Main comparison table */}
            <div className="overflow-x-auto">
              <table className="w-full text-sm border-collapse">
                <thead>
                  <tr className="border-b border-white/8">
                    <th className="text-left py-2 pr-4 text-xs text-white/40 font-medium uppercase tracking-wider">Metric</th>
                    <th className="text-center py-2 px-3 text-xs text-white/40 font-medium uppercase tracking-wider">
                      Phase 2<br /><span className="normal-case text-white/25 tracking-normal">Tabular baseline</span>
                    </th>
                    <th className="text-center py-2 px-3 text-xs text-guard-400 font-medium uppercase tracking-wider">
                      Phase 3<br /><span className="normal-case text-guard-300/50 tracking-normal">+ Graph features</span>
                    </th>
                    <th className="text-center py-2 pl-3 text-xs text-white/40 font-medium uppercase tracking-wider">Change</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    { key: 'precision', label: 'Precision' },
                    { key: 'recall',    label: 'Recall' },
                    { key: 'f1',        label: 'F1 Score' },
                    { key: 'pr_auc',    label: 'PR-AUC ★' },
                  ].map(({ key, label }) => {
                    const base  = metrics.baseline[key as keyof typeof metrics.baseline] as number
                    const graph = metrics.graph[key as keyof typeof metrics.graph] as number
                    const delta = graph - base
                    const isPrimary = key === 'pr_auc'
                    return (
                      <tr key={key} className={`border-b border-white/5 ${isPrimary ? 'bg-white/3' : ''}`}>
                        <td className={`py-2.5 pr-4 text-xs font-medium ${isPrimary ? 'text-white' : 'text-white/60'}`}>
                          {label}
                        </td>
                        <td className="py-2.5 px-3 text-center">
                          <span className="font-mono text-sm text-white/70">{(base * 100).toFixed(1)}%</span>
                        </td>
                        <td className="py-2.5 px-3 text-center">
                          <span className={`font-mono text-sm font-semibold ${isPrimary ? 'text-guard-300' : 'text-white/80'}`}>
                            {(graph * 100).toFixed(1)}%
                          </span>
                        </td>
                        <td className="py-2.5 pl-3 text-center">
                          <span className={`font-mono text-xs flex items-center justify-center gap-0.5 ${
                            delta > 0.001 ? 'text-emerald-400' : delta < -0.001 ? 'text-danger-400' : 'text-white/30'
                          }`}>
                            {delta > 0.001 ? <TrendingUp size={10} /> : delta < -0.001 ? <TrendingDown size={10} /> : <Minus size={10} />}
                            {delta >= 0 ? '+' : ''}{(delta * 100).toFixed(2)}%
                          </span>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>

            {/* Why PR-AUC callout */}
            <div className="mt-4 p-3 bg-guard-500/8 border border-guard-500/20 rounded-lg">
              <p className="text-xs text-guard-300/80 leading-relaxed">
                <strong className="text-guard-300">★ Primary metric is PR-AUC</strong>, not accuracy or ROC-AUC.
                At 0.15% fraud rate, a model that flags <em>nothing</em> has 99.8% accuracy.
                PR-AUC directly measures the precision/recall tradeoff on the minority fraud class.
              </p>
            </div>

            {/* Expandable confusion matrices */}
            <div className="mt-4 grid grid-cols-2 gap-3">
              {[
                { key: 'baseline', label: 'Phase 2 — Tabular', m: metrics.baseline },
                { key: 'graph',    label: 'Phase 3 — + Graph', m: metrics.graph },
              ].map(({ key, label, m }) => (
                <div key={key}>
                  <button
                    onClick={() => setExpanded(expanded === key as 'baseline' | 'graph' ? null : key as 'baseline' | 'graph')}
                    className="w-full text-left text-xs text-white/40 hover:text-white/60 transition-colors flex items-center gap-1"
                  >
                    <span className={`transition-transform ${expanded === key ? 'rotate-90' : ''}`}>▶</span>
                    {label}
                  </button>
                  {expanded === key && (
                    <ConfusionMatrix tp={m.tp} fp={m.fp} tn={m.tn} fn={m.fn} />
                  )}
                </div>
              ))}
            </div>

            {/* Threshold info */}
            <div className="mt-4 flex gap-3 text-xs text-white/35">
              <span>Baseline threshold: <span className="font-mono text-white/55">{metrics.baseline.threshold.toFixed(4)}</span></span>
              <span>·</span>
              <span>Graph threshold: <span className="font-mono text-white/55">{metrics.graph.threshold.toFixed(4)}</span></span>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
