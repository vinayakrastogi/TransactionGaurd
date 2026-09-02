import { useState, useEffect } from 'react'
import { api, ScoreResponse, FeatureExplanation, FlaggedTransaction } from '../api'
import { BarChart, Bar, XAxis, YAxis, Cell, Tooltip, ResponsiveContainer } from 'recharts'
import { Brain, TrendingUp, TrendingDown, AlertCircle, CheckCircle2, Loader2 } from 'lucide-react'

interface Props {
  transaction: FlaggedTransaction | null
}

function ShapBar({ features }: { features: FeatureExplanation[] }) {
  const maxAbs = Math.max(...features.map(f => Math.abs(f.shap_value)), 0.001)
  const data = features.map(f => ({
    name: f.feature.replace(/_/g, ' '),
    value: f.shap_value,
    abs: Math.abs(f.shap_value),
    description: f.description,
    feature_value: f.feature_value,
    direction: f.direction,
  })).sort((a, b) => b.abs - a.abs)

  const CustomTooltip = ({ active, payload }: { active?: boolean; payload?: { payload: typeof data[0] }[] }) => {
    if (!active || !payload?.length) return null
    const d = payload[0].payload
    return (
      <div className="glass-card p-3 text-xs max-w-[260px]">
        <p className="text-white font-medium mb-1">{d.name}</p>
        <p className="text-white/50 mb-2">{d.description}</p>
        <p className="font-mono">
          Value: <span className="text-guard-300">{d.feature_value.toFixed(4)}</span>
        </p>
        <p className="font-mono">
          SHAP: <span className={d.value > 0 ? 'text-danger-400' : 'text-emerald-400'}>
            {d.value > 0 ? '+' : ''}{d.value.toFixed(4)}
          </span>
        </p>
      </div>
    )
  }

  return (
    <div className="mt-4">
      <p className="text-xs text-white/40 mb-3 uppercase tracking-wider">Top Feature Contributions</p>
      <ResponsiveContainer width="100%" height={data.length * 40 + 20}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 0, right: 16, left: 0, bottom: 0 }}
        >
          <XAxis
            type="number"
            domain={[-maxAbs * 1.1, maxAbs * 1.1]}
            tick={{ fill: 'rgba(255,255,255,0.3)', fontSize: 10 }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            dataKey="name"
            type="category"
            width={160}
            tick={{ fill: 'rgba(255,255,255,0.6)', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
            {data.map((entry, i) => (
              <Cell
                key={i}
                fill={entry.value > 0
                  ? `rgba(255, 54, 54, ${0.5 + 0.5 * (entry.abs / maxAbs)})`
                  : `rgba(34, 197, 94, ${0.5 + 0.5 * (entry.abs / maxAbs)})`
                }
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

export default function ShapExplanation({ transaction }: Props) {
  const [explanation, setExplanation] = useState<ScoreResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!transaction) { setExplanation(null); return }
    setLoading(true)
    setError(null)

    // Build a minimal transaction payload — backend will use test set if ID matches
    api.scoreTransaction({
      transaction_id: transaction.transaction_id,
      type: transaction.type,
      amount: transaction.amount,
      nameOrig: transaction.nameOrig,
      oldbalanceOrg: 0,
      newbalanceOrig: 0,
      nameDest: transaction.nameDest,
      oldbalanceDest: 0,
      newbalanceDest: 0,
    })
      .then(res => setExplanation(res))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [transaction?.transaction_id])

  if (!transaction) {
    return (
      <div className="glass-card flex flex-col items-center justify-center h-64 gap-3">
        <Brain className="text-white/20" size={32} />
        <p className="text-white/30 text-sm">Select a transaction to see its SHAP explanation</p>
      </div>
    )
  }

  return (
    <div className="glass-card animate-slide-up">
      {/* Header */}
      <div className="flex items-center justify-between p-5 border-b border-white/8">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-guard-500/20 flex items-center justify-center">
            <Brain className="text-guard-400" size={16} />
          </div>
          <div>
            <h2 className="font-semibold text-white text-sm">SHAP Explanation</h2>
            <p className="text-xs text-white/40 font-mono">{transaction.transaction_id}</p>
          </div>
        </div>
        {explanation && (
          <div className="flex items-center gap-2">
            {explanation.is_fraud_predicted ? (
              <AlertCircle className="text-danger-400" size={18} />
            ) : (
              <CheckCircle2 className="text-emerald-400" size={18} />
            )}
            <div className="text-right">
              <div className={`text-lg font-bold font-mono ${explanation.fraud_prob > 0.5 ? 'text-danger-400' : 'text-emerald-400'}`}>
                {(explanation.fraud_prob * 100).toFixed(1)}%
              </div>
              <div className="text-xs text-white/40">fraud probability</div>
            </div>
          </div>
        )}
      </div>

      <div className="p-5">
        {loading ? (
          <div className="flex items-center justify-center h-48 gap-3">
            <Loader2 className="animate-spin text-guard-400" size={24} />
            <span className="text-white/40 text-sm">Computing SHAP values...</span>
          </div>
        ) : error ? (
          <div className="text-danger-400 text-sm p-4 bg-danger-500/10 rounded-lg">
            {error}
          </div>
        ) : explanation ? (
          <>
            {/* Transaction Info */}
            <div className="grid grid-cols-4 gap-3 mb-5">
              {[
                { label: 'Type', value: explanation.transaction_info.type },
                { label: 'Amount', value: `$${explanation.transaction_info.amount.toLocaleString(undefined, { maximumFractionDigits: 0 })}` },
                { label: 'Origin', value: explanation.transaction_info.orig?.slice(0, 12) },
                { label: 'Dest', value: explanation.transaction_info.dest?.slice(0, 12) },
              ].map(({ label, value }) => (
                <div key={label} className="bg-white/4 rounded-lg p-3">
                  <p className="text-xs text-white/35 mb-1">{label}</p>
                  <p className="text-sm font-mono text-white/80 font-medium truncate">{value}</p>
                </div>
              ))}
            </div>

            {/* Plain-English Reason */}
            <div className={`rounded-lg p-4 mb-5 border ${
              explanation.is_fraud_predicted
                ? 'bg-danger-500/10 border-danger-500/25'
                : 'bg-emerald-500/10 border-emerald-500/25'
            }`}>
              <div className="flex items-start gap-2">
                {explanation.is_fraud_predicted
                  ? <TrendingUp className="text-danger-400 mt-0.5 shrink-0" size={14} />
                  : <TrendingDown className="text-emerald-400 mt-0.5 shrink-0" size={14} />
                }
                <p className={`text-sm leading-relaxed ${explanation.is_fraud_predicted ? 'text-danger-200' : 'text-emerald-200'}`}>
                  {explanation.plain_english_reason}
                </p>
              </div>
            </div>

            {/* Feature details */}
            <div className="mb-4">
              <ShapBar features={explanation.top_features} />
            </div>

            {/* Feature list */}
            <div className="space-y-2">
              {explanation.top_features.map((feat, i) => (
                <div key={feat.feature} className="flex items-center gap-3 p-3 bg-white/3 rounded-lg hover:bg-white/5 transition-colors">
                  <span className="text-xs text-white/25 font-mono w-4">{i + 1}</span>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono text-white/70 font-medium">{feat.feature}</span>
                      {feat.direction === 'increases_fraud_score'
                        ? <TrendingUp size={10} className="text-danger-400" />
                        : <TrendingDown size={10} className="text-emerald-400" />
                      }
                    </div>
                    <p className="text-xs text-white/35 truncate mt-0.5">{feat.description}</p>
                  </div>
                  <div className="text-right shrink-0">
                    <span className={`font-mono text-xs font-bold ${feat.shap_value > 0 ? 'text-danger-400' : 'text-emerald-400'}`}>
                      {feat.shap_value > 0 ? '+' : ''}{feat.shap_value.toFixed(4)}
                    </span>
                    <p className="text-xs text-white/25 font-mono">{feat.feature_value.toFixed(2)}</p>
                  </div>
                </div>
              ))}
            </div>
          </>
        ) : null}
      </div>
    </div>
  )
}
