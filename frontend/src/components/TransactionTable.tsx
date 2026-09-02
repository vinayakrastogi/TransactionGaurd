import { useState, useEffect, useCallback } from 'react'
import { api, FlaggedTransaction } from '../api'
import {
  AlertTriangle, ChevronUp, ChevronDown, RefreshCw,
  Search, Filter, ExternalLink
} from 'lucide-react'

interface Props {
  onSelectTransaction: (tx: FlaggedTransaction) => void
  selectedId?: string
}

type SortField = 'fraud_prob' | 'amount' | 'step'
type SortDir = 'asc' | 'desc'

const TYPE_COLORS: Record<string, string> = {
  TRANSFER: 'text-purple-400 bg-purple-400/10',
  CASH_OUT: 'text-orange-400 bg-orange-400/10',
  CASH_IN:  'text-emerald-400 bg-emerald-400/10',
  PAYMENT:  'text-blue-400 bg-blue-400/10',
  DEBIT:    'text-yellow-400 bg-yellow-400/10',
}

function ProbBar({ prob }: { prob: number }) {
  const color = prob >= 0.8 ? '#ff3636' : prob >= 0.5 ? '#f97316' : '#22c55e'
  return (
    <div className="flex items-center gap-2">
      <div className="probability-bar flex-1" style={{ minWidth: 64 }}>
        <div
          style={{ width: `${prob * 100}%`, height: '100%', background: color, borderRadius: 2, transition: 'width 0.5s ease' }}
        />
      </div>
      <span className="font-mono text-xs" style={{ color, minWidth: 44 }}>
        {(prob * 100).toFixed(1)}%
      </span>
    </div>
  )
}

export default function TransactionTable({ onSelectTransaction, selectedId }: Props) {
  const [transactions, setTransactions] = useState<FlaggedTransaction[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [sortField, setSortField] = useState<SortField>('fraud_prob')
  const [sortDir, setSortDir] = useState<SortDir>('desc')
  const [search, setSearch] = useState('')
  const [minProb, setMinProb] = useState(0)
  const LIMIT = 20

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await api.getFlaggedTransactions(page, LIMIT, minProb)
      setTransactions(res.transactions)
      setTotal(res.total)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to load transactions')
    } finally {
      setLoading(false)
    }
  }, [page, minProb])

  useEffect(() => { load() }, [load])

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    } else {
      setSortField(field)
      setSortDir('desc')
    }
  }

  const sorted = [...transactions].sort((a, b) => {
    const va = a[sortField] ?? 0
    const vb = b[sortField] ?? 0
    const cmp = (va as number) - (vb as number)
    return sortDir === 'asc' ? cmp : -cmp
  })

  const filtered = sorted.filter(tx =>
    !search || tx.transaction_id.includes(search) ||
    tx.nameOrig.includes(search) || tx.nameDest.includes(search)
  )

  const SortIcon = ({ field }: { field: SortField }) => {
    if (sortField !== field) return <ChevronUp className="opacity-20" size={12} />
    return sortDir === 'asc' ? <ChevronUp size={12} /> : <ChevronDown size={12} />
  }

  const totalPages = Math.ceil(total / LIMIT)

  return (
    <div className="glass-card flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between p-5 border-b border-white/8">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-danger-500/20 flex items-center justify-center">
            <AlertTriangle className="text-danger-400" size={16} />
          </div>
          <div>
            <h2 className="font-semibold text-white text-sm">Flagged Transactions</h2>
            <p className="text-xs text-white/40">{total.toLocaleString()} total flagged</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {/* Min prob filter */}
          <div className="flex items-center gap-1.5 bg-white/5 rounded-lg px-3 py-1.5">
            <Filter size={12} className="text-white/40" />
            <span className="text-xs text-white/50">Min prob:</span>
            <select
              value={minProb}
              onChange={e => { setMinProb(Number(e.target.value)); setPage(1) }}
              className="bg-transparent text-xs text-white/80 outline-none cursor-pointer"
            >
              <option value={0}>All</option>
              <option value={0.5}>≥50%</option>
              <option value={0.7}>≥70%</option>
              <option value={0.9}>≥90%</option>
            </select>
          </div>
          <button
            onClick={load}
            disabled={loading}
            className="p-1.5 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
          >
            <RefreshCw size={14} className={loading ? 'animate-spin text-guard-400' : 'text-white/60'} />
          </button>
        </div>
      </div>

      {/* Search */}
      <div className="px-5 py-3 border-b border-white/5">
        <div className="relative">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-white/30" />
          <input
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search transaction ID or account..."
            className="w-full bg-white/5 border border-white/8 rounded-lg pl-8 pr-4 py-2 text-sm text-white/80 placeholder:text-white/25 outline-none focus:border-guard-500/50 transition-colors"
          />
        </div>
      </div>

      {/* Table */}
      <div className="overflow-auto flex-1">
        {error ? (
          <div className="flex items-center justify-center h-32 text-danger-400 text-sm">
            {error}
          </div>
        ) : (
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-[#0d0f2b]">
              <tr className="text-white/40 text-xs border-b border-white/5">
                <th className="text-left px-5 py-3 font-medium">Transaction ID</th>
                <th className="text-left px-3 py-3 font-medium">Type</th>
                <th
                  className="text-left px-3 py-3 font-medium cursor-pointer hover:text-white/70 select-none"
                  onClick={() => handleSort('amount')}
                >
                  <div className="flex items-center gap-1">
                    Amount <SortIcon field="amount" />
                  </div>
                </th>
                <th className="text-left px-3 py-3 font-medium">Origin → Dest</th>
                <th
                  className="text-left px-3 py-3 font-medium cursor-pointer hover:text-white/70 select-none"
                  onClick={() => handleSort('fraud_prob')}
                >
                  <div className="flex items-center gap-1">
                    Fraud Score <SortIcon field="fraud_prob" />
                  </div>
                </th>
                <th className="text-left px-3 py-3 font-medium">Label</th>
                <th className="px-3 py-3" />
              </tr>
            </thead>
            <tbody>
              {loading && filtered.length === 0 ? (
                [...Array(8)].map((_, i) => (
                  <tr key={i} className="border-b border-white/5">
                    {[...Array(7)].map((_, j) => (
                      <td key={j} className="px-5 py-3">
                        <div className="h-3 bg-white/5 rounded animate-pulse" style={{ width: `${60 + Math.random() * 40}%` }} />
                      </td>
                    ))}
                  </tr>
                ))
              ) : filtered.map(tx => (
                <tr
                  key={tx.transaction_id}
                  onClick={() => onSelectTransaction(tx)}
                  className={`table-row-hover border-b border-white/5 animate-fade-in ${
                    selectedId === tx.transaction_id
                      ? 'bg-guard-500/10 border-l-2 border-l-guard-400'
                      : ''
                  }`}
                >
                  <td className="px-5 py-3">
                    <span className="font-mono text-xs text-white/70">{tx.transaction_id}</span>
                  </td>
                  <td className="px-3 py-3">
                    <span className={`text-xs px-2 py-0.5 rounded-md font-medium ${TYPE_COLORS[tx.type] || 'text-white/60 bg-white/5'}`}>
                      {tx.type}
                    </span>
                  </td>
                  <td className="px-3 py-3">
                    <span className="font-mono text-xs text-white/80">
                      ${tx.amount.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                    </span>
                  </td>
                  <td className="px-3 py-3">
                    <div className="text-xs text-white/50 font-mono max-w-[180px]">
                      <span className="text-white/70">{tx.nameOrig.slice(0, 10)}</span>
                      <span className="text-white/25 mx-1">→</span>
                      <span>{tx.nameDest.slice(0, 10)}</span>
                    </div>
                  </td>
                  <td className="px-3 py-3 min-w-[140px]">
                    <ProbBar prob={tx.fraud_prob} />
                  </td>
                  <td className="px-3 py-3">
                    {tx.actual_label === 1 ? (
                      <span className="badge-fraud">Fraud</span>
                    ) : (
                      <span className="badge-legit">Legit</span>
                    )}
                  </td>
                  <td className="px-3 py-3">
                    <ExternalLink size={13} className="text-white/20 hover:text-guard-400 transition-colors" />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Pagination */}
      <div className="flex items-center justify-between px-5 py-3 border-t border-white/8">
        <span className="text-xs text-white/40">
          Page {page} of {totalPages || 1}
        </span>
        <div className="flex gap-1">
          <button
            onClick={() => setPage(p => Math.max(1, p - 1))}
            disabled={page <= 1}
            className="px-3 py-1.5 text-xs rounded-lg bg-white/5 hover:bg-white/10 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            ← Prev
          </button>
          <button
            onClick={() => setPage(p => Math.min(totalPages, p + 1))}
            disabled={page >= totalPages}
            className="px-3 py-1.5 text-xs rounded-lg bg-white/5 hover:bg-white/10 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            Next →
          </button>
        </div>
      </div>
    </div>
  )
}
