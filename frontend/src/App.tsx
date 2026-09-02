import { useState } from 'react'
import { FlaggedTransaction } from './api'
import TransactionTable from './components/TransactionTable'
import ShapExplanation from './components/ShapExplanation'
import GraphView from './components/GraphView'
import MetricsPanel from './components/MetricsPanel'
import { Shield, Activity, BarChart2, Network, Menu, X } from 'lucide-react'

type Tab = 'dashboard' | 'metrics'

export default function App() {
  const [selectedTx, setSelectedTx] = useState<FlaggedTransaction | null>(null)
  const [activeTab, setActiveTab] = useState<Tab>('dashboard')
  const [sidebarOpen, setSidebarOpen] = useState(true)

  return (
    <div className="min-h-screen flex flex-col" style={{ background: 'var(--color-bg-primary)' }}>
      {/* Top Nav */}
      <header className="sticky top-0 z-50 border-b border-white/8 backdrop-blur-xl" style={{ background: 'rgba(13, 15, 43, 0.85)' }}>
        <div className="max-w-screen-2xl mx-auto px-6 h-14 flex items-center gap-4">
          {/* Logo */}
          <div className="flex items-center gap-2.5 mr-6">
            <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-guard-500 to-guard-700 flex items-center justify-center shadow-lg shadow-guard-500/30">
              <Shield size={16} className="text-white" />
            </div>
            <span className="font-bold text-white text-sm tracking-tight">
              Transaction<span className="text-guard-400">Guard</span>
            </span>
          </div>

          {/* Nav */}
          <nav className="flex items-center gap-1">
            <button
              onClick={() => setActiveTab('dashboard')}
              className={`flex items-center gap-2 px-4 py-1.5 rounded-lg text-sm font-medium transition-all ${
                activeTab === 'dashboard'
                  ? 'bg-guard-500/15 text-guard-300 border border-guard-500/25'
                  : 'text-white/50 hover:text-white/80 hover:bg-white/5'
              }`}
            >
              <Activity size={14} />
              Dashboard
            </button>
            <button
              onClick={() => setActiveTab('metrics')}
              className={`flex items-center gap-2 px-4 py-1.5 rounded-lg text-sm font-medium transition-all ${
                activeTab === 'metrics'
                  ? 'bg-guard-500/15 text-guard-300 border border-guard-500/25'
                  : 'text-white/50 hover:text-white/80 hover:bg-white/5'
              }`}
            >
              <BarChart2 size={14} />
              Model Metrics
            </button>
          </nav>

          <div className="flex-1" />

          {/* Status indicator */}
          <div className="flex items-center gap-2 px-3 py-1.5 bg-emerald-500/10 border border-emerald-500/20 rounded-lg">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse-slow" />
            <span className="text-xs text-emerald-400 font-medium">API Connected</span>
          </div>

          <button
            onClick={() => setSidebarOpen(o => !o)}
            className="p-1.5 rounded-lg hover:bg-white/8 text-white/50 hover:text-white/80 transition-colors lg:hidden"
          >
            {sidebarOpen ? <X size={18} /> : <Menu size={18} />}
          </button>
        </div>
      </header>

      {/* Page content */}
      {activeTab === 'dashboard' ? (
        <div className="flex-1 max-w-screen-2xl mx-auto w-full px-6 py-5 grid gap-5" style={{
          gridTemplateColumns: sidebarOpen ? '1fr 420px' : '1fr',
          gridTemplateRows: 'auto',
        }}>
          {/* Main column: Transaction Table */}
          <div className="min-h-[600px] flex flex-col gap-5">
            <TransactionTable
              onSelectTransaction={setSelectedTx}
              selectedId={selectedTx?.transaction_id}
            />
          </div>

          {/* Sidebar: SHAP + Graph */}
          {sidebarOpen && (
            <div className="flex flex-col gap-5 min-w-0">
              <ShapExplanation transaction={selectedTx} />
              <GraphView transaction={selectedTx} />
            </div>
          )}
        </div>
      ) : (
        <div className="flex-1 max-w-4xl mx-auto w-full px-6 py-8">
          {/* Page header */}
          <div className="mb-8">
            <h1 className="text-2xl font-bold text-white mb-2">Model Performance</h1>
            <p className="text-white/50 text-sm">
              Comparison between Phase 2 (tabular features only) and Phase 3 (tabular + graph features).
              The graph-augmented model adds network-based signals to detect coordinated fraud rings.
            </p>
          </div>

          <MetricsPanel />

          {/* Architecture note */}
          <div className="mt-6 glass-card p-6">
            <div className="flex items-center gap-3 mb-4">
              <Network className="text-purple-400" size={18} />
              <h3 className="font-semibold text-white text-sm">Graph Feature Explanations</h3>
            </div>
            <div className="grid grid-cols-2 gap-4">
              {[
                {
                  feat: 'Degree Centrality',
                  desc: 'Counts how many unique accounts an account sends to (out-degree) or receives from (in-degree). Mule accounts have unusually high in-degree from many ring participants funneling money to them.',
                },
                {
                  feat: 'PageRank',
                  desc: 'Measures "influence" by propagating scores along edges iteratively. Ring hub accounts collect from many high-activity sources and accumulate disproportionate PageRank relative to their transaction count.',
                },
                {
                  feat: 'Clustering Coefficient',
                  desc: 'Fraction of an account\'s neighbors that also transact with each other. Fraud rings have high clustering (members transact only within the ring). Legitimate accounts have low clustering.',
                },
                {
                  feat: 'Shared Neighbors',
                  desc: 'How many accounts both the sender and receiver have previously transacted with. High shared-neighbor count means both parties are embedded in the same tight cluster — a layering ring signature.',
                },
                {
                  feat: 'Community Detection',
                  desc: 'Louvain algorithm groups densely-connected accounts. A high-value TRANSFER between different communities is a classic cross-community layering signal (originator in ring A, mule in ring B).',
                },
                {
                  feat: 'Cross-Community Flag',
                  desc: 'Binary flag: 1 if sender and receiver belong to different Louvain communities, 0 otherwise. Fraudulent layering transfers frequently cross community boundaries while legitimate transfers cluster within them.',
                },
              ].map(({ feat, desc }) => (
                <div key={feat} className="p-4 bg-white/3 rounded-xl border border-white/6">
                  <h4 className="text-xs font-semibold text-guard-300 mb-2">{feat}</h4>
                  <p className="text-xs text-white/50 leading-relaxed">{desc}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer className="border-t border-white/5 py-4 px-6">
        <div className="max-w-screen-2xl mx-auto flex items-center justify-between text-xs text-white/20">
          <span>TransactionGuard v1.0 · Portfolio project · Synthetic PaySim-schema dataset</span>
          <span>XGBoost + NetworkX + SHAP · FastAPI + React</span>
        </div>
      </footer>
    </div>
  )
}
