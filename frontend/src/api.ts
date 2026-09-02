// API client for TransactionGuard backend

const API_BASE = '/api'

export interface FlaggedTransaction {
  transaction_id: string
  fraud_prob: number
  is_fraud_predicted: boolean
  actual_label: number
  type: string
  amount: number
  nameOrig: string
  nameDest: string
  step?: number
}

export interface FeatureExplanation {
  feature: string
  shap_value: number
  direction: 'increases_fraud_score' | 'decreases_fraud_score'
  feature_value: number
  description: string
}

export interface ScoreResponse {
  transaction_id: string
  fraud_prob: number
  is_fraud_predicted: boolean
  threshold_used: number
  plain_english_reason: string
  top_features: FeatureExplanation[]
  transaction_info: {
    type: string
    amount: number
    orig: string
    dest: string
  }
}

export interface GraphNode {
  id: string
  is_flagged: boolean
  degree: number
  pagerank: number
  community_id: number
}

export interface GraphEdge {
  source: string
  target: string
  weight: number
  is_fraud: boolean
}

export interface GraphResponse {
  account_id: string
  nodes: GraphNode[]
  edges: GraphEdge[]
}

export interface MetricSet {
  model: string
  precision: number
  recall: number
  f1: number
  pr_auc: number
  tp: number
  fp: number
  tn: number
  fn: number
  threshold: number
}

export interface MetricsResponse {
  baseline: MetricSet
  graph: MetricSet
  improvement: Record<string, number>
}

export interface FlaggedTransactionsResponse {
  total: number
  page: number
  limit: number
  transactions: FlaggedTransaction[]
}

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(error.detail || `HTTP ${response.status}`)
  }
  return response.json()
}

export const api = {
  getFlaggedTransactions: (page = 1, limit = 20, minProb = 0) =>
    apiFetch<FlaggedTransactionsResponse>(
      `/transactions/flagged?page=${page}&limit=${limit}&min_prob=${minProb}`
    ),

  scoreTransaction: (tx: { transaction_id: string; [key: string]: unknown }) =>
    apiFetch<ScoreResponse>('/score', {
      method: 'POST',
      body: JSON.stringify(tx),
    }),

  getAccountGraph: (accountId: string) =>
    apiFetch<GraphResponse>(`/graph/ring/${encodeURIComponent(accountId)}`),

  getMetrics: () => apiFetch<MetricsResponse>('/metrics'),
}
