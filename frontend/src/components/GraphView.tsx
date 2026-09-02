import { useState, useEffect, useRef, useCallback } from 'react'
import { api, GraphResponse, FlaggedTransaction } from '../api'
import { Network, Loader2, Info } from 'lucide-react'

interface Props {
  transaction: FlaggedTransaction | null
}

// Color by community
const COMMUNITY_COLORS = [
  '#5561fa', '#f97316', '#22c55e', '#a855f7', '#ec4899',
  '#14b8a6', '#eab308', '#06b6d4', '#8b5cf6', '#6366f1',
]

function getCommunityColor(id: number): string {
  if (id < 0) return '#6b7280'
  return COMMUNITY_COLORS[id % COMMUNITY_COLORS.length]
}

export default function GraphView({ transaction }: Props) {
  const [graphData, setGraphData] = useState<GraphResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [hoveredNode, setHoveredNode] = useState<string | null>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animRef = useRef<number>(0)
  const nodesRef = useRef<Map<string, { x: number; y: number; vx: number; vy: number }>>(new Map())

  const accountId = transaction?.nameOrig || null

  useEffect(() => {
    if (!accountId) { setGraphData(null); return }
    setLoading(true)
    setError(null)
    api.getAccountGraph(accountId)
      .then(data => { setGraphData(data); nodesRef.current.clear() })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [accountId])

  // Simple force-directed simulation on canvas
  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas || !graphData) return
    const ctx = canvas.getContext('2d')!
    const W = canvas.width
    const H = canvas.height

    // Initialize node positions
    if (nodesRef.current.size === 0) {
      graphData.nodes.forEach((node, i) => {
        const angle = (i / graphData.nodes.length) * 2 * Math.PI
        const r = Math.min(W, H) * 0.35
        nodesRef.current.set(node.id, {
          x: W / 2 + r * Math.cos(angle) + (Math.random() - 0.5) * 20,
          y: H / 2 + r * Math.sin(angle) + (Math.random() - 0.5) * 20,
          vx: 0, vy: 0,
        })
      })
    }

    const positions = nodesRef.current
    const nodes = graphData.nodes
    const edges = graphData.edges

    // Force simulation step
    const k = Math.sqrt((W * H) / Math.max(nodes.length, 1)) * 0.8
    const gravity = 0.02
    const damping = 0.85

    // Reset forces
    positions.forEach(p => { p.vx *= damping; p.vy *= damping })

    // Repulsion between nodes
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const pi = positions.get(nodes[i].id)!
        const pj = positions.get(nodes[j].id)!
        const dx = pi.x - pj.x
        const dy = pi.y - pj.y
        const dist = Math.sqrt(dx * dx + dy * dy) + 1
        const force = (k * k) / (dist * dist) * 0.5
        pi.vx += (dx / dist) * force
        pi.vy += (dy / dist) * force
        pj.vx -= (dx / dist) * force
        pj.vy -= (dy / dist) * force
      }
    }

    // Attraction along edges
    edges.forEach(edge => {
      const ps = positions.get(edge.source)
      const pt = positions.get(edge.target)
      if (!ps || !pt) return
      const dx = pt.x - ps.x
      const dy = pt.y - ps.y
      const dist = Math.sqrt(dx * dx + dy * dy) + 1
      const ideal = k * 1.5
      const force = (dist - ideal) / dist * 0.05
      ps.vx += dx * force
      ps.vy += dy * force
      pt.vx -= dx * force
      pt.vy -= dy * force
    })

    // Gravity toward center
    positions.forEach(p => {
      p.vx += (W / 2 - p.x) * gravity
      p.vy += (H / 2 - p.y) * gravity
    })

    // Update positions
    positions.forEach(p => {
      p.x = Math.max(20, Math.min(W - 20, p.x + p.vx))
      p.y = Math.max(20, Math.min(H - 20, p.y + p.vy))
    })

    // Draw
    ctx.clearRect(0, 0, W, H)
    ctx.fillStyle = 'rgba(13, 15, 43, 0)'
    ctx.fillRect(0, 0, W, H)

    // Draw edges
    edges.forEach(edge => {
      const ps = positions.get(edge.source)
      const pt = positions.get(edge.target)
      if (!ps || !pt) return
      ctx.beginPath()
      ctx.moveTo(ps.x, ps.y)
      ctx.lineTo(pt.x, pt.y)
      ctx.strokeStyle = edge.is_fraud
        ? 'rgba(255, 54, 54, 0.6)'
        : 'rgba(255, 255, 255, 0.08)'
      ctx.lineWidth = edge.is_fraud ? 1.5 : 0.8
      ctx.stroke()

      // Arrow
      if (edge.is_fraud) {
        const angle = Math.atan2(pt.y - ps.y, pt.x - ps.x)
        const mx = (ps.x + pt.x) / 2
        const my = (ps.y + pt.y) / 2
        ctx.beginPath()
        ctx.moveTo(mx, my)
        ctx.lineTo(mx - 6 * Math.cos(angle - 0.4), my - 6 * Math.sin(angle - 0.4))
        ctx.lineTo(mx - 6 * Math.cos(angle + 0.4), my - 6 * Math.sin(angle + 0.4))
        ctx.closePath()
        ctx.fillStyle = 'rgba(255, 54, 54, 0.7)'
        ctx.fill()
      }
    })

    // Draw nodes
    nodes.forEach(node => {
      const p = positions.get(node.id)!
      const isCenter = node.id === accountId
      const isHovered = node.id === hoveredNode
      const r = isCenter ? 10 : node.is_flagged ? 8 : 5

      // Glow for flagged nodes
      if (node.is_flagged || isCenter) {
        ctx.beginPath()
        ctx.arc(p.x, p.y, r + 4, 0, 2 * Math.PI)
        ctx.fillStyle = isCenter
          ? 'rgba(85, 97, 250, 0.2)'
          : 'rgba(255, 54, 54, 0.2)'
        ctx.fill()
      }

      ctx.beginPath()
      ctx.arc(p.x, p.y, r, 0, 2 * Math.PI)

      if (isCenter) {
        ctx.fillStyle = '#5561fa'
        ctx.strokeStyle = 'rgba(255,255,255,0.8)'
        ctx.lineWidth = 2
        ctx.fill()
        ctx.stroke()
      } else if (node.is_flagged) {
        ctx.fillStyle = '#ff3636'
        ctx.strokeStyle = 'rgba(255,54,54,0.5)'
        ctx.lineWidth = 1
        ctx.fill()
        ctx.stroke()
      } else {
        ctx.fillStyle = getCommunityColor(node.community_id)
        ctx.globalAlpha = isHovered ? 1.0 : 0.7
        ctx.fill()
        ctx.globalAlpha = 1.0
      }

      // Label for center + hovered + flagged
      if (isCenter || isHovered || node.is_flagged) {
        ctx.font = `${isCenter ? 'bold ' : ''}9px JetBrains Mono, monospace`
        ctx.fillStyle = 'rgba(255,255,255,0.85)'
        ctx.fillText(node.id.slice(0, 11), p.x + r + 3, p.y + 3)
      }
    })

    animRef.current = requestAnimationFrame(draw)
  }, [graphData, accountId, hoveredNode])

  useEffect(() => {
    if (!graphData) return
    animRef.current = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(animRef.current)
  }, [graphData, draw])

  // Handle canvas resize
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const observer = new ResizeObserver(() => {
      const parent = canvas.parentElement!
      canvas.width = parent.offsetWidth
      canvas.height = parent.offsetHeight
    })
    observer.observe(canvas.parentElement!)
    const parent = canvas.parentElement!
    canvas.width = parent.offsetWidth
    canvas.height = parent.offsetHeight
    return () => observer.disconnect()
  }, [])

  // Mouse hover
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!graphData) return
    const canvas = canvasRef.current!
    const rect = canvas.getBoundingClientRect()
    const mx = e.clientX - rect.left
    const my = e.clientY - rect.top
    let found: string | null = null
    graphData.nodes.forEach(node => {
      const p = nodesRef.current.get(node.id)
      if (!p) return
      const dx = p.x - mx
      const dy = p.y - my
      if (dx * dx + dy * dy < 100) found = node.id
    })
    setHoveredNode(found)
  }, [graphData])

  if (!transaction) {
    return (
      <div className="glass-card flex flex-col items-center justify-center h-64 gap-3">
        <Network className="text-white/20" size={32} />
        <p className="text-white/30 text-sm">Select a transaction to explore its account graph</p>
      </div>
    )
  }

  return (
    <div className="glass-card flex flex-col">
      <div className="flex items-center justify-between p-5 border-b border-white/8">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-purple-500/20 flex items-center justify-center">
            <Network className="text-purple-400" size={16} />
          </div>
          <div>
            <h2 className="font-semibold text-white text-sm">Transaction Graph</h2>
            <p className="text-xs text-white/40">
              {graphData ? `${graphData.nodes.length} nodes · ${graphData.edges.length} edges (2-hop subgraph)` : accountId}
            </p>
          </div>
        </div>
        {graphData && (
          <div className="flex items-center gap-3 text-xs text-white/40">
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-guard-500" />
              Center account
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-danger-500" />
              Flagged
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-0.5 bg-danger-500/60" />
              Fraud edge
            </span>
          </div>
        )}
      </div>

      <div className="relative flex-1" style={{ minHeight: 380 }}>
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center gap-3 z-10">
            <Loader2 className="animate-spin text-guard-400" size={24} />
            <span className="text-white/40 text-sm">Building subgraph...</span>
          </div>
        )}
        {error && (
          <div className="absolute inset-0 flex items-center justify-center text-danger-400 text-sm p-8 text-center">
            {error}
          </div>
        )}
        <canvas
          ref={canvasRef}
          onMouseMove={handleMouseMove}
          style={{ width: '100%', height: '100%', cursor: hoveredNode ? 'pointer' : 'default' }}
        />
        {hoveredNode && graphData && (
          <div
            className="absolute bottom-4 left-4 glass-card p-3 text-xs pointer-events-none"
            style={{ maxWidth: 240 }}
          >
            {(() => {
              const node = graphData.nodes.find(n => n.id === hoveredNode)
              if (!node) return null
              return (
                <>
                  <p className="font-mono text-white/80 font-medium mb-2">{node.id}</p>
                  <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-white/50">
                    <span>Degree: <span className="text-white/70">{node.degree}</span></span>
                    <span>Community: <span className="text-white/70">{node.community_id}</span></span>
                    <span>PageRank: <span className="text-white/70">{node.pagerank.toFixed(5)}</span></span>
                    <span className={node.is_flagged ? 'text-danger-400' : 'text-emerald-400'}>
                      {node.is_flagged ? '⚠ Flagged' : '✓ Clean'}
                    </span>
                  </div>
                </>
              )
            })()}
          </div>
        )}
        {!graphData && !loading && !error && (
          <div className="absolute inset-0 flex items-center justify-center">
            <p className="text-white/20 text-sm">Graph will appear here</p>
          </div>
        )}
      </div>

      {graphData && (
        <div className="px-5 pb-4 pt-2 flex items-start gap-2 border-t border-white/5">
          <Info size={12} className="text-white/30 mt-0.5 shrink-0" />
          <p className="text-xs text-white/30">
            Node color = Louvain community. Colors match cross-community clusters (same-color nodes are in the same detected fraud ring). Hover nodes for details.
          </p>
        </div>
      )}
    </div>
  )
}
