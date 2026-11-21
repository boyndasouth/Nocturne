// The memory graph, drawn as a night sky: facts are stars, themes are the bright bodies they
// orbit, outdated facts are hollow, and "replaced by" edges are dashed.
import { useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D, { type ForceGraphMethods, type LinkObject, type NodeObject } from "react-force-graph-2d";
import type { Edge, Graph, LinkType, Note } from "../api";

type N = NodeObject<Note & { degree: number }>;
type L = LinkObject<N, { link_type: LinkType }>;

const C = {
  fact: "#9cc3e6",
  theme: "#d6b2ec",
  episode: "#8fcab6",
  outdated: "#7c829f",
  moon: "#f1eff8",
  dim: "rgba(160,166,196,0.10)",
};

function radius(n: N): number {
  if (n.tier === "theme") return 7 + Math.min(n.degree, 10) * 0.45;
  if (n.tier === "episode") return 1.8;
  return 2.4 + 2.2 * n.strength + Math.min(n.degree, 6) * 0.25;
}

function clip(s: string, max: number): string {
  return s.length > max ? `${s.slice(0, max - 1).trimEnd()}…` : s;
}

export function Sky(props: {
  graph: Graph;
  selected: string | null;
  query: string;
  onSelect: (id: string) => void;
  /** How note text is shown (addressed to the reader, or as stored). */
  label: (s: string) => string;
}) {
  const box = useRef<HTMLDivElement>(null);
  const fg = useRef<ForceGraphMethods<N, L> | undefined>(undefined);
  const [size, setSize] = useState({ w: 600, h: 400 });
  const [hover, setHover] = useState<string | null>(null);
  const reduced = useMemo(
    () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false,
    [],
  );

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) =>
      setSize({ w: Math.floor(e.contentRect.width), h: Math.floor(e.contentRect.height) }),
    );
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const data = useMemo(() => {
    const degree = new Map<string, number>();
    const ids = new Set(props.graph.nodes.map((n) => n.id));
    const edges = props.graph.edges.filter((e: Edge) => ids.has(e.src_id) && ids.has(e.dst_id));
    for (const e of edges) {
      degree.set(e.src_id, (degree.get(e.src_id) ?? 0) + 1);
      degree.set(e.dst_id, (degree.get(e.dst_id) ?? 0) + 1);
    }
    return {
      nodes: props.graph.nodes.map((n) => ({ ...n, degree: degree.get(n.id) ?? 0 })) as N[],
      links: edges.map((e) => ({ source: e.src_id, target: e.dst_id, link_type: e.link_type })) as L[],
    };
  }, [props.graph]);

  const neighbors = useMemo(() => {
    const m = new Map<string, Set<string>>();
    for (const e of props.graph.edges) {
      if (!m.has(e.src_id)) m.set(e.src_id, new Set());
      if (!m.has(e.dst_id)) m.set(e.dst_id, new Set());
      m.get(e.src_id)!.add(e.dst_id);
      m.get(e.dst_id)!.add(e.src_id);
    }
    return m;
  }, [props.graph]);

  const q = props.query.trim().toLowerCase();
  const matches = (n: Note) => !q || n.content.toLowerCase().includes(q) || n.keywords.includes(q);
  const focus = props.selected ?? hover;
  const near = focus ? neighbors.get(focus) ?? new Set<string>() : null;

  useEffect(() => {
    const f = fg.current;
    if (!f) return;
    f.d3Force("charge")?.strength(-38);
    f.d3Force("link")?.distance((l: L) => (l.link_type === "summarizes" ? 34 : 46));
  }, [data]);

  return (
    <div className="sky" ref={box}>
      <ForceGraph2D<N, L>
        ref={fg}
        graphData={data}
        width={size.w}
        height={size.h}
        backgroundColor="rgba(0,0,0,0)"
        cooldownTicks={reduced ? 0 : 160}
        warmupTicks={reduced ? 160 : 40}
        onEngineStop={() => fg.current?.zoomToFit(400, 60)}
        nodeId="id"
        nodeLabel={(n) => props.label(n.content)}
        nodeRelSize={1}
        nodeVal={(n) => radius(n) ** 2}
        onNodeClick={(n) => props.onSelect(n.id)}
        onNodeHover={(n) => setHover(n ? n.id : null)}
        linkColor={(l) => {
          const s = (l.source as N).id;
          const t = (l.target as N).id;
          const lit = focus && (s === focus || t === focus);
          if (l.link_type === "supersedes") return lit ? C.moon : "rgba(124,130,159,0.75)";
          if (l.link_type === "summarizes") return lit ? C.theme : "rgba(214,178,236,0.18)";
          return lit ? C.fact : "rgba(156,195,230,0.32)";
        }}
        linkWidth={(l) => (l.link_type === "summarizes" ? 0.6 : 1)}
        linkLineDash={(l) => (l.link_type === "supersedes" ? [3, 2.5] : null)}
        linkDirectionalArrowLength={(l) => (l.link_type === "supersedes" ? 3.5 : 0)}
        linkDirectionalArrowRelPos={0.92}
        nodeCanvasObject={(n, ctx, scale) => {
          const r = radius(n);
          const x = n.x ?? 0;
          const y = n.y ?? 0;
          const faded = (q && !matches(n)) || (near && n.id !== focus && !near.has(n.id));
          const gone = n.status === "archived" || n.status === "forgotten";
          const color = n.status === "superseded" ? C.outdated : gone ? C.dim : C[n.tier];
          ctx.save();
          ctx.globalAlpha = faded ? 0.18 : gone ? 0.5 : 1;
          if (n.tier === "theme") {
            ctx.beginPath();
            ctx.arc(x, y, r + 3.5, 0, 2 * Math.PI);
            ctx.strokeStyle = "rgba(214,178,236,0.35)";
            ctx.lineWidth = 1;
            ctx.stroke();
          }
          ctx.beginPath();
          ctx.arc(x, y, r, 0, 2 * Math.PI);
          if (n.status === "superseded") {
            ctx.strokeStyle = color;
            ctx.lineWidth = 1.2;
            ctx.stroke();
          } else {
            ctx.shadowColor = color;
            ctx.shadowBlur = n.tier === "theme" ? 14 : 6;
            ctx.fillStyle = color;
            ctx.fill();
          }
          ctx.shadowBlur = 0;
          if (n.id === props.selected) {
            ctx.beginPath();
            ctx.arc(x, y, r + 2.5, 0, 2 * Math.PI);
            ctx.strokeStyle = C.moon;
            ctx.lineWidth = 1.5 / Math.max(scale / 2, 1);
            ctx.stroke();
          }
          const label =
            n.id === focus ||
            (near?.has(n.id) ?? false) ||
            (n.tier === "theme" && !faded) ||
            (q !== "" && matches(n)) ||
            (scale > 2.6 && !faded && n.tier !== "episode");
          if (label) {
            const size = Math.max(11 / scale, 2.2);
            ctx.font = `${n.tier === "theme" ? "italic " : ""}${size}px Spectral, Georgia, serif`;
            ctx.fillStyle = n.status === "superseded" ? C.outdated : "rgba(241,239,248,0.92)";
            ctx.textAlign = "center";
            ctx.textBaseline = "top";
            const shown = props.label(n.content);
            const text = n.tier === "theme" ? clip(shown.split(":")[0], 28) : clip(shown, 44);
            ctx.fillText(text, x, y + r + 3);
          }
          ctx.restore();
        }}
        nodePointerAreaPaint={(n, color, ctx) => {
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(n.x ?? 0, n.y ?? 0, radius(n) + 3, 0, 2 * Math.PI);
          ctx.fill();
        }}
      />
      <ul className="legend" aria-label="Legend">
        <li>
          <span className="dot fact" /> Fact
        </li>
        <li>
          <span className="dot theme" /> Theme
        </li>
        <li>
          <span className="dot outdated" /> Outdated fact
        </li>
        <li>
          <span className="dash" /> Replaced by
        </li>
      </ul>
    </div>
  );
}
