"use client";

import { useEffect, useState } from "react";
import { knowledgeApi, VideoArtifactItem } from "@/lib/api";

interface Props {
  sessionId: string;
  folderIds?: number[];
  refreshKey?: number;
}

type RelationItem = {
  source: string;
  relation: string;
  target: string;
  evidence?: string;
};

type GraphNode = {
  id: string;
  label: string;
  type?: string;
  x: number;
  y: number;
  side: "left" | "right";
};

type GraphEdge = RelationItem & {
  id: string;
  sourceNodeId: string;
  targetNodeId: string;
};

function truncate(value: string, max: number) {
  return value.length > max ? `${value.slice(0, max)}...` : value;
}

function normalizeRelation(item: VideoArtifactItem): RelationItem[] {
  return Array.from(
    new Map(
      item.relations
        .map((relation) => ({
          source: (relation.source || "").trim(),
          relation: (relation.relation || "related").trim() || "related",
          target: (relation.target || "").trim(),
          evidence: (relation.evidence || "").trim(),
        }))
        .filter((relation) => relation.source && relation.target && relation.source !== relation.target)
        .map((relation) => [
          `${relation.source.toLowerCase()}__${relation.relation.toLowerCase()}__${relation.target.toLowerCase()}`,
          relation,
        ])
    ).values()
  ).slice(0, 12);
}

function rankRelations(relations: RelationItem[], limit = 8): RelationItem[] {
  const scored = relations
    .map((relation) => {
      const relationLen = relation.relation.length;
      const specificScore = Math.min(10, Math.max(0, relationLen - 1));
      const evidenceScore = relation.evidence ? 3 : 0;
      return {
        relation,
        score: specificScore + evidenceScore,
      };
    })
    .sort((a, b) => b.score - a.score || a.relation.relation.length - b.relation.relation.length)
    .map((item) => item.relation);

  return scored.slice(0, limit);
}

function pickTopLabels(values: string[], max = 6): string[] {
  const count = new Map<string, number>();
  for (const value of values) {
    count.set(value, (count.get(value) || 0) + 1);
  }
  return Array.from(count.entries())
    .sort((a, b) => b[1] - a[1] || a[0].length - b[0].length)
    .slice(0, max)
    .map(([label]) => label);
}

function buildGraph(item: VideoArtifactItem): { nodes: GraphNode[]; edges: GraphEdge[]; height: number; width: number } {
  const normalized = rankRelations(normalizeRelation(item), 8);
  const entityTypeMap = new Map(
    item.entities
      .map((entity) => [((entity.name || "").trim()), entity.type || ""] as const)
      .filter(([name]) => !!name)
  );

  const leftLabels = pickTopLabels(normalized.map((edge) => edge.source), 6);
  const rightLabels = pickTopLabels(normalized.map((edge) => edge.target), 6);

  const filteredEdges = normalized.filter(
    (edge) => leftLabels.includes(edge.source) && rightLabels.includes(edge.target)
  );

  const rowCount = Math.max(2, leftLabels.length, rightLabels.length);
  const rowGap = 48;
  const top = 32;
  const width = 520;
  const height = Math.max(220, top * 2 + (rowCount - 1) * rowGap + 16);
  const leftX = 120;
  const rightX = width - 120;

  const nodes: GraphNode[] = [];
  leftLabels.forEach((label, index) => {
    nodes.push({
      id: `left:${label}`,
      label,
      type: entityTypeMap.get(label),
      x: leftX,
      y: top + index * rowGap,
      side: "left",
    });
  });

  rightLabels.forEach((label, index) => {
    nodes.push({
      id: `right:${label}`,
      label,
      type: entityTypeMap.get(label),
      x: rightX,
      y: top + index * rowGap,
      side: "right",
    });
  });

  const edges: GraphEdge[] = filteredEdges.map((edge, index) => ({
    ...edge,
    id: `${edge.source}-${edge.relation}-${edge.target}-${index}`,
    sourceNodeId: `left:${edge.source}`,
    targetNodeId: `right:${edge.target}`,
  }));

  return { nodes, edges, height, width };
}

function getKeyEntities(item: VideoArtifactItem): Array<{ name?: string; type?: string; description?: string }> {
  const relations = normalizeRelation(item);
  const degree = new Map<string, number>();
  for (const relation of relations) {
    degree.set(relation.source, (degree.get(relation.source) || 0) + 1);
    degree.set(relation.target, (degree.get(relation.target) || 0) + 1);
  }

  const inputEntities = item.entities
    .map((entity) => ({
      name: (entity.name || "").trim(),
      type: (entity.type || "").trim(),
      description: (entity.description || "").trim(),
    }))
    .filter((entity) => !!entity.name);

  const entityMap = new Map(inputEntities.map((entity) => [entity.name.toLowerCase(), entity]));

  for (const relation of relations) {
    for (const name of [relation.source, relation.target]) {
      const key = name.toLowerCase();
      if (!entityMap.has(key)) {
        entityMap.set(key, { name, type: "", description: "" });
      }
    }
  }

  return Array.from(entityMap.values())
    .sort((a, b) => {
      const ad = degree.get(a.name || "") || 0;
      const bd = degree.get(b.name || "") || 0;
      if (bd !== ad) {
        return bd - ad;
      }
      const at = a.type ? 1 : 0;
      const bt = b.type ? 1 : 0;
      if (bt !== at) {
        return bt - at;
      }
      return (a.name || "").length - (b.name || "").length;
    })
    .slice(0, 8);
}

function ArtifactGraph({ item }: { item: VideoArtifactItem }) {
  const { nodes, edges, height, width } = buildGraph(item);
  if (nodes.length === 0 || edges.length === 0) {
    return null;
  }

  const nodeById = new Map(nodes.map((node) => [node.id, node]));

  return (
    <div className="artifact-graph">
      <svg viewBox={`0 0 ${width} ${height}`} className="artifact-graph-svg" role="img" aria-label={`${item.title} relation graph`}>
        <defs>
          <marker id="artifact-arrow" markerWidth="10" markerHeight="8" refX="8" refY="4" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,8 L8,4 z" className="artifact-graph-arrow" />
          </marker>
        </defs>

        <text x={36} y={16} className="artifact-graph-side-label">来源实体</text>
        <text x={width - 96} y={16} className="artifact-graph-side-label">目标实体</text>

        {edges.map((edge, index) => {
          const source = nodeById.get(edge.sourceNodeId);
          const target = nodeById.get(edge.targetNodeId);
          if (!source || !target) {
            return null;
          }

          const dx = target.x - source.x;
          const dy = target.y - source.y;
          const controlX = source.x + dx * 0.45;
          const controlY = source.y + dy * 0.2;
          const path = `M ${source.x} ${source.y} Q ${controlX} ${controlY} ${target.x} ${target.y}`;
          const labelX = source.x + dx * 0.5;
          const labelY = source.y + dy * 0.5 - 6;

          return (
            <g key={edge.id || `${edge.source}-${edge.target}-${index}`}>
              <path d={path} className="artifact-graph-edge" markerEnd="url(#artifact-arrow)" />
              <text x={labelX} y={labelY} textAnchor="middle" className="artifact-graph-edge-label">
                {truncate(edge.relation, 10)}
              </text>
            </g>
          );
        })}

        {nodes.map((node) => (
          <g key={node.id}>
            <rect
              x={node.x - 48}
              y={node.y - 18}
              width={96}
              height={36}
              rx={10}
              className={node.side === "left" ? "artifact-graph-node artifact-graph-node-source" : "artifact-graph-node artifact-graph-node-target"}
            />
            <text x={node.x} y={node.y - 1} textAnchor="middle" className="artifact-graph-node-label">
              {truncate(node.label, 8)}
            </text>
            {node.type ? (
              <text x={node.x} y={node.y + 12} textAnchor="middle" className="artifact-graph-node-type">
                {truncate(node.type, 6)}
              </text>
            ) : null}
          </g>
        ))}
      </svg>
    </div>
  );
}

export default function VideoArtifactsPanel({ sessionId, folderIds, refreshKey }: Props) {
  const [items, setItems] = useState<VideoArtifactItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const folderIdsKey = (folderIds || []).join(",");

  useEffect(() => {
    let cancelled = false;
    const effectiveFolderIds = folderIdsKey
      ? folderIdsKey.split(",").map((value) => Number(value)).filter((value) => !Number.isNaN(value))
      : undefined;

    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await knowledgeApi.getVideoArtifacts(sessionId, effectiveFolderIds, 24);
        if (!cancelled) {
          setItems(data);
        }
      } catch {
        if (!cancelled) {
          setError("摘要与知识图谱加载失败");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    if (sessionId) {
      load();
    }

    return () => {
      cancelled = true;
    };
  }, [sessionId, refreshKey, folderIdsKey]);

  return (
    <section className="panel panel-artifacts">
      <div className="panel-header">
        <div>
          <div className="panel-title">视频摘要与知识图谱</div>
          <div className="panel-subtitle">展示已入库视频的摘要、关键实体和核心关系</div>
        </div>
      </div>

      <div className="panel-body">
        {loading ? (
          <div className="text-sm text-[var(--muted)]">加载中...</div>
        ) : error ? (
          <div className="text-sm text-[var(--muted)]">{error}</div>
        ) : items.length === 0 ? (
          <div className="text-sm text-[var(--muted)]">
            还没有可展示的摘要。先选择收藏夹并完成入库或更新后，这里会自动出现每个视频的整理结果。
          </div>
        ) : (
          <div className="artifact-grid">
            {items.map((item) => (
              <article key={item.bvid} className="artifact-card">
                {(() => {
                  const keyEntities = getKeyEntities(item);
                  const keyRelations = rankRelations(normalizeRelation(item), 5);
                  return (
                    <>
                <div className="artifact-head">
                  <a href={item.url} target="_blank" rel="noopener noreferrer" className="artifact-title">
                    {item.title}
                  </a>
                  <div className="artifact-meta">
                    {item.source || "unknown"}
                    {item.transcript_segment_count > 0 ? ` · ${item.transcript_segment_count} 段文字稿` : ""}
                  </div>
                </div>

                {item.summary ? <p className="artifact-summary">{item.summary}</p> : null}

                {!item.summary && item.entities.length === 0 && item.relations.length === 0 ? (
                  <p className="artifact-summary">这条视频的摘要与知识图谱还在生成或回填中，刷新页面后会自动显示。</p>
                ) : null}

                {item.key_points.length > 0 ? (
                  <div className="artifact-section">
                    <div className="artifact-section-title">关键要点</div>
                    <ul className="artifact-list">
                      {item.key_points.map((point, index) => (
                        <li key={`${item.bvid}-point-${index}`}>{point}</li>
                      ))}
                    </ul>
                  </div>
                ) : null}

                {keyEntities.length > 0 ? (
                  <div className="artifact-section">
                    <div className="artifact-section-title">关键实体</div>
                    <div className="artifact-tags">
                      {keyEntities.map((entity, index) => (
                        <span key={index} className="artifact-tag" title={entity.description || entity.name}>
                          {entity.name || "未命名"}
                          {entity.type ? ` · ${entity.type}` : ""}
                        </span>
                      ))}
                    </div>
                  </div>
                ) : null}

                {item.relations.length > 0 ? (
                  <div className="artifact-section">
                    <div className="artifact-section-title">核心关系</div>
                    <ArtifactGraph item={item} />
                    {keyRelations.length > 0 ? (
                      <ul className="artifact-relations-inline">
                        {keyRelations.map((relation, idx) => (
                          <li key={`${item.bvid}-rel-${idx}`}>
                            <strong>{truncate(relation.source, 12)}</strong>
                            <span>{truncate(relation.relation, 12)}</span>
                            <strong>{truncate(relation.target, 12)}</strong>
                          </li>
                        ))}
                      </ul>
                    ) : null}
                  </div>
                ) : null}
                    </>
                  );
                })()}
              </article>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
