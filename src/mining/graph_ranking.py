from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set


def calculate_pagerank(
    nodes: Sequence[str],
    edges: Sequence[Tuple[str, str]],
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Dict[str, float]:
    """
    标题: 计算有向图的 PageRank 分数

    Input:
      - 参数:
          - nodes (Sequence[str], required): 节点 ID 列表（论文 OpenAlex Work ID）
          - edges (Sequence[Tuple[str,str]], required): 有向边列表 (src_id -> dst_id)
          - damping (float, optional, default=0.85, range=(0,1)): 阻尼系数
          - max_iter (int, optional, default=100, >=1): 最大迭代次数
          - tol (float, optional, default=1e-6, >0): 收敛阈值（L1 范数）
      - 上下文:
          - 无（纯函数）
      - 依赖:
          - Python 标准库

    Output:
      - 返回:
          - Dict[str,float]: 每个节点的 PageRank 分数（非负；总和约等于 1）
      - 副作用:
          - 无
      - 错误:
          - ValueError: 当 damping/tol/max_iter 非法时抛出

    Why:
      - 本实现采用幂迭代 (power iteration) 且不依赖 NetworkX，避免额外依赖与安装成本，
        适合在本项目的“Top-N 局部引文子图”场景中快速计算权威度先验。

    References:
      - Page, Brin, Motwani, Winograd (1999) - The PageRank Citation Ranking: https://doi.org/10.48550/arXiv.cs/0601119
      - Wikipedia - PageRank: https://en.wikipedia.org/wiki/PageRank

    Calls:
      - 无（纯算法）

    Example:
      - 输入:
          - nodes=["A","B","C"]
          - edges=[("A","B"),("B","C"),("C","B")]
      - 输出:
          - {"A":0.05..., "B":0.47..., "C":0.47...} (示意)
      - 边界/错误:
          - nodes=[] -> {}
    """
    if not (0.0 < damping < 1.0):
        raise ValueError("damping must be in (0, 1)")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    if tol <= 0:
        raise ValueError("tol must be > 0")

    n = len(nodes)
    if n == 0:
        return {}

    # 去重并保证顺序稳定：nodes 由调用方提供，通常已是 Top-N 列表
    index: Dict[str, int] = {}
    ordered_nodes: List[str] = []
    for node in nodes:
        if node not in index:
            index[node] = len(ordered_nodes)
            ordered_nodes.append(node)

    n = len(ordered_nodes)
    if n == 0:
        return {}

    # 构建 outlinks 与入边列表
    out_degree = [0] * n
    incoming: List[List[int]] = [[] for _ in range(n)]

    for src, dst in edges:
        if src not in index or dst not in index:
            continue
        s = index[src]
        d = index[dst]
        out_degree[s] += 1
        incoming[d].append(s)

    pr = [1.0 / n] * n
    teleport = (1.0 - damping) / n

    for _ in range(max_iter):
        sink_mass = 0.0
        for i in range(n):
            if out_degree[i] == 0:
                sink_mass += pr[i]

        sink_contrib = damping * sink_mass / n
        new_pr = [teleport + sink_contrib] * n

        for d in range(n):
            acc = 0.0
            for s in incoming[d]:
                # out_degree[s] > 0 by construction for sources in incoming
                acc += pr[s] / out_degree[s]
            new_pr[d] += damping * acc

        # L1 diff for convergence
        diff = 0.0
        for i in range(n):
            diff += abs(new_pr[i] - pr[i])
        pr = new_pr
        if diff < tol:
            break

    # 数值稳定：归一化到 sum=1（避免浮点误差传播到上游融合逻辑）
    s = sum(pr)
    if s > 0:
        pr = [x / s for x in pr]

    return {node: float(pr[index[node]]) for node in ordered_nodes}


def detect_communities_louvain(
    nodes: Sequence[str],
    edges: Sequence[Tuple[str, str]],
    resolution: float = 1.0,
    max_levels: int = 10,
    max_iter: int = 50,
    min_edges: int = 2,
) -> Dict[str, int]:
    """
    标题: 使用 Louvain 方法对局部论文网络进行社区检测（结构化综述分簇）

    Input:
      - 参数:
          - nodes (Sequence[str], required): 节点 ID 列表（论文 OpenAlex Work ID）
          - edges (Sequence[Tuple[str,str]], required): 边列表（可传有向引文边；本函数会自动对称化为无向加权图）
          - resolution (float, optional, default=1.0, >0): 模块度分辨率参数 γ（越大越倾向产生更多小社区）
          - max_levels (int, optional, default=10, >=1): Louvain 多层聚合的最大层数
          - max_iter (int, optional, default=50, >=1): 每一层“单层优化”的最大迭代轮数
          - min_edges (int, optional, default=2, >=0): 少于该边数时直接退化为单社区（避免噪声分簇）
      - 上下文:
          - 无（纯函数）
      - 依赖:
          - Python 标准库

    Output:
      - 返回:
          - Dict[str,int]: node_id -> community_id（community_id 为从 0 开始的连续整数）
      - 副作用:
          - 无
      - 错误:
          - ValueError: 当 resolution/max_levels/max_iter/min_edges 非法时抛出

    Why:
      - “结构化综述”需要把 Top-N 论文按“流派/方法/主题簇”分组。
        Louvain 是经典的模块度最大化社区检测方法，适合小图快速聚类且不依赖外部模型。

    References:
      - Blondel et al. (2008) - Fast unfolding of communities in large networks: https://doi.org/10.1088/1742-5468/2008/10/P10008
      - Wikipedia - Louvain method: https://en.wikipedia.org/wiki/Louvain_method

    Calls:
      - 无（纯算法）

    Example:
      - 输入:
          - nodes=["W1","W2","W3"]
          - edges=[("W1","W2"),("W2","W1"),("W2","W3")]
      - 输出:
          - {"W1":0,"W2":0,"W3":1} (示意)
      - 边界/错误:
          - edges=[] -> 所有节点同一社区 0
    """
    if resolution <= 0:
        raise ValueError("resolution must be > 0")
    if max_levels < 1:
        raise ValueError("max_levels must be >= 1")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    if min_edges < 0:
        raise ValueError("min_edges must be >= 0")

    # 去重且稳定顺序
    node_index: Dict[str, int] = {}
    ordered_nodes: List[str] = []
    for n in nodes:
        if n not in node_index:
            node_index[n] = len(ordered_nodes)
            ordered_nodes.append(n)
    if not ordered_nodes:
        return {}

    # --- build undirected weighted adjacency from (possibly directed) edges ---
    adj: List[Dict[int, float]] = [dict() for _ in range(len(ordered_nodes))]
    edge_counter = 0
    seen_pairs: Set[Tuple[int, int]] = set()

    for a, b in edges:
        if a not in node_index or b not in node_index:
            continue
        i = node_index[a]
        j = node_index[b]
        if i == j:
            continue
        # Treat directed citation as undirected evidence; accumulate weight
        adj[i][j] = adj[i].get(j, 0.0) + 1.0
        adj[j][i] = adj[j].get(i, 0.0) + 1.0
        # count unique undirected pairs for min_edges heuristic
        pair = (i, j) if i < j else (j, i)
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            edge_counter += 1

    # Too few edges => one community
    if edge_counter < min_edges:
        return {n: 0 for n in ordered_nodes}

    def _degrees(g_adj: List[Dict[int, float]]) -> List[float]:
        return [sum(nei.values()) for nei in g_adj]

    def _total_weight(degs: List[float]) -> float:
        # For undirected graph, sum(deg) = 2m
        return sum(degs)

    def _one_level(
        g_adj: List[Dict[int, float]],
        deg: List[float],
        resolution_: float,
        max_iter_: int,
    ) -> List[int]:
        """
        Greedy modularity optimization phase (single level).
        Returns community assignment per node (community ids are arbitrary integers).
        """
        n_ = len(g_adj)
        # initial: each node its own community
        com = list(range(n_))
        # community total degree
        tot = deg.copy()
        m2 = _total_weight(deg)
        # If the graph at this level has no edges (all isolated super-nodes),
        # keep each node as its own community. Returning all-zeros would incorrectly
        # merge disconnected components in later levels.
        if m2 <= 1e-12:
            return list(range(n_))

        for _ in range(max_iter_):
            moved = False
            for i in range(n_):
                ci = com[i]
                ki = deg[i]
                if ki <= 1e-12:
                    continue

                # accumulate weights from i to each neighboring community
                neigh_communities: Dict[int, float] = {}
                for j, w in g_adj[i].items():
                    cj = com[j]
                    neigh_communities[cj] = neigh_communities.get(cj, 0.0) + w

                # remove i from its community
                tot[ci] -= ki

                best_c = ci
                best_gain = 0.0

                # evaluate moving i into each neighbor community
                for cj, k_i_in in neigh_communities.items():
                    gain = k_i_in - resolution_ * (tot[cj] * ki / m2)
                    if gain > best_gain:
                        best_gain = gain
                        best_c = cj

                # insert i back (possibly to new community)
                com[i] = best_c
                tot[best_c] += ki
                if best_c != ci:
                    moved = True

            if not moved:
                break
        return com

    def _renumber(com: List[int]) -> List[int]:
        mapping: Dict[int, int] = {}
        out: List[int] = []
        for c in com:
            if c not in mapping:
                mapping[c] = len(mapping)
            out.append(mapping[c])
        return out

    def _aggregate_graph(g_adj: List[Dict[int, float]], com: List[int]) -> List[Dict[int, float]]:
        """Build induced graph of communities (undirected weighted)."""
        com = _renumber(com)
        k = max(com) + 1 if com else 0
        new_adj: List[Dict[int, float]] = [dict() for _ in range(k)]
        for i, neigh in enumerate(g_adj):
            ci = com[i]
            for j, w in neigh.items():
                cj = com[j]
                if ci == cj:
                    continue
                new_adj[ci][cj] = new_adj[ci].get(cj, 0.0) + w
        # symmetrize (since we built from directed iteration over i, weights should be symmetric-ish)
        for i in range(k):
            for j, w in list(new_adj[i].items()):
                if i == j:
                    continue
                if i not in new_adj[j]:
                    new_adj[j][i] = w
        return new_adj

    # --- multi-level louvain ---
    current_adj = adj
    current_deg = _degrees(current_adj)

    # mapping from original nodes to communities in current level
    orig_to_current = list(range(len(ordered_nodes)))

    for _level in range(max_levels):
        com_level = _one_level(current_adj, current_deg, resolution, max_iter)
        com_level = _renumber(com_level)

        # update orig_to_current mapping
        for idx in range(len(orig_to_current)):
            orig_to_current[idx] = com_level[orig_to_current[idx]]

        # if no aggregation possible (each node stays unique), stop
        if len(set(com_level)) == len(current_adj):
            break

        # build next level graph
        current_adj = _aggregate_graph(current_adj, com_level)
        current_deg = _degrees(current_adj)
        if len(current_adj) <= 1:
            break

    # final renumber to ensure 0..K-1
    final = _renumber(orig_to_current)
    return {ordered_nodes[i]: int(final[i]) for i in range(len(ordered_nodes))}


