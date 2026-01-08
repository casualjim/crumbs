use std::collections::{HashMap, HashSet, VecDeque};

use eyre::{Result, eyre};

use crate::repository::{CochangeEdge, DependencyEdge, Repository};

const DEPENDENCY_WEIGHT: f64 = 0.7;
const COCHANGE_WEIGHT: f64 = 0.3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EdgeKind {
    Dependency,
    Cochange,
}

#[derive(Clone, Debug)]
struct Edge {
    src: usize,
    dst: usize,
    weight: f64,
    kind: EdgeKind,
}

#[derive(Clone, Debug)]
pub struct TopologyStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub component_count: usize,
    pub scc_count: usize,
    pub cyclic_scc_count: usize,
    pub cyclic_node_count: usize,
    pub betti_0: usize,
    pub betti_1: usize,
    pub solid_score: f64,
    pub avg_out_degree: f64,
    pub density: f64,
}

#[derive(Clone, Debug)]
pub struct CycleEdge {
    pub src: String,
    pub dst: String,
    pub weight: f64,
    pub cochange_weight: f64,
    pub persistence: f64,
    pub cut_score: f64,
}

#[derive(Clone, Debug)]
pub struct CycleComponent {
    pub id: usize,
    pub nodes: Vec<String>,
    pub edges: Vec<CycleEdge>,
    pub cycle_rank: usize,
    pub total_weight: f64,
    pub max_weight: f64,
    pub min_weight: f64,
}

#[derive(Clone, Debug)]
pub struct RefactorCut {
    pub scc_id: usize,
    pub src: String,
    pub dst: String,
    pub weight: f64,
    pub persistence: f64,
    pub cut_score: f64,
}

#[derive(Clone, Debug)]
pub struct RefactorPlan {
    pub total_cycles: usize,
    pub cuts: Vec<RefactorCut>,
    pub warnings: Vec<String>,
}

#[derive(Clone, Copy, Debug)]
pub struct RefactorOptions {
    pub max_cuts_per_cycle: usize,
    pub max_total_cuts: usize,
    pub min_cut_score: f64,
}

impl Default for RefactorOptions {
    fn default() -> Self {
        Self {
            max_cuts_per_cycle: 3,
            max_total_cuts: 20,
            min_cut_score: 0.2,
        }
    }
}

#[derive(Clone, Debug)]
pub struct StarNeighbor {
    pub path: String,
    pub distance: usize,
    pub in_weight: f64,
    pub out_weight: f64,
    pub total_weight: f64,
}

#[derive(Clone, Debug)]
pub struct TopologySnapshot {
    pub stats: TopologyStats,
    pub cycles: Vec<CycleComponent>,
    nodes: Vec<String>,
    index: HashMap<String, usize>,
    edges: Vec<Edge>,
    outgoing: Vec<Vec<usize>>,
    incoming: Vec<Vec<usize>>,
    undirected: Vec<Vec<usize>>,
}

impl TopologySnapshot {
    pub async fn load(db: &dyn Repository) -> Result<Self> {
        let dependency_edges = db.list_dependency_edges().await?;
        let cochange_edges = db.list_cochange_edges().await?;
        let files = db.list_files().await?;
        Ok(Self::from_edges(&files, &dependency_edges, &cochange_edges))
    }

    pub fn star_neighborhood(&self, center: &str, depth: usize) -> Result<Vec<StarNeighbor>> {
        let Some(&center_idx) = self.index.get(center) else {
            return Err(eyre!("unknown path: {center}"));
        };
        if depth == 0 {
            return Ok(Vec::new());
        }

        let mut distance = vec![usize::MAX; self.nodes.len()];
        let mut queue = VecDeque::new();
        distance[center_idx] = 0;
        queue.push_back(center_idx);

        while let Some(node) = queue.pop_front() {
            let next = distance[node].saturating_add(1);
            if next > depth {
                continue;
            }
            for &neighbor in &self.undirected[node] {
                if distance[neighbor] == usize::MAX {
                    distance[neighbor] = next;
                    queue.push_back(neighbor);
                }
            }
        }

        let mut neighbors = Vec::new();
        for (idx, &dist) in distance.iter().enumerate() {
            if idx == center_idx || dist == usize::MAX || dist == 0 {
                continue;
            }
            let (in_weight, out_weight) = self.neighborhood_weights(idx, &distance, depth);
            let total_weight = in_weight + out_weight;
            neighbors.push(StarNeighbor {
                path: self.nodes[idx].clone(),
                distance: dist,
                in_weight,
                out_weight,
                total_weight,
            });
        }

        neighbors.sort_by(|a, b| {
            a.distance
                .cmp(&b.distance)
                .then_with(|| b.total_weight.partial_cmp(&a.total_weight).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| a.path.cmp(&b.path))
        });
        Ok(neighbors)
    }

    pub fn has_path(&self, path: &str) -> bool {
        self.index.contains_key(path)
    }

    pub fn refactor_plan(&self, options: RefactorOptions) -> RefactorPlan {
        build_refactor_plan(self, options)
    }

    fn neighborhood_weights(&self, node_idx: usize, distances: &[usize], depth: usize) -> (f64, f64) {
        let mut in_weight = 0.0;
        let mut out_weight = 0.0;
        for &edge_idx in &self.incoming[node_idx] {
            let edge = &self.edges[edge_idx];
            if edge.kind != EdgeKind::Dependency {
                continue;
            }
            if distances[edge.src] <= depth {
                in_weight += edge.weight.max(0.0);
            }
        }
        for &edge_idx in &self.outgoing[node_idx] {
            let edge = &self.edges[edge_idx];
            if edge.kind != EdgeKind::Dependency {
                continue;
            }
            if distances[edge.dst] <= depth {
                out_weight += edge.weight.max(0.0);
            }
        }
        (in_weight, out_weight)
    }

    fn from_edges(
        files: &[String],
        dependency_edges: &[DependencyEdge],
        cochange_edges: &[CochangeEdge],
    ) -> Self {
        let mut nodes: Vec<String> = Vec::new();
        let mut index = HashMap::new();
        for file in files {
            let idx = nodes.len();
            nodes.push(file.clone());
            index.insert(file.clone(), idx);
        }
        for edge in dependency_edges {
            ensure_node(&mut nodes, &mut index, &edge.src_path);
            ensure_node(&mut nodes, &mut index, &edge.dst_path);
        }
        for edge in cochange_edges {
            ensure_node(&mut nodes, &mut index, &edge.src_path);
            ensure_node(&mut nodes, &mut index, &edge.dst_path);
        }

        let mut edges = Vec::new();
        for edge in dependency_edges {
            let src = index[&edge.src_path];
            let dst = index[&edge.dst_path];
            edges.push(Edge {
                src,
                dst,
                weight: edge.reference_count.max(0) as f64,
                kind: EdgeKind::Dependency,
            });
        }
        for edge in cochange_edges {
            let src = index[&edge.src_path];
            let dst = index[&edge.dst_path];
            edges.push(Edge {
                src,
                dst,
                weight: edge.weight,
                kind: EdgeKind::Cochange,
            });
        }

        let mut outgoing = vec![Vec::new(); nodes.len()];
        let mut incoming = vec![Vec::new(); nodes.len()];
        let mut undirected = vec![Vec::new(); nodes.len()];
        for (idx, edge) in edges.iter().enumerate() {
            if edge.kind == EdgeKind::Dependency {
                outgoing[edge.src].push(idx);
                incoming[edge.dst].push(idx);
                if edge.src != edge.dst {
                    undirected[edge.src].push(edge.dst);
                    undirected[edge.dst].push(edge.src);
                }
            }
        }
        for neighbors in &mut undirected {
            neighbors.sort_unstable();
            neighbors.dedup();
        }

        let (cochange_weights, max_cochange) = cochange_map(cochange_edges, &index);
        let (stats, cycles) = analyze_topology(
            &nodes,
            &edges,
            &outgoing,
            &undirected,
            &cochange_weights,
            max_cochange,
        );

        Self {
            stats,
            cycles,
            nodes,
            index,
            edges,
            outgoing,
            incoming,
            undirected,
        }
    }
}

fn ensure_node(nodes: &mut Vec<String>, index: &mut HashMap<String, usize>, path: &str) {
    if index.contains_key(path) {
        return;
    }
    let idx = nodes.len();
    nodes.push(path.to_string());
    index.insert(path.to_string(), idx);
}

fn cochange_map(
    edges: &[CochangeEdge],
    index: &HashMap<String, usize>,
) -> (HashMap<(usize, usize), f64>, f64) {
    let mut weights = HashMap::new();
    let mut max_weight = 0.0;
    for edge in edges {
        let Some(&src) = index.get(&edge.src_path) else { continue };
        let Some(&dst) = index.get(&edge.dst_path) else { continue };
        let key = if src <= dst { (src, dst) } else { (dst, src) };
        let entry = weights.entry(key).or_insert(0.0);
        if edge.weight > *entry {
            *entry = edge.weight;
        }
        if *entry > max_weight {
            max_weight = *entry;
        }
    }
    (weights, max_weight)
}

fn analyze_topology(
    nodes: &[String],
    edges: &[Edge],
    outgoing: &[Vec<usize>],
    undirected: &[Vec<usize>],
    cochange_weights: &HashMap<(usize, usize), f64>,
    max_cochange: f64,
) -> (TopologyStats, Vec<CycleComponent>) {
    let node_count = nodes.len();
    let edge_count = edges.iter().filter(|edge| edge.kind == EdgeKind::Dependency).count();

    let components = connected_components(undirected);
    let component_count = components.len();
    let mut undirected_edge_set = HashSet::new();
    for edge in edges.iter().filter(|edge| edge.kind == EdgeKind::Dependency) {
        let (a, b) = if edge.src <= edge.dst {
            (edge.src, edge.dst)
        } else {
            (edge.dst, edge.src)
        };
        undirected_edge_set.insert((a, b));
    }
    let undirected_edges = undirected_edge_set.len();
    let betti_0 = component_count;
    let betti_1 = ((undirected_edges as isize) - (node_count as isize) + (component_count as isize))
        .max(0) as usize;

    let sccs = tarjan_scc(node_count, outgoing, edges);
    let scc_count = sccs.len();
    let mut cycles = Vec::new();
    let mut cyclic_scc_count = 0;
    let mut cyclic_nodes = HashSet::new();

    for (id, scc) in sccs.iter().enumerate() {
        let mut edge_indices = Vec::new();
        let mut has_self_loop = false;
        let scc_set: HashSet<usize> = scc.iter().copied().collect();
        for (edge_idx, edge) in edges.iter().enumerate() {
            if edge.kind != EdgeKind::Dependency {
                continue;
            }
            if scc_set.contains(&edge.src) && scc_set.contains(&edge.dst) {
                if edge.src == edge.dst {
                    has_self_loop = true;
                }
                edge_indices.push(edge_idx);
            }
        }
        if scc.len() <= 1 && !has_self_loop {
            continue;
        }
        cyclic_scc_count += 1;
        for node in scc {
            cyclic_nodes.insert(*node);
        }

        let mut max_weight: f64 = 0.0;
        let mut min_weight: f64 = f64::MAX;
        let mut total_weight = 0.0;
        for edge_idx in &edge_indices {
            let edge = &edges[*edge_idx];
            max_weight = max_weight.max(edge.weight);
            min_weight = min_weight.min(edge.weight);
            total_weight += edge.weight;
        }

        let mut cycle_edges = Vec::new();
        for edge_idx in &edge_indices {
            let edge = &edges[*edge_idx];
            let cochange = cochange_for_edge(edge, cochange_weights);
            let dep_norm = if max_weight > 0.0 { edge.weight / max_weight } else { 0.0 };
            let co_norm = if max_cochange > 0.0 { cochange / max_cochange } else { 0.0 };
            let persistence = DEPENDENCY_WEIGHT * dep_norm + COCHANGE_WEIGHT * co_norm;
            let cut_score = 1.0 - persistence;
            cycle_edges.push(CycleEdge {
                src: nodes[edge.src].clone(),
                dst: nodes[edge.dst].clone(),
                weight: edge.weight,
                cochange_weight: cochange,
                persistence,
                cut_score,
            });
        }

        if min_weight == f64::MAX {
            min_weight = 0.0;
        }

        let cycle_rank =
            ((edge_indices.len() as isize) - (scc.len() as isize) + 1).max(0) as usize;
        let mut node_paths: Vec<String> = scc.iter().map(|idx| nodes[*idx].clone()).collect();
        node_paths.sort();
        cycle_edges.sort_by(|a, b| b.cut_score.partial_cmp(&a.cut_score).unwrap_or(std::cmp::Ordering::Equal));

        cycles.push(CycleComponent {
            id,
            nodes: node_paths,
            edges: cycle_edges,
            cycle_rank,
            total_weight,
            max_weight,
            min_weight,
        });
    }

    let avg_out_degree = if node_count == 0 {
        0.0
    } else {
        edge_count as f64 / node_count as f64
    };
    let density = if node_count <= 1 {
        0.0
    } else {
        edge_count as f64 / (node_count * (node_count - 1)) as f64
    };
    let solid_score = if edge_count == 0 {
        1.0
    } else {
        1.0 / (1.0 + (betti_1 as f64 / edge_count as f64))
    };

    let stats = TopologyStats {
        node_count,
        edge_count,
        component_count,
        scc_count,
        cyclic_scc_count,
        cyclic_node_count: cyclic_nodes.len(),
        betti_0,
        betti_1,
        solid_score,
        avg_out_degree,
        density,
    };

    (stats, cycles)
}

fn cochange_for_edge(edge: &Edge, cochange_weights: &HashMap<(usize, usize), f64>) -> f64 {
    let key = if edge.src <= edge.dst {
        (edge.src, edge.dst)
    } else {
        (edge.dst, edge.src)
    };
    cochange_weights.get(&key).copied().unwrap_or(0.0)
}

fn connected_components(undirected: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut visited = vec![false; undirected.len()];
    let mut components = Vec::new();
    for start in 0..undirected.len() {
        if visited[start] {
            continue;
        }
        let mut stack = vec![start];
        visited[start] = true;
        let mut component = Vec::new();
        while let Some(node) = stack.pop() {
            component.push(node);
            for &neighbor in &undirected[node] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    stack.push(neighbor);
                }
            }
        }
        components.push(component);
    }
    components
}

fn tarjan_scc(node_count: usize, outgoing: &[Vec<usize>], edges: &[Edge]) -> Vec<Vec<usize>> {
    let mut index = 0usize;
    let mut stack = Vec::new();
    let mut on_stack = vec![false; node_count];
    let mut indices = vec![None; node_count];
    let mut lowlink = vec![0usize; node_count];
    let mut result = Vec::new();

    fn strongconnect(
        v: usize,
        index: &mut usize,
        stack: &mut Vec<usize>,
        on_stack: &mut Vec<bool>,
        indices: &mut Vec<Option<usize>>,
        lowlink: &mut Vec<usize>,
        outgoing: &[Vec<usize>],
        edges: &[Edge],
        result: &mut Vec<Vec<usize>>,
    ) {
        indices[v] = Some(*index);
        lowlink[v] = *index;
        *index += 1;
        stack.push(v);
        on_stack[v] = true;

        for &edge_idx in &outgoing[v] {
            let edge = &edges[edge_idx];
            let w = edge.dst;
            if indices[w].is_none() {
                strongconnect(
                    w, index, stack, on_stack, indices, lowlink, outgoing, edges, result,
                );
                lowlink[v] = lowlink[v].min(lowlink[w]);
            } else if on_stack[w] {
                lowlink[v] = lowlink[v].min(indices[w].unwrap());
            }
        }

        if indices[v] == Some(lowlink[v]) {
            let mut scc = Vec::new();
            while let Some(w) = stack.pop() {
                on_stack[w] = false;
                scc.push(w);
                if w == v {
                    break;
                }
            }
            result.push(scc);
        }
    }

    for v in 0..node_count {
        if indices[v].is_none() {
            strongconnect(
                v,
                &mut index,
                &mut stack,
                &mut on_stack,
                &mut indices,
                &mut lowlink,
                outgoing,
                edges,
                &mut result,
            );
        }
    }

    result
}

fn build_refactor_plan(snapshot: &TopologySnapshot, options: RefactorOptions) -> RefactorPlan {
    let mut cuts = Vec::new();
    let mut warnings = Vec::new();
    let total_cycles = snapshot.cycles.len();

    for cycle in &snapshot.cycles {
        if cuts.len() >= options.max_total_cuts {
            warnings.push("cut limit reached before all cycles were handled".to_string());
            break;
        }
        let mut remaining_edges = cycle.edges.clone();
        let mut removed = Vec::new();
        let node_set: HashSet<String> = cycle.nodes.iter().cloned().collect();

        while has_cycle_in_component(&node_set, &remaining_edges) {
            if removed.len() >= options.max_cuts_per_cycle {
                warnings.push(format!(
                    "cycle {} still has cycles after {} cuts",
                    cycle.id, options.max_cuts_per_cycle
                ));
                break;
            }
            if cuts.len() >= options.max_total_cuts {
                warnings.push("cut limit reached before all cycles were handled".to_string());
                break;
            }

            remaining_edges.sort_by(|a, b| {
                b.cut_score
                    .partial_cmp(&a.cut_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.weight.partial_cmp(&b.weight).unwrap_or(std::cmp::Ordering::Equal))
                    .then_with(|| a.src.cmp(&b.src))
                    .then_with(|| a.dst.cmp(&b.dst))
            });
            let candidate = remaining_edges
                .iter()
                .find(|edge| edge.cut_score >= options.min_cut_score)
                .cloned();
            let Some(candidate) = candidate else {
                warnings.push(format!(
                    "cycle {} has no edges above min_cut_score {:.2}",
                    cycle.id, options.min_cut_score
                ));
                break;
            };

            removed.push(candidate.clone());
            remaining_edges.retain(|edge| !(edge.src == candidate.src && edge.dst == candidate.dst));
        }

        for cut in removed {
            cuts.push(RefactorCut {
                scc_id: cycle.id,
                src: cut.src,
                dst: cut.dst,
                weight: cut.weight,
                persistence: cut.persistence,
                cut_score: cut.cut_score,
            });
            if cuts.len() >= options.max_total_cuts {
                break;
            }
        }
    }

    RefactorPlan {
        total_cycles,
        cuts,
        warnings,
    }
}

fn has_cycle_in_component(node_set: &HashSet<String>, edges: &[CycleEdge]) -> bool {
    if node_set.is_empty() {
        return false;
    }
    let mut index = HashMap::new();
    for (idx, node) in node_set.iter().enumerate() {
        index.insert(node.clone(), idx);
    }
    let node_count = index.len();
    let mut indegree = vec![0usize; node_count];
    let mut outgoing = vec![Vec::new(); node_count];
    for edge in edges {
        let Some(&src) = index.get(&edge.src) else { continue };
        let Some(&dst) = index.get(&edge.dst) else { continue };
        outgoing[src].push(dst);
        indegree[dst] += 1;
    }
    let mut queue = VecDeque::new();
    for (idx, &deg) in indegree.iter().enumerate() {
        if deg == 0 {
            queue.push_back(idx);
        }
    }
    let mut visited = 0usize;
    while let Some(node) = queue.pop_front() {
        visited += 1;
        for &neighbor in &outgoing[node] {
            indegree[neighbor] = indegree[neighbor].saturating_sub(1);
            if indegree[neighbor] == 0 {
                queue.push_back(neighbor);
            }
        }
    }
    visited != node_count
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot_from_edges(edges: &[(&str, &str, i64)]) -> TopologySnapshot {
        let dependency_edges = edges
            .iter()
            .map(|(src, dst, weight)| DependencyEdge {
                src_path: src.to_string(),
                dst_path: dst.to_string(),
                reference_count: *weight,
            })
            .collect::<Vec<_>>();
        TopologySnapshot::from_edges(&[], &dependency_edges, &[])
    }

    #[test]
    fn betti_numbers_for_triangle() {
        let snapshot = snapshot_from_edges(&[("a", "b", 1), ("b", "c", 1), ("c", "a", 1)]);
        assert_eq!(snapshot.stats.betti_0, 1);
        assert_eq!(snapshot.stats.betti_1, 1);
        assert_eq!(snapshot.stats.cyclic_scc_count, 1);
    }

    #[test]
    fn refactor_plan_picks_lightest_edge() {
        let snapshot = snapshot_from_edges(&[("a", "b", 5), ("b", "c", 2), ("c", "a", 9)]);
        let plan = snapshot.refactor_plan(RefactorOptions {
            max_cuts_per_cycle: 1,
            max_total_cuts: 5,
            min_cut_score: 0.0,
        });
        assert_eq!(plan.cuts.len(), 1);
        assert_eq!(plan.cuts[0].src, "b");
        assert_eq!(plan.cuts[0].dst, "c");
    }
}
