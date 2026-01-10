use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};

use eyre::{Result, eyre};
use serde::{Deserialize, Serialize};

use crate::repository::{CochangeEdge, DependencyEdge, Repository};

#[cfg(test)]
mod refactor;
pub mod layers;
pub mod workspace;
use workspace::WorkspaceInfo;
use workspace::detect_workspace_info;

const DEPENDENCY_WEIGHT: f64 = 0.7;
const COCHANGE_WEIGHT: f64 = 0.3;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
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
    pub betti_2: usize,
    pub triangle_count: usize,
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TopologyNode {
    pub path: String,
    pub package: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TopologyEdge {
    pub src: String,
    pub dst: String,
    pub weight: f64,
    pub kind: EdgeKind,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TopologyExport {
    pub nodes: Vec<TopologyNode>,
    pub edges: Vec<TopologyEdge>,
}

#[derive(Clone, Debug)]
pub struct FeatureVolume {
    pub id: usize,
    pub nodes: Vec<String>,
    pub triangle_count: usize,
    pub cohesion: f64,
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
    pub workspace: Option<WorkspaceInfo>,
    nodes: Vec<String>,
    index: HashMap<String, usize>,
    package_by_path: HashMap<String, String>,
    edges: Vec<Edge>,
    outgoing: Vec<Vec<usize>>,
    incoming: Vec<Vec<usize>>,
    undirected: Vec<Vec<usize>>,
}

impl TopologySnapshot {
    pub async fn load_with_workspace(
        db: &dyn Repository,
        repo_root: &Path,
        start_path: &Path,
    ) -> Result<Self> {
        let dependency_edges = db.list_dependency_edges().await?;
        let cochange_edges = db.list_cochange_edges().await?;
        let files = db.list_files().await?;
        let workspace = detect_workspace_info(start_path, &files)?;
        let (scoped_files, scoped_dependencies, scoped_cochanges) =
            scope_workspace_files(repo_root, &workspace, &files, &dependency_edges, &cochange_edges);
        let mut snapshot =
            Self::from_edges(&scoped_files, &scoped_dependencies, &scoped_cochanges);
        snapshot.apply_workspace(workspace, &scoped_files);
        Ok(snapshot)
    }

    pub fn nodes(&self) -> &[String] {
        &self.nodes
    }

    pub fn package_for_path(&self, path: &str) -> Option<&str> {
        self.package_by_path.get(path).map(String::as_str)
    }

    pub fn dependency_edges(&self) -> Vec<TopologyEdge> {
        self.edges
            .iter()
            .filter(|edge| edge.kind == EdgeKind::Dependency)
            .map(|edge| TopologyEdge {
                src: self.nodes[edge.src].clone(),
                dst: self.nodes[edge.dst].clone(),
                weight: edge.weight,
                kind: edge.kind,
            })
            .collect()
    }

    pub fn export_graph(&self, include_cochange: bool) -> TopologyExport {
        let nodes = self
            .nodes
            .iter()
            .map(|path| TopologyNode {
                path: path.clone(),
                package: self.package_by_path.get(path).cloned(),
            })
            .collect();
        let edges = self
            .edges
            .iter()
            .filter(|edge| include_cochange || edge.kind == EdgeKind::Dependency)
            .map(|edge| TopologyEdge {
                src: self.nodes[edge.src].clone(),
                dst: self.nodes[edge.dst].clone(),
                weight: edge.weight,
                kind: edge.kind,
            })
            .collect();
        TopologyExport { nodes, edges }
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

    pub fn shortest_path(&self, start: &str, end: &str) -> Result<Vec<String>> {
        let Some(&start_idx) = self.index.get(start) else {
            return Err(eyre!("unknown path: {start}"));
        };
        let Some(&end_idx) = self.index.get(end) else {
            return Err(eyre!("unknown path: {end}"));
        };
        if start_idx == end_idx {
            return Ok(vec![start.to_string()]);
        }

        let mut queue = VecDeque::new();
        let mut visited = vec![false; self.nodes.len()];
        let mut prev = vec![None; self.nodes.len()];
        visited[start_idx] = true;
        queue.push_back(start_idx);

        while let Some(node) = queue.pop_front() {
            for &edge_idx in &self.outgoing[node] {
                let edge = &self.edges[edge_idx];
                if edge.kind != EdgeKind::Dependency {
                    continue;
                }
                let next = edge.dst;
                if visited[next] {
                    continue;
                }
                visited[next] = true;
                prev[next] = Some(node);
                if next == end_idx {
                    break;
                }
                queue.push_back(next);
            }
        }

        if !visited[end_idx] {
            return Ok(Vec::new());
        }

        let mut path = Vec::new();
        let mut current = Some(end_idx);
        while let Some(idx) = current {
            path.push(self.nodes[idx].clone());
            current = prev[idx];
        }
        path.reverse();
        Ok(path)
    }

    pub fn hotspots(
        &self,
        limit: usize,
        iterations: usize,
        damping: f64,
    ) -> Vec<(String, f64)> {
        let ranks = pagerank(self, iterations, damping);
        let mut pairs: Vec<(String, f64)> = ranks
            .into_iter()
            .enumerate()
            .map(|(idx, score)| (self.nodes[idx].clone(), score))
            .collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if limit > 0 && pairs.len() > limit {
            pairs.truncate(limit);
        }
        pairs
    }

    pub fn feature_volumes(
        &self,
        max_triangles: usize,
        limit: usize,
    ) -> Vec<FeatureVolume> {
        let triangles = find_triangles(self, max_triangles);
        if triangles.is_empty() {
            return Vec::new();
        }
        let volumes = group_triangles(self, &triangles);
        let mut volumes = volumes
            .into_iter()
            .enumerate()
            .map(|(id, volume)| FeatureVolume {
                id,
                nodes: volume.nodes,
                triangle_count: volume.triangle_count,
                cohesion: volume.cohesion,
            })
            .collect::<Vec<_>>();
        volumes.sort_by(|a, b| {
            b.cohesion
                .partial_cmp(&a.cohesion)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if limit > 0 && volumes.len() > limit {
            volumes.truncate(limit);
        }
        volumes
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
            workspace: None,
            nodes,
            index,
            package_by_path: HashMap::new(),
            edges,
            outgoing,
            incoming,
            undirected,
        }
    }

    fn apply_workspace(&mut self, workspace: WorkspaceInfo, files: &[String]) {
        let files_are_absolute = files.iter().any(|path| Path::new(path).is_absolute());
        let mut prefixes: Vec<(PathBuf, String)> = workspace
            .packages
            .iter()
            .map(|pkg| {
                let prefix = if files_are_absolute {
                    Path::new(&workspace.root).join(&pkg.path)
                } else {
                    if workspace.root.trim().is_empty() || workspace.root == "." {
                        PathBuf::from(&pkg.path)
                    } else {
                        Path::new(&workspace.root).join(&pkg.path)
                    }
                };
                (prefix, pkg.name.clone())
            })
            .collect();
        prefixes.sort_by(|a, b| b.0.as_os_str().len().cmp(&a.0.as_os_str().len()));

        let mut package_by_path = HashMap::new();
        for file in files {
            let path = PathBuf::from(file);
            for (prefix, name) in &prefixes {
                if path.starts_with(prefix) {
                    package_by_path.insert(file.clone(), name.clone());
                    break;
                }
            }
        }

        self.workspace = Some(workspace);
        self.package_by_path = package_by_path;
    }
}

fn scope_workspace_files(
    repo_root: &Path,
    workspace: &WorkspaceInfo,
    files: &[String],
    dependency_edges: &[DependencyEdge],
    cochange_edges: &[CochangeEdge],
) -> (Vec<String>, Vec<DependencyEdge>, Vec<CochangeEdge>) {
    let root = workspace.root.trim();
    if root.is_empty() || root == "." {
        return (
            files.to_vec(),
            dependency_edges.to_vec(),
            cochange_edges.to_vec(),
        );
    }

    let files_are_absolute = files.iter().any(|path| Path::new(path).is_absolute());
    let mut root_path = PathBuf::from(root);
    if files_are_absolute && root_path.is_relative() {
        root_path = repo_root.join(root_path);
    } else if !files_are_absolute && root_path.is_absolute() {
        if let Ok(relative) = root_path.strip_prefix(repo_root) {
            root_path = if relative.as_os_str().is_empty() {
                PathBuf::from(".")
            } else {
                relative.to_path_buf()
            };
        }
    }

    let mut allowed = HashSet::new();
    for file in files {
        if path_in_scope(file, &root_path) {
            allowed.insert(file.clone());
        }
    }
    if allowed.is_empty() {
        return (
            files.to_vec(),
            dependency_edges.to_vec(),
            cochange_edges.to_vec(),
        );
    }

    let scoped_files: Vec<String> = files
        .iter()
        .filter(|file| allowed.contains(*file))
        .cloned()
        .collect();
    let scoped_dependencies: Vec<DependencyEdge> = dependency_edges
        .iter()
        .filter(|edge| allowed.contains(&edge.src_path) && allowed.contains(&edge.dst_path))
        .cloned()
        .collect();
    let scoped_cochanges: Vec<CochangeEdge> = cochange_edges
        .iter()
        .filter(|edge| allowed.contains(&edge.src_path) && allowed.contains(&edge.dst_path))
        .cloned()
        .collect();

    (scoped_files, scoped_dependencies, scoped_cochanges)
}

fn path_in_scope(path: &str, root: &Path) -> bool {
    let cleaned = path.trim_start_matches("./");
    let file_path = Path::new(cleaned);
    if root.as_os_str().is_empty() {
        return true;
    }
    if file_path == root {
        return true;
    }
    file_path.starts_with(root)
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
    let triangles = collect_triangles(undirected, 0);
    let triangle_count = triangles.len();
    let betti_2 = if triangle_count == 0 || undirected_edges == 0 {
        0
    } else {
        let edge_index = build_edge_index(&undirected_edge_set);
        let rank = triangle_boundary_rank(&triangles, &edge_index, undirected_edges);
        triangle_count.saturating_sub(rank)
    };

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
        betti_2,
        triangle_count,
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

fn pagerank(snapshot: &TopologySnapshot, iterations: usize, damping: f64) -> Vec<f64> {
    let node_count = snapshot.nodes.len();
    if node_count == 0 {
        return Vec::new();
    }
    let mut ranks = vec![1.0 / node_count as f64; node_count];
    let mut out_weight = vec![0.0; node_count];
    for edge in &snapshot.edges {
        if edge.kind != EdgeKind::Dependency {
            continue;
        }
        out_weight[edge.src] += edge.weight.max(0.0);
    }

    let base = (1.0 - damping) / node_count as f64;
    for _ in 0..iterations {
        let mut next = vec![base; node_count];
        let mut sink_rank = 0.0;
        for (idx, weight) in out_weight.iter().enumerate() {
            if *weight == 0.0 {
                sink_rank += ranks[idx];
            }
        }
        let sink_share = if sink_rank > 0.0 {
            damping * sink_rank / node_count as f64
        } else {
            0.0
        };
        if sink_share > 0.0 {
            for value in &mut next {
                *value += sink_share;
            }
        }

        for edge in &snapshot.edges {
            if edge.kind != EdgeKind::Dependency {
                continue;
            }
            let denom = out_weight[edge.src];
            if denom <= 0.0 {
                continue;
            }
            let weight = edge.weight.max(0.0);
            if weight == 0.0 {
                continue;
            }
            next[edge.dst] += damping * ranks[edge.src] * (weight / denom);
        }
        ranks = next;
    }
    ranks
}

fn find_triangles(snapshot: &TopologySnapshot, max_triangles: usize) -> Vec<[usize; 3]> {
    collect_triangles(&snapshot.undirected, max_triangles)
}

fn collect_triangles(undirected: &[Vec<usize>], max_triangles: usize) -> Vec<[usize; 3]> {
    let mut triangles = Vec::new();
    if undirected.len() < 3 {
        return triangles;
    }
    let mut neighbor_sets = Vec::with_capacity(undirected.len());
    for neighbors in undirected {
        let set: HashSet<usize> = neighbors.iter().copied().collect();
        neighbor_sets.push(set);
    }

    for u in 0..undirected.len() {
        let neighbors = &undirected[u];
        for i in 0..neighbors.len() {
            let v = neighbors[i];
            if v <= u {
                continue;
            }
            for j in (i + 1)..neighbors.len() {
                let w = neighbors[j];
                if w <= v {
                    continue;
                }
                if neighbor_sets[v].contains(&w) {
                    triangles.push([u, v, w]);
                    if max_triangles > 0 && triangles.len() >= max_triangles {
                        return triangles;
                    }
                }
            }
        }
    }
    triangles
}

fn build_edge_index(edges: &HashSet<(usize, usize)>) -> HashMap<(usize, usize), usize> {
    let mut list: Vec<(usize, usize)> = edges.iter().copied().collect();
    list.sort_unstable();
    let mut map = HashMap::with_capacity(list.len());
    for (idx, edge) in list.into_iter().enumerate() {
        map.insert(edge, idx);
    }
    map
}

fn triangle_boundary_rank(
    triangles: &[[usize; 3]],
    edge_index: &HashMap<(usize, usize), usize>,
    edge_count: usize,
) -> usize {
    if edge_count == 0 || triangles.is_empty() {
        return 0;
    }
    let word_len = (edge_count + 63) / 64;
    let mut rows: Vec<Vec<u64>> = Vec::with_capacity(triangles.len());

    for triangle in triangles {
        let mut row = vec![0u64; word_len];
        let mut edges = [
            (triangle[0], triangle[1]),
            (triangle[0], triangle[2]),
            (triangle[1], triangle[2]),
        ];
        for (a, b) in &mut edges {
            if *a > *b {
                std::mem::swap(a, b);
            }
            if let Some(&idx) = edge_index.get(&(*a, *b)) {
                row[idx / 64] ^= 1u64 << (idx % 64);
            }
        }
        rows.push(row);
    }

    let mut rank = 0usize;
    for col in 0..edge_count {
        let word = col / 64;
        let mask = 1u64 << (col % 64);
        let mut pivot = None;
        for r in rank..rows.len() {
            if rows[r][word] & mask != 0 {
                pivot = Some(r);
                break;
            }
        }
        let Some(pivot) = pivot else {
            continue;
        };
        rows.swap(rank, pivot);
        for r in (rank + 1)..rows.len() {
            if rows[r][word] & mask != 0 {
                for w in word..word_len {
                    rows[r][w] ^= rows[rank][w];
                }
            }
        }
        rank += 1;
        if rank == rows.len() {
            break;
        }
    }

    rank
}

struct VolumeGroup {
    nodes: Vec<String>,
    triangle_count: usize,
    cohesion: f64,
}

fn group_triangles(
    snapshot: &TopologySnapshot,
    triangles: &[[usize; 3]],
) -> Vec<VolumeGroup> {
    if triangles.is_empty() {
        return Vec::new();
    }
    let mut parent: Vec<usize> = (0..triangles.len()).collect();

    fn find(parent: &mut [usize], idx: usize) -> usize {
        if parent[idx] != idx {
            parent[idx] = find(parent, parent[idx]);
        }
        parent[idx]
    }

    fn union(parent: &mut [usize], a: usize, b: usize) {
        let root_a = find(parent, a);
        let root_b = find(parent, b);
        if root_a != root_b {
            parent[root_b] = root_a;
        }
    }

    let mut edge_to_triangles: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (idx, triangle) in triangles.iter().enumerate() {
        let mut edges = [
            (triangle[0], triangle[1]),
            (triangle[0], triangle[2]),
            (triangle[1], triangle[2]),
        ];
        for (a, b) in &mut edges {
            if *a > *b {
                std::mem::swap(a, b);
            }
            edge_to_triangles.entry((*a, *b)).or_default().push(idx);
        }
    }

    for indices in edge_to_triangles.values() {
        if indices.len() < 2 {
            continue;
        }
        let first = indices[0];
        for other in indices.iter().skip(1) {
            union(&mut parent, first, *other);
        }
    }

    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for idx in 0..triangles.len() {
        let root = find(&mut parent, idx);
        groups.entry(root).or_default().push(idx);
    }

    let mut volumes = Vec::new();
    for (_, triangle_indices) in groups {
        let mut node_set = HashSet::new();
        for idx in &triangle_indices {
            let triangle = triangles[*idx];
            node_set.insert(triangle[0]);
            node_set.insert(triangle[1]);
            node_set.insert(triangle[2]);
        }
        let mut nodes: Vec<String> = node_set
            .iter()
            .map(|idx| snapshot.nodes[*idx].clone())
            .collect();
        nodes.sort();

        let node_count = nodes.len();
        let triangle_count = triangle_indices.len();
        let possible = if node_count >= 3 {
            (node_count * (node_count - 1) * (node_count - 2)) as f64 / 6.0
        } else {
            0.0
        };
        let cohesion = if possible > 0.0 {
            triangle_count as f64 / possible
        } else {
            0.0
        };
        volumes.push(VolumeGroup {
            nodes,
            triangle_count,
            cohesion,
        });
    }

    volumes
}

#[cfg(test)]
mod extra_tests {
    use super::*;
    use crate::topology::layers::{check_layers, Layer, LayerConfig};
    use std::collections::{BTreeSet, HashSet};

    fn snapshot_from_edges(edges: &[(&str, &str, i64)]) -> TopologySnapshot {
        let mut files = BTreeSet::new();
        let mut dependency_edges = Vec::new();
        for (src, dst, weight) in edges {
            files.insert(src.to_string());
            files.insert(dst.to_string());
            dependency_edges.push(DependencyEdge {
                src_path: (*src).to_string(),
                dst_path: (*dst).to_string(),
                reference_count: *weight,
            });
        }
        let files: Vec<String> = files.into_iter().collect();
        TopologySnapshot::from_edges(&files, &dependency_edges, &[])
    }

    #[test]
    fn shortest_path_finds_dependency_route() -> Result<()> {
        let snapshot = snapshot_from_edges(&[
            ("src/a.rs", "src/b.rs", 1),
            ("src/b.rs", "src/c.rs", 1),
        ]);

        let path = snapshot.shortest_path("src/a.rs", "src/c.rs")?;
        assert_eq!(path, vec!["src/a.rs", "src/b.rs", "src/c.rs"]);

        let missing = snapshot.shortest_path("src/c.rs", "src/a.rs")?;
        assert!(missing.is_empty());
        Ok(())
    }

    #[test]
    fn feature_volumes_detect_triangle() {
        let snapshot = snapshot_from_edges(&[
            ("src/a.rs", "src/b.rs", 1),
            ("src/b.rs", "src/c.rs", 1),
            ("src/c.rs", "src/a.rs", 1),
        ]);

        let volumes = snapshot.feature_volumes(10, 10);
        assert_eq!(volumes.len(), 1);
        assert_eq!(volumes[0].triangle_count, 1);
        assert_eq!(volumes[0].nodes.len(), 3);
    }

    #[test]
    fn hotspots_include_all_nodes() {
        let snapshot = snapshot_from_edges(&[
            ("src/a.rs", "src/b.rs", 5),
            ("src/b.rs", "src/c.rs", 1),
        ]);

        let hotspots = snapshot.hotspots(0, 10, 0.85);
        let nodes: HashSet<&str> = hotspots.iter().map(|(node, _)| node.as_str()).collect();
        assert_eq!(nodes.len(), 3);
        assert!(nodes.contains("src/a.rs"));
        assert!(nodes.contains("src/b.rs"));
        assert!(nodes.contains("src/c.rs"));
    }

    #[test]
    fn layers_detect_disallowed_dependency() {
        let snapshot = snapshot_from_edges(&[(
            "src/domain/model.rs",
            "src/application/service.rs",
            1,
        )]);

        let config = LayerConfig {
            layers: vec![
                Layer {
                    name: "domain".to_string(),
                    patterns: vec!["domain".to_string()],
                    allowed_deps: Vec::new(),
                },
                Layer {
                    name: "application".to_string(),
                    patterns: vec!["application".to_string()],
                    allowed_deps: vec!["domain".to_string()],
                },
            ],
        };

        let result = check_layers(&snapshot, &config);
        assert!(!result.is_valid);
        assert_eq!(result.violations.len(), 1);
        assert_eq!(result.violations[0].from_layer, "domain");
        assert_eq!(result.violations[0].to_layer, "application");
    }
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

    fn snapshot_from_edges_with_cochange(
        dependencies: &[(&str, &str, i64)],
        cochanges: &[(&str, &str, f64)],
    ) -> TopologySnapshot {
        let dependency_edges = dependencies
            .iter()
            .map(|(src, dst, weight)| DependencyEdge {
                src_path: src.to_string(),
                dst_path: dst.to_string(),
                reference_count: *weight,
            })
            .collect::<Vec<_>>();
        let cochange_edges = cochanges
            .iter()
            .map(|(src, dst, weight)| CochangeEdge {
                src_path: src.to_string(),
                dst_path: dst.to_string(),
                weight: *weight,
                commit_count: 0,
            })
            .collect::<Vec<_>>();
        TopologySnapshot::from_edges(&[], &dependency_edges, &cochange_edges)
    }

    #[test]
    fn betti_numbers_for_triangle() {
        let snapshot = snapshot_from_edges(&[("a", "b", 1), ("b", "c", 1), ("c", "a", 1)]);
        assert_eq!(snapshot.stats.betti_0, 1);
        assert_eq!(snapshot.stats.betti_1, 1);
        assert_eq!(snapshot.stats.betti_2, 0);
        assert_eq!(snapshot.stats.triangle_count, 1);
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


    #[test]
    fn star_neighborhood_respects_depth() {
        let snapshot = snapshot_from_edges(&[
            ("a", "b", 1),
            ("b", "c", 1),
            ("c", "d", 1),
        ]);
        let depth1 = snapshot.star_neighborhood("a", 1).expect("star");
        assert_eq!(depth1.len(), 1);
        assert_eq!(depth1[0].path, "b");
        assert_eq!(depth1[0].distance, 1);

        let depth2 = snapshot.star_neighborhood("a", 2).expect("star");
        assert!(depth2.iter().any(|item| item.path == "b" && item.distance == 1));
        assert!(depth2.iter().any(|item| item.path == "c" && item.distance == 2));
        assert!(!depth2.iter().any(|item| item.path == "d"));
    }

    #[test]
    fn cochange_weight_is_attached_to_cycle_edges() {
        let snapshot = snapshot_from_edges_with_cochange(
            &[("a", "b", 1), ("b", "c", 1), ("c", "a", 1)],
            &[("a", "b", 10.0)],
        );
        let cycle = snapshot.cycles.first().expect("cycle");
        let edge = cycle
            .edges
            .iter()
            .find(|edge| edge.src == "a" && edge.dst == "b")
            .expect("edge a->b");
        assert!(edge.cochange_weight > 0.0);
    }

    #[test]
    fn refactor_plan_warns_on_high_min_cut_score() {
        let snapshot = snapshot_from_edges(&[("a", "b", 1), ("b", "c", 1), ("c", "a", 1)]);
        let plan = snapshot.refactor_plan(RefactorOptions {
            max_cuts_per_cycle: 1,
            max_total_cuts: 1,
            min_cut_score: 1.1,
        });
        assert_eq!(plan.cuts.len(), 0);
        assert_eq!(plan.warnings.len(), 1);
    }
}
