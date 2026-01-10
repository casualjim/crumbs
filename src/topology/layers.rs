use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use super::{TopologySnapshot, TopologyEdge};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerConfig {
    pub layers: Vec<Layer>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Layer {
    pub name: String,
    pub patterns: Vec<String>,
    pub allowed_deps: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct LayerViolation {
    pub from_node: String,
    pub to_node: String,
    pub from_layer: String,
    pub to_layer: String,
    pub reason: String,
}

#[derive(Clone, Debug)]
pub struct LayerCheckResult {
    pub is_valid: bool,
    pub violations: Vec<LayerViolation>,
    pub orphaned_nodes: Vec<String>,
}

impl LayerConfig {
    pub fn default_config() -> Self {
        Self {
            layers: vec![
                Layer {
                    name: "domain".to_string(),
                    patterns: vec!["domain".to_string(), "model".to_string()],
                    allowed_deps: Vec::new(),
                },
                Layer {
                    name: "application".to_string(),
                    patterns: vec!["service".to_string(), "handler".to_string()],
                    allowed_deps: vec!["domain".to_string()],
                },
                Layer {
                    name: "infrastructure".to_string(),
                    patterns: vec![
                        "infra".to_string(),
                        "adapter".to_string(),
                        "repository".to_string(),
                    ],
                    allowed_deps: vec!["domain".to_string(), "application".to_string()],
                },
            ],
        }
    }
}

pub fn check_layers(snapshot: &TopologySnapshot, config: &LayerConfig) -> LayerCheckResult {
    let mut layer_lookup = HashMap::new();
    for layer in &config.layers {
        layer_lookup.insert(layer.name.to_ascii_lowercase(), layer);
    }

    let mut node_layers = HashMap::new();
    let mut orphaned = Vec::new();
    for node in snapshot.nodes() {
        if let Some(layer) = match_layer(node, config) {
            node_layers.insert(node.clone(), layer.name.clone());
        } else {
            orphaned.push(node.clone());
        }
    }

    let mut violations = Vec::new();
    for edge in snapshot.dependency_edges() {
        let Some(from_layer) = node_layers.get(&edge.src) else { continue };
        let Some(to_layer) = node_layers.get(&edge.dst) else { continue };
        if from_layer == to_layer {
            continue;
        }
        let Some(layer) = layer_lookup.get(&from_layer.to_ascii_lowercase()) else { continue };
        if layer.allowed_deps.is_empty() {
            violations.push(build_violation(&edge, from_layer, to_layer, "disallowed dependency"));
            continue;
        }
        let allowed: HashSet<String> = layer
            .allowed_deps
            .iter()
            .map(|item| item.to_ascii_lowercase())
            .collect();
        if !allowed.contains(&to_layer.to_ascii_lowercase()) {
            violations.push(build_violation(&edge, from_layer, to_layer, "disallowed dependency"));
        }
    }

    LayerCheckResult {
        is_valid: violations.is_empty(),
        violations,
        orphaned_nodes: orphaned,
    }
}

fn match_layer<'a>(path: &str, config: &'a LayerConfig) -> Option<&'a Layer> {
    let normalized = path.to_ascii_lowercase();
    for layer in &config.layers {
        for pattern in &layer.patterns {
            let pattern = pattern.to_ascii_lowercase();
            if normalized.contains(&pattern) {
                return Some(layer);
            }
        }
    }
    None
}

fn build_violation(
    edge: &TopologyEdge,
    from_layer: &str,
    to_layer: &str,
    reason: &str,
) -> LayerViolation {
    LayerViolation {
        from_node: edge.src.clone(),
        to_node: edge.dst.clone(),
        from_layer: from_layer.to_string(),
        to_layer: to_layer.to_string(),
        reason: reason.to_string(),
    }
}
