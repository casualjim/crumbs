pub(crate) struct BatchItem {
    pub(crate) chunk_id: String,
    pub(crate) file_path: String,
    pub(crate) text: String,
    pub(crate) token_count: usize,
}

pub(crate) struct ChunkBatch {
    pub(crate) batch_id: usize,
    pub(crate) items: Vec<BatchItem>,
}

pub(crate) struct StreamBatcher {
    buffer: Vec<BatchItem>,
    chunk_count: usize,
    max_batch_size: usize,
    batch_id: usize,
}

impl StreamBatcher {
    pub(crate) fn new(max_batch_size: usize) -> Self {
        Self {
            buffer: Vec::new(),
            chunk_count: 0,
            max_batch_size: max_batch_size.max(1),
            batch_id: 0,
        }
    }

    pub(crate) fn add(&mut self, item: BatchItem) -> Option<ChunkBatch> {
        self.chunk_count = self.chunk_count.saturating_add(1);
        self.buffer.push(item);

        if self.chunk_count >= self.max_batch_size {
            return self.flush();
        }

        None
    }

    pub(crate) fn flush(&mut self) -> Option<ChunkBatch> {
        if self.buffer.is_empty() {
            return None;
        }

        let batch = ChunkBatch {
            batch_id: self.batch_id,
            items: std::mem::take(&mut self.buffer),
        };
        self.batch_id = self.batch_id.saturating_add(1);
        self.chunk_count = 0;
        Some(batch)
    }

    pub(crate) fn remove_file(&mut self, file_path: &str) {
        if self.buffer.is_empty() {
            return;
        }
        self.buffer.retain(|item| item.file_path != file_path);
        self.chunk_count = self.buffer.len();
    }
}

pub(crate) fn split_by_tokens(
    items: Vec<BatchItem>,
    max_tokens_per_batch: usize,
    max_items_per_batch: usize,
) -> Vec<Vec<BatchItem>> {
    let max_tokens = max_tokens_per_batch.max(1);
    let max_items = max_items_per_batch.max(1);
    let mut result = Vec::new();
    let mut current = Vec::new();
    let mut current_tokens = 0usize;

    for item in items {
        let item_tokens = item.token_count;
        if !current.is_empty()
            && (current.len() >= max_items
                || current_tokens.saturating_add(item_tokens) > max_tokens)
        {
            result.push(std::mem::take(&mut current));
            current_tokens = 0;
        }

        current_tokens = current_tokens.saturating_add(item_tokens);
        current.push(item);
    }

    if !current.is_empty() {
        result.push(current);
    }

    result
}
