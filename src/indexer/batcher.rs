pub(crate) struct BatchItem {
    pub(crate) chunk_id: String,
    pub(crate) file_path: String,
    pub(crate) text: String,
    pub(crate) token_count: usize,
}

pub(crate) struct BatchMeta {
    pub(crate) chunk_id: String,
    pub(crate) file_path: String,
}

pub(crate) struct TokenAwareBatcher {
    max_tokens_per_batch: usize,
    max_items_per_batch: usize,
    current_tokens: usize,
    current: Vec<BatchItem>,
}

impl TokenAwareBatcher {
    pub(crate) fn new(max_tokens_per_batch: usize, max_items_per_batch: usize) -> Self {
        Self {
            max_tokens_per_batch: max_tokens_per_batch.max(1),
            max_items_per_batch: max_items_per_batch.max(1),
            current_tokens: 0,
            current: Vec::new(),
        }
    }

    pub(crate) fn add(&mut self, item: BatchItem) -> Option<Vec<BatchItem>> {
        if !self.current.is_empty()
            && (self.current.len() >= self.max_items_per_batch
                || self.current_tokens.saturating_add(item.token_count) > self.max_tokens_per_batch)
        {
            let batch = std::mem::take(&mut self.current);
            self.current_tokens = 0;
            self.current_tokens = self.current_tokens.saturating_add(item.token_count);
            self.current.push(item);
            return Some(batch);
        }

        self.current_tokens = self.current_tokens.saturating_add(item.token_count);
        self.current.push(item);
        None
    }

    pub(crate) fn flush(&mut self) -> Option<Vec<BatchItem>> {
        if self.current.is_empty() {
            None
        } else {
            self.current_tokens = 0;
            Some(std::mem::take(&mut self.current))
        }
    }

    pub(crate) fn remove_file(&mut self, file_path: &str) {
        if self.current.is_empty() {
            return;
        }
        self.current.retain(|item| item.file_path != file_path);
        self.current_tokens = self.current.iter().map(|item| item.token_count).sum();
    }

    pub(crate) fn has_pending(&self) -> bool {
        !self.current.is_empty()
    }
}
