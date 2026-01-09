use std::io::IsTerminal;
use std::time::Duration;

use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use tokio::sync::watch;

pub(crate) fn spinner(message: &str) -> Option<ProgressBar> {
    if !std::io::stderr().is_terminal() {
        return None;
    }

    let progress = ProgressBar::new_spinner();
    progress.set_draw_target(ProgressDrawTarget::stderr());
    progress.set_style(ProgressStyle::default_spinner());
    progress.set_message(message.to_string());
    progress.enable_steady_tick(Duration::from_millis(120));
    Some(progress)
}

pub(crate) struct IndexProgress {
    _multi: MultiProgress,
    files: ProgressBar,
    embedding: ProgressBar,
    history: ProgressBar,
}

impl IndexProgress {
    pub(crate) fn new() -> Option<Self> {
        if !std::io::stderr().is_terminal() {
            return None;
        }

        let multi = MultiProgress::with_draw_target(ProgressDrawTarget::stderr());
        let style = ProgressStyle::with_template(
            "{prefix:>9} [{bar:40.cyan/blue}] {pos:>5}/{len:<5} {msg}",
        )
        .unwrap()
        .progress_chars("##-");

        let files = multi.add(ProgressBar::new(0));
        files.set_style(style.clone());
        files.set_prefix("files");
        files.set_message("starting");

        let embedding = multi.add(ProgressBar::new(0));
        embedding.set_style(style.clone());
        embedding.set_prefix("embedding");
        embedding.set_message("idle");

        let history = multi.add(ProgressBar::new(1));
        history.set_style(style);
        history.set_prefix("history");
        history.set_message("pending");

        Some(Self {
            _multi: multi,
            files,
            embedding,
            history,
        })
    }

    pub(crate) fn update_files(
        &self,
        files_processed: usize,
        total_files: usize,
        stream_done: bool,
    ) {
        let total = total_files.max(1) as u64;
        self.files.set_length(total);
        self.files.set_position((files_processed as u64).min(total));
        let message = if stream_done { "done" } else { "scanning" };
        self.files.set_message(message.to_string());
    }

    pub(crate) fn update_embedding(
        &self,
        completed_batches: usize,
        total_batches: usize,
        stream_done: bool,
    ) {
        let total = total_batches.max(1) as u64;
        self.embedding.set_length(total);
        self.embedding
            .set_position((completed_batches as u64).min(total));
        let pending = total_batches.saturating_sub(completed_batches);
        let message = if pending > 0 {
            format!("{pending} pending")
        } else if stream_done && total_batches > 0 {
            "done".to_string()
        } else {
            "idle".to_string()
        };
        self.embedding.set_message(message);
    }

    pub(crate) fn start_history(&self) {
        self.history.set_length(1);
        self.history.set_position(0);
        self.history.set_message("running");
    }

    pub(crate) fn finish_history(&self) {
        self.history.set_length(1);
        self.history.set_position(1);
        self.history.set_message("done");
    }

    pub(crate) fn finish_and_clear(&self) {
        self.files.finish_and_clear();
        self.embedding.finish_and_clear();
        self.history.finish_and_clear();
    }
}

pub(crate) fn watch_spinner(
    message: &'static str,
) -> Option<(ProgressBar, watch::Sender<&'static str>)> {
    let progress = spinner(message)?;
    let (tx, mut rx) = watch::channel(message);
    let progress_clone = progress.clone();
    tokio::spawn(async move {
        while rx.changed().await.is_ok() {
            let message = *rx.borrow();
            progress_clone.set_message(message);
            progress_clone.tick();
        }
    });
    Some((progress, tx))
}
