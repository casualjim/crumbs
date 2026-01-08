use std::io::IsTerminal;
use std::time::Duration;

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
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
