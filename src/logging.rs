use eyre::{Result, eyre};
use tracing_subscriber::EnvFilter;

pub(crate) fn init() -> Result<()> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .try_init()
        .map_err(|err| eyre!(err))?;
    Ok(())
}
