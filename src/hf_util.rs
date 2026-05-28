// Unified local/remote access to a HuggingFace model repository.
//
// - Local mode  – created via `HfRepo::from_local(path)` — operations
//   read directly from the local filesystem.
// - HF mode     – created via `HfRepo::from_hf(repo_id)` — files are
//   downloaded on demand into a temp staging directory and can be
//   removed after processing.

use std::fs;
use std::path::{Path, PathBuf};

pub struct HfRepo {
    repo_id: Option<String>,
    staging: PathBuf,
}

impl HfRepo {
    pub fn from_local(path: PathBuf) -> Self {
        Self { repo_id: None, staging: path }
    }

    pub fn from_hf(repo_id: &str) -> Result<Self, String> {
        eprintln!("  Input '{}' not found locally — treating as HF repo ID", repo_id);
        let staging = PathBuf::from(format!("hub/models--{}", repo_id.replace('/', "--")));
        fs::create_dir_all(&staging).map_err(|e| e.to_string())?;
        Ok(Self { repo_id: Some(repo_id.to_owned()), staging })
    }

    pub fn is_hf(&self) -> bool { self.repo_id.is_some() }

    pub fn path(&self) -> &Path { &self.staging }

    /// List immediate children of *dir* (defaults to root).  Behaves like UNIX ``ls``:
    /// returns names of files and directories at that level, not recursive.
    /// - Local: reads the filesystem.
    /// - HF: fetches the full file tree from the Hub API and filters to one level.
    pub fn ls(&self, dir: Option<&str>) -> Result<Vec<String>, String> {
        let prefix = dir.unwrap_or("");
        match &self.repo_id {
            None => {
                let target = if prefix.is_empty() {
                    self.staging.clone()
                } else {
                    self.staging.join(prefix)
                };
                let mut entries = Vec::new();
                for entry in fs::read_dir(&target).map_err(|e| e.to_string())? {
                    let entry = entry.map_err(|e| e.to_string())?;
                    entries.push(entry.file_name().to_string_lossy().to_string());
                }
                entries.sort();
                Ok(entries)
            }
            Some(repo_id) => {
                let url = format!("https://huggingface.co/api/models/{}/tree/main", repo_id);
                let body_str = ureq::get(&url)
                    .call()
                    .map_err(|e| format!("HF API error: {e}"))?
                    .into_body()
                    .read_to_string()
                    .map_err(|e| format!("read API response: {e}"))?;
                let all: Vec<serde_json::Value> =
                    serde_json::from_str(&body_str).map_err(|e| e.to_string())?;
                // Collect all paths from the API
                let all_paths: Vec<&str> = all
                    .iter()
                    .filter_map(|v| v["path"].as_str())
                    .collect();
                // Find immediate children of *prefix*: strip prefix + "/",
                // keep only entries that have exactly one more component
                let prefix_len = if prefix.is_empty() { 0 } else { prefix.len() + 1 };
                let mut seen = std::collections::BTreeSet::new();
                for path in &all_paths {
                    if prefix_len > 0 && !path.starts_with(prefix) {
                        continue;
                    }
                    let rest = &path[prefix_len..];
                    if let Some(slash) = rest.find('/') {
                        seen.insert(rest[..slash].to_string());
                    } else if !rest.is_empty() {
                        seen.insert(rest.to_string());
                    }
                }
                Ok(seen.into_iter().collect())
            }
        }
    }

    /// Ensure a file is available locally, downloading if in HF mode.
    /// Returns the local path to the file.
    pub fn ensure(&self, filename: &str) -> Result<PathBuf, String> {
        let local = self.staging.join(filename);
        if local.exists() {
            return Ok(local);
        }
        match &self.repo_id {
            None => Err(format!("file not found: {}", local.display())),
            Some(repo_id) => download_hf(repo_id, filename, &self.staging),
        }
    }

    /// Download multiple files in parallel.  Returns paths in the same order.
    /// For local repos, just verifies all files exist.
    pub fn ensure_batch(&self, filenames: &[String]) -> Result<Vec<PathBuf>, String> {
        match &self.repo_id {
            None => {
                let mut paths = Vec::with_capacity(filenames.len());
                for f in filenames {
                    let local = self.staging.join(f);
                    if !local.exists() {
                        return Err(format!("file not found: {}", local.display()));
                    }
                    paths.push(local);
                }
                Ok(paths)
            }
            Some(repo_id) => {
                let repo_id = repo_id.clone();
                let staging = self.staging.clone();
                std::thread::scope(move |s| {
                    let mut handles = Vec::with_capacity(filenames.len());
                    for f in filenames {
                        let rid = repo_id.clone();
                        let stg = staging.clone();
                        let f = f.clone();
                        handles.push(s.spawn(move || download_hf(&rid, &f, &stg)));
                    }
                    let mut results = Vec::with_capacity(filenames.len());
                    for h in handles {
                        results.push(h.join().map_err(|_| "thread panicked".to_string())??);
                    }
                    Ok(results)
                })
            }
        }
    }

    /// Remove a file from the staging directory (no-op if absent).
    pub fn remove(&self, filename: &str) {
        fs::remove_file(self.staging.join(filename)).ok();
    }
}

fn download_hf(repo_id: &str, filename: &str, dest_dir: &Path) -> Result<PathBuf, String> {
    let url = format!("https://huggingface.co/{}/resolve/main/{}", repo_id, filename);
    let dest = dest_dir.join(filename);

    if dest.exists() {
        return Ok(dest);
    }

    eprint!("  Downloading {} ...", filename);

    let mut resp = ureq::get(&url)
        .call()
        .map_err(|e| format!("HTTP error for {filename}: {e}"))?;

    let total: u64 = resp
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let data = resp
        .body_mut()
        .with_config()
        .limit(5_000_000_000) // 5 GB — shard files can be several GB
        .read_to_vec()
        .map_err(|e| format!("read {filename}: {e}"))?;
    let n = data.len() as u64;

    fs::write(&dest, &data).map_err(|e| format!("write {filename}: {e}"))?;

    eprintln!(" {:.1} MB", n as f64 / 1e6);

    if total > 0 && n != total {
        eprintln!("  WARNING: expected {total} bytes, got {n}");
    }

    Ok(dest)
}
