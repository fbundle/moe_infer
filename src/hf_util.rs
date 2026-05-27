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
    _tmp_dir: Option<tempfile::TempDir>,
}

impl HfRepo {
    pub fn from_local(path: PathBuf) -> Self {
        Self { repo_id: None, staging: path, _tmp_dir: None }
    }

    pub fn from_hf(repo_id: &str) -> Result<Self, String> {
        eprintln!("  Input '{}' not found locally — treating as HF repo ID", repo_id);
        let tmp = tempfile::tempdir().map_err(|e| e.to_string())?;
        let staging = tmp.path().to_path_buf();
        Ok(Self { repo_id: Some(repo_id.to_owned()), staging, _tmp_dir: Some(tmp) })
    }

    pub fn is_hf(&self) -> bool { self.repo_id.is_some() }

    pub fn path(&self) -> &Path { &self.staging }

    /// List files in the repo.
    /// - Local: lists files in the directory.
    /// - HF: fetches file list from the Hub API.
    pub fn ls(&self) -> Result<Vec<String>, String> {
        match &self.repo_id {
            None => {
                let mut files = Vec::new();
                for entry in fs::read_dir(&self.staging).map_err(|e| e.to_string())? {
                    let entry = entry.map_err(|e| e.to_string())?;
                    if entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                        files.push(entry.file_name().to_string_lossy().to_string());
                    }
                }
                files.sort();
                Ok(files)
            }
            Some(repo_id) => {
                let url = format!("https://huggingface.co/api/models/{}/tree/main", repo_id);
                let body_str = ureq::get(&url)
                    .call()
                    .map_err(|e| format!("HF API error: {e}"))?
                    .into_body()
                    .read_to_string()
                    .map_err(|e| format!("read API response: {e}"))?;
                let entries: Vec<serde_json::Value> =
                    serde_json::from_str(&body_str).map_err(|e| e.to_string())?;
                let files: Vec<String> = entries
                    .iter()
                    .filter(|v| v["type"] == "file")
                    .filter_map(|v| v["path"].as_str().map(String::from))
                    .collect();
                Ok(files)
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

    let resp = ureq::get(&url)
        .call()
        .map_err(|e| format!("HTTP error for {filename}: {e}"))?;

    let total: u64 = resp
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let data = resp
        .into_body()
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
