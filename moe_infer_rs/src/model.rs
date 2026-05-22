pub mod config;
pub mod expert;
pub mod weights;

use std::os::fd::IntoRawFd;
use std::path::PathBuf;

use self::config::{load_model_config, ModelConfig};
use self::expert::ExpertFile;
use self::weights::WeightFile;

pub use self::expert::ExpertFile as ExpertFileType;

pub struct Model {
    pub config: ModelConfig,
    pub wf: WeightFile,
    pub expert_files: Vec<ExpertFile>,
}

impl Model {
    pub fn load(model_path: &str) -> Result<Self, String> {
        let dir = PathBuf::from(model_path);
        if !dir.exists() {
            return Err(format!("not found: {}", dir.display()));
        }
        let config = load_model_config(&dir).map_err(|e| format!("config: {}", e))?;
        let wf = WeightFile::open(
            &dir.join("model_weights.bin"),
            &dir.join("model_weights.json"),
        )
        .map_err(|e| format!("weights: {}", e))?;

        let packed_dir = dir.join("packed_experts");
        let lz4_dir = dir.join("packed_experts_lz4");
        let expert_size = config.expert_size_4bit;

        let mut expert_files = Vec::with_capacity(config.num_layers);
        for layer in 0..config.num_layers {
            let lz4_path = lz4_dir.join(format!("layer_{:02}.bin", layer));
            if lz4_path.exists() {
                use std::io::Read;
                // Header: [u32 num_experts][u32 off_0]...[u32 off_{N-1}][u32 total_size]
                let mut f = std::fs::File::open(&lz4_path)
                    .map_err(|e| format!("lz4 expert {}: {}", layer, e))?;
                let mut hdr4 = [0u8; 4];
                f.read_exact(&mut hdr4).map_err(|e| format!("lz4 hdr {}: {}", layer, e))?;
                let n = u32::from_le_bytes(hdr4) as usize;
                let off_len = n + 1;
                let mut off = vec![0u32; off_len];
                let mut off_buf = vec![0u8; off_len * 4];
                f.read_exact(&mut off_buf).map_err(|e| format!("lz4 off {}: {}", layer, e))?;
                for i in 0..off_len {
                    off[i] = u32::from_le_bytes([
                        off_buf[i*4], off_buf[i*4+1], off_buf[i*4+2], off_buf[i*4+3]
                    ]);
                }
                let fd = f.into_raw_fd();
                expert_files.push(ExpertFile::Lz4 { fd, offsets: off, expert_size });
            } else {
                let f = std::fs::File::open(packed_dir.join(format!("layer_{:02}.bin", layer)))
                    .map_err(|e| format!("expert {}: {}", layer, e))?;
                expert_files.push(ExpertFile::Raw { fd: f.into_raw_fd(), expert_size });
            }
        }

        let lz4_count = expert_files.iter().filter(|e| matches!(e, ExpertFile::Lz4 { .. })).count();
        eprintln!(
            "[model] {} layers hidden={} experts={} lz4_layers={}",
            config.num_layers, config.hidden_dim, config.num_experts, lz4_count
        );
        Ok(Model { config, wf, expert_files })
    }
}
