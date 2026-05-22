use std::fmt;

#[derive(Debug)]
pub enum MoEError {
    Metal(String),
    Io(std::io::Error),
    Config(String),
    Shader(String),
}

impl fmt::Display for MoEError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MoEError::Metal(s) => write!(f, "Metal error: {}", s),
            MoEError::Io(e) => write!(f, "I/O error: {}", e),
            MoEError::Config(s) => write!(f, "Config error: {}", s),
            MoEError::Shader(s) => write!(f, "Shader error: {}", s),
        }
    }
}

impl std::error::Error for MoEError {}

impl From<std::io::Error> for MoEError {
    fn from(e: std::io::Error) -> Self { MoEError::Io(e) }
}

impl From<String> for MoEError {
    fn from(s: String) -> Self { MoEError::Metal(s) }
}
