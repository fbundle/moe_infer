//! Weight file I/O -- ports moe_infer_mlx/core_src/model_weights.h
//!
//! Manages two assets:
//!   - `model_weights.json` -- manifest that lists every tensor name, its byte
//!     offset/size/shape/dtype within the flat binary blob.
//!   - The binary `.safetensors` / `.bin` file itself, read into memory for a
//!     stable address usable by Metal GPU buffers.
//!
//! Lookup from tensor name to byte offset is O(1) via an FNV-1a hash table with
//! open addressing (linear probing), matching the C implementation.

use crate::types::*;
use std::fs;
use std::mem::ManuallyDrop;

// ============================================================================
// FNV-1a hash
// ============================================================================

const FNV_OFFSET_BASIS: u32 = 2_166_136_261;
const FNV_PRIME: u32 = 16_777_619;

/// FNV-1a non-cryptographic hash, same implementation as the C code.
fn fnv1a(s: &str) -> u32 {
    let mut h = FNV_OFFSET_BASIS;
    for &b in s.as_bytes() {
        h ^= b as u32;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

// ============================================================================
// Hash table for O(1) tensor lookup
// (replaces the O(N) linear scan the manifest would otherwise require)
// ============================================================================

/// Power-of-two table size. Must be > 4x the expected number of tensors
/// (~2092 for a typical MoE model) to keep probe chains short.
const TENSOR_HT_SIZE: usize = 8192;

/// O(1) tensor-name lookup table backed by FNV-1a with open addressing.
///
/// Each slot stores an optional index into the manifest's `tensors` Vec.
/// An empty slot is `None`; a collision is resolved by walking forward
/// (with wrap-around via the mask `TENSOR_HT_SIZE - 1`).
pub struct TensorHashTable<'a> {
    /// Slot array: `Some(index_into_manifest_tensors)` or `None`.
    slots: Vec<Option<usize>>,
    /// Borrowed reference so `find()` can return `&TensorInfo` without
    /// requiring the caller to carry the manifest separately.
    tensors: &'a [TensorInfo],
}

impl<'a> TensorHashTable<'a> {
    /// Build the hash table from a loaded manifest.
    ///
    /// Prints nothing on success (the caller's `load_manifest` already
    /// printed the summary). Runs in O(N) where N = manifest.tensors.len().
    pub fn new(manifest: &'a TensorManifest) -> Self {
        let mut slots = vec![None; TENSOR_HT_SIZE];
        for (idx, ti) in manifest.tensors.iter().enumerate() {
            let mut slot = (fnv1a(&ti.name) as usize) & (TENSOR_HT_SIZE - 1);
            // Linear probe until we find an empty slot.
            while slots[slot].is_some() {
                slot = (slot + 1) & (TENSOR_HT_SIZE - 1);
            }
            slots[slot] = Some(idx);
        }
        TensorHashTable {
            slots,
            tensors: &manifest.tensors,
        }
    }

    /// Look up a tensor by name.
    ///
    /// Returns `Some(&TensorInfo)` on a match or `None` when the tensor is
    /// not present in the manifest.
    pub fn find(&self, name: &str) -> Option<&TensorInfo> {
        let hash = fnv1a(name) as usize;
        let mut slot = hash & (TENSOR_HT_SIZE - 1);
        for _ in 0..TENSOR_HT_SIZE {
            match self.slots[slot] {
                None => return None, // empty slot => definitively absent
                Some(idx) => {
                    if self.tensors[idx].name == name {
                        return Some(&self.tensors[idx]);
                    }
                }
            }
            slot = (slot + 1) & (TENSOR_HT_SIZE - 1);
        }
        None // table full (should never happen with the 4x sizing)
    }
}

// ============================================================================
// Minimal JSON parser (no serde dependency)
// ============================================================================

/// Byte-level recursive-descent JSON parser specialised for the structure
/// of `model_weights.json`:
///
/// ```json
/// {
///   "tensors": {
///     "<name>": { "offset": N, "size": N, "shape": [...], "dtype": "..." },
///     ...
///   }
/// }
/// ```
struct JsonParser<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn new(src: &'a [u8]) -> Self {
        JsonParser { src, pos: 0 }
    }

    // -- helpers ---------------------------------------------------------------

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<u8> {
        let b = self.src.get(self.pos).copied();
        if b.is_some() {
            self.pos += 1;
        }
        b
    }

    fn skip_ws(&mut self) {
        while self.src.get(self.pos).is_some_and(|&b| b.is_ascii_whitespace()) {
            self.pos += 1;
        }
    }

    fn expect(&mut self, expected: u8) -> Result<(), String> {
        self.skip_ws();
        match self.advance() {
            Some(b) if b == expected => Ok(()),
            Some(b) => Err(format!(
                "JSON: expected byte {expected} ('{}') but found {b} ('{}') at offset {}",
                expected as char,
                b as char,
                self.pos - 1
            )),
            None => Err(format!(
                "JSON: expected byte {expected} ('{}') but reached end of input",
                expected as char
            )),
        }
    }

    // -- value parsers ---------------------------------------------------------

    /// Parse a JSON string (including escape sequences).
    fn parse_string(&mut self) -> Result<String, String> {
        self.skip_ws();
        self.expect(b'"')?;

        let mut s = String::new();
        loop {
            match self.advance() {
                None => return Err("JSON: unterminated string".into()),
                Some(b'"') => return Ok(s),
                Some(b'\\') => match self.advance() {
                    None => return Err("JSON: unterminated escape sequence".into()),
                    Some(b'"') => s.push('"'),
                    Some(b'\\') => s.push('\\'),
                    Some(b'/') => s.push('/'),
                    Some(b'n') => s.push('\n'),
                    Some(b'r') => s.push('\r'),
                    Some(b't') => s.push('\t'),
                    Some(b'u') => {
                        return Err("JSON: unicode escape sequences not supported".into());
                    }
                    Some(c) => {
                        s.push('\\');
                        s.push(c as char);
                    }
                },
                Some(c) => s.push(c as char),
            }
        }
    }

    /// Parse a non-negative integer (u64).
    fn parse_u64(&mut self) -> Result<u64, String> {
        self.skip_ws();
        let start = self.pos;
        while self
            .src
            .get(self.pos)
            .is_some_and(|&b| b.is_ascii_digit())
        {
            self.pos += 1;
        }
        if self.pos == start {
            return Err(format!(
                "JSON: expected integer at offset {start}"
            ));
        }
        let raw = &self.src[start..self.pos];
        let s = std::str::from_utf8(raw)
            .map_err(|e| format!("JSON: invalid UTF-8 in number: {e}"))?;
        s.parse::<u64>()
            .map_err(|e| format!("JSON: failed to parse integer '{s}': {e}"))
    }

    /// Parse `[ int, int, ... ]` — the shape field of a tensor.
    fn parse_i32_array(&mut self) -> Result<Vec<i32>, String> {
        self.skip_ws();
        self.expect(b'[')?;
        let mut arr = Vec::new();
        loop {
            self.skip_ws();
            if self.peek() == Some(b']') {
                self.pos += 1;
                break;
            }
            if !arr.is_empty() {
                self.expect(b',')?;
                self.skip_ws();
            }
            arr.push(self.parse_u64()? as i32);
        }
        Ok(arr)
    }

    /// Parse a single tensor-info object `{ "offset":.., "size":.., ... }`.
    fn parse_tensor_info(&mut self) -> Result<TensorInfo, String> {
        self.skip_ws();
        self.expect(b'{')?;

        let mut offset: u64 = 0;
        let mut size: u64 = 0;
        let mut shape = [0i32; 4];
        let mut dtype = String::new();
        let mut ndim: i32 = 0;
        let mut first = true;

        loop {
            self.skip_ws();
            match self.peek() {
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                None => return Err("JSON: unterminated tensor-info object".into()),
                _ => {}
            }

            if !first {
                self.expect(b',')?;
            }
            first = false;

            let key = self.parse_string()?;
            self.expect(b':')?;

            match key.as_str() {
                "offset" => offset = self.parse_u64()?,
                "size" => size = self.parse_u64()?,
                "shape" => {
                    let s = self.parse_i32_array()?;
                    ndim = s.len() as i32;
                    for (i, &v) in s.iter().enumerate().take(4) {
                        shape[i] = v;
                    }
                }
                "dtype" => dtype = self.parse_string()?,
                _ => {
                    self.skip_value()?;
                }
            }
        }

        if dtype.is_empty() {
            return Err("JSON: tensor info missing required 'dtype' field".into());
        }

        Ok(TensorInfo {
            name: String::new(), // filled in by the caller
            offset,
            size,
            ndim,
            shape,
            dtype,
        })
    }

    /// Skip over any JSON value (string, number, object, array, true/false/null).
    fn skip_value(&mut self) -> Result<(), String> {
        self.skip_ws();
        match self.peek() {
            Some(b'"') => {
                self.parse_string()?;
            }
            Some(b'{') => {
                self.skip_balanced(b'{', b'}')?;
            }
            Some(b'[') => {
                self.skip_balanced(b'[', b']')?;
            }
            Some(b't') if self.src[self.pos..].starts_with(b"true") => {
                self.pos += 4;
            }
            Some(b'f') if self.src[self.pos..].starts_with(b"false") => {
                self.pos += 5;
            }
            Some(b'n') if self.src[self.pos..].starts_with(b"null") => {
                self.pos += 4;
            }
            Some(b) if b == b'-' || b.is_ascii_digit() => {
                self.parse_u64()?;
            }
            Some(b) => {
                return Err(format!(
                    "JSON: unexpected byte 0x{b:02x} ('{}') at offset {}",
                    b as char, self.pos
                ));
            }
            None => {
                return Err(format!(
                    "JSON: unexpected end of input at offset {}",
                    self.pos
                ));
            }
        }
        Ok(())
    }

    /// Skip a balanced pair of delimiters (e.g. `{ }` or `[ ]`),
    /// correctly handling nested strings.
    fn skip_balanced(&mut self, open: u8, close: u8) -> Result<(), String> {
        self.skip_ws();
        self.expect(open)?;
        let mut depth: u32 = 1;
        while depth > 0 {
            match self.advance() {
                None => return Err(format!("JSON: unterminated container (depth {depth})")),
                Some(b) if b == close => depth -= 1,
                Some(b) if b == open => depth += 1,
                Some(b'"') => {
                    // Skip through the entire string literal (interpreting escapes).
                    loop {
                        match self.advance() {
                            None => return Err("JSON: unterminated string inside skip".into()),
                            Some(b'"') => break,
                            Some(b'\\') => {
                                self.advance(); // skip the escaped char
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    // -- top-level manifest parser ------------------------------------------

    /// Parse the entire `model_weights.json` root object, extracting every
    /// tensor entry under the `"tensors"` key.
    fn parse_manifest(&mut self) -> Result<TensorManifest, String> {
        self.skip_ws();
        self.expect(b'{')?;

        let mut tensors: Vec<TensorInfo> = Vec::new();
        let mut first = true;

        loop {
            self.skip_ws();
            match self.peek() {
                None => return Err("JSON: unexpected end of root object".into()),
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                _ => {}
            }

            if !first {
                self.expect(b',')?;
            }
            first = false;

            let key = self.parse_string()?;
            self.expect(b':')?;

            if key == "tensors" {
                self.skip_ws();
                self.expect(b'{')?;

                let mut tensor_first = true;
                loop {
                    self.skip_ws();
                    match self.peek() {
                        Some(b'}') => {
                            self.pos += 1;
                            break;
                        }
                        None => {
                            return Err(
                                "JSON: unexpected end of 'tensors' object".into(),
                            );
                        }
                        _ => {}
                    }

                    if !tensor_first {
                        self.expect(b',')?;
                    }
                    tensor_first = false;

                    let tensor_name = self.parse_string()?;
                    self.expect(b':')?;

                    let mut info = self.parse_tensor_info()?;
                    info.name = tensor_name;
                    tensors.push(info);
                }
            } else {
                // Skip any other top-level keys (e.g. "__metadata__").
                self.skip_value()?;
            }
        }

        Ok(TensorManifest { tensors })
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Load a `model_weights.json` manifest file.
///
/// Parses the JSON and returns a `TensorManifest` with one `TensorInfo` per
/// entry in the `"tensors"` object.
pub fn load_manifest(json_path: &str) -> Result<TensorManifest, String> {
    let data =
        fs::read(json_path).map_err(|e| format!("Cannot read {json_path}: {e}"))?;

    let mut parser = JsonParser::new(&data);
    let manifest = parser.parse_manifest()?;

    println!(
        "[manifest] Loaded {} tensors from {json_path}",
        manifest.tensors.len()
    );

    Ok(manifest)
}

/// Open a weights file: read the binary blob into memory and load the JSON
/// manifest.  Returns a `WeightFile` whose `data` pointer is stable for the
/// remainder of the process lifetime (the backing `Vec<u8>` is leaked).
///
/// This matches the `MALLOC_WEIGHTS` codepath in the C code (aligned posix_memalign
/// + read).  The data pointer can be passed directly to Metal via `newBufferWithBytes*
///  NoCopy` or similar.
pub fn open_weights(bin_path: &str, json_path: &str) -> Result<WeightFile, String> {
    let bytes =
        fs::read(bin_path).map_err(|e| format!("Cannot read {bin_path}: {e}"))?;

    let size = bytes.len();

    // Leak the Vec so that the backing allocation is never freed and the
    // pointer stays valid for Metal's GPU<->CPU shared memory.
    let data_ptr = {
        let md = ManuallyDrop::new(bytes);
        md.as_ptr() as *mut u8
    };

    println!("[weights] loaded {:.2} GB from {bin_path}", size as f64 / 1e9);

    let manifest = load_manifest(json_path)?;

    Ok(WeightFile {
        data: data_ptr,
        size,
        manifest,
    })
}

/// Return a raw pointer to the tensor data within the weight file.
///
/// The pointer is computed as `wf.data + tensor.offset` and is valid for
/// `tensor.size` bytes.  Returns `None` when the tensor name is not found
/// in the manifest.
pub fn get_tensor_ptr<'a>(
    wf: &'a WeightFile,
    ht: &TensorHashTable,
    name: &str,
) -> Option<*const u8> {
    let info = ht.find(name)?;
    // SAFETY: `wf.data` points to a valid allocation of at least `wf.size`
    // bytes, and the manifest guarantees that `offset + size <= wf.size`.
    Some(unsafe { wf.data.add(info.offset as usize) as *const u8 })
}

/// Return a reference to the tensor metadata struct, or `None` if the name
/// is not present in the manifest.
pub fn get_tensor_info<'a>(
    ht: &'a TensorHashTable<'a>,
    name: &str,
) -> Option<&'a TensorInfo> {
    ht.find(name)
}

// ============================================================================
// Owned tensor hash table — stores indices into an owned Vec<TensorInfo>
// ============================================================================

/// Owned version of TensorHashTable that doesn't borrow from the manifest.
/// Can be stored directly in FlashMoEContext.
#[derive(Debug, Clone)]
pub struct OwnedTensorHashTable {
    slots: Vec<Option<usize>>,
    tensors: Vec<TensorInfo>,
}

impl OwnedTensorHashTable {
    pub fn new(manifest: &TensorManifest) -> Self {
        let mut slots = vec![None; TENSOR_HT_SIZE];
        let tensors = manifest.tensors.clone();
        for (idx, ti) in tensors.iter().enumerate() {
            let mut slot = (fnv1a(&ti.name) as usize) & (TENSOR_HT_SIZE - 1);
            while slots[slot].is_some() {
                slot = (slot + 1) & (TENSOR_HT_SIZE - 1);
            }
            slots[slot] = Some(idx);
        }
        Self { slots, tensors }
    }

    pub fn find(&self, name: &str) -> Option<&TensorInfo> {
        let hash = fnv1a(name) as usize;
        let mut slot = hash & (TENSOR_HT_SIZE - 1);
        for _ in 0..TENSOR_HT_SIZE {
            match self.slots[slot] {
                None => return None,
                Some(idx) => {
                    if self.tensors[idx].name == name {
                        return Some(&self.tensors[idx]);
                    }
                }
            }
            slot = (slot + 1) & (TENSOR_HT_SIZE - 1);
        }
        None
    }
}
