/// BPE tokenizer — Rust port of tokenizer.h.
///
/// Loads the .bin format created by export_tokenizer.py.
/// Supports encode (text -> token IDs) with the same pretokenization
/// and BPE merge algorithm as the C implementation.
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

const BPE_MAX_TOKEN_LEN: usize = 256;
const BPE_MAX_PIECES: usize = 8192;

/// Tokenizer state (port of bpe_tokenizer struct).
pub struct BpeTokenizer {
    pub vocab: Vec<VocabEntry>,
    pub merges: Vec<MergeEntry>,
    pub added: Vec<AddedToken>,
    /// Vocab hash: key (string bytes) -> token ID
    vocab_map: HashMap<Vec<u8>, u32>,
    /// Merge hash: key (a_bytes + b'\xff' + b_bytes) -> priority index
    merge_map: HashMap<Vec<u8>, u32>,
    byte_char: [u32; 256],
    char_byte: [u8; 512],
    vocab_size: u32,
}

#[derive(Debug, Clone)]
pub struct VocabEntry {
    pub id: u32,
    pub str_bytes: Vec<u8>, // UTF-8 bytes of the BPE token string
}

#[derive(Debug, Clone)]
pub struct MergeEntry {
    pub a: Vec<u8>,
    pub b: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct AddedToken {
    pub id: u32,
    pub str_bytes: Vec<u8>,
}

fn read_u32_be(f: &mut File) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u16_be(f: &mut File) -> std::io::Result<u16> {
    let mut buf = [0u8; 2];
    f.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn fnv1a_hash(data: &[u8]) -> u32 {
    let mut h: u32 = 0x811c9dc5;
    for &b in data {
        h ^= b as u32;
        h = h.wrapping_mul(0x01000193);
    }
    h
}

impl BpeTokenizer {
    /// Load a tokenizer from a .bin file (created by export_tokenizer.py).
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let mut f = File::open(path)?;

        // Magic header
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != b"BPET" {
            anyhow::bail!("Invalid tokenizer magic: {:?}", magic);
        }

        let version = read_u32_be(&mut f)?;
        if version != 1 {
            anyhow::bail!("Unsupported tokenizer version: {}", version);
        }

        let vocab_size = read_u32_be(&mut f)?;
        let num_merges = read_u32_be(&mut f)?;
        let num_added = read_u32_be(&mut f)?;

        // Read vocab
        let mut vocab = Vec::with_capacity(vocab_size as usize);
        for _ in 0..vocab_size {
            let id = read_u32_be(&mut f)?;
            let str_len = read_u16_be(&mut f)? as usize;
            let mut str_bytes = vec![0u8; str_len];
            f.read_exact(&mut str_bytes)?;
            vocab.push(VocabEntry { id, str_bytes });
        }

        // Read merges
        let mut merges = Vec::with_capacity(num_merges as usize);
        for _ in 0..num_merges {
            let len_a = read_u16_be(&mut f)? as usize;
            let mut a = vec![0u8; len_a];
            f.read_exact(&mut a)?;
            let len_b = read_u16_be(&mut f)? as usize;
            let mut b = vec![0u8; len_b];
            f.read_exact(&mut b)?;
            merges.push(MergeEntry { a, b });
        }

        // Read added tokens
        let mut added = Vec::with_capacity(num_added as usize);
        for _ in 0..num_added {
            let id = read_u32_be(&mut f)?;
            let str_len = read_u16_be(&mut f)? as usize;
            let mut str_bytes = vec![0u8; str_len];
            f.read_exact(&mut str_bytes)?;
            added.push(AddedToken { id, str_bytes });
        }

        // Build vocab hashmap
        let mut vocab_map = HashMap::new();
        for entry in &vocab {
            vocab_map.insert(entry.str_bytes.clone(), entry.id);
        }

        // Build merge hashmap
        let mut merge_map = HashMap::new();
        for (i, merge) in merges.iter().enumerate() {
            let mut key = merge.a.clone();
            key.push(0xff);
            key.extend_from_slice(&merge.b);
            merge_map.insert(key, i as u32);
        }

        // Build byte-unicode table (GPT-2 compatible)
        let mut byte_char = [0u32; 256];
        let mut char_byte = [0u8; 512];
        let mut n = 0;
        for b in 0..256u32 {
            if (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF) {
                byte_char[b as usize] = b;
            } else {
                byte_char[b as usize] = 256 + n;
                n += 1;
            }
        }
        for b in 0..256 {
            let cp = byte_char[b] as usize;
            if cp < 512 {
                char_byte[cp] = b as u8;
            }
        }

        eprintln!(
            "bpe_load: {} vocab, {} merges, {} added tokens",
            vocab_size, num_merges, num_added
        );

        Ok(BpeTokenizer {
            vocab,
            merges,
            added,
            vocab_map,
            merge_map,
            byte_char,
            char_byte,
            vocab_size,
        })
    }

    /// Encode text to token IDs. Returns the list of token IDs.
    pub fn encode(&self, text: &str, max_ids: usize) -> Vec<u32> {
        let text_bytes = text.as_bytes();
        let text_len = text_bytes.len();
        let mut out_ids = Vec::with_capacity(max_ids.min(4096));
        let mut pos = 0;

        while pos < text_len && out_ids.len() < max_ids {
            // Check for added tokens (special tokens like <|im_start|>)
            let mut found_added = false;
            let mut best_len = 0;
            let mut best_id = 0;

            for added in &self.added {
                let alen = added.str_bytes.len();
                if alen > best_len && pos + alen <= text_len {
                    if &text_bytes[pos..pos + alen] == added.str_bytes.as_slice() {
                        best_len = alen;
                        best_id = added.id;
                        found_added = true;
                    }
                }
            }

            if found_added {
                out_ids.push(best_id);
                pos += best_len;
                continue;
            }

            // Find next added token to know chunk boundary
            let mut chunk_end = text_len;
            for added in &self.added {
                let alen = added.str_bytes.len();
                if let Some(j) = text_bytes[pos..].windows(alen).position(|w| w == added.str_bytes.as_slice()) {
                    let j_abs = pos + j;
                    if j_abs < chunk_end {
                        chunk_end = j_abs;
                    }
                }
            }

            let _chunk_len = chunk_end - pos;
            let chunk = &text_bytes[pos..chunk_end];

            // Pretokenize
            let spans = self.pretokenize(chunk);
            let mut bpe_buf = Vec::with_capacity(BPE_MAX_TOKEN_LEN * 4);

            for span in spans {
                let piece = &chunk[span.0..span.1];
                bpe_buf.clear();
                self.bytes_to_bpe_str(piece, &mut bpe_buf);

                let ids = self.bpe_process(&bpe_buf, max_ids - out_ids.len());
                out_ids.extend_from_slice(&ids);
                if out_ids.len() >= max_ids {
                    break;
                }
            }

            pos = chunk_end;
        }

        out_ids
    }

    /// Convert raw bytes to BPE-safe UTF-8 string.
    fn bytes_to_bpe_str(&self, raw: &[u8], out: &mut Vec<u8>) {
        out.clear();
        for &byte in raw {
            let cp = self.byte_char[byte as usize];
            if cp < 0x80 {
                out.push(cp as u8);
            } else if cp < 0x800 {
                out.push(0xC0 | (cp >> 6) as u8);
                out.push(0x80 | (cp & 0x3F) as u8);
            } else {
                out.push(0xE0 | (cp >> 12) as u8);
                out.push(0x80 | ((cp >> 6) & 0x3F) as u8);
                out.push(0x80 | (cp & 0x3F) as u8);
            }
        }
    }

    /// Pretokenize into spans.
    fn pretokenize(&self, text: &[u8]) -> Vec<(usize, usize)> {
        let mut spans = Vec::new();
        let text_len = text.len();
        let mut i = 0;

        while i < text_len && spans.len() < BPE_MAX_PIECES {
            let c = text[i];

            // Whitespace handling
            if is_ws(c) {
                let start = i;
                let mut has_nl = false;
                let mut j = i;
                while j < text_len && is_ws(text[j]) {
                    if is_nl(text[j]) { has_nl = true; }
                    j += 1;
                }
                if has_nl || j >= text_len {
                    spans.push((start, j));
                    i = j;
                    continue;
                }
                if j - start > 1 {
                    spans.push((start, j - 1));
                    i = j - 1;
                    continue;
                }
            }

            let lead_sp = c == b' ' && i + 1 < text_len;
            let ws = i;
            let wi = if lead_sp { i + 1 } else { i };

            if wi < text_len {
                let wc = text[wi];

                // Contractions: 's, 't, 'm, 'd, 're, 've, 'll
                if !lead_sp && wc == b'\'' && wi + 1 < text_len {
                    let nc = text[wi + 1] | 0x20;
                    if nc == b's' || nc == b't' || nc == b'm' || nc == b'd' {
                        spans.push((wi, wi + 2));
                        i = wi + 2;
                        continue;
                    }
                    if wi + 2 < text_len {
                        let nc2 = text[wi + 2] | 0x20;
                        if (nc == b'r' && nc2 == b'e') || (nc == b'v' && nc2 == b'e') || (nc == b'l' && nc2 == b'l') {
                            spans.push((wi, wi + 3));
                            i = wi + 3;
                            continue;
                        }
                    }
                }

                if wc >= 0xC0 || is_alpha(wc) {
                    let mut j = wi;
                    while j < text_len {
                        let jc = text[j];
                        if jc >= 0xC0 {
                            j += if jc < 0xE0 { 2 } else if jc < 0xF0 { 3 } else { 4 };
                        } else if is_alpha(jc) {
                            j += 1;
                        } else {
                            break;
                        }
                    }
                    if j > wi {
                        spans.push((ws, j));
                        i = j;
                        continue;
                    }
                }

                if wc >= b'0' && wc <= b'9' {
                    spans.push((ws, wi + 1));
                    i = wi + 1;
                    continue;
                }

                if !is_alnum_ws(wc) {
                    let mut j = wi;
                    while j < text_len && !is_alnum_ws(text[j]) {
                        j += 1;
                    }
                    while j < text_len && is_nl(text[j]) {
                        j += 1;
                    }
                    spans.push((ws, j));
                    i = j;
                    continue;
                }
            }

            spans.push((i, i + 1));
            i += 1;
        }

        spans
    }

    /// BPE merge processing: iterate until no more merges can be applied.
    fn bpe_process(&self, bpe_str: &[u8], max_ids: usize) -> Vec<u32> {
        if bpe_str.is_empty() {
            return Vec::new();
        }

        #[derive(Clone, Copy)]
        struct Piece {
            start: usize,
            len: u16,
            prev: i32,
            next: i32,
        }

        let mut pieces: Vec<Piece> = Vec::with_capacity(BPE_MAX_PIECES);
        let mut i = 0;
        while i < bpe_str.len() && pieces.len() < BPE_MAX_PIECES {
            let c = bpe_str[i];
            let clen = if c < 0x80 { 1 } else if c < 0xE0 { 2 } else if c < 0xF0 { 3 } else { 4 };
            let clen = clen.min(bpe_str.len() - i);
            let num = pieces.len();
            pieces.push(Piece {
                start: i,
                len: clen as u16,
                prev: num as i32 - 1,
                next: num as i32 + 1,
            });
            i += clen;
        }
        if pieces.is_empty() {
            return Vec::new();
        }
        let last = pieces.len() - 1;
        pieces[last].next = -1;

        // Arena for merged strings
        let mut arena: Vec<u8> = Vec::with_capacity(1024 * 16);
        let mut active = pieces.len();

        while active > 1 {
            let mut best_prio = u32::MAX;
            let mut best_idx = -1i32;

            let mut ci = 0i32;
            while ci != -1 {
                let ni = pieces[ci as usize].next;
                if ni == -1 { break; }

                let a = &bpe_str[pieces[ci as usize].start..][..pieces[ci as usize].len as usize];
                let b = &bpe_str[pieces[ni as usize].start..][..pieces[ni as usize].len as usize];

                let mut key = a.to_vec();
                key.push(0xff);
                key.extend_from_slice(b);

                if let Some(&prio) = self.merge_map.get(&key) {
                    if prio < best_prio {
                        best_prio = prio;
                        best_idx = ci;
                    }
                }
                ci = ni;
            }

            if best_idx == -1 { break; }

            let bi = best_idx as usize;
            let ni = pieces[bi].next as usize;
            let new_len = pieces[bi].len + pieces[ni].len;
            if new_len as usize > BPE_MAX_TOKEN_LEN { break; }

            // Merge: if pieces are adjacent in the original string, just extend
            if pieces[bi].start + pieces[bi].len as usize == pieces[ni].start {
                pieces[bi].len = new_len;
            } else {
                // Need to copy to arena
                if arena.len() + new_len as usize > arena.capacity() {
                    arena.clear();
                }
                let arena_start = arena.len();
                arena.extend_from_slice(&bpe_str[pieces[bi].start..][..pieces[bi].len as usize]);
                arena.extend_from_slice(&bpe_str[pieces[ni].start..][..pieces[ni].len as usize]);
                pieces[bi].start = arena_start;
                pieces[bi].len = new_len;
            }

            pieces[bi].next = pieces[ni].next;
            if pieces[ni].next != -1 {
                let nni = pieces[ni].next as usize;
                pieces[nni].prev = best_idx;
            }
            active -= 1;
        }

        // Convert final pieces to token IDs
        let mut out_ids = Vec::new();
        let mut ci2 = 0i32;
        while ci2 != -1 && out_ids.len() < max_ids {
            let piece = &bpe_str[pieces[ci2 as usize].start..][..pieces[ci2 as usize].len as usize];
            if let Some(&id) = self.vocab_map.get(piece) {
                out_ids.push(id);
            } else {
                // Fallback: encode individual bytes
                for &byte in piece {
                    let cp = self.byte_char[byte as usize];
                    let single = if cp < 0x80 {
                        vec![cp as u8]
                    } else if cp < 0x800 {
                        vec![0xC0 | (cp >> 6) as u8, 0x80 | (cp & 0x3F) as u8]
                    } else {
                        vec![0xE0 | (cp >> 12) as u8, 0x80 | ((cp >> 6) & 0x3F) as u8]
                    };
                    if let Some(&byte_id) = self.vocab_map.get(&single) {
                        if out_ids.len() < max_ids {
                            out_ids.push(byte_id);
                        }
                    }
                }
            }
            ci2 = pieces[ci2 as usize].next;
        }

        out_ids
    }
}

fn is_ws(c: u8) -> bool { c == b' ' || c == b'\t' || c == b'\n' || c == b'\r' }
fn is_nl(c: u8) -> bool { c == b'\n' || c == b'\r' }
fn is_alpha(c: u8) -> bool { (c >= b'A' && c <= b'Z') || (c >= b'a' && c <= b'z') }
fn is_alnum_ws(c: u8) -> bool { is_alpha(c) || (c >= b'0' && c <= b'9') || is_ws(c) || c >= 0xC0 }
