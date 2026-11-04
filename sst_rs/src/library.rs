//! Reading `path_library.npz`, the saved paths the Python demo drives with (`loadPathLibrary` in `sst/smoothing.py`).
//!
//! An .npz is a zip of .npy arrays, one (n, 8) float64 array per path: x, y, heading, speed, steering, cost, time,
//! and whether the state is a tree node.
use std::io::Read;

use flate2::read::DeflateDecoder;

use crate::tree::State;

fn u16_at(b: &[u8], i: usize) -> usize {
    u16::from_le_bytes([b[i], b[i + 1]]) as usize
}

fn u32_at(b: &[u8], i: usize) -> usize {
    u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]) as usize
}

/// The (name, contents) of every file in a zip archive, from its central directory.
fn unzip(zip: &[u8]) -> Result<Vec<(String, Vec<u8>)>, String> {
    let eocd = (0..zip.len().saturating_sub(21)).rev().find(|&i| zip[i..i + 4] == [0x50, 0x4b, 0x05, 0x06])
        .ok_or("not a zip file")?;
    let (count, mut at) = (u16_at(zip, eocd + 10), u32_at(zip, eocd + 16));
    let mut files = Vec::new();
    for _ in 0..count {
        if zip.get(at..at + 4) != Some(&[0x50, 0x4b, 0x01, 0x02]) {
            return Err("bad central directory".into());
        }
        let (method, stored_size) = (u16_at(zip, at + 10), u32_at(zip, at + 20));
        let (name_len, extra_len, comment_len) = (u16_at(zip, at + 28), u16_at(zip, at + 30), u16_at(zip, at + 32));
        let local = u32_at(zip, at + 42);
        let name = String::from_utf8_lossy(&zip[at + 46..at + 46 + name_len]).into_owned();
        at += 46 + name_len + extra_len + comment_len;
        let data = local + 30 + u16_at(zip, local + 26) + u16_at(zip, local + 28);
        let contents = match method {
            8 => {
                let mut out = Vec::new();
                DeflateDecoder::new(&zip[data..]).read_to_end(&mut out).map_err(|e| e.to_string())?;
                out
            }
            0 if stored_size != 0xFFFF_FFFF => zip[data..data + stored_size].to_vec(),
            _ => return Err(format!("{name}: unsupported zip entry")),
        };
        files.push((name, contents));
    }
    Ok(files)
}

/// The rows of a 2-D little-endian float64 .npy array.
fn npy_rows(npy: &[u8]) -> Result<Vec<Vec<f64>>, String> {
    if npy.get(..6) != Some(b"\x93NUMPY") {
        return Err("not an .npy file".into());
    }
    let (header_len, start) = if npy[6] == 1 { (u16_at(npy, 8), 10) } else { (u32_at(npy, 8), 12) };
    let header = String::from_utf8_lossy(&npy[start..start + header_len]);
    if !header.contains("'<f8'") || header.contains("'fortran_order': True") {
        return Err(format!("unsupported array: {header}"));
    }
    let shape = header.split('(').nth(1).and_then(|s| s.split(')').next()).ok_or("no shape")?;
    let dims: Vec<usize> = shape.split(',').filter_map(|d| d.trim().parse().ok()).collect();
    let [rows, cols] = dims[..] else { return Err(format!("not 2-D: {shape}")) };
    let data = &npy[start + header_len..];
    Ok((0..rows)
        .map(|r| (0..cols).map(|c| f64::from_le_bytes(data[(r * cols + c) * 8..][..8].try_into().unwrap())).collect())
        .collect())
}

/// The paths in a path library file, in order (path_0, path_1, ...).
pub fn load_path_library(file: &str) -> Result<Vec<Vec<State>>, String> {
    let zip = std::fs::read(file).map_err(|e| format!("{file}: {e}"))?;
    let mut files = unzip(&zip)?;
    files.sort_by_key(|(name, _)| name.trim_start_matches("path_").trim_end_matches(".npy").parse::<usize>().unwrap_or(0));
    files
        .iter()
        .map(|(_, npy)| {
            Ok(npy_rows(npy)?
                .iter()
                .map(|r| State { x: r[0], y: r[1], a: r[2], s: r[3], steer: r[4], cost: r[5], time: r[6], node: r[7] != 0.0 })
                .collect())
        })
        .collect()
}
