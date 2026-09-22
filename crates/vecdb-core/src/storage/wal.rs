use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::errors::{Result, VecDbError};
use crate::types::{VectorId, VectorRecord};

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn checksum(data: &[u8]) -> u32 {
    xxhash_rust::xxh3::xxh3_64(data) as u32
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalEntry {
    Insert {
        id: VectorId,
        mmap_index: usize,
        record: VectorRecord,
    },
    Delete {
        id: VectorId,
    },
    Checkpoint {
        entry_count: usize,
        timestamp: u64,
    },
}

pub struct WriteAheadLog {
    pub path: PathBuf,
    pub entry_count: usize,
    file: File,
}

fn open_rw(path: &Path, create: bool) -> Result<File> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(create)
        .open(path)
        .map_err(|e| VecDbError::StorageError(format!("wal open failed: {e}")))
}

impl WriteAheadLog {
    pub fn open(path: &Path) -> Result<Self> {
        let file = open_rw(path, true)?;

        let mut wal = Self {
            path: path.to_path_buf(),
            entry_count: 0,
            file,
        };

        // Count existing valid entries.
        let entries = wal.replay()?;
        wal.entry_count = entries.len();
        Ok(wal)
    }

    pub fn append(&mut self, entry: &WalEntry) -> Result<()> {
        let payload =
            serde_json::to_vec(entry).map_err(|e| VecDbError::SerializationError(e.to_string()))?;

        let length = payload.len() as u32;
        let cksum = checksum(&payload);

        // Seek to end before writing (replaces O_APPEND — safe for single-writer WAL).
        self.file
            .seek(SeekFrom::End(0))
            .map_err(|e| VecDbError::StorageError(format!("wal seek failed: {e}")))?;

        self.file
            .write_all(&length.to_le_bytes())
            .map_err(|e| VecDbError::StorageError(format!("wal write failed: {e}")))?;
        self.file
            .write_all(&payload)
            .map_err(|e| VecDbError::StorageError(format!("wal write failed: {e}")))?;
        self.file
            .write_all(&cksum.to_le_bytes())
            .map_err(|e| VecDbError::StorageError(format!("wal write failed: {e}")))?;
        // sync_all = FlushFileBuffers on Windows: commits write buffer to page cache
        // so that reads via any handle (including std::fs::read) see current data.
        self.file
            .sync_all()
            .map_err(|e| VecDbError::StorageError(format!("wal flush failed: {e}")))?;

        self.entry_count += 1;
        tracing::debug!("WAL append: entry_count={}", self.entry_count);
        Ok(())
    }

    /// Append many entries with a single `sync_all` at the end. Used by bulk
    /// load, where per-entry fsync would dominate (one fsync per vector). The
    /// batch is still crash-consistent: on a crash mid-batch, replay stops at
    /// the last fully-written entry (length/checksum framing), and any partial
    /// tail is ignored.
    pub fn append_batch(&mut self, entries: &[WalEntry]) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        self.file
            .seek(SeekFrom::End(0))
            .map_err(|e| VecDbError::StorageError(format!("wal seek failed: {e}")))?;

        let mut buf = Vec::new();
        for entry in entries {
            let payload = serde_json::to_vec(entry)
                .map_err(|e| VecDbError::SerializationError(e.to_string()))?;
            let length = payload.len() as u32;
            let cksum = checksum(&payload);
            buf.extend_from_slice(&length.to_le_bytes());
            buf.extend_from_slice(&payload);
            buf.extend_from_slice(&cksum.to_le_bytes());
        }
        self.file
            .write_all(&buf)
            .map_err(|e| VecDbError::StorageError(format!("wal write failed: {e}")))?;
        self.file
            .sync_all()
            .map_err(|e| VecDbError::StorageError(format!("wal flush failed: {e}")))?;

        self.entry_count += entries.len();
        Ok(())
    }

    pub fn replay(&mut self) -> Result<Vec<WalEntry>> {
        // Read entire file via a fresh open/read/close cycle.
        // This is the most reliable approach on all platforms — avoids any
        // per-handle buffering or file-pointer aliasing issues.
        let data = std::fs::read(&self.path)
            .map_err(|e| VecDbError::StorageError(format!("wal read failed: {e}")))?;
        let mut cursor = std::io::Cursor::new(data);

        let mut entries = Vec::new();

        loop {
            // Read 4-byte length prefix.
            let mut len_buf = [0u8; 4];
            match cursor.read_exact(&mut len_buf) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => {
                    tracing::warn!("WAL replay: error reading length: {e}");
                    break;
                }
            }
            let length = u32::from_le_bytes(len_buf) as usize;

            let mut payload = vec![0u8; length];
            if let Err(e) = cursor.read_exact(&mut payload) {
                tracing::warn!("WAL replay: truncated payload: {e}");
                break;
            }

            let mut cksum_buf = [0u8; 4];
            if let Err(e) = cursor.read_exact(&mut cksum_buf) {
                tracing::warn!("WAL replay: truncated checksum: {e}");
                break;
            }
            let stored = u32::from_le_bytes(cksum_buf);
            let computed = checksum(&payload);
            if computed != stored {
                tracing::warn!(
                    "WAL replay: checksum mismatch (computed={computed}, stored={stored}), stopping"
                );
                break;
            }

            match serde_json::from_slice::<WalEntry>(&payload) {
                Ok(entry) => entries.push(entry),
                Err(e) => {
                    tracing::warn!("WAL replay: deserialization error: {e}, stopping");
                    break;
                }
            }
        }

        tracing::info!("WAL replay: {} entries recovered", entries.len());
        Ok(entries)
    }

    pub fn checkpoint(&mut self) -> Result<()> {
        let entry = WalEntry::Checkpoint {
            entry_count: self.entry_count,
            timestamp: now_unix(),
        };
        self.append(&entry)
    }

    pub fn truncate_after_checkpoint(&mut self) -> Result<()> {
        let entries = self.replay()?;

        let Some(checkpoint_pos) = entries
            .iter()
            .rposition(|e| matches!(e, WalEntry::Checkpoint { .. }))
        else {
            return Ok(());
        };

        let remaining: Vec<_> = entries.into_iter().skip(checkpoint_pos + 1).collect();
        let remaining_count = remaining.len();

        // Truncate and rewrite via a separate handle so we can resize from 0.
        let mut new_file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&self.path)
            .map_err(|e| VecDbError::StorageError(format!("wal truncate open failed: {e}")))?;

        for entry in &remaining {
            let payload = serde_json::to_vec(entry)
                .map_err(|e| VecDbError::SerializationError(e.to_string()))?;
            let length = payload.len() as u32;
            let cksum = checksum(&payload);
            new_file
                .write_all(&length.to_le_bytes())
                .map_err(|e| VecDbError::StorageError(format!("wal rewrite failed: {e}")))?;
            new_file
                .write_all(&payload)
                .map_err(|e| VecDbError::StorageError(format!("wal rewrite failed: {e}")))?;
            new_file
                .write_all(&cksum.to_le_bytes())
                .map_err(|e| VecDbError::StorageError(format!("wal rewrite failed: {e}")))?;
        }
        new_file
            .flush()
            .map_err(|e| VecDbError::StorageError(format!("wal rewrite flush failed: {e}")))?;
        drop(new_file);

        // Reopen own handle so it sees the truncated content.
        self.file = open_rw(&self.path, false)?;
        // Position at end ready for future appends.
        self.file
            .seek(SeekFrom::End(0))
            .map_err(|e| VecDbError::StorageError(format!("wal seek failed: {e}")))?;

        self.entry_count = remaining_count;
        tracing::info!("WAL truncated: {} entries remaining", self.entry_count);
        Ok(())
    }

    pub fn entry_count(&self) -> usize {
        self.entry_count
    }
}
