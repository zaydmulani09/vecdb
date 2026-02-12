use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use memmap2::MmapMut;

use crate::errors::{Result, VecDbError};
use crate::types::Vector;

const MAGIC: u32 = 0x5645_4344;
const VERSION: u32 = 1;
const HEADER_SIZE: usize = 64;

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub struct MmapVectorStore {
    pub path: PathBuf,
    pub dimension: usize,
    pub count: usize,
    pub capacity: usize,
    mmap: MmapMut,
    file: File,
}

impl MmapVectorStore {
    pub fn create(path: &Path, dimension: usize, initial_capacity: usize) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|e| VecDbError::StorageError(format!("create failed: {e}")))?;

        let total_size = (HEADER_SIZE + initial_capacity * dimension * 4) as u64;
        file.set_len(total_size)
            .map_err(|e| VecDbError::StorageError(format!("set_len failed: {e}")))?;

        let mut mmap = unsafe {
            MmapMut::map_mut(&file)
                .map_err(|e| VecDbError::StorageError(format!("mmap failed: {e}")))?
        };

        let now = now_unix();
        mmap[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        mmap[4..8].copy_from_slice(&VERSION.to_le_bytes());
        mmap[8..16].copy_from_slice(&(dimension as u64).to_le_bytes());
        mmap[16..24].copy_from_slice(&0u64.to_le_bytes());
        mmap[24..32].copy_from_slice(&(initial_capacity as u64).to_le_bytes());
        mmap[32..40].copy_from_slice(&now.to_le_bytes());
        mmap[40..48].copy_from_slice(&now.to_le_bytes());
        // bytes 48..64 reserved — zeroed by set_len

        mmap.flush()
            .map_err(|e| VecDbError::StorageError(format!("flush failed: {e}")))?;

        Ok(Self {
            path: path.to_path_buf(),
            dimension,
            count: 0,
            capacity: initial_capacity,
            mmap,
            file,
        })
    }

    pub fn open(path: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|e| VecDbError::StorageError(format!("open failed: {e}")))?;

        let mmap = unsafe {
            MmapMut::map_mut(&file)
                .map_err(|e| VecDbError::StorageError(format!("mmap failed: {e}")))?
        };

        // Performance: on Linux, tell the kernel to pre-fault pages into the page cache.
        // Reduces first-read latency when the store is small enough to fit in RAM.
        // Before: OS demand-faults pages one at a time on first access
        // After:  kernel prefetches pages in the background via MADV_WILLNEED
        #[cfg(target_os = "linux")]
        {
            let _ = mmap.advise(memmap2::Advice::WillNeed);
        }

        let magic = u32::from_le_bytes(mmap[0..4].try_into().unwrap());
        if magic != MAGIC {
            return Err(VecDbError::StorageError(format!(
                "invalid magic: {magic:#x}"
            )));
        }
        let version = u32::from_le_bytes(mmap[4..8].try_into().unwrap());
        if version != VERSION {
            return Err(VecDbError::StorageError(format!(
                "unsupported version: {version}"
            )));
        }

        let dimension = u64::from_le_bytes(mmap[8..16].try_into().unwrap()) as usize;
        let count = u64::from_le_bytes(mmap[16..24].try_into().unwrap()) as usize;
        let capacity = u64::from_le_bytes(mmap[24..32].try_into().unwrap()) as usize;

        Ok(Self {
            path: path.to_path_buf(),
            dimension,
            count,
            capacity,
            mmap,
            file,
        })
    }

    pub fn append(&mut self, vector: &Vector) -> Result<usize> {
        if vector.len() != self.dimension {
            return Err(VecDbError::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }
        if self.count >= self.capacity {
            self.grow()?;
        }

        let offset = HEADER_SIZE + self.count * self.dimension * 4;
        for (i, &val) in vector.iter().enumerate() {
            self.mmap[offset + i * 4..offset + i * 4 + 4].copy_from_slice(&val.to_le_bytes());
        }

        let index = self.count;
        self.count += 1;
        self.mmap[16..24].copy_from_slice(&(self.count as u64).to_le_bytes());
        self.mmap[40..48].copy_from_slice(&now_unix().to_le_bytes());
        self.mmap
            .flush()
            .map_err(|e| VecDbError::StorageError(format!("flush failed: {e}")))?;

        Ok(index)
    }

    pub fn overwrite(&mut self, index: usize, vector: &Vector) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(VecDbError::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }
        if index >= self.count {
            return Err(VecDbError::StorageError("index out of bounds".into()));
        }

        let offset = HEADER_SIZE + index * self.dimension * 4;
        for (i, &val) in vector.iter().enumerate() {
            self.mmap[offset + i * 4..offset + i * 4 + 4].copy_from_slice(&val.to_le_bytes());
        }
        self.mmap[40..48].copy_from_slice(&now_unix().to_le_bytes());
        self.mmap
            .flush()
            .map_err(|e| VecDbError::StorageError(format!("flush failed: {e}")))?;
        Ok(())
    }

    pub fn get(&self, index: usize) -> Result<Vector> {
        if index >= self.count {
            return Err(VecDbError::StorageError("index out of bounds".into()));
        }
        let offset = HEADER_SIZE + index * self.dimension * 4;
        let mut vector = Vec::with_capacity(self.dimension);
        for i in 0..self.dimension {
            let bytes: [u8; 4] = self.mmap[offset + i * 4..offset + i * 4 + 4]
                .try_into()
                .unwrap();
            vector.push(f32::from_le_bytes(bytes));
        }
        Ok(vector)
    }

    pub fn get_all(&self) -> Result<Vec<(usize, Vector)>> {
        // Performance: on Linux, hint that pages will be read sequentially.
        // Before: random-access page faults on each vector read
        // After:  kernel uses read-ahead (MADV_SEQUENTIAL) to prefetch contiguous pages
        #[cfg(target_os = "linux")]
        {
            let _ = self.mmap.advise(memmap2::Advice::Sequential);
        }
        let mut out = Vec::with_capacity(self.count);
        for i in 0..self.count {
            out.push((i, self.get(i)?));
        }
        Ok(out)
    }

    fn grow(&mut self) -> Result<()> {
        let new_capacity = self.capacity * 2;
        let new_size = (HEADER_SIZE + new_capacity * self.dimension * 4) as u64;

        // Swap out the current mmap with a small anon mapping so the file can be resized.
        let placeholder = MmapMut::map_anon(1)
            .map_err(|e| VecDbError::StorageError(format!("anon mmap failed: {e}")))?;
        drop(std::mem::replace(&mut self.mmap, placeholder));

        self.file
            .set_len(new_size)
            .map_err(|e| VecDbError::StorageError(format!("set_len failed: {e}")))?;

        self.mmap = unsafe {
            MmapMut::map_mut(&self.file)
                .map_err(|e| VecDbError::StorageError(format!("remap failed: {e}")))?
        };

        self.capacity = new_capacity;
        self.mmap[24..32].copy_from_slice(&(new_capacity as u64).to_le_bytes());

        tracing::info!("MmapVectorStore grew to capacity {}", new_capacity);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn flush(&self) -> Result<()> {
        self.mmap
            .flush()
            .map_err(|e| VecDbError::StorageError(format!("flush failed: {e}")))
    }
}
