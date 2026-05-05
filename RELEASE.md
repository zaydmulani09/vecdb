# Release Process

## Prerequisites
- Rust stable toolchain installed
- `cargo` in PATH
- On Windows: MSYS2 with mingw64, dlltool in PATH
- GitHub repo with Actions enabled
- (Optional) crates.io account with API token for publishing vecdb-core

## Steps

### 1. Verify tests pass

```
PATH="$PATH:/c/msys64/mingw64/bin" cargo test --workspace
```

### 2. Verify clippy is clean

```
PATH="$PATH:/c/msys64/mingw64/bin" cargo clippy --workspace -- -D warnings
```

### 3. Bump versions
Ensure all crates in `Cargo.toml` are set to the target version (e.g. 0.1.0).

### 4. Update CHANGELOG.md
Fill in the release date. Add any last-minute entries.

### 5. Commit and tag

```
git add -A
git commit -m "chore: release v0.1.0"
git tag v0.1.0
git push origin main --tags
```

Pushing the tag triggers the GitHub Actions release workflow, which builds
binaries for all targets and uploads them as release assets.

### 6. Create GitHub Release
- Go to Releases → Draft a new release
- Select tag v0.1.0
- Copy the CHANGELOG.md v0.1.0 section as the release body
- Publish

### 7. (Optional) Publish vecdb-core to crates.io

```
PATH="$PATH:/c/msys64/mingw64/bin" cargo publish -p vecdb-core
```

## Release Targets (built by CI)
- x86_64-unknown-linux-musl
- aarch64-unknown-linux-musl
- x86_64-apple-darwin
- x86_64-pc-windows-gnu

## Binary naming convention
vecdb-{version}-{target}.tar.gz (linux/mac)
vecdb-{version}-{target}.zip (windows)
