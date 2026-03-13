use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;

use crate::{
    client::VecDbClient,
    output,
    types::{UpsertRecord, UpsertRequest, UpsertResponse},
};

// ── JSONL deserialization ──────────────────────────────────────────────────

#[derive(Deserialize)]
struct JsonlRecord {
    id: String,
    vector: Vec<f32>,
    text: Option<String>,
    payload: Option<serde_json::Value>,
}

// ── Simple CSV parser ─────────────────────────────────────────────────────
//
// Handles quoted fields (RFC 4180 style). Vectors are stored as JSON arrays
// inside quoted fields, e.g.: doc1,"[0.1,0.2,0.3]",optional text
//
// No external `csv` crate — keeps the dep tree minimal.

fn parse_csv_fields(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut chars = line.chars().peekable();

    while chars.peek().is_some() {
        let field = if chars.peek() == Some(&'"') {
            chars.next(); // consume opening quote
            let mut s = String::new();
            loop {
                match chars.next() {
                    None => break,
                    Some('"') => {
                        if chars.peek() == Some(&'"') {
                            chars.next(); // "" escape → one literal "
                            s.push('"');
                        } else {
                            // end of quoted field; consume trailing comma if present
                            if chars.peek() == Some(&',') {
                                chars.next();
                            }
                            break;
                        }
                    }
                    Some(c) => s.push(c),
                }
            }
            s
        } else {
            let mut s = String::new();
            while chars.peek().is_some() && chars.peek() != Some(&',') {
                s.push(chars.next().unwrap());
            }
            if chars.peek() == Some(&',') {
                chars.next();
            }
            s
        };
        fields.push(field);
    }
    fields
}

// ── Record parsing ────────────────────────────────────────────────────────

fn parse_jsonl_records(content: &str) -> (Vec<UpsertRecord>, usize) {
    let mut records = Vec::new();
    let mut errors = 0usize;

    for (line_no, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match serde_json::from_str::<JsonlRecord>(line) {
            Ok(r) => records.push(UpsertRecord {
                id: r.id,
                vector: r.vector,
                text: r.text,
                payload: r.payload,
            }),
            Err(e) => {
                eprintln!("warning: skipping line {}: {}", line_no + 1, e);
                errors += 1;
            }
        }
    }
    (records, errors)
}

fn parse_csv_records(content: &str) -> Result<(Vec<UpsertRecord>, usize)> {
    let mut lines = content.lines();
    let header_line = lines
        .next()
        .ok_or_else(|| anyhow!("CSV file is empty"))?
        .trim()
        .to_string();

    let headers = parse_csv_fields(&header_line);
    let id_col = headers
        .iter()
        .position(|h| h == "id")
        .ok_or_else(|| anyhow!("CSV missing required 'id' column"))?;
    let vec_col = headers
        .iter()
        .position(|h| h == "vector")
        .ok_or_else(|| anyhow!("CSV missing required 'vector' column"))?;
    let text_col = headers.iter().position(|h| h == "text");

    let mut records = Vec::new();
    let mut errors = 0usize;

    for (line_no, line) in lines.enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fields = parse_csv_fields(line);

        let id = match fields.get(id_col) {
            Some(v) if !v.is_empty() => v.clone(),
            _ => {
                eprintln!("warning: skipping CSV row {}: missing 'id'", line_no + 2);
                errors += 1;
                continue;
            }
        };

        let vector: Vec<f32> = match fields.get(vec_col) {
            Some(v) => match serde_json::from_str(v) {
                Ok(vv) => vv,
                Err(e) => {
                    eprintln!(
                        "warning: skipping CSV row {}: invalid vector: {}",
                        line_no + 2,
                        e
                    );
                    errors += 1;
                    continue;
                }
            },
            None => {
                eprintln!(
                    "warning: skipping CSV row {}: missing 'vector'",
                    line_no + 2
                );
                errors += 1;
                continue;
            }
        };

        let text = text_col
            .and_then(|col| fields.get(col))
            .filter(|s| !s.is_empty())
            .cloned();

        records.push(UpsertRecord {
            id,
            vector,
            text,
            payload: None,
        });
    }
    Ok((records, errors))
}

// ── Main ingest function ───────────────────────────────────────────────────

pub async fn run(
    client: &VecDbClient,
    collection: &str,
    file: Option<&Path>,
    batch_size: usize,
    dry_run: bool,
) -> Result<()> {
    // Read input: file or stdin
    let content = if let Some(path) = file {
        std::fs::read_to_string(path)
            .with_context(|| format!("cannot read file: {}", path.display()))?
    } else {
        let stdin = std::io::stdin();
        let reader = BufReader::new(stdin.lock());
        reader
            .lines()
            .collect::<std::io::Result<Vec<_>>>()
            .context("failed to read stdin")?
            .join("\n")
    };

    // Determine format from extension (default: jsonl)
    let use_csv = file
        .and_then(|p| p.extension())
        .map(|e| e.eq_ignore_ascii_case("csv"))
        .unwrap_or(false);

    let (records, parse_errors) = if use_csv {
        parse_csv_records(&content)?
    } else {
        parse_jsonl_records(&content)
    };

    let total = records.len();
    eprintln!(
        "Parsed {} records ({} parse errors){}",
        total,
        parse_errors,
        if dry_run {
            " — dry run, not sending"
        } else {
            ""
        }
    );

    if dry_run || total == 0 {
        return Ok(());
    }

    // Send in batches
    let path = format!("/collections/{}/vectors", collection);
    let mut inserted_total = 0usize;
    let mut updated_total = 0usize;
    let mut error_total = 0usize;
    let mut batch_no = 0usize;

    for chunk in records.chunks(batch_size) {
        batch_no += 1;
        let body = UpsertRequest {
            records: chunk.to_vec(),
        };
        let resp: UpsertResponse = client
            .post(&path, &body)
            .await
            .with_context(|| format!("batch {} failed", batch_no))?;

        inserted_total += resp.inserted;
        updated_total += resp.updated;
        error_total += resp.errors.len();

        eprintln!(
            "Batch {}: inserted={} updated={} errors={}",
            batch_no,
            resp.inserted,
            resp.updated,
            resp.errors.len()
        );
        for e in &resp.errors {
            eprintln!("  error [{}]: {}", e.id, e.error);
        }
    }

    let summary = UpsertResponse {
        inserted: inserted_total,
        updated: updated_total,
        errors: vec![],
        time_ms: 0,
    };
    output::print_upsert_result(&summary);
    eprintln!("Total errors: {}", error_total);

    Ok(())
}
