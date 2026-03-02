use crate::errors::{Result, VecDbError};
use crate::types::Vector;

// ─────────────────────────────────────────────────────────────────
// Token
// ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // ── Keywords ─────────────────────────────────────────────────
    Select,
    From,
    Where,
    And,
    Or,
    Not,
    Order,
    By,
    Asc,
    Desc,
    Limit,
    Like,
    True,
    False,
    // ── Functions ─────────────────────────────────────────────────
    VectorSim,
    Embed,
    // ── Identifiers and literals ──────────────────────────────────
    Identifier(String),
    StringLit(String),
    IntLit(i64),
    FloatLit(f64),
    /// Inline vector literal `[f32, ...]`.
    VectorLit(Vector),
    // ── Comparison operators ──────────────────────────────────────
    Eq,    // =
    NotEq, // != or <>
    Lt,    // <
    LtEq,  // <=
    Gt,    // >
    GtEq,  // >=
    // ── JSON path operators ───────────────────────────────────────
    Arrow,     // ->
    ArrowText, // ->>
    // ── Punctuation ──────────────────────────────────────────────
    LParen,    // (
    RParen,    // )
    LBracket,  // [
    RBracket,  // ]
    Comma,     // ,
    Dot,       // .
    Star,      // *
    Semicolon, // ;
    Eof,
}

// ─────────────────────────────────────────────────────────────────
// Lexer
// ─────────────────────────────────────────────────────────────────

pub struct Lexer {
    input: Vec<char>,
    pos: usize,
}

impl Lexer {
    pub fn new(input: &str) -> Self {
        Self {
            input: input.chars().collect(),
            pos: 0,
        }
    }

    /// Tokenize the entire input.
    pub fn tokenize(&mut self) -> Result<Vec<Token>> {
        let mut tokens = Vec::new();
        loop {
            let tok = self.next_token()?;
            let done = tok == Token::Eof;
            tokens.push(tok);
            if done {
                break;
            }
        }
        Ok(tokens)
    }

    // ── Low-level helpers ─────────────────────────────────────────

    fn peek(&self) -> Option<char> {
        self.input.get(self.pos).copied()
    }

    fn peek_next(&self) -> Option<char> {
        self.input.get(self.pos + 1).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let ch = self.input.get(self.pos).copied();
        if ch.is_some() {
            self.pos += 1;
        }
        ch
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(c) if c.is_whitespace()) {
            self.advance();
        }
    }

    // ── Main dispatch ─────────────────────────────────────────────

    fn next_token(&mut self) -> Result<Token> {
        self.skip_whitespace();

        match self.peek() {
            None => Ok(Token::Eof),
            Some(c) => match c {
                '(' => {
                    self.advance();
                    Ok(Token::LParen)
                }
                ')' => {
                    self.advance();
                    Ok(Token::RParen)
                }
                '[' => {
                    self.advance();
                    self.scan_vector_lit()
                }
                ']' => {
                    self.advance();
                    Ok(Token::RBracket)
                }
                ',' => {
                    self.advance();
                    Ok(Token::Comma)
                }
                '.' => {
                    self.advance();
                    Ok(Token::Dot)
                }
                '*' => {
                    self.advance();
                    Ok(Token::Star)
                }
                ';' => {
                    self.advance();
                    Ok(Token::Semicolon)
                }
                '=' => {
                    self.advance();
                    Ok(Token::Eq)
                }
                '!' => {
                    self.advance();
                    if self.peek() == Some('=') {
                        self.advance();
                        Ok(Token::NotEq)
                    } else {
                        Err(VecDbError::InvalidQuery(format!(
                            "unexpected '!' at pos {}",
                            self.pos
                        )))
                    }
                }
                '<' => {
                    self.advance();
                    match self.peek() {
                        Some('=') => {
                            self.advance();
                            Ok(Token::LtEq)
                        }
                        Some('>') => {
                            self.advance();
                            Ok(Token::NotEq)
                        }
                        _ => Ok(Token::Lt),
                    }
                }
                '>' => {
                    self.advance();
                    if self.peek() == Some('=') {
                        self.advance();
                        Ok(Token::GtEq)
                    } else {
                        Ok(Token::Gt)
                    }
                }
                '-' => {
                    self.advance();
                    match self.peek() {
                        Some('>') => {
                            self.advance();
                            if self.peek() == Some('>') {
                                self.advance();
                                Ok(Token::ArrowText) // ->>
                            } else {
                                Ok(Token::Arrow) // ->
                            }
                        }
                        Some(d) if d.is_ascii_digit() => self.scan_number(true),
                        _ => Err(VecDbError::InvalidQuery(format!(
                            "unexpected '-' at pos {}",
                            self.pos
                        ))),
                    }
                }
                '\'' => self.scan_string(),
                c if c.is_ascii_digit() => self.scan_number(false),
                c if c.is_alphabetic() || c == '_' => self.scan_ident_or_keyword(),
                other => Err(VecDbError::InvalidQuery(format!(
                    "unexpected character '{}' at pos {}",
                    other, self.pos
                ))),
            },
        }
    }

    // ── Single-quoted string ──────────────────────────────────────

    /// Scan `'...'` — `''` inside the string is an escaped single-quote.
    fn scan_string(&mut self) -> Result<Token> {
        self.advance(); // consume opening '
        let mut s = String::new();
        loop {
            match self.advance() {
                None => {
                    return Err(VecDbError::InvalidQuery(
                        "unterminated string literal".into(),
                    ))
                }
                Some('\'') => {
                    if self.peek() == Some('\'') {
                        self.advance(); // consume second '
                        s.push('\'');
                    } else {
                        break;
                    }
                }
                Some(c) => s.push(c),
            }
        }
        Ok(Token::StringLit(s))
    }

    // ── Number ────────────────────────────────────────────────────

    /// Scan an integer or float.  `negative` is true when the `-` has
    /// already been consumed by the caller.
    fn scan_number(&mut self, negative: bool) -> Result<Token> {
        let mut s = String::new();
        if negative {
            s.push('-');
        }
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            s.push(self.advance().unwrap());
        }
        // Detect float: digit `.` digit ...
        let is_float =
            self.peek() == Some('.') && matches!(self.peek_next(), Some(c) if c.is_ascii_digit());
        if is_float {
            s.push(self.advance().unwrap()); // '.'
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                s.push(self.advance().unwrap());
            }
            // Optional exponent `e` / `E`
            if matches!(self.peek(), Some('e') | Some('E')) {
                s.push(self.advance().unwrap());
                if matches!(self.peek(), Some('+') | Some('-')) {
                    s.push(self.advance().unwrap());
                }
                while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                    s.push(self.advance().unwrap());
                }
            }
            let v: f64 = s
                .parse()
                .map_err(|_| VecDbError::InvalidQuery(format!("invalid float literal: {s}")))?;
            Ok(Token::FloatLit(v))
        } else {
            let v: i64 = s
                .parse()
                .map_err(|_| VecDbError::InvalidQuery(format!("invalid integer literal: {s}")))?;
            Ok(Token::IntLit(v))
        }
    }

    // ── Inline vector literal  `[f32, ...]` ──────────────────────

    /// Scan the body of `[...]`.  The opening `[` has already been consumed.
    fn scan_vector_lit(&mut self) -> Result<Token> {
        let mut floats: Vec<f32> = Vec::new();
        self.skip_whitespace();

        // Empty vector `[]`
        if self.peek() == Some(']') {
            self.advance();
            return Ok(Token::VectorLit(floats));
        }

        loop {
            self.skip_whitespace();

            let negative = if self.peek() == Some('-') {
                self.advance();
                true
            } else {
                false
            };

            let mut s = String::new();
            if negative {
                s.push('-');
            }
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                s.push(self.advance().unwrap());
            }
            if self.peek() == Some('.') {
                s.push(self.advance().unwrap());
                while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                    s.push(self.advance().unwrap());
                }
            }
            // optional exponent
            if matches!(self.peek(), Some('e') | Some('E')) {
                s.push(self.advance().unwrap());
                if matches!(self.peek(), Some('+') | Some('-')) {
                    s.push(self.advance().unwrap());
                }
                while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                    s.push(self.advance().unwrap());
                }
            }
            if s.is_empty() || s == "-" {
                return Err(VecDbError::InvalidQuery(
                    "expected number in vector literal".into(),
                ));
            }
            let v: f32 = s.parse().map_err(|_| {
                VecDbError::InvalidQuery(format!("invalid float in vector literal: {s}"))
            })?;
            floats.push(v);

            self.skip_whitespace();
            match self.peek() {
                Some(',') => {
                    self.advance();
                }
                Some(']') => {
                    self.advance();
                    break;
                }
                other => {
                    return Err(VecDbError::InvalidQuery(format!(
                        "expected ',' or ']' in vector literal, got {:?}",
                        other
                    )))
                }
            }
        }
        Ok(Token::VectorLit(floats))
    }

    // ── Identifier / keyword ──────────────────────────────────────

    fn scan_ident_or_keyword(&mut self) -> Result<Token> {
        let mut s = String::new();
        while matches!(self.peek(), Some(c) if c.is_alphanumeric() || c == '_') {
            s.push(self.advance().unwrap());
        }
        Ok(match s.to_uppercase().as_str() {
            "SELECT" => Token::Select,
            "FROM" => Token::From,
            "WHERE" => Token::Where,
            "AND" => Token::And,
            "OR" => Token::Or,
            "NOT" => Token::Not,
            "ORDER" => Token::Order,
            "BY" => Token::By,
            "ASC" => Token::Asc,
            "DESC" => Token::Desc,
            "LIMIT" => Token::Limit,
            "LIKE" => Token::Like,
            "TRUE" => Token::True,
            "FALSE" => Token::False,
            "VECTOR_SIM" => Token::VectorSim,
            "EMBED" => Token::Embed,
            _ => Token::Identifier(s),
        })
    }
}
