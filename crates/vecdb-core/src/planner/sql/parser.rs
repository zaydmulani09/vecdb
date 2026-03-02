use std::mem::discriminant;

use crate::errors::{Result, VecDbError};

use super::ast::{
    Condition, Literal, Operator, OrderBy, ScalarCondition, SelectStatement, VectorCondition,
};
use super::lexer::{Lexer, Token};

// ─────────────────────────────────────────────────────────────────
// SqlParser
// ─────────────────────────────────────────────────────────────────

pub struct SqlParser {
    tokens: Vec<Token>,
    pos: usize,
}

impl SqlParser {
    /// Parse a SQL string into a `SelectStatement`.
    pub fn parse(sql: &str) -> Result<SelectStatement> {
        let mut lexer = Lexer::new(sql);
        let tokens = lexer.tokenize()?;
        let mut parser = SqlParser { tokens, pos: 0 };
        parser.parse_select()
    }

    // ── Token navigation ─────────────────────────────────────────

    /// Peek at the current token without consuming it.
    fn peek(&self) -> Token {
        self.tokens.get(self.pos).cloned().unwrap_or(Token::Eof)
    }

    /// Consume and return the current token.
    fn advance(&mut self) -> Token {
        if self.pos < self.tokens.len() {
            let tok = self.tokens[self.pos].clone();
            self.pos += 1;
            tok
        } else {
            Token::Eof
        }
    }

    /// Consume the current token, returning an error if its variant does not
    /// match `expected`.  Only the variant discriminant is compared — inner
    /// values are ignored (e.g. `expect(Token::Identifier(String::new()))` will
    /// match any `Identifier(_)`).
    fn expect(&mut self, expected: Token) -> Result<()> {
        let tok = self.advance();
        if discriminant(&tok) == discriminant(&expected) {
            Ok(())
        } else {
            Err(VecDbError::InvalidQuery(format!(
                "expected {:?}, got {:?}",
                expected, tok
            )))
        }
    }

    /// Return `true` if the current token's variant matches `expected`.
    fn is_next(&self, expected: &Token) -> bool {
        discriminant(&self.peek()) == discriminant(expected)
    }

    // ── Grammar rules ─────────────────────────────────────────────

    fn parse_select(&mut self) -> Result<SelectStatement> {
        self.expect(Token::Select)?;
        let columns = self.parse_columns()?;
        self.expect(Token::From)?;

        let table = match self.advance() {
            Token::Identifier(s) => s,
            tok => {
                return Err(VecDbError::InvalidQuery(format!(
                    "expected table name, got {:?}",
                    tok
                )))
            }
        };

        let where_clause = if self.is_next(&Token::Where) {
            self.advance(); // consume WHERE
            Some(self.parse_where_clause()?)
        } else {
            None
        };

        let order_by = if self.is_next(&Token::Order) {
            self.advance(); // consume ORDER
            self.expect(Token::By)?;
            Some(self.parse_order_by()?)
        } else {
            None
        };

        let limit = if self.is_next(&Token::Limit) {
            self.advance(); // consume LIMIT
            match self.advance() {
                Token::IntLit(n) => Some(n as usize),
                tok => {
                    return Err(VecDbError::InvalidQuery(format!(
                        "expected integer after LIMIT, got {:?}",
                        tok
                    )))
                }
            }
        } else {
            None
        };

        // Optional trailing semicolon.
        if self.is_next(&Token::Semicolon) {
            self.advance();
        }

        Ok(SelectStatement {
            table,
            columns,
            where_clause,
            order_by,
            limit,
        })
    }

    fn parse_columns(&mut self) -> Result<Vec<String>> {
        if self.is_next(&Token::Star) {
            self.advance();
            return Ok(vec![]);
        }

        let mut cols = Vec::new();
        loop {
            match self.peek() {
                Token::Star => {
                    self.advance();
                    return Ok(vec![]);
                }
                Token::Identifier(_) => {
                    if let Token::Identifier(s) = self.advance() {
                        cols.push(s);
                    }
                }
                tok => {
                    return Err(VecDbError::InvalidQuery(format!(
                        "expected column name or '*', got {:?}",
                        tok
                    )))
                }
            }
            if self.is_next(&Token::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        Ok(cols)
    }

    fn parse_where_clause(&mut self) -> Result<Condition> {
        self.parse_or_condition()
    }

    fn parse_or_condition(&mut self) -> Result<Condition> {
        let mut left = self.parse_and_condition()?;
        while self.is_next(&Token::Or) {
            self.advance(); // consume OR
            let right = self.parse_and_condition()?;
            left = Condition::Or(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_and_condition(&mut self) -> Result<Condition> {
        let mut left = self.parse_atom_condition()?;
        while self.is_next(&Token::And) {
            self.advance(); // consume AND
            let right = self.parse_atom_condition()?;
            left = Condition::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_atom_condition(&mut self) -> Result<Condition> {
        // NOT <condition>
        if self.is_next(&Token::Not) {
            self.advance();
            let inner = self.parse_atom_condition()?;
            return Ok(Condition::Not(Box::new(inner)));
        }

        // Parenthesised condition
        if self.is_next(&Token::LParen) {
            self.advance(); // consume (
            let cond = self.parse_or_condition()?;
            self.expect(Token::RParen)?;
            return Ok(cond);
        }

        // VECTOR_SIM(...)
        if self.is_next(&Token::VectorSim) {
            let vc = self.parse_vector_condition()?;
            return Ok(Condition::Vector(vc));
        }

        // Scalar condition
        let sc = self.parse_scalar_condition()?;
        Ok(Condition::Scalar(sc))
    }

    fn parse_vector_condition(&mut self) -> Result<VectorCondition> {
        self.expect(Token::VectorSim)?;
        self.expect(Token::LParen)?;

        let column = match self.advance() {
            Token::Identifier(s) => s,
            tok => {
                return Err(VecDbError::InvalidQuery(format!(
                    "expected column name in VECTOR_SIM, got {:?}",
                    tok
                )))
            }
        };

        self.expect(Token::Comma)?;

        let vector = match self.advance() {
            Token::VectorLit(v) => v,
            tok => {
                return Err(VecDbError::InvalidQuery(format!(
                    "expected vector literal in VECTOR_SIM, got {:?}",
                    tok
                )))
            }
        };

        self.expect(Token::RParen)?;

        let op = self.parse_operator()?;

        let threshold = match self.advance() {
            Token::FloatLit(f) => f as f32,
            Token::IntLit(i) => i as f32,
            tok => {
                return Err(VecDbError::InvalidQuery(format!(
                    "expected similarity threshold, got {:?}",
                    tok
                )))
            }
        };

        Ok(VectorCondition {
            column,
            vector,
            op,
            threshold,
        })
    }

    fn parse_scalar_condition(&mut self) -> Result<ScalarCondition> {
        let field = self.parse_field_path()?;
        let op = self.parse_operator()?;
        let value = self.parse_literal()?;
        Ok(ScalarCondition { field, op, value })
    }

    /// Parse a (possibly path-qualified) field name.
    ///
    /// Handles:
    /// - `field`
    /// - `table.field`
    /// - `field->'key'`
    /// - `field->>'key'`
    fn parse_field_path(&mut self) -> Result<String> {
        let mut path = match self.advance() {
            Token::Identifier(s) => s,
            tok => {
                return Err(VecDbError::InvalidQuery(format!(
                    "expected field name, got {:?}",
                    tok
                )))
            }
        };

        loop {
            match self.peek() {
                Token::Dot => {
                    self.advance();
                    match self.advance() {
                        Token::Identifier(s) => path = format!("{path}.{s}"),
                        tok => {
                            return Err(VecDbError::InvalidQuery(format!(
                                "expected field after '.', got {:?}",
                                tok
                            )))
                        }
                    }
                }
                Token::Arrow => {
                    self.advance();
                    match self.advance() {
                        Token::StringLit(s) => path = format!("{path}->'{s}'"),
                        Token::Identifier(s) => path = format!("{path}->{s}"),
                        tok => {
                            return Err(VecDbError::InvalidQuery(format!(
                                "expected key after '->', got {:?}",
                                tok
                            )))
                        }
                    }
                }
                Token::ArrowText => {
                    self.advance();
                    match self.advance() {
                        Token::StringLit(s) => path = format!("{path}->>'{}'", s),
                        Token::Identifier(s) => path = format!("{path}->>{s}"),
                        tok => {
                            return Err(VecDbError::InvalidQuery(format!(
                                "expected key after '->>', got {:?}",
                                tok
                            )))
                        }
                    }
                }
                _ => break,
            }
        }
        Ok(path)
    }

    fn parse_operator(&mut self) -> Result<Operator> {
        match self.advance() {
            Token::Eq => Ok(Operator::Eq),
            Token::NotEq => Ok(Operator::NotEq),
            Token::Lt => Ok(Operator::Lt),
            Token::LtEq => Ok(Operator::LtEq),
            Token::Gt => Ok(Operator::Gt),
            Token::GtEq => Ok(Operator::GtEq),
            Token::Like => Ok(Operator::Like),
            tok => Err(VecDbError::InvalidQuery(format!(
                "expected comparison operator, got {:?}",
                tok
            ))),
        }
    }

    fn parse_literal(&mut self) -> Result<Literal> {
        match self.advance() {
            Token::StringLit(s) => Ok(Literal::String(s)),
            Token::FloatLit(f) => Ok(Literal::Float(f)),
            Token::IntLit(i) => Ok(Literal::Integer(i)),
            Token::True => Ok(Literal::Bool(true)),
            Token::False => Ok(Literal::Bool(false)),
            tok => Err(VecDbError::InvalidQuery(format!(
                "expected literal value, got {:?}",
                tok
            ))),
        }
    }

    fn parse_order_by(&mut self) -> Result<OrderBy> {
        let field = match self.advance() {
            Token::Identifier(s) => s,
            tok => {
                return Err(VecDbError::InvalidQuery(format!(
                    "expected field for ORDER BY, got {:?}",
                    tok
                )))
            }
        };
        let descending = if self.is_next(&Token::Desc) {
            self.advance();
            true
        } else if self.is_next(&Token::Asc) {
            self.advance();
            false
        } else {
            false // default: ascending
        };
        Ok(OrderBy { field, descending })
    }
}
