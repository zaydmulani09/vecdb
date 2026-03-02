#[cfg(test)]
mod sql_tests {
    use chrono::Utc;
    use serde_json::json;
    use tempfile::tempdir;

    use crate::planner::sql::ast::{Condition, Literal, Operator};
    use crate::planner::sql::lexer::{Lexer, Token};
    use crate::planner::sql::{AstConverter, SqlParser};
    use crate::planner::QueryPlanner;
    use crate::storage::Storage;
    use crate::types::{CollectionConfig, VectorRecord};

    // ── Lexer tests ───────────────────────────────────────────────

    /// Test 1 — keywords tokenise correctly
    #[test]
    fn test_lexer_keywords() {
        let mut lex = Lexer::new("SELECT FROM WHERE AND OR NOT ORDER BY ASC DESC LIMIT LIKE");
        let tokens = lex.tokenize().unwrap();
        let expected = vec![
            Token::Select,
            Token::From,
            Token::Where,
            Token::And,
            Token::Or,
            Token::Not,
            Token::Order,
            Token::By,
            Token::Asc,
            Token::Desc,
            Token::Limit,
            Token::Like,
            Token::Eof,
        ];
        assert_eq!(tokens, expected);
    }

    /// Test 2 — comparison operators
    #[test]
    fn test_lexer_comparison_operators() {
        let mut lex = Lexer::new("= != <> < <= > >=");
        let tokens = lex.tokenize().unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Eq,
                Token::NotEq,
                Token::NotEq,
                Token::Lt,
                Token::LtEq,
                Token::Gt,
                Token::GtEq,
                Token::Eof,
            ]
        );
    }

    /// Test 3 — vector literal `[1.0, -2.5, 0.3]`
    #[test]
    fn test_lexer_vector_literal() {
        let mut lex = Lexer::new("[1.0, -2.5, 0.3]");
        let tokens = lex.tokenize().unwrap();
        assert_eq!(tokens.len(), 2); // VectorLit + Eof
        match &tokens[0] {
            Token::VectorLit(v) => {
                assert_eq!(v.len(), 3);
                assert!((v[0] - 1.0_f32).abs() < 1e-6);
                assert!((v[1] - (-2.5_f32)).abs() < 1e-6);
                assert!((v[2] - 0.3_f32).abs() < 1e-5);
            }
            other => panic!("expected VectorLit, got {:?}", other),
        }
    }

    /// Test 4 — single-quoted string with escaped quote
    #[test]
    fn test_lexer_string_escaping() {
        let mut lex = Lexer::new("'it''s a test'");
        let tokens = lex.tokenize().unwrap();
        assert_eq!(tokens[0], Token::StringLit("it's a test".into()));
    }

    /// Test 5 — JSON path operators `->` and `->>`
    #[test]
    fn test_lexer_arrows() {
        let mut lex = Lexer::new("-> ->>");
        let tokens = lex.tokenize().unwrap();
        assert_eq!(tokens[0], Token::Arrow);
        assert_eq!(tokens[1], Token::ArrowText);
        assert_eq!(tokens[2], Token::Eof);
    }

    // ── Parser tests ──────────────────────────────────────────────

    /// Test 6 — `SELECT * FROM collection LIMIT 5`
    #[test]
    fn test_parse_select_star() {
        let stmt = SqlParser::parse("SELECT * FROM mycol LIMIT 5").unwrap();
        assert_eq!(stmt.table, "mycol");
        assert!(stmt.columns.is_empty(), "SELECT * → empty columns vec");
        assert_eq!(stmt.limit, Some(5));
        assert!(stmt.where_clause.is_none());
    }

    /// Test 7 — VECTOR_SIM condition
    #[test]
    fn test_parse_vector_sim_condition() {
        let sql = "SELECT * FROM docs WHERE VECTOR_SIM(embedding, [1.0, 0.0, 0.5]) > 0.8";
        let stmt = SqlParser::parse(sql).unwrap();
        match stmt.where_clause.unwrap() {
            Condition::Vector(vc) => {
                assert_eq!(vc.column, "embedding");
                assert_eq!(vc.vector.len(), 3);
                assert_eq!(vc.op, Operator::Gt);
                assert!((vc.threshold - 0.8_f32).abs() < 1e-5);
            }
            other => panic!("expected Vector condition, got {:?}", other),
        }
    }

    /// Test 8 — scalar equality condition
    #[test]
    fn test_parse_scalar_equality() {
        let sql = "SELECT * FROM docs WHERE category = 'science'";
        let stmt = SqlParser::parse(sql).unwrap();
        match stmt.where_clause.unwrap() {
            Condition::Scalar(sc) => {
                assert_eq!(sc.field, "category");
                assert_eq!(sc.op, Operator::Eq);
                assert_eq!(sc.value, Literal::String("science".into()));
            }
            other => panic!("expected Scalar condition, got {:?}", other),
        }
    }

    /// Test 9 — combined VECTOR_SIM AND scalar condition
    #[test]
    fn test_parse_combined_condition() {
        let sql =
            "SELECT * FROM docs WHERE VECTOR_SIM(vec, [1.0, 2.0]) > 0.7 AND region = 'US' LIMIT 10";
        let stmt = SqlParser::parse(sql).unwrap();
        assert_eq!(stmt.limit, Some(10));
        match stmt.where_clause.unwrap() {
            Condition::And(left, right) => {
                assert!(
                    matches!(*left, Condition::Vector(_)),
                    "left must be Vector condition"
                );
                assert!(
                    matches!(*right, Condition::Scalar(_)),
                    "right must be Scalar condition"
                );
            }
            other => panic!("expected And condition, got {:?}", other),
        }
    }

    /// Test 10 — ORDER BY … DESC
    #[test]
    fn test_parse_order_by_desc() {
        let sql = "SELECT * FROM docs WHERE VECTOR_SIM(v, [1.0]) > 0.5 ORDER BY score DESC LIMIT 3";
        let stmt = SqlParser::parse(sql).unwrap();
        let ob = stmt.order_by.expect("order_by must be Some");
        assert_eq!(ob.field, "score");
        assert!(ob.descending);
    }

    /// Test 11 — NOT condition
    #[test]
    fn test_parse_not_condition() {
        let sql = "SELECT * FROM docs WHERE VECTOR_SIM(v, [0.1]) > 0.5 AND NOT active = false";
        let stmt = SqlParser::parse(sql).unwrap();
        match stmt.where_clause.unwrap() {
            Condition::And(_, right) => {
                assert!(
                    matches!(*right, Condition::Not(_)),
                    "right must be NOT condition"
                );
            }
            other => panic!("expected And condition, got {:?}", other),
        }
    }

    /// Test 12 — OR condition
    #[test]
    fn test_parse_or_condition() {
        let sql =
            "SELECT * FROM docs WHERE VECTOR_SIM(v, [1.0]) > 0.5 AND (cat = 'A' OR cat = 'B')";
        let stmt = SqlParser::parse(sql).unwrap();
        // Top level must be And
        assert!(
            matches!(stmt.where_clause.unwrap(), Condition::And(_, _)),
            "top-level must be And"
        );
    }

    /// Test 13 — invalid SQL returns error
    #[test]
    fn test_parse_invalid_sql_errors() {
        assert!(
            SqlParser::parse("").is_err(),
            "empty string should be an error"
        );
        assert!(
            SqlParser::parse("SELECT FROM").is_err(),
            "missing table name after FROM should error"
        );
        assert!(
            SqlParser::parse("SELECT * docs").is_err(),
            "missing FROM keyword should error"
        );
    }

    // ── Converter tests ───────────────────────────────────────────

    /// Test 14 — converter produces a plan with the correct query vector
    #[test]
    fn test_converter_dense_vector_plan() {
        let sql = "SELECT * FROM testcol WHERE VECTOR_SIM(vec, [1.0, 0.0, 0.5, 0.2]) > 0.6 LIMIT 5";
        let stmt = SqlParser::parse(sql).unwrap();
        let qv = AstConverter::extract_query_vector(&stmt);
        assert!(qv.is_some(), "query vector must be extracted");
        let qv = qv.unwrap();
        assert_eq!(qv.len(), 4);
        assert!((qv[0] - 1.0_f32).abs() < 1e-6);

        let config = CollectionConfig::new("testcol", 4);
        let converter = AstConverter::new(QueryPlanner::new(config));
        let plan = converter.convert(stmt, 500).unwrap();

        assert_eq!(plan.output_k, 5);
        assert!(!plan.use_hybrid, "no query_text → not hybrid");
        assert!(plan.filter_predicate.is_none(), "no scalar filter");
    }

    /// Test 15 — scalar equality filter is emitted in structured condition format
    #[test]
    fn test_converter_filter_is_json_object() {
        let sql = "SELECT * FROM docs WHERE VECTOR_SIM(v, [1.0, 2.0]) > 0.5 AND category = 'science' LIMIT 10";
        let stmt = SqlParser::parse(sql).unwrap();

        let config = CollectionConfig::new("docs", 2);
        let converter = AstConverter::new(QueryPlanner::new(config));
        let plan = converter.convert(stmt, 100).unwrap();

        let pred = plan
            .filter_predicate
            .as_deref()
            .expect("filter_predicate must be Some");

        let parsed: serde_json::Value =
            serde_json::from_str(pred).expect("filter predicate must be valid JSON");

        // New structured format: {"field": "category", "op": "=", "value": "science"}
        assert_eq!(
            parsed["field"].as_str(),
            Some("category"),
            "filter must have field=category"
        );
        assert_eq!(parsed["op"].as_str(), Some("="), "filter must have op==");
        assert_eq!(
            parsed["value"].as_str(),
            Some("science"),
            "filter must have value=science"
        );
    }

    // ── Storage integration tests ─────────────────────────────────

    /// Test 16 — execute_sql returns results for a dense VECTOR_SIM query
    #[test]
    fn test_execute_sql_dense() {
        let dir = tempdir().unwrap();
        let config = CollectionConfig::new("sqlcol", 4);
        let mut storage = Storage::create(dir.path(), &config).unwrap();

        let now = Utc::now();
        for i in 0..10usize {
            storage
                .upsert(VectorRecord {
                    id: format!("d{i}"),
                    vector: vec![i as f32, i as f32 + 1.0, 0.0, 1.0],
                    payload: json!({ "idx": i }),
                    text: None,
                    created_at: now,
                    updated_at: now,
                })
                .unwrap();
        }

        let sql =
            "SELECT * FROM sqlcol WHERE VECTOR_SIM(embedding, [4.0, 5.0, 0.0, 1.0]) > 0.0 LIMIT 5";
        let results = storage.execute_sql(sql).unwrap();

        assert!(
            !results.is_empty(),
            "execute_sql must return at least one result"
        );
        assert!(results.len() <= 5, "must respect LIMIT 5");
        // Results must be sorted descending by score.
        for i in 1..results.len() {
            assert!(
                results[i - 1].score >= results[i].score,
                "results not sorted at index {i}"
            );
        }
    }

    /// Test 17 — scalar filter in SQL reduces results to matching rows
    #[test]
    fn test_execute_sql_with_filter() {
        let dir = tempdir().unwrap();
        let config = CollectionConfig::new("filtcol", 4);
        let mut storage = Storage::create(dir.path(), &config).unwrap();

        let now = Utc::now();
        // Insert 5 "A" records and 5 "B" records.
        for i in 0..5usize {
            storage
                .upsert(VectorRecord {
                    id: format!("a{i}"),
                    vector: vec![i as f32, 1.0, 0.0, 0.0],
                    payload: json!({ "cat": "A" }),
                    text: None,
                    created_at: now,
                    updated_at: now,
                })
                .unwrap();
            storage
                .upsert(VectorRecord {
                    id: format!("b{i}"),
                    vector: vec![i as f32, 0.0, 1.0, 0.0],
                    payload: json!({ "cat": "B" }),
                    text: None,
                    created_at: now,
                    updated_at: now,
                })
                .unwrap();
        }

        let sql = "SELECT * FROM filtcol WHERE VECTOR_SIM(vec, [2.0, 1.0, 0.0, 0.0]) > 0.0 AND cat = 'A' LIMIT 10";
        let results = storage.execute_sql(sql).unwrap();

        assert!(
            !results.is_empty(),
            "filtered query must return at least one result"
        );
        for r in &results {
            assert_eq!(
                r.payload.get("cat").and_then(|v| v.as_str()),
                Some("A"),
                "all results must have cat='A', got payload: {}",
                r.payload
            );
        }
    }

    // ── Additional lexer coverage ─────────────────────────────────

    /// Test 18 — negative floats inside vector literal parse correctly
    #[test]
    fn test_lexer_negative_numbers_in_vector() {
        let mut lex = Lexer::new("[-1.5, 2.0, -0.5, 3.14]");
        let tokens = lex.tokenize().unwrap();
        match &tokens[0] {
            Token::VectorLit(v) => {
                assert_eq!(v.len(), 4);
                assert!((v[0] - (-1.5_f32)).abs() < 1e-6);
                assert!((v[1] - 2.0_f32).abs() < 1e-6);
                assert!((v[2] - (-0.5_f32)).abs() < 1e-6);
                assert!((v[3] - 3.14_f32).abs() < 1e-4);
            }
            other => panic!("expected VectorLit, got {:?}", other),
        }
    }
}
