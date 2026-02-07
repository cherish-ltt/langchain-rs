//! 输出解析器
//!
//! 提供从 LLM 文本输出中提取结构化数据的解析器。

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// 解析器错误
#[derive(Debug, Error)]
pub enum ParseError {
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Pattern not found: {0}")]
    PatternNotFound(String),

    #[error("Invalid format: {0}")]
    InvalidFormat(String),

    #[error("Empty output")]
    EmptyOutput,
}

/// 输出解析器 trait
pub trait OutputParser<T>: Send + Sync {
    /// 解析文本并返回结构化数据
    fn parse(&self, text: &str) -> Result<T, ParseError>;

    /// 获取解析器的格式提示
    fn get_format_instructions(&self) -> String;
}

/// JSON 解析器
pub struct JsonParser<T> {
    phantom: std::marker::PhantomData<T>,
}

impl<T: for<'de> Deserialize<'de>> JsonParser<T> {
    pub fn new() -> Self {
        Self {
            phantom: std::marker::PhantomData,
        }
    }
}

impl<T: for<'de> Deserialize<'de>> Default for JsonParser<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: for<'de> Deserialize<'de> + Send + Sync> OutputParser<T> for JsonParser<T> {
    fn parse(&self, text: &str) -> Result<T, ParseError> {
        let json_str = extract_json(text)?;
        serde_json::from_str(&json_str).map_err(ParseError::Json)
    }

    fn get_format_instructions(&self) -> String {
        "Output must be valid JSON format. Wrap the JSON in ```json ``` code blocks if needed."
            .to_owned()
    }
}

/// 从文本中提取 JSON
fn extract_json(text: &str) -> Result<String, ParseError> {
    let text = text.trim();

    // 1. 查找 ```json 代码块
    if let Some(start) = text.find("```json") {
        let json_start = start + 7;
        if let Some(end) = text[json_start..].find("```") {
            return Ok(text[json_start..json_start + end].trim().to_owned());
        }
    }

    // 2. 查找普通 ``` 代码块
    if let Some(start) = text.find("```") {
        let json_start = start + 3;
        if let Some(end) = text[json_start..].find("```") {
            let content = text[json_start..json_start + end].trim();
            // 验证是否是 JSON
            if content.starts_with('{') || content.starts_with('[') {
                return Ok(content.to_owned());
            }
        }
    }

    // 3. 查找 JSON 对象（以 { 开始）
    if let Some(brace_start) = text.find('{') {
        let mut brace_count = 0;
        let mut end = brace_start;

        for (i, c) in text[brace_start..].char_indices() {
            match c {
                '{' => brace_count += 1,
                '}' => {
                    brace_count -= 1;
                    if brace_count == 0 {
                        end = brace_start + i + 1;
                        break;
                    }
                }
                _ => {}
            }
        }

        if brace_count == 0 {
            return Ok(text[brace_start..end].to_owned());
        }
    }

    // 4. 查找 JSON 数组（以 [ 开始）
    if let Some(bracket_start) = text.find('[') {
        let mut bracket_count = 0;
        let mut end = bracket_start;

        for (i, c) in text[bracket_start..].char_indices() {
            match c {
                '[' => bracket_count += 1,
                ']' => {
                    bracket_count -= 1;
                    if bracket_count == 0 {
                        end = bracket_start + i + 1;
                        break;
                    }
                }
                _ => {}
            }
        }

        if bracket_count == 0 {
            return Ok(text[bracket_start..end].to_owned());
        }
    }

    Err(ParseError::PatternNotFound("No JSON found".to_owned()))
}

/// 列表解析器
pub struct ListParser {
    separator: String,
}

impl ListParser {
    pub fn new(separator: &str) -> Self {
        Self {
            separator: separator.to_owned(),
        }
    }

    /// 创建逗号分隔的列表解析器
    pub fn comma_separated() -> Self {
        Self::new(",")
    }

    /// 创建换行分隔的列表解析器
    pub fn newline_separated() -> Self {
        Self::new("\n")
    }
}

impl OutputParser<Vec<String>> for ListParser {
    fn parse(&self, text: &str) -> Result<Vec<String>, ParseError> {
        let items: Vec<String> = text
            .split(&self.separator)
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty())
            .collect();

        if items.is_empty() {
            return Err(ParseError::EmptyOutput);
        }

        Ok(items)
    }

    fn get_format_instructions(&self) -> String {
        format!(
            "Output a list of items separated by '{}'. Each item should be on its own line if using newline separator.",
            self.separator
        )
    }
}

/// 键值对解析器
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyValue {
    pub key: String,
    pub value: String,
}

pub struct KeyValueParser {
    pair_separator: String,
    kv_separator: String,
}

impl KeyValueParser {
    pub fn new(pair_separator: &str, kv_separator: &str) -> Self {
        Self {
            pair_separator: pair_separator.to_owned(),
            kv_separator: kv_separator.to_owned(),
        }
    }

    /// 创建标准的 "key: value" 解析器（每行一个键值对）
    pub fn standard() -> Self {
        Self::new("\n", ":")
    }

    /// 创建逗号分隔的 "key=value" 解析器
    pub fn csv_style() -> Self {
        Self::new(",", "=")
    }
}

impl OutputParser<Vec<KeyValue>> for KeyValueParser {
    fn parse(&self, text: &str) -> Result<Vec<KeyValue>, ParseError> {
        let pairs: Vec<KeyValue> = text
            .split(&self.pair_separator)
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .filter_map(|line| {
                let parts: Vec<&str> = line.splitn(2, &self.kv_separator).collect();
                if parts.len() == 2 {
                    Some(KeyValue {
                        key: parts[0].trim().to_owned(),
                        value: parts[1].trim().to_owned(),
                    })
                } else {
                    None
                }
            })
            .collect();

        if pairs.is_empty() {
            return Err(ParseError::EmptyOutput);
        }

        Ok(pairs)
    }

    fn get_format_instructions(&self) -> String {
        format!(
            "Output key-value pairs with '{}' separating pairs and '{}' separating keys from values.",
            self.pair_separator, self.kv_separator
        )
    }
}

/// 正则表达式解析器
#[cfg(feature = "regex")]
pub struct RegexParser {
    pattern: regex::Regex,
    capture_group: usize,
}

#[cfg(feature = "regex")]
impl RegexParser {
    pub fn new(pattern: &str) -> Result<Self, regex::Error> {
        Ok(Self {
            pattern: regex::Regex::new(pattern)?,
            capture_group: 0,
        })
    }

    pub fn with_capture_group(mut self, group: usize) -> Self {
        self.capture_group = group;
        self
    }
}

#[cfg(feature = "regex")]
impl OutputParser<String> for RegexParser {
    fn parse(&self, text: &str) -> Result<String, ParseError> {
        let matches = self
            .pattern
            .captures(text)
            .ok_or_else(|| ParseError::PatternNotFound("No matches".to_owned()))?;

        matches
            .get(self.capture_group)
            .map(|m| m.as_str().to_owned())
            .ok_or_else(|| ParseError::PatternNotFound("Capture group empty".to_owned()))
    }

    fn get_format_instructions(&self) -> String {
        format!("Output must match the pattern: {}", self.pattern.as_str())
    }
}

/// 组合解析器 - 依次尝试多个解析器
pub struct OrParser<T> {
    parsers: Vec<Box<dyn OutputParser<T>>>,
}

impl<T> OrParser<T> {
    pub fn new() -> Self {
        Self {
            parsers: Vec::new(),
        }
    }

    pub fn add_parser(mut self, parser: impl OutputParser<T> + 'static) -> Self {
        self.parsers.push(Box::new(parser));
        self
    }
}

impl<T> Default for OrParser<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Send + Sync> OutputParser<T> for OrParser<T> {
    fn parse(&self, text: &str) -> Result<T, ParseError> {
        let mut last_error = ParseError::PatternNotFound("No parsers".to_owned());

        for parser in &self.parsers {
            match parser.parse(text) {
                Ok(result) => return Ok(result),
                Err(e) => last_error = e,
            }
        }

        Err(last_error)
    }

    fn get_format_instructions(&self) -> String {
        let instructions: Vec<String> = self
            .parsers
            .iter()
            .map(|p| p.get_format_instructions())
            .collect();

        instructions.join("\nOR\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Deserialize, PartialEq)]
    struct TestData {
        name: String,
        value: i32,
    }

    #[test]
    fn test_json_parser_with_code_block() {
        let parser = JsonParser::<TestData>::new();
        let text = r#"
Here's the data:
```json
{
  "name": "test",
  "value": 42
}
```
"#;

        let result = parser.parse(text).unwrap();
        assert_eq!(result.name, "test");
        assert_eq!(result.value, 42);
    }

    #[test]
    fn test_json_parser_without_code_block() {
        let parser = JsonParser::<TestData>::new();
        let text = r#"{"name": "test", "value": 42}"#;

        let result = parser.parse(text).unwrap();
        assert_eq!(result.name, "test");
        assert_eq!(result.value, 42);
    }

    #[test]
    fn test_list_parser_comma() {
        let parser = ListParser::comma_separated();
        let text = "apple, banana, cherry";

        let result = parser.parse(text).unwrap();
        assert_eq!(result, vec!["apple", "banana", "cherry"]);
    }

    #[test]
    fn test_list_parser_newline() {
        let parser = ListParser::newline_separated();
        let text = "apple\nbanana\ncherry";

        let result = parser.parse(text).unwrap();
        assert_eq!(result, vec!["apple", "banana", "cherry"]);
    }

    #[test]
    fn test_key_value_parser() {
        let parser = KeyValueParser::standard();
        let text = "name: John\nage: 30\ncity: New York";

        let result = parser.parse(text).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].key, "name");
        assert_eq!(result[0].value, "John");
    }

    #[test]
    fn test_key_value_parser_csv() {
        let parser = KeyValueParser::csv_style();
        let text = "name=John,age=30,city=New York";

        let result = parser.parse(text).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].key, "name");
        assert_eq!(result[0].value, "John");
    }

    #[test]
    fn test_or_parser_with_list() {
        // 测试 OrParser 尝试不同的列表解析策略
        // 注意：OrParser 会使用第一个成功的解析器

        // 只有逗号的文本 - 第一个 parser 成功
        let parser1 = OrParser::new()
            .add_parser(ListParser::comma_separated())
            .add_parser(ListParser::newline_separated());

        let comma_text = "apple, banana, cherry";
        let result: Result<Vec<String>, _> = parser1.parse(comma_text);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vec!["apple", "banana", "cherry"]);

        // 只有换行的文本 - 第一个 parser 会把整个字符串当作一个元素
        // 为了正确处理，我们需要先尝试 newline parser
        let parser2 = OrParser::new()
            .add_parser(ListParser::newline_separated())
            .add_parser(ListParser::comma_separated());

        let newline_text = "apple\nbanana\ncherry";
        let result: Result<Vec<String>, _> = parser2.parse(newline_text);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vec!["apple", "banana", "cherry"]);
    }

    #[test]
    fn test_json_extract_from_code_block() {
        let json_str = extract_json(
            r#"
Here's the result:
```json
{
  "name": "test",
  "value": 42
}
```
Thanks!
"#,
        )
        .unwrap();

        let parsed: TestData = serde_json::from_str(&json_str).unwrap();
        assert_eq!(parsed.name, "test");
        assert_eq!(parsed.value, 42);
    }

    #[test]
    fn test_json_extract_plain() {
        let json_str = extract_json(r#"{"name": "test", "value": 42}"#).unwrap();

        let parsed: TestData = serde_json::from_str(&json_str).unwrap();
        assert_eq!(parsed.name, "test");
        assert_eq!(parsed.value, 42);
    }
}
