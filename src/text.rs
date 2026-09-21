use regex::Regex;
use std::collections::HashSet;
use std::sync::LazyLock;
use sha2::{Digest, Sha256};

pub fn normalize_text(text: &str) -> String {
    text.trim().chars().flat_map(|ch| {
        let c = match ch {
            '\u{3000}' => ' ',
            '\u{ff01}'..='\u{ff5e}' => char::from_u32(ch as u32 - 0xfee0).unwrap_or(ch),
            '\u{2018}' | '\u{2019}' => '\'',
            '\u{201c}' | '\u{201d}' => '"',
            _ => ch,
        };
        c.to_lowercase()
    }).collect::<String>()
}

pub fn is_cjk(ch: char) -> bool {
    matches!(ch as u32, 0x3400..=0x4dbf | 0x4e00..=0x9fff | 0xf900..=0xfaff | 0x20000..=0x323af)
}

/// Keep document token frequencies; query callers can deduplicate separately.
pub fn tokenize(text: &str) -> Vec<String> {
    let normalized = normalize_text(text);
    let mut tokens = Vec::new();
    let mut word = String::new();
    let mut run = Vec::new();
    fn flush(word: &mut String, run: &mut Vec<char>, tokens: &mut Vec<String>) {
        if !word.is_empty() { tokens.push(std::mem::take(word)); }
        tokens.extend(run.iter().map(char::to_string));
        tokens.extend(run.windows(2).map(|p| format!("{}{}", p[0], p[1])));
        run.clear();
    }
    for ch in normalized.chars() {
        if is_cjk(ch) {
            if !word.is_empty() { tokens.push(std::mem::take(&mut word)); }
            run.push(ch);
        } else if ch.is_alphanumeric() || ch == '_' {
            if !run.is_empty() { flush(&mut word, &mut run, &mut tokens); }
            word.push(ch);
        } else { flush(&mut word, &mut run, &mut tokens); }
    }
    flush(&mut word, &mut run, &mut tokens);
    tokens
}

pub(crate) fn query_terms(text: &str, strict: bool) -> Vec<String> {
    let tokens = tokenize(text);
    let mut seen = HashSet::new();
    tokens.into_iter().filter(|t| {
        // Retaining unigrams also preserves standalone characters in mixed
        // queries such as "上海 茶". Bigrams enforce adjacency in strict mode.
        (strict || !(t.chars().count() == 2 && t.chars().all(is_cjk))) && seen.insert(t.clone())
    }).collect()
}

/// token 计数的字符密度，单位 1/100 token：ASCII ≈ 4 字符/token，非 ASCII（CJK 等）≈ 2 字符/token。
/// 取值照抄天使之魂在大样本上校准过的 0.25 / 0.48，不另行估计。
const ASCII_UNITS: u64 = 25;
const NON_ASCII_UNITS: u64 = 48;

/// 估算文本的 token 数：按字符密度累加、向上取整。
/// 不做分词、不调模型，成本是一次字符遍历。
pub(crate) fn count_tokens(text: &str) -> usize {
    let units: u64 = text.chars().map(|ch| if (ch as u32) < 128 { ASCII_UNITS } else { NON_ASCII_UNITS }).sum();
    units.div_ceil(100) as usize
}

/// 按 token 预算把文本截断到不超过 `budget` 个 token，口径与 `count_tokens` 同源。
/// 返回原文前缀，不改写内容。
pub(crate) fn truncate_to_tokens(text: &str, budget: usize) -> String {
    if budget == 0 { return String::new(); }
    let limit = budget as u64 * 100;
    let mut units: u64 = 0;
    let mut end = 0usize;
    for (index, ch) in text.char_indices() {
        let cost = if (ch as u32) < 128 { ASCII_UNITS } else { NON_ASCII_UNITS };
        if units + cost > limit { break; }
        units += cost;
        end = index + ch.len_utf8();
    }
    text[..end].to_string()
}

pub(crate) fn digest(text: &str) -> String { format!("{:x}", Sha256::digest(text.as_bytes())) }pub(crate) fn normalized_tag(text: &str) -> String {
    normalize_text(text).split_whitespace().collect::<Vec<_>>().join(" ")
}

// ── 统一 Markdown 清洗 ────────────────────────────────────────────────
//
// 规则：去掉标记、只留可读文本。切片与正文
// 照旧保留原文，清洗只作用于送进全文索引的文本，检索命中因此不被标记符干扰。

fn pattern(source: &str) -> Regex { Regex::new(source).expect("static markdown pattern compiles") }

static MD_HTML: LazyLock<Regex> = LazyLock::new(|| pattern(r"<[^>]+>"));
static MD_HEADING: LazyLock<Regex> = LazyLock::new(|| pattern(r"#+\s?"));
static MD_BOLD: LazyLock<Regex> = LazyLock::new(|| pattern(r"(\*\*|__)(.*?)(\*\*|__)"));
static MD_ITALIC: LazyLock<Regex> = LazyLock::new(|| pattern(r"(\*|_)(.*?)(\*|_)"));
static MD_LINK: LazyLock<Regex> = LazyLock::new(|| pattern(r"\[(.*?)\]\(.*?\)"));
static MD_IMAGE: LazyLock<Regex> = LazyLock::new(|| pattern(r"!\[.*?\]\(.*?\)"));
static MD_FENCE: LazyLock<Regex> = LazyLock::new(|| pattern(r"(?s)```.*?```"));
static MD_CODE: LazyLock<Regex> = LazyLock::new(|| pattern(r"`([^`]+)`"));
static MD_BULLET: LazyLock<Regex> = LazyLock::new(|| pattern(r"(?m)^[-*+]\s+"));
static MD_ORDERED: LazyLock<Regex> = LazyLock::new(|| pattern(r"(?m)^\d+\.\s+"));
static MD_QUOTE: LazyLock<Regex> = LazyLock::new(|| pattern(r"(?m)^>\s+"));
static MD_RULE: LazyLock<Regex> = LazyLock::new(|| pattern(r"---+"));
static MD_PIPE: LazyLock<Regex> = LazyLock::new(|| pattern(r"\|"));

/// 把 Markdown 清洗成纯文本。输入输出都是原文以外的
/// 派生文本，调用方负责决定拿它做什么（库只拿它喂索引分词）。
pub fn clean_markdown(input: &str) -> String {
    let text = input.to_string();
    let text = MD_HTML.replace_all(&text, "");
    let text = MD_HEADING.replace_all(&text, "");
    let text = MD_BOLD.replace_all(&text, "$2");
    let text = MD_ITALIC.replace_all(&text, "$2");
    let text = MD_LINK.replace_all(&text, "$1");
    let text = MD_IMAGE.replace_all(&text, "");
    let text = MD_FENCE.replace_all(&text, "");
    let text = MD_CODE.replace_all(&text, "$1");
    let text = MD_BULLET.replace_all(&text, "");
    let text = MD_ORDERED.replace_all(&text, "");
    let text = MD_QUOTE.replace_all(&text, "");
    let text = MD_RULE.replace_all(&text, "");
    let text = MD_PIPE.replace_all(&text, " ");
    text.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::clean_markdown;

    /// 期望值逐条对拍得到。
    #[test]
    fn clean_markdown_matches_reference() {
        let cases = [
            ("# 标题\n\n正文 **粗体** 与 *斜体* 和 _下划线_", "标题\n\n正文 粗体 与 斜体 和 下划线"),
            ("见 [链接](http://a.b) 与 ![图片](http://c.d)", "见 链接 与 !图片"),
            ("```py\nprint(1)\n```\n后面 `行内` 文字", "后面 行内 文字"),
            ("- 项目一\n- 项目二\n1. 有序\n> 引用\n\n---", "项目一\n项目二\n有序\n引用"),
            ("| 列A | 列B |\n|---|---|\n| 1 | 2 |", "列A   列B  \n   \n  1   2"),
            ("<div>标签</div> 普通文本", "标签 普通文本"),
            ("混合 **粗** [链](u) `码` 尾", "混合 粗 链 码 尾"),
            ("  ## 缩进标题  ", "缩进标题"),
        ];
        for (input, expected) in cases {
            assert_eq!(clean_markdown(input), expected, "input={input:?}");
        }
    }
}
