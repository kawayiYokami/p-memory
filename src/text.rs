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

/// 从宿主传入的 source 路径里取出每一级父目录名，作为精确整词关键字。
/// 只读传入的字符串、不碰文件系统；分隔符兼容 `/` 与 `\`，末段视为叶子（文件名/资源）丢弃。
pub(crate) fn ancestor_dirs(source: &str) -> Vec<String> {
    let normalized = normalize_text(source);
    let parts: Vec<&str> = normalized.split(['/', '\\']).filter(|p| !p.is_empty()).collect();
    if parts.len() < 2 { return Vec::new(); }
    let mut dirs: Vec<String> = parts[..parts.len() - 1].iter().map(|p| p.trim().to_string()).filter(|p| !p.is_empty()).collect();
    dirs.sort();
    dirs.dedup();
    dirs
}

/// 查询侧关键字词：不做 1+2 切分，按空白切成整词后归一化，用于命中精确整词字段。
pub(crate) fn keyword_terms(query: &str) -> Vec<String> {
    let mut seen = HashSet::new();
    normalize_text(query).split_whitespace().filter(|w| seen.insert(w.to_string())).map(|w| w.to_string()).collect()
}

/// 按与 `tokenize` 同源的规则，把文本截断到不超过 `budget` 个 token。
///
/// 计数方式是分词规则的直接映射：一个 CJK 字算 1 个 token，它与前一个字的
/// 相邻二元组再算 1 个；连续的字母数字串整体算 1 个。宿主不必自己数 token，
/// 也不必知道库用的是哪套切分。返回的是原文前缀，不改写内容。
pub(crate) fn truncate_to_tokens(text: &str, budget: usize) -> String {
    if budget == 0 { return String::new(); }
    let mut count = 0usize;
    let mut end = 0usize;
    let mut in_word = false;
    let mut run = 0usize;
    for (index, ch) in text.char_indices() {
        let mut added = 0usize;
        if is_cjk(ch) {
            if in_word { added += 1; in_word = false; }
            run += 1;
            added += 1;
            if run >= 2 { added += 1; }
        } else if ch.is_alphanumeric() || ch == '_' {
            if run > 0 { added += run + run.saturating_sub(1); run = 0; }
            in_word = true;
        } else {
            if in_word { added += 1; in_word = false; }
            if run > 0 { added += run + run.saturating_sub(1); run = 0; }
        }
        if count + added > budget { break; }
        count += added;
        end = index + ch.len_utf8();
    }
    text[..end].to_string()
}

pub(crate) fn digest(text: &str) -> String { format!("{:x}", Sha256::digest(text.as_bytes())) }pub(crate) fn normalized_tag(text: &str) -> String {
    normalize_text(text).split_whitespace().collect::<Vec<_>>().join(" ")
}

// ── 统一 Markdown 清洗 ────────────────────────────────────────────────
//
// 规则与 story 的 clean_markdown 一致：去掉标记、只留可读文本。切片与正文
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

/// 把 Markdown 清洗成纯文本，替换顺序与 story 一致。输入输出都是原文以外的
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

    /// 期望值与 story 的 `clean_markdown` 逐条对拍得到（见 .pai/temp/clean_compare.py）。
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
