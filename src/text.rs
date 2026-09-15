use std::collections::HashSet;
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
