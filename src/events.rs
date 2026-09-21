//! 库内事件流：把「这次执行发生了什么、各阶段花了多久」交给宿主。
//!
//! 库只产出事件，不决定去向——落盘、轮转、保留多久都是宿主日志体系的事。
//! 未注册 sink 时全程不构造事件、不格式化，开销为零。

use crate::types::Degrade;
use chrono::{Local, SecondsFormat};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

/// 事件接收回调。库在检索线程里同步调用它，因此它必须非阻塞：
/// 在 sink 里做同步 IO 或网络上报，会把检索拖住，和慢的重排回调一样。
pub type EventSink = Arc<dyn Fn(&LogEvent) + Send + Sync>;

/// 一条事件。`kind` 决定哪些字段有值，无值的字段不出现。
///
/// 事件里不含查询原文与正文：宿主本来就知道查询词，库再抄一遍只是把检索词
/// 散进宿主的日志文件。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEvent {
    /// 本地时间的 RFC3339 时间戳，毫秒精度。
    pub ts: String,
    /// 事件类型：`search`、`index_rebuild`。
    pub kind: String,
    /// 本次执行的总耗时（毫秒）。
    pub ms: u64,
    /// 各阶段耗时（毫秒）。`rerank` 是「等宿主重排回调返回」的时间，也就是模型
    /// 推理时间，不是库的开销；`embed` 同理。
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub stages: BTreeMap<String, u64>,
    /// 参与融合的候选条数。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidates: Option<usize>,
    /// 折叠后剩下的条数。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub folded: Option<usize>,
    /// 实际送进重排回调的文档数。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rerank_docs: Option<usize>,
    /// 实际送进重排回调的文档 token 数（含查询词）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rerank_tokens: Option<usize>,
    /// 最终返回的命中条数。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hits: Option<usize>,
    /// 重建写入索引的文档数。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub documents: Option<usize>,
    /// 索引格式串。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    /// 本次落在哪几档降级。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub degraded: Vec<Degrade>,
}

impl LogEvent {
    /// 新事件，时间戳取当前本地时间。
    pub fn new(kind: &str) -> Self {
        Self {
            ts: Local::now().to_rfc3339_opts(SecondsFormat::Millis, false),
            kind: kind.to_string(),
            ms: 0,
            stages: BTreeMap::new(),
            candidates: None,
            folded: None,
            rerank_docs: None,
            rerank_tokens: None,
            hits: None,
            documents: None,
            format: None,
            degraded: Vec::new(),
        }
    }
}

/// 进程内的事件接收位。检索线程取出 `Arc` 后立刻放掉这把锁。
#[derive(Default)]
pub(crate) struct EventRegistry {
    sink: Mutex<Option<EventSink>>,
}

impl EventRegistry {
    pub fn get(&self) -> Option<EventSink> {
        self.sink.lock().clone()
    }
    pub fn set(&self, sink: EventSink) {
        *self.sink.lock() = Some(sink);
    }
    pub fn clear(&self) -> bool {
        self.sink.lock().take().is_some()
    }
    pub fn is_registered(&self) -> bool {
        self.sink.lock().is_some()
    }
}

/// 阶段计时：每 `mark` 一次，记下距上一次 `mark` 的毫秒数。
pub(crate) struct StageTimer {
    last: Instant,
    marks: BTreeMap<String, u64>,
}

impl StageTimer {
    pub fn start() -> Self {
        Self { last: Instant::now(), marks: BTreeMap::new() }
    }
    pub fn mark(&mut self, name: &str) {
        let now = Instant::now();
        self.marks.insert(name.to_string(), (now - self.last).as_millis() as u64);
        self.last = now;
    }
    /// 收尾：把最后一段也记进去，返回整张表。
    pub fn finish(mut self, name: &str) -> BTreeMap<String, u64> {
        self.mark(name);
        self.marks
    }
}
