use crate::{storage::{self, KnowledgeBase}, text, types::*, Error, Result};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::BTreeSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MemoryState {
    pub pinned: bool,
    pub strength: i64,
    pub useful_count: i64,
    pub useful_score: f64,
    pub last_recalled_at_us: Option<i64>,
    pub last_decay_at_us: Option<i64>,
}
impl Default for MemoryState {
    fn default() -> Self { Self { pinned: false, strength: 1, useful_count: 0, useful_score: 0.0, last_recalled_at_us: None, last_decay_at_us: None } }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    #[serde(flatten)] pub header: RecordHeader,
    pub memory_type: String,
    pub judgment: String,
    pub reasoning: String,
    pub state: MemoryState,
}

fn knowledge() -> String { "knowledge".into() }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInput {
    #[serde(flatten)] pub record: RecordInput,
    #[serde(default = "knowledge")] pub memory_type: String,
    pub judgment: String,
    #[serde(default)] pub reasoning: String,
    /// None preserves an existing memory's lifecycle state.
    #[serde(default)] pub state: Option<MemoryState>,
}
impl MemoryInput {
    pub fn new(judgment: impl Into<String>) -> Self {
        Self { record: RecordInput::default(), memory_type: knowledge(), judgment: judgment.into(), reasoning: String::new(), state: None }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DecayPolicy {
    pub tier0_threshold: f64,
    pub tier1_threshold: f64,
    pub useful_score_boost: f64,
    pub strength_boost: i64,
    pub tier0_cycle_days: u32,
}
impl Default for DecayPolicy {
    fn default() -> Self { Self { tier0_threshold: 3.0, tier1_threshold: 10.0, useful_score_boost: 2.5, strength_boost: 1, tier0_cycle_days: 3 } }
}
impl DecayPolicy {
    fn validate(&self) -> Result<()> {
        if !self.tier0_threshold.is_finite() || !self.tier1_threshold.is_finite() || !self.useful_score_boost.is_finite()
            || self.tier0_threshold <= 0.0 || self.tier1_threshold <= self.tier0_threshold
            || self.useful_score_boost <= 0.0 || self.strength_boost < 1 || self.tier0_cycle_days == 0 {
            return Err(Error::Validation("invalid decay policy".into()));
        }
        Ok(())
    }
    pub fn tier(&self, score: f64) -> u8 {
        if score >= self.tier1_threshold { 2 } else if score >= self.tier0_threshold { 1 } else { 0 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FeedbackRequest {
    #[serde(default)] pub filter: ReadFilter,
    #[serde(default)] pub recalled_ids: Vec<i64>,
    #[serde(default)] pub useful_ids: Vec<i64>,
    #[serde(default)] pub now_us: Option<i64>,
    #[serde(default)] pub policy: DecayPolicy,
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FeedbackReport { pub recalled: usize, pub boosted: usize, pub penalized: usize }
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DecayReport { pub decayed: usize, pub retirement_candidates: Vec<RecordKey> }

#[derive(Clone)]
pub struct MemoryStore(pub(crate) KnowledgeBase);

pub(crate) fn upsert(conn: &Connection, input: &MemoryInput) -> Result<(Memory, crate::index::IndexDocument)> {
    storage::validate_identity("memory_type", &input.memory_type)?;
    if input.judgment.trim().is_empty() { return Err(Error::Validation("judgment is required".into())); }
    let existing: Option<Memory> = if let Some(id) = input.record.id {
        storage::record_value(conn, &RecordKey { id })?.map(serde_json::from_value).transpose()?
    } else { None };
    let state = input.state.clone().or_else(|| existing.map(|m| m.state)).unwrap_or_default();
    if state.strength < 0 || state.useful_count < 0 || !state.useful_score.is_finite() || state.useful_score < 0.0 {
        return Err(Error::Validation("memory lifecycle values must be finite and nonnegative".into()));
    }
    let judgment = input.judgment.trim().to_string();
    let reasoning = input.reasoning.trim().to_string();
    let memory_type_id = storage::term_id(conn, &input.memory_type)?;
    let payload = json!({"memory_type_id": memory_type_id, "judgment": judgment, "reasoning": reasoning,
        "state": state, "judgment_key": text::normalized_tag(&judgment)});
    let (header, document) = storage::put_record(conn, RecordKind::Memory, &input.record, &payload, &judgment, "")?;
    Ok((Memory { header, memory_type: input.memory_type.clone(), judgment, reasoning, state }, document))
}

pub(crate) fn as_input(memory: &Memory, at: i64) -> MemoryInput {
    MemoryInput {
        record: RecordInput { id: Some(memory.header.id), namespace: memory.header.namespace.clone(),
            scope: memory.header.scope.clone(), tags: memory.header.tags.clone(), evidence: memory.header.evidence.clone(),
            metadata: memory.header.metadata.clone(), created_at_us: Some(memory.header.created_at_us),
            updated_at_us: Some(at.max(memory.header.updated_at_us)), expected_revision: Some(memory.header.revision) },
        memory_type: memory.memory_type.clone(), judgment: memory.judgment.clone(), reasoning: memory.reasoning.clone(), state: Some(memory.state.clone()),
    }
}

impl MemoryStore {
    pub fn upsert(&self, input: MemoryInput) -> Result<WriteReceipt<Memory>> {
        let receipt = self.0.mutate(|tx| upsert(tx, &input))?;
        let WriteReceipt { value: (memory, document), revision } = receipt;
        // 索引文档由写入流程就地交过来，这里只写进 writer，提交交给 update_index。
        self.0.index_documents(&[document])?;
        // 写入即向量化：库内部补齐，宿主只给正文。
        self.0.vectorize(&[memory.header.id]);
        Ok(WriteReceipt { value: memory, revision })
    }
    pub fn upsert_many(&self, inputs: &[MemoryInput]) -> Result<WriteReceipt<Vec<Memory>>> {
        let receipt = self.0.mutate(|tx| inputs.iter().map(|input| upsert(tx, input)).collect::<Result<Vec<_>>>())?;
        let WriteReceipt { value, revision } = receipt;
        let (memories, documents): (Vec<Memory>, Vec<crate::index::IndexDocument>) = value.into_iter().unzip();
        self.0.index_documents(&documents)?;
        let ids: Vec<i64> = memories.iter().map(|memory| memory.header.id).collect();
        self.0.vectorize(&ids);
        Ok(WriteReceipt { value: memories, revision })
    }
    pub fn upsert_by_judgment(&self, mut input: MemoryInput) -> Result<WriteReceipt<Memory>> {
        let receipt = self.0.mutate(|tx| {
            let mut stmt = tx.prepare("SELECT id FROM records WHERE namespace_id=(SELECT id FROM strings WHERE text=?1)
                AND scope_id=(SELECT id FROM strings WHERE text=?2) AND kind=?3 AND json_extract(payload_json,'$.judgment_key')=?4 ORDER BY id LIMIT 2")?;
            let ids = stmt.query_map(params![text::normalized_tag(&input.record.namespace), text::normalized_tag(&input.record.scope),
                RecordKind::Memory.code(), text::normalized_tag(&input.judgment)], |r| r.get::<_, i64>(0))?.collect::<std::result::Result<Vec<_>, _>>()?;
            if ids.len() > 1 { return Err(Error::Conflict("multiple memories have this judgment; update by ID".into())); }
            if let Some(id) = ids.first() {
                if input.record.id.is_some_and(|given| given != *id) { return Err(Error::Conflict("judgment belongs to a different ID".into())); }
                input.record.id = Some(*id);
                let key = RecordKey { id: *id };
                let existing: Memory = serde_json::from_value(storage::record_value(tx, &key)?.ok_or_else(|| Error::NotFound(id.to_string()))?)?;
                let mut metadata = existing.header.metadata;
                metadata.extend(input.record.metadata.clone());
                input.record.metadata = metadata;
                if input.record.evidence.is_empty() { input.record.evidence = existing.header.evidence; }
            }
            upsert(tx, &input)
        })?;
        let WriteReceipt { value: (memory, document), revision } = receipt;
        self.0.index_documents(&[document])?;
        self.0.vectorize(&[memory.header.id]);
        Ok(WriteReceipt { value: memory, revision })
    }
    pub fn get(&self, id: i64, filter: &ReadFilter) -> Result<Memory> {
        let state = self.0.read()?;
        storage::get(state.conn(), &RecordKey { id }, filter)
    }
    pub fn list(&self, request: &PageRequest) -> Result<Page<Memory>> {
        storage::list(self.0.read()?.conn(), RecordKind::Memory, request)
    }
    pub fn delete(&self, id: i64, filter: &ReadFilter) -> Result<WriteReceipt<bool>> {
        self.0.mutate(|tx| {
            let key = RecordKey { id };
            if !storage::matches_filter(tx, &key, filter)? { return Err(Error::NotFound(id.to_string())); }
            storage::delete_record(tx, &key)
        })
    }
    pub fn feedback(&self, request: &FeedbackRequest) -> Result<WriteReceipt<FeedbackReport>> {
        request.policy.validate()?;
        let recalled: BTreeSet<_> = request.recalled_ids.iter().copied().collect();
        let useful: BTreeSet<_> = request.useful_ids.iter().copied().collect();
        if !useful.is_subset(&recalled) { return Err(Error::Validation("useful_ids must be a subset of recalled_ids".into())); }
        let receipt = self.0.mutate(|tx| {
            let mut report = FeedbackReport { recalled: recalled.len(), ..Default::default() };
            let mut documents = Vec::new();
            let at = request.now_us.unwrap_or_else(storage::now_us);
            for id in &recalled {
                let key = RecordKey { id: *id };
                let mut memory: Memory = storage::get(tx, &key, &request.filter)?;
                if useful.contains(id) {
                    memory.state.strength = memory.state.strength.checked_add(request.policy.strength_boost).ok_or_else(|| Error::Validation("strength overflow".into()))?;
                    memory.state.useful_count = memory.state.useful_count.checked_add(1).ok_or_else(|| Error::Validation("useful_count overflow".into()))?;
                    memory.state.useful_score += request.policy.useful_score_boost;
                    memory.state.last_recalled_at_us = Some(at);
                    report.boosted += 1;
                } else if !memory.state.pinned && request.policy.tier(memory.state.useful_score) == 1 {
                    memory.state.strength = (memory.state.strength - 1).max(0);
                    report.penalized += 1;
                } else { continue; }
                let (_, document) = upsert(tx, &as_input(&memory, at))?;
                documents.push(document);
            }
            Ok((report, documents))
        })?;
        let WriteReceipt { value: (report, documents), revision } = receipt;
        self.0.index_documents(&documents)?;
        Ok(WriteReceipt { value: report, revision })
    }
    pub fn decay(&self, filter: &ReadFilter, policy: &DecayPolicy, at: Option<i64>) -> Result<WriteReceipt<DecayReport>> {
        policy.validate()?;
        let receipt = self.0.mutate(|tx| {
            let at = at.unwrap_or_else(storage::now_us);
            let keys = storage::select_keys(tx, filter, &[RecordKind::Memory], usize::MAX, None)?;
            let cycle = i128::from(policy.tier0_cycle_days) * 86_400_000_000;
            let mut documents = Vec::new();
            let mut report = DecayReport::default();
            for key in keys {
                let mut memory: Memory = storage::get(tx, &key, filter)?;
                if memory.state.pinned { continue; }
                if policy.tier(memory.state.useful_score) == 0 && memory.state.strength > 0 {
                    let reference = memory.header.created_at_us.max(memory.state.last_recalled_at_us.unwrap_or(i64::MIN)).max(memory.state.last_decay_at_us.unwrap_or(i64::MIN));
                    let steps = (i128::from(at) - i128::from(reference)) / cycle;
                    if steps > 0 {
                        memory.state.strength = (i128::from(memory.state.strength) - steps).max(0) as i64;
                        memory.state.last_decay_at_us = Some((i128::from(reference) + steps * cycle) as i64);
                        let (_, document) = upsert(tx, &as_input(&memory, at))?;
                        documents.push(document);
                        report.decayed += 1;
                    }
                }
                if memory.state.strength == 0 && policy.tier(memory.state.useful_score) < 2 { report.retirement_candidates.push(key); }
            }
            Ok((report, documents))
        })?;
        let WriteReceipt { value: (report, documents), revision } = receipt;
        self.0.index_documents(&documents)?;
        Ok(WriteReceipt { value: report, revision })
    }
}
