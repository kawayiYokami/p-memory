use crate::{storage::{self, KnowledgeBase}, text, types::*, Error, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet, HashSet};

fn concept() -> String { "concept".into() }
fn confidence() -> f64 { 0.8 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityInput {
    #[serde(flatten)] pub record: RecordInput,
    pub name: String,
    #[serde(default = "concept")] pub entity_type: String,
    #[serde(default)] pub aliases: Vec<String>,
    #[serde(default)] pub attributes: BTreeMap<String, Vec<String>>,
    #[serde(default)] pub summary: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    #[serde(flatten)] pub header: RecordHeader,
    pub name: String, pub entity_type: String, pub aliases: Vec<String>,
    pub attributes: BTreeMap<String, Vec<String>>, pub summary: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationInput {
    #[serde(flatten)] pub record: RecordInput,
    pub subject_id: i64, pub predicate: String, pub object_id: i64,
    #[serde(default = "confidence")] pub confidence: f64,
    #[serde(default)] pub reason: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation {
    #[serde(flatten)] pub header: RecordHeader,
    pub subject_id: i64, pub predicate: String, pub object_id: i64,
    pub confidence: f64, pub reason: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventInput {
    #[serde(flatten)] pub record: RecordInput,
    pub name: String,
    #[serde(default)] pub summary: String,
    #[serde(default)] pub participants: Vec<i64>,
    #[serde(default = "confidence")] pub confidence: f64,
    #[serde(default)] pub reason: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    #[serde(flatten)] pub header: RecordHeader,
    pub name: String, pub summary: String, pub participants: Vec<i64>,
    pub confidence: f64, pub reason: String,
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GraphBatch {
    #[serde(default)] pub entities: Vec<EntityInput>,
    #[serde(default)] pub relations: Vec<RelationInput>,
    #[serde(default)] pub events: Vec<EventInput>,
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GraphBatchResult { pub entities: Vec<Entity>, pub relations: Vec<Relation>, pub events: Vec<Event> }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neighborhood { pub entities: Vec<Entity>, pub relations: Vec<Relation> }
#[derive(Clone)]
pub struct GraphStore(pub(crate) KnowledgeBase);

fn check_confidence(score: f64) -> Result<()> {
    if !score.is_finite() || !(0.0..=1.0).contains(&score) { return Err(Error::Validation("confidence must be in [0,1]".into())); }
    Ok(())
}

/// 单次扩散返回的词元上限：防止宽泛等价词灌入过多词元、稀释检索精度。
const EXPAND_QUERY_LIMIT: usize = 64;

/// 按文本查已登记的领域 id；没登记返回 None（只读接口不新建标记）。
fn namespace_term(conn: &Connection, namespace: &str) -> Result<Option<i64>> {
    Ok(conn.query_row("SELECT id FROM strings WHERE text=?1", [text::normalized_tag(namespace)], |r| r.get(0)).optional()?)
}

/// 扩散核心，直接在给定连接上做：GraphStore::expand_query 与全文检索路共用。
/// 先找出查询里出现的登记谓词，再取这些谓词所在等价组的全部同义词。
pub(crate) fn expand_query_conn(conn: &Connection, namespace: &str, text_value: &str) -> Result<Vec<String>> {
    let Some(namespace_id) = namespace_term(conn, namespace)? else { return Ok(Vec::new()); };
    let normalized = text::normalized_tag(text_value);
    let mut stmt = conn.prepare("SELECT s.text, pe.canonical_id FROM predicate_equivalents pe \
        JOIN strings s ON s.id=pe.predicate_id WHERE pe.namespace_id=?1")?;
    let mut canonical_ids: BTreeSet<i64> = BTreeSet::new();
    for row in stmt.query_map([namespace_id], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))? {
        let (predicate_text, canonical_id) = row?;
        let probe = text::normalized_tag(&predicate_text);
        if !probe.is_empty() && normalized.contains(&probe) { canonical_ids.insert(canonical_id); }
    }
    if canonical_ids.is_empty() { return Ok(Vec::new()); }
    let mut member_stmt = conn.prepare("SELECT s.text FROM predicate_equivalents pe JOIN strings s ON s.id=pe.predicate_id \
        WHERE pe.namespace_id=?1 AND pe.canonical_id=?2 ORDER BY s.text")?;
    let mut seen: HashSet<String> = HashSet::new();
    let mut expanded: Vec<String> = Vec::new();
    for canonical_id in &canonical_ids {
        for row in member_stmt.query_map(params![namespace_id, canonical_id], |r| r.get::<_, String>(0))? {
            let term = row?;
            if seen.insert(term.clone()) { expanded.push(term); }
            if expanded.len() >= EXPAND_QUERY_LIMIT { return Ok(expanded); }
        }
    }
    Ok(expanded)
}

/// 查询期扩散：把命中的同义词追加到查询词后面，让「老公」也能召回写「丈夫」的记录。
/// 领域没登记等价词、或查询里没出现登记词时，原样返回。查询词用空格连接追加，
/// 追加部分自成词元，不影响原有部分的分词结果。
pub(crate) fn match_predicate_synonyms(conn: &Connection, namespace: &str, query: &str) -> Result<String> {
    let extra = expand_query_conn(conn, namespace, query)?;
    if extra.is_empty() { return Ok(query.to_string()); }
    Ok(format!("{query} {}", extra.join(" ")))
}

fn referenced_entity(conn: &Connection, record: &RecordInput, id: i64) -> Result<Entity> {
    storage::get(conn, &RecordKey { id },
        &ReadFilter { namespace: record.namespace.clone(), scopes: vec![record.scope.clone()], tags: vec![] })
}

pub(crate) fn upsert_entity(conn: &Connection, input: &EntityInput) -> Result<(Entity, crate::index::IndexDocument)> {
    storage::validate_identity("entity name", &input.name)?;
    storage::validate_identity("entity_type", &input.entity_type)?;
    let aliases: Vec<_> = input.aliases.iter().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect::<BTreeSet<_>>().into_iter().collect();
    let attributes = input.attributes.clone();
    let attr_text = attributes.iter().map(|(k, values)| format!("{k} {}", values.join(" "))).collect::<Vec<_>>().join(" ");
    let body = format!("{} {} {}", aliases.join(" "), input.summary, attr_text);
    let (header, document) = storage::put_record(conn, RecordKind::Entity, &input.record,
        &json!({"name":input.name,"entity_type":input.entity_type,"aliases":aliases,"attributes":attributes,"summary":input.summary}), &body)?;
    let entity_type_id = storage::term_id(conn, &input.entity_type)?;
    conn.execute("INSERT INTO entities(record_id,name,entity_type_id) VALUES (?1,?2,?3)
        ON CONFLICT(record_id) DO UPDATE SET name=excluded.name,entity_type_id=excluded.entity_type_id",
        params![header.id, input.name, entity_type_id])?;
    conn.execute("DELETE FROM entity_aliases WHERE entity_id=?1", [header.id])?;
    for alias in std::iter::once(&input.name).chain(aliases.iter()) {
        let alias_id = storage::term_id(conn, alias)?;
        conn.execute("INSERT OR IGNORE INTO entity_aliases(entity_id,alias_id) VALUES (?1,?2)", params![header.id, alias_id])?;
    }
    conn.execute("DELETE FROM entity_attributes WHERE entity_id=?1", [header.id])?;
    for (key, values) in &attributes {
        let key_id = storage::term_id(conn, key)?;
        for value in values {
            conn.execute("INSERT OR IGNORE INTO entity_attributes(entity_id,attr_key_id,attr_value) VALUES (?1,?2,?3)", params![header.id, key_id, value])?;
        }
    }
    Ok((Entity { header, name: input.name.clone(), entity_type: input.entity_type.clone(), aliases, attributes, summary: input.summary.clone() }, document))
}

pub(crate) fn upsert_relation(conn: &Connection, input: &RelationInput) -> Result<(Relation, crate::index::IndexDocument)> {
    storage::validate_identity("predicate", &input.predicate)?;
    check_confidence(input.confidence)?;
    let subject = referenced_entity(conn, &input.record, input.subject_id)?;
    let object = referenced_entity(conn, &input.record, input.object_id)?;
    let body = format!("{} {} {} {}", subject.name, input.predicate, object.name, input.reason);
    let (header, document) = storage::put_record(conn, RecordKind::Relation, &input.record,
        &json!({"subject_id":input.subject_id,"predicate":input.predicate,"object_id":input.object_id,"confidence":input.confidence,"reason":input.reason,
            "subject_name":subject.name,"object_name":object.name}), &body)?;
    let predicate_id = storage::term_id(conn, &input.predicate)?;
    conn.execute("INSERT INTO relations(record_id,subject_id,predicate_id,object_id) VALUES (?1,?2,?3,?4)
        ON CONFLICT(record_id) DO UPDATE SET subject_id=excluded.subject_id,predicate_id=excluded.predicate_id,object_id=excluded.object_id",
        params![header.id, input.subject_id, predicate_id, input.object_id])?;
    Ok((Relation { header, subject_id: input.subject_id, predicate: input.predicate.clone(), object_id: input.object_id, confidence: input.confidence, reason: input.reason.clone() }, document))
}

pub(crate) fn upsert_event(conn: &Connection, input: &EventInput) -> Result<(Event, crate::index::IndexDocument)> {
    storage::validate_identity("event name", &input.name)?;
    check_confidence(input.confidence)?;
    let participants: Vec<_> = input.participants.iter().copied().collect::<BTreeSet<_>>().into_iter().collect();
    let names = participants.iter().map(|id| referenced_entity(conn, &input.record, *id).map(|e| e.name)).collect::<Result<Vec<_>>>()?;
    let body = format!("{} {} {} {}", input.name, input.summary, names.join(" "), input.reason);
    let (header, document) = storage::put_record(conn, RecordKind::Event, &input.record,
        &json!({"name":input.name,"summary":input.summary,"participants":participants,"confidence":input.confidence,"reason":input.reason,
            "participant_names":names}), &body)?;
    conn.execute("DELETE FROM event_participants WHERE event_id=?1", [header.id])?;
    for id in &participants {
        conn.execute("INSERT INTO event_participants(event_id,entity_id) VALUES (?1,?2)", params![header.id, id])?;
    }
    Ok((Event { header, name: input.name.clone(), summary: input.summary.clone(), participants, confidence: input.confidence, reason: input.reason.clone() }, document))
}

pub(crate) fn apply_batch(conn: &Connection, batch: &GraphBatch) -> Result<(GraphBatchResult, Vec<crate::index::IndexDocument>)> {
    let mut documents = Vec::new();
    // Entities first permits references to entities created in this transaction.
    let mut entities = Vec::new();
    for input in &batch.entities { let (entity, document) = upsert_entity(conn, input)?; entities.push(entity); documents.push(document); }
    let mut relations = Vec::new();
    for input in &batch.relations { let (relation, document) = upsert_relation(conn, input)?; relations.push(relation); documents.push(document); }
    let mut events = Vec::new();
    for input in &batch.events { let (event, document) = upsert_event(conn, input)?; events.push(event); documents.push(document); }
    // 改名会改写引用它的关系与事件正文，这些记录的索引与向量都要一并刷新。
    documents.extend(refresh_dependents(conn, &entities)?);
    Ok((GraphBatchResult { entities, relations, events }, documents))
}

/// 返回因引用正文变化而被重写的记录的索引文档。
fn refresh_dependents(conn: &Connection, entities: &[Entity]) -> Result<Vec<crate::index::IndexDocument>> {
    let mut keys = BTreeSet::new();
    let mut documents = Vec::new();
    for entity in entities {
        let mut stmt = conn.prepare("SELECT record_id FROM relations WHERE subject_id=?1 OR object_id=?1
            UNION SELECT event_id FROM event_participants WHERE entity_id=?1")?;
        for row in stmt.query_map([entity.header.id], |r| r.get::<_, i64>(0))? { keys.insert(RecordKey { id: row? }); }
    }
    for key in keys {
        let value = storage::record_value(conn, &key)?.ok_or_else(|| Error::NotFound(key.id.to_string()))?;
        let kind_code: i64 = conn.query_row("SELECT kind FROM records WHERE id=?1", [key.id], |r| r.get(0))?;
        let kind = RecordKind::from_code(kind_code).ok_or_else(|| Error::Validation("invalid stored record kind".into()))?;
        // 正文由 payload 字段现拼；实体改名时，把新名字写回 payload 的名称快照并推进指纹。
        let (body, patch) = if kind == RecordKind::Relation {
            let r: Relation = serde_json::from_value(value.clone())?;
            let input = r.header.as_input();
            let subject = referenced_entity(conn, &input, r.subject_id)?.name;
            let object = referenced_entity(conn, &input, r.object_id)?.name;
            (format!("{} {} {} {}", subject, r.predicate, object, r.reason), json!({"subject_name":subject,"object_name":object}))
        } else {
            let e: Event = serde_json::from_value(value.clone())?;
            let names = e.participants.iter().map(|id| referenced_entity(conn, &e.header.as_input(), *id).map(|v| v.name)).collect::<Result<Vec<_>>>()?;
            (format!("{} {} {} {}", e.name, e.summary, names.join(" "), e.reason), json!({"participant_names":names}))
        };
        let old = storage::record_text(kind, &value);
        if old != body {
            let raw: String = conn.query_row("SELECT payload_json FROM records WHERE id=?1", [key.id], |r| r.get(0))?;
            let mut payload: serde_json::Value = serde_json::from_str(&raw)?;
            if let Some(object) = payload.as_object_mut() {
                if let Some(extra) = patch.as_object() {
                    for (name, value) in extra { object.insert(name.clone(), value.clone()); }
                }
            }
            let tags: Vec<String> = value.get("tags").and_then(|v| v.as_array())
                .map(|list| list.iter().filter_map(|v| v.as_str()).map(str::to_string).collect()).unwrap_or_default();
            let fingerprint = storage::record_fingerprint(&body, &tags);
            let revision = storage::next_revision(conn, key.id)?;
            conn.execute("UPDATE records SET payload_json=?2,fingerprint=?3,revision=?4,updated_at_us=MAX(updated_at_us,?5) WHERE id=?1",
                params![key.id, serde_json::to_string(&payload)?, fingerprint, revision, storage::now_us()])?;
            conn.execute("DELETE FROM embeddings WHERE record_id=?1", [key.id])?;
            documents.push(storage::index_document(conn, key.id, kind, body.clone())?);
        }
    }
    Ok(documents)
}

impl GraphStore {
    pub fn apply_batch(&self, batch: &GraphBatch) -> Result<WriteReceipt<GraphBatchResult>> {
        let mut documents: Vec<crate::index::IndexDocument> = Vec::new();
        let receipt = self.0.mutate(|tx| {
            let (result, staged) = apply_batch(tx, batch)?;
            documents = staged;
            Ok(result)
        })?;
        self.0.index_documents(&documents)?;
        Ok(receipt)
    }
    pub fn get(&self, kind: RecordKind, id: i64, filter: &ReadFilter) -> Result<serde_json::Value> {
        if !matches!(kind, RecordKind::Entity | RecordKind::Relation | RecordKind::Event) { return Err(Error::Validation("expected a graph record kind".into())); }
        storage::get(self.0.read()?.conn(), &RecordKey { id }, filter)
    }
    pub fn list(&self, kind: RecordKind, page: &PageRequest) -> Result<Page<serde_json::Value>> {
        if !matches!(kind, RecordKind::Entity | RecordKind::Relation | RecordKind::Event) { return Err(Error::Validation("expected a graph record kind".into())); }
        storage::list(self.0.read()?.conn(), kind, page)
    }
    /// 登记一批谓词等价组到某个知识领域：`groups` 是一组组互等同义词，
    /// 组内第一个词当规范词（组内代表），同组词据此归并。重复登记同一个词会改写它的归属。
    /// 表由上游提供、库不内置领域数据；只影响查询期扩散，不改谓词的落盘写法。
    pub fn set_predicate_equivalents(&self, namespace: &str, groups: &[Vec<String>]) -> Result<WriteReceipt<usize>> {
        storage::validate_identity("namespace", namespace)?;
        for group in groups {
            if group.is_empty() { return Err(Error::Validation("an equivalent group must not be empty".into())); }
            for term in group { storage::validate_identity("predicate", term)?; }
        }
        self.0.mutate(|tx| {
            let namespace_id = storage::term_id(tx, namespace)?;
            let mut count = 0usize;
            for group in groups {
                let canonical_id = storage::term_id(tx, &group[0])?;
                for term in group {
                    let predicate_id = storage::term_id(tx, term)?;
                    tx.execute("INSERT INTO predicate_equivalents(namespace_id,predicate_id,canonical_id) VALUES (?1,?2,?3) \
                        ON CONFLICT(namespace_id,predicate_id) DO UPDATE SET canonical_id=excluded.canonical_id",
                        params![namespace_id, predicate_id, canonical_id])?;
                    count += 1;
                }
            }
            Ok(count)
        })
    }

    /// 列出某个知识领域已登记的等价组；每个组是一组互等同义词。没登记过就返回空。
    pub fn predicate_equivalents(&self, namespace: &str) -> Result<Vec<Vec<String>>> {
        storage::validate_identity("namespace", namespace)?;
        let state = self.0.read()?;
        let conn = state.conn();
        let Some(namespace_id) = namespace_term(conn, namespace)? else { return Ok(Vec::new()); };
        let mut stmt = conn.prepare("SELECT pe.canonical_id, s.text FROM predicate_equivalents pe \
            JOIN strings s ON s.id=pe.predicate_id WHERE pe.namespace_id=?1 ORDER BY pe.canonical_id, s.text")?;
        let mut groups: BTreeMap<i64, Vec<String>> = BTreeMap::new();
        for row in stmt.query_map([namespace_id], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))? {
            let (canonical_id, term) = row?;
            groups.entry(canonical_id).or_default().push(term);
        }
        Ok(groups.into_values().map(|mut group| { group.sort(); group.dedup(); group }).collect())
    }

    /// 扩散：找出 `text` 里出现了哪些已登记谓词，返回这些谓词所在等价组的全部同义词。
    /// 命中判定是「登记词作为子串出现在查询里」，与 story 的关键词表同一路数。
    /// 返回的是可直接追加进检索的补充词元；没有命中返回空。
    pub fn expand_query(&self, namespace: &str, text_value: &str) -> Result<Vec<String>> {
        storage::validate_identity("namespace", namespace)?;
        expand_query_conn(self.0.read()?.conn(), namespace, text_value)
    }

    pub fn resolve(&self, name: &str, filter: &ReadFilter, limit: usize) -> Result<Vec<Entity>> {
        storage::validate_filter(filter)?; storage::validate_limit(limit)?;
        let state = self.0.read()?;
        let conn = state.conn();
        let mut stmt = conn.prepare("SELECT DISTINCT entity_id FROM entity_aliases WHERE alias_id=(SELECT id FROM strings WHERE text=?1) ORDER BY entity_id")?;
        let mut result = Vec::new();
        for row in stmt.query_map([text::normalized_tag(name)], |r| r.get::<_, i64>(0))? {
            let key = RecordKey { id: row? };
            if storage::matches_filter(conn, &key, filter)? {
                result.push(storage::get(conn, &key, filter)?);
                if result.len() == limit { break; }
            }
        }
        Ok(result)
    }
    pub fn neighbors(&self, id: i64, filter: &ReadFilter, limit: usize) -> Result<Neighborhood> {
        storage::validate_limit(limit)?;
        let state = self.0.read()?;
        let conn = state.conn();
        let root = RecordKey { id };
        let _: Entity = storage::get(conn, &root, filter)?;
        let mut stmt = conn.prepare("SELECT record_id FROM relations WHERE subject_id=?1 OR object_id=?1 ORDER BY record_id")?;
        let mut entities = BTreeMap::new(); let mut relations = Vec::new();
        for row in stmt.query_map([id], |r| r.get::<_, i64>(0))? {
            let key = RecordKey { id: row? };
            if !storage::matches_filter(conn, &key, filter)? { continue; }
            let relation: Relation = storage::get(conn, &key, filter)?;
            let endpoint = if relation.subject_id == id { relation.object_id } else { relation.subject_id };
            let entity_key = RecordKey { id: endpoint };
            // Tags constrain returned relations, while scope constrains endpoints too.
            let entity_filter = ReadFilter { tags: vec![], ..filter.clone() };
            if !storage::matches_filter(conn, &entity_key, &entity_filter)? { continue; }
            entities.insert(endpoint, storage::get(conn, &entity_key, &entity_filter)?);
            relations.push(relation);
            if relations.len() == limit { break; }
        }
        Ok(Neighborhood { entities: entities.into_values().collect(), relations })
    }
    pub fn events_for_entity(&self, id: i64, filter: &ReadFilter, limit: usize) -> Result<Vec<Event>> {
        storage::validate_limit(limit)?;
        let state = self.0.read()?;
        let conn = state.conn();
        let _: Entity = storage::get(conn, &RecordKey { id }, &ReadFilter { tags: vec![], ..filter.clone() })?;
        let mut stmt = conn.prepare("SELECT event_id FROM event_participants WHERE entity_id=?1 ORDER BY event_id")?;
        let mut events = Vec::new();
        for row in stmt.query_map([id], |r| r.get::<_, i64>(0))? {
            let key = RecordKey { id: row? };
            if storage::matches_filter(conn, &key, filter)? { events.push(storage::get(conn, &key, filter)?); }
            if events.len() == limit { break; }
        }
        Ok(events)
    }
    pub fn delete(&self, kind: RecordKind, id: i64, filter: &ReadFilter) -> Result<WriteReceipt<bool>> {
        if !matches!(kind, RecordKind::Entity | RecordKind::Relation | RecordKind::Event) { return Err(Error::Validation("expected a graph record kind".into())); }
        self.0.mutate(|tx| {
            let key = RecordKey { id };
            if !storage::matches_filter(tx, &key, filter)? { return Err(Error::NotFound(id.to_string())); }
            storage::delete_record(tx, &key)
        })
    }
}
