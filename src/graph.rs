use crate::{storage::{self, KnowledgeBase}, text, types::*, Error, Result};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};

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

fn referenced_entity(conn: &Connection, record: &RecordInput, id: i64) -> Result<Entity> {
    storage::get(conn, &RecordKey { id },
        &ReadFilter { namespace: record.namespace.clone(), scopes: vec![record.scope.clone()], tags: vec![] })
}

pub(crate) fn upsert_entity(conn: &Connection, input: &EntityInput) -> Result<Entity> {
    storage::validate_identity("entity name", &input.name)?;
    storage::validate_identity("entity_type", &input.entity_type)?;
    let aliases: Vec<_> = input.aliases.iter().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect::<BTreeSet<_>>().into_iter().collect();
    let attributes = input.attributes.clone();
    let attr_text = attributes.iter().map(|(k, values)| format!("{k} {}", values.join(" "))).collect::<Vec<_>>().join(" ");
    let body = format!("{} {} {} {}", input.name, aliases.join(" "), input.summary, attr_text);
    let header = storage::put_record(conn, RecordKind::Entity, &input.record,
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
    Ok(Entity { header, name: input.name.clone(), entity_type: input.entity_type.clone(), aliases, attributes, summary: input.summary.clone() })
}

pub(crate) fn upsert_relation(conn: &Connection, input: &RelationInput) -> Result<Relation> {
    storage::validate_identity("predicate", &input.predicate)?;
    check_confidence(input.confidence)?;
    let subject = referenced_entity(conn, &input.record, input.subject_id)?;
    let object = referenced_entity(conn, &input.record, input.object_id)?;
    let body = format!("{} {} {} {}", subject.name, input.predicate, object.name, input.reason);
    let header = storage::put_record(conn, RecordKind::Relation, &input.record,
        &json!({"subject_id":input.subject_id,"predicate":input.predicate,"object_id":input.object_id,"confidence":input.confidence,"reason":input.reason,
            "subject_name":subject.name,"object_name":object.name}), &body)?;
    let predicate_id = storage::term_id(conn, &input.predicate)?;
    conn.execute("INSERT INTO relations(record_id,subject_id,predicate_id,object_id) VALUES (?1,?2,?3,?4)
        ON CONFLICT(record_id) DO UPDATE SET subject_id=excluded.subject_id,predicate_id=excluded.predicate_id,object_id=excluded.object_id",
        params![header.id, input.subject_id, predicate_id, input.object_id])?;
    Ok(Relation { header, subject_id: input.subject_id, predicate: input.predicate.clone(), object_id: input.object_id, confidence: input.confidence, reason: input.reason.clone() })
}

pub(crate) fn upsert_event(conn: &Connection, input: &EventInput) -> Result<Event> {
    storage::validate_identity("event name", &input.name)?;
    check_confidence(input.confidence)?;
    let participants: Vec<_> = input.participants.iter().copied().collect::<BTreeSet<_>>().into_iter().collect();
    let names = participants.iter().map(|id| referenced_entity(conn, &input.record, *id).map(|e| e.name)).collect::<Result<Vec<_>>>()?;
    let body = format!("{} {} {} {}", input.name, input.summary, names.join(" "), input.reason);
    let header = storage::put_record(conn, RecordKind::Event, &input.record,
        &json!({"name":input.name,"summary":input.summary,"participants":participants,"confidence":input.confidence,"reason":input.reason,
            "participant_names":names}), &body)?;
    conn.execute("DELETE FROM event_participants WHERE event_id=?1", [header.id])?;
    for id in &participants {
        conn.execute("INSERT INTO event_participants(event_id,entity_id) VALUES (?1,?2)", params![header.id, id])?;
    }
    Ok(Event { header, name: input.name.clone(), summary: input.summary.clone(), participants, confidence: input.confidence, reason: input.reason.clone() })
}

pub(crate) fn apply_batch(conn: &Connection, batch: &GraphBatch) -> Result<(GraphBatchResult, Vec<i64>)> {
    // Entities first permits references to entities created in this transaction.
    let entities = batch.entities.iter().map(|v| upsert_entity(conn, v)).collect::<Result<Vec<_>>>()?;
    let relations = batch.relations.iter().map(|v| upsert_relation(conn, v)).collect::<Result<Vec<_>>>()?;
    let events = batch.events.iter().map(|v| upsert_event(conn, v)).collect::<Result<Vec<_>>>()?;
    // 改名会改写引用它的关系与事件正文，这些记录的向量必须一并重算。
    let refreshed = refresh_dependents(conn, &entities)?;
    let mut ids: Vec<i64> = entities.iter().map(|e| e.header.id)
        .chain(relations.iter().map(|r| r.header.id)).chain(events.iter().map(|e| e.header.id)).collect();
    ids.extend(refreshed);
    ids.sort_unstable();
    ids.dedup();
    Ok((GraphBatchResult { entities, relations, events }, ids))
}

/// 返回因引用正文变化而被重写的记录 id。
fn refresh_dependents(conn: &Connection, entities: &[Entity]) -> Result<Vec<i64>> {
    let mut keys = BTreeSet::new();
    let mut rewritten = Vec::new();
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
        let old = storage::search_text(conn, kind, &value)?;
        if old != body {
            let raw: String = conn.query_row("SELECT payload_json FROM records WHERE id=?1", [key.id], |r| r.get(0))?;
            let mut payload: serde_json::Value = serde_json::from_str(&raw)?;
            if let Some(object) = payload.as_object_mut() {
                if let Some(extra) = patch.as_object() {
                    for (name, value) in extra { object.insert(name.clone(), value.clone()); }
                }
            }
            let revision = storage::next_revision(conn, key.id)?;
            conn.execute("UPDATE records SET payload_json=?2,fingerprint=?3,revision=?4,updated_at_us=MAX(updated_at_us,?5) WHERE id=?1",
                params![key.id, serde_json::to_string(&payload)?, text::digest(&format!("text-v1\n{body}")), revision, storage::now_us()])?;
            conn.execute("DELETE FROM embeddings WHERE record_id=?1", [key.id])?;
            rewritten.push(key.id);
        }
    }
    Ok(rewritten)
}

impl GraphStore {
    pub fn apply_batch(&self, batch: &GraphBatch) -> Result<WriteReceipt<GraphBatchResult>> {
        let mut ids: Vec<i64> = Vec::new();
        let receipt = self.0.mutate(|tx| {
            let (result, written) = apply_batch(tx, batch)?;
            ids = written;
            Ok(result)
        })?;
        self.0.vectorize(&ids);
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
