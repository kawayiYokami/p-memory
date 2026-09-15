//! Read-only, repeatable migration into a separate p-memory directory.
use crate::{graph::{self, *}, memory::{self, *}, notes::{self, NoteInput},
    schema, storage::{self, KnowledgeBase}, text, types::*, Error, Result};
use rusqlite::{params, types::ValueRef, Connection, OpenFlags, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{collections::{BTreeMap, BTreeSet, HashMap}, path::{Path, PathBuf}};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LegacySource { #[serde(rename = "p_ai")] Pai, WorldTree, AngelMemory }
fn dry_run() -> bool { true }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportRequest {
    pub source: LegacySource,
    /// A stable host identity, e.g. "my-qq-bot". Keep it on subsequent imports.
    pub source_id: String,
    pub database: PathBuf,
    pub destination: PathBuf,
    #[serde(default)] pub graph_database: Option<PathBuf>,
    #[serde(default)] pub lookup_database: Option<PathBuf>,
    /// Only files below this explicit directory can be read as note sources.
    #[serde(default)] pub notes_root: Option<PathBuf>,
    #[serde(default = "default_namespace")] pub namespace: String,
    #[serde(default = "public_scope")] pub scope: String,
    #[serde(default = "dry_run")] pub dry_run: bool,
}
/// 来源 ID 与内部自增 ID 的映射；内部主键与外部 ID 无关，只在导入时关联。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdMapping { pub source_table: String, pub source_id: String, pub target_id: i64 }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportReport {
    pub source_id: String, pub source_fingerprint: String,
    pub applied: bool, pub already_imported: bool,
    pub counts: BTreeMap<String,usize>, pub id_map: Vec<IdMapping>,
    pub conflicts: Vec<String>, pub missing_sources: Vec<String>, pub warnings: Vec<String>,
    pub index_ready: bool, pub index_error: Option<String>,
}

/// 逻辑键，仅在导入计划内部用于把引用接到尚未分配的实体上。
fn key(table: &str, old: &str) -> String { format!("{table}\u{1f}{old}") }

#[derive(Serialize)]
struct MemoryDraft { table: String, source_id: String, input: MemoryInput }
#[derive(Serialize)]
struct EntityDraft { table: String, source_id: String, input: EntityInput }
impl EntityDraft { fn key(&self) -> String { key(&self.table, &self.source_id) } }
#[derive(Serialize)]
struct RelationDraft { table: String, source_id: String, record: RecordInput, subject: String, predicate: String, object: String, confidence: f64, reason: String }
#[derive(Serialize)]
struct EventDraft { table: String, source_id: String, record: RecordInput, name: String, summary: String, participants: Vec<String>, confidence: f64, reason: String }
#[derive(Serialize)]
struct NoteDraft { table: String, source_id: String, input: NoteInput }
#[derive(Default, Serialize)]
struct Plan { memories: Vec<MemoryDraft>, entities: Vec<EntityDraft>, relations: Vec<RelationDraft>, events: Vec<EventDraft>,
    notes: Vec<NoteDraft>, missing: BTreeSet<String>, warnings: Vec<String> }
type Row = serde_json::Map<String,Value>;

fn source_conn(path: &Path) -> Result<Connection> {
    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX)?;
    conn.execute_batch("PRAGMA query_only=ON; BEGIN DEFERRED;")?;
    Ok(conn)
}
fn has_table(conn: &Connection, name: &str) -> Result<bool> {
    Ok(conn.query_row("SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name=?1)", [name], |r|r.get(0))?)
}
fn rows(conn: &Connection, table: &str) -> Result<Vec<Row>> {
    if !table.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') { return Err(Error::Validation("invalid source table".into())); }
    if !has_table(conn,table)? { return Ok(vec![]); }
    let mut stmt=conn.prepare(&format!("SELECT * FROM {table} ORDER BY 1"))?;
    let names:Vec<_>=stmt.column_names().into_iter().map(String::from).collect();
    let mut query=stmt.query([])?;let mut result=Vec::new();
    while let Some(row)=query.next()? {
        let mut object=Row::new();
        for (i,name) in names.iter().enumerate() {
            let value=match row.get_ref(i)? {
                ValueRef::Null=>Value::Null,ValueRef::Integer(v)=>json!(v),ValueRef::Real(v)=>json!(v),
                ValueRef::Text(v)=>Value::String(String::from_utf8(v.to_vec()).map_err(|_|Error::Validation(format!("non-UTF-8 source in {table}.{name}")))?),
                ValueRef::Blob(v)=>json!({"bytes":v}),
            };
            object.insert(name.clone(),value);
        }
        result.push(object);
    }
    Ok(result)
}
fn string(row:&Row,key:&str)->String {
    match row.get(key) { Some(Value::String(s))=>s.clone(),Some(Value::Number(n))=>n.to_string(),_=>String::new() }
}
fn strings(value:Option<&Value>)->Vec<String> {
    match value { Some(Value::Array(a))=>a.iter().filter_map(|v|v.as_str().map(String::from)).collect(),Some(Value::String(s))=>vec![s.clone()],_=>vec![] }
}
fn number(row:&Row,key:&str,default:f64)->f64 { row.get(key).and_then(Value::as_f64).unwrap_or(default) }
fn timestamp(row:&Row,key:&str)->Result<Option<i64>> {
    let Some(value)=row.get(key).filter(|v|!v.is_null()) else {return Ok(None)};
    if let Some(s)=value.as_str() {
        if s.trim().is_empty(){return Ok(None)}
        if let Ok(dt)=chrono::DateTime::parse_from_rfc3339(s){return Ok(Some(dt.timestamp_micros()))}
        if let Ok(dt)=chrono::NaiveDateTime::parse_from_str(s.trim(),"%Y-%m-%dT%H:%M:%S%.f"){return Ok(Some(dt.and_utc().timestamp_micros()))}
        if let Ok(seconds)=s.parse::<f64>(){return seconds_to_us(seconds).map(Some)}
    }
    if let Some(seconds)=value.as_f64(){return seconds_to_us(seconds).map(Some)}
    Err(Error::Validation(format!("invalid source timestamp {key}: {value}")))
}
fn seconds_to_us(seconds:f64)->Result<i64>{
    let value=seconds*1_000_000.0;
    if !value.is_finite() || value < i64::MIN as f64 || value >= i64::MAX as f64 {return Err(Error::Validation("source timestamp out of range".into()))}
    Ok(value.round() as i64)
}
fn metadata(row:&Row)->Result<Row>{
    match row.get("metadata_json") {
        Some(Value::String(s)) if !s.trim().is_empty()=>Ok(serde_json::from_str(s)?),
        _=>Ok(Row::new()),
    }
}
fn namespace(req:&ImportRequest,domain:&str)->String {
    if matches!(req.source,LegacySource::WorldTree) && !domain.trim().is_empty(){format!("{}/{}",req.namespace,domain.trim())}else{req.namespace.clone()}
}
/// 只构造记录内容；内部自增 ID 由写入时分配，来源 ID 只进 metadata 溯源。
fn record(req:&ImportRequest,table:&str,old:&str,row:&Row,domain:&str)->Result<RecordInput>{
    if old.is_empty(){return Err(Error::Validation(format!("missing source ID in {table}")))}
    let namespace=namespace(req,domain);
    let mut metadata=metadata(row)?;
    metadata.insert("legacy".into(),json!({"format":req.source,"source_id":req.source_id,"table":table,"id":old,"row":row}));
    let scope=string(row,"memory_scope");
    Ok(RecordInput{id:None,namespace,scope:if scope.trim().is_empty(){req.scope.clone()}else{scope},metadata,
        created_at_us:Some(timestamp(row,"created_at")?.unwrap_or(0)),
        updated_at_us:Some(timestamp(row,"updated_at")?.or(timestamp(row,"created_at")?).unwrap_or(0)),..Default::default()})
}
fn tag_map(conn:&Connection,tags:&str,relations:&str,id_col:&str)->Result<BTreeMap<String,Vec<String>>>{
    let names:BTreeMap<_,_>=rows(conn,tags)?.iter().map(|r|(string(r,"id"),string(r,"name"))).collect();
    let mut map:BTreeMap<String,Vec<String>>=BTreeMap::new();
    for row in rows(conn,relations)? {if let Some(name)=names.get(&string(&row,"tag_id")){map.entry(string(&row,id_col)).or_default().push(name.clone());}}
    Ok(map)
}
fn evidence(value:&Value,fallback:&Row)->Vec<Evidence>{
    let values=match value {Value::Array(a)=>a.clone(),Value::Object(_)=>vec![value.clone()],_=>vec![]};
    let mut result=Vec::new();
    for item in values {
        if let Some(row)=item.as_object(){
            let source=string(row,"source");let source=if source.is_empty(){string(row,"file_path")}else{source};
            if source.is_empty(){continue}
            let start=row.get("line_start").and_then(Value::as_u64).filter(|v|*v>0).map(|v|v as usize);
            let end=row.get("line_end").and_then(Value::as_u64).map(|v|v as usize);
            let range=start.zip(end).filter(|(s,e)|e>=s);
            result.push(Evidence{source,source_revision:None,chunk_id:None,
                offset:range.map(|v|v.0),limit:range.map(|(s,e)|e-s+1),quote:string(row,"quote"),metadata:row.clone()});
        }
    }
    if result.is_empty(){
        let source=string(fallback,"file_path");
        if !source.is_empty(){result.push(Evidence{source,..Default::default()});}
    }
    result
}
fn memory_rows(req:&ImportRequest,plan:&mut Plan,conn:&Connection)->Result<Vec<Row>>{
    let (table,tags,rel)=match req.source{LegacySource::Pai=>("memory_record","global_tag","memory_tag_rel"),LegacySource::AngelMemory=>("memory_records","global_tags","memory_tag_rel"),LegacySource::WorldTree=>("world_tree_memory","world_tree_tag","world_tree_memory_tag")};
    if !has_table(conn,table)? {return Err(Error::Validation(format!("expected legacy table {table}")))}
    let memory_tags=tag_map(conn,tags,rel,"memory_id")?;
    for row in rows(conn,table)? {
        let old=string(&row,"id");let metadata=metadata(&row)?;
        let mut rec=record(req,table,&old,&row,&string(&metadata,"domain"))?;
        rec.tags=memory_tags.get(&old).cloned().unwrap_or_default();
        rec.evidence=evidence(metadata.get("evidence").unwrap_or(&Value::Null),&metadata);
        let kind=string(&row,"memory_type");
        let memory_type=match kind.as_str(){"知识记忆"=>"knowledge","事件记忆"=>"event","技能记忆"=>"skill","任务记忆"=>"task","情感记忆"=>"emotion",""=>"knowledge",_=>&kind}.to_string();
        let state=MemoryState{pinned:row.get("is_active").is_some_and(|v|v.as_bool()==Some(true)||v.as_i64()==Some(1)),
            strength:row.get("strength").and_then(Value::as_i64).unwrap_or(1),useful_count:row.get("useful_count").and_then(Value::as_i64).unwrap_or(0),useful_score:number(&row,"useful_score",0.0),
            last_recalled_at_us:timestamp(&row,"last_recalled_at")?.filter(|v|*v!=0),last_decay_at_us:timestamp(&row,"last_decay_at")?.filter(|v|*v!=0)};
        plan.memories.push(MemoryDraft{table:table.into(),source_id:old.clone(),
            input:MemoryInput{record:rec,memory_type,judgment:string(&row,"judgment"),reasoning:string(&row,"reasoning"),state:Some(state)}});
    }
    let note_table=if matches!(req.source,LegacySource::Pai){"note_index_record"}else{"note_index_records"};
    let mut notes=rows(conn,note_table)?;
    if matches!(req.source,LegacySource::Pai){
        let tags=tag_map(conn,"global_tag","note_tag_rel","source_id")?;
        for row in &mut notes {row.insert("tags".into(),json!(tags.get(&string(row,"source_id")).cloned().unwrap_or_default()));}
    }
    Ok(notes)
}
fn raw_graph(req:&ImportRequest,plan:&mut Plan)->Result<Vec<Row>>{
    let Some(path)=&req.graph_database else{return Ok(vec![])};
    let conn=source_conn(path)?;
    if !has_table(&conn,"world_tree_graph")?{return Err(Error::Validation("expected world_tree_graph table".into()))}
    let rows=rows(&conn,"world_tree_graph")?;
    let tags=tag_map(&conn,"world_tree_graph_tag","world_tree_graph_tag_map","graph_id")?;
    for row in &rows {
        let metadata=metadata(row)?;let old=string(row,"id");
        let mut rec=record(req,"world_tree_graph",&old,row,&string(&metadata,"domain"))?;
        rec.tags=tags.get(&old).cloned().unwrap_or_default();rec.tags.push("legacy:graph-extraction".into());
        rec.evidence=evidence(metadata.get("evidence").unwrap_or(&Value::Null),&metadata);
        plan.memories.push(MemoryDraft{table:"world_tree_graph".into(),source_id:old.clone(),
            input:MemoryInput{record:rec,memory_type:"graph_extraction".into(),judgment:string(row,"judgment"),reasoning:string(row,"reasoning"),state:None}});
    }
    Ok(rows)
}
fn lookup_graph(req:&ImportRequest,plan:&mut Plan,path:&Path)->Result<()> {
    let conn=source_conn(path)?;
    if !has_table(&conn,"entities")?{return Err(Error::Validation("expected entities table in lookup database".into()))}
    let mut aliases:BTreeMap<String,Vec<String>>=BTreeMap::new();
    for row in rows(&conn,"entity_aliases")?{aliases.entry(string(&row,"entity_id")).or_default().push(string(&row,"term"));}
    let mut attributes:BTreeMap<String,BTreeMap<String,Vec<String>>>=BTreeMap::new();
    for row in rows(&conn,"entity_attributes")?{attributes.entry(string(&row,"entity_id")).or_default().entry(string(&row,"attr_key")).or_default().push(string(&row,"attr_value"));}
    for row in rows(&conn,"entities")? {
        let old=string(&row,"entity_id");let rec=record(req,"entities",&old,&row,&string(&row,"domain"))?;
        plan.entities.push(EntityDraft{table:"entities".into(),source_id:old.clone(),
            input:EntityInput{record:rec,name:string(&row,"canonical_name"),entity_type:{let t=string(&row,"entity_type");if t.is_empty(){"concept".into()}else{t}},
                summary:string(&row,"summary"),aliases:aliases.remove(&old).unwrap_or_default(),attributes:attributes.remove(&old).unwrap_or_default()}});
    }
    for row in rows(&conn,"relations")?{
        let old=string(&row,"relation_id");let rec=record(req,"relations",&old,&row,&string(&row,"domain"))?;
        plan.relations.push(RelationDraft{table:"relations".into(),source_id:old,record:rec,
            subject:key("entities",&string(&row,"subject_entity_id")),object:key("entities",&string(&row,"object_entity_id")),
            predicate:string(&row,"predicate"),reason:string(&row,"reason"),confidence:number(&row,"confidence",0.8)});
    }
    let mut participants:BTreeMap<String,Vec<String>>=BTreeMap::new();
    for row in rows(&conn,"event_participants")?{participants.entry(string(&row,"event_id")).or_default().push(key("entities",&string(&row,"entity_id")));}
    let mut event_aliases:BTreeMap<String,Vec<String>>=BTreeMap::new();
    for row in rows(&conn,"event_aliases")?{event_aliases.entry(string(&row,"event_id")).or_default().push(string(&row,"term"));}
    for row in rows(&conn,"events")?{
        let old=string(&row,"event_id");let mut rec=record(req,"events",&old,&row,&string(&row,"domain"))?;
        rec.metadata.insert("legacy_event_aliases".into(),json!(event_aliases.remove(&old).unwrap_or_default()));
        plan.events.push(EventDraft{table:"events".into(),source_id:old.clone(),record:rec,name:string(&row,"event_name"),
            summary:string(&row,"summary"),reason:string(&row,"reason"),confidence:number(&row,"confidence",0.8),
            participants:participants.remove(&old).unwrap_or_default()});
    }
    Ok(())
}
fn attributes(value:Option<&Value>)->BTreeMap<String,Vec<String>>{
    value.and_then(Value::as_object).map(|m|m.iter().map(|(k,v)|(k.clone(),match v {Value::String(s)=>vec![s.clone()],Value::Array(a)=>a.iter().map(|v|v.as_str().map(String::from).unwrap_or_else(||v.to_string())).collect(),_=>vec![v.to_string()]})).collect()).unwrap_or_default()
}
fn extraction_graph(req:&ImportRequest,plan:&mut Plan,raw:&[Row])->Result<()> {
    if !raw.is_empty(){plan.warnings.push("No canonical lookup database supplied: extraction-local entities remain separate across source records.".into());}
    for row in raw {
        let metadata=metadata(row)?;let old=string(row,"id");let domain=string(&metadata,"domain");
        let mut local:BTreeMap<String,Vec<String>>=BTreeMap::new();
        for (i,item) in metadata.get("entities").and_then(Value::as_array).into_iter().flatten().enumerate(){
            let Some(item)=item.as_object()else{continue};let name=string(item,"name");
            let sid=format!("{old}:{i}");
            let mut rec=record(req,"extracted_entities",&sid,row,&domain)?;
            rec.metadata.insert("extraction".into(),json!(item));rec.evidence=evidence(item.get("evidence").unwrap_or(&Value::Null),&metadata);
            let entity_key=key("extracted_entities",&sid);let aliases=strings(item.get("aliases"));
            for alias in std::iter::once(&name).chain(aliases.iter()){let ids=local.entry(text::normalized_tag(alias)).or_default();if !ids.contains(&entity_key){ids.push(entity_key.clone());}}
            let kind=string(item,"type");plan.entities.push(EntityDraft{table:"extracted_entities".into(),source_id:sid,
                input:EntityInput{record:rec,name,entity_type:if kind.is_empty(){"concept".into()}else{kind},aliases,attributes:attributes(item.get("attributes")),summary:string(item,"summary")}});
        }
        // Legacy extraction can reference an entity omitted from the entities array.
        let resolve=|name:&str,plan:&mut Plan,local:&mut BTreeMap<String,Vec<String>>|->Result<String>{
            let name=name.trim();if name.is_empty(){return Err(Error::Validation(format!("empty entity reference in graph {old}")))}
            if let Some(ids)=local.get(&text::normalized_tag(name)){
                if ids.len()!=1{return Err(Error::Conflict(format!("ambiguous entity {name} in graph {old}")))}return Ok(ids[0].clone())
            }
            let sid=format!("{old}:{name}");
            let mut rec=record(req,"inferred_entities",&sid,row,&domain)?;
            rec.metadata.insert("inferred_from_legacy_reference".into(),json!(true));
            let entity_key=key("inferred_entities",&sid);local.insert(text::normalized_tag(name),vec![entity_key.clone()]);
            plan.entities.push(EntityDraft{table:"inferred_entities".into(),source_id:sid,
                input:EntityInput{record:rec,name:name.into(),entity_type:"concept".into(),aliases:vec![],attributes:BTreeMap::new(),summary:String::new()}});Ok(entity_key)
        };
        for (i,item) in metadata.get("relations").and_then(Value::as_array).into_iter().flatten().enumerate(){
            let Some(item)=item.as_object()else{continue};
            let subject=resolve(&string(item,"subject"),plan,&mut local)?;let object=resolve(&string(item,"object"),plan,&mut local)?;
            let sid=format!("{old}:{i}");
            let mut rec=record(req,"extracted_relations",&sid,row,&domain)?;
            rec.metadata.insert("extraction".into(),json!(item));rec.evidence=evidence(item.get("evidence").unwrap_or(&Value::Null),&metadata);
            plan.relations.push(RelationDraft{table:"extracted_relations".into(),source_id:sid,record:rec,subject,object,
                predicate:string(item,"predicate"),reason:string(item,"reason"),confidence:number(item,"confidence",0.8)});
        }
        for (i,item) in metadata.get("events").and_then(Value::as_array).into_iter().flatten().enumerate(){
            let Some(item)=item.as_object()else{continue};let mut participants=Vec::new();
            for name in strings(item.get("participants")){participants.push(resolve(&name,plan,&mut local)?);}
            let sid=format!("{old}:{i}");
            let mut rec=record(req,"extracted_events",&sid,row,&domain)?;
            rec.metadata.insert("extraction".into(),json!(item));rec.evidence=evidence(item.get("evidence").unwrap_or(&Value::Null),&metadata);
            plan.events.push(EventDraft{table:"extracted_events".into(),source_id:sid,record:rec,name:string(item,"name"),
                summary:string(item,"summary"),participants,confidence:number(item,"confidence",0.8),reason:string(item,"reason")});
        }
    }
    Ok(())
}

// Canonical IDs remain authoritative; attach source snapshots only when names
// resolve uniquely. Raw extraction records are always preserved independently.
fn attach_evidence(req:&ImportRequest,plan:&mut Plan,raw:&[Row])->Result<()> {
    let mut aliases:BTreeMap<(String,String),BTreeSet<String>>=BTreeMap::new();
    for entity in &plan.entities{for name in std::iter::once(&entity.input.name).chain(entity.input.aliases.iter()){
        aliases.entry((entity.input.record.namespace.clone(),text::normalized_tag(name))).or_default().insert(entity.key());}}
    let entity_index:HashMap<String,usize>=plan.entities.iter().enumerate().map(|(i,e)|(e.key().to_owned(),i)).collect();
    let mut relation_index:HashMap<(String,String,String,String),Vec<usize>>=HashMap::new();
    for (i,r) in plan.relations.iter().enumerate(){relation_index.entry((r.record.namespace.clone(),r.subject.clone(),r.object.clone(),r.predicate.clone())).or_default().push(i);}
    let mut event_index:HashMap<(String,String),Vec<usize>>=HashMap::new();
    for (i,e) in plan.events.iter().enumerate(){event_index.entry((e.record.namespace.clone(),e.name.clone())).or_default().push(i);}
    for row in raw{
        let meta=metadata(row)?;let ns=namespace(req,&string(&meta,"domain"));
        let resolve=|name:&str|->Option<String>{let ids=aliases.get(&(ns.clone(),text::normalized_tag(name)))?;if ids.len()==1{ids.first().cloned()}else{None}};
        for item in meta.get("entities").and_then(Value::as_array).into_iter().flatten(){
            let Some(item)=item.as_object()else{continue};let Some(id)=resolve(&string(item,"name"))else{continue};
            if let Some(&i)=entity_index.get(&id){plan.entities[i].input.record.evidence.extend(evidence(item.get("evidence").unwrap_or(&Value::Null),&meta));}
        }
        for item in meta.get("relations").and_then(Value::as_array).into_iter().flatten(){
            let Some(item)=item.as_object()else{continue};let (Some(s),Some(o))=(resolve(&string(item,"subject")),resolve(&string(item,"object")))else{continue};
            if let Some(idx)=relation_index.get(&(ns.clone(),s,o,string(item,"predicate"))){for &i in idx{plan.relations[i].record.evidence.extend(evidence(item.get("evidence").unwrap_or(&Value::Null),&meta));}}
        }
        for item in meta.get("events").and_then(Value::as_array).into_iter().flatten(){
            let Some(item)=item.as_object()else{continue};
            if let Some(idx)=event_index.get(&(ns.clone(),string(item,"name"))){for &i in idx{plan.events[i].record.evidence.extend(evidence(item.get("evidence").unwrap_or(&Value::Null),&meta));}}
        }
    }
    Ok(())
}
fn collect_files(root:&Path)->Result<Vec<PathBuf>>{
    let mut result=Vec::new();let mut stack=vec![root.to_path_buf()];
    while let Some(path)=stack.pop(){
        for entry in std::fs::read_dir(path)?{let entry=entry?;let kind=entry.file_type()?;
            if kind.is_symlink(){continue}if kind.is_dir(){stack.push(entry.path())}else if kind.is_file(){
                let ext=entry.path().extension().and_then(|s|s.to_str()).unwrap_or("").to_lowercase();if matches!(ext.as_str(),"md"|"markdown"|"txt"){result.push(entry.path());}
            }
        }
    }
    result.sort();Ok(result)
}
fn import_notes(req:&ImportRequest,plan:&mut Plan,index_rows:Vec<Row>,raw:&[Row])->Result<()> {
    let root=req.notes_root.as_ref().map(std::fs::canonicalize).transpose()?;
    let mut sources:BTreeMap<String,Row>=BTreeMap::new();
    for row in index_rows {let source=string(&row,"source_file_path");if !source.is_empty(){sources.insert(source,row);}}
    for row in raw {let meta=metadata(row)?;let source=string(&meta,"file_path");if !source.is_empty(){sources.entry(source).or_insert(meta);}}
    if let Some(root)=&root {
        for file in collect_files(root)? {let relative=file.strip_prefix(root).map_err(|e|Error::Validation(e.to_string()))?.to_string_lossy().replace('\\',"/");sources.entry(relative).or_default();}
    }
    let mut seen=BTreeSet::new();
    for (source,row) in sources {
        let Some(root)=&root else{plan.missing.insert(source);continue};
        let candidate=root.join(source.replace('\\',"/"));
        let path=match std::fs::canonicalize(&candidate){Ok(p)=>p,Err(e) if e.kind()==std::io::ErrorKind::NotFound=>{plan.missing.insert(source);continue},Err(e)=>return Err(e.into())};
        if !path.starts_with(root){plan.missing.insert(source.clone());plan.warnings.push(format!("Skipped note outside notes_root: {source}"));continue}
        let relative=path.strip_prefix(root).map_err(|e|Error::Validation(e.to_string()))?.to_string_lossy().replace('\\',"/");
        let domain=if matches!(req.source,LegacySource::WorldTree){relative.split('/').next().filter(|_|relative.contains('/')).unwrap_or("")}else{""};
        let domain={let explicit=string(&row,"domain");if explicit.is_empty(){domain.to_string()}else{explicit}};
        let ns=namespace(req,&domain);
        if !seen.insert((ns.clone(),path.clone())){continue}
        let sid=format!("{ns}:{relative}");
        let mut rec=record(req,"notes",&sid,&row,&domain)?;
        rec.tags=strings(row.get("tags"));
        let note=NoteInput{record:rec,source:relative,title:{let title=string(&row,"title");if title.is_empty(){string(&row,"heading_h1")}else{title}},content:std::fs::read_to_string(&path)?,chunk_chars:220};
        plan.notes.push(NoteDraft{table:"notes".into(),source_id:sid,input:note});
    }
    Ok(())
}
fn build(req:&ImportRequest)->Result<Plan>{
    storage::validate_identity("source_id",&req.source_id)?;storage::validate_identity("namespace",&req.namespace)?;storage::validate_identity("scope",&req.scope)?;
    if !matches!(req.source,LegacySource::WorldTree)&&(req.graph_database.is_some()||req.lookup_database.is_some()){return Err(Error::Validation("graph_database and lookup_database apply only to world_tree".into()))}
    let conn=source_conn(&req.database)?;let mut plan=Plan::default();let note_rows=memory_rows(req,&mut plan,&conn)?;
    let raw=raw_graph(req,&mut plan)?;
    if let Some(path)=&req.lookup_database{lookup_graph(req,&mut plan,path)?;attach_evidence(req,&mut plan,&raw)?;}else{extraction_graph(req,&mut plan,&raw)?;}
    import_notes(req,&mut plan,note_rows,&raw)?;
    plan.warnings.push("Legacy Tantivy/FAISS/vector caches are not copied. Vectors are generated inside the library: register an embedder and run embeddings.sync to fill them.".into());
    Ok(plan)
}
/// 两阶段写入：先建被引用方拿到内部 id，再按逻辑键把引用翻译成内部 id。
fn apply(conn:&Connection,plan:&Plan,mappings:&mut Vec<IdMapping>,conflicts:&mut Vec<String>)->Result<()> {
    for draft in &plan.memories { let memory=memory::upsert(conn,&draft.input)?;
        mappings.push(IdMapping{source_table:draft.table.clone(),source_id:draft.source_id.clone(),target_id:memory.header.id}); }
    let mut map:HashMap<String,i64>=HashMap::new();
    for draft in &plan.entities {
        let entity=graph::upsert_entity(conn,&draft.input)?;
        map.insert(draft.key(),entity.header.id);
        mappings.push(IdMapping{source_table:draft.table.clone(),source_id:draft.source_id.clone(),target_id:entity.header.id});
    }
    for draft in &plan.relations {
        let (Some(&subject),Some(&object))=(map.get(&draft.subject),map.get(&draft.object)) else {
            conflicts.push(format!("skipped relation {}: unresolved entity reference {} -> {}",draft.source_id,draft.subject,draft.object));continue;};
        let relation=graph::upsert_relation(conn,&RelationInput{record:draft.record.clone(),subject_id:subject,predicate:draft.predicate.clone(),object_id:object,confidence:draft.confidence,reason:draft.reason.clone()})?;
        mappings.push(IdMapping{source_table:draft.table.clone(),source_id:draft.source_id.clone(),target_id:relation.header.id});
    }
    for draft in &plan.events {
        let mut participants=Vec::new();
        for participant in &draft.participants { match map.get(participant){Some(id)=>participants.push(*id),None=>conflicts.push(format!("event {}: dropped unresolved participant {participant}",draft.source_id))} }
        let event=graph::upsert_event(conn,&EventInput{record:draft.record.clone(),name:draft.name.clone(),summary:draft.summary.clone(),participants,confidence:draft.confidence,reason:draft.reason.clone()})?;
        mappings.push(IdMapping{source_table:draft.table.clone(),source_id:draft.source_id.clone(),target_id:event.header.id});
    }
    for draft in &plan.notes { let note=notes::upsert(conn,&draft.input)?;
        mappings.push(IdMapping{source_table:draft.table.clone(),source_id:draft.source_id.clone(),target_id:note.header.id}); }
    Ok(())
}
/// Dry run validates in an in-memory database without creating the destination.
/// Applying requires an empty directory. Repeating an identical import returns
/// its receipt; changed sources never overwrite data previously imported.
pub fn import_legacy(req:&ImportRequest)->Result<ImportReport>{
    let plan=build(req)?;
    let fingerprint=text::digest(&serde_json::to_string(&plan)?);
    let mut report=ImportReport{source_id:req.source_id.clone(),source_fingerprint:fingerprint.clone(),applied:false,already_imported:false,
        counts:BTreeMap::new(),id_map:vec![],conflicts:vec![],missing_sources:plan.missing.iter().cloned().collect(),warnings:plan.warnings.clone(),index_ready:false,index_error:None};
    let mut preview=Connection::open_in_memory()?;schema::initialize(&mut preview)?;
    let tx=preview.transaction()?;
    let mut mappings=Vec::new();
    let mut preview_conflicts=Vec::new();
    if let Err(e)=apply(&tx,&plan,&mut mappings,&mut preview_conflicts){preview_conflicts.push(e.to_string());report.conflicts=preview_conflicts;return Ok(report)}
    report.id_map=mappings;
    {
        let mut stmt=tx.prepare("SELECT kind,COUNT(*) FROM records GROUP BY kind")?;for row in stmt.query_map([],|r|Ok((r.get::<_,i64>(0)?,r.get::<_,i64>(1)?)))?{
        let(code,count)=row?;let name=RecordKind::from_code(code).map(|k|k.as_str().to_string()).unwrap_or_else(||code.to_string());report.counts.insert(name,count as usize);}}
    drop(tx);
    if req.destination.exists()&&std::fs::read_dir(&req.destination)?.next().is_some(){
        let database=req.destination.join("store.sqlite3");
        if database.is_file(){
            let conn=source_conn(&database)?;
            let app:i64=conn.pragma_query_value(None,"application_id",|r|r.get(0))?;
            if app==schema::APPLICATION_ID&&has_table(&conn,"import_runs")?{
                let previous:Option<(String,String)>=conn.query_row("SELECT source_fingerprint,report_json FROM import_runs WHERE source_id=?1",[&req.source_id],|r|Ok((r.get(0)?,r.get(1)?))).optional()?;
                if let Some((old,serialized))=previous{
                    if old==fingerprint{let mut previous:ImportReport=serde_json::from_str(&serialized)?;previous.already_imported=true;previous.applied=false;return Ok(previous)}
                    report.conflicts.push("Source changed since the previous import; use a new destination for a new snapshot.".into());return Ok(report)
                }
            }
        }
        report.conflicts.push("Destination must be a new or empty directory; existing user data is never overwritten.".into());return Ok(report)
    }
    if req.dry_run{report.conflicts=preview_conflicts;return Ok(report)}
    let kb=KnowledgeBase::open(&req.destination)?;
    let receipt=kb.mutate(|tx|{
        let count:i64=tx.query_row("SELECT COUNT(*) FROM records",[],|r|r.get(0))?;
        if count!=0{return Err(Error::Conflict("destination became nonempty during import".into()))}
        let mut mappings=Vec::new();report.conflicts.clear();apply(tx,&plan,&mut mappings,&mut report.conflicts)?;report.id_map=mappings;report.applied=true;
        tx.execute("INSERT INTO import_runs(source_id,source_fingerprint,report_json) VALUES (?1,?2,?3)",params![req.source_id,fingerprint,serde_json::to_string(&report)?])?;
        Ok(())
    })?;
    report.index_ready=receipt.index_ready;report.index_error=receipt.index_error;
    kb.write(|writer| Ok(writer.conn.execute("UPDATE import_runs SET report_json=?2 WHERE source_id=?1",params![req.source_id,serde_json::to_string(&report)?])?))?;
    kb.close()?;Ok(report)
}
