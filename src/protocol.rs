//! Versioned JSON boundary for thin foreign-language bindings. Domain behavior
//! lives in the typed Rust stores; this module only decodes and dispatches.
use crate::{KnowledgeBase, Result, Error, types::*, memory::DecayPolicy};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};

pub const PROTOCOL_VERSION:u32=1;
fn decode<T:DeserializeOwned>(value:Value)->Result<T>{Ok(serde_json::from_value(value)?)}
fn encode<T:Serialize>(value:T)->Result<Value>{Ok(serde_json::to_value(value)?)}
fn field<T:DeserializeOwned>(value:&Value,name:&str)->Result<T>{decode(value.get(name).cloned().ok_or_else(||Error::Validation(format!("missing {name}")))?)}
fn optional<T:DeserializeOwned+Default>(value:&Value,name:&str)->Result<T>{value.get(name).cloned().map(decode).transpose().map(|v|v.unwrap_or_default())}
#[derive(Deserialize)]
struct IdRequest { id:i64, #[serde(default)] filter:ReadFilter }
#[derive(Deserialize)]
struct GraphId { id:i64, kind:RecordKind, #[serde(default)] filter:ReadFilter }

pub fn dispatch(kb:&KnowledgeBase,operation:&str,args:Value)->Result<Value>{
    match operation {
        "health"=>encode(kb.health()?),
        "rebuild_indexes"=>encode(kb.rebuild_indexes()?),
        "backup"=>{kb.backup(field::<String>(&args,"path")?)?;Ok(Value::Null)},
        "close"=>{kb.close()?;Ok(Value::Null)},
        "search"=>encode(kb.search(&decode(args)?)?),
        "memories.upsert"=>encode(kb.memories().upsert(decode(args)?)?),
        "memories.upsert_many"=>encode(kb.memories().upsert_many(&decode::<Vec<_>>(args)?)?),
        "memories.upsert_by_judgment"=>encode(kb.memories().upsert_by_judgment(decode(args)?)?),
        "memories.get"=>{let r:IdRequest=decode(args)?;encode(kb.memories().get(r.id,&r.filter)?)},
        "memories.list"=>encode(kb.memories().list(&decode(args)?)?),
        "memories.delete"=>{let r:IdRequest=decode(args)?;encode(kb.memories().delete(r.id,&r.filter)?)},
        "memories.feedback"=>encode(kb.memories().feedback(&decode(args)?)?),
        "memories.decay"=>encode(kb.memories().decay(&optional::<ReadFilter>(&args,"filter")?,&optional::<DecayPolicy>(&args,"policy")?,optional(&args,"now_us")?)?),
        "graph.apply_batch"=>encode(kb.graph().apply_batch(&decode(args)?)?),
        "graph.get"=>{let r:GraphId=decode(args)?;encode(kb.graph().get(r.kind,r.id,&r.filter)?)},
        "graph.list"=>encode(kb.graph().list(field(&args,"kind")?,&optional::<PageRequest>(&args,"page")?)?),
        "graph.delete"=>{let r:GraphId=decode(args)?;encode(kb.graph().delete(r.kind,r.id,&r.filter)?)},
        "graph.resolve"=>encode(kb.graph().resolve(&field::<String>(&args,"name")?,&optional::<ReadFilter>(&args,"filter")?,args.get("limit").cloned().map(decode).transpose()?.unwrap_or(10))?),
        "graph.neighbors"=>encode(kb.graph().neighbors(field::<i64>(&args,"id")?,&optional::<ReadFilter>(&args,"filter")?,args.get("limit").cloned().map(decode).transpose()?.unwrap_or(50))?),
        "graph.events_for_entity"=>encode(kb.graph().events_for_entity(field::<i64>(&args,"id")?,&optional::<ReadFilter>(&args,"filter")?,args.get("limit").cloned().map(decode).transpose()?.unwrap_or(50))?),
        "graph.ego"=>{let id:i64=field(&args,"id")?;let depth:usize=args.get("depth").cloned().map(decode).transpose()?.unwrap_or(1);let limit:usize=args.get("limit").cloned().map(decode).transpose()?.unwrap_or(50);encode(kb.graph().ego(id,depth,&optional::<ReadFilter>(&args,"filter")?,limit)?)},
        "graph.path"=>encode(kb.graph().path(field::<i64>(&args,"from")?,field::<i64>(&args,"to")?,&optional::<ReadFilter>(&args,"filter")?)?),
        "graph.strongly_connected"=>encode(kb.graph().strongly_connected(&optional::<ReadFilter>(&args,"filter")?)?),
        "graph.component_count"=>encode(kb.graph().build_graph(&optional::<ReadFilter>(&args,"filter")?)?.component_count()),
        "notes.upsert"=>encode(kb.notes().upsert(decode(args)?)?),
        "notes.get"=>{let r:IdRequest=decode(args)?;encode(kb.notes().get(r.id,&r.filter)?)},
        "notes.list"=>encode(kb.notes().list(&decode(args)?)?),
        "notes.delete"=>{let r:IdRequest=decode(args)?;encode(kb.notes().delete(r.id,&r.filter)?)},
        "notes.chunks"=>{let r:IdRequest=decode(args)?;encode(kb.notes().chunks(r.id,&r.filter)?)},
        "notes.get_chunk"=>{let r:IdRequest=decode(args)?;encode(kb.notes().get_chunk(r.id,&r.filter)?)},
        "embeddings.register_space"=>encode(kb.embeddings().register_space(decode(args)?)?),
        "embeddings.spaces"=>encode(kb.embeddings().spaces()?),
        "embeddings.embedder_space"=>encode(kb.embeddings().embedder_space(&field::<String>(&args,"space_id")?)?),
        "embeddings.sync"=>encode(kb.embeddings().sync(&field::<String>(&args,"space_id")?,field(&args,"batch")?)?),
        "embeddings.unregister_embedder"=>encode(kb.embeddings().unregister_embedder(&field::<String>(&args,"space_id")?)?),
        "embeddings.namespace_vectorization"=>encode(kb.embeddings().namespace_vectorization(&field::<String>(&args,"namespace")?)?),
        "embeddings.set_namespace_vectorization"=>encode(kb.embeddings().set_namespace_vectorization(&field::<String>(&args,"namespace")?,field(&args,"enabled")?)?),
        "embeddings.delete_space"=>encode(kb.embeddings().delete_space(&field::<String>(&args,"id")?)?),
        _=>Err(Error::Validation(format!("unknown operation: {operation}"))),
    }
}

pub fn envelope(result:Result<Value>)->Value {
    match result {Ok(value)=>json!({"ok":true,"result":value}),Err(error)=>json!({"ok":false,"error":{"code":error.code(),"message":error.to_string()}})}
}
