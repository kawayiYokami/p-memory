use p_memory::{protocol, EmbedCallbackError, EmbedErrorKind, Embedder, EmbedderOptions,
               KnowledgeBase, Reranker, RerankerOptions};
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyList};
use serde_json::Value;

/// Operations release the GIL and use the core's shared engine mutex.
#[pyclass(frozen)]
struct NativeKnowledgeBase { inner: KnowledgeBase }

fn native_error(error:p_memory::Error)->PyErr{
    PyRuntimeError::new_err(serde_json::json!({"code":error.code(),"message":error.to_string()}).to_string())
}

/// 宿主嵌入回调的 Rust 侧代理。回调是 Python 函数，无法走 JSON 协议，
/// 因此注册与调用都留在绑定层：核心释放 GIL 调到这里，再重新取 GIL 回 Python。
struct PyEmbedder{callback:Py<PyAny>}

impl Embedder for PyEmbedder{
    fn embed(&mut self,texts:&[String])->std::result::Result<Vec<Vec<f32>>,EmbedCallbackError>{
        Python::attach(|py|{
            let argument=PyList::new(py,texts).map_err(|error|EmbedCallbackError::other(error.to_string()))?;
            match self.callback.bind(py).call1((argument,)){
                Ok(value)=>value.extract::<Vec<Vec<f32>>>().map_err(|error|
                    EmbedCallbackError::other(format!("embedder must return a list of float vectors: {error}"))),
                Err(error)=>Err(embed_error(py,&error)),
            }
        })
    }
}

/// 把宿主抛出的异常映射成带类别的回调错误。宿主在异常上挂 `kind`
/// （`too_large` / `rate_limited` / `other`）显式声明类别，库不解析错误文案。
fn embed_error(py:Python<'_>,error:&PyErr)->EmbedCallbackError{
    let message=error.value(py).to_string();
    let kind=error.value(py).getattr("kind").ok()
        .and_then(|value|value.extract::<String>().ok())
        .and_then(|code|EmbedErrorKind::from_code(&code))
        .unwrap_or(EmbedErrorKind::Other);
    EmbedCallbackError::new(kind,message)
}

struct PyReranker{callback:Py<PyAny>}

impl Reranker for PyReranker{
    fn rerank(&mut self,query:&str,documents:&[String])->std::result::Result<Vec<f32>,String>{
        Python::attach(|py|{
            let documents=PyList::new(py,documents).map_err(|error|error.to_string())?;
            let value=self.callback.bind(py).call1((query,documents)).map_err(|error|error.value(py).to_string())?;
            value.extract::<Vec<f32>>().map_err(|error|format!("reranker must return a list of floats: {error}"))
        })
    }
}

#[pymethods]
impl NativeKnowledgeBase {
    #[new]
    fn new(py:Python<'_>,path:String)->PyResult<Self>{
        py.detach(move ||KnowledgeBase::open(path)).map(|inner|Self{inner}).map_err(native_error)
    }
    #[staticmethod]
    fn restore(py:Python<'_>,snapshot:String,directory:String)->PyResult<Self>{
        py.detach(move ||KnowledgeBase::restore(snapshot,directory)).map(|inner|Self{inner}).map_err(native_error)
    }
    fn call(&self,py:Python<'_>,operation:String,args_json:String)->String{
        let kb=self.inner.clone();
        py.detach(move ||{
            let result=serde_json::from_str::<Value>(&args_json).map_err(p_memory::Error::from).and_then(|args|protocol::dispatch(&kb,&operation,args));
            protocol::envelope(result).to_string()
        })
    }
    /// 注册嵌入回调：一个回调对应一个向量空间，注册即用样本校验，不符即拒绝绑定。
    #[pyo3(signature=(space_id,callback,max_batch=32,max_tokens_per_text=None))]
    fn register_embedder(&self,py:Python<'_>,space_id:String,callback:Py<PyAny>,
                         max_batch:usize,max_tokens_per_text:Option<usize>)->PyResult<()>{
        let kb=self.inner.clone();
        let embedder=PyEmbedder{callback};
        py.detach(move ||kb.embeddings().register_embedder_with(
            &space_id,embedder,EmbedderOptions{max_batch,max_tokens_per_text})).map_err(native_error)
    }
    /// 注册重排回调：库在调用前按声明的条数上限与 token 预算强制截断，两者是与门。
    #[pyo3(signature=(callback,max_tokens_total=8192,max_candidates=50,max_tokens_per_doc=1024,max_tokens_query=None))]
    fn register_reranker(&self,py:Python<'_>,callback:Py<PyAny>,max_tokens_total:usize,
                         max_candidates:usize,max_tokens_per_doc:usize,max_tokens_query:Option<usize>)->PyResult<()>{
        let kb=self.inner.clone();
        let reranker=PyReranker{callback};
        py.detach(move ||kb.register_reranker_with(reranker,
            RerankerOptions{max_tokens_total,max_candidates,max_tokens_per_doc,max_tokens_query})).map_err(native_error)
    }
    /// 注册事件回调：`callback(event: dict) -> None`。库在检索线程里同步调用它，
    /// 回调必须非阻塞——在里面做同步 IO 或网络上报会把检索拖住。
    /// 回调抛错只丢这一条事件，不影响检索。
    fn register_event_sink(&self,callback:Py<PyAny>){
        self.inner.register_event_sink(move |event|{
            let Ok(payload)=serde_json::to_string(event) else { return };
            Python::attach(|py|{
                if let Ok(dict)=py.import("json").and_then(|json|json.call_method1("loads",(payload,))){
                    let _=callback.bind(py).call1((dict,));
                }
            });
        });
    }
    /// 注销事件回调，返回此前是否有注册。
    fn unregister_event_sink(&self)->bool{ self.inner.unregister_event_sink() }
    /// 当前是否注册了事件回调。
    fn event_sink_registered(&self)->bool{ self.inner.event_sink_registered() }
}
#[pyfunction]
fn import_legacy(py:Python<'_>,request_json:String)->String{
    py.detach(move ||{
        let result=serde_json::from_str::<p_memory::legacy::ImportRequest>(&request_json).map_err(p_memory::Error::from)
            .and_then(|request|p_memory::legacy::import_legacy(&request))
            .and_then(|report|serde_json::to_value(report).map_err(p_memory::Error::from));
        protocol::envelope(result).to_string()
    })
}
#[pyfunction]
fn delete_import_run(py:Python<'_>,request_json:String)->String{
    py.detach(move ||{
        let result=serde_json::from_str::<serde_json::Value>(&request_json).map_err(p_memory::Error::from)
            .and_then(|args|{
                let destination=args.get("destination").and_then(|v|v.as_str())
                    .ok_or_else(||p_memory::Error::Validation("destination is required".into()))?;
                let source_id=args.get("source_id").and_then(|v|v.as_str())
                    .ok_or_else(||p_memory::Error::Validation("source_id is required".into()))?;
                let removed=p_memory::legacy::delete_import_run(std::path::Path::new(destination),source_id)?;
                serde_json::to_value(removed).map_err(p_memory::Error::from)
            });
        protocol::envelope(result).to_string()
    })
}
#[pymodule]
fn _native(module:&Bound<'_,PyModule>)->PyResult<()>{
    module.add_class::<NativeKnowledgeBase>()?;
    module.add_function(wrap_pyfunction!(import_legacy,module)?)?;
    module.add_function(wrap_pyfunction!(delete_import_run,module)?)?;
    module.add("PROTOCOL_VERSION",protocol::PROTOCOL_VERSION)?;
    module.add("__version__",env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
