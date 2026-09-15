use p_memory::{protocol, KnowledgeBase};
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use serde_json::Value;

/// Operations release the GIL and use the core's shared engine mutex.
#[pyclass(frozen)]
struct NativeKnowledgeBase { inner: KnowledgeBase }

fn native_error(error:p_memory::Error)->PyErr{
    PyRuntimeError::new_err(serde_json::json!({"code":error.code(),"message":error.to_string()}).to_string())
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
#[pymodule]
fn _native(module:&Bound<'_,PyModule>)->PyResult<()>{
    module.add_class::<NativeKnowledgeBase>()?;
    module.add_function(wrap_pyfunction!(import_legacy,module)?)?;
    module.add("PROTOCOL_VERSION",protocol::PROTOCOL_VERSION)?;
    module.add("__version__",env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
