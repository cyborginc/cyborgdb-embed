//! C shim over the HuggingFace tokenizers library.
//!
//! The exported surface is deliberately tiny: everything else is localised out
//! of the archive so Rust's runtime symbols never reach a consuming link.

use std::ffi::{c_char, CStr};
use tokenizers::tokenizer::Tokenizer;

pub struct Handle {
    tokenizer: Tokenizer,
}

/// Builds a tokenizer from the bytes of a `tokenizer.json`.
///
/// Returns null when the definition cannot be parsed.
#[no_mangle]
pub extern "C" fn cyborgdb_tokenizer_new(json: *const u8, len: usize) -> *mut Handle {
    if json.is_null() {
        return std::ptr::null_mut();
    }
    let bytes = unsafe { std::slice::from_raw_parts(json, len) };
    match Tokenizer::from_bytes(bytes) {
        Ok(tokenizer) => Box::into_raw(Box::new(Handle { tokenizer })),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn cyborgdb_tokenizer_free(handle: *mut Handle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) };
    }
}

/// Encodes one string, writing ids into `ids` and returning the count written.
/// Returns the required length when the buffer is too small, so the caller can
/// size and retry without a second tokenisation elsewhere.
#[no_mangle]
pub extern "C" fn cyborgdb_tokenizer_encode(
    handle: *const Handle,
    text: *const c_char,
    ids: *mut u32,
    capacity: usize,
    written: *mut usize,
) -> i32 {
    if handle.is_null() || text.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*handle };
    let Ok(text) = (unsafe { CStr::from_ptr(text) }).to_str() else {
        return -1;
    };
    let Ok(encoding) = handle.tokenizer.encode(text, true) else {
        return -1;
    };
    let source = encoding.get_ids();
    unsafe { *written = source.len() };
    if source.len() > capacity || ids.is_null() {
        return 1;
    }
    unsafe { std::ptr::copy_nonoverlapping(source.as_ptr(), ids, source.len()) };
    0
}
