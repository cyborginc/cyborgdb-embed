//! C shim over the HuggingFace tokenizers library.
//!
//! The exported surface is deliberately tiny: everything else is localised out
//! of the archive so Rust's runtime symbols never reach a consuming link.

use std::ffi::{c_char, CStr};
use tokenizers::tokenizer::Tokenizer;
use tokenizers::{PaddingParams, TruncationParams};

pub struct Handle {
    tokenizer: Tokenizer,
}

/// Builds a tokenizer from the bytes of a `tokenizer.json`, truncating at
/// `max_length` tokens.
///
/// Truncation is configured here rather than by trimming ids afterwards: the
/// tokenizer keeps the trailing separator token, and a tail slice does not.
///
/// Any padding strategy in the definition is cleared. Several models ship
/// `{"strategy": {"Fixed": 128}}`, which pads every short input to a constant
/// width — identical output for many times the work. Callers pad to the batch.
///
/// Returns null when the definition cannot be parsed.
#[no_mangle]
pub extern "C" fn cyborgdb_tokenizer_new(
    json: *const u8,
    len: usize,
    max_length: usize,
) -> *mut Handle {
    if json.is_null() || max_length == 0 {
        return std::ptr::null_mut();
    }
    let bytes = unsafe { std::slice::from_raw_parts(json, len) };
    let Ok(mut tokenizer) = Tokenizer::from_bytes(bytes) else {
        return std::ptr::null_mut();
    };
    let truncation = TruncationParams {
        max_length,
        ..Default::default()
    };
    if tokenizer.with_truncation(Some(truncation)).is_err() {
        return std::ptr::null_mut();
    }
    tokenizer.with_padding(None::<PaddingParams>);
    Box::into_raw(Box::new(Handle { tokenizer }))
}

#[no_mangle]
pub extern "C" fn cyborgdb_tokenizer_free(handle: *mut Handle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) };
    }
}

/// Encodes one string, writing ids into `ids` and setting `written`.
///
/// The encoding is already truncated to the length fixed at construction, so a
/// buffer of that size is always sufficient. A short buffer is a caller bug and
/// returns 1 without writing.
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
