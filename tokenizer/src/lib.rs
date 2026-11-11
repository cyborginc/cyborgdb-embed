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
///
/// When `truncated` is non-null it is set to 1 if the text did not fit. The
/// tokenizer moves whatever it cut into the overflowing encodings, so this costs
/// nothing; `written == max_length` cannot tell a cut text from an exact fit.
#[no_mangle]
pub extern "C" fn cyborgdb_tokenizer_encode(
    handle: *const Handle,
    text: *const c_char,
    ids: *mut u32,
    capacity: usize,
    written: *mut usize,
    truncated: *mut u8,
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
    if !truncated.is_null() {
        unsafe { *truncated = !encoding.get_overflowing().is_empty() as u8 };
    }
    if source.len() > capacity || ids.is_null() {
        return 1;
    }
    unsafe { std::ptr::copy_nonoverlapping(source.as_ptr(), ids, source.len()) };
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    const MAX_LENGTH: usize = 16;

    // Shaped like the registry's tokenizers: [CLS] and [SEP] added by a
    // post-processor, so the limit covers two special tokens.
    const DEFINITION: &str = r#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": {"type": "BertProcessing", "sep": ["[SEP]", 2], "cls": ["[CLS]", 1]},
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "word": 3},
            "unk_token": "[UNK]"
        }
    }"#;

    fn handle() -> *mut Handle {
        let handle = cyborgdb_tokenizer_new(DEFINITION.as_ptr(), DEFINITION.len(), MAX_LENGTH);
        assert!(!handle.is_null());
        handle
    }

    fn encode(handle: *const Handle, words: usize) -> (usize, u8) {
        let text = CString::new(vec!["word"; words].join(" ")).unwrap();
        let mut ids = [0u32; MAX_LENGTH];
        let mut written = 0usize;
        let mut truncated = 0xffu8;
        let rc = cyborgdb_tokenizer_encode(
            handle,
            text.as_ptr(),
            ids.as_mut_ptr(),
            ids.len(),
            &mut written,
            &mut truncated,
        );
        assert_eq!(rc, 0);
        (written, truncated)
    }

    #[test]
    fn flags_only_text_that_was_cut() {
        let handle = handle();
        let text_capacity = MAX_LENGTH - 2;
        assert_eq!(encode(handle, 3), (5, 0));
        assert_eq!(encode(handle, text_capacity), (MAX_LENGTH, 0), "exact fit");
        assert_eq!(encode(handle, text_capacity + 1), (MAX_LENGTH, 1));
        assert_eq!(encode(handle, 600), (MAX_LENGTH, 1));
        cyborgdb_tokenizer_free(handle);
    }

    #[test]
    fn null_flag_is_ignored() {
        let handle = handle();
        let text = CString::new("word word").unwrap();
        let mut ids = [0u32; MAX_LENGTH];
        let mut written = 0usize;
        let rc = cyborgdb_tokenizer_encode(
            handle,
            text.as_ptr(),
            ids.as_mut_ptr(),
            ids.len(),
            &mut written,
            std::ptr::null_mut(),
        );
        assert_eq!((rc, written), (0, 4));
        cyborgdb_tokenizer_free(handle);
    }
}
