use std::{ffi::OsStr, path::Path};

/// Checks if the filename has the extension.
/// The case of ext is ignored, and should not contain the leading period.
pub fn has_extension(filename: &str, ext: &str) -> bool {
    let file_ext = get_extension_from_filename(filename);
    if let Some(file_ext) = file_ext {
        file_ext.eq_ignore_ascii_case(ext)
    } else {
        false
    }
}

pub fn get_extension_from_filename(filename: &str) -> Option<&str> {
    Path::new(filename).extension().and_then(OsStr::to_str)
}
