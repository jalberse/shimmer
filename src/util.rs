// Downcase the string and remove any '-' or '_' characters; thus we can be
// a little flexible in what we match for argument names.
pub fn normalize_arg(arg: &str) -> String {
    let mut ret = String::new();
    for c in arg.chars() {
        if c != "_".chars().next().unwrap() && c != "-".chars().next().unwrap() {
            ret.push(c.to_ascii_lowercase())
        }
    }
    ret
}

pub fn is_quoted_string(s: &str) -> bool {
    s.len() >= 2 && s.starts_with('\"') && s.ends_with('\"')
}

pub fn dequote_string(s: &str) -> &str {
    assert!(is_quoted_string(s));
    &s[1..s.len() - 1]
}
