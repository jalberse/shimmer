// Adapted from pbrt4 crate under apache license.

use super::token::Token;

/// Tokenizer splits a string into an iterator of tokens.
pub(crate) struct Tokenizer<'a> {
    str: &'a str,
    offset: usize,
}

impl<'a> Tokenizer<'a> {
    pub fn new(str: &'a str) -> Self {
        Self { str, offset: 0 }
    }

    fn rewind_until(&mut self, chars: &[char]) -> usize {
        let mut offset = 0;

        loop {
            // Peek next char
            let Some(ch) = self.peek_char() else {
                break;
            };

            if chars.contains(&ch) {
                break;
            }

            // Take next char
            if let Some((pos, _)) = self.next_char() {
                offset = pos;
            }
        }

        offset
    }

    fn peek_char(&mut self) -> Option<char> {
        if self.offset > self.str.len() {
            None
        } else {
            self.str[self.offset..].chars().next()
        }
    }

    /// Get current char and step forward.
    fn next_char(&mut self) -> Option<(usize, char)> {
        match self.peek_char() {
            Some(ch) => {
                let offset = self.offset;
                self.offset += 1;
                Some((offset, ch))
            }
            None => None,
        }
    }

    /// Get current token without moving forward.
    pub fn peek_token(&mut self) -> Option<Token<'a>> {
        let offset = self.offset;
        let token = self.next();
        self.offset = offset;

        token
    }

    pub fn token(&self, start: usize, end: usize) -> Token<'a> {
        let token = Token::new(&self.str[start..end]);
        dbg!(token)
    }

    /// Return current offset within string.
    pub fn offset(&self) -> usize {
        self.offset
    }
}

impl<'a> Iterator for Tokenizer<'a> {
    type Item = Token<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let Some((start, ch)) = self.next_char() else {
                return None;
            };

            let token = match ch {
                '[' | ']' => self.token(start, start + 1),
                ' ' | '\n' | '\t' | '\r' => continue,
                '"' => {
                    let mut end = self.rewind_until(&['"']);

                    // Consume remaining "
                    if let Some((pos, _)) = self.next_char() {
                        end = pos;
                    }

                    self.token(start, end + 1)
                }
                '#' => {
                    // Skip comment line
                    self.rewind_until(&['\r', '\n']);
                    continue;
                }
                _ => {
                    let mut end = self.rewind_until(&[' ', '\r', '\n', '\t', '"', '[', ']']);
                    if end == 0 {
                        end = start;
                    }

                    self.token(start, end + 1)
                }
            };

            return Some(token);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_line() {
        let mut t = Tokenizer::new("");

        assert_eq!(t.next(), None);
        assert_eq!(t.next(), None);
    }

    #[test]
    fn single_token() {
        let mut t = Tokenizer::new("Scale");

        assert_eq!(t.next(), Some(Token::new("Scale")));
        assert_eq!(t.next(), None);
    }

    #[test]
    fn two_tokens() {
        let mut t = Tokenizer::new("Scale Scale");

        assert_eq!(t.next(), Some(Token::new("Scale")));
        assert_eq!(t.next(), Some(Token::new("Scale")));
        assert_eq!(t.next(), None);
    }

    #[test]
    fn skip_newlines() {
        let str = r#"


        "#;

        let mut t = Tokenizer::new(str);
        assert_eq!(t.next(), None);
    }

    #[test]
    fn brackets() {
        let mut t = Tokenizer::new("[ abc ]");

        assert_eq!(t.next(), Some(Token::new("[")));
        assert_eq!(t.next(), Some(Token::new("abc")));
        assert_eq!(t.next(), Some(Token::new("]")));
        assert_eq!(t.next(), None);
    }

    #[test]
    fn skip_comments() {
        let str = r#"
# Comment

Scale

"#;

        let mut t = Tokenizer::new(str);

        assert_eq!(t.next(), Some(Token::new("Scale")));
        assert_eq!(t.next(), None);
    }

    #[test]
    fn comment_middle() {
        let str = r#"
Scale

# Comment"#;

        let mut t = Tokenizer::new(str);

        assert_eq!(t.next(), Some(Token::new("Scale")));
        assert_eq!(t.next(), None);
    }

    #[test]
    fn quotes_single() {
        let mut t = Tokenizer::new(r#" "test" "#);

        assert_eq!(t.next(), Some(Token::new("\"test\"")));
        assert_eq!(t.next(), None);
    }

    #[test]
    fn quotes_two() {
        let mut t = Tokenizer::new(r#" "foo" [] "bar" "#);

        assert_eq!(t.next(), Some(Token::new("\"foo\"")));

        assert_eq!(t.next(), Some(Token::new("[")));
        assert_eq!(t.next(), Some(Token::new("]")));

        assert_eq!(t.next(), Some(Token::new("\"bar\"")));

        assert_eq!(t.next(), None);
    }

    #[test]
    fn single_quote() {
        let mut t = Tokenizer::new("foo \"abc");

        assert_eq!(t.next(), Some(Token::new("foo")));
        assert_eq!(t.next(), Some(Token::new("\"abc")));

        assert_eq!(t.next(), None);
    }

    #[test]
    fn single_quote_with_spaces() {
        let mut t = Tokenizer::new("foo \"abc test [] ");

        assert_eq!(t.next(), Some(Token::new("foo")));
        assert_eq!(t.next(), Some(Token::new("\"abc test [] ")));

        assert_eq!(t.next(), None);
    }

    #[test]
    fn just_quote() {
        let mut t = Tokenizer::new("\"");

        assert_eq!(t.next(), Some(Token::new("\"")));
        assert_eq!(t.next(), None);
    }

    #[test]
    fn parse_scale() {
        let mut t = Tokenizer::new("Scale -1 1 1");

        assert_eq!(t.next(), Some(Token::new("Scale")));
        assert_eq!(t.next(), Some(Token::new("-1")));
        assert_eq!(t.next(), Some(Token::new("1")));
        assert_eq!(t.next(), Some(Token::new("1")));

        assert_eq!(t.next(), None);
    }
}
