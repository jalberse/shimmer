use itertools::Itertools;
use log::warn;

use crate::Float;

use std::fs;

pub fn read_float_file(filename: &str) -> Vec<Float> {
    // PAPERDOC - This is a bit more robust (the parse() is better) than PBRT's version,
    //  because of the ease of the parse() function. It's easier to tell it's floats separated by whitespace easily.
    let contents = fs::read_to_string(filename);
    match contents {
        Ok(s) => s
            .split_whitespace()
            .map(|t| {
                let float_val: Float = t.parse().expect("Unable to parse float value!");
                float_val
            })
            .collect_vec(),
        Err(_) => {
            warn!("Erroor reading this file! {}", filename);
            Vec::new()
        }
    }
}
