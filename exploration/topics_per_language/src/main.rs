use std::{
    collections::HashMap,
    io::{BufRead, BufReader},
    ops::AddAssign,
};

use serde::Deserialize;

const FILE_PATH: &str = "./../page_views.json";

#[derive(Deserialize)]
struct JsonEntry {
    pageviews: u64,
    page_id: u64,
    wiki_db: heapless::String<10>,
}

fn main() -> anyhow::Result<()> {
    let file = std::fs::File::open(FILE_PATH).unwrap();
    let reader = BufReader::new(file);

    let mut index: HashMap<(u64, heapless::String<10>), u64, _> = HashMap::new();

    for l in reader.lines() {
        if let Ok(l) = l {
            let s: JsonEntry = serde_json::from_str(&l)?;

            if let Some(e) = index.get_mut(&(s.page_id, s.wiki_db.clone())) {
                e.add_assign(s.pageviews);
            } else {
                let _ = index.insert((s.page_id, s.wiki_db), s.pageviews);
            }
        }
    }

    let index: HashMap<_, _> = index
        .iter()
        .map(|e| (format!("{}: {}", e.0.0, e.0.1), e.1))
        .collect();

    let res = serde_json::to_string(&index)?;

    std::fs::write("result.json", res)?;

    Ok(())
}
