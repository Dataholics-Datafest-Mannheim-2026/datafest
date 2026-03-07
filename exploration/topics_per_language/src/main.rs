use std::{
    collections::HashMap,
    io::{BufRead, BufReader},
    ops::AddAssign,
};

use serde::{Deserialize, Serialize};

const FILE_PATH: &str = "../../data/data/page_views.json";

#[derive(Serialize, Deserialize)]
struct JsonEntry {
    pageviews: u64,
    #[serde(flatten)]
    key: Key,
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Hash)]
struct Key {
    page_id: u64,
    wiki_db: heapless::String<10>,
}

fn main() -> anyhow::Result<()> {
    let file = std::fs::File::open(FILE_PATH).unwrap();
    let reader = BufReader::new(file);

    let mut index: HashMap<Key, u64, _> = HashMap::new();

    for l in reader.lines() {
        if let Ok(l) = l {
            let s: JsonEntry = serde_json::from_str(&l)?;

            if let Some(e) = index.get_mut(&s.key) {
                e.add_assign(s.pageviews);
            } else {
                let _ = index.insert(s.key, s.pageviews);
            }
        }
    }

    let index: Vec<_> = index
        .into_iter()
        .map(|e| JsonEntry {
            pageviews: e.1,
            key: e.0,
        })
        .collect();

    let res = serde_json::to_string(&index)?;

    std::fs::write("result.json", res)?;

    Ok(())
}
