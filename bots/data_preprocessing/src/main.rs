use std::{
    collections::HashMap,
    io::{BufRead, BufReader, BufWriter},
    net::{Ipv4Addr, Ipv6Addr},
    str::FromStr,
};

use chrono::{DateTime, Timelike, Utc};
use serde::{Deserialize, Serialize};

const USER_TEXT_CAP: usize = 50;
const REVISION_TAGS_CAP: usize = 25;
const FILES: [&str; 8] = [
    "../../data/data/edit_types/arwiki.json",
    "../../data/data/edit_types/dewiki.json",
    "../../data/data/edit_types/eswiki.json",
    "../../data/data/edit_types/itwiki.json",
    "../../data/data/edit_types/nlwiki.json",
    "../../data/data/edit_types/plwiki.json",
    "../../data/data/edit_types/ruwiki.json",
    "../../data/data/edit_types/svwiki.json",
];

// Mapping of tag names to group names (from tags_yes only)
fn get_tag_group_mapping() -> HashMap<&'static str, &'static str> {
    let mut map = HashMap::new();

    // mobile_editing
    map.insert("advanced mobile edit", "mobile_editing");
    map.insert("android app edit", "mobile_editing");
    map.insert("ios app edit", "mobile_editing");
    map.insert("mobile app edit", "mobile_editing");
    map.insert("mobile edit", "mobile_editing");
    map.insert("mobile web edit", "mobile_editing");

    // translation_tools
    map.insert("contenttranslation", "translation_tools");
    map.insert("contenttranslation-v2", "translation_tools");
    map.insert("sectiontranslation", "translation_tools");

    // visual_editors
    map.insert("visualeditor", "visual_editors");
    map.insert("visualeditor-switched", "visual_editors");
    map.insert("wikieditor", "visual_editors");
    map.insert("emoji", "visual_editors");

    // edit_checking
    map.insert("editcheck-newcontent", "edit_checking");
    map.insert("editcheck-newreference", "edit_checking");

    // reversion_and_rollback
    map.insert("mw-reverted", "reversion_and_rollback");
    map.insert("mw-rollback", "reversion_and_rollback");
    map.insert("mw-undo", "reversion_and_rollback");
    map.insert("mw-manual-revert", "reversion_and_rollback");

    // bot_assisted_editing
    map.insert("huggle", "bot_assisted_editing");
    map.insert("twinkle", "bot_assisted_editing");

    // other
    map.insert("mw-blank", "other");

    map
}

/// Maps a tag name to its group name if it's in tags_yes, returns None otherwise
fn get_tag_group(tag: &str) -> Option<&'static str> {
    static TAG_MAPPING: std::sync::OnceLock<HashMap<&'static str, &'static str>> =
        std::sync::OnceLock::new();

    let mapping = TAG_MAPPING.get_or_init(get_tag_group_mapping);
    mapping.get(tag).copied()
}

#[derive(Deserialize)]
struct JsonEntry {
    page_id: i64,
    revision_id: i64,
    revision_timestamp: DateTime<Utc>,
    user_text: Option<heapless::String<USER_TEXT_CAP>>,
    revision_tags: Option<heapless::Vec<String, 20>>,
    is_bot: bool,
    revision_comment: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct RevisionTags {
    pub tag_name: String,
    pub times_used: u64,
}

impl RevisionTags {
    fn from_json_entry(
        e: Option<heapless::Vec<String, 20>>,
    ) -> heapless::Vec<Self, REVISION_TAGS_CAP> {
        if let Some(e) = e {
            e.into_iter()
                .filter_map(|tag| {
                    // Only include tags that are in tags_yes
                    get_tag_group(&tag).map(|group| Self {
                        tag_name: group.to_string(),
                        times_used: 1,
                    })
                })
                .collect()
        } else {
            heapless::Vec::new()
        }
    }

    fn reverted(&self) -> bool {
        self.tag_name == "reversion_and_rollback"
    }

    fn reverting(&self) -> bool {
        self.tag_name == "reversion_and_rollback"
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Entity {
    pub user_text_is_ip: bool,
    pub avg_revision_comment_length: u64,
    // pub avg_edit_hours: u32,
    pub revision_tags: heapless::Vec<RevisionTags, REVISION_TAGS_CAP>,
    pub total_edits: u64,
    pub pct_of_reverted_edits: u8,  // percent 0..=100
    pub pct_of_reverting_edits: u8, // percent 0..=100
    pub unique_edited_articles: u64,
    pub unique_edited_languages: u64,
    pub median_seconds_between_edits: u64,
    pub edited_articles: HashMap<i64, u64>, // page_id -> edit_count
    pub edited_languages: HashMap<String, u64>, // language -> edit_count
    #[serde(skip)]
    pub edit_timestamps: Vec<DateTime<Utc>>,
}
impl Entity {
    fn update(&mut self, entry: JsonEntry, language: &str) {
        let new_tags = RevisionTags::from_json_entry(entry.revision_tags);

        // Update running averages using weighted average formula
        let new_comment_len = entry.revision_comment.len() as u64;
        self.avg_revision_comment_length = (self.avg_revision_comment_length * self.total_edits
            + new_comment_len)
            / (self.total_edits + 1);

        let new_hour = entry.revision_timestamp.time().hour() as u64;
        // self.avg_edit_hours = ((self.avg_edit_hours as u64 * self.total_edits + new_hour)
        // / (self.total_edits + 1)) as u32;

        // Merge revision tags - increment count if tag exists, otherwise add it
        for new_tag in new_tags.iter() {
            if let Some(existing_tag) = self
                .revision_tags
                .iter_mut()
                .find(|t| t.tag_name == new_tag.tag_name)
            {
                existing_tag.times_used += 1;
            } else if self.revision_tags.len() < REVISION_TAGS_CAP {
                let _ = self.revision_tags.push(new_tag.clone());
            }
            // If cap is reached and tag doesn't exist, it's silently dropped
        }

        // Update reverted/reverting percentages
        // First, recover approximate counts from current percentages
        let old_reverted_count = (self.pct_of_reverted_edits as u64 * self.total_edits) / 100;
        let new_is_reverted = new_tags.iter().any(RevisionTags::reverted);
        let new_reverted_count = old_reverted_count + if new_is_reverted { 1 } else { 0 };

        let old_reverting_count = (self.pct_of_reverting_edits as u64 * self.total_edits) / 100;
        let new_is_reverting = new_tags.iter().any(RevisionTags::reverting);
        let new_reverting_count = old_reverting_count + if new_is_reverting { 1 } else { 0 };

        // Increment total edits count
        self.total_edits += 1;

        // Recalculate percentages with new totals
        self.pct_of_reverted_edits = ((new_reverted_count * 100) / self.total_edits) as u8;
        self.pct_of_reverting_edits = ((new_reverting_count * 100) / self.total_edits) as u8;

        *self.edited_articles.entry(entry.page_id).or_insert(0) += 1;
        self.unique_edited_articles = self.edited_articles.len() as u64;

        *self
            .edited_languages
            .entry(language.to_string())
            .or_insert(0) += 1;
        self.unique_edited_languages = self.edited_languages.len() as u64;

        self.edit_timestamps.push(entry.revision_timestamp);
        self.edit_timestamps.sort_unstable();
        if self.edit_timestamps.len() >= 2 {
            let mut intervals: Vec<u64> = self
                .edit_timestamps
                .windows(2)
                .map(|w| (w[1] - w[0]).num_seconds().unsigned_abs())
                .collect();
            intervals.sort_unstable();
            let mid = intervals.len() / 2;
            self.median_seconds_between_edits = if intervals.len() % 2 == 0 {
                (intervals[mid - 1] + intervals[mid]) / 2
            } else {
                intervals[mid]
            };
        }
    }
}

impl Entity {
    fn new(entry: JsonEntry, language: &str) -> Self {
        let revision_tags = RevisionTags::from_json_entry(entry.revision_tags);

        let mut edited_articles = HashMap::new();
        edited_articles.insert(entry.page_id, 1);

        let mut edited_languages = HashMap::new();
        edited_languages.insert(language.to_string(), 1);

        let user_text_is_ip = if let Ok(_) =
            Ipv4Addr::from_str(entry.user_text.clone().unwrap().to_string().as_str())
        {
            true
        } else if let Ok(_) =
            Ipv6Addr::from_str(entry.user_text.clone().unwrap().to_string().as_str())
        {
            true
        } else {
            false
        };

        Entity {
            avg_revision_comment_length: entry.revision_comment.len() as u64,
            // avg_edit_hours: entry.revision_timestamp.time().hour(),
            total_edits: 1,
            pct_of_reverted_edits: if revision_tags.iter().any(RevisionTags::reverted) {
                100
            } else {
                0
            },
            pct_of_reverting_edits: if revision_tags.iter().any(RevisionTags::reverting) {
                100
            } else {
                0
            },
            revision_tags,
            unique_edited_articles: 1,
            unique_edited_languages: 1,
            median_seconds_between_edits: 0,
            edited_articles,
            edited_languages,
            edit_timestamps: vec![entry.revision_timestamp],
            user_text_is_ip,
        }
    }
}

fn main() -> anyhow::Result<()> {
    let entities_file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open("./../entity_list.json")?;

    let mut index: HashMap<heapless::String<USER_TEXT_CAP>, Entity> = HashMap::new();

    for file_path in FILES {
        // Extract language code from filename (e.g., "arwiki.json" -> "ar")
        let language = std::path::Path::new(file_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| s.strip_suffix("wiki"))
            .unwrap_or("unknown");

        let file = std::fs::File::open(file_path).unwrap();
        let reader = BufReader::new(file);

        for l in reader.lines() {
            if let Ok(l) = l {
                let l = l.trim();
                if l.is_empty() {
                    continue;
                }

                let entries = if let Ok(entry) = serde_json::from_str::<JsonEntry>(l) {
                    vec![entry]
                } else if let Ok(entries) = serde_json::from_str::<Vec<JsonEntry>>(l) {
                    entries
                } else {
                    // Ignore non-entry lines (e.g., metadata/wrappers/malformed rows).
                    continue;
                };

                for entry in entries {
                    if let Some(user_text) = entry.user_text.clone()
                        && !user_text.trim().is_empty()
                    {
                        if let Some(index_entry) = index.get_mut(&user_text) {
                            index_entry.update(entry, language)
                        } else {
                            let index_entry = Entity::new(entry, language);
                            let _ = index.insert(user_text, index_entry);
                        }
                    }
                }
            }
        }
    }

    let writer = BufWriter::new(entities_file);
    serde_json::to_writer(writer, &index)?;

    Ok(())
}
