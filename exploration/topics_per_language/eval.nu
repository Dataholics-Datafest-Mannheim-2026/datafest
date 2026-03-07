let page_info = open --raw ../../data/data/page_info.json | from json --objects
def get_title []: int -> string {
  try { $page_info | where page_id == $in | first | page_title } catch { null }
}

let grouped = open result.json | group-by wiki_db --to-table | update items {|r| $r.items | select pageviews page_id | sort-by pageviews | last 10 | insert page_title {|e| $e.page_id | get_title }}

$grouped | save -f ./topics_per_language.yaml
