"""Collect REAL faculty publication data from OpenAlex (free, public, CC0 metadata) -- UNTESTED OFFLINE, test on your machine.

1. Fill data/faculty_input.csv with columns:  faculty_id, faculty_name, department, research_interests, openalex_author_id
   (find the id at https://openalex.org/ -> search the author -> URL ends in A1234567890; or leave blank and use --search)
2. python scripts/collect_openalex.py --max-papers 8 --email you@nmims.edu
3. Output: data/faculty_research.csv  (the schema the engine expects). MANUALLY CHECK it: author disambiguation errors happen.

Privacy: only public bibliographic metadata (titles/abstracts/keywords/venue year) is collected -- no student data.
Use faculty names only with permission or replace names with IDs before sharing outside the team.
"""
import argparse, csv, json, os, time, urllib.parse, urllib.request

API = "https://api.openalex.org"


def get(url, email):
    req = urllib.request.Request(url + ("&" if "?" in url else "?") + "mailto=" + urllib.parse.quote(email),
                                 headers={"User-Agent": f"SNLP-Group7 ({email})"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def abstract_from_inverted(inv):
    if not inv:
        return ""
    pos = {}
    for w, idxs in inv.items():
        for i in idxs:
            pos[i] = w
    return " ".join(pos[i] for i in sorted(pos))


def find_author(name, email):
    res = get(f"{API}/authors?search={urllib.parse.quote(name)}&per-page=5", email)["results"]
    for r in res:
        print(f"   candidate: {r['id']}  {r['display_name']}  works={r['works_count']}  inst={[i['display_name'] for i in r.get('last_known_institutions', [])][:1]}")
    return res[0]["id"].split("/")[-1] if res else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/faculty_input.csv"); ap.add_argument("--output", default="data/faculty_research.csv")
    ap.add_argument("--max-papers", type=int, default=8); ap.add_argument("--email", required=True); ap.add_argument("--search", action="store_true")
    a = ap.parse_args()
    rows = []
    for f in csv.DictReader(open(a.input, encoding="utf-8")):
        aid = (f.get("openalex_author_id") or "").strip()
        if not aid and a.search:
            print("searching", f["faculty_name"]); aid = find_author(f["faculty_name"], a.email)
        if not aid:
            print("!! no author id for", f["faculty_id"]); continue
        url = (f"{API}/works?filter=author.id:{aid},has_abstract:true&sort=publication_year:desc&per-page={a.max_papers}"
               "&select=id,title,abstract_inverted_index,keywords,topics,publication_year")
        for w in get(url, a.email)["results"]:
            kws = [k["display_name"] for k in w.get("keywords", [])] or [t["display_name"] for t in w.get("topics", [])][:4]
            rows.append({"faculty_id": f["faculty_id"], "faculty_name": f["faculty_name"], "department": f["department"],
                         "research_interests": f["research_interests"], "paper_title": w["title"],
                         "abstract": abstract_from_inverted(w.get("abstract_inverted_index")), "keywords": "; ".join(kws), "areas": ""})
        time.sleep(0.2)
    with open(a.output, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["faculty_id", "faculty_name", "department", "research_interests", "paper_title", "abstract", "keywords", "areas"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} papers to {a.output}")


if __name__ == "__main__":
    main()
