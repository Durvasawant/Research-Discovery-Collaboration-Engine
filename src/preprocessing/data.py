"""Dataset loading, validation and researcher-profile construction."""
from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd

REQUIRED = ["faculty_id", "faculty_name", "department", "research_interests", "paper_title", "abstract", "keywords"]
OPTIONAL = ["areas"]
PROFILE_MODES = ("full", "interests_only", "titles_keywords")


@dataclass
class Paper:
    title: str
    abstract: str
    keywords: List[str]

    @property
    def text(self) -> str:
        return f"{self.title}. {self.abstract}. {', '.join(self.keywords)}"


@dataclass
class Researcher:
    faculty_id: str
    name: str
    department: str
    interests: str
    papers: List[Paper] = field(default_factory=list)
    areas: Optional[set] = None

    def documents(self, mode: str = "full") -> List[str]:
        """The individual text units that describe this researcher (interests + papers)."""
        if mode == "interests_only":
            return [self.interests]
        if mode == "titles_keywords":
            return [self.interests] + [f"{p.title}. {', '.join(p.keywords)}" for p in self.papers]
        return [self.interests] + [p.text for p in self.papers]

    def profile_text(self, mode: str = "full") -> str:
        """One concatenated document. Interests and curated keywords are repeated once to up-weight them."""
        if mode == "interests_only":
            return self.interests
        parts = [self.interests, self.interests]
        for p in self.papers:
            kws = ", ".join(p.keywords)
            if mode == "titles_keywords":
                parts += [p.title, kws, kws]
            else:
                parts += [p.title, p.abstract, kws, kws]
        return " . ".join(parts)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}. Required: {REQUIRED}")
    df = df.copy()
    for c in REQUIRED:
        df[c] = df[c].fillna("").astype(str).str.strip()
    df = df[df.faculty_id != ""]
    df = df.drop_duplicates(subset=["faculty_id", "paper_title", "abstract"])
    return df.reset_index(drop=True)


def load_dataframe(path: str) -> pd.DataFrame:
    return validate(pd.read_csv(path))


def split_keywords(s: str) -> List[str]:
    return [k.strip() for k in str(s).replace(",", ";").split(";") if k.strip()]


def build_researchers(df: pd.DataFrame) -> List[Researcher]:
    """Group the per-paper table into one Researcher per faculty_id (stable, sorted by id)."""
    out = []
    for fid, g in df.groupby("faculty_id", sort=True):
        first = g.iloc[0]
        papers = [Paper(r.paper_title, r.abstract, split_keywords(r.keywords))
                  for r in g.itertuples() if (r.paper_title or r.abstract)]
        areas = None
        if "areas" in g.columns and str(first.get("areas", "")).strip():
            areas = {a.strip() for a in str(first["areas"]).split(";") if a.strip()}
        out.append(Researcher(fid, first.faculty_name, first.department, first.research_interests, papers, areas))
    return out


def dataset_stats(df: pd.DataFrame) -> dict:
    """Numbers required by the handout's dataset documentation (section 8)."""
    wc = df.abstract.str.split().str.len()
    return {
        "faculty": int(df.faculty_id.nunique()),
        "paper_rows": int(len(df)),
        "departments": int(df.department.nunique()),
        "papers_per_faculty_mean": round(len(df) / max(df.faculty_id.nunique(), 1), 2),
        "abstract_words_mean": round(float(wc.mean()), 1),
        "abstract_words_min": int(wc.min()),
        "abstract_words_max": int(wc.max()),
        "empty_abstracts": int((df.abstract == "").sum()),
        "empty_keywords": int((df.keywords == "").sum()),
    }
