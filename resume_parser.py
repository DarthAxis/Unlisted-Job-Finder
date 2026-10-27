"""Resume parsing: PDF -> text -> skills and target job titles (100% local)."""
import io
import re
from collections import Counter
from dataclasses import dataclass, field

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class ResumeError(Exception):
    """Raised with a user-friendly message when a resume can't be processed."""


SKILL_TERMS = [
    # Languages / data
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "ruby", "php", "scala", "r",
    "sql", "nosql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "snowflake", "bigquery",
    "spark", "hadoop", "kafka", "airflow", "dbt", "etl", "data modeling", "data warehouse", "data pipeline",
    # Web / cloud / devops
    "react", "angular", "vue", "node.js", "django", "flask", "fastapi", "graphql", "rest api", "html", "css",
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ci/cd", "jenkins", "git", "linux",
    # ML / analytics
    "machine learning", "deep learning", "nlp", "llm", "pytorch", "tensorflow", "scikit-learn", "pandas",
    "numpy", "statistics", "a/b testing", "data analysis", "data visualization", "tableau", "power bi", "looker",
    "excel", "data science",
    # Information / knowledge management
    "metadata", "metadata management", "taxonomy", "ontology", "information architecture", "knowledge graph",
    "knowledge management", "content strategy", "content modeling", "data governance", "data catalog",
    "digital asset management", "cataloging", "controlled vocabulary", "schema.org", "rdf", "sparql", "xml",
    "json", "search relevance", "seo", "cms", "dita", "technical writing", "document management",
    # Product / design / business
    "product management", "roadmap", "user research", "ux", "ui design", "figma", "wireframing", "agile",
    "scrum", "jira", "confluence", "stakeholder management", "project management", "program management",
    "business analysis", "requirements gathering", "process improvement", "salesforce", "sap", "crm",
    "marketing", "analytics", "google analytics", "customer success", "sales", "accounting", "budgeting",
    "compliance", "risk management", "leadership", "communication", "cross-functional",
]


def _term_regex(term: str) -> re.Pattern:
    return re.compile(r"(?<![\w+#])" + re.escape(term) + r"(?![\w+#])", re.IGNORECASE)


_SKILL_PATTERNS = {t: _term_regex(t) for t in SKILL_TERMS}

_TITLE_RE = re.compile(
    r"\b((?:Senior|Junior|Lead|Principal|Staff|Associate|Head of|Chief)?\s?"
    r"(?:[A-Z][\w/&\-]*\s){0,3}"
    r"(?:Engineer|Developer|Architect|Analyst|Manager|Designer|Scientist|Librarian|Specialist|"
    r"Consultant|Strategist|Director|Administrator|Coordinator|Researcher|Writer|Editor|Curator|Archivist))\b"
)


@dataclass
class ResumeProfile:
    text: str = ""
    skills: list = field(default_factory=list)
    titles: list = field(default_factory=list)


def extract_text(file_obj) -> str:
    """Extract text from an uploaded PDF. Raises ResumeError on any problem."""
    try:
        data = file_obj.read() if hasattr(file_obj, "read") else file_obj
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                raise ResumeError("This PDF is password-protected. Please upload an unlocked copy.")
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    except ResumeError:
        raise
    except (PdfReadError, ValueError, KeyError, OSError):
        raise ResumeError("That file doesn't look like a valid PDF. Please try re-exporting your resume as PDF.")
    except Exception as exc:  # noqa: BLE001
        raise ResumeError(f"Couldn't read the PDF ({type(exc).__name__}). Try another file.")
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) < 80:
        raise ResumeError("I couldn't find readable text. Scanned/image-only PDFs aren't supported; "
                          "please upload a text-based PDF.")
    return text


def extract_skills(text: str) -> list:
    found = []
    for term, pat in _SKILL_PATTERNS.items():
        if pat.search(text):
            found.append(term)
    return found


def extract_titles(text: str, limit: int = 6) -> list:
    counts = Counter(m.group(1).strip() for m in _TITLE_RE.finditer(text))
    return [t for t, _ in counts.most_common(limit)]


def parse_resume(file_obj) -> ResumeProfile:
    text = extract_text(file_obj)
    return ResumeProfile(text=text, skills=extract_skills(text), titles=extract_titles(text))
