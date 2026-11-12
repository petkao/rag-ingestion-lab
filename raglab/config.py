"""Central settings. Every knob that changes ingestion quality lives here so you can experiment."""

from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"          # everything ingestion produces
CHROMA_DIR = OUTPUT_DIR / "chroma"            # the vector store (persistent, on disk)
ENRICH_CACHE = OUTPUT_DIR / "enrich_cache.json"  # LLM captions/summaries, so re-runs are free

# --- Models -------------------------------------------------------------
CLAUDE_MODEL = "claude-opus-5-5"              # vision captions, table summaries, RAG agent
EMBED_MODEL = "text-embedding-3-small"        # OpenAI, 1536 dims (same as ad-engine)

# --- Parsing ------------------------------------------------------------
MIN_IMAGE_PX = 64          # skip icons, bullets, spacer images smaller than this (either side)
HEADING_SIZE_RATIO = 1.15  # a span this much larger than the body font size counts as a heading

# --- Chunking -----------------------------------------------------------
CHUNK_MAX_CHARS = 1200     # ~300 tokens. Smaller = precise but loses context; larger = the opposite
CHUNK_OVERLAP_CHARS = 200  # tail of the previous chunk repeated at the start of the next

# --- Retrieval ----------------------------------------------------------
COLLECTION = "pdf_chunks"
DEFAULT_TOP_K = 5


def doc_output_dir(pdf_path: Path) -> Path:
    """output/<pdf-stem>/ holds the intermediate JSON of each stage, for inspection."""
    d = OUTPUT_DIR / pdf_path.stem
    (d / "images").mkdir(parents=True, exist_ok=True)
    return d
