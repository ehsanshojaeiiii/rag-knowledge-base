import json
from pathlib import Path

from rag_kb.main import app

SPEC = Path("docs/openapi.json")
REGENERATE = (
    'uv run python -c "import json; from rag_kb.main import app; '
    'print(json.dumps(app.openapi(), indent=2, ensure_ascii=False))" > docs/openapi.json'
)


def test_committed_openapi_spec_matches_the_app() -> None:
    """docs/openapi.json is what reviewers import into Postman; it must describe the real API."""
    committed = json.loads(SPEC.read_text(encoding="utf-8"))
    assert app.openapi() == committed, f"Regenerate: {REGENERATE}"
