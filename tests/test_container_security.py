from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_production_and_local_images_run_api_only_as_unprivileged_user():
    for filename in ("Dockerfile", "Dockerfile.render"):
        content = (ROOT / filename).read_text(encoding="utf-8")
        assert "USER app" in content
        assert "adduser --system" in content
        assert "uvicorn app.api_main:app" in content
        assert "/ready" in content
        assert "streamlit run" not in content
        assert "frontend/dist" not in content


def test_container_writable_paths_are_explicitly_created():
    for filename in ("Dockerfile", "Dockerfile.render"):
        content = (ROOT / filename).read_text(encoding="utf-8")
        assert "/app/chat_history" in content
        assert "/app/.cache/huggingface" in content
        assert "/home/app" in content
