from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_production_and_local_images_run_as_unprivileged_user():
    for filename in ("Dockerfile", "Dockerfile.render"):
        content = (ROOT / filename).read_text(encoding="utf-8")
        assert "USER app" in content
        assert "adduser --system" in content
        assert "chown -R app:app /app/chat_history /app/.cache /home/app" in content

        if filename == "Dockerfile.render":
            assert "curl -f http://127.0.0.1:${PORT:-8000}/ready" in content
            assert "uvicorn app.api_main:app" in content
            assert "COPY --from=frontend-build /frontend/dist /app/frontend/dist" in content
        else:
            # The legacy image remains available for rollback during migration.
            assert "curl -f http://127.0.0.1:${PORT:-8501}/_stcore/health" in content
            assert "streamlit run app/streamlit_app.py" in content


def test_container_writable_paths_are_explicitly_created():
    for filename in ("Dockerfile", "Dockerfile.render"):
        content = (ROOT / filename).read_text(encoding="utf-8")
        assert "/app/chat_history" in content
        assert "/app/.cache/huggingface" in content
        assert "/home/app" in content
