from __future__ import annotations

from scripts.backfill_graph_entities import backfill_entity_terms, build_backfill_query


def test_backfill_query_can_be_limited_to_one_owner_and_document_ids():
    query = build_backfill_query(
        owner_scope="a" * 64,
        document_ids=["doc-a", "doc-b", "doc-a"],
    )

    filters = query["query"]["bool"]["filter"]
    assert {"exists": {"field": "passage_text"}} in filters
    assert {"term": {"owner_scope": "a" * 64}} in filters
    assert {"terms": {"document_id": ["doc-a", "doc-b"]}} in filters


def test_backfill_updates_only_legacy_chunks_and_never_logs_or_returns_text(monkeypatch):
    from opensearchpy import helpers

    scanned = [
        {"_id": "old:1", "_source": {
            "passage_text": "Northbridge Holdings references FDA controls.",
        }},
        {"_id": "new:1", "_source": {
            "passage_text": "Acme Corporation is a provider.",
            "entity_terms": ["acme corporation"],
        }},
        {"_id": "plain:1", "_source": {
            "passage_text": "Payment is due within 30 days.",
        }},
    ]
    actions = []

    monkeypatch.setattr(helpers, "scan", lambda *args, **kwargs: iter(scanned))

    def fake_bulk(client, batch, **kwargs):
        actions.extend(batch)
        return len(batch), []

    monkeypatch.setattr(helpers, "bulk", fake_bulk)

    result = backfill_entity_terms(
        object(),
        "tenant-index",
        batch_size=2,
        dry_run=False,
    )

    assert result == {
        "scanned": 3,
        "eligible": 1,
        "updated": 1,
        "skipped_existing": 1,
        "without_entities": 1,
    }
    assert len(actions) == 1
    assert actions[0]["_op_type"] == "update"
    assert actions[0]["_id"] == "old:1"
    assert "northbridge holdings" in actions[0]["doc"]["entity_terms"]
    assert "passage_text" not in actions[0]["doc"]


def test_backfill_dry_run_never_writes_to_opensearch(monkeypatch):
    from opensearchpy import helpers

    scanned = [{
        "_id": "old:1",
        "_source": {"passage_text": "Northbridge Holdings references FDA."},
    }]
    monkeypatch.setattr(helpers, "scan", lambda *args, **kwargs: iter(scanned))

    def unexpected_bulk(*args, **kwargs):
        raise AssertionError("Dry-run must not modify the index.")

    monkeypatch.setattr(helpers, "bulk", unexpected_bulk)

    result = backfill_entity_terms(
        object(),
        "tenant-index",
        dry_run=True,
    )

    assert result["scanned"] == 1
    assert result["eligible"] == 1
    assert result["updated"] == 0




def test_backfill_cli_updates_existing_mapping_without_deploying_sparse_models(monkeypatch):
    from scripts import backfill_graph_entities as command

    events = []

    class Indices:
        def exists(self, *, index):
            events.append(("exists", index))
            return True

    class Client:
        indices = Indices()

    monkeypatch.setattr(command.rag_pipeline, "build_opensearch_client", lambda: Client())
    monkeypatch.setattr(command.rag_pipeline, "opensearch_index_name", lambda *args: "tenant-index")
    monkeypatch.setattr(
        command.rag_pipeline,
        "ensure_opensearch_index",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Backfill must not trigger unrelated sparse-model registration.")
        ),
    )
    monkeypatch.setattr(
        command.rag_pipeline,
        "_ensure_entity_mapping",
        lambda client, index: events.append(("mapping", index)),
    )
    monkeypatch.setattr(
        command,
        "backfill_entity_terms",
        lambda *args, **kwargs: {"scanned": 0, "eligible": 0, "updated": 0, "skipped_existing": 0, "without_entities": 0},
    )

    assert command.main([]) == 0
    assert events == [("exists", "tenant-index"), ("mapping", "tenant-index")]
