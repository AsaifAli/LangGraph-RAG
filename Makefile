# Run project commands inside the Docker app container so the host does not need Python.
.PHONY: test compile

test:
	docker compose exec app pytest -q

compile:
	docker compose exec app python -m compileall -q agents app citations config evaluation retrieval scripts shared tools
