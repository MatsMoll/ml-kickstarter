
.PHONY: clean
clean:
	docker system prune -f

.PHONY: build
build:
	docker compose build

.PHONY: infra-up
infra-up:
	docker compose up aligned-catalog mlflow-tracker prefect-server pipeline-worker

.PHONY: ollama
ollama:
	@docker compose up ollama -d ; sleep 5 ; \
	CONTAINER_ID=$$(docker ps | grep ollama | awk '{print $$1}') ; \
	echo "Container ID: $$CONTAINER_ID" ; \
	docker exec $$CONTAINER_ID ollama pull nomic-embed-text

.PHONY: models-up
models-up:
	docker compose up movie-review-is-negative wine-model

.PHONY: test
test:
	docker compose run test-action
