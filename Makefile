

.PHONY: infra-up
infra-up:
	docker compose --profile infra up

.PHONY: build
build:
	docker compose build

.PHONY: clean
clean:
	docker system prune -f

.PHONY: ollama
ollama:
	@docker compose up ollama -d ; sleep 5 ; \
	CONTAINER_ID=$$(docker ps | grep ollama | awk '{print $$1}') ; \
	echo "Container ID: $$CONTAINER_ID" ; \
	docker exec $$CONTAINER_ID ollama pull nomic-embed-text

.PHONY: models-up
models-up:
	docker compose --profile model up

.PHONY: test
test:
	docker compose run test-action
