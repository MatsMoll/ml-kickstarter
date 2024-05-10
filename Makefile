

.PHONY: up
infra-up: 
	docker compose up aligned-catalog mlflow-tracker

.PHONY: train
train: 
	docker compose run trainer

.PHONY: ollama
ollama:
	@docker compose up ollama -d ; sleep 5 ; \
	CONTAINER_ID=$$(docker ps | grep ollama | awk '{print $$1}') ; \
	echo "Container ID: $$CONTAINER_ID" ; \
	docker exec $$CONTAINER_ID ollama pull nomic-embed-text

.PHONY: models
models-up:
	docker compose up movie_review_is_negative wine-model
