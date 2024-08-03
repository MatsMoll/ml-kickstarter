from datetime import timedelta
from aligned import (
    EventTimestamp,
    FileSource,
    Int32,
    model_contract,
    feature_view,
    String,
    UInt64,
    CustomMethodDataSource,
)
from aligned.sources.lancedb import LanceDBConfig
from aligned.exposed_model.langchain import LangChain
from aligned.exposed_model.ollama import ollama_embedding_contract
from aligned.data_source.batch_data_source import DummyDataSource
from src.kickstarter_docs.load_chunks import parse_code_chunks

vector_db = LanceDBConfig("data/lancedb")


@feature_view(
    name="module_chunks",
    source=CustomMethodDataSource.from_load(parse_code_chunks).with_loaded_at(),
    materialized_source=FileSource.csv_at("data/module_chunks.csv"),
    acceptable_freshness=timedelta(minutes=15),
    unacceptable_freshness=timedelta(days=1),
)
class ModuleChunk:
    id = String().as_entity()
    source = String()
    lineno = Int32()
    name = String()

    source_type = String().accepted_values(
        [
            "docs",
            "import",
            "class",
            "function",
            "attribute",
        ]
    )
    content = String().is_optional()
    loaded_at = EventTimestamp()


docs = ModuleChunk()

ollama_api = "http://host.docker.internal:11434"

KickstarterDocsEmbedding = ollama_embedding_contract(
    contract_name="kickstarter_docs_embedding",
    model="nomic-embed-text",
    entities=docs.id,
    input=[docs.source, docs.source_type, docs.lineno, docs.content, docs.name],
    output_source=vector_db.table("kickstarter-docs").as_vector_index(
        name="kickstater-docs"
    ),
    endpoint=ollama_api,
    prompt_template="Type: '{source_type}': named: '{name}' at line nr. '{lineno}' in '{source}'\n\nBehavior: {content}",
    model_version_field=String().with_name("prompt_version"),
    acceptable_freshness=timedelta(minutes=15),
)


def rag_chain():
    from langchain_community.llms import Ollama
    from langchain.chains import RetrievalQA

    llm = Ollama(model="llama2", base_url=ollama_api)

    docs_context = KickstarterDocsEmbedding.as_langchain_retriver(
        number_of_docs=15,
        # The embedding is referencing to a doc chunk
        # Therefore, it is currently needed to add this
        needed_views=[ModuleChunk],
    )
    return RetrievalQA.from_chain_type(
        llm=llm, retriever=docs_context, return_source_documents=True
    )


@feature_view(name="kickstarter-docs-question", source=DummyDataSource())
class KickstarterQAInputFormat:
    query = String()
    question_id = UInt64().as_entity()


question = KickstarterQAInputFormat()


@model_contract(
    name="kickstarter-docs-questions",
    input_features=[question.query],
    exposed_model=LangChain.from_chain(
        rag_chain(),
        chain_output="result",
        output_key="content",
        depends_on=[KickstarterDocsEmbedding],
    ),
)
class KickstarterDocsQuestionAnswer:
    content = String()
    responded_at = EventTimestamp()
