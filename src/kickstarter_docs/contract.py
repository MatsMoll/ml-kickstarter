import ast
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

import polars as pl

vector_db = LanceDBConfig("data/lancedb")


def chunks_from(stm: ast.stmt, hir: list[str]) -> list[dict]:
    import ast

    def signature(function: ast.AsyncFunctionDef | ast.FunctionDef) -> str:
        signature = ""
        if function.args.args:
            args = ", ".join(
                [
                    f"{arg.arg} of type '{ast.unparse(arg.annotation)}'"
                    if arg.annotation
                    else arg.arg
                    for arg in function.args.args
                ]
            )
            signature += f"input: {args}"

        if function.returns:
            signature += f" returns: {ast.unparse(function.returns)}"

        return signature

    if isinstance(stm, ast.ClassDef):
        cur_hist = hir + [stm.name]

        components = []
        sub_content = [f"Subclassing: {ast.unparse(base)}" for base in stm.bases]
        sub_content.extend(
            [f"Decorated: {ast.unparse(dec)}" for dec in stm.decorator_list]
        )

        if isinstance(stm.body[0], ast.Expr) and isinstance(
            stm.body[0].value, ast.Constant
        ):
            sub_content.append(ast.unparse(stm.body[0].value))

        for sub in stm.body:
            if isinstance(sub, ast.AsyncFunctionDef) or isinstance(
                sub, ast.FunctionDef
            ):
                sub_content.append(f"Function: {sub.name}. Signature: {signature(sub)}")
            elif isinstance(sub, ast.Assign):
                sub_content.extend(
                    [
                        f"Attribute: {name.id}"
                        for name in sub.targets
                        if isinstance(name, ast.Name)
                    ]
                )
            else:
                components.extend(chunks_from(sub, cur_hist))

        components.append(
            {
                "id": ".".join(cur_hist),
                "source": ".".join(cur_hist),
                "source_type": "class",
                "lineno": stm.lineno,
                "name": stm.name,
                "content": "\n".join(sub_content),
            }
        )
        return components
    elif isinstance(stm, ast.FunctionDef) or isinstance(stm, ast.AsyncFunctionDef):
        sub_content = [f"Signature: {signature(stm)}"]
        sub_content.extend(
            [f"Decorated: {ast.unparse(dec)}" for dec in stm.decorator_list]
        )
        if isinstance(stm.body[0], ast.Expr) and isinstance(
            stm.body[0].value, ast.Constant
        ):
            sub_content.append(ast.unparse(stm.body[0].value))

        cur_hist = hir + [stm.name]
        return [
            {
                "id": ".".join(cur_hist),
                "source": ".".join(cur_hist),
                "source_type": "function",
                "lineno": stm.lineno,
                "name": stm.name,
                "content": "\n".join(sub_content),
            }
        ]
    else:
        return []


def parse_md_file(file: str, hir: list[str]) -> list[dict]:
    """
    Parsing a file into different components based on it's format.
    """
    # First level will be the file name -> find #
    section_level = len(hir)
    next_section = "#" * section_level

    id = "/".join(hir)

    if file.startswith(f"{next_section} "):
        splits = file.split("\n", maxsplit=1)
        assert len(splits) == 2, splits
        line, rest = splits
        source = {
            "id": id,
            "source": id + "/" + line,
            "content": rest,
            "name": line,
            "lineno": 0,
            "source_type": "docs",
        }
        return [source] + parse_md_file(rest, hir + [line])

    if f"\n{next_section} " not in file:
        return []

    all_sections = []
    for section in file.split(f"\n{next_section} "):
        splits = section.split("\n", maxsplit=1)

        if len(splits) == 1:
            continue

        assert len(splits) == 2, splits
        line, rest = splits
        source = {
            "id": id,
            "source": id + "/" + line,
            "content": rest,
            "lineno": 0,
            "name": line,
            "source_type": "docs",
        }
        all_sections.extend([source] + parse_md_file(rest, hir + [line]))
    return all_sections


async def parse_code_chunks(request) -> pl.LazyFrame:
    from pathlib import Path
    import ast

    components = []
    files = Path("./src").glob("**/*.py")
    for file in files:
        module_name = file.as_posix().replace("/", ".").replace(".py", "")
        tree = ast.parse(file.read_text(), filename=file.name)

        for stm in tree.body:
            components.extend(chunks_from(stm, [module_name]))

    md_root = Path(".")
    for file in list(md_root.glob("**/*.md")) + list(md_root.glob("*.md")):
        components.extend(parse_md_file(file.read_text(), [file.as_posix()]))

    return pl.DataFrame(components).lazy()


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
    responeded_at = EventTimestamp()
