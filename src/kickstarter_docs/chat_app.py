import asyncio

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Protocol
from aligned.exposed_model.interface import StreamablePredictor
from aligned.feature_store import ModelFeatureStore
from aligned.schemas.feature import FeatureType
from src.load_store import load_store

try:
    import streamlit as st
    from streamlit.delta_generator import DeltaGenerator
except ImportError:
    st = None
    DeltaGenerator = None


@dataclass
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str


class ChatStore(Protocol):
    def read(self) -> list[ChatMessage]: ...

    def add(self, message: ChatMessage) -> None: ...


class StreamlitChatStore(ChatStore):
    cache_key: str = field(default="messages")

    def read(self) -> list[ChatMessage]:
        if self.cache_key not in st.session_state:
            st.session_state[self.cache_key] = []
        return st.session_state[self.cache_key]

    def add(self, message: ChatMessage) -> None:
        if self.cache_key not in st.session_state:
            st.session_state[self.cache_key] = []
        st.session_state[self.cache_key].append(message)


async def sidebar(col: DeltaGenerator, model_contract: ModelFeatureStore) -> None:
    col.title("About")
    col.write(
        "This is a simple chatbot that understands understands the Python and Markdown files in the project directory."
    )
    col.write(
        f"To view the code. Checkout the `{model_contract.model.name}` model contract."
    )
    freshness = await model_contract.freshness()
    now = datetime.now()

    for loc, fresh in freshness.items():
        if loc.location == "model":
            # Not good, but oh well
            expected = model_contract.store.model(
                loc.name
            ).model.predictions_view.acceptable_freshness
        else:
            expected = model_contract.store.feature_view(
                loc.name
            ).view.acceptable_freshness

        if expected and fresh:
            dt_since_last = now.replace(tzinfo=fresh.tzinfo) - fresh

            if dt_since_last <= expected:
                col.success(f"Data for `{loc.name}` is up to date.")
            else:
                col.error(
                    f"Data for `{loc.name}` is {dt_since_last} old. This needs to be updated."
                )
        else:
            col.write(f"`{loc}` was last updated at {fresh}")


async def run(chat_store: ChatStore):
    store = await load_store()

    model_contract = store.model("kickstarter-docs-questions")
    exposed_model = model_contract.model.exposed_model

    if exposed_model is None:
        st.warning("Found no model to query")
        return

    await sidebar(st.sidebar, model_contract)

    st.title("Kickstarter Docs Chat")

    # Display chat messages from history on app rerun
    current_messages = chat_store.read()
    for message in current_messages:
        with st.chat_message(message.role):
            st.markdown(message.content)

    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)

        # Add user message to chat history
        chat_store.add(ChatMessage(role="user", content=prompt))

        response = await display_response(model_contract, prompt)
        chat_store.add(response)


async def display_response(contract: ModelFeatureStore, prompt: str) -> ChatMessage:
    from streamlit_components.async_stream import write_stream

    exposed_model = contract.model.exposed_model
    assert exposed_model is not None

    response_keys = [
        feature.name
        for feature in contract.model.predictions_view.features
        if feature.dtype == FeatureType.string() and feature.constraints is None
    ]
    assert len(response_keys) == 1, f"Found {response_keys}"

    response_key = response_keys[0]

    if isinstance(exposed_model, StreamablePredictor):
        with st.chat_message("assistant"):
            response = await write_stream(
                exposed_model.stream_predict({"full_prompt": prompt, "query": prompt}),
                write_property=response_key,
            )
    else:
        with st.spinner("Querying the LLM"):
            response_df = await contract.predict_over(
                {"full_prompt": [prompt], "query": [prompt]}
            ).to_polars()

        response = response_df.to_dicts()[0][response_key]

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)

    return ChatMessage(role="assistant", content=response)


if __name__ == "__main__":
    asyncio.run(run(chat_store=StreamlitChatStore()))
