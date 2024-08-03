from typing import Any, Final

import streamlit as st
from streamlit.errors import StreamlitAPIException
from streamlit.delta_generator import DeltaGenerator
from streamlit import type_util


async def write_stream(stream, write_property: str) -> list[Any] | str:
    _TEXT_CURSOR: Final = "▕"

    # Just apply some basic checks for common iterable types that should
    # not be passed in here.
    if isinstance(stream, str) or type_util.is_dataframe_like(stream):
        raise StreamlitAPIException(
            "`st.write_stream` expects a generator or stream-like object as input "
            f"not {type(stream)}. Please use `st.write` instead for "
            "this data type."
        )

    stream_container: DeltaGenerator | None = None
    streamed_response: str = ""
    written_content: list[Any] = []

    def flush_stream_response():
        """Write the full response to the app."""
        nonlocal streamed_response
        nonlocal stream_container

        if streamed_response and stream_container:
            # Replace the stream_container element the full response
            stream_container.markdown(streamed_response)
            written_content.append(streamed_response)
            stream_container = None
            streamed_response = ""

    # Iterate through the generator and write each chunk to the app
    # with a type writer effect.
    with st.spinner():
        async for chunk in stream:  # type: ignore
            if type_util.is_openai_chunk(chunk):
                # Try to convert OpenAI chat completion chunk to a string:
                try:
                    if len(chunk.choices) == 0:
                        # The choices list can be empty. E.g. when using the
                        # AzureOpenAI client, the first chunk will always be empty.
                        chunk = ""
                    else:
                        chunk = chunk.choices[0].delta.content or ""
                except AttributeError as err:
                    raise StreamlitAPIException(
                        "Failed to parse the OpenAI ChatCompletionChunk. "
                        "The most likely cause is a change of the chunk object structure "
                        "due to a recent OpenAI update. You might be able to fix this "
                        "by downgrading the OpenAI library or upgrading Streamlit. Also, "
                        "please report this issue to: https://github.com/streamlit/streamlit/issues."
                    ) from err

            if type_util.is_type(chunk, "langchain_core.messages.ai.AIMessageChunk"):
                # Try to convert LangChain message chunk to a string:
                try:
                    chunk = chunk.content or ""
                except AttributeError as err:
                    raise StreamlitAPIException(
                        "Failed to parse the LangChain AIMessageChunk. "
                        "The most likely cause is a change of the chunk object structure "
                        "due to a recent LangChain update. You might be able to fix this "
                        "by downgrading the OpenAI library or upgrading Streamlit. Also, "
                        "please report this issue to: https://github.com/streamlit/streamlit/issues."
                    ) from err

            if isinstance(chunk, dict):
                if write_property in chunk:
                    chunk = chunk[write_property]
                else:
                    continue

            if isinstance(chunk, str):
                if not chunk:
                    # Empty strings can be ignored
                    continue

                first_text = False
                if not stream_container:
                    stream_container = st.empty()
                    first_text = True
                streamed_response += chunk
                # Only add the streaming symbol on the second text chunk
                stream_container.markdown(
                    streamed_response + ("" if first_text else _TEXT_CURSOR),
                )
            elif callable(chunk):
                flush_stream_response()
                chunk()
            else:
                flush_stream_response()
                st.write(chunk)
                written_content.append(chunk)

    flush_stream_response()

    if not written_content:
        # If nothing was streamed, return an empty string.
        return ""
    elif len(written_content) == 1 and isinstance(written_content[0], str):
        # If the output only contains a single string, return it as a string
        return written_content[0]

    # Otherwise return it as a list of write-compatible objects
    return written_content
