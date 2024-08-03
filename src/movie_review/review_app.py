import asyncio
import streamlit as st
from src.load_store import load_store


async def main() -> None:
    with st.spinner("Loading"):
        store = await load_store()

    st.title("Write a movie review")

    with st.form("Review form"):
        review = st.text_area("Review")

        st.form_submit_button()

    if not review:
        st.warning("Need to write a review")
        return

    pred = (
        await store.model("movie_review_is_negative")
        .predict_over({"text": [review], "review_id": ["ddddd"]})
        .to_polars()
    ).to_dicts()[0]

    if pred["is_negative_pred"]:
        st.error("It was a negative review")
    else:
        st.success("It is a positive review")


if __name__ == "__main__":
    asyncio.run(main())
