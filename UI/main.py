import streamlit as st
import sys
from bs4 import BeautifulSoup
import requests

sys.path.append("../")
from Logic import utils
import time
from enum import Enum
import random
from Logic.core.utility.snippet import Snippet

snippet_obj = Snippet(
    number_of_words_on_each_side=5
)  # You can change this parameter, if needed.


class color(Enum):
    RED = "#FF0000"
    GREEN = "#00FF00"
    BLUE = "#0000FF"
    YELLOW = "#FFFF00"
    WHITE = "#FFFFFF"
    CYAN = "#00FFFF"
    MAGENTA = "#FF00FF"


def get_movie_img(url):
    try:
        response = requests.get(url, headers = {'User-Agent': 'IMDB Crawler'}, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        image_element = soup.find('img', {'class': 'ipc-image'})
        if image_element:
            image_url = image_element['src']
            return image_url
        else:
            return "https://m.media-amazon.com/images/M/MV5BNDE3ODcxYzMtY2YzZC00NmNlLWJiNDMtZDViZWM2MzIxZDYwXkEyXkFqcGdeQXVyNjAwNDUxODI@._V1_.jpg"
    except Exception:
        return "https://m.media-amazon.com/images/M/MV5BNDE3ODcxYzMtY2YzZC00NmNlLWJiNDMtZDViZWM2MzIxZDYwXkEyXkFqcGdeQXVyNjAwNDUxODI@._V1_.jpg"


def get_summary_with_snippet(movie_info, query):
    summary = movie_info["first_page_summary"]
    snippet, not_exist_words = snippet_obj.find_snippet(summary, query)
    if "***" in snippet:
        snippet = snippet.split()
        for i in range(len(snippet)):
            current_word = snippet[i]
            if current_word.startswith("***") and current_word.endswith("***"):
                current_word_without_star = current_word[3:-3]
                summary = summary.lower().replace(
                    current_word_without_star,
                    f"<b><font size='4' color={random.choice(list(color)).value}>{current_word_without_star}</font></b>",
                )
    return summary


def search_time(start, end):
    st.success("Search took: {:.6f} milli-seconds".format((end - start) * 1e3))


def search_handling(
    search_button,
    search_term,
    search_max_num,
    search_weights,
    search_method,
    unigram_smoothing,
    alpha,
    lamda,
    correct_query
):
    if search_button:
        if correct_query:
            corrected_query = utils.correct_text(search_term, utils.movies_dataset)
            if corrected_query != search_term:
                st.warning(f"Your search terms were corrected to: {corrected_query}")
                search_term = corrected_query

        

        with st.spinner("Searching..."):
            time.sleep(0.5)  # for showing the spinner! (can be removed)
            start_time = time.time()
            result = utils.search(
                search_term,
                search_max_num,
                search_method,
                search_weights,
                smoothing_method=unigram_smoothing,
                alpha=alpha,
                lamda=lamda,
            )
            print(f"Result: {result}")
            end_time = time.time()
            if len(result) == 0:
                st.warning("No results found!")
                return

            search_time(start_time, end_time)

            for i in range(len(result)):
                card = st.columns([3, 1])
                info = utils.get_movie_by_id(result[i][0], utils.movies_dataset)
                with card[0].container():
                    st.title(info["title"])
                    st.markdown(f"[Link to movie]({info['URL']})")
                    st.write(f"Relevance Score: {result[i][1]}")
                    st.markdown(
                        f"<b><font size = '4'>Summary:</font></b> {get_summary_with_snippet(info, search_term)}",
                        unsafe_allow_html=True,
                    )

                with st.container():
                    st.markdown("**Directors:**")
                    if info["directors"] is not None:
                        num_authors = len(info["directors"])
                        for j in range(num_authors):
                            st.text(info["directors"][j])

                with st.container():
                    st.markdown("**Stars:**")
                    if info["stars"] is not None:
                        num_authors = len(info["stars"])
                        stars = "".join(star + ", " for star in info["stars"])
                        st.text(stars[:-2])

                    topic_card = st.columns(1)
                    with topic_card[0].container():
                        st.write("Genres:")
                        num_topics = len(info["genres"])
                        for j in range(num_topics):
                            st.markdown(
                                f"<span style='color:{random.choice(list(color)).value}'>{info['genres'][j]}</span>",
                                unsafe_allow_html=True,
                            )
                with card[1].container():
                    st.image(get_movie_img(info['URL']), use_column_width=True)

                st.divider()


def main():
    st.title("Search Engine")
    st.write(
        "This is a simple search engine for IMDB movies. You can search through IMDB dataset and find the most relevant movie to your search terms."
    )
    st.markdown(
        '<span style="color:yellow">Developed By: MIR Team at Sharif University</span>',
        unsafe_allow_html=True,
    )

    search_term = st.text_input("Seacrh Term")
    # search_summary_terms = st.text_input("Search in summary of movie")
    with st.expander("Advanced Search"):
        search_max_num = st.number_input(
            "Maximum number of results", min_value=5, max_value=100, value=10, step=5
        )
        correct_query = st.checkbox("Use Spell Correction")
        weight_stars = st.slider(
            "Weight of stars in search",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
        )

        weight_genres = st.slider(
            "Weight of genres in search",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
        )

        weight_summary = st.slider(
            "Weight of summary in search",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
        )

        search_weights = [weight_stars, weight_genres, weight_summary]
        search_method = st.selectbox(
            "Search method",
            ("ltn.lnn", "ltc.lnc", "OkapiBM25", "Unigram"),
        )

        if search_method == "Unigram":
            unigram_smoothing = st.selectbox(
            "Unigram Smoothing method",
            ("mixture", "bayes", "naive"),
            )
            alpha = st.slider(
                "Unigram alpha",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.1,
            )

            lamda = st.slider(
                "Unigram lambda",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.1,
            )
        else:
            unigram_smoothing = None
            alpha=0.0
            lamda=0.0

    search_button = st.button("Search!")

    search_handling(
        search_button,
        search_term,
        search_max_num,
        search_weights,
        search_method,
        unigram_smoothing,
        alpha,
        lamda,
        correct_query
    )


if __name__ == "__main__":
    main()
