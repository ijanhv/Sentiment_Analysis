#include "longintrepr.h"
import streamlit as st
import altair as alt
# from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import joblib
pipe_lr = joblib.load(open("emotion_classifier_nv_model.pkl", "rb"))


def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]


def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results


emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗",
                       "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"}


def main():
    menu = ["Home", "Monitor", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home-Emotion In Text")
        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1, col2 = st.columns(2)
            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]
            with col1:
                st.success('Original Text')
                st.write(raw_text)

                st.success("Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write("{}:{}".format(prediction, emoji_icon))
                st.write("Confidence:{}".format(np.max(probability)))

                fig_pie = alt.Chart(proba_df_clean).mark_arc().encode(
                    theta='probability',
                    color='emotions',
                    tooltip=['emotions', 'probability']
                ).interactive()
                st.altair_chart(fig_pie, use_container_width=True)

            with col2:
                st.success("Prediction Probability")
                st.write(probability)
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                # st.write(proba_df.T)
                # proba_df_clean = proba_df.T.reset_index()
                # proba_df_clean.columns = ["emotions","probability"]
                fig = alt.Chart(proba_df_clean).mark_bar().encode(
                    x='emotions', y='probability', color='emotions')
                st.altair_chart(fig, use_container_width=True)

            # wordcloud = WordCloud().generate(raw_text)
            # # Display the generated image
            # wordcloud_image = wordcloud.to_image()
            # st.image(wordcloud_image)


    elif choice == "Monitor":
        st.subheader("Monitor App")

    else:
        st.subheader("About")


if _name_ == '_main_':
    main()