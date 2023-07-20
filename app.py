import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import plotly.express as px

import joblib
pipe_lr = joblib.load(open("emotion_classifier.pkl", "rb"))

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

emotions_emoji_dict = {
    "anger": "😠",
    "disgust": "🤮",
    "fear": "😨😱",
    "happy": "🤗",
    "joy": "🤗",
    "neutral": "😐",
    "sad": "😔",
    "sadness": "😔",
    "shame": "😳",
    "surprise": "😮"
}

st.title("Sentiment Analysis")

def main():
    menu = ["Home","Data","About"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Home":
        st.subheader("Text Emotion Detection")
        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')
            
        if submit_text:
            col1, col2 = st.columns(2)
            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)
            with col1:
                st.success('Original Text')
                st.write(raw_text)
                
                st.success("Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write("{}:{}".format(prediction, emoji_icon))
                st.write("Confidence:{}".format(np.max(probability)))
                
            with col2:
                st.success("Prediction Probability")
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions","probability"]
                fig_bar = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
                st.altair_chart(fig_bar, use_container_width=True)
            
        
                
    elif choice == "Data":
        st.subheader("CSV Data")
        data_file = st.file_uploader("Upload CSV", type=['csv'])
        if data_file is not None:
            df = pd.read_csv(data_file)
            # st.dataframe(df)

            # find the emotions along with emoji icons

            # apply predict_emotions() on every row of the dataframe

            emotions = []
            for text in df['Text']:
                pred = predict_emotions(text)
                emotions.append(pred)
                # emotions.append(emotions_emoji_dict[pred])

            df['emotions'] = emotions
            # df['emoji'] = emotions_emoji_dict[pred]
            st.dataframe(df)

            # plot and visualize the count of emotions

            st.subheader("Count of Emotions")
            fig = alt.Chart(df).mark_bar().encode(x='emotions', y='count()', color='emotions')
            st.altair_chart(fig, use_container_width=True)       
          
        
    else:
        st.subheader("About")
        st.text("Built with Streamlit")

        st.info("This is a sentiment analysis app to analyze the emotions of the text you enter")


if __name__ == '__main__':
    main()
