import streamlit as st
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from nltk.sentiment import SentimentIntensityAnalyzer
from rake_nltk import Rake
from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Streamlit UI
st.markdown(
    """
    <style>
        .title {
            font-size: 36px;  /* Increase font size for the title */
            color: darkblue;   /* Change title color to dark blue */
        }
        .intro {
            font-size: 20px;  /* Set font size for the introductory text */
            color: darkgreen;  /* Change color of the introductory text */
        }
        .thick-hr {
            border: 4px solid #FFA500;  /* Increase thickness and change color as needed */
            margin: 10px 0;  /* Space above and below the line */
        }
        h2, h3, h4, h5, h6 {
            font-size: 24px;  /* Increase font size for all headers */
        }
        p {
            font-size: 20px;  /* Increase font size for paragraphs */
        }
        .summary {
            color: darkgreen;  /* Change color for summarized text */
            font-size: 20px;   /* Set font size for summarized text */
        }
        .keywords {
            color: darkpurple; /* Change color for keywords */
            font-size: 20px;   /* Set font size for keywords */
        }
        .sentiment {
            color: darkpurple; /* Change color for sentiment analysis */
            font-size: 20px;   /* Set font size for sentiment analysis */
        }
        .topic {
            color: darkorange; /* Change color for topic sentences */
            font-family: 'Courier New', Courier, monospace; /* Change font for topic sentences */
            font-size: 22px; /* Increase font size for topic sentences */
        }
    </style>
    """, 
    unsafe_allow_html=True
)

# Title
st.markdown('<h1 class="title">Analyze Me</h1>', unsafe_allow_html=True)
st.markdown('<p class="intro">We help you to create concise summaries, sentiment analysis, keyword extraction, and find the best topic for you.</p>', unsafe_allow_html=True)

# Step 5: AI Summary Image
st.subheader("AI Summary Image:")
# Correct the path to your image and reduce size (width set to 400 pixels)
st.image('images/ai-summary.jpg', caption='AI Generated Summary', use_column_width=False, width=400)

# Input text area
text = st.text_area("Enter your text here:", height=300)

if st.button("Analyze"):
    if text:
        # Step 1: Text Summarization
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, 2)

        # Add thick horizontal line before summary
        st.markdown('<div class="thick-hr"></div>', unsafe_allow_html=True)
        st.subheader("Summary:")
        for sentence in summary:
            st.markdown(f'<p class="summary">{sentence}</p>', unsafe_allow_html=True)  # Use class for color

        # Step 2: Sentiment Analysis using VADER
        sia = SentimentIntensityAnalyzer()
        sentiment_scores = sia.polarity_scores(text)

        # Add thick horizontal line before sentiment analysis
        st.markdown('<div class="thick-hr"></div>', unsafe_allow_html=True)
        st.subheader("Sentiment Analysis:")
        st.markdown(f'<p class="sentiment">*Negative:* {sentiment_scores["neg"]:.2f}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="sentiment">*Neutral:* {sentiment_scores["neu"]:.2f}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="sentiment">*Positive:* {sentiment_scores["pos"]:.2f}</p>', unsafe_allow_html=True)

        if sentiment_scores['compound'] >= 0.05:
            sentiment_label = "Positive"
        elif sentiment_scores['compound'] <= -0.05:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"

        st.markdown(f'<p class="sentiment">*Overall Sentiment:* {sentiment_label}</p>', unsafe_allow_html=True)

        # Step 3: Keyword Extraction
        rake = Rake()
        rake.extract_keywords_from_text(text)
        keywords = rake.get_ranked_phrases()

        # Add thick horizontal line before keyword extraction
        st.markdown('<div class="thick-hr"></div>', unsafe_allow_html=True)
        st.subheader("Keywords:")
        st.markdown(f'<p class="keywords">{", ".join(keywords)}</p>', unsafe_allow_html=True)  # Use class for color

        # Step 4: Topic Modeling
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in nltk.word_tokenize(text.lower()) if word.isalpha() and word not in stop_words]

        if len(tokens) >= 10:  # Ensure we have enough tokens for topic modeling
            texts = [tokens]
            dictionary = corpora.Dictionary(texts)
            corpus = [dictionary.doc2bow(text) for text in texts]

            num_topics = 2
            lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

            # Create topic sentences
            topic_sentences = []
            for idx, topic in lda_model.print_topics(num_words=3):
                # Extract words for the topic
                words = [word.split("*")[1].strip(' " ') for word in topic.split(" + ")]
                # Generate a simple sentence for the topic
                sentence = f" {' '.join(words)}."
                topic_sentences.append(sentence)

            # Add thick horizontal line before topic modeling
            st.markdown('<div class="thick-hr"></div>', unsafe_allow_html=True)
            st.subheader("Topic Modeling:")
            for idx, topic_sentence in enumerate(topic_sentences):
                st.markdown(f'<p class="topic">Topic {idx + 1}: {topic_sentence}</p>', unsafe_allow_html=True)

        else:
            st.warning("Not enough words for topic modeling. Please enter more text.")
    else:
        st.warning("Please enter some text for analysis.")
