import re
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import streamlit as st
import xml.etree.ElementTree as ET
import io

def extract_data(content):
    views_match = re.search(r'(\d[\d,]*) views', content)  # Extract numeric views
    likes_match = re.search(r'(\d[\d,]*) likes', content)  # Extract numeric likes
    comments_match = re.findall(r'(?<=\n).*?(?=\n)', content)  # Extract all comments as a list
    heatmap_match = re.search(r'## Heatmap SVG\n(.*?)\n\n', content, re.DOTALL)  # Adjust for multi-line SVG

    # Extract views, likes
    views = views_match.group(1).replace(',', '') if views_match else '0'
    likes = likes_match.group(1).replace(',', '') if likes_match else '0'
    
    heatmap_svg = heatmap_match.group(1) if heatmap_match else None

    # Convert extracted values to integers
    return int(views), int(likes), comments_match, heatmap_svg

def perform_sentiment_analysis(comments):
    sentiments = []
    for comment in comments:
        analysis = TextBlob(comment)
        sentiments.append((comment, analysis.sentiment))  # Store comment with its sentiment
    return sentiments


def calculate_ratios(views, likes, comments):
    view_to_comment_ratio = views / len(comments) if comments else 0
    views_to_like_ratio = views / likes if likes else 0
    comment_to_like_ratio = len(comments) / likes if likes else 0
    return view_to_comment_ratio, views_to_like_ratio, comment_to_like_ratio

def normalize_ratios(view_to_comment_ratio, views_to_like_ratio, comment_to_like_ratio):
    max_view_to_comment_ratio = 100
    max_views_to_like_ratio = 50
    max_comment_to_like_ratio = 10

    score_view_to_comment = min(view_to_comment_ratio / max_view_to_comment_ratio * 10, 10)
    score_views_to_like = min(views_to_like_ratio / max_views_to_like_ratio * 10, 10)
    score_comment_to_like = min(comment_to_like_ratio / max_comment_to_like_ratio * 10, 10)

    return score_view_to_comment, score_views_to_like, score_comment_to_like


def extract_attention_data(svg_content):
    # Check if svg_content is None
    if svg_content is None:
        return []  # or raise an Exception, or return a default value

    attention_data = []
    try:
        # Use the ET.fromstring method to parse the SVG content
        root = ET.fromstring(svg_content)

        for element in root.iter():
            if element.tag in ['{http://www.w3.org/2000/svg}circle', '{http://www.w3.org/2000/svg}rect']:
                attention_data.append({
                    'tag': element.tag,
                    'attributes': element.attrib,
                })
    except ET.ParseError as e:
        print(f"Error parsing SVG content: {e}")
        # Handle parse error appropriately

    return attention_data


def plot_attention_graph(attention_data):
    if not attention_data:
        return None

    x_data, y_data = zip(*attention_data)
    plt.figure(figsize=(10, 5))
    plt.scatter(x_data, y_data, marker='o', color='orange')
    plt.title('Attention Heatmap Analysis')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)

    # Save to a BytesIO stream instead of a file
    img_stream = io.BytesIO()
    plt.savefig(img_stream, format='png')
    plt.close()
    img_stream.seek(0)  # Reset the stream to the beginning
    return img_stream

def plot_sentiment_analysis(sentiments):
    plt.figure(figsize=(10, 5))
    plt.hist(sentiments, bins=20, color='blue', alpha=0.7)
    plt.title('Comment Sentiment Analysis')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.grid(axis='y')

    # Save to a BytesIO stream instead of a file
    img_stream = io.BytesIO()
    plt.savefig(img_stream, format='png')
    plt.close()
    img_stream.seek(0)  # Reset the stream to the beginning
    return img_stream

def main():
    st.title("Video Analysis Tool")
    st.sidebar.title("Navigation")
    
    uploaded_file = st.file_uploader("Upload your file", type=["txt", "csv" , "md"])
    
    if uploaded_file is not None:
        content = uploaded_file.read().decode("utf-8")
        views, likes, comments, heatmap_svg = extract_data(content)

        # Only extract attention data if heatmap_svg is not None
        if heatmap_svg:
            attention_data = extract_attention_data(heatmap_svg)
        else:
            st.warning("No heatmap data available.")
            attention_data = []  # or handle this case as needed
        
        views, likes, comments, heatmap_svg = extract_data(content)
        sentiments = perform_sentiment_analysis(comments)
        ratios = calculate_ratios(views, likes, comments)
        normalized_scores = normalize_ratios(*ratios)
        attention_data = extract_attention_data(heatmap_svg)

        # Display results
        st.header("Analysis Results")
        st.metric("Views", views)
        st.metric("Likes", likes)
        st.metric("Comments Count", len(comments))
        st.metric("View-to-Comment Ratio Score", f"{normalized_scores[0]:.2f}/10")
        st.metric("Views-to-Like Ratio Score", f"{normalized_scores[1]:.2f}/10")
        st.metric("Comment-to-Like Ratio Score", f"{normalized_scores[2]:.2f}/10")
        st.metric("Sentiment Mean", f"{sum(sentiments) / len(sentiments) if sentiments else 0:.2f}")

        # Plot sentiment analysis
        st.subheader("Sentiment Analysis")
        sentiment_img_stream = plot_sentiment_analysis(sentiments)
        st.image(sentiment_img_stream, caption='Sentiment Analysis Histogram', use_column_width=True)

        # Plot attention graph
        st.subheader("Attention Heatmap Analysis")
        attention_img_stream = plot_attention_graph(attention_data)
        st.image(attention_img_stream, caption='Attention Heatmap', use_column_width=True)

if __name__ == "__main__":
    main()
