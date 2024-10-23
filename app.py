import re
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import streamlit as st
import xml.etree.ElementTree as ET

def extract_data(content):
    views = re.search(r'- \*\*Views:\*\* (.+)', content).group(1)
    likes = re.search(r'- \*\*Likes:\*\* (.+)', content).group(1)
    comments = re.findall(r'- \*\*(.+?)\*\*: (.+)', content)
    heatmap_svg = re.search(r'## Heatmap\n(.+)', content, re.DOTALL).group(1)

    return int(views.replace(',', '')), int(likes.replace(',', '')), comments, heatmap_svg

def perform_sentiment_analysis(comments):
    sentiments = []
    for author, comment in comments:
        analysis = TextBlob(comment)
        sentiments.append(analysis.sentiment.polarity)
    return sentiments

def calculate_ratios(views, likes, comments):
    view_to_comment_ratio = views / len(comments) if comments else 0
    views_to_like_ratio = views / likes if likes else 0
    comment_to_like_ratio = len(comments) / likes if likes else 0
    return view_to_comment_ratio, views_to_like_ratio, comment_to_like_ratio

def normalize_ratios(view_to_comment_ratio, views_to_like_ratio, comment_to_like_ratio):
    # Define maximum expected values for normalization
    max_view_to_comment_ratio = 100  # Example maximum value
    max_views_to_like_ratio = 50      # Example maximum value
    max_comment_to_like_ratio = 10     # Example maximum value

    # Normalize ratios to a score out of 10
    score_view_to_comment = min(view_to_comment_ratio / max_view_to_comment_ratio * 10, 10)
    score_views_to_like = min(views_to_like_ratio / max_views_to_like_ratio * 10, 10)
    score_comment_to_like = min(comment_to_like_ratio / max_comment_to_like_ratio * 10, 10)

    return score_view_to_comment, score_views_to_like, score_comment_to_like

def analyze_heatmap(heatmap_svg):
    attention_data = extract_attention_data(heatmap_svg)
    return attention_data

def extract_attention_data(svg_content):
    attention_data = []
    
    # Use regex or XML parsing to extract attention points
    root = ET.fromstring(svg_content)
    
    for element in root.iter():
        if element.tag in ['{http://www.w3.org/2000/svg}circle', '{http://www.w3.org/2000/svg}rect']:
            x = float(element.get('cx', 0)) if 'cx' in element.attrib else float(element.get('x', 0))
            y = float(element.get('cy', 0)) if 'cy' in element.attrib else float(element.get('y', 0))
            attention_data.append((x, y))

    return attention_data

def plot_attention_graph(attention_data):
    if not attention_data:
        return None  # No data to plot

    x_data, y_data = zip(*attention_data)
    plt.figure(figsize=(10, 5))
    plt.plot(x_data, y_data, marker='o')
    plt.title('Attention Heatmap Analysis')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Attention')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('attention_heatmap_analysis.png')
    plt.close()  # Close the plot to free up memory

def plot_sentiment_analysis(sentiments):
    plt.figure(figsize=(10, 5))
    plt.hist(sentiments, bins=20, color='blue', alpha=0.7)
    plt.title('Comment Sentiment Analysis')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.grid(axis='y')
    plt.tight_layout()

    plt.savefig('sentiment_analysis_chart.png')
    plt.close()  # Close the plot to free up memory

def main():
    st.title("Video Analysis Tool")
    
    uploaded_file = st.file_uploader("Upload a Markdown file", type=["md"])
    
    if uploaded_file is not None:
        content = uploaded_file.read().decode("utf-8")
        
        views, likes, comments, heatmap_svg = extract_data(content)
        sentiments = perform_sentiment_analysis(comments)
        ratios = calculate_ratios(views, likes, comments)
        normalized_scores = normalize_ratios(*ratios)
        attention_data = analyze_heatmap(heatmap_svg)

        # Display results
        st.write(f"**Views:** {views}")
        st.write(f"**Likes:** {likes}")
        st.write(f"**Comments Count:** {len(comments)}")
        st.write(f"**View-to-Comment Ratio Score:** {normalized_scores[0]:.2f}/10")
        st.write(f"**Views-to-Like Ratio Score:** {normalized_scores[1]:.2f}/10")
        st.write(f"**Comment-to-Like Ratio Score:** {normalized_scores[2]:.2f}/10")
        st.write(f"**Sentiment Mean:** {sum(sentiments) / len(sentiments) if sentiments else 0:.2f}")

        # Plot sentiment analysis
        plot_sentiment_analysis(sentiments)
        st.image('sentiment_analysis_chart.png')

        # Plot attention graph
        plot_attention_graph(attention_data)
        st.image('attention_heatmap_analysis.png')

if __name__ == "__main__":
    main()
