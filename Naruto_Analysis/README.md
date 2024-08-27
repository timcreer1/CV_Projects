# 🍥 Naruto Project

This project showcases a comprehensive approach to interacting with computer systems through hand gestures and explores various natural language processing and machine learning techniques applied to data from the Naruto series. It involves controlling a computer’s mouse and volume through hand gestures, scraping detailed Jutsu data, analyzing character relationships in Naruto subtitles, and building sophisticated theme and Jutsu classifiers.

## 📄 Overview

This Python-based project combines computer vision, web scraping, natural language processing, and deep learning to achieve multiple objectives:

### 🔍 Key Components

- **Subtitles**:  Contains a zipfile of the subtitles which are utilised in parts 2 & 3 below. 

- **1. Jutsu Crawler**:  
  The code builds a Scrapy spider to scrape data from a Naruto wiki page, specifically focusing on extracting detailed information about various Jutsu techniques. The spider navigates through the list of Jutsu, retrieves relevant details such as the Jutsu name, type, and description, and cleans the extracted HTML content using BeautifulSoup before returning the structured data.

- **2. Character Network**:  
  The code processes Naruto subtitles to extract character interactions within a sliding window of 10 sentences, using SpaCy for named entity recognition. The extracted relationships are then visualized as a network graph using NetworkX and Pyvis, highlighting the connections between characters based on their co-occurrence in the text.

- **3. Theme Classifier**:  
  The code trains a theme classifier using a Hugging Face zero-shot classification model, categorizing Naruto-related texts into themes like friendship, battle, and sacrifice. The model is fine-tuned on a dataset using PyTorch, with class weights applied to handle class imbalance, and evaluated using metrics like accuracy and F1-score.

- **4. Jutsu Classifier**:  
  The code preprocesses a dataset of Jutsu descriptions, simplifying them into three main categories (Ninjutsu, Taijutsu, and Genjutsu), and fine-tunes a DistilBERT model for sequence classification. The model training includes tokenization, data augmentation, and custom loss functions, ensuring high-quality classification performance evaluated on precision, recall, and accuracy metrics.

## 🛠️ Requirements

Before running the code in this project, make sure you have installed the following packages:

```bash
pip install scrapy bs4 pandas glob2 spacy networkx pyvis transformers tqdm torch json nltk datasets sklearn evaluate numpy
