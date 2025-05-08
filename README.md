# StyloGuard

**StyloGuard** is a Streamlit-based authorship verification and stylometric analysis tool designed to support academic integrity through detailed writing style analysis. It allows evaluators to analyze essays, compare stylistic features across submissions, and use deep learning models to detect potential authorship inconsistencies.

---

## 📌 Project Overview

With the increasing use of AI-generated or paraphrased content, traditional plagiarism checkers are no longer sufficient on their own. *StyloGuard* provides an additional layer of verification by analyzing and comparing the **writing style** of students based on linguistic features and fine-tuned machine learning models.

The tool includes three major functionalities:

1. **Direct Analysis** – Analyze the writing style of a single essay.
2. **Feature Comparison** – Compare the stylistic similarity between two essays using extracted features.
3. **Stylometric Similarity Checker** – Use a fine-tuned BERT model to detect whether two essays are likely written by the same author, regardless of topic.

---

## 🚀 Running the App

You can run this app either locally or on **Streamlit Cloud**.

### 👉 On Streamlit Cloud:
1. Upload this project to a public GitHub repository.
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud) and sign in.
3. Select the GitHub repository and deploy the app.
4. Once the app is live, refer to the **Home** page within the application for a detailed guide on how to use each feature effectively.

### 👉 Locally (for developers):
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/StyloGuard.git
    cd StyloGuard
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download required NLP resources:
    ```bash
    python -m spacy download en_core_web_sm
    ```

4. Run the app:
    ```bash
    streamlit run StyloGuard.py
    ```

---

## 🧠 Features

- 20+ unique stylometric features including readability, lexical diversity, sentiment, and syntactic structure.
- Save analyzed essays and metadata to a PostgreSQL database.
- Weighted similarity comparison using radar plots.
- Deep learning–based stylometric similarity detection using a fine-tuned Sentence-BERT model.
- Support for both text input and file uploads (`.pdf` and `.docx` formats).

---

## 📚 Learning How to Use It

Once you launch the app, visit the **Home** page (formerly the About page). It provides a step-by-step explanation of how to use each part of the tool, including:
- What each page does,
- What kind of input is expected,
- What results to expect, and
- How to interpret the similarity scores.

This built-in documentation will guide any user—technical or non-technical—through the tool effectively.

---

## 🔧 Technical Stack

- **Frontend:** Streamlit
- **NLP Tools:** SpaCy, TextBlob, NLTK, Sentence-BERT
- **Database:** PostgreSQL (for saving student and essay data)
- **Visualization:** Plotly

---

## 📜 License

This project is available under the MIT License.

---

## 🙋‍♂️ Contributions

Feel free to fork the repository or suggest improvements through pull requests or issues. Any suggestions that help improve usability or functionality are welcome.

