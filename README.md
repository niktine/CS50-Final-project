# Suicide Prediction and Explanation Tool

## Overview

This project is an advanced tool for suicide-related text classification using Natural Language Processing (NLP) and Machine Learning. The system is designed to process a given text, classify it as either **"Suicide"** or **"Non-Suicide"**, and provide an interpretable explanation for the model's prediction using LIME (Local Interpretable Model-agnostic Explanations). This tool can aid researchers, mental health practitioners, and developers in understanding and predicting suicide risk based on text inputs.

[Watch the introductory video here.](https://www.youtube.com/watch?v=GmlNISxKQUQ)

---

## Features

- **Preprocessing Pipeline:**
  - Text cleaning (removing punctuation, numbers, and URLs).
  - Text lemmatization using NLTK.
- **Text Classification:**
  - Utilizes Logistic Regression and Naïve Bayes models.
  - TF-IDF vectorization for feature extraction.
- **Explainable AI:**
  - Generates interpretable explanations for predictions using LIME.
- **Test Suite:**
  - Comprehensive test cases with Pytest to ensure robust functionality.

---

## Technologies Used

- **Programming Language:** Python
- **Libraries:**
  - `numpy` - Numerical computations.
  - `nltk` - Natural Language Toolkit for text preprocessing.
  - `lime` - Model interpretability.
  - `scikit-learn` - Machine Learning models and TF-IDF vectorization.
  - `joblib` - Model persistence.
  - `pytest` - Testing framework.

---

## Installation

### Prerequisites

- Python 3.7 or higher
- A virtual environment is recommended for dependency management.

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/niktine/CS50-Final-project
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download NLTK resources:
   ```bash
   python -m nltk.downloader wordnet omw-1.4
   ```

---

## Usage

### Run the Application

1. Execute the main script:
   ```bash
   python project.py
   ```
2. Enter a paragraph of text when prompted.
3. View the cleaned, lemmatized text, prediction, and explanation.
4. The LIME explanation will be saved as an HTML file (`prediction_result.html`).

### Running Tests

Execute the test suite using Pytest:

```bash
python -m pytest test_project.py
```

---

## Example Output

**Input:**

```
I feel like I can't go on anymore. Life is too hard.
```

**Output:**

- **Prediction:** Suicide
- **Explanation:**
  - Highlights words contributing to the prediction (e.g., "can't go on", "too hard").
  - Full explanation available in `prediction_result.html`.

---

## File Structure

```
.
├── project.py             # Main application script
├── test_project.py        # Test suite
├── requirements.txt       # Dependency file
├── prediction_result.html # LIME explanation output (generated dynamically)
└── README.md              # Documentation
```

---

## Future Enhancements

- Integration with real-time data sources (e.g., social media platforms).
- Incorporation of advanced NLP models (e.g., BERT, GPT).
- Enhanced multilingual support for broader applicability.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with your changes. Ensure that all new code is accompanied by relevant tests.

---

## Disclaimer

This tool is intended for research and educational purposes only. It is not a substitute for professional mental health services.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments

Special thanks to the open-source community and mental health professionals for their contributions to suicide prevention research.

