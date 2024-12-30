import pytest
import numpy as np
from project import preprocess_text, p_lemmatizer, predict_and_explain_instance

@pytest.mark.parametrize("text,expected", [
    ("Hello World!", "hello world"),
    ("This is a TEST.", "this is a test"),
    ("Remove 123 numbers!", "remove numbers"),
    ("Check this link: https://example.com", "check this link"),
])

def test_preprocess_text(text, expected):
    assert preprocess_text(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("dogs running", "dog running"),
    ("children playing", "child playing"),
    ("feet walking", "foot walking"),
])
def test_p_lemmatizer(text, expected):

    assert p_lemmatizer(text) == expected


def test_predict_and_explain_instance(monkeypatch):
   
    def mock_tfidf_transform(texts):
        return texts

    def mock_predict_proba(X):
            
        return np.array([[0.8, 0.2]] * len(X))

    monkeypatch.setattr("project.tfidf.transform", mock_tfidf_transform)
    monkeypatch.setattr("project.logreg.predict_proba", mock_predict_proba)


    try:
        result = predict_and_explain_instance("test text")
        assert result == 0  
    except Exception as e:
        pytest.fail(f"Unexpected exception occurred: {e}")
