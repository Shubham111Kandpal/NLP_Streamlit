import pytest
from fastapi.testclient import TestClient
from main import app, get_word_embeddings, load_glove_model, label_mapping

client = TestClient(app)

@pytest.fixture(scope='module')
def glove_model():
    return load_glove_model()

def test_get_word_embeddings(glove_model):
    sentence = "Abbreviations : GEMS , Global Enteric Multicenter Study ; VIP , ventilated improved pit ."
    embeddings = get_word_embeddings(sentence, glove_model)
    assert len(embeddings) == len(sentence.split())

def test_predict_empty_input():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 200
    assert response.json() == {"error": "No embeddings generated for the input text."}

def test_predict_valid_input(glove_model):
    sentence = "Fractions from FPLC purification were treated with Laemmli buffer [ 82 ] with 10 mM 1,4 - dithiothreitol ( DTT ) and heated for 5 m at 85 ° C then analyzed on a 4 % to 15 % discontinuous SDS gel with a 6 % stacking gel run at ambient temperature at a constant 100 V."
    embeddings = get_word_embeddings(sentence, glove_model)
    if embeddings:
        response = client.post("/predict", json={"text": sentence})
        assert response.status_code == 200
        predictions = response.json()["predictions"]
        assert len(predictions) == len(sentence.split())
        for label in predictions:
            assert label in label_mapping.values()

def test_predict_unknown_words(glove_model):
    sentence = "Two epithelial cytokines other than IL33 , IL25 , and thymic stromal lymphopoietin ( TSLP ) are known to activate ILC2 in the lung [ 22,24 ] .."
    response = client.post("/predict", json={"text": sentence})
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == len(sentence.split())

def test_logging(tmp_path):
    log_file = tmp_path / "interaction_logs.txt"
    sentence = "Two epithelial cytokines other than IL33 , IL25 , and thymic stromal lymphopoietin ( TSLP ) are known to activate ILC2 in the lung [ 22,24 ] .."
    response = client.post("/predict", json={"text": sentence})
    assert response.status_code == 200

    with open(log_file, 'r') as f:
        log_entry = f.read().strip()
        assert "user_input" in log_entry
        assert sentence in log_entry
        assert "predicted_labels" in log_entry