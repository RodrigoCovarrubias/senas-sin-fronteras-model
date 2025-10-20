# app.py
from typing import List
import numpy as np
from flask import Flask, request, jsonify, abort
from keras.models import load_model

# your modules
from evaluate_model import normalize_keypoints
from helpers import get_word_ids, words_text
from constants import MODEL_PATH, MODEL_FRAMES, WORDS_JSON_PATH

app = Flask(__name__)

MODEL = load_model(MODEL_PATH)
WORD_IDS = get_word_ids(WORDS_JSON_PATH)

try:
    shape = MODEL.input_shape
    if isinstance(shape, list): shape = shape[0]
    EXPECTED_FRAMES = int(shape[1]) if shape[1] is not None else int(MODEL_FRAMES)
    EXPECTED_FEATS  = int(shape[2]) if shape[2] is not None else None
except Exception:
    EXPECTED_FRAMES = int(MODEL_FRAMES)
    EXPECTED_FEATS  = None

def _validate_sequence(seq: List[List[float]]):
    if not seq or not isinstance(seq[0], list):
        abort(422, description="Each sequence must be a non-empty list of frame vectors")
    lens = {len(f) for f in seq}
    if len(lens) != 1:
        abort(422, description=f"Inconsistent frame lengths inside a sequence: {sorted(list(lens))}")
    feat_len = next(iter(lens))
    if EXPECTED_FEATS is not None and feat_len != EXPECTED_FEATS:
        abort(422, description=f"Feature length {feat_len} != model expected {EXPECTED_FEATS}")

def _predict_word(seq_2d: np.ndarray, threshold: float) -> str:
    x = np.expand_dims(seq_2d, axis=0)                
    probs = MODEL.predict(x, verbose=0)[0]            
    idx = int(np.argmax(probs))
    if float(probs[idx]) < threshold:
        return ""                                     
    raw = WORD_IDS[idx]                               
    label_id = raw.split("-")[0]
    return words_text.get(label_id, raw)              

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_frames": int(EXPECTED_FRAMES),
        "features": int(EXPECTED_FEATS) if EXPECTED_FEATS is not None else None,
        "num_classes": len(WORD_IDS)
    })

@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    data = request.get_json(silent=True)
    if not data or "sequences" not in data:
        abort(400, description="JSON body must include 'sequences': List[List[List[float]]]")
    sequences = data["sequences"]
    threshold = float(data.get("threshold", 0.7))

    if not isinstance(sequences, list) or not sequences:
        abort(422, description="'sequences' must be a non-empty list of sequences")

    out: List[str] = []
    for seq in sequences:
        _validate_sequence(seq)
        norm = normalize_keypoints(seq, int(EXPECTED_FRAMES))  # -> (frames, features)
        out.append(_predict_word(norm, threshold))

    return jsonify(out)

if __name__ == "__main__":
    # For dev only; use gunicorn/uwsgi in production
    app.run(host="0.0.0.0", port=8000, debug=False)
