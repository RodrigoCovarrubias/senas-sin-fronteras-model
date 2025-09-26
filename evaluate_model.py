# --- poner los flags ANTES de importar TF/MP ---
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # silencia INFO/WARNING de TF

from absl import logging
logging.set_verbosity(logging.ERROR)

import cv2
import numpy as np

# MediaPipe 0.10.x
from mediapipe.python.solutions.holistic import Holistic

# usa tf.keras, no el paquete 'keras'
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from helpers import *
from constants import *
from text_to_speech import text_to_speech


def interpolate_keypoints(keypoints, target_length=15):
    current_length = len(keypoints)
    if current_length == target_length:
        return keypoints

    indices = np.linspace(0, current_length - 1, target_length)
    interpolated_keypoints = []
    for i in indices:
        lower_idx = int(np.floor(i))
        upper_idx = int(np.ceil(i))
        weight = i - lower_idx
        if lower_idx == upper_idx:
            interpolated_keypoints.append(keypoints[lower_idx])
        else:
            p0 = np.asarray(keypoints[lower_idx], dtype=np.float32)
            p1 = np.asarray(keypoints[upper_idx], dtype=np.float32)
            interpolated_point = (1 - weight) * p0 + weight * p1
            interpolated_keypoints.append(interpolated_point.tolist())
    return interpolated_keypoints


def normalize_keypoints(keypoints, target_length=15):
    current_length = len(keypoints)
    if current_length < target_length:
        return interpolate_keypoints(keypoints, target_length)
    elif current_length > target_length:
        step = current_length / target_length
        indices = np.arange(0, current_length, step).astype(int)[:target_length]
        return [keypoints[i] for i in indices]
    else:
        return keypoints


def evaluate_model(src=None, threshold=0.8, margin_frame=1, delay_frames=3):
    """
    src: None -> webcam 0
    threshold: prob mínima para aceptar predicción
    margin_frame: frames de “colchón” tras detectar mano antes de empezar a acumular keypoints
    delay_frames: frames de “colchón” tras dejar de ver mano antes de cerrar la seña
    """
    kp_seq, sentence = [], []
    word_ids = get_word_ids(WORDS_JSON_PATH)

    # carga del modelo (tf.keras)
    model = load_model(MODEL_PATH)

    count_frame = 0
    fix_frames = 0
    recording = False

    # Opcional: params del Holistic (puedes afinarlos)
    with Holistic(static_image_mode=False, model_complexity=1, smooth_landmarks=True) as holistic_model:
        video = cv2.VideoCapture(src or 0)
        # Opcional: baja resolución para más FPS
        # video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # video.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        last_pred_id = None         # para “suavizar” cambios bruscos
        stable_hits = 0             # cuántas veces seguidas mantuvo la misma clase
        stable_needed = 2           # p.ej. exigir 2 aciertos seguidos

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            results = mediapipe_detection(frame, holistic_model)

            # Si hay mano o ya estamos grabando, acumula frames
            if there_hand(results) or recording:
                recording = False
                count_frame += 1
                if count_frame > margin_frame:
                    kp_frame = extract_keypoints(results)
                    kp_seq.append(kp_frame)

            else:
                # Se dejó de ver mano: ¿tenemos secuencia suficiente como para intentar?
                if count_frame >= MIN_LENGTH_FRAMES + margin_frame:
                    fix_frames += 1
                    if fix_frames < delay_frames:
                        # espera unos frames extra por si la seña aún no terminó
                        recording = True
                        continue

                    # descarta los frames de margen y delay del final
                    if margin_frame + delay_frames > 0 and len(kp_seq) > (margin_frame + delay_frames):
                        kp_seq = kp_seq[: -(margin_frame + delay_frames)]

                    # normaliza al largo del modelo y asegura dtype
                    kp_normalized = normalize_keypoints(kp_seq, int(MODEL_FRAMES))
                    X = np.expand_dims(kp_normalized, axis=0).astype(np.float32)

                    # predicción
                    res = model.predict(X, verbose=0)[0]
                    pred_id = int(np.argmax(res))
                    pred_conf = float(res[pred_id])

                    print(pred_id, f"({pred_conf * 100:.2f}%)")

                    # Pequeña lógica de estabilización
                    if pred_id == last_pred_id:
                        stable_hits += 1
                    else:
                        last_pred_id = pred_id
                        stable_hits = 1

                    if pred_conf > threshold and stable_hits >= stable_needed:
                        word_id = word_ids[pred_id].split('-')[0]
                        sent = words_text.get(word_id, word_id)
                        sentence.insert(0, sent)
                        text_to_speech(sent)  # solo local
                        # resetea estabilizador tras aceptar
                        stable_hits = 0
                        last_pred_id = None

                # reset de acumuladores para la próxima seña
                recording = False
                fix_frames = 0
                count_frame = 0
                kp_seq = []

            # UI
            if not src:
                cv2.rectangle(frame, (0, 0), (640, 35), (245, 117, 16), -1)
                cv2.putText(frame, ' | '.join(sentence[:3]), FONT_POS, FONT, FONT_SIZE, (255, 255, 255), 1, cv2.LINE_AA)

                draw_keypoints(frame, results)
                cv2.imshow('Traductor LSP', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        video.release()
        cv2.destroyAllWindows()
        return sentence


if __name__ == "__main__":
    evaluate_model()
