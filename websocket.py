"""
WS prototipo 'todo en el servidor':
- Carga modelo Keras
- Recibe 'phrase_frames' (lista de palabras -> lista de frames JPEG base64)
- Extrae keypoints con MediaPipe Holistic
- Normaliza a MODEL_FRAMES
- Predice palabra por palabra
- Notifica a doctores 'phrase_result' y 'word_result'
"""
# ===================== Imports extra (arriba del archivo) =====================

from dotenv import load_dotenv
import os
from helpers import mediapipe_detection, extract_keypoints, get_word_ids
from evaluate_model import normalize_keypoints
from constants import MODEL_FRAMES

from openai import OpenAI
import requests
import asyncio
import json
import logging
import time
import base64
import re
import io
from dataclasses import dataclass
from typing import Dict, Optional, List

import numpy as np
import cv2
import websockets
from websockets.exceptions import ConnectionClosed


# ====== ML / Keypoints ======
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp

# ===================== Config =====================
WS_HOST = "0.0.0.0"
WS_PORT = 8080

MODEL_PATH = "./models/actions_30.h5"           # <<--- AJUSTA
WORDS_JSON_PATH = "./models/words.json"    # <<--- AJUSTA si usas un mapeo id->texto
MODEL_FRAMES = 30                 # nº de frames por secuencia que espera el modelo
THRESHOLD = 0.7                     # umbral prob. para aceptar palabra

# WebSocket settings
WS_MAX_MSG_SIZE = 4 * 1024 * 1024   # 4MB por mensaje
WS_PING_INTERVAL = 20
WS_PING_TIMEOUT  = 20

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger("ssf-ws")


# ===================== AGENTE =====================

OPENROUTER_API_KEY = ""  
client = OpenAI(
    api_key=OPENROUTER_API_KEY
)

# ===================== AGENTE =====================

# ===================== Utilidades =====================
data_url_re = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<b64>.+)$")

def decode_data_url_to_bgr(data_url: str) -> Optional[np.ndarray]:
    """
    data:image/jpeg;base64,/9j/... -> np.ndarray BGR (OpenCV)
    """
    m = data_url_re.match(data_url)
    if not m:
        return None
    b = base64.b64decode(m.group("b64"))
    img_array = np.frombuffer(b, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img  # BGR

def zero_pad_or_trim(sequence: np.ndarray, target_len: int) -> np.ndarray:
    """
    Ajusta la secuencia a target_len en el primer eje.
    Si es más larga, recorta centrado; si es más corta, zero-pad al final.
    """
    n = sequence.shape[0]
    if n == target_len:
        return sequence
    if n > target_len:
        # recorte centrado
        start = (n - target_len) // 2
        return sequence[start:start+target_len]
    # pad al final
    pad_shape = (target_len - n,) + sequence.shape[1:]
    pad = np.zeros(pad_shape, dtype=sequence.dtype)
    return np.concatenate([sequence, pad], axis=0)

def load_words_map(words_json_path: str) -> List[str]:
    """
    Si tienes un JSON con el orden de clases del modelo, cárgalo.
    Por simplicidad, aquí devolvemos una lista dummy si no existe.
    Formato esperado: array de etiquetas en el mismo orden que las salidas del modelo.
    """
    try:
        import json as _json
        with open(words_json_path, "r", encoding="utf-8") as f:
            data = _json.load(f)
        # soporta dict {class_id: label} o lista directa
        if isinstance(data, dict):
            # ordénalo por clave numérica si corresponde
            keys = sorted(data.keys(), key=lambda k: int(str(k).split("-")[0]) if str(k).split("-")[0].isdigit() else str(k))
            return [data[k] for k in keys]
        if isinstance(data, list):
            return data
    except Exception as e:
        logger.warning(f"No se pudo leer {words_json_path}: {e}")
    # fallback
    return [f"label_{i}" for i in range(100)]

# ===================== Keypoints con MediaPipe =====================
mp_holistic = mp.solutions.holistic

def extract_keypoints(results: mp_holistic.Holistic) -> np.ndarray:
    """
    Mismo layout clásico:
      pose: 33*4 (x,y,z,visibility)
      face: 468*3 (x,y,z)
      manos: 21*3 por mano (izq/der)
    """
    # Solo manos: 21*3*2 = 126
    lh = []
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            lh.extend([lm.x, lm.y, lm.z])
    else:
        lh = [0.0] * (21 * 3)

    rh = []
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            rh.extend([lm.x, lm.y, lm.z])
    else:
        rh = [0.0] * (21 * 3)

    return np.array(lh + rh, dtype=np.float32)

def frames_to_keypoints_sequence(frames_bgr: List[np.ndarray], holistic: mp_holistic.Holistic) -> np.ndarray:
    """
    Convierte lista de frames BGR -> [T, D] keypoints.
    """
    seq = []
    for bgr in frames_bgr:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = holistic.process(rgb)
        kps = extract_keypoints(res)
        seq.append(kps)
    return np.stack(seq, axis=0)  # [T, D]

# ===================== WebSocket Server =====================
@dataclass
class ClientInfo:
    ws: websockets.WebSocketServerProtocol
    session_id: Optional[str]
    client_id: str
    client_type: str  # 'doctor' | 'patient'

class WebSocketServer:
    def __init__(self):
        self.doctors: Dict[str, ClientInfo] = {}
        self.patients: Dict[str, ClientInfo] = {}
        self.sessions: Dict[str, Dict[str, Optional[str]]] = {}

        # ML
        logger.info("Cargando modelo Keras...")
        self.model = load_model(MODEL_PATH)
        logger.info("Modelo cargado.")
        self.class_labels = get_word_ids(WORDS_JSON_PATH)
        logger.info(f"Clases cargadas: {len(self.class_labels)} etiquetas -> {self.class_labels}")  

        # MediaPipe Holistic: crear uno por servidor (uso en to_thread)
        self.holistic = mp_holistic.Holistic(static_image_mode=False, model_complexity=1)

    async def start(self):
        logger.info(f"🚀 WS en {WS_HOST}:{WS_PORT}")
        async with websockets.serve(
            self._handle_client,
            WS_HOST,
            WS_PORT,
            max_size=None,
            ping_interval=WS_PING_INTERVAL,
            ping_timeout=WS_PING_TIMEOUT,
        ):
            logger.info("✅ Servidor WebSocket listo (modelo en memoria)")
            await asyncio.Future()

    async def _handle_client(self, websocket):
        peer = getattr(websocket, "remote_address", ("?", "?"))[0]
        logger.info(f"🔗 Conexión desde {peer}")
        try:
            async for raw in websocket:
                try:
                    data = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8"))
                except Exception:
                    await self._send_error(websocket, "JSON inválido")
                    continue

                msg_type = data.get("type")
                if msg_type == "register":
                    await self._handle_register(websocket, data)
                elif msg_type == "phrase_frames":
                    await self._handle_phrase_frames(websocket, data)
                elif msg_type == "word_frames":
                    # Soporte opcional por compatibilidad
                    await self._handle_word_frames(websocket, data)
                else:
                    await self._send_error(websocket, f"Tipo de mensaje desconocido: {msg_type}")
        except ConnectionClosed:
            logger.info("🔌 Cliente desconectado")
        except Exception as e:
            logger.exception(f"Error cliente: {e}")
        finally:
            await self._cleanup_client(websocket)

    async def _handle_register(self, websocket, data):
        cid = data.get("id")
        ctype = data.get("clientType")
        sid = data.get("sessionId")
        if not cid or ctype not in ("doctor", "patient"):
            await self._send_error(websocket, "Campos requeridos: id, clientType ('doctor'|'patient')")
            return

        info = ClientInfo(ws=websocket, session_id=sid, client_id=cid, client_type=ctype)
        if ctype == "doctor":
            self.doctors[cid] = info
        else:
            self.patients[cid] = info

        if sid:
            self.sessions.setdefault(sid, {"patient": None, "doctor": None})
            self.sessions[sid][ctype] = cid

        await websocket.send(json.dumps({
            "type": "registered",
            "clientType": ctype,
            "id": cid,
            "sessionId": sid
        }))
 


    def generate_medical_sentence(self, words):
        if not words:
            return ""

        prompt = f"""
        Contexto: el usuario está describiendo síntomas médicos en lenguaje de señas.

        Tarea: une las siguientes palabras en una frase médica coherente, breve y natural en español. 
        Debe sonar como una descripción de síntomas o malestar.
        No añadas ninguna explicación, comentario ni comillas; responde únicamente con la frase final.

        Palabras: {words}
        """

        # Inicializa el cliente de OpenAI

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # puedes cambiar a gpt-4o o gpt-4o-mini
                messages=[
                    {"role": "system", "content": "Eres un asistente médico que genera frases naturales basadas en síntomas."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.3  
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"❌ Error generando frase con OpenAI: {e}")
            return "Error al generar la frase"


    # ---------- Phrase (lista de lista) ----------
    async def _handle_phrase_frames(self, websocket, data):
            session_id = data.get("sessionId")
            patient_id = data.get("patientId")
            phrase_id  = data.get("phraseId")
            words      = data.get("words", [])
            logger.info(f"Recibidas {len(words)} palabras en phrase_frames para session_id={session_id}")
            if not session_id or not isinstance(words, list) or len(words) == 0:
                await self._send_error(websocket, "phrase_frames: payload inválido")
                return

            MAX_WORDS = 20
            MAX_FRAMES_PER_WORD = 120
            words = words[:MAX_WORDS]

            # --- Predicción por palabra ---
            word_preds = await asyncio.to_thread(self._predict_phrase, words, MAX_FRAMES_PER_WORD)
            logger.info(f"Predicciones frase: {word_preds}")

            word_results = []
            final_texts = []
            for pred_label, prob in word_preds:
                if pred_label is None:
                    word_results.append({"label": None, "prob": 0.0})
                    continue
                word_results.append({"label": pred_label, "prob": float(prob)})
                if prob >= THRESHOLD:
                    final_texts.append(pred_label)

            # --- Generar frase final ---
            if len(final_texts) == 1:
                # ✅ Caso: una sola palabra → no usamos el agente
                phrase_text = final_texts[0].capitalize()
                logger.info(f"🗣️ Frase directa (una palabra): {phrase_text}")
            elif len(final_texts) > 1:
                # ✅ Caso: varias palabras → usamos LLM
                phrase_text = self.generate_medical_sentence(final_texts)
                logger.info(f"🩺 Frase generada por LLM: {phrase_text}")
            else:
                phrase_text = "(sin detección válida)"
                logger.info("⚠️ No se detectaron palabras válidas")

            # --- Notificar doctores ---
            msg = {
                "type": "phrase_result",
                "sessionId": session_id,
                "patientId": patient_id,
                "phraseId": phrase_id,
                "words": word_results,
                "phraseText": phrase_text,
                "timestamp": int(time.time()*1000),
                "threshold": THRESHOLD
            }
            await self._broadcast_to_doctors(session_id, msg)

    def _predict_phrase(self, words_payload: List[List[dict]], max_frames_per_word: int):
        """
        Sin async: corre en hilo via asyncio.to_thread
        - Decodifica frames
        - Extrae keypoints con helpers
        - Normaliza con normalize_keypoints
        - Predice usando el modelo Keras
        Devuelve lista de tuplas (label, prob) por palabra en el mismo orden.
        """
        results = []
        for w_idx, word_frames in enumerate(words_payload):
            if not isinstance(word_frames, list) or not word_frames:
                results.append((None, 0.0))
                continue

            # 1️⃣ Decode frames a BGR
            frames_bgr = []
            for i, fr in enumerate(word_frames[:max_frames_per_word]):
                data_url = fr.get("data")
                if not data_url:
                    continue
                img = decode_data_url_to_bgr(data_url)
                if img is None:
                    continue
                frames_bgr.append(img)
            if not frames_bgr:
                results.append((None, 0.0))
                continue

            # 2️⃣ Extraer keypoints con el helper (mismo flujo del PyQt)
            seq = []
            for frame in frames_bgr:
                results_mp = mediapipe_detection(frame, self.holistic)
                kps = extract_keypoints(results_mp)
                seq.append(kps)

            # 3️⃣ Normalizar igual que en evaluate_model.py
            seq_fixed = normalize_keypoints(seq, int(MODEL_FRAMES))

            # 4️⃣ Predicción con el modelo cargado
            inp = np.expand_dims(seq_fixed, axis=0)
            probs = self.model.predict(inp, verbose=0)[0]
            cls_idx = int(np.argmax(probs))
            prob = float(probs[cls_idx])

            # 5️⃣ Etiqueta legible
            label = self.class_labels[cls_idx] if cls_idx < len(self.class_labels) else f"class_{cls_idx}"
            results.append((label, prob))

        return results


    # ---------- Word (compat) ----------
    async def _handle_word_frames(self, websocket, data):
        """
        Igual a 'phrase' pero para una sola palabra.
        """
        session_id = data.get("sessionId")
        patient_id = data.get("patientId")
        word_id    = data.get("wordId")
        frames     = data.get("frames", [])
        if not session_id or not isinstance(frames, list) or len(frames) == 0:
            await self._send_error(websocket, "word_frames: payload inválido")
            return

        preds = await asyncio.to_thread(self._predict_phrase, [frames], 120)
        label, prob = preds[0] if preds else (None, 0.0)
        msg = {
            "type": "word_result",
            "sessionId": session_id,
            "patientId": patient_id,
            "wordId": word_id,
            "label": label,
            "prob": float(prob),
            "accepted": bool(label and prob >= THRESHOLD),
            "timestamp": int(time.time()*1000),
            "threshold": THRESHOLD
        }
        await self._broadcast_to_doctors(session_id, msg)

    # ---------- Utils ----------
    async def _broadcast_to_doctors(self, session_id: str, payload: dict):
        sent = 0
        for did, d in list(self.doctors.items()):
            if d.session_id == session_id:
                try:
                    await d.ws.send(json.dumps(payload))
                    sent += 1
                except Exception as e:
                    logger.warning(f"Falló envío a doctor {did}: {e}")
        if sent == 0:
            logger.warning(f"Sin doctores en sesión {session_id}")

    async def _cleanup_client(self, websocket):
        removed_id = None
        removed_type = None
        for did, d in list(self.doctors.items()):
            if d.ws == websocket:
                del self.doctors[did]
                removed_id, removed_type = did, "doctor"
                break
        if not removed_id:
            for pid, p in list(self.patients.items()):
                if p.ws == websocket:
                    del self.patients[pid]
                    removed_id, removed_type = pid, "patient"
                    break
        if removed_id:
            for sid, parts in list(self.sessions.items()):
                if parts.get(removed_type) == removed_id:
                    parts[removed_type] = None
                    if not parts.get("doctor") and not parts.get("patient"):
                        logger.info(f"🗑️ Sesión {sid} cerrada (sin clientes conectados)")
                        del self.sessions[sid]
                    break

    async def _send_error(self, websocket, msg: str):
        try:
            await websocket.send(json.dumps({
                "type": "error",
                "message": msg,
                "timestamp": int(time.time()*1000)
            }))
        except Exception:
            pass

# ===================== main =====================
async def main():
    server = WebSocketServer()
    await server.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Detenido por usuario")
