import os
import cv2

# SETTINGS
MIN_LENGTH_FRAMES = 5
LENGTH_KEYPOINTS = 126 # 258 # 1662
MODEL_FRAMES = 30

# PATHS
ROOT_PATH = os.getcwd()
FRAME_ACTIONS_PATH = os.path.join(ROOT_PATH, "frame_actions")
DATA_PATH = os.path.join(ROOT_PATH, "data")
DATA_JSON_PATH = os.path.join(DATA_PATH, "data.json")
MODEL_FOLDER_PATH = os.path.join(ROOT_PATH, "models")
MODEL_PATH = os.path.join(MODEL_FOLDER_PATH, f"actions_{MODEL_FRAMES}.keras")
MODEL_PATH2 = os.path.join(MODEL_FOLDER_PATH, f"actions_{MODEL_FRAMES}.h5")
KEYPOINTS_PATH = os.path.join(DATA_PATH, "keypoints")
WORDS_JSON_PATH = os.path.join(MODEL_FOLDER_PATH, "words.json")

# SHOW IMAGE PARAMETERS
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_SIZE = 1.5
FONT_POS = (5, 30)

words_text = {
    "alergia": "ALERGIA",
    "ayer": "AYER",
    "frio": "FRIO",
    "mucho": "MUCHO",
    "malestar_garganta": "MALASTAR DE GARGANTA",
    "mareo": "MAREO",
    "noche": "NOCHE",
    "semana": "SEMANA",
    "tos": "TOS",
    "cansado": "CANSADO",
    "dolor_de_estomago": "DOLOR DE ESTOMAGO",
    "congestion": "CONGESTIÓN",
    "hoy": "HOY",
    
}