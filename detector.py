# REQUIERE PYTHON <= 3.12 (Problemas con mediapipe en 3.13+)
import platform, os
if platform.system() == "Linux":
    os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2, time, numpy as np
import serial, math, mediapipe as mp

# ==== CONFIG ====
PORT = '/dev/ttyUSB0'      # Cambia según tu sistema
BAUD = 115200
SHOW_FPS = True

SMILE_RATIO_TH = 70
EYE_RATIO_TH   = 0.22

# ==== Serial ====
try:
    esp = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)
    print("✅ ESP32 conectada")
except Exception as e:
    print("⚠️ No se pudo abrir el puerto serie:", e)
    esp = None

# ==== MediaPipe ====
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==== Cámara ====
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ No se pudo abrir la cámara.")
    exit()

t0, frames = time.time(), 0
last_state = {"face": 0, "eyes": 0, "smile": 0}

def dist(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])

def ratio_eye(lm, w, h, L, R, T, B):
    Lp, Rp, Tp, Bp = np.array([lm[L].x*w, lm[L].y*h]), np.array([lm[R].x*w, lm[R].y*h]), np.array([lm[T].x*w, lm[T].y*h]), np.array([lm[B].x*w, lm[B].y*h])
    return dist(Tp, Bp) / (dist(Lp, Rp) + 1e-6)

def ratio_mouth(lm, w, h):
    L, R, U, D = np.array([lm[61].x*w, lm[61].y*h]), np.array([lm[291].x*w, lm[291].y*h]), np.array([lm[13].x*w, lm[13].y*h]), np.array([lm[14].x*w, lm[14].y*h])
    return dist(L, R) / (dist(U, D) + 1e-6)

LEFT_L, LEFT_R, LEFT_T, LEFT_B = 33, 133, 159, 145
RIGHT_L, RIGHT_R, RIGHT_T, RIGHT_B = 362, 263, 386, 374

print("🎥 Detector iniciado. Presiona Q para salir.")

# --- Creamos una única ventana persistente ---
cv2.namedWindow("Detector Facial", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detector Facial", 640, 480)

while True:
    ok, frame = cap.read()
    if not ok:
        print("⚠️ Error al leer la cámara.")
        break

    frames += 1
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    state = {"face": 0, "eyes": 0, "smile": 0}

    if res.multi_face_landmarks:
        state["face"] = 1
        lm = res.multi_face_landmarks[0].landmark

        smile_ratio = ratio_mouth(lm, w, h)
        eye_left  = ratio_eye(lm, w, h, LEFT_L, LEFT_R, LEFT_T, LEFT_B)
        eye_right = ratio_eye(lm, w, h, RIGHT_L, RIGHT_R, RIGHT_T, RIGHT_B)
        eye_ratio = (eye_left + eye_right) / 2.0

        if smile_ratio > SMILE_RATIO_TH: state["smile"] = 1
        if eye_ratio > EYE_RATIO_TH:     state["eyes"] = 1

        cv2.putText(frame, f"Smile={smile_ratio:.2f}", (10, 30), 0, 0.6, (0,255,0), 2)
        cv2.putText(frame, f"Eyes={eye_ratio:.3f}", (10, 55), 0, 0.6, (0,255,0), 2)

    # Enviar si hay cambio
    if state != last_state and esp:
        msg = f"face={state['face']} eyes={state['eyes']} smile={state['smile']}\n"
        esp.write(msg.encode('utf-8'))
        print("➡️", msg.strip())
        last_state = state.copy()

    cv2.putText(frame, f"FACE:{state['face']}  EYES:{state['eyes']}  SMILE:{state['smile']}", (10, 80), 0, 0.6, (0,255,255), 2)

    if SHOW_FPS and frames % 10 == 0:
        now = time.time(); fps = frames / (now - t0 + 1e-6)
        cv2.putText(frame, f"FPS:{fps:.1f}", (10, 105), 0, 0.6, (255,255,0), 2)
        frames, t0 = 0, now

    # --- mostramos SIEMPRE en la misma ventana ---
    cv2.imshow("Detector Facial", frame)

    if cv2.getWindowProperty("Detector Facial", cv2.WND_PROP_VISIBLE) < 1:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if esp: esp.close()
