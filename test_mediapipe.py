import mediapipe as mp

print("MediaPipe Version:", mp.__version__)

mp_hands = mp.solutions.hands

hands = mp_hands.Hands()

print("MediaPipe Hands Loaded Successfully")