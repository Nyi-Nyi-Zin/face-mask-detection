import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from collections import deque



def create_model():
    # MobileNetV2 base
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None
    )

    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Rescaling(1. / 127.5, offset=-1),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    return model



model = create_model()


try:
    model.load_weights('face_mask_classifier.h5')
    print("Success: Weights loaded from .h5 file!")
except Exception as e:
    try:
        model.load_weights('face_mask_classifier.keras')
        print("Success: Weights loaded from .keras file!")
    except Exception as e2:
        print(f"Error: Could not load weights. Make sure the file exists. {e2}")


prediction_history = deque(maxlen=8)


cap = cv2.VideoCapture(0)

print("Starting Webcam... Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break


    img = cv2.resize(frame, (224, 224))
    img = np.expand_dims(img, axis=0)


    raw_pred = model.predict(img, verbose=0)[0][0]


    prediction_history.append(raw_pred)

    # ပျမ်းမျှ (Average) ကို တွက်မယ်
    avg_pred = np.mean(prediction_history)

    # --- ပိုပြီး တည်ငြိမ်သွားစေမည့် Threshold သတ်မှတ်ချက် ---
    # 0.7 ထက်နည်းရင် (Model က သေချာသလောက်ရှိရင်) Mask ရှိတယ်လို့ ပြမယ်
    if avg_pred < 0.1:
        label = "With Mask"
        color = (0, 255, 0)  # အစိမ်းရောင်
        confidence = (1 - avg_pred) * 100
    else:
        label = "No Mask"
        color = (0, 0, 255)  # အနီရောင်
        confidence = avg_pred * 100

    # Screen ပေါ်မှာ စာသားနဲ့ ယုံကြည်မှုနှုန်း (%) ပြမယ်
    display_text = f"{label}: {confidence:.1f}%"
    cv2.putText(frame, display_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Video Window ပြသခြင်း
    cv2.imshow('Face Mask Detector - Press q to Quit', frame)

    # 'q' နှိပ်ရင် ပိတ်မယ်
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ပိတ်သိမ်းခြင်း
cap.release()
cv2.destroyAllWindows()