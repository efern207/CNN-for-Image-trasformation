import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd

NUM_IMAGES = 2500
VAL = NUM_IMAGES - 100
W = 100
H = 100
COLOR = (255,)


all_images = np.empty((NUM_IMAGES, W, H), dtype=np.float32)
all_corners = np.zeros((NUM_IMAGES, 8), dtype=np.float32)
for i in range(NUM_IMAGES):
    background = np.zeros((H, W), dtype=np.uint8)  # Create a black image
    corners = np.array([[[0, 0],
                         [0, W],
                         [H, W],
                         [H, 0]]], dtype=np.int32)    # Coordinates of shape

    # Generate random corners
 
    corners[0, 0, 0] = np.random.randint(0, W//4)     # Top-left x coordinate
    corners[0, 0, 1] = np.random.randint(0, H//4)     # Top-left y coordinate
    corners[0, 1, 0] = np.random.randint(3*W//4, W)     # Top right x coordinate
    corners[0, 1, 1] = np.random.randint(0, H//4)     # Top-right y coordinate
 
    corners[0, 2, 0] = np.random.randint(3*W//4, W)     # Bottom right x coordinate
    corners[0, 2, 1] = np.random.randint(3*H//4, H)     # Bottom right y coordinate
    corners[0, 3, 0] = np.random.randint(0, W//4)     # Bottom-left x coordinate
    corners[0, 3, 1] = np.random.randint(3*H//4, H)     # Bottom left y coordinate
 
    
    
    shape = cv.fillPoly(background, corners, COLOR)   # Generate the shape

    all_images[i, :, :] = shape[:, :] * (1 / 255)
    all_corners[i, :] = np.reshape(corners, 8 , order="C") * (1 / max(W, H))


model = keras.models.Sequential([
    layers.Normalization(input_shape=(W, H, 1)),
    layers.Conv2D(9, 5, padding="same", activation="tanh"), ###
    layers.Conv2D(9, 5, padding="same", activation="relu"),
    layers.Conv2D(5, 3, padding="same", activation="tanh"),
    layers.Flatten(),
    layers.Dense(48, activation="sigmoid"),
    # layers.Dense(16, activation="sigmoid"),
    layers.Dense(8, activation="sigmoid")
])


model.summary()
model.compile(
    "adam",
    loss="mean_squared_error", ###
    metrics=["accuracy", "mean_squared_error", "mean_absolute_error"]
)

history = model.fit(
    all_images[:VAL, :, :],
    all_corners[:VAL, :],
    epochs=10,
    batch_size=100, ###
    verbose=1
)
for k in history.history:
    print(k, history.history[k], type(history.history[k]))

h = [[k] + history.history[k] for k in history.history]

df=pd.DataFrame(h)
df.to_excel("Stats_set11.xlsx",sheet_name="Stats")



results = model.evaluate(
    all_images[VAL:, :, :],
    all_corners[VAL:, :],
    batch_size=100
)

print(results)

df=pd.DataFrame(results)
df.to_excel("Stats2_set11.xlsx",sheet_name="Stats2")

predictions = model.predict(
    all_images[VAL:, :, :],
)


for i in range(VAL, NUM_IMAGES):
    copy = np.copy(all_images[i, :, :])
    center = (int(H*predictions[i-VAL, 0]), int(W*predictions[i-VAL, 1]))
    # center = (int(H*all_corners[i, 0]), int(W*all_corners[i, 1]))
    cv.circle(copy, center, 10, (0.5,))
    print(
        (center[0]/H - predictions[i-VAL, 0]) **2 + (center[1]/W - predictions[i-VAL, 1]) **2
    )


    src = np.float32([
        (int(W*predictions[i-VAL, 0]), int(H*predictions[i-VAL, 1])),
        (int(W*predictions[i-VAL, 2]), int(H*predictions[i-VAL, 3])),
        (int(W*predictions[i-VAL, 4]), int(H*predictions[i-VAL, 5])),
        (int(W*predictions[i-VAL, 6]), int(H*predictions[i-VAL, 7])),
    ])
    #src = np.float32([
    truth = np.float32([
        (int(W*all_corners[i, 0]), int(H*all_corners[i, 1])),
        (int(W*all_corners[i, 2]), int(H*all_corners[i, 3])),
        (int(W*all_corners[i, 4]), int(H*all_corners[i, 5])),
        (int(W*all_corners[i, 6]), int(H*all_corners[i, 7])),
    ])
    dst = np.float32([[0, 0], [H,0], [H ,W], [0,W]])
    txm = cv.getPerspectiveTransform(src, dst)
    out = cv.warpPerspective(copy, txm, (W, H))
    print(src)
    #print(truth)
    print('')

    cv.imshow("Display Orientation", out)
    cv.waitKey(0)
