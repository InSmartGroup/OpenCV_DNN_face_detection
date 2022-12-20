import cv2
import numpy as np

# Webcam stream
video_cap = cv2.VideoCapture(0)

# Named window to display the output
win_name = 'Web camera preview'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# Load the DNN model
prototext_path = "caffe_model/deploy.prototxt"
model_path = "caffe_model/res10_300x300_ssd_iter_140000.caffemodel"

model = cv2.dnn.readNetFromCaffe(prototext_path, model_path)

# Model parameters
mean = [104, 117, 123]
scale = 1.
in_width = 300
in_height = 300

# Face detection threshold
detection_threshold = 0.5

# Annotation settings
font_style = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_thickness = 1

while True:
    has_frame, frame = video_cap.read()
    if not has_frame:
        break

    height = frame.shape[0]
    width = frame.shape[1]

    # Flip the frame
    frame = cv2.flip(frame, 1)

    # Convert the frame into a blob format
    blob = cv2.dnn.blobFromImage(frame, scalefactor=scale, size=(in_width, in_height),
                                 mean=mean, swapRB=False, crop=False)

    # Pass the blob to the DNN model
    model.setInput(blob)

    # Retrieve detections from the DNN model
    detections = model.forward()

    # Process each detection
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > detection_threshold:
            # Extract the bounding box coordinates from the detection
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x1, y1, x2, y2) = box.astype('int')

            # Annotate the frame with results
            cv2.rectangle(frame, (x1, y2), (x2, y2), (0, 255, 255), 2)
            label = 'Confidence: %.4f' % confidence
            label_size, base_line = cv2.getTextSize(label, font_style, font_scale, font_thickness)
            cv2.rectangle(frame, (x1, y1 - label_size[1]), (x1 + label_size[0], y1 + base_line),
                          (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (x1, y1), font_style, font_scale, (0, 0, 0))

    cv2.imshow(win_name, frame)

    key = cv2.waitKey(1)
    if key == 27 or key == ord('q') or key == ord('Q'):
        break

video_cap.release()
cv2.destroyWindow(win_name)
