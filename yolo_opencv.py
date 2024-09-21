import cv2
import argparse
import numpy as np

# Argument parser to take inputs from the command line
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help='path to input image')
ap.add_argument('-c', '--config', required=True,
                help='path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help='path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help='path to text file containing class names')
args = ap.parse_args()

# Function to get the output layer names
def get_output_layers(net):
    layer_names = net.getLayerNames()

    # Handle both scalar and list of lists cases for different OpenCV versions
    try:
        # For newer versions (returns scalars)
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except TypeError:
        # For older versions (returns list of lists)
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

# Function to draw bounding boxes and labels on the image
def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f"{classes[class_id]}: {confidence:.2f}"

    # Change color based on confidence
    if confidence >= 0.97:  # If confidence is 0.98 or higher
        color = (0, 255, 255)  # Yellow color (BGR format)
    else:  # For all other confidence levels
        color = (255, 0, 0)  # Green color (BGR format)

    # Draw bounding box and label
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 4, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
# Read the input image
image = cv2.imread(args.image)
Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392

# Load the class names
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Generate random colors for each class
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Load the YOLO model with pre-trained weights and config
net = cv2.dnn.readNet(args.weights, args.config)

# Preprocess the input image for YOLO
blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)

# Perform forward pass and get output
outs = net.forward(get_output_layers(net))

class_ids = []
confidences = []
boxes = []
conf_threshold = 0.4
nms_threshold = 0.3

# Parse the YOLO output
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.4:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w // 2
            y = center_y - h // 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

# Perform non-maxima suppression to remove overlapping boxes
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# Draw bounding boxes for each valid detection
if len(indices) > 0:
    for i in indices.flatten():  # Use flatten to handle np.int64 scalar indices
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
else:
    print("No objects detected.")

# Display the final image with the detection boxes
cv2.imshow("Object Detection", image)
cv2.waitKey()

# Save the result image to a file
cv2.imwrite("O_apple_detected.jpg", image)
cv2.destroyAllWindows()
