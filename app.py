
import os
import io
import base64
import traceback

import cv2
import numpy as np
import torch

from PIL import Image, ImageDraw, ImageFont
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO

from scipy.spatial.distance import pdist
from segment_anything import sam_model_registry, SamPredictor


# =========================================================
# PATHS AND CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "epoch71.pt"
)

SAM_CHECKPOINT_PATH = os.path.join(
    BASE_DIR, "models", "sam_vit_b_01ec64.pth"
)

SAM_MODEL_TYPE = "vit_b"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIDENCE_THRESHOLD = 0.25

MAX_IMAGE_SIZE = 10 * 1024 * 1024

# Estimated pixel spacing
PIXEL_SPACING_MM = 0.75


# =========================================================
# FLASK APPLICATION
# =========================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_SIZE

CORS(
    app,
    resources={
        r"/api/*": {
            "origins": [
                r"http://localhost:\d+",
                r"http://127\.0\.0\.1:\d+",
            ]
        }
    },
)


# =========================================================
# LOAD YOLO MODEL
# =========================================================

print("Loading YOLOv11 model...")

if not os.path.exists(YOLO_MODEL_PATH):
    raise FileNotFoundError(
        f"YOLO model not found: {YOLO_MODEL_PATH}"
    )

yolo_model = YOLO(YOLO_MODEL_PATH)

print("YOLOv11 model loaded successfully.")


# =========================================================
# LOAD ORIGINAL SAM MODEL
# =========================================================

print(f"Loading Original SAM ViT-B on {DEVICE}...")

if not os.path.exists(SAM_CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"SAM checkpoint not found: {SAM_CHECKPOINT_PATH}"
    )

sam_model = sam_model_registry[SAM_MODEL_TYPE](
    checkpoint=SAM_CHECKPOINT_PATH
)

sam_model.to(device=DEVICE)
sam_model.eval()

sam_predictor = SamPredictor(sam_model)

print("Original SAM loaded successfully.")
print(f"All models ready. Device: {DEVICE}")


# =========================================================
# IMAGE CONVERSION
# =========================================================

def image_to_data_uri(image_array):

    image = Image.fromarray(
        image_array.astype(np.uint8)
    )

    buffer = io.BytesIO()

    image.save(buffer, format="PNG")

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return f"data:image/png;base64,{encoded}"


# =========================================================
# BOX VALIDATION
# =========================================================

def clamp_box(box, image_width, image_height):

    x1, y1, x2, y2 = map(float, box)

    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))

    x2 = max(0, min(x2, image_width - 1))
    y2 = max(0, min(y2, image_height - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    return np.array(
        [x1, y1, x2, y2],
        dtype=np.float32
    )


# =========================================================
# ORIGINAL SAM SEGMENTATION
# =========================================================

def sam_segment(image_array, box):

    input_box = np.array(
        box,
        dtype=np.float32
    )

    masks, scores, logits = sam_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box,
        multimask_output=False
    )

    mask = masks[0].astype(np.uint8)

    sam_score = float(scores[0])

    return mask, sam_score


# =========================================================
# MEASURE SEGMENTATION MASK
# =========================================================

def measure_mask(mask):

    y_coordinates, x_coordinates = np.where(mask > 0)

    if len(x_coordinates) == 0:

        return {
            "area_px": 0,
            "width_px": None,
            "height_px": None,
            "max_distance_px": 0,
            "estimated_size_mm": 0
        }

    x_min = int(x_coordinates.min())
    x_max = int(x_coordinates.max())

    y_min = int(y_coordinates.min())
    y_max = int(y_coordinates.max())

    width = x_max - x_min + 1
    height = y_max - y_min + 1

    area = int(np.count_nonzero(mask))

    # Coordinates of segmented pixels
    points = np.column_stack((
        x_coordinates,
        y_coordinates
    )).astype(np.float32)

    # Convex hull reduces the number of points
    # while preserving the maximum distance.
    hull = cv2.convexHull(points)

    hull_points = hull.reshape(-1, 2)

    if len(hull_points) > 1:

        max_distance = float(
            pdist(hull_points).max()
        )

    else:
        max_distance = 0.0

    # Convert pixels into estimated millimeters
    estimated_size_mm = (
        max_distance * PIXEL_SPACING_MM
    )

    return {
        "area_px": area,
        "width_px": int(width),
        "height_px": int(height),
        "max_distance_px": round(
            max_distance, 2
        ),
        "estimated_size_mm": round(
            estimated_size_mm, 2
        )
    }


# =========================================================
# DRAW YOLO OUTPUT IMAGES
# =========================================================

def draw_yolo_images(image_array, detections):

    plain_image = Image.fromarray(
        image_array.copy()
    )

    labeled_image = Image.fromarray(
        image_array.copy()
    )

    plain_draw = ImageDraw.Draw(plain_image)
    labeled_draw = ImageDraw.Draw(labeled_image)

    for detection in detections:

        x1, y1, x2, y2 = detection["box"]

        box = (
            int(x1),
            int(y1),
            int(x2),
            int(y2),
        )

        plain_draw.rectangle(
            box,
            outline=(255, 0, 0),
            width=2,
        )

        labeled_draw.rectangle(
            box,
            outline=(255, 0, 0),
            width=2,
        )

        label = (
            f'{detection["class_name"]} '
            f'{detection["confidence"]:.2f}%'
        )

        text_y = max(0, int(y1) - 20)

        labeled_draw.text(
            (int(x1), text_y),
            label,
            fill=(255, 255, 0),
        )

    return (
        np.array(plain_image),
        np.array(labeled_image),
    )


# =========================================================
# SEGMENTATION OVERLAY
# =========================================================

def make_segmentation_overlay(image_array, combined_mask):

    overlay = image_array.copy()

    mask = combined_mask.astype(bool)

    if np.any(mask):

        green_layer = np.zeros_like(image_array)

        green_layer[:, :] = (0, 255, 0)

        alpha = 0.40

        overlay[mask] = (
            image_array[mask] * (1 - alpha)
            + green_layer[mask] * alpha
        ).astype(np.uint8)

        mask_uint8 = (
            mask.astype(np.uint8) * 255
        )

        contours, _ = cv2.findContours(
            mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        cv2.drawContours(
            overlay,
            contours,
            -1,
            (255, 0, 0),
            2,
        )

    return overlay


# =========================================================
# SEGMENTATION IMAGE WITH ESTIMATED SIZE LABELS
# =========================================================

def make_size_labeled_image(
    image_array,
    detections,
    combined_mask
):

    # Start with segmentation overlay
    output = make_segmentation_overlay(
        image_array,
        combined_mask
    )

    output_image = Image.fromarray(output)

    draw = ImageDraw.Draw(output_image)

    try:
        font = ImageFont.truetype(
            "arial.ttf",
            20
        )
    except OSError:
        font = ImageFont.load_default()

    for index, detection in enumerate(detections):

        dimensions = detection["dimensions"]

        if dimensions["area_px"] == 0:
            continue

        x1, y1, x2, y2 = detection["box"]

        max_distance = dimensions["max_distance_px"]

        size_mm = dimensions["estimated_size_mm"]

        label = (
            f'Stone {index + 1}: '
            f'{size_mm:.2f} mm'
        )

        text_x = int(x1)

        text_y = max(
            5,
            int(y1) - 30
        )

        text_bbox = draw.textbbox(
            (text_x, text_y),
            label,
            font=font
        )

        draw.rectangle(
            (
                text_bbox[0] - 4,
                text_bbox[1] - 3,
                text_bbox[2] + 4,
                text_bbox[3] + 3
            ),
            fill=(0, 0, 0)
        )

        draw.text(
            (text_x, text_y),
            label,
            fill=(255, 255, 0),
            font=font
        )

    return np.array(output_image)


# =========================================================
# HOME ROUTE
# =========================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "Kidney Stone Detection API is running.",
        "device": DEVICE,
    })


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({
        "status": "success",
        "yolo_loaded": yolo_model is not None,
        "sam_loaded": sam_predictor is not None,
        "device": DEVICE,
    })


# =========================================================
# PREDICTION API
# =========================================================

@app.route("/api/predict", methods=["POST"])
def predict():

    try:

        # CHECK UPLOADED FILE

        if "file" not in request.files:

            return jsonify({
                "error": "No image file received. Please upload an image."
            }), 400

        uploaded_file = request.files["file"]

        if uploaded_file.filename == "":

            return jsonify({
                "error": "No file selected."
            }), 400

        # READ IMAGE

        pil_image = Image.open(
            uploaded_file.stream
        ).convert("RGB")

        image_array = np.array(
            pil_image,
            dtype=np.uint8
        )

        image_height, image_width = image_array.shape[:2]

        print("Running YOLOv11 detection...")

        # YOLO PREDICTION

        yolo_results = yolo_model.predict(
            source=image_array,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )

        detections = []

        if yolo_results and len(yolo_results) > 0:

            result = yolo_results[0]

            if result.boxes is not None:

                boxes = result.boxes

                for index in range(len(boxes)):

                    raw_box = (
                        boxes.xyxy[index]
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    valid_box = clamp_box(
                        raw_box,
                        image_width,
                        image_height,
                    )

                    if valid_box is None:
                        continue

                    class_id = int(
                        boxes.cls[index].item()
                    )

                    confidence = float(
                        boxes.conf[index].item()
                    ) * 100

                    class_name = yolo_model.names.get(
                        class_id,
                        str(class_id)
                    )

                    detections.append({

                        "box": [
                            round(float(value), 2)
                            for value in valid_box
                        ],

                        "class_id": class_id,

                        "class_name": str(class_name),

                        "confidence": round(
                            confidence,
                            2
                        ),

                        "dimensions": {
                            "area_px": 0,
                            "width_px": None,
                            "height_px": None,
                            "max_distance_px": 0,
                            "estimated_size_mm": 0
                        },

                        "sam_score": None,

                    })

        print(f"YOLO detections: {len(detections)}")

        # YOLO OUTPUT IMAGES

        yolo_image, yolo_labeled_image = draw_yolo_images(
            image_array,
            detections,
        )

        # ORIGINAL SAM SEGMENTATION

        print("Running Original SAM segmentation...")

        combined_mask = np.zeros(
            (image_height, image_width),
            dtype=np.uint8,
        )

        if detections:

            sam_predictor.set_image(
                image_array,
                image_format="RGB"
            )

            for index, detection in enumerate(detections):

                print(
                    f"SAM segmentation: "
                    f"{index + 1}/{len(detections)}"
                )

                mask, sam_score = sam_segment(
                    image_array,
                    detection["box"],
                )

                if mask.shape != (
                    image_height,
                    image_width
                ):

                    raise ValueError(
                        "SAM returned a mask with unexpected dimensions."
                    )

                detection["sam_score"] = round(
                    sam_score,
                    4
                )

                # Calculate each stone's dimensions
                detection["dimensions"] = measure_mask(
                    mask
                )

                combined_mask = np.maximum(
                    combined_mask,
                    mask
                )

        print("YOLO + Original SAM completed.")

        # SEGMENTATION OVERLAY

        segmentation_overlay = make_segmentation_overlay(
            image_array,
            combined_mask,
        )

        # TOTAL MASK MEASUREMENTS

        total_dimensions = measure_mask(
            combined_mask
        )

        # IMAGE WITH SIZE LABELS

        size_labeled_image = make_size_labeled_image(
            image_array,
            detections,
            combined_mask
        )

        detection_count = len(detections)

        # JSON RESPONSE

        response_data = {

            "detected": detection_count > 0,

            "no_stone_detected": detection_count == 0,

            "detection_count": detection_count,

            "detections": detections,

            "device": DEVICE,

            "dimensions": total_dimensions,

            "pixel_spacing_mm": PIXEL_SPACING_MM,

            "measurement_unit": "mm",

            "measurement_note": (
                "Estimated size is calculated using maximum "
                "distance between segmented mask pixels multiplied "
                "by the configured pixel spacing of 0.75 mm/pixel. "
                "Use verified CT pixel spacing for clinical measurements."
            ),

            "image": {

                "filename": uploaded_file.filename,

                "format": "PNG/JPG",

                "width": image_width,

                "height": image_height,

            },

            "message": (
                "YOLOv11 detection and Original SAM segmentation completed."
            ),

            # IMAGE RESULTS

            "original_image": image_to_data_uri(
                image_array
            ),

            "yolo_image": image_to_data_uri(
                yolo_image
            ),

            "yolo_labeled_image": image_to_data_uri(
                yolo_labeled_image
            ),

            "segmentation_image": image_to_data_uri(
                segmentation_overlay
            ),

            "segmentation_size_image": image_to_data_uri(
                size_labeled_image
            ),

        }

        return jsonify(response_data), 200

    except Exception as error:

        print("Prediction error:", str(error))

        traceback.print_exc()

        return jsonify({
            "error": str(error),
            "message": "Image processing failed.",
        }), 500


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )
