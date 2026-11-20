
import os
import io
import json
import base64
import traceback

import cv2
import numpy as np
import torch

from PIL import Image, ImageDraw, ImageFont, ImageOps
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from torchvision import models, transforms

from scipy.spatial.distance import pdist
from segment_anything import sam_model_registry, SamPredictor


# =========================================================
# 1. PATHS AND CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(BASE_DIR, "models")

YOLO_MODEL_PATH = os.path.join(
    MODELS_DIR, "epoch71.pt"
)

SAM_CHECKPOINT_PATH = os.path.join(
    MODELS_DIR, "sam_vit_b_01ec64.pth"
)

GATEKEEPER_MODEL_PATH = os.path.join(
    MODELS_DIR, "MobileNet_Gatekeeper.pth"
)

CLASS_NAMES_PATH = os.path.join(
    MODELS_DIR, "class_names.json"
)

SAM_MODEL_TYPE = "vit_b"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIDENCE_THRESHOLD = 0.25

GATEKEEPER_THRESHOLD = 0.60

GATEKEEPER_IMAGE_SIZE = 224

MAX_IMAGE_SIZE = 10 * 1024 * 1024

PIXEL_SPACING_MM = 0.75

GATEKEEPER_CLASSES = [
    "brain_ct",
    "chest_ct",
    "kidney_ct",
    "mri",
    "nonmedical",
    "normal_human",
    "other_body_ct",
    "ultrasound",
    "xray"
]


# =========================================================
# 2. MOBILENET PREPROCESSING
# =========================================================

GATEKEEPER_TRANSFORM = transforms.Compose([
    transforms.Resize(
        (GATEKEEPER_IMAGE_SIZE, GATEKEEPER_IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# 3. FLASK APPLICATION
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
# 4. LOAD MOBILENET GATEKEEPER
# =========================================================

print("Loading MobileNetV3 Gatekeeper...")

if not os.path.exists(GATEKEEPER_MODEL_PATH):
    raise FileNotFoundError(
        f"Gatekeeper model not found: {GATEKEEPER_MODEL_PATH}"
    )

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(
        f"Class names file not found: {CLASS_NAMES_PATH}"
    )


with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8-sig"
) as file:

    loaded_classes = json.load(file)


if loaded_classes != GATEKEEPER_CLASSES:

    raise ValueError(
        "class_names.json does not match the expected "
        "MobileNet class order. Check train_ds.classes "
        "and the saved model's training configuration."
    )


gatekeeper_model = models.mobilenet_v3_large(
    weights=None
)

in_features = gatekeeper_model.classifier[3].in_features

gatekeeper_model.classifier[3] = torch.nn.Linear(
    in_features,
    len(GATEKEEPER_CLASSES)
)


try:

    checkpoint = torch.load(
        GATEKEEPER_MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

except TypeError:

    checkpoint = torch.load(
        GATEKEEPER_MODEL_PATH,
        map_location=DEVICE
    )


if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]


gatekeeper_model.load_state_dict(
    checkpoint,
    strict=True
)

gatekeeper_model.to(DEVICE)

gatekeeper_model.eval()

print("MobileNetV3 Gatekeeper loaded successfully.")


# =========================================================
# 5. LOAD YOLO MODEL
# =========================================================

print("Loading YOLOv11 model...")

if not os.path.exists(YOLO_MODEL_PATH):

    raise FileNotFoundError(
        f"YOLO model not found: {YOLO_MODEL_PATH}"
    )


yolo_model = YOLO(YOLO_MODEL_PATH)

print("YOLOv11 model loaded successfully.")


# =========================================================
# 6. LOAD ORIGINAL SAM MODEL
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
# 7. MOBILENET GATEKEEPER PREDICTION
# =========================================================

def verify_kidney_ct(image):

    image_tensor = GATEKEEPER_TRANSFORM(image)

    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = gatekeeper_model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_index = torch.max(
            probabilities,
            dim=1
        )

    predicted_index = int(predicted_index.item())

    predicted_class = GATEKEEPER_CLASSES[predicted_index]

    predicted_probability = float(confidence.item())

    kidney_index = GATEKEEPER_CLASSES.index(
        "kidney_ct"
    )

    kidney_probability = float(
        probabilities[0][kidney_index].item()
    )

    accepted = (
        predicted_class == "kidney_ct"
        and kidney_probability >= GATEKEEPER_THRESHOLD
    )

    return {

        "accepted": accepted,

        "predicted_class": predicted_class,

        "confidence": round(
            predicted_probability * 100,
            2
        ),

        "kidney_ct_probability": round(
            kidney_probability * 100,
            2
        ),

        "threshold": GATEKEEPER_THRESHOLD * 100

    }


# =========================================================
# 8. IMAGE CONVERSION
# =========================================================

def image_to_data_uri(image_array):

    image = Image.fromarray(
        image_array.astype(np.uint8)
    )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return f"data:image/png;base64,{encoded}"


# =========================================================
# 9. BOX VALIDATION
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
# 10. ORIGINAL SAM SEGMENTATION
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
# 11. MEASURE SEGMENTATION MASK
# =========================================================

def measure_mask(mask):

    y_coordinates, x_coordinates = np.where(
        mask > 0
    )

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

    points = np.column_stack((
        x_coordinates,
        y_coordinates
    )).astype(np.float32)

    hull = cv2.convexHull(points)

    hull_points = hull.reshape(-1, 2)

    if len(hull_points) > 1:

        max_distance = float(
            pdist(hull_points).max()
        )

    else:

        max_distance = 0.0

    estimated_size_mm = (
        max_distance * PIXEL_SPACING_MM
    )

    return {

        "area_px": area,

        "width_px": int(width),

        "height_px": int(height),

        "max_distance_px": round(
            max_distance,
            2
        ),

        "estimated_size_mm": round(
            estimated_size_mm,
            2
        )

    }


# =========================================================
# 12. DRAW YOLO OUTPUT IMAGES
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

        text_y = max(
            0,
            int(y1) - 20
        )

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
# 13. SEGMENTATION OVERLAY
# =========================================================

def make_segmentation_overlay(
    image_array,
    combined_mask
):

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
# 14. SEGMENTATION IMAGE WITH SIZE LABELS
# =========================================================

def make_size_labeled_image(
    image_array,
    detections,
    combined_mask
):

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

        size_mm = dimensions["estimated_size_mm"]

        label = (
            f"Stone {index + 1}: "
            f"{size_mm:.2f} mm"
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
# 15. HOME ROUTE
# =========================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({

        "message": "Kidney Stone Detection API is running.",

        "device": DEVICE,

    })


# =========================================================
# 16. HEALTH CHECK
# =========================================================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({

        "status": "success",

        "gatekeeper_loaded": gatekeeper_model is not None,

        "yolo_loaded": yolo_model is not None,

        "sam_loaded": sam_predictor is not None,

        "gatekeeper_threshold": GATEKEEPER_THRESHOLD,

        "gatekeeper_classes": GATEKEEPER_CLASSES,

        "device": DEVICE,

    })


# =========================================================
# 17. PREDICTION API
# =========================================================

@app.route("/api/predict", methods=["POST"])
def predict():

    try:

        # ---------------------------------------------
        # CHECK UPLOADED FILE
        # ---------------------------------------------

        if "file" not in request.files:

            return jsonify({

                "error": (
                    "No image file received. "
                    "Please upload an image."
                )

            }), 400

        uploaded_file = request.files["file"]

        if uploaded_file.filename == "":

            return jsonify({

                "error": "No file selected."

            }), 400

        # ---------------------------------------------
        # READ IMAGE
        # ---------------------------------------------

        try:

            pil_image = Image.open(
                uploaded_file.stream
            )

            pil_image = ImageOps.exif_transpose(
                pil_image
            ).convert("RGB")

        except Exception:

            return jsonify({

                "error": "Invalid or unsupported image file."

            }), 400

        image_array = np.array(
            pil_image,
            dtype=np.uint8
        )

        image_height, image_width = image_array.shape[:2]

        # ---------------------------------------------
        # STEP 1: MOBILENET GATEKEEPER
        # ---------------------------------------------

        print("Running MobileNet Gatekeeper...")

        gatekeeper_result = verify_kidney_ct(
            pil_image
        )

        print(
            "Gatekeeper Result:",
            gatekeeper_result
        )

        # ---------------------------------------------
        # STEP 2: REJECT NON-KIDNEY CT
        # ---------------------------------------------

        if not gatekeeper_result["accepted"]:

            print("Image rejected by Gatekeeper.")

            return jsonify({

                "accepted": False,

                "gatekeeper": gatekeeper_result,

                "detected": False,

                "no_stone_detected": False,

                "detection_count": 0,

                "detections": [],

                "device": DEVICE,

                "message": (
                    "Image rejected by MobileNet Gatekeeper. "
                    "Only kidney_ct images with at least "
                    "60% kidney_ct probability are accepted."
                ),

                "image": {

                    "filename": uploaded_file.filename,

                    "format": "PNG/JPG",

                    "width": image_width,

                    "height": image_height,

                },

                "original_image": image_to_data_uri(
                    image_array
                ),

                "yolo_image": None,

                "yolo_labeled_image": None,

                "segmentation_image": None,

                "segmentation_size_image": None

            }), 200

        # ---------------------------------------------
        # STEP 3: ACCEPTED KIDNEY CT
        # ---------------------------------------------

        print(
            "Kidney CT accepted by Gatekeeper."
        )

        print("Running YOLOv11 detection...")

        # ---------------------------------------------
        # STEP 4: YOLO PREDICTION
        # ---------------------------------------------

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

        print(
            f"YOLO detections: {len(detections)}"
        )

        # ---------------------------------------------
        # STEP 5: YOLO OUTPUT IMAGES
        # ---------------------------------------------

        yolo_image, yolo_labeled_image = draw_yolo_images(
            image_array,
            detections,
        )

        # ---------------------------------------------
        # STEP 6: ORIGINAL SAM SEGMENTATION
        # ---------------------------------------------

        print(
            "Running Original SAM segmentation..."
        )

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

                detection["dimensions"] = measure_mask(
                    mask
                )

                combined_mask = np.maximum(
                    combined_mask,
                    mask
                )

        print(
            "YOLO + Original SAM completed."
        )

        # ---------------------------------------------
        # STEP 7: SEGMENTATION OVERLAY
        # ---------------------------------------------

        segmentation_overlay = make_segmentation_overlay(
            image_array,
            combined_mask,
        )

        # ---------------------------------------------
        # STEP 8: TOTAL MASK MEASUREMENTS
        # ---------------------------------------------

        total_dimensions = measure_mask(
            combined_mask
        )

        # ---------------------------------------------
        # STEP 9: IMAGE WITH SIZE LABELS
        # ---------------------------------------------

        size_labeled_image = make_size_labeled_image(
            image_array,
            detections,
            combined_mask
        )

        detection_count = len(detections)

        # ---------------------------------------------
        # STEP 10: JSON RESPONSE
        # ---------------------------------------------

        response_data = {

            "accepted": True,

            "gatekeeper": gatekeeper_result,

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
                "MobileNet Gatekeeper accepted the image. "
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

        print(
            "Prediction error:",
            str(error)
        )

        traceback.print_exc()

        return jsonify({

            "error": str(error),

            "message": "Image processing failed.",

        }), 500


# =========================================================
# 18. RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )
