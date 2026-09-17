import streamlit as st
import cv2
import easyocr
from ultralytics import YOLO
from datetime import datetime
import os
import csv
import math

# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="AI Traffic Violation Detection",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 AI Traffic Violation Detection System")
st.write("Detect vehicles, violations, number plates and generate fines dynamically.")

# ============================================================
# FINE AMOUNTS - DEMO VALUES
# ============================================================

FINE_AMOUNTS = {
    "Without Helmet": 1500,
    "No Seat Belt": 1000,
    "Red Light Violation": 1000,
    "Triple Riding": 2000
}

# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    vehicle_model = YOLO("yolo11n.pt")

    helmet_model = YOLO(
        "runs/detect/train-5/weights/best.pt"
    )

    plate_model = YOLO(
        "models/license_plate_best.pt"
    )

    reader = easyocr.Reader(
        ["en"],
        gpu=False
    )

    return (
        vehicle_model,
        helmet_model,
        plate_model,
        reader
    )


vehicle_model, helmet_model, plate_model, reader = load_models()

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_center(box):
    """
    Get center point of bounding box.
    """
    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2
    )


def point_inside_box(point, box, margin=0.15):
    """
    Check whether a point is inside or near a vehicle box.
    """

    px, py = point

    x1, y1, x2, y2 = box

    width = x2 - x1
    height = y2 - y1

    x1 -= width * margin
    y1 -= height * margin
    x2 += width * margin
    y2 += height * margin

    return (
        x1 <= px <= x2
        and
        y1 <= py <= y2
    )


def center_distance(point1, point2):

    return math.sqrt(
        (point1[0] - point2[0]) ** 2
        +
        (point1[1] - point2[1]) ** 2
    )


def associate_plate_to_vehicle(
    plate_center,
    vehicles
):
    """
    Find the vehicle closest to the detected plate.
    """

    best_vehicle = None
    best_distance = float("inf")

    for vehicle in vehicles:

        vehicle_center = get_center(
            vehicle["box"]
        )

        distance = center_distance(
            plate_center,
            vehicle_center
        )

        # Plate should normally be inside/near vehicle
        if point_inside_box(
            plate_center,
            vehicle["box"],
            margin=0.35
        ):

            if distance < best_distance:

                best_distance = distance
                best_vehicle = vehicle

    return best_vehicle


# ============================================================
# LOCATION
# ============================================================

location = st.text_input(
    "📍 Violation Location",
    "Bapatla Main Road"
)

# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📷 Upload ANY Traffic Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # ========================================================
    # CREATE DIRECTORIES
    # ========================================================

    os.makedirs(
        "input/images",
        exist_ok=True
    )

    os.makedirs(
        "output/violations",
        exist_ok=True
    )

    # Use unique filename so every uploaded image is processed
    timestamp_file = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    image_path = (
        f"input/images/"
        f"traffic_{timestamp_file}.jpg"
    )

    with open(
        image_path,
        "wb"
    ) as file:

        file.write(
            uploaded_file.getbuffer()
        )

    image = cv2.imread(
        image_path
    )

    if image is None:

        st.error(
            "❌ Could not read the uploaded image."
        )

        st.stop()

    # ========================================================
    # DISPLAY IMAGE
    # ========================================================

    st.subheader(
        "📷 Captured Traffic Image"
    )

    st.image(
        image,
        channels="BGR",
        width="stretch"
    )

    # ========================================================
    # VEHICLE DETECTION
    # ========================================================

    st.subheader(
        "🚗 Vehicle Detection"
    )

    vehicle_results = vehicle_model.predict(
        source=image,
        conf=0.40,
        verbose=False
    )

    vehicle_classes = {

        2: "Car",
        3: "Motorcycle",
        5: "Bus",
        7: "Truck"
    }

    vehicle_counts = {

        "Car": 0,
        "Motorcycle": 0,
        "Bus": 0,
        "Truck": 0
    }

    vehicles = []

    for result in vehicle_results:

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            if class_id not in vehicle_classes:
                continue

            x1, y1, x2, y2 = map(
                float,
                box.xyxy[0]
            )

            vehicle_type = (
                vehicle_classes[class_id]
            )

            vehicle_counts[
                vehicle_type
            ] += 1

            vehicles.append({

                "id": len(vehicles) + 1,

                "type": vehicle_type,

                "box": (
                    x1,
                    y1,
                    x2,
                    y2
                ),

                "violations": [],

                "plate": "Not Recognized",

                "plate_confidence": 0.0
            })

    # ========================================================
    # VEHICLE COUNTS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "🚗 Cars",
        vehicle_counts["Car"]
    )

    col2.metric(
        "🏍️ Motorcycles",
        vehicle_counts["Motorcycle"]
    )

    col3.metric(
        "🚌 Buses",
        vehicle_counts["Bus"]
    )

    col4.metric(
        "🚚 Trucks",
        vehicle_counts["Truck"]
    )

    st.info(
        f"Total vehicles detected: {len(vehicles)}"
    )

    if len(vehicles) == 0:

        st.warning(
            "⚠️ No supported vehicles detected."
        )

        st.stop()

    # ========================================================
    # HELMET DETECTION
    # ========================================================

    st.subheader(
        "🪖 Helmet Detection"
    )

    helmet_results = helmet_model.predict(
        source=image,
        conf=0.40,
        verbose=False
    )

    helmet_image = helmet_results[0].plot()

    st.image(
        helmet_image,
        channels="BGR",
        width="stretch"
    )

    helmet_detections = []

    for result in helmet_results:

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            class_name = (
                result.names[class_id]
            )

            x1, y1, x2, y2 = map(
                float,
                box.xyxy[0]
            )

            helmet_detections.append({

                "class": class_name,

                "box": (
                    x1,
                    y1,
                    x2,
                    y2
                ),

                "center": get_center(
                    (
                        x1,
                        y1,
                        x2,
                        y2
                    )
                )
            })

    # ========================================================
    # ASSOCIATE HELMET VIOLATIONS WITH MOTORCYCLES
    # ========================================================

    without_helmet_count = 0

    for vehicle in vehicles:

        # Only motorcycles are considered
        # for helmet violation.

        if vehicle["type"] != "Motorcycle":
            continue

        vehicle_box = vehicle["box"]

        vehicle_helmet_detections = []

        for helmet in helmet_detections:

            if point_inside_box(
                helmet["center"],
                vehicle_box,
                margin=0.20
            ):

                vehicle_helmet_detections.append(
                    helmet
                )

        # Check ONLY this vehicle
        # for Without Helmet.

        without_helmet = False

        for helmet in vehicle_helmet_detections:

            if helmet["class"] == "Without Helmet":

                without_helmet = True
                break

        if without_helmet:

            vehicle["violations"].append(
                "Without Helmet"
            )

            without_helmet_count += 1

    st.metric(
        "⚠️ Vehicles Without Helmet",
        without_helmet_count
    )

    # ========================================================
    # TRIPLE RIDING DETECTION
    # ========================================================

    # Detect PERSON class using YOLO model.
    # COCO person class = 0.

    person_detections = []

    for result in vehicle_results:

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            if class_id != 0:
                continue

            x1, y1, x2, y2 = map(
                float,
                box.xyxy[0]
            )

            person_detections.append({

                "box": (
                    x1,
                    y1,
                    x2,
                    y2
                ),

                "center": get_center(
                    (
                        x1,
                        y1,
                        x2,
                        y2
                    )
                )
            })

    triple_riding_count = 0

    for vehicle in vehicles:

        if vehicle["type"] != "Motorcycle":
            continue

        rider_count = 0

        for person in person_detections:

            if point_inside_box(
                person["center"],
                vehicle["box"],
                margin=0.10
            ):

                rider_count += 1

        if rider_count >= 3:

            vehicle["violations"].append(
                "Triple Riding"
            )

            triple_riding_count += 1

    # ========================================================
    # NUMBER PLATE DETECTION
    # ========================================================

    st.subheader(
        "🔢 Number Plate Detection"
    )

    plate_results = plate_model.predict(
        source=image,
        conf=0.30,
        verbose=False
    )

    plate_image = plate_results[0].plot()

    st.image(
        plate_image,
        channels="BGR",
        width="stretch"
    )

    # ========================================================
    # OCR
    # ========================================================

    st.subheader(
        "🔍 Number Plate Recognition"
    )

    detected_plates = []

    for result in plate_results:

        for box in result.boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # Keep coordinates inside image

            x1 = max(
                0,
                x1
            )

            y1 = max(
                0,
                y1
            )

            x2 = min(
                image.shape[1],
                x2
            )

            y2 = min(
                image.shape[0],
                y2
            )

            plate = image[
                y1:y2,
                x1:x2
            ]

            if plate.size == 0:
                continue

            # =================================================
            # ENLARGE
            # =================================================

            plate = cv2.resize(
                plate,
                None,
                fx=8,
                fy=8,
                interpolation=cv2.INTER_CUBIC
            )

            # =================================================
            # GRAYSCALE
            # =================================================

            gray = cv2.cvtColor(
                plate,
                cv2.COLOR_BGR2GRAY
            )

            # =================================================
            # CONTRAST
            # =================================================

            gray = cv2.equalizeHist(
                gray
            )

            # =================================================
            # OCR
            # =================================================

            ocr_results = reader.readtext(
                gray,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                detail=1
            )

            best_text = ""
            best_confidence = 0.0

            for detection in ocr_results:

                text = detection[1]

                confidence = float(
                    detection[2]
                )

                text = (
                    text
                    .replace(" ", "")
                    .replace("-", "")
                    .upper()
                )

                if (
                    text
                    and
                    confidence > best_confidence
                ):

                    best_text = text

                    best_confidence = (
                        confidence
                    )

            plate_center = get_center(
                (
                    x1,
                    y1,
                    x2,
                    y2
                )
            )

            detected_plates.append({

                "plate": (
                    best_text
                    if best_text
                    else "Not Recognized"
                ),

                "confidence": (
                    best_confidence
                ),

                "center": plate_center,

                "box": (
                    x1,
                    y1,
                    x2,
                    y2
                )
            })

    # ========================================================
    # ASSOCIATE EACH PLATE WITH VEHICLE
    # ========================================================

    for plate_data in detected_plates:

        vehicle = associate_plate_to_vehicle(
            plate_data["center"],
            vehicles
        )

        if vehicle is not None:

            # Only replace if OCR produced text

            if (
                plate_data["plate"]
                != "Not Recognized"
            ):

                # Keep better OCR result

                if (
                    vehicle["plate"]
                    == "Not Recognized"
                    or
                    plate_data["confidence"]
                    >
                    vehicle["plate_confidence"]
                ):

                    vehicle["plate"] = (
                        plate_data["plate"]
                    )

                    vehicle[
                        "plate_confidence"
                    ] = (
                        plate_data["confidence"]
                    )

    # ========================================================
    # DISPLAY PLATES
    # ========================================================

    if detected_plates:

        for i, plate_data in enumerate(
            detected_plates,
            start=1
        ):

            if (
                plate_data["plate"]
                != "Not Recognized"
            ):

                st.success(
                    f"🚘 Plate {i}: "
                    f"{plate_data['plate']}"
                )

                st.write(
                    "OCR Confidence: "
                    f"{plate_data['confidence']:.2f}"
                )

            else:

                st.warning(
                    f"⚠️ Plate {i} detected, "
                    "but text could not be recognized."
                )

    else:

        st.warning(
            "⚠️ No number plate detected."
        )

    # ========================================================
    # VIOLATION + FINE DETAILS
    # ========================================================

    st.subheader(
        "🚨 Vehicle-wise Violation & Fine Details"
    )

    current_datetime = datetime.now()

    current_date = (
        current_datetime.strftime(
            "%Y-%m-%d"
        )
    )

    current_time = (
        current_datetime.strftime(
            "%H:%M:%S"
        )
    )

    violation_found = False

    # ========================================================
    # DISPLAY EACH VEHICLE
    # ========================================================

    for vehicle in vehicles:

        violations = vehicle["violations"]

        # No violation
        if not violations:

            continue

        violation_found = True

        total_fine = sum(
            FINE_AMOUNTS[v]
            for v in violations
        )

        st.error(
            f"🚨 VEHICLE {vehicle['id']} "
            f"VIOLATION DETECTED"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.write(
                f"🚗 **Vehicle Type:** "
                f"{vehicle['type']}"
            )

            st.write(
                f"🔢 **Number Plate:** "
                f"{vehicle['plate']}"
            )

            if vehicle[
                "plate"
            ] != "Not Recognized":

                st.write(
                    "🔍 **OCR Confidence:** "
                    f"{vehicle['plate_confidence']:.2f}"
                )

            st.write(
                "🚨 **Violation:** "
                +
                ", ".join(violations)
            )

        with col2:

            st.write(
                f"💰 **Total Fine:** "
                f"₹{total_fine}"
            )

            st.write(
                f"📅 **Date:** "
                f"{current_date}"
            )

            st.write(
                f"🕐 **Time:** "
                f"{current_time}"
            )

            st.write(
                f"📍 **Location:** "
                f"{location}"
            )

        st.divider()

    # ========================================================
    # NO VIOLATION
    # ========================================================

    if not violation_found:

        st.success(
            "✅ No supported traffic violation "
            "was detected."
        )

    # ========================================================
    # SAVE VIOLATIONS TO CSV
    # ========================================================

    if violation_found:

        csv_file = (
            "output/violations/"
            "violation_records.csv"
        )

        file_exists = os.path.exists(
            csv_file
        )

        with open(
            csv_file,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(
                file
            )

            if not file_exists:

                writer.writerow([
                    "Vehicle ID",
                    "Vehicle Type",
                    "Number Plate",
                    "Violation",
                    "Fine Amount",
                    "Date",
                    "Time",
                    "Location",
                    "OCR Confidence"
                ])

            for vehicle in vehicles:

                if not vehicle["violations"
                ]:

                    continue

                total_fine = sum(
                    FINE_AMOUNTS[v]
                    for v in vehicle[
                        "violations"
                    ]
                )

                writer.writerow([

                    vehicle["id"],

                    vehicle["type"],

                    vehicle["plate"],

                    ", ".join(
                        vehicle[
                            "violations"
                        ]
                    ),

                    total_fine,

                    current_date,

                    current_time,

                    location,

                    round(
                        vehicle[
                            "plate_confidence"
                        ],
                        2
                    )
                ])

        st.success(
            "✅ Vehicle-wise violation "
            "records saved successfully!"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    st.subheader(
        "📊 Detection Summary"
    )

    violating_vehicles = sum(
        1
        for vehicle in vehicles
        if vehicle["violations"]
    )

    total_fine_generated = sum(
        sum(
            FINE_AMOUNTS[v]
            for v in vehicle["violations"]
        )
        for vehicle in vehicles
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "🚗 Vehicles",
        len(vehicles)
    )

    c2.metric(
        "⚠️ Violating Vehicles",
        violating_vehicles
    )

    c3.metric(
        "💰 Total Fine",
        f"₹{total_fine_generated}"
    )