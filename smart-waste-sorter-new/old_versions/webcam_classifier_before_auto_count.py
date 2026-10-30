import cv2
import csv
from ultralytics import YOLO

# Load the trained AI model
model_path = (
    "runs/classify/runs/classify/"
    "smart_waste_six/weights/best.pt"
)

model = YOLO(model_path)

# Instructions for each waste category
class_instructions = {
    "Organic": "Use the organic waste bin.",
    "Glass": "Use the glass recycling bin.",
    "Can_Metal": "Use the metal recycling bin.",
    "Other": "Check local disposal instructions.",
    "Paper": "Use the paper recycling bin.",
    "Plastic": "Use the plastic recycling bin.",
}

# Minimum confidence required
CONFIDENCE_THRESHOLD = 60.0

# Prevent counting the same item more than once
item_counted = False

# Statistics for six categories
item_counts = {
    "Organic": 0,
    "Glass": 0,
    "Can_Metal": 0,
    "Other": 0,
    "Paper": 0,
    "Plastic": 0,
}


# Save statistics to a CSV file
def save_statistics():
    with open(
        "waste_statistics.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["Category", "Count"])

        for name, count in item_counts.items():
            writer.writerow([name, count])

        writer.writerow(["TOTAL", sum(item_counts.values())])


# Open the webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit

print("Smart Waste Sorter started!")
print("Show ONE waste item to the camera.")
print("Press C to count the item once.")
print("Remove the item, then press R for the next item.")
print("Press S to show statistics.")
print("Press Q to quit.")

# Initial prediction values
category = "Uncertain"
confidence = 0.0
instruction = "Show one item clearly."
status = "Waiting for prediction"

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read webcam image.")
            break

        # Predict the waste category
        results = model.predict(
            frame,
            imgsz=224,
            verbose=False
        )

        result = results[0]
        class_id = int(result.probs.top1)
        predicted_category = result.names[class_id]
        confidence = float(result.probs.top1conf) * 100

        # Check confidence before accepting the category
        if confidence >= CONFIDENCE_THRESHOLD:
            category = predicted_category
            status = "Waste classified"
            instruction = class_instructions.get(
                category,
                "Check local disposal instructions."
            )
        else:
            category = "Uncertain"
            status = "Try better lighting"
            instruction = "Please check manually."

        # Display category
        cv2.putText(
            frame,
            f"Category: {category}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if category != "Uncertain"
            else (0, 165, 255),
            2,
        )

        # Display confidence
        cv2.putText(
            frame,
            f"Confidence: {confidence:.1f}%",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Display status
        cv2.putText(
            frame,
            status,
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # Display sorting instruction
        cv2.putText(
            frame,
            instruction,
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            2,
        )

        # Display counting status
        count_status = (
            "Item already counted"
            if item_counted
            else "Ready to count"
        )

        cv2.putText(
            frame,
            count_status,
            (20, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255) if item_counted
            else (0, 255, 0),
            2,
        )

        cv2.imshow("Smart Waste Sorter", frame)

        # Read keyboard input
        key = cv2.waitKey(1) & 0xFF

        # C: Count the current item once
        if key == ord("c"):
            if item_counted:
                print(
                    "Already counted! Remove the item "
                    "and press R for the next item."
                )

            elif (
                confidence >= CONFIDENCE_THRESHOLD
                and category in item_counts
            ):
                item_counts[category] += 1
                item_counted = True
                save_statistics()

                print(
                    f"Counted {category}. "
                    f"Total: {sum(item_counts.values())}"
                )

            else:
                print("Uncertain item. Not counted.")

        # R: Unlock counting for the next item
        elif key == ord("r"):
            item_counted = False
            print("Ready to count the next item.")

        # S: Display statistics
        elif key == ord("s"):
            print("\n--- Waste Statistics ---")

            for name, count in item_counts.items():
                print(f"{name}: {count}")

            print(f"Total items: {sum(item_counts.values())}")
            print("------------------------")

        # Q: Quit the program
        elif key == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("Smart Waste Sorter stopped.")