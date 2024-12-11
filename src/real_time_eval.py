from transformers import AutoModelForObjectDetection, AutoImageProcessor, pipeline
import torch
from PIL import ImageDraw, Image
import numpy as np
import cv2
import time

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def plot_results(image, results, threshold=0.7):
    image = Image.fromarray(np.uint8(image))
    draw = ImageDraw.Draw(image)
    for result in results:
        score = result["score"]
        label = result["label"]
        box = list(result["box"].values())
        if score > threshold:
            x, y, x2, y2 = tuple(box)
            draw.rectangle((x, y, x2, y2), outline="red", width=1)
            draw.text((x, y), label, fill="white")
            draw.text(
                (x + 0.5, y - 0.5),
                text=str(score),
                fill="green" if score > 0.7 else "red",
            )
    return image


def main():
    model_path = r"C:\Users\isaac\dev\CV_Garbage_Detection\Models\conditional-detr-200-epochs-new-data"

    model = AutoModelForObjectDetection.from_pretrained(model_path)
    processor = AutoImageProcessor.from_pretrained(model_path)
    obj = pipeline("object-detection", model=model, image_processor=processor,device=device)
    print("Model loaded")
    
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    print("Press 'q' to quit.")

    # Real-time processing loop
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        cv2.imshow("Original Camera Feed", frame)

        # Convert the frame to RGB for inference
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        # Run inference on the frame
        results = obj(pil_img)

        # Use the plot_results function to annotate the frame
        annotated_image = plot_results(rgb_frame, results, threshold=0.7)

        # Convert the PIL Image back to OpenCV format (BGR) for display
        annotated_frame = np.array(annotated_image)
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        print(f"FPS: {fps:.2f}")

        # Display the frame
        cv2.imshow("Real-Time Object Detection", annotated_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
