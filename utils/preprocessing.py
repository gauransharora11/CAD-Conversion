import cv2
import os

def create_edge_dataset(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            path = os.path.join(input_folder, file)
            img = cv2.imread(path)

            # Resize
            img = cv2.resize(img, (512, 512))

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Remove noise
            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            # Edge detection
            edges = cv2.Canny(blur, 50, 150)

            cv2.imwrite(os.path.join(output_folder, file), edges)

if __name__ == "__main__":
    create_edge_dataset("data/photos", "data/edges")
