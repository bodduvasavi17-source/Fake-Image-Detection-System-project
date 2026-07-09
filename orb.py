import cv2
import numpy as np

def detect_copy_move(image_path):
    # 1. Rewind the file
    image_path.seek(0)
    
    # 2. Read the image
    file_bytes = np.asarray(bytearray(image_path.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 0)
    
    if img is None:
        return 0

    # 3. Actual ORB Logic
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(img, None)
    
    # If no descriptors are found, return 0
    if des is None:
        return 0

    # 4. Match the image against itself to find copied areas
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des, des)
    
    # We filter out matches that are just the exact same pixel (distance 0)
    real_matches = [m for m in matches if m.distance > 0]
    
    # 5. Return the count of matches
    return len(real_matches)
