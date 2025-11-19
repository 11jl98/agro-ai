import cv2
import numpy as np
from PIL import Image, ImageEnhance
import config


class ImagePreprocessor:
    
    @staticmethod
    def remove_background(image_array):
        hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([90, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        result = cv2.bitwise_and(image_array, image_array, mask=mask)
        return result
    
    @staticmethod
    def enhance_contrast(image):
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(1.5)
    
    @staticmethod
    def normalize_lighting(image_array):
        lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    @staticmethod
    def denoise(image_array):
        return cv2.fastNlMeansDenoisingColored(image_array, None, 10, 10, 7, 21)
    
    @staticmethod
    def center_crop_leaf(image_array):
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image_array.shape[1] - x, w + 2*padding)
            h = min(image_array.shape[0] - y, h + 2*padding)
            return image_array[y:y+h, x:x+w]
        
        return image_array
    
    @staticmethod
    def preprocess_real_world_image(image_path, apply_background_removal=True):
        image = Image.open(image_path).convert('RGB')
        image = ImagePreprocessor.enhance_contrast(image)
        image_array = np.array(image)
        
        if apply_background_removal:
            image_array = ImagePreprocessor.remove_background(image_array)
        
        image_array = ImagePreprocessor.normalize_lighting(image_array)
        image_array = ImagePreprocessor.denoise(image_array)
        image_array = ImagePreprocessor.center_crop_leaf(image_array)
        
        image = Image.fromarray(image_array.astype('uint8'))
        image = image.resize(config.IMAGE_SIZE)
        
        return np.array(image) / 255.0