import cv2
import numpy as np
import streamlit as st

# def find_rotation_angle(image):
#     # Функция поиска и удаления выбросов
#     def reject_outliers(data, m=1.1):
#         return data[abs(data - np.median(data)) < m * np.std(data)]
#
#     # Преобразование изображения в оттенки серого
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # Увеличение контрастности фото
#     clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
#     enhanced_contrast = clahe.apply(gray)
#     # Применение метода Гаусса для сглаживания изображения
#     blur = cv2.GaussianBlur(enhanced_contrast, (5, 5), 0)
#
#     # Выделение границ методом Canny
#     edges = cv2.Canny(blur, 80, 180) # 80, 180
#
#     # Определение линий с помощью преобразования Хафа
#     lines = cv2.HoughLinesP(edges, 1, np.pi/180,
#                             threshold=80, minLineLength=50, maxLineGap=10) # 80, 50, 10
#
#     # Вычисление угла поворота линий относительно горизонта
#     if lines is not None:
#         angles = []
#         for ind, line in enumerate(lines):
#             x1, y1, x2, y2 = line[0]
#             cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             angle = np.arctan2((y2 - y1), (x2 - x1)) * 180.0 / np.pi
#             angles.append(angle)
#         if len(angles) >= 2:
#             final_angles = reject_outliers(np.array(angles))
#         else:
#             final_angles = np.array(angles)
#         if len(final_angles):
#             median_angle = np.median(final_angles) #np.median(angles)
#         else:
#             median_angle = None
#     else:
#         median_angle = None
#     return image, median_angle

def preprocess_image(img):
    # Преобразуем изображение в серой масштаб
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Улучшение контраста с помощью линейной коррекции глобального среднего значения
    alpha = 1.2
    beta = -40
    clahe = cv2.createCLAHE(clipLimit=alpha, tileGridSize=(8, 8))
    enhanced_contrast = clahe.apply(gray)
    return enhanced_contrast


def detect_barcodes(enhanced_contrast):
    barcode_detector = cv2.QRCodeDetector()
    data, _, _ = barcode_detector.detectAndDecode(enhanced_contrast)
    print(data)
    if (data is not None) and (data != ''):
        print('barcode_detector.points:', barcode_detector.points)
        x, y, w, h = barcode_detector.points[0].decodeData()[::-2]
        roi = enhanced_contrast[y:y + h, x:x + w]
        return roi
    else:
        return None


def blur_roi(roi, kernel_size=7):
    blurred_roi = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
    return blurred_roi


def mask_roi(blurred_roi, enhanced_contrast, threshold=1.5):
    _, binary_mask = cv2.threshold(blurred_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_thresh = int(np.mean([c.area for c in contours]) * threshold)
    filtered_contours = [c for c in contours if
                         c.area > area_thresh and cv2.contourArea(c) / cv2.boundingRect(c)[2] < 0.9]
    if len(filtered_contours) != 0:
        largest_contour = max(filtered_contours, key=lambda x: cv2.contourArea(x))
        x, y, w, h = cv2.boundingRect(largest_contour)
        mask = np.zeros_like(enhanced_contrast)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        return mask
    else:
        return None


def apply_mask(edges, mask):
    if mask is not None:
        masked_edges = cv2.bitwise_and(edges, edges, mask=mask)
        return masked_edges
    else:
        return edges


def canny_edge_detection(enhanced_contrast, shape):
    if shape <= 180:
        threshold1 = 70
        threshold2 = 150
    else:
        threshold1 = 80
        threshold2 = 180
    edges = cv2.Canny(enhanced_contrast,
                      threshold1=threshold1, threshold2=threshold2)
    roi = detect_barcodes(enhanced_contrast)
    if roi is not None:
        blurred_roi = blur_roi(roi)
        mask = mask_roi(blurred_roi, enhanced_contrast)
        edges = apply_mask(edges, mask)
    return edges


def blur_image(im):
    # Преобразование изображения в оттенки серого
    gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # Применение фильтра Гаусса для уменьшения бликов
    blurred_image = cv2.GaussianBlur(gray_image, (0, 0), 10)
    # Восстановление цветового изображения
    restored_image = cv2.cvtColor(blurred_image, cv2.COLOR_GRAY2BGR)
    return restored_image

def find_rotation_angle(image):
    # Функция поиска и удаления выбросов
    def reject_outliers(data, m=1.1):
        return data[abs(data - np.median(data)) < m * np.std(data)]

    # Преобразование изображения в оттенки серого
    shape = image.shape[0]
    st.write('shape:', shape)
    gray = preprocess_image(image)
    edges = canny_edge_detection(gray, shape)
    if shape <= 180:
        threshold = 70
        minLineLength = 70
    else:
        threshold = 100
        minLineLength = 100
    # Определение линий с помощью преобразования Хафа
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=threshold,
                            minLineLength=minLineLength,
                            maxLineGap=10)  # 100, 100, 10

    # Вычисление угла поворота линий относительно горизонта
    if lines is not None:
        angles = []
        for ind, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            angle = np.arctan2((y2 - y1), (x2 - x1)) * 180.0 / np.pi
            angles.append(angle)
        if len(angles) >= 2:
            final_angles = reject_outliers(np.array(angles))
        else:
            final_angles = np.array(angles)
        if len(final_angles):
            median_angle = np.median(final_angles)  # np.median(angles)
        else:
            median_angle = None
    else:
        median_angle = None
    return image, median_angle

def rotate_image(image, angle):
    # Поворот исходного изображения на найденный угол
    if angle is not None:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h),
                                    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    else:
        rotated_image = image
    return rotated_image