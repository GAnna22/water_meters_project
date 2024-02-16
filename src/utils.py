import cv2
import numpy as np

def find_rotation_angle(image):
    # Функция поиска и удаления выбросов
    def reject_outliers(data, m=1.1):
        return data[abs(data - np.median(data)) < m * np.std(data)]
    
    # Преобразование изображения в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Применение метода Гаусса для сглаживания изображения
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Выделение границ методом Canny
    edges = cv2.Canny(blur, 80, 180)

    # Определение линий с помощью преобразования Хафа
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=50, maxLineGap=10)

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
            median_angle = np.median(final_angles) #np.median(angles)
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