import cv2
import numpy as np
from datetime import datetime

# Загрузка изображения
input_path = "C:/Projects/2025/python/del_distorsion/3717.jpeg"
img = cv2.imread(input_path)

if img is None:
    raise FileNotFoundError(f"Не удалось загрузить изображение по пути: {input_path}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Параметры шахматной доски (внутренние углы)
pattern_size = (6, 9)  # (ширина, высота) - количество внутренних углов
ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

if ret:
    # Уточнение положения углов
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    
    # Создание массива 3D точек (объектных точек)
    # Ключевое изменение: правильное создание трехмерного массива точек
    objp = np.zeros((pattern_size[0] * pattern_size[1], 1, 3), np.float32)
    objp[:, 0, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    
    # Для fisheye.calibrate нужно передавать списки списков точек
    obj_points = [objp]  # Должен быть список массивов
    img_points = [corners.reshape(-1, 1, 2)]  # Должен быть список массивов, правильно форматированных
    
    # Инициализация параметров камеры
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    
    # Калибровка
    try:
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            obj_points,
            img_points,
            gray.shape[::-1],  # (width, height)
            K,
            D,
            flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
        
        print(f"Калибровка успешна. RMS = {rms}")
        print(f"Матрица камеры K:\n{K}")
        print(f"Коэффициенты дисторсии D:\n{D}")
        
        # Коррекция дисторсии
        undistorted = cv2.fisheye.undistortImage(img, K, D, Knew=K)
        
        # Сохранение результатов
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"C:/Projects/2025/python/del_distorsion/undistorted_{timestamp}.jpg"
        cv2.imwrite(output_path, undistorted)
        
        # Сохранение параметров калибровки
        np.savez(
            f"C:/Projects/2025/python/del_distorsion/calibration_{timestamp}.npz",
            K=K, D=D, rms=rms
        )
        
        print(f"Результат сохранен в: {output_path}")
        
    except Exception as e:
        print(f"Ошибка при калибровке: {str(e)}")
        # Выводим более подробную информацию об ошибке для отладки
        import traceback
        traceback.print_exc()
        
        # Печать информации о форме и типе данных
        print(f"Форма objp: {obj_points[0].shape}, тип: {obj_points[0].dtype}")
        print(f"Форма corners: {img_points[0].shape}, тип: {img_points[0].dtype}")
else:
    print("Углы шахматной доски не найдены. Проверьте:")
    print("- Видимость всех углов на изображении")
    print("- Правильность pattern_size (должны быть внутренние углы)")
    print("- Контрастность изображения")
    
    # Визуализация изображения с найденными углами (если есть) для отладки
    debug_img = cv2.drawChessboardCorners(img.copy(), pattern_size, corners if ret else np.array([]), ret)
    debug_path = "C:/Projects/2025/python/del_distorsion/debug_corners.jpg"
    cv2.imwrite(debug_path, debug_img)
    print(f"Отладочное изображение сохранено в: {debug_path}")

# Калибровка успешна. RMS = 0.38014544632888914
# Матрица камеры K:
# [[1.61491391e+03 7.30706032e+00 7.61585340e+02]
#  [0.00000000e+00 1.55626792e+03 5.62851364e+02]
#  [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
# Коэффициенты дисторсии D:
# [[-0.45328041]
#  [ 0.08986565]
#  [-2.66772286]
#  [15.47672051]]
# Результат сохранен в: C:/Projects/2025/python/del_distorsion/undistorted_20250515_141138.jpg