import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def estimate_vignette_params(image, samples=1000):
    """
    Оценивает параметры виньетирования на основе анализа изображения.
    
    Parameters:
    - image: Исходное изображение
    - samples: Количество семплов для анализа
    
    Returns:
    - Параметры виньетирования (radius, falloff)
    """
    # Преобразуем в grayscale, если цветное
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Размеры изображения
    height, width = gray.shape
    center_x, center_y = width // 2, height // 2
    
    # Создаем случайные точки для анализа
    np.random.seed(42)  # Для воспроизводимости результатов
    points_y = np.random.randint(0, height, samples)
    points_x = np.random.randint(0, width, samples)
    
    # Вычисляем расстояния до центра и соответствующие значения яркости
    distances = np.sqrt((points_x - center_x)**2 + (points_y - center_y)**2)
    brightnesses = gray[points_y, points_x].astype(np.float32)
    
    # Нормализуем расстояния и яркости
    max_dist = np.sqrt(center_x**2 + center_y**2)
    distances = distances / max_dist
    brightnesses = brightnesses / 255.0
    
    # Определяем функцию для оптимизации параметров
    def vignette_model(params):
        radius, falloff = params
        # Модель виньетирования: cos(dist * pi/2 * radius)^falloff
        model_values = np.cos(distances * np.pi/2 * radius)**falloff
        # Вычисляем ошибку (чем меньше, тем лучше подгонка модели)
        error = np.mean((brightnesses - model_values)**2)
        return error
    
    # Начальные параметры: radius=1.0, falloff=1.0
    initial_params = [1.0, 1.0]
    
    # Оптимизируем параметры
    result = minimize(vignette_model, initial_params, 
                     bounds=[(0.1, 2.0), (0.1, 5.0)],
                     method='L-BFGS-B')
    
    return result.x

def remove_vignette_advanced(image, params=None):
    """
    Продвинутый метод удаления виньетирования с автоматическим определением параметров.
    
    Parameters:
    - image: Исходное изображение
    - params: Параметры виньетирования (radius, falloff). Если None, они будут оценены автоматически.
    
    Returns:
    - Изображение с устраненным виньетированием
    """
    # Преобразуем в float для математических операций
    img_float = image.astype(np.float32) / 255.0
    
    # Оцениваем параметры виньетирования, если они не предоставлены
    if params is None:
        params = estimate_vignette_params(image)
    
    radius, falloff = params
    print(f"Используемые параметры виньетирования: radius={radius:.2f}, falloff={falloff:.2f}")
    
    # Получаем размеры изображения
    height, width = img_float.shape[:2]
    
    # Создаем координатную сетку
    y, x = np.mgrid[0:height, 0:width]
    
    # Находим центр изображения
    center_x, center_y = width // 2, height // 2
    
    # Вычисляем нормализованное расстояние от каждого пикселя до центра
    distances = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    max_distance = np.sqrt(center_x ** 2 + center_y ** 2)
    distances = distances / max_distance
    
    # Создаем маску виньетирования с учетом оцененных параметров
    vignette_mask = np.cos(distances * np.pi/2 * radius) ** falloff
    
    # Применяем коррекцию к каждому каналу
    if len(img_float.shape) == 3:  # Для цветных изображений
        vignette_mask = np.dstack([vignette_mask] * img_float.shape[2])
    
    # Коррекция виньетирования
    corrected = img_float / (vignette_mask + 1e-6)  # Избегаем деления на ноль
    
    # Клипируем значения для диапазона [0, 1]
    corrected = np.clip(corrected, 0.0, 1.0)
    
    # Преобразуем обратно в uint8
    corrected = (corrected * 255).astype(np.uint8)
    
    return corrected, params

# Пример использования
if __name__ == "__main__":
    # Загружаем изображение с виньетированием
    image_path = "C:/Projects/2025/python/del_vignette/1.jpg"
    image = cv2.imread(image_path)
    
    if image is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение по пути: {image_path}")
    
    # Преобразуем BGR в RGB для корректного отображения
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Применяем коррекцию виньетирования с автоматической оценкой параметров
    corrected_image, params = remove_vignette_advanced(image_rgb)
    
    # Отображаем результаты
    plt.figure(figsize=(15, 10))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Исходное изображение")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(corrected_image)
    plt.title(f"Коррекция с radius={params[0]:.2f}, falloff={params[1]:.2f}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("advanced_vignette_correction.jpg", dpi=300)
    plt.show()
    
    # Сохраняем параметры и результат
    np.save("vignette_params.npy", params)
    cv2.imwrite("advanced_corrected_image.jpg", cv2.cvtColor(corrected_image, cv2.COLOR_RGB2BGR))
    print("Изображение без виньетирования сохранено, а параметры записаны в 'vignette_params.npy'")