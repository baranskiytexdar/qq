import cv2
import numpy as np
import matplotlib.pyplot as plt

def correct_chromatic_aberration_laplacian(image, strength=0.5, blur_size=3):
    """
    Коррекция хроматической аберрации с использованием лаплассиана для определения границ.
    
    Parameters:
    - image: BGR изображение из OpenCV
    - strength: Сила коррекции (0.0 - 1.0)
    - blur_size: Размер ядра размытия для снижения шума
    
    Returns:
    - Изображение с уменьшенной хроматической аберрацией
    """
    # Преобразуем в RGB для удобства работы с каналами
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Разделяем изображение на каналы
    r, g, b = cv2.split(rgb)
    
    # Уменьшаем шум в каналах
    r_blur = cv2.GaussianBlur(r, (blur_size, blur_size), 0)
    g_blur = cv2.GaussianBlur(g, (blur_size, blur_size), 0)
    b_blur = cv2.GaussianBlur(b, (blur_size, blur_size), 0)
    
    # Вычисляем лаплассиан для каждого канала для определения границ
    r_laplacian = cv2.Laplacian(r_blur, cv2.CV_32F)
    g_laplacian = cv2.Laplacian(g_blur, cv2.CV_32F)
    b_laplacian = cv2.Laplacian(b_blur, cv2.CV_32F)
    
    # Преобразуем в абсолютные значения и нормализуем для лучшей визуализации
    r_edges = cv2.convertScaleAbs(r_laplacian)
    g_edges = cv2.convertScaleAbs(g_laplacian)
    b_edges = cv2.convertScaleAbs(b_laplacian)
    
    # Находим разницу между краями в разных каналах
    rg_diff = cv2.absdiff(r_edges, g_edges)
    rb_diff = cv2.absdiff(r_edges, b_edges)
    gb_diff = cv2.absdiff(g_edges, b_edges)
    
    # Комбинируем разницы для создания маски хроматической аберрации
    edge_diff = cv2.add(cv2.add(rg_diff, rb_diff), gb_diff)
    
    # Применяем порог для выделения значительных аберраций
    _, aberration_mask = cv2.threshold(edge_diff, 20, 255, cv2.THRESH_BINARY)
    
    # Расширяем маску для более плавной коррекции
    kernel = np.ones((3, 3), np.uint8)
    aberration_mask = cv2.dilate(aberration_mask, kernel, iterations=1)
    
    # Преобразуем маску для весовой коррекции
    aberration_weight = aberration_mask.astype(np.float32) / 255.0 * strength
    
    # Создаем средний канал как эталон (обычно зеленый канал наиболее четкий)
    reference = g.copy()
    
    # Корректируем красный и синий каналы к эталону с учетом весов
    r_corrected = r * (1 - aberration_weight) + reference * aberration_weight
    b_corrected = b * (1 - aberration_weight) + reference * aberration_weight
    
    # Объединяем каналы обратно
    corrected_rgb = cv2.merge([r_corrected.astype(np.uint8), 
                             g, 
                             b_corrected.astype(np.uint8)])
    
    # Преобразуем обратно в BGR для OpenCV
    corrected = cv2.cvtColor(corrected_rgb, cv2.COLOR_RGB2BGR)
    
    return corrected, aberration_mask

# Пример использования
if __name__ == "__main__":
    # Загружаем изображение с хроматической аберрацией
    image_path = "C:/Projects/2025/python/del_aberation/1.jpeg"
    image = cv2.imread(image_path)
    
    if image is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение по пути: {image_path}")
    
    # Применяем коррекцию
    corrected_image, aberration_mask = correct_chromatic_aberration_laplacian(image, strength=0.6)
    
    # Отображаем результаты
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Исходное изображение")
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB))
    plt.title("Скорректированное изображение")
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(aberration_mask, cmap='gray')
    plt.title("Маска хроматической аберрации")
    plt.axis('off')
    
    # Увеличенное сравнение до/после
    h, w = image.shape[:2]
    x, y = w // 3, h // 3  # Выберите интересную область
    crop_size = 200
    
    # Вырезаем фрагменты
    original_crop = image[y:y+crop_size, x:x+crop_size]
    corrected_crop = corrected_image[y:y+crop_size, x:x+crop_size]
    
    # Объединяем для сравнения
    comparison = np.hstack((original_crop, corrected_crop))
    
    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB))
    plt.title("Сравнение фрагментов (до/после)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("laplacian_ca_correction.jpg", dpi=300)
    plt.show()
    
    # Сохраняем результат
    cv2.imwrite("laplacian_corrected_image.jpg", corrected_image)
    print("Изображение с исправленной хроматической аберрацией сохранено как 'laplacian_corrected_image.jpg'")