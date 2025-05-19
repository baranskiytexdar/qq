import cv2
import numpy as np
import matplotlib.pyplot as plt

def correct_chromatic_aberration_edges(image, threshold=50, blur_size=5, strength=0.8):
    """
    Коррекция хроматической аберрации с использованием детектирования краев.
    
    Parameters:
    - image: BGR изображение из OpenCV
    - threshold: Порог детектирования краев
    - blur_size: Размер ядра размытия для снижения шума
    - strength: Сила коррекции (0.0 - 1.0)
    
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
    
    # Находим края в зеленом канале (обычно наиболее четком)
    edges = cv2.Canny(g_blur, threshold, threshold * 2)
    
    # Расширяем области краев
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=2)
    
    # Создаем маску для областей с хроматической аберрацией
    # Находим разницу между каналами в областях краев
    rb_diff = cv2.absdiff(r_blur, b_blur)
    rg_diff = cv2.absdiff(r_blur, g_blur)
    bg_diff = cv2.absdiff(b_blur, g_blur)
    
    # Комбинируем разницы
    color_diff = cv2.add(cv2.add(rb_diff, rg_diff), bg_diff)
    
    # Применяем порог к разнице цветов для выделения аберраций
    _, aberration_mask = cv2.threshold(color_diff, 20, 255, cv2.THRESH_BINARY)
    
    # Применяем маску краев для ограничения областей коррекции
    aberration_mask = cv2.bitwise_and(aberration_mask, dilated_edges)
    
    # Преобразуем маску в float для весовой коррекции
    aberration_weight = aberration_mask.astype(np.float32) / 255.0 * strength
    
    # Создаем корректированные каналы
    # ИСПРАВЛЕНИЕ: Убедимся, что все каналы имеют одинаковый тип данных и размер
    r_corrected = np.uint8(r * (1 - aberration_weight) + g * aberration_weight)
    g_corrected = g.copy()  # Зеленый канал оставляем без изменений
    b_corrected = np.uint8(b * (1 - aberration_weight) + g * aberration_weight)
    
    # ИСПРАВЛЕНИЕ: Проверяем формы и типы массивов перед объединением
    assert r_corrected.shape == g_corrected.shape == b_corrected.shape, "Каналы имеют разные размеры"
    assert r_corrected.dtype == g_corrected.dtype == b_corrected.dtype, "Каналы имеют разные типы данных"
    
    # Объединяем каналы обратно
    corrected_rgb = cv2.merge([r_corrected, g_corrected, b_corrected])
    
    # Преобразуем обратно в BGR для OpenCV
    corrected = cv2.cvtColor(corrected_rgb, cv2.COLOR_RGB2BGR)
    
    return corrected, aberration_mask

# Пример использования
if __name__ == "__main__":
    # Загружаем изображение с хроматической аберрацией
    image_path = "C:/Projects/2025/python/del_aberation/1.jpeg"  # Укажите правильный путь к изображению
    image = cv2.imread(image_path)
    
    if image is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение по пути: {image_path}")
    
    # Применяем коррекцию
    corrected_image, aberration_mask = correct_chromatic_aberration_edges(image, threshold=30, strength=0.7)
    
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
    
    # Создаем увеличенный фрагмент для сравнения
    h, w = image.shape[:2]
    # Выберите интересную область с хроматической аберрацией
    x, y = w // 4, h // 4
    crop_size = min(200, h//2, w//2)  # Убедимся, что размер фрагмента не выходит за границы
    
    # Вырезаем фрагменты
    original_crop = image[y:y+crop_size, x:x+crop_size]
    corrected_crop = corrected_image[y:y+crop_size, x:x+crop_size]
    
    # Объединяем фрагменты для сравнения
    comparison = np.hstack((original_crop, corrected_crop))
    
    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB))
    plt.title("Сравнение фрагментов (до/после)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("C:/Projects/2025/python/del_aberation/edge_based_ca_correction.jpg", dpi=300)
    plt.show()
    
    # Сохраняем результат
    cv2.imwrite("C:/Projects/2025/python/del_aberation/edge_corrected_image.jpg", corrected_image)
    print("Изображение с исправленной хроматической аберрацией сохранено как 'edge_corrected_image.jpg'")