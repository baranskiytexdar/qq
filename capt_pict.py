import requests
import time
import os
from datetime import datetime

# Настройки
CAMERA_URL = "http://192.168.0.107:8080/shot.jpg"  # URL для получения кадра
SAVE_FOLDER = "C:/Projects/2025/python/capture_picture_from_ipcoamera/images"  # Папка для сохранения (можно изменить)
INTERVAL_SEC = 1  # Интервал между снимками в секундах

def capture_and_save():
    # Создаем папку, если ее нет
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
    
    while True:
        try:
            # Получаем текущее время для имени файла
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(SAVE_FOLDER, f"capture_{timestamp}.jpg")
            
            # Загружаем изображение с камеры
            response = requests.get(CAMERA_URL, stream=True, timeout=10)
            
            if response.status_code == 200:
                # Сохраняем изображение
                with open(filename, 'wb') as f:
                    f.write(response.content)
                print(f"Сохранено: {filename}")
            else:
                print(f"Ошибка: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"Ошибка: {str(e)}")
        
        # Ждем указанный интервал
        time.sleep(INTERVAL_SEC)

if __name__ == "__main__":
    print(f"Начинаем захват изображений с камеры. Сохранение в папку: {SAVE_FOLDER}")
    capture_and_save()