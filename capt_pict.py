import requests
import time
import os
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO  # Добавлена строка импорта YOLO

# Настройки
CONFIG_FILE = "C:/Projects/2025/python/capture_picture_from_ipcoamera/cameras.conf"  # Файл с IP-адресами камер
SAVE_FOLDER = "C:/Projects/2025/python/capture_picture_from_ipcoamera/images"  # Базовая папка для сохранения
INTERVAL_SEC = 1  # Интервал между снимками в секундах
MAX_WORKERS = 10  # Максимальное количество параллельных потоков
MODEL_PATH = "C:/Projects/2025/python/stew_bad_recog/models/yolo11s_with_dataset_160525/best.pt"  # Путь к модели YOLO

def load_camera_config(config_file):
    """Загружает список IP-адресов камер из файла конфигурации."""
    cameras = []
    try:
        with open(config_file, 'r') as f:
            for line in f:
                # Удаляем пробелы и символ перевода строки
                line = line.strip()
                # Пропускаем пустые строки и комментарии
                if line and not line.startswith('#'):
                    cameras.append(line)
        return cameras
    except Exception as e:
        print(f"Ошибка при чтении файла конфигурации: {str(e)}")
        return []

def capture_from_camera(camera_url, camera_index, yolo_model):
    """Функция для захвата изображений с одной камеры в отдельном потоке и их обработки YOLO."""
    # Создаем отдельную папку для этой камеры
    camera_folder = os.path.join(SAVE_FOLDER, f"camera_{camera_index}")
    if not os.path.exists(camera_folder):
        os.makedirs(camera_folder)
    
    # Создаем папку для результатов YOLO для этой камеры
    yolo_results_folder = os.path.join(camera_folder, "yolo_results")
    if not os.path.exists(yolo_results_folder):
        os.makedirs(yolo_results_folder)
    
    print(f"Начат захват с камеры {camera_index}: {camera_url}")
    
    # Формируем полный URL для получения кадра
    if not camera_url.endswith("/shot.jpg"):
        if not camera_url.endswith("/"):
            camera_url += "/"
        camera_url += "shot.jpg"
    
    while True:
        try:
            # Получаем текущее время для имени файла
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(camera_folder, f"capture_{timestamp}.jpg")
            
            # Загружаем изображение с камеры
            response = requests.get(camera_url, stream=True, timeout=10)
            
            if response.status_code == 200:
                # Сохраняем изображение
                with open(filename, 'wb') as f:
                    f.write(response.content)
                print(f"Камера {camera_index}: Сохранено в {filename}")
                
                # Обрабатываем изображение с помощью YOLO
                try:
                    start_time = time.time()  # Начало измерения времени обработки
                    
                    # Применяем модель YOLO к полученному изображению
                    results = yolo_model.predict(source=filename, save=False)  # Выключаем автосохранение YOLO
                    
                    # Сохраняем результаты в папку yolo_results
                    yolo_filename = os.path.join(yolo_results_folder, f"yolo_{timestamp}.jpg")
                    
                    # Сохраняем изображение с аннотациями (bounding boxes)
                    for r in results:
                        im_array = r.plot()  # Получаем изображение с аннотациями
                        # Сохраняем как изображение
                        import cv2
                        cv2.imwrite(yolo_filename, im_array)
                    
                    process_time = time.time() - start_time
                    print(f"Камера {camera_index}: YOLO обработка завершена за {process_time:.2f}с, сохранено в {yolo_filename}")
                    
                    # Сохраняем метаданные результатов в отдельный JSON файл (опционально)
                    if len(results) > 0 and len(results[0].boxes) > 0:
                        json_filename = os.path.join(yolo_results_folder, f"yolo_{timestamp}.json")
                        # Извлекаем и сохраняем данные о детекциях
                        detections = []
                        for r in results:
                            for box in r.boxes:
                                det_data = {
                                    "xyxy": box.xyxy.tolist() if hasattr(box, 'xyxy') else [],
                                    "conf": float(box.conf) if hasattr(box, 'conf') else 0,
                                    "cls": int(box.cls) if hasattr(box, 'cls') else -1,
                                    "cls_name": yolo_model.names[int(box.cls)] if hasattr(box, 'cls') else "unknown"
                                }
                                detections.append(det_data)
                        
                        # Сохраняем данные в JSON
                        import json
                        with open(json_filename, 'w') as f:
                            json.dump(detections, f, indent=4)
                
                except Exception as e:
                    print(f"Камера {camera_index}: Ошибка при обработке YOLO: {str(e)}")
            else:
                print(f"Камера {camera_index}: Ошибка HTTP {response.status_code}")
        
        except requests.exceptions.Timeout:
            print(f"Камера {camera_index}: Тайм-аут подключения")
        except requests.exceptions.ConnectionError:
            print(f"Камера {camera_index}: Ошибка соединения")
        except Exception as e:
            print(f"Камера {camera_index}: Ошибка: {str(e)}")
        
        # Ждем указанный интервал
        time.sleep(INTERVAL_SEC)

def capture_with_thread_pool(camera_urls, yolo_model):
    """Использует пул потоков для параллельной обработки камер."""
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(camera_urls))) as executor:
        # Запускаем задачи и сохраняем ссылки на них
        futures = []
        for index, url in enumerate(camera_urls):
            future = executor.submit(capture_from_camera, url, index, yolo_model)
            futures.append(future)
        
        # Ожидаем завершения всех задач (практически никогда не произойдет, 
        # так как функции работают в бесконечном цикле)
        for future in futures:
            try:
                future.result()  # Это заблокирует выполнение до завершения задачи
            except Exception as e:
                print(f"Ошибка в потоке: {str(e)}")

def capture_with_threads(camera_urls, yolo_model):
    """Запускает отдельный поток для каждой камеры."""
    threads = []
    for index, url in enumerate(camera_urls):
        thread = threading.Thread(
            target=capture_from_camera, 
            args=(url, index, yolo_model),
            daemon=True  # Делаем потоки фоновыми, чтобы они завершились при выходе из программы
        )
        threads.append(thread)
        thread.start()
    
    # Ожидание завершения всех потоков (практически не произойдет)
    for thread in threads:
        thread.join()

def main():
    # Создаем базовую папку для сохранения
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
    
    # Загружаем список URL камер
    camera_urls = load_camera_config(CONFIG_FILE)
    
    if not camera_urls:
        print(f"Не найдены URL камер в файле {CONFIG_FILE}")
        print("Формат файла: один URL на строку, например:")
        print("http://192.168.0.107:8080")
        print("http://192.168.0.108:8080")
        return
    
    print(f"Найдено {len(camera_urls)} камер в конфигурации.")
    
    # Загружаем модель YOLO
    print(f"Загрузка модели YOLO из {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)  # Загружаем модель YOLO
    print("Модель YOLO успешно загружена.")
    
    print(f"Начинаем параллельный захват и обработку YOLO. Изображения будут сохраняться в {SAVE_FOLDER}")
    
    try:
        # Выберите один из методов параллельного выполнения:
        # capture_with_threads(camera_urls, model)  # Используя чистые потоки
        capture_with_thread_pool(camera_urls, model)  # Используя пул потоков
    except KeyboardInterrupt:
        print("/nЗахват остановлен пользователем.")
    except Exception as e:
        print(f"Неожиданная ошибка: {str(e)}")

if __name__ == "__main__":
    main()