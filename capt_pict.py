import requests
import time
import os
import threading
import queue
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import configparser

# Значения по умолчанию (будут использоваться, если параметры отсутствуют в конфигурационном файле)
DEFAULT_CONFIG = {
    "General": {
        "config_file": "cameras.conf",
        "save_folder": "C:/Projects/2025/python/capture_picture_from_ipcoamera/images",
        "interval_sec": "1",
        "max_workers": "10"
    },
    "YOLO": {
        "model_path": "C:/Projects/2025/python/stew_bad_recog/models/yolo11s_with_dataset_160525/best.pt"
    },
    "UI": {
        "thumbnail_width": "320",
        "thumbnail_height": "240",
        "ui_update_interval": "100",
        "grid_columns": "2"
    }
}
CONFIG_INI_FILE = "C:/Projects/2025/python/qq/config.conf"

# Глобальные настройки (будут заполнены из файла конфигурации)
CONFIG = {
    "config_file": "",
    "save_folder": "",
    "interval_sec": 1,
    "max_workers": 10,
    "model_path": "",
    "thumbnail_width": 320,
    "thumbnail_height": 240,
    "ui_update_interval": 100,
    "grid_columns": 2
}
def load_config():
    """Загружает настройки из файла конфигурации."""
    config = configparser.ConfigParser()
    
    # Если файл конфигурации не существует, создаем его с настройками по умолчанию
    if not os.path.exists(CONFIG_INI_FILE):
        print(f"Файл конфигурации {CONFIG_INI_FILE} не найден. Создаю новый с настройками по умолчанию.")
        for section in DEFAULT_CONFIG:
            config[section] = DEFAULT_CONFIG[section]
            
        with open(CONFIG_INI_FILE, 'w') as configfile:
            config.write(configfile)
    else:
        # Загружаем существующий файл конфигурации
        config.read(CONFIG_INI_FILE)
        print(f"Загружен файл конфигурации: {CONFIG_INI_FILE}")
    
    # Заполняем глобальные настройки из файла конфигурации
    # Если какая-то настройка отсутствует, используем значение по умолчанию
    global CONFIG
    
    # Настройки из секции General
    if 'General' in config:
        CONFIG["config_file"] = config.get('General', 'config_file', fallback=DEFAULT_CONFIG['General']['config_file'])
        CONFIG["save_folder"] = config.get('General', 'save_folder', fallback=DEFAULT_CONFIG['General']['save_folder'])
        CONFIG["interval_sec"] = config.getfloat('General', 'interval_sec', fallback=float(DEFAULT_CONFIG['General']['interval_sec']))
        CONFIG["max_workers"] = config.getint('General', 'max_workers', fallback=int(DEFAULT_CONFIG['General']['max_workers']))
    
    # Настройки из секции YOLO
    if 'YOLO' in config:
        CONFIG["model_path"] = config.get('YOLO', 'model_path', fallback=DEFAULT_CONFIG['YOLO']['model_path'])
    
    # Настройки из секции UI
    if 'UI' in config:
        CONFIG["thumbnail_width"] = config.getint('UI', 'thumbnail_width', fallback=int(DEFAULT_CONFIG['UI']['thumbnail_width']))
        CONFIG["thumbnail_height"] = config.getint('UI', 'thumbnail_height', fallback=int(DEFAULT_CONFIG['UI']['thumbnail_height']))
        CONFIG["ui_update_interval"] = config.getint('UI', 'ui_update_interval', fallback=int(DEFAULT_CONFIG['UI']['ui_update_interval']))
        CONFIG["grid_columns"] = config.getint('UI', 'grid_columns', fallback=int(DEFAULT_CONFIG['UI']['grid_columns']))
    
    print("Загружены следующие настройки:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    return CONFIG


# Очереди для передачи изображений между потоками (по одной на камеру)
image_queues = {}

# Флаги для управления работой потоков захвата
camera_active = {}

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

def resize_image_with_aspect_ratio(image, width=None, height=None):
    """Изменяет размер изображения, сохраняя соотношение сторон."""
    if image is None:
        # Создаем пустое черное изображение требуемого размера
        return np.zeros((height or 240, width or 320, 3), dtype=np.uint8)
        
    h, w = image.shape[:2]
    
    if width and height:
        return cv2.resize(image, (width, height))
    
    if width:
        aspect_ratio = width / float(w)
        dim = (width, int(h * aspect_ratio))
    else:
        aspect_ratio = height / float(h)
        dim = (int(w * aspect_ratio), height)
    
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def draw_text_with_background(img, text, pos, font=cv2.FONT_HERSHEY_SIMPLEX, 
                             font_scale=0.6, text_color=(255, 255, 255), 
                             bg_color=(0, 0, 0), thickness=1, padding=5):
    """Рисует текст с фоном на изображении."""
    # Получаем размер текста
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Координаты фонового прямоугольника
    x, y = pos
    rec_x1 = x - padding
    rec_y1 = y - text_height - padding
    rec_x2 = x + text_width + padding
    rec_y2 = y + padding
    
    # Рисуем фоновый прямоугольник
    cv2.rectangle(img, (rec_x1, rec_y1), (rec_x2, rec_y2), bg_color, -1)
    
    # Рисуем текст
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)
    
    return img

def capture_from_camera(camera_url, camera_index, yolo_model):
    """Функция для захвата изображений с одной камеры в отдельном потоке и их обработки YOLO."""
    global camera_active
    
    # Используем настройки из CONFIG
    save_folder = CONFIG["save_folder"]
    interval_sec = CONFIG["interval_sec"]
    
    # Создаем отдельную папку для этой камеры
    camera_folder = os.path.join(save_folder, f"camera_{camera_index}")
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
    
    while camera_active[camera_index]:
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
                
                # Преобразуем изображение из байтов в OpenCV формат
                img_array = np.frombuffer(response.content, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                print(f"Камера {camera_index}: Сохранено в {filename}")
                
                # Обрабатываем изображение с помощью YOLO
                try:
                    start_time = time.time()  # Начало измерения времени обработки
                    
                    # Применяем модель YOLO к полученному изображению
                    results = yolo_model.predict(source=img, save=False)  # Передаем изображение напрямую
                    
                    has_detections = False
                    
                    # Проверяем, есть ли обнаружения
                    if len(results) > 0 and len(results[0].boxes) > 0:
                        has_detections = True
                    
                    # Получаем изображение с аннотациями
                    for r in results:
                        annotated_img = r.plot()  # Изображение с отображенными bounding boxes
                    
                    # Сохраняем результаты в папку yolo_results
                    yolo_filename = os.path.join(yolo_results_folder, f"yolo_{timestamp}.jpg")
                    cv2.imwrite(yolo_filename, annotated_img)
                    
                    process_time = time.time() - start_time
                    
                    # Готовим изображение для отображения
                    display_img = annotated_img.copy()
                    
                    # Добавляем информацию на изображение
                    camera_text = f"Камера {camera_index}"
                    time_text = f"Время: {timestamp}"
                    process_text = f"YOLO: {process_time:.2f}с"
                    detection_text = "Дефекты ОБНАРУЖЕНЫ!" if has_detections else "Дефектов нет"
                    
                    # Рисуем информацию на изображении с меньшим масштабом для миниатюр
                    display_img = draw_text_with_background(display_img, camera_text, (10, 20))
                    display_img = draw_text_with_background(display_img, time_text, (10, 40))
                    display_img = draw_text_with_background(display_img, process_text, (10, 60))
                    
                    # Рисуем информацию о детекции с цветом в зависимости от результата
                    bg_color = (0, 0, 255) if has_detections else (0, 128, 0)  # Красный если есть дефекты, зеленый если нет
                    display_img = draw_text_with_background(display_img, detection_text, 
                                                           (10, 80), bg_color=bg_color)
                    
                    # Помещаем изображение в очередь для отображения в UI
                    try:
                        image_queues[camera_index].put({
                            'camera_index': camera_index,
                            'image': display_img,
                            'timestamp': timestamp,
                            'has_detections': has_detections
                        }, block=False)  # Неблокирующая операция
                    except queue.Full:
                        # Если очередь заполнена, пропускаем добавление (не блокируем поток)
                        pass
                    
                    print(f"Камера {camera_index}: YOLO обработка завершена за {process_time:.2f}с, сохранено в {yolo_filename}")
                    
                    # Сохраняем метаданные результатов
                    if has_detections:
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
        time.sleep(interval_sec)
    
    print(f"Захват с камеры {camera_index} остановлен.")

class MultiCameraMonitorApp:
    def __init__(self, root, camera_urls):
        self.root = root
        self.root.title("Система мониторинга дефектов - Параллельный режим")
        
        # Получаем размер экрана для более оптимального отображения
    # Используем настройки из CONFIG
        thumbnail_width = CONFIG["thumbnail_width"]
        thumbnail_height = CONFIG["thumbnail_height"]
        grid_columns = CONFIG["grid_columns"]

        # Получаем размер экрана для более оптимального отображения
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()    

        # Устанавливаем размер окна
        window_width = min(screen_width - 100, thumbnail_width * grid_columns + 40)
        window_height = min(screen_height - 100, 
                           (thumbnail_height * ((len(camera_urls) + grid_columns - 1) // grid_columns)) + 100)
        
        self.root.geometry(f"{window_width}x{window_height}")
        
        self.camera_urls = camera_urls
        self.num_cameras = len(camera_urls)
        
        # Создаем флаги активности камер
        global camera_active
        for i in range(self.num_cameras):
            camera_active[i] = True
        
        # Создаем очереди для изображений
        global image_queues
        for i in range(self.num_cameras):
            image_queues[i] = queue.Queue(maxsize=3)
        
        # Фрейм для кнопок управления
        control_frame = ttk.Frame(root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Метка статуса
        self.status_var = tk.StringVar(value="Система мониторинга запущена")
        status_label = ttk.Label(control_frame, textvariable=self.status_var, font=("Arial", 10, "bold"))
        status_label.pack(side=tk.LEFT, padx=5)
        
        # Кнопка для выхода
        exit_button = ttk.Button(control_frame, text="Выход", command=self.on_exit)
        exit_button.pack(side=tk.RIGHT, padx=5)
        
        # Создаем главный фрейм
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Создаем холст с полосой прокрутки для размещения камер
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Добавляем вертикальную полосу прокрутки
        scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Создаем холст
        self.canvas = tk.Canvas(canvas_frame, yscrollcommand=scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Настраиваем scrollbar
        scrollbar.config(command=self.canvas.yview)
        
        # Создаем фрейм внутри холста для размещения сетки камер
        self.cameras_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.cameras_frame, anchor=tk.NW)
        
        # Создаем сетку для отображения камер
        self.setup_camera_grid()
        
        # Настраиваем прокрутку
        self.cameras_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        
        # Запускаем обновление UI
        self.update_display()
    
    def setup_camera_grid(self):
        """Создаем сетку для отображения всех камер одновременно."""
        grid_columns = CONFIG["grid_columns"]

        self.camera_frames = []
        self.camera_labels = []
        self.camera_buttons = []
        self.camera_vars = []
        
        # Определяем количество столбцов для сетки
        columns = grid_columns
        rows = (self.num_cameras + columns - 1) // columns  # Округляем вверх
        
        # Создаем фреймы и метки для каждой камеры
        for i in range(self.num_cameras):
            row = i // columns
            col = i % columns
            
            # Создаем фрейм для камеры
            frame = ttk.LabelFrame(self.cameras_frame, text=f"Камера {i}")
            frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            
            # Создаем метку для отображения изображения
            label = ttk.Label(frame)
            label.pack(padx=2, pady=2)
            
            # Создаем фрейм для элементов управления камерой
            control_frame = ttk.Frame(frame)
            control_frame.pack(fill=tk.X, padx=2, pady=2)
            
            # Создаем переменную и чекбокс для включения/выключения камеры
            var = tk.BooleanVar(value=True)
            check = ttk.Checkbutton(control_frame, text="Активна", 
                                   variable=var, command=lambda idx=i: self.toggle_camera(idx))
            check.pack(side=tk.LEFT, padx=2)
            
            # Создаем кнопку для просмотра в полный размер
            button = ttk.Button(control_frame, text="Просмотр", 
                               command=lambda idx=i: self.open_fullscreen(idx))
            button.pack(side=tk.RIGHT, padx=2)
            
            # Сохраняем ссылки на элементы
            self.camera_frames.append(frame)
            self.camera_labels.append(label)
            self.camera_buttons.append(button)
            self.camera_vars.append(var)
        
        # Настраиваем веса строк и столбцов для равномерного распределения
        for i in range(rows):
            self.cameras_frame.grid_rowconfigure(i, weight=1)
        for i in range(columns):
            self.cameras_frame.grid_columnconfigure(i, weight=1)
    
    def update_display(self):
        """Обновляет все миниатюры камер."""
        
        # Используем настройки из CONFIG
        ui_update_interval = CONFIG["ui_update_interval"]
        thumbnail_width = CONFIG["thumbnail_width"]
        thumbnail_height = CONFIG["thumbnail_height"]

        # Обновляем изображения для всех камер
        for camera_idx in range(self.num_cameras):
            # Проверяем, есть ли новое изображение в очереди этой камеры
            try:
                if not image_queues[camera_idx].empty():
                    # Получаем новое изображение
                    image_data = image_queues[camera_idx].get_nowait()
                    
                    img = image_data['image']
                    has_detections = image_data['has_detections']
                    
                    # Изменяем размер для отображения в миниатюре
                    thumbnail = resize_image_with_aspect_ratio(img, width=THUMBNAIL_WIDTH, height=THUMBNAIL_HEIGHT)
                    
                    # Добавляем индикатор обнаружения дефекта
                    if has_detections:
                        # Рисуем красную рамку вокруг миниатюры
                        cv2.rectangle(thumbnail, (0, 0), (THUMBNAIL_WIDTH-1, THUMBNAIL_HEIGHT-1), (0, 0, 255), 3)
                    
                    # Преобразуем в формат Tkinter
                    thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(thumbnail)
                    tk_img = ImageTk.PhotoImage(image=pil_img)
                    
                    # Обновляем изображение в соответствующей метке
                    self.camera_labels[camera_idx].config(image=tk_img)
                    self.camera_labels[camera_idx].image = tk_img  # Сохраняем ссылку
                    
                    # Обновляем заголовок фрейма камеры
                    detection_status = "🔴 ДЕФЕКТ!" if has_detections else "✓ Нет дефектов"
                    self.camera_frames[camera_idx].config(text=f"Камера {camera_idx} - {detection_status}")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Ошибка при обновлении дисплея для камеры {camera_idx}: {str(e)}")
        
        # Планируем следующее обновление
        self.root.after(ui_update_interval, self.update_display)
    
    def toggle_camera(self, camera_idx):
        """Включает или выключает выбранную камеру."""
        global camera_active
        is_active = self.camera_vars[camera_idx].get()
        camera_active[camera_idx] = is_active
        
        status = "включена" if is_active else "отключена"
        self.status_var.set(f"Камера {camera_idx} {status}")
        
        if not is_active:
            # Если камера отключена, очищаем изображение
            # Создаем пустое изображение с сообщением
            img = np.zeros((THUMBNAIL_HEIGHT, THUMBNAIL_WIDTH, 3), dtype=np.uint8)
            cv2.putText(img, f"Камера {camera_idx} отключена", (10, THUMBNAIL_HEIGHT // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Преобразуем в формат Tkinter
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            tk_img = ImageTk.PhotoImage(image=pil_img)
            
            # Обновляем изображение
            self.camera_labels[camera_idx].config(image=tk_img)
            self.camera_labels[camera_idx].image = tk_img
            
            # Обновляем заголовок фрейма
            self.camera_frames[camera_idx].config(text=f"Камера {camera_idx} - ОТКЛЮЧЕНА")
    
    def open_fullscreen(self, camera_idx):
        """Открывает выбранную камеру в отдельном окне для просмотра в полный размер."""
        try:
            # Проверяем, активна ли камера
            if not camera_active[camera_idx]:
                messagebox.showinfo("Информация", f"Камера {camera_idx} отключена. Включите её перед просмотром.")
                return
            
            # Создаем новое окно для просмотра
            fullscreen_window = tk.Toplevel(self.root)
            fullscreen_window.title(f"Камера {camera_idx} - Полноразмерный просмотр")
            fullscreen_window.geometry("1024x768")
            
            # Создаем метку для отображения изображения
            image_label = ttk.Label(fullscreen_window)
            image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Панель управления
            control_frame = ttk.Frame(fullscreen_window)
            control_frame.pack(fill=tk.X, padx=10, pady=5)
            
            # Информационная метка
            info_label = ttk.Label(control_frame, text="Ожидание изображения...", font=("Arial", 10))
            info_label.pack(side=tk.LEFT, padx=5)
            
            # Кнопка закрытия
            close_button = ttk.Button(control_frame, text="Закрыть", 
                                     command=fullscreen_window.destroy)
            close_button.pack(side=tk.RIGHT, padx=5)
            
            # Функция обновления изображения в полноэкранном режиме
            def update_fullscreen():
                try:
                    # Проверяем активность камеры
                    if not camera_active[camera_idx]:
                        info_label.config(text=f"Камера {camera_idx} отключена")
                        fullscreen_window.after(500, update_fullscreen)
                        return
                    
                    # Проверяем, есть ли изображение в очереди
                    if not image_queues[camera_idx].empty():
                        # Берем копию последнего изображения, не удаляя его из очереди
                        image_data = image_queues[camera_idx].queue[-1]
                        
                        img = image_data['image']
                        timestamp = image_data['timestamp']
                        has_detections = image_data['has_detections']
                        
                        # Получаем размер окна для масштабирования
                        win_width = fullscreen_window.winfo_width() - 20
                        win_height = fullscreen_window.winfo_height() - 60
                        
                        # Изменяем размер изображения для полноэкранного отображения
                        display_img = resize_image_with_aspect_ratio(img, width=win_width, height=win_height)
                        
                        # Преобразуем в формат Tkinter
                        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(display_img)
                        tk_img = ImageTk.PhotoImage(image=pil_img)
                        
                        # Обновляем изображение
                        image_label.config(image=tk_img)
                        image_label.image = tk_img
                        
                        # Обновляем информацию
                        detection_status = "ОБНАРУЖЕНЫ ДЕФЕКТЫ!" if has_detections else "Нет дефектов"
                        detection_color = "red" if has_detections else "green"
                        
                        info_label.config(
                            text=f"Камера {camera_idx} | Время: {timestamp} | {detection_status}",
                            foreground=detection_color
                        )
                    
                    # Планируем следующее обновление
                    fullscreen_window.after(100, update_fullscreen)
                
                except Exception as e:
                    info_label.config(text=f"Ошибка: {str(e)}")
                    fullscreen_window.after(1000, update_fullscreen)
            
            # Запускаем обновление
            fullscreen_window.after(100, update_fullscreen)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть окно просмотра: {str(e)}")
    
    def on_exit(self):
        """Корректно завершает работу приложения."""
        if messagebox.askyesno("Выход", "Вы уверены, что хотите выйти?"):
            # Останавливаем все камеры
            global camera_active
            for idx in camera_active.keys():
                camera_active[idx] = False
            
            # Даем время потокам завершиться
            time.sleep(0.5)
            
            # Закрываем главное окно
            self.root.destroy()

def start_camera_threads(camera_urls, yolo_model):
    """Запускает потоки для всех камер."""
    threads = []
    
    for index, url in enumerate(camera_urls):
        # Создаем поток для камеры
        thread = threading.Thread(
            target=capture_from_camera,
            args=(url, index, yolo_model),
            daemon=True
        )
        threads.append(thread)
        thread.start()
    
    return threads

def main():
    # Загружаем настройки из файла конфигурации
    load_config()
    
    # Используем настройки из CONFIG
    save_folder = CONFIG["save_folder"]
    config_file = CONFIG["config_file"]
    model_path = CONFIG["model_path"]
    max_workers = CONFIG["max_workers"]

    # Создаем базовую папку для сохранения
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # Загружаем список URL камер
    camera_urls = load_camera_config(config_file)
    
    if not camera_urls:
        print(f"Не найдены URL камер в файле {config_file}")
        print("Создаем демонстрационную конфигурацию...")
        # Создаем демо-конфигурацию
        camera_urls = ["http://192.168.0.100:8080", "http://192.168.0.101:8080"]
        
        # Сохраняем демо-конфигурацию
        with open(config_file, 'w') as f:
            for url in camera_urls:
                f.write(f"{url}\n")
    
    print(f"Найдено {len(camera_urls)} камер в конфигурации.")
    
    # Загружаем модель YOLO
    print(f"Загрузка модели YOLO из {model_path}...")
    try:
        model = YOLO(model_path)  # Загружаем модель YOLO
        print("Модель YOLO успешно загружена.")
    except Exception as e:
        print(f"Ошибка при загрузке модели YOLO: {str(e)}")
        messagebox.showerror("Ошибка", f"Не удалось загрузить модель YOLO: {str(e)}")
        return
    
    # Инициализируем флаги активности камер
    for i in range(len(camera_urls)):
        camera_active[i] = True
    
    # Запускаем GUI
    root = tk.Tk()
    app = MultiCameraMonitorApp(root, camera_urls)
    
    # Запускаем потоки захвата
    camera_threads = start_camera_threads(camera_urls, model)
    # save_folder = CONFIG["save_folder"]
    print(f"Запущено {len(camera_threads)} потоков захвата изображений")
    print(f"Изображения сохраняются в {save_folder}")
    
    try:
        # Запускаем главный цикл Tkinter
        root.mainloop()
    except KeyboardInterrupt:
        print("\nПрограмма остановлена пользователем.")
    except Exception as e:
        print(f"Неожиданная ошибка: {str(e)}")
    
    # Останавливаем все потоки камер
    for idx in camera_active.keys():
        camera_active[idx] = False
    
    print("Завершение работы программы...")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Критическая ошибка при запуске программы: {str(e)}")
        # В случае ошибки отображаем сообщение в графическом интерфейсе
        try:
            import tkinter.messagebox as mb
            mb.showerror("Критическая ошибка", f"Ошибка при запуске программы:\n{str(e)}")
        except:
            pass  # Если даже tkinter не работает, просто выходим