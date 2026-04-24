# Лабораторная работа 1
## Основы обработки изображений
### Выполнил: студент группы 8Е21 Плотников Юрий Андреевич

Для 1 лаб работы по CV необходимо реализовать базовый минимум операций над изображениями
Входное изображение в формате (RGB, не чёрно-белое)
1. Фильтры
<br>1.1 Медианный фильтр
<br>1.2 Фильтр гаусса
2. Морфологические операции
<br>2.1 Эрозия
<br>2.2 Дилатация
3. Прочие операции
<br>3.1 пороговая бинаризация (для rgb и grayscale изображения)
<br>3.2 выравнивание гистограммы
<br>3.3 поворот изображений на угол кратный 90 градусов

В данной лабораторной работе изучаются базовые принципы обработки изображений. Несмотря на использование библиотек (numpy, cv2, matplotlib), по заданию нельзя применять готовые функции для обработки изображений — необходимо реализовать их самостоятельно, чтобы глубже понять теоретический материал.

Хоть использование библиотек и упрощает нам жизнь, но по заданию нельзя использовать готовые функции библиотек по обработке изображений.
Необходимо научиться их самостоятельному применению и прийти к полному пониманию теоритического материала.
Загрузим файлы изображений из папки прокета.

```python
# Чтение файла
image1=cv2.imread('Lab_1_pictures/Filters/DSC01644.JPG') # Кот на диване баланс
image2=cv2.imread('Lab_1_pictures/Filters/DSC01645.JPG') # Кот на диване тёплый
image3=cv2.imread('Lab_1_pictures/Filters/DSC01649_NOIZEE.jpg') # Морда кота зашумлённая
image4=cv2.imread('Lab_1_pictures/Filters/photo_2026-04-11_17-51-54.jpg') # Максим с гитарой
image5=cv2.imread('Lab_1_pictures/AntonAntonDipDipDeDon/VideoCapture_20240130-182722(1).jpg') # Антон в корридоре
image6=cv2.imread('Lab_1_pictures/Filters/20221205_003642.jpg') # Максим в корридоре пересвет
image7=cv2.imread('Lab_1_pictures/Filters/20240824_134749.jpg') # Котята
image8=cv2.imread('Lab_1_pictures/dmAaD.jpg') # Круг трансмутации
image9=cv2.imread('Lab_1_pictures/Filters/IMG_20220524_143433.jpg') # Я в недосвете
image10=cv2.imread('Lab_1_pictures/DSC01626.JPG') # Кошка (я хз зачем)

image = [image1, 
         image2, 
         image3,  
         image5, 
         image6, 
         image8, 
         image10]
```
<img width="1990" height="747" alt="image" src="https://github.com/user-attachments/assets/2a822ac5-b88d-4ecb-9f8a-2818567aff15" />

Как оказалось, по умолчанию, изображения выводятся, используя неверный порядок каналов. 
Предположительный порядок - BGR. Нам нужен RGB.
Исправим этот сюрприз с помощью создания функции для изменения порядка каналов цветового пространства.

```python
def bgr_to_rgb(img_bgr):
    """
    Преобразование из BGR в RGB.
    Просто меняем местами первый и третий каналы.
    """
    if len(img_bgr.shape) != 3:
        raise ValueError("Изображение должно быть цветным (3 канала)")
    
    h, w, c = img_bgr.shape
    
    # Создаем новое изображение
    img_rgb = np.zeros_like(img_bgr)
    
    # Меняем каналы местами: BGR -> RGB
    # Канал 0 (B) становится каналом 2 (R)
    # Канал 1 (G) остается каналом 1 (G)
    # Канал 2 (R) становится каналом 0 (B)
    for i in range(h):
        for j in range(w):
            img_rgb[i, j, 0] = img_bgr[i, j, 2]  # R = B
            img_rgb[i, j, 1] = img_bgr[i, j, 1]  # G = G
            img_rgb[i, j, 2] = img_bgr[i, j, 0]  # B = R
    return img_rgb
```
Проверим работоспособность функции
```python
image_RGB = image
for i in range(len(image)):
    image_RGB[i] = bgr_to_rgb(image[i])

plt.figure(figsize=(20, 20))
# Ряд 1: Изображения
for i in range(len(image_RGB)):
    plt.subplot(5, 5, i+1)
    plt.imshow(image_RGB[i])
plt.tight_layout()
plt.show()
```
<img width="1990" height="747" alt="image" src="https://github.com/user-attachments/assets/e8e83507-61f9-424e-a2c3-735a08b11006" />

Теперь, после изменения цветового формата, можно назвать выведенные изображения исходыми.
В основе многих видов фильтрации изображений также лежит перевод изображения в ч/б формат реализуем его функцию, потому что в дальнейшем это нам обязательно пригодится

```python
def rgb_to_gray(img_rgb):
    """
    Преобразование из RGB в Grayscale.
    Используем взвешенную сумму каналов.
    Формула: Gray = 0.299*R + 0.587*G + 0.114*B
    """
    if len(img_rgb.shape) != 3:
        raise ValueError("Изображение должно быть цветным (3 канала)")
    
    h, w = img_rgb.shape[:2]
    img_gray = np.zeros((h, w), dtype=np.uint8)
    
    # Коэффициенты для перевода в grayscale
    # Зеленый канал имеет наибольший вес, так как человеческий глаз 
    # наиболее чувствителен к зеленому цвету
    for i in range(h):
        for j in range(w):
            r = img_rgb[i, j, 0]
            g = img_rgb[i, j, 1]
            b = img_rgb[i, j, 2]
            
            # Применяем формулу
            gray_value = 0.299 * r + 0.587 * g + 0.114 * b
            img_gray[i, j] = int(gray_value)
    
    return img_gray
```

Проверим работоспособность функции

<img width="1990" height="775" alt="image" src="https://github.com/user-attachments/assets/8a186641-c1aa-46d2-aa22-b2691c24bab5" />

# 1 Фильтры
## 1.1 Фильтр линейной коррекции
### 1.1.1 Создание функции
```python
def linear_brightness_filter(img):
    """
    Линейная коррекция яркости (контрастная растяжка).
    Находит минимальное и максимальное значение яркости,
    затем пересчитывает все пиксели по формуле:
    I_new = (I_old - I_min) * 255 / (I_max - I_min)
    
    Args:
        img: входное изображение (RGB или grayscale)
    Returns:
        Отфильтрованное изображение
    """
    if len(img.shape) == 3:  # Цветное RGB изображение
        result = np.zeros_like(img)
        
        # Применяем к каждому каналу отдельно
        for channel in range(3):
            channel_data = img[:, :, channel].astype(np.float32)
            
            # Находим мин и макс значения
            I_min = np.min(channel_data)
            I_max = np.max(channel_data)
            
            # Применяем линейное преобразование
            # Формула: I_new = (I_old - I_min) * 255 / (I_max - I_min)
            I_range = I_max - I_min
            
            if I_range > 0:  # Избегаем деления на ноль
                channel_corrected = (channel_data - I_min) * 255.0 / I_range
            else:
                channel_corrected = channel_data
            
            # Ограничиваем значения диапазоном [0, 255] и преобразуем в uint8
            result[:, :, channel] = np.clip(channel_corrected, 0, 255).astype(np.uint8)
        
        return result
    else:  # Grayscale изображение
        img_float = img.astype(np.float32)
        
        # Находим мин и макс значения
        I_min = np.min(img_float)
        I_max = np.max(img_float)
        
        # Применяем линейное преобразование
        I_range = I_max - I_min
        
        if I_range > 0:
            result = (img_float - I_min) * 255.0 / I_range
        else:
            result = img_float
        
        return np.clip(result, 0, 255).astype(np.uint8)
```
### 1.1.2 Применение и визуализация

<img width="1985" height="994" alt="image" src="https://github.com/user-attachments/assets/07250b7c-d819-4836-9b09-e18a7ee414fb" />


СТАТИСТИКА ЛИНЕЙНОГО ФИЛЬТРА

Original RGB:

  R - Min:  54, Max: 255, Mean: 139.72
  
  G - Min:  47, Max: 255, Mean: 132.53
  
  B - Min:  69, Max: 255, Mean: 143.94

After Linear Filter:

  R - Min:   0, Max: 255, Mean: 108.25
  
  G - Min:   0, Max: 255, Mean: 104.36
  
  B - Min:   0, Max: 255, Mean: 102.26

Grayscale Original - Min:  54, Max: 255

Grayscale Linear   - Min:   0, Max: 255

## 1.2 Фильтры нелинейной коррекции
### 1.2.1 Создание функции
#### 1.2.1.1 Фильтр логарифмической коррекции

```python
def manual_log_correction(img):
    """
    Применяет логарифмическое преобразование к изображению.
    Формула: s = c * ln(1 + r)
    Это расширяет диапазон темных значений и сжимает светлые.
    """
    # Преобразуем в float, чтобы избежать переполнения и работать с дробными числами
    img_float = img.astype(np.float64)
    
    # Константа c для масштабирования результата обратно в [0, 255]
    # Максимальное значение ln(1 + 255) = ln(256)
    c = 255.0 / np.log(1 + 255.0)
    
    # Применяем формулу
    # Обратите внимание: мы используем (1 + img), так как ln(0) не определен
    log_img = c * np.log(1 + img_float)
    
    # Обрезаем значения (на случай ошибок округления) и возвращаем в uint8
    return np.clip(log_img, 0, 255).astype(np.uint8)
```

#### 1.2.1.2 Фильтр гамма-корреции
Чтобы правильно воспроизвести функцию гамма-коррекции, нам потребуется реализовать переход в другое цветовое пространство (из RGB в YUV) и обратно.
Для начала, напишем функцию для смены цветового пространства.

```python
# ==========================================
# RGB В YUV
# ==========================================
def rgb_to_yuv(img_rgb):
    """
    Преобразование из RGB в YUV.
    Y - яркость (luminance)
    U и V - цветность (chrominance)
    
    Формулы:
    Y = 0.299*R + 0.587*G + 0.114*B
    U = -0.147*R - 0.289*G + 0.436*B
    V = 0.615*R - 0.515*G - 0.100*B
    """
    if len(img_rgb.shape) != 3:
        raise ValueError("Изображение должно быть цветным (3 канала)")
    
    h, w = img_rgb.shape[:2]
    img_yuv = np.zeros_like(img_rgb, dtype=np.float32)
    
    # Нормализуем RGB значения в диапазон [0, 1]
    r_norm = img_rgb[:, :, 0].astype(np.float32) / 255.0
    g_norm = img_rgb[:, :, 1].astype(np.float32) / 255.0
    b_norm = img_rgb[:, :, 2].astype(np.float32) / 255.0
    
    for i in range(h):
        for j in range(w):
            r = r_norm[i, j]
            g = g_norm[i, j]
            b = b_norm[i, j]
            
            # Применяем формулы преобразования
            y = 0.299 * r + 0.587 * g + 0.114 * b
            u = -0.147 * r - 0.289 * g + 0.436 * b
            v = 0.615 * r - 0.515 * g - 0.100 * b
            
            # Y остается в диапазоне [0, 1], U и V в диапазоне [-0.5, 0.5]
            # Масштабируем для хранения в uint8
            img_yuv[i, j, 0] = y * 255.0           # Y: 0-255
            img_yuv[i, j, 1] = (u + 0.5) * 255.0   # U: 0-255 (было -0.5 до 0.5)
            img_yuv[i, j, 2] = (v + 0.5) * 255.0   # V: 0-255 (было -0.5 до 0.5)
    
    return img_yuv.astype(np.uint8)
# ==========================================
# RGB В YUV
# ==========================================
def rgb_to_yuv(img_rgb):
    """
    Преобразование из RGB в YUV.
    Y - яркость (luminance)
    U и V - цветность (chrominance)
    
    Формулы:
    Y = 0.299*R + 0.587*G + 0.114*B
    U = -0.147*R - 0.289*G + 0.436*B
    V = 0.615*R - 0.515*G - 0.100*B
    """
    if len(img_rgb.shape) != 3:
        raise ValueError("Изображение должно быть цветным (3 канала)")
    
    h, w = img_rgb.shape[:2]
    img_yuv = np.zeros_like(img_rgb, dtype=np.float32)
    
    # Нормализуем RGB значения в диапазон [0, 1]
    r_norm = img_rgb[:, :, 0].astype(np.float32) / 255.0
    g_norm = img_rgb[:, :, 1].astype(np.float32) / 255.0
    b_norm = img_rgb[:, :, 2].astype(np.float32) / 255.0
    
    for i in range(h):
        for j in range(w):
            r = r_norm[i, j]
            g = g_norm[i, j]
            b = b_norm[i, j]
            
            # Применяем формулы преобразования
            y = 0.299 * r + 0.587 * g + 0.114 * b
            u = -0.147 * r - 0.289 * g + 0.436 * b
            v = 0.615 * r - 0.515 * g - 0.100 * b
            
            # Y остается в диапазоне [0, 1], U и V в диапазоне [-0.5, 0.5]
            # Масштабируем для хранения в uint8
            img_yuv[i, j, 0] = y * 255.0           # Y: 0-255
            img_yuv[i, j, 1] = (u + 0.5) * 255.0   # U: 0-255 (было -0.5 до 0.5)
            img_yuv[i, j, 2] = (v + 0.5) * 255.0   # V: 0-255 (было -0.5 до 0.5)
    
    return img_yuv.astype(np.uint8)
```
Теперь напишем функцию для возвращения в RGB пространство.

```python
# ==========================================
# YUV В RGB (обратное преобразование)
# ==========================================
def yuv_to_rgb(img_yuv):
    """
    Обратное преобразование из YUV в RGB.
    """
    if len(img_yuv.shape) != 3:
        raise ValueError("Изображение должно быть в формате YUV (3 канала)")
    
    h, w = img_yuv.shape[:2]
    img_rgb = np.zeros_like(img_yuv, dtype=np.float32)
    
    # Нормализуем значения
    y_norm = img_yuv[:, :, 0].astype(np.float32) / 255.0
    u_norm = (img_yuv[:, :, 1].astype(np.float32) / 255.0) - 0.5
    v_norm = (img_yuv[:, :, 2].astype(np.float32) / 255.0) - 0.5
    
    for i in range(h):
        for j in range(w):
            y = y_norm[i, j]
            u = u_norm[i, j]
            v = v_norm[i, j]
            
            # Обратные формулы
            r = y + 1.140 * v
            g = y - 0.395 * u - 0.581 * v
            b = y + 2.032 * u
            
            # Ограничиваем диапазон [0, 1]
            r = max(0, min(1, r))
            g = max(0, min(1, g))
            b = max(0, min(1, b))
            
            img_rgb[i, j, 0] = r * 255.0
            img_rgb[i, j, 1] = g * 255.0
            img_rgb[i, j, 2] = b * 255.0
    
    return img_rgb.astype(np.uint8)
```

```python
def manual_gamma_correction(img, gamma):
    """
    Применяет гамма-коррекцию к каналу яркости (Y) в пространстве YUV.
    gamma < 1: осветляет изображение (расширяет темные тона)
    gamma > 1: затемняет изображение (сжимает светлые тона)
    """
    # 1. Перевод в YUV 
    img_yuv = rgb_to_yuv(img)
    
    # 2. Извлекаем канал яркости Y
    y_channel = img_yuv[:, :, 0].astype(np.float64)
    
    # 3. Нормализуем значения в диапазон [0, 1]
    y_normalized = y_channel / 255.0
    
    # 4. Применяем степенную функцию (гамма-коррекцию)
    # Формула из методички: y_new = (y_old / 255) ** (1/gamma) * 255
    y_corrected = np.power(y_normalized, 1.0 / gamma)
    
    # 5. Возвращаем в диапазон [0, 255]
    y_corrected = (y_corrected * 255.0).astype(np.uint8)
    
    # 6. Записываем исправленный канал обратно
    img_yuv[:, :, 0] = y_corrected
    
    # 7. Перевод обратно в RGB
    img_corrected = yuv_to_rgb(img_yuv)
    
    return img_corrected
```

### 1.2.2 Применение и визуализация

```python
k = 7 # номер изображения из списка
img = image_RGB[k-1]
# 1. Логарифмическая коррекция
img_log = manual_log_correction(img)

# 2. Гамма-коррекция (два варианта: осветление и затемнение)
# gamma = 0.5 -> Осветление (полезно для вечерних фото)
img_gamma_bright = manual_gamma_correction(img, gamma=0.5)

# gamma = 1.5 -> Затемнение (полезно для пересвеченных фото)
img_gamma_dark = manual_gamma_correction(img, gamma=1.5)
```
<img width="1189" height="980" alt="image" src="https://github.com/user-attachments/assets/976cd616-366d-4ffa-afef-dca9f1e52422" />

## 1.3 Фильтр баланса белого

<img width="1227" height="462" alt="image" src="https://github.com/user-attachments/assets/e55a795e-360e-47f2-99ec-7eeb71adcc27" />

По счастливому стечению обстоятельств были найдены два изображения, как раз различающиеся балансом белого. Попробуем узнать, на каком из этих изображений баланс белого нарушен. Для этого пропустим через фильтр одно из них.
Возьмём для проверки второе изображение, потому что оттенок шерсти кота на нём выглядит более тёплым, чем на первом.
Для создания фильтра баланса белого, усредним значения всех каналов, добившись эффекта "Серого мира".
Создадим функцию, в которой будет реализовано выполнение следующей математической формулы:

![image.png](attachment:image.png)

где R_avg, G_avg, B_avg - это средние значения каждого из цветовых каналов.

### 1.3.1 Создание функции

```python
def manual_gray_world(img):
    """
    Реализация алгоритма балансировки белого "Серый мир" вручную.
    Args:
        img: входное изображение в формате RGB (numpy array)
    Returns:
        img_balanced: изображение с исправленным балансом белого
    """
    # Разделяем изображение на каналы R, G, B
    r = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    b = img[:, :, 2].astype(np.float32)
    
    # 1. Находим средние значения интенсивности каждого канала
    r_avg = np.mean(r)
    g_avg = np.mean(g)
    b_avg = np.mean(b)
    
    # 2. Находим общее среднее значение интенсивности всех трех каналов
    k = (r_avg + g_avg + b_avg) / 3
    
    # 3. Определяем коэффициенты коррекции
    # Избегаем деления на ноль, добавляя маленькое число epsilon, если среднее равно 0
    epsilon = 1e-9
    kr = k / (r_avg + epsilon)
    kg = k / (g_avg + epsilon)
    kb = k / (b_avg + epsilon)
    
    # 4. Корректируем интенсивность пикселей
    r_corrected = r * kr
    g_corrected = g * kg
    b_corrected = b * kb
    
    # Обрезаем значения, вышедшие за диапазон [0, 255], и приводим к uint8
    r_corrected = np.clip(r_corrected, 0, 255).astype(np.uint8)
    g_corrected = np.clip(g_corrected, 0, 255).astype(np.uint8)
    b_corrected = np.clip(b_corrected, 0, 255).astype(np.uint8)
    
    # Собираем каналы обратно в изображение
    img_balanced = np.dstack([r_corrected, g_corrected, b_corrected])
    
    return img_balanced
```
### 1.3.2 Применение и визуализация

<img width="1990" height="1486" alt="image" src="https://github.com/user-attachments/assets/5f02f4d1-c8dd-4f0f-a4ed-575ced0aacc2" />

Мы сделали удачный выбор! Кот действительно был слишком тёплым. Теперь сравним отфильтрованное второе изображение с первым изображением кота

<img width="1565" height="567" alt="image" src="https://github.com/user-attachments/assets/8788dd56-e0e6-4501-bcbf-bf5e8e3d73e8" />

Сравним численные прараметры цветовых каналов изображений

Статистика каналов:

Original_1 - R_avg: 121.77, G_avg: 103.69, B_avg: 91.53

Original_2 - R_avg: 129.60, G_avg: 125.06, B_avg: 122.46

Balanced_2 - R_avg: 105.17, G_avg: 105.19, B_avg: 103.38

Фотографии почти идентичны как визуально, так и по средним значениям цветовых каналов, что означает, что первый вариант изначально был выровнен через данный фильтр ещё на устройсве, с которого была сделана фотография (или идеально совпали все обстоятельства освещения)

## 1.4 Фильтр Гаусса
### 1.4.1 Создание функции

```python
def create_gaussian_kernel(size, sigma=1.0):
    """Создает ядро Гаусса заданного размера"""
    kernel = np.zeros((size, size))
    center = size // 2

    # Формула Гаусса: G(x,y) = (1/(2*pi*sigma^2)) * exp(-(x^2+y^2)/(2*sigma^2))
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))

    # Нормируем ядро, чтобы сумма всех элементов была равна 1
    # Это нужно, чтобы яркость изображения не изменилась глобально
    kernel /= np.sum(kernel)
    return kernel

def convolution_2d(image, kernel):
    """Базовая операция 2D свертки"""
    h, w = image.shape
    k_h, k_w = kernel.shape
    pad_h = k_h // 2
    pad_w = k_w // 2

    result = np.zeros_like(image, dtype=np.float64)
    # Паддинг для обработки краев
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

    for i in range(h):
        for j in range(w):
            # Извлекаем область изображения под ядром
            region = padded_image[i : i + k_h, j : j + k_w]
            # Поэлементное умножение и суммирование (свертка)
            result[i, j] = np.sum(region * kernel)
    return result       

def manual_gaussian_filter(img, kernel_size=5, sigma=1.0):
    """Применяет фильтр Гаусса через свертку"""
    kernel = create_gaussian_kernel(kernel_size, sigma)

    if len(img.shape) == 3:
        channels = []
        for i in range(3):
            channel = img[:, :, i]
            filtered_channel = convolution_2d(channel, kernel)
            channels.append(filtered_channel)
        return np.dstack(channels).astype(np.uint8)
    else:
        return convolution_2d(img, kernel).astype(np.uint8)
```
```python
# Применяем фильтр
k = 3 # номер изображения из списка
img = image_RGB[k-1]

# Зададим размер ядра свёртки
gaussian_kernel_size = 7

# Зададим значения sigma
gaussian_sigma = [1, 1.5, 3]

img_gaussian = []
# Применим функцию фильтра Гаусса
for i in range(len(img_gaussian)):
    img_gaussian[i] = manual_gaussian_filter(img, gaussian_kernel_size, gaussian_sigma[i])
```

<img width="1338" height="989" alt="9" src="https://github.com/user-attachments/assets/118f67a6-9c99-4cb4-8ca8-c541dcfa1da1" />

Ядро Гаусса 7x7:

[[0.0013 0.0041 0.0079 0.0099 0.0079 0.0041 0.0013]

 [0.0041 0.0124 0.0241 0.0301 0.0241 0.0124 0.0041]
 
 [0.0079 0.0241 0.047  0.0587 0.047  0.0241 0.0079]
 
 [0.0099 0.0301 0.0587 0.0733 0.0587 0.0301 0.0099]
 
 [0.0079 0.0241 0.047  0.0587 0.047  0.0241 0.0079]
 
 [0.0041 0.0124 0.0241 0.0301 0.0241 0.0124 0.0041]
 
 [0.0013 0.0041 0.0079 0.0099 0.0079 0.0041 0.0013]]

Как можно увидеть, фильтр Гаусса не справился с частью шума, потому что на изображении присутствует  множество разных видов шума.

### 1.4.3 Визуализация тепловой карты

<img width="675" height="636" alt="image" src="https://github.com/user-attachments/assets/6fc9d96c-2b3a-4a18-a0e8-0369cf22551e" />

<img width="675" height="636" alt="image" src="https://github.com/user-attachments/assets/0de7a8c5-3473-4e7f-8849-299734feda5a" />

<img width="675" height="636" alt="image" src="https://github.com/user-attachments/assets/98d5d7bb-6d5f-4697-afa4-cede651e37d1" />

## 1.5 Медианный фильтр


```python
# Преобразуем загруженное изображение в float для корректной работы фильтров, если нужно,
# но наши функции принимают uint8. Убедимся, что image_RGB имеет тип uint8.
for i in range(len(image_RGB)):
    if image_RGB[i].dtype != np.uint8:
        image_RGB[i] = (image_RGB[i] * 255).astype(np.uint8)
        print(f"Преобразован тип изображения {i}")
    else:
        print("Всё гуд!")
```

### 1.5.1 Создание функции

```python
def manual_median_filter(img, kernel_size=3):
    """
    Применяет медианный фильтр к изображению.
    Если изображение цветное (RGB), применяет фильтр к каждому каналу отдельно.
    """
    # Если изображение цветное, обрабатываем каждый канал (R, G, B) отдельно
    if len(img.shape) == 3:
        channels = []
        for i in range(3):
            channel = img[:, :, i]
            filtered_channel = manual_median_filter_grayscale(channel, kernel_size)
            channels.append(filtered_channel)
        return np.dstack(channels) # Собираем каналы обратно
    else:
        return manual_median_filter_grayscale(img, kernel_size)

def manual_median_filter_grayscale(gray_img, kernel_size=3):
    """Медианный фильтр для одноканального (серого) изображения"""
    h, w = gray_img.shape
    result = np.zeros_like(gray_img)
    pad = kernel_size // 2

    # Добавляем "рамку" из нулей или копируем края, чтобы не выйти за границы
    # Здесь используем копирование краев (edge padding)
    padded_img = np.pad(gray_img, pad, mode='edge')

    for i in range(h):
        for j in range(w):
            # Вырезаем окно размером kernel_size x kernel_size вокруг пикселя (i, j)
            window = padded_img[i : i + kernel_size, j : j + kernel_size]

            # Находим медиану всех значений в окне
            # Медиана - это значение, которое находится в середине отсортированного списка
            median_val = np.median(window)

            result[i, j] = median_val

    return result.astype(np.uint8)
```

### 1.5.2 Применение и визуализация

```python
k = 3 #  1 <= k <= 10 (включительно)
img = image_RGB[k-1]
# Применяем фильтр
median_kernel_size = [3, 7, 11]
img_median = [0, 0, 0]
for i in range(len(median_kernel_size)):
    img_median[i] = manual_median_filter(img, median_kernel_size[i])
```

<img width="1338" height="989" alt="13" src="https://github.com/user-attachments/assets/e15ce2e9-4e65-4f69-bdad-8c3f77c6df52" />

# 2.1 ПОРОГОВАЯ БИНАРИЗАЦИЯ
### 2.1.1 Создание функции

```python
def manual_threshold(img, threshold=127):
    """
    Простая пороговая бинаризация.
    Если яркость пикселя > threshold -> 255 (белый), иначе -> 0 (черный).
    """
    gray = rgb_to_gray(img)
    h, w = gray.shape
    binary = np.zeros_like(gray)

    for i in range(h):
        for j in range(w):
            if gray[i, j] > threshold:
                binary[i, j] = 255
            else:
                binary[i, j] = 0

    return binary
```

### 2.1.2 Примениение и визуализация 

```python
# Выберем изображение
k = 1
img = image_RGB[k-1]

# зададим несколько порогов
threshold_val = [60, 130, 200]

# Бинаризуем исходное изображение
img_binary = [0, 0, 0]
for i in range(len(threshold_val)):
    img_binary[i] = manual_threshold(img, threshold_val[i])
```
<img width="641" height="508" alt="image" src="https://github.com/user-attachments/assets/73f0e92d-8e12-4f01-bca5-d596b488ecab" />

# 3 МОРФОЛОГИЧЕСКИЕ ОПЕРАЦИИ
## 3.1 ЭРОЗИЯ

```python
def manual_erosion(binary_img, kernel_size=3):
    """
    Эрозия: Пиксель остается белым (255) ТОЛЬКО ЕСЛИ все пиксели в окне ядра тоже белые.
    Иначе становится черным (0). Это "съедает" границы объектов.
    """
    h, w = binary_img.shape
    result = np.zeros_like(binary_img)
    pad = kernel_size // 2

    # Нормализуем входные данные к 0 и 1 для удобства логики
    # Предполагаем, что binary_img содержит только 0 и 255
    norm_img = (binary_img // 255).astype(np.uint8)
    padded_img = np.pad(norm_img, pad, mode='constant', constant_values=0)

    for i in range(h):
        for j in range(w):
            window = padded_img[i : i + kernel_size, j : j + kernel_size]
            # Логическое И (минимум): если есть хоть один 0, результат 0
            if np.all(window == 1):
                result[i, j] = 255
            else:
                result[i, j] = 0

    return result
```

## 3.2 ДИЛАТАЦИЯ

```python
def manual_dilation(binary_img, kernel_size=3):
    """
    Дилатация: Пиксель становится белым (255), ЕСЛИ ХОТЯ БЫ ОДИН пиксель в окне ядра белый.
    Иначе остается черным (0). Это "расширяет" объекты.
    """
    h, w = binary_img.shape
    result = np.zeros_like(binary_img)
    pad = kernel_size // 2

    norm_img = (binary_img // 255).astype(np.uint8)
    padded_img = np.pad(norm_img, pad, mode='constant', constant_values=0)

    for i in range(h):
        for j in range(w):
            window = padded_img[i : i + kernel_size, j : j + kernel_size]
            # Логическое ИЛИ (максимум): если есть хоть один 1, результат 1
            if np.any(window == 1):
                result[i, j] = 255
            else:
                result[i, j] = 0

    return result
```

```python
# ==========================================
# ПРИМЕНЕНИЕ И ВИЗУАЛИЗАЦИЯ
# ==========================================
img = img_binary[1]
# Применяем морфологию
erosion_kernel_size = 21 # Размер ядра
img_eroded = manual_erosion(img, erosion_kernel_size)
dilation_kernel_size = 9
img_dilated = manual_dilation(img, dilation_kernel_size)
```

<img width="1489" height="402" alt="image" src="https://github.com/user-attachments/assets/70464b7e-69a5-407d-a524-bc5e62bd34ee" />

```python

```

## 3.2 ВЫРАВНИВАНИЕ ГИСТОГРАММЫ 
Гистограмма – это график распределения яркостей на изображении. 
На горизонтальной оси - шкала яркостей тонов от белого до черного, 
а на вертикальной оси - число пикселей заданной яркости.

```python
def manual_histogram_equalization(img):
    """
    Выравнивание гистограммы для улучшения контраста.
    Работает с grayscale изображением.
    Если подано RGB, применяет эквализацию к каждому каналу отдельно
    (или можно перевести в YUV и выровнять только канал яркости Y, но для простоты сделаем поканально).
    """
    if len(img.shape) == 3:
        # Для RGB лучше работать в пространстве YUV или LAB, но для учебной цели
        # сделаем поканальную эквализацию в RGB (может исказить цвета, но покажет принцип)
        channels = []
        for i in range(3):
            channel = img[:, :, i]
            eq_channel = equalize_grayscale(channel)
            channels.append(eq_channel)
        return np.dstack(channels)
    else:
        return equalize_grayscale(img)

def equalize_grayscale(gray_img):
    """Алгоритм выравнивания гистограммы для одного канала"""
    h, w = gray_img.shape
    total_pixels = h * w

    # 1. Подсчет гистограммы (частота встречаемости каждого уровня яркости 0-255)
    hist = np.zeros(256, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            pixel_val = gray_img[i, j]
            hist[pixel_val] += 1

    # 2. Вычисление кумулятивной функции распределения (CDF)
    cdf = np.cumsum(hist)

    # 3. Нормализация CDF до диапазона [0, 255]
    # Формула: CDF_norm[i] = round((CDF[i] - CDF_min) / (total_pixels - CDF_min)) * 255
    # Упрощенная версия (если CDF_min != 0):
    cdf_min = cdf[np.nonzero(cdf)][0] # Первое ненулевое значение CDF

    cdf_normalized = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        if cdf[i] >= cdf_min:
            # Линейное растягивание накопленной вероятности
            cdf_normalized[i] = int(round((cdf[i] - cdf_min) / (total_pixels - cdf_min) * 255))

    # 4. Применение таблицы преобразования (Look-Up Table) к изображению
    result = np.zeros_like(gray_img)
    for i in range(h):
        for j in range(w):
            old_val = gray_img[i, j]
            result[i, j] = cdf_normalized[old_val]

    return result
```

```python
# ==========================================
# ПРИМЕНЕНИЕ И ВИЗУАЛИЗАЦИЯ
# ==========================================
k = 7
img = image_RGB[k-1]

# Выравнивание гистограммы
median_kernel_size_9=3
img_median_9 = manual_median_filter(img, median_kernel_size_9)
img_eq = manual_histogram_equalization(img_median_9)
```
<img width="1182" height="438" alt="image" src="https://github.com/user-attachments/assets/b08f9065-b288-40c8-8772-f893153aec1b" />

<img width="1490" height="490" alt="image" src="https://github.com/user-attachments/assets/15a5057d-c0ec-439a-bb0f-8682bcc547d3" />

## 3.3 ПОВОРОТ ИЗОБРАЖЕНИЯ

```python
def manual_rotate_90(img):
    """Поворот на 90 градусов по часовой стрелке"""
    h, w = img.shape[:2]
    if len(img.shape) == 2:
        new_img = np.zeros((w, h), dtype=img.dtype)
        for i in range(h):
            for j in range(w):
                # Новая координата x = старая y
                # Новая координата y = (h - 1) - старая x
                new_img[j, h - 1 - i] = img[i, j]
    else:
        new_img = np.zeros((w, h, img.shape[2]), dtype=img.dtype)
        for i in range(h):
            for j in range(w):
                new_img[j, h - 1 - i, :] = img[i, j, :]
    return new_img

def manual_rotate_180(img):
    """Поворот на 180 градусов"""
    h, w = img.shape[:2]
    if len(img.shape) == 2:
        new_img = np.zeros_like(img)
        for i in range(h):
            for j in range(w):
                new_img[h - 1 - i, w - 1 - j] = img[i, j]
    else:
        new_img = np.zeros_like(img)
        for i in range(h):
            for j in range(w):
                new_img[h - 1 - i, w - 1 - j, :] = img[i, j, :]
    return new_img

def manual_rotate_270(img):
    """Поворот на 270 градусов по часовой стрелке (или 90 против)"""
    h, w = img.shape[:2]
    if len(img.shape) == 2:
        new_img = np.zeros((w, h), dtype=img.dtype)
        for i in range(h):
            for j in range(w):
                # Новая координата x = (w - 1) - старая y
                # Новая координата y = старая x
                new_img[w - 1 - j, i] = img[i, j]
    else:
        new_img = np.zeros((w, h, img.shape[2]), dtype=img.dtype)
        for i in range(h):
            for j in range(w):
                new_img[w - 1 - j, i, :] = img[i, j, :]
    return new_img
```

```python
# 2. Повороты
k = 4
img = image_RGB[k-1]

img_rot90 = manual_rotate_90(img)
img_rot180 = manual_rotate_180(img)
img_rot270 = manual_rotate_270(img)
```

<img width="1289" height="1489" alt="image" src="https://github.com/user-attachments/assets/ae0b3b44-a70d-4b5c-a21e-079937d5da53" />

### Вывод
В резульате выполнения лабораторной работы были на практике закреплены полученные знания по базовым методам работы с изображениями, среди которых:
•	Коррекция освещения.
•	Фильтрация.
•	Бинаризация.
•	Морфологические операции.



