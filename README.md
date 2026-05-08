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

В данной лабораторной работе изучаются базовые принципы обработки изображений. Несмотря на использование библиотек (numpy, cv2, matplotlib), по заданию нельзя использовать готовые функции библиотек по обработке изображений. Необходимо научиться их самостоятельному применению и прийти к более глубокому пониманию теоритического материала.
Загрузим файлы изображений из папки прокета.

```python
# Чтение файла
image1=cv2.imread('Lab_1_pictures/Filters/DSC01644.JPG') # Кот на диване баланс
image2=cv2.imread('Lab_1_pictures/Filters/DSC01645.JPG') # Кот на диване тёплый
image3=cv2.imread('Lab_1_pictures/Filters/DSC01649_NOIZEE.jpg') # Морда кота зашумлённая
image5=cv2.imread('Lab_1_pictures/AntonAntonDipDipDeDon/VideoCapture_20240130-182722(1).jpg') # Антон в корридоре
image6=cv2.imread('Lab_1_pictures/Filters/20221205_003642.jpg') # Максим в корридоре пересвет
image10=cv2.imread('Lab_1_pictures/DSC01626.JPG') # Кошка (я хз зачем)

image = [image1, 
         image2, 
         image3,  
         image5, 
         image6,  
         image10]
```
<img width="1990" height="993" alt="image" src="https://github.com/user-attachments/assets/a6313aa3-9000-428e-a4a9-1adfcf1049c8" />


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

<img width="1990" height="993" alt="image" src="https://github.com/user-attachments/assets/bda25a77-8ca4-4872-8b99-7f6d55152808" />

Теперь, после изменения цветового формата, можно назвать выведенные изображения исходыми.
В основе многих видов фильтрации изображений также лежит перевод изображения в ч/б формат реализуем его функцию, потому что в дальнейшем это нам пригодится

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
Для начала, напишем функции для смены цветового пространства.

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

Составим функцию гамма-коррекции

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

Специально для наглядности работы этого фильтра были найдены два изображения, как раз различающиеся балансом белого. Попробуем узнать, на каком из этих изображений баланс белого нарушен. Для этого пропустим через фильтр одно из них.
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

# Лабораторная работа 2
## Визуальная одометрия (навигация)
### Выполнил: студент группы 8Е21 Плотников Юрий Андреевич

Цель: Разработать систему визуальной одометрии (навигации) по группе фотографий.
Ход работы: сделайте не менее 8 фото с переносом камеры или ноутбука по квадрату (то есть двиньте сначала вправо, потом вперед, потом влево, потом назад и обратно в начальную точку). Используя данные фотографии реализуйте следующее:
<p> 1.	Определите на каждой фотографии ключевые точки </p>
<p>2.	Отфильтруйте самые наилучшие применяю адаптивный радиус и локальные максимумы, не забудьте так же выровнять по яркости изображения.</p>
<p>3.	Постройте по каждой точке дескриптор (можете использовать любой, рекомендуется SIFT)</p>
<p>4.	Сопоставьте два соседних изображения на предмет соответствия ключевых точек. То есть определите пары одинаковых точек.</p>
<p>5.	Постройте модель преобразования изображений, учитывайте только поворот и сдвиг.</p>
<p>6.	С учетом полученных моделей постройте траекторию движения камеры.</p>
<p>Проверка работоспособности: будет осуществляться на специальной группе фото, предоставленных преподавателем. Траектория движения, для которых недоступна.</p>
<p>В процессе выполнения вы можете использовать готовые функции по погрузке данных, перевода в цветовые пространства, фильтрации, для построения прямых и траекторий. Функции 1-6 описанные выше должны быть реализованы самостоятельно.</p>

Подключим библиотеки, которые будут использоваться в дальнейшем
```python
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import random
#from PIL import Image
from IPython.display import Image
%matplotlib inline
import math
import lab1_functions as lb1
from lab1_functions import intensity_grayscale
import lab2_functions as lb2
import os
print(os.getcwd())
print(os.listdir())
```
d:\Libraries\Documents\VScode\CV_labs_Plotnikov_8E21\Lab_2
['gradient.png', 'harris1.png', 'lab1_functions.py', 'lab2.ipynb', 'lab2.py', 'lab2_functions.py', 'Pictures4', '__pycache__', 'Лишнее_2']

# 2.1 Загрузить изображения
Импортируем изображения с движением камеры и выведем их для проверки:

```python
image1=cv2.cvtColor(cv2.imread('D:/Libraries/Documents/VScode/CV_labs_Plotnikov_8E21/Lab_2/Pictures4/1.JPG'), cv2.COLOR_BGR2RGB)
image2=cv2.cvtColor(cv2.imread('D:/Libraries/Documents/VScode/CV_labs_Plotnikov_8E21/Lab_2/Pictures4/2.JPG'), cv2.COLOR_BGR2RGB)
image3=cv2.cvtColor(cv2.imread('D:/Libraries/Documents/VScode/CV_labs_Plotnikov_8E21/Lab_2/Pictures4/3.JPG'), cv2.COLOR_BGR2RGB)
image4=cv2.cvtColor(cv2.imread('D:/Libraries/Documents/VScode/CV_labs_Plotnikov_8E21/Lab_2/Pictures4/4.JPG'), cv2.COLOR_BGR2RGB)
image5=cv2.cvtColor(cv2.imread('D:/Libraries/Documents/VScode/CV_labs_Plotnikov_8E21/Lab_2/Pictures4/5.JPG'), cv2.COLOR_BGR2RGB)
image6=cv2.cvtColor(cv2.imread('D:/Libraries/Documents/VScode/CV_labs_Plotnikov_8E21/Lab_2/Pictures4/6.JPG'), cv2.COLOR_BGR2RGB)
image7=cv2.cvtColor(cv2.imread('D:/Libraries/Documents/VScode/CV_labs_Plotnikov_8E21/Lab_2/Pictures4/7.JPG'), cv2.COLOR_BGR2RGB)
image8=cv2.cvtColor(cv2.imread('D:/Libraries/Documents/VScode/CV_labs_Plotnikov_8E21/Lab_2/Pictures4/8.JPG'), cv2.COLOR_BGR2RGB)

images_sequence = [image1, image2, image3, image4, image5, image6, image7, image8]
```
<img width="1010" height="452" alt="image" src="https://github.com/user-attachments/assets/450b4c0c-5619-4c0c-ab48-0ad3b8a50694" />
Уменьшим разрешение изображений для более быстрой их обработки.
```python
def resize_images(images_sequence, scale_factor=2):
    """
    Уменьшает разрешение всех изображений в последовательности
    
    Параметры:
    - images_sequence: список изображений (в формате numpy array)
    - scale_factor: коэффициент уменьшения (во сколько раз уменьшить)
                    Например: scale_factor=2 уменьшит разрешение в 2 раза
    
    Возвращает:
    - список сжатых изображений
    """
    resized_images = []
    
    for img in images_sequence:
        if img is not None:
            # Получаем новые размеры
            height, width = img.shape[:2]
            new_width = width // scale_factor
            new_height = height // scale_factor
            
            # Изменяем размер изображения
            resized_img = cv2.resize(img, (new_width, new_height), 
                                    interpolation=cv2.INTER_AREA)  # INTER_AREA лучше для уменьшения
            resized_images.append(resized_img)
        else:
            print("Предупреждение: обнаружено пустое изображение")
            resized_images.append(None)
    
    return resized_images
```

```python
# Пример использования для ваших изображений:
# Уменьшаем в 2 раза (4032x3024 -> 2016x1512)
images_compressed = resize_images(images_sequence, scale_factor=2)

# Уменьшаем в 4 раза (4032x3024 -> 1008x756)
images_compressed_4x = resize_images(images_sequence, scale_factor=4)

# Уменьшаем в 8 раз (4032x3024 -> 504x378)
images_compressed_8x = resize_images(images_sequence, scale_factor=8)

# вот тут менять____________________________________________________________________
images_sequence = images_compressed_4x
```

Работа будет вестись над grayscale изображениями. То что мы будем использовать - не зависит от цветов. Все что возможно - можно сделать для RGB повторяя те же операций и преобразования, просто трижды - по разу для каждого цветового канала. Это не цель лабораторной работы.

Переведем изображения в черно-белый формат, используя функцию из прошлой лабы.
```python
images_sequence_gray = []
for img in images_sequence:
    images_sequence_gray.append(lb1.intensity_grayscale(img))
```

<img width="1002" height="452" alt="image" src="https://github.com/user-attachments/assets/b3b08b65-65a4-4c9b-8b53-180ca9e7d3b3" />
Составим необходимый алгоритм действий для выполнения лабораторной:
1. Загрузить изображения
2. Привести их к одинаковой яркости / grayscale
3. Найти ключевые точки
4. Отфильтровать точки
5. Построить дескрипторы (SIFT)
6. Сопоставить точки между соседними кадрами
7. Вычислить преобразование (поворот + сдвиг)
8. Накопить преобразования и построить траекторию камеры

# 2.2 Приведение к одинаковой яркости / grayscale

Исправим проблемы с яркостью, применив функцию для выравнивания гистограммы:


```python
images_hist_equalized = images_sequence_gray
for img in images_hist_equalized:
    img = lb1.hist_equalize(img)
    
show_images(images_sequence_gray, 2, 4)
```
<img width="1189" height="494" alt="image" src="https://github.com/user-attachments/assets/032a4c03-1576-4f18-85b5-141654f9f729" />
# 2.3 Найти ключевые точки

На изображениях есть фотография на бумаге и четыре магнита, удерживающие её на стенке холодильника, но нам могут значительно помешать: блики, тень от фотографа, другие нерелватные в этом контексте детали. Это визуальный <b>шум</b>. От шума надо избавиться, рассмотрим изображение 1, применим для подавления шумов фильтр Гаусса.
```python
images_temp = [images_hist_equalized[0], lb1.gaussian_2d(images_hist_equalized[0], 1.2, 21)]
show_images(images_temp, 1, 2)
```

limits for Z kernel 1.0 1.0
limits for Z kernel after normalization 7.658057422279819e-32 0.11052426603583844
limits for gaussian output 3.163609320933093e-13 253.00000870842632
<img width="1189" height="453" alt="image" src="https://github.com/user-attachments/assets/904a5e8c-8337-410d-87e4-e42be7c57492" />

Предыдущее действие сделало изображение немного более подходящим для дальнейшей работы с ним.

Перейдём к теории по следующему этапу:

Если бы мы смотрели на голубое небо и взяли его кусочек как ключевую точку, то распределение интенсивностей было бы +- одинаковое между ним и другим кусочком неба. Это плохая ключевая точка.

Если бы мы смотрели на фото чего-то геометрически простого, как например мебель, а конкретнее тумбочка, то могли бы предположить, что край тумбочки - хорошая ключевая точка, т.к. интенсивность достигает пика на моменте перехода от края тумбочки к ее боковой части в тени. По идее - уже неплохо, но если двигаться вдоль этого края - ситуация не поменяется при сравнении двух кадров.

Из всего этого следует, что лучший вариант - когда меняются по двум направлениям тренды. Например, угол тумбочки. Вдоль него не подвигаться, то есть изменение интенсивности слева и спереди (перед стеной) достаточно легко отслеживаются.

Из математики следует, что производная функции показывает скорость изменения ее значения. А градиент - вектор, показывающий <b>НАПРАВЛЕНИЕ</b> наибыстрейшего увеличения функции. Это то, что надо нам. Формула для градиента в общей форме выглядит так:
<img width="552" height="135" alt="image" src="https://github.com/user-attachments/assets/d790072a-178b-40e6-9b2a-aa35a4162503" />
Исходя из вышесказанного, понятно, что алгоритмы поиска ключевых точек ищут точки, где изменение яркости происходит во многих направлениях одновременно.

Математически - ищем градиенты изображения, по x и y координатам. 

Пусть I_x = изменение по х, I_y = изменение по y.
<p>Значит, место, где I_x = 0, I_y = 0 это однородная область.
<p>Если I_x большое, I_y маленькое = это край.
<p>Если I_x большое, I_y большое = это угол (ключевая точка).

Из первой лабораторной роботы мы понияли, что смотреть на сам один пиксель недостаточно. Нужно брать апертуру/кернел/область/окно. Так и сделаем. Идейно по определению подходит фильтр Хариса. Источник: https://docs.exponenta.ru/R2021a/visionhdl/ug/corner-detection.html

### 2.3.1 Разбор фильтра

Фактически у нас есть исходное изображение, minor_size, k, threshold_ratio.

minor_size - так же как в прошлой лабе, размер апертуры/кернел/окно. маленькое окно = чувствительность к мелким деталям, большое окно = реагирует только на крупные структуры

источник: https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html
<p> k - коэффициент, испольуемый в формуле Хариса:

<img width="552" height="160" alt="image" src="https://github.com/user-attachments/assets/875e1d99-9a53-48bc-b052-bb6a29c4c240" />

Этот коэффициент регулирует насколько алгоритм строго смотрит края.
<p>Маленький = алгоритм более терпим к краям, может принимать некоторые края за углы
<p>Большой = алгоритм строгий, оставляет только очень выраженные углы

<b> threshold_ratio: </b>

После вычисления Harris response R нужно решить какие точки считать ключевыми.

Для этого берётся максимум:
R_max = max(R)

и строится порог:
threshold = threshold_ratio * R_max

Если threshold_ratio = 0.01, то берутся точки у которых R > 1% от максимального

Если увеличить до 0.1 = останутся только самые сильные углы.

```python
def harris_keypoints(image, minor_size=5, k=0.04, threshold_ratio=0.01):

    if len(image.shape) == 3:
        image = intensity_grayscale(image)

    image = image.astype(float)

    height, width = image.shape

    Ix = np.zeros_like(image)
    Iy = np.zeros_like(image)

    # градиенты (центральные разности)
    for r in range(1, height-1):
        for c in range(1, width-1):
            Ix[r][c] = (image[r][c+1] - image[r][c-1]) / 2
            Iy[r][c] = (image[r+1][c] - image[r-1][c]) / 2

    pad = minor_size // 2

    R = np.zeros_like(image)

    for r in range(pad, height-pad):
        for c in range(pad, width-pad):

            sum_Ix2 = 0
            sum_Iy2 = 0
            sum_Ixy = 0

            for i in range(-pad, pad+1):
                for j in range(-pad, pad+1):
                    gx = Ix[r+i][c+j]
                    gy = Iy[r+i][c+j]

                    sum_Ix2 += gx*gx
                    sum_Iy2 += gy*gy
                    sum_Ixy += gx*gy

            det = sum_Ix2 * sum_Iy2 - sum_Ixy**2
            trace = sum_Ix2 + sum_Iy2

            R[r][c] = det - k*(trace**2)

    R_max = np.max(R)
    threshold = threshold_ratio * R_max

    keypoints = []

    for r in range(pad, height-pad):
        for c in range(pad, width-pad):

            if R[r][c] > threshold:

                local_max = True

                for i in range(-1,2):
                    for j in range(-1,2):
                        if R[r+i][c+j] > R[r][c]:
                            local_max = False

                if local_max:
                    keypoints.append((r,c))

    return keypoints, Ix, Iy
```
Что тут происходит? Сначала вычисляются упомянутые градиенты изображения.
Градиент показывает, насколько быстро меняется яркость:

Ix — изменение яркости по горизонтали

Iy — изменение яркости по вертикали

Это делается с помощью центральной разности: берётся разница между соседними пикселями.

После этого алгоритм начинает рассматривать каждую точку изображения и маленькое окно вокруг неё (minor_size). Внутри этого окна суммируются значения градиентов:

квадрат горизонтального градиента

квадрат вертикального градиента

произведение двух градиентов

Эти суммы используются для вычисления величины R. Она показывает, насколько вероятно, что точка является углом.

Дальше выбираются только точки, у которых R достаточно большое (больше порога).
После этого выполняется проверка локального максимума: точка должна быть больше всех своих соседей. Это нужно, чтобы оставить только самые сильные углы и убрать лишние точки вокруг них.

В итоге функция возвращает:

keypoints — координаты найденных углов

Ix — карту горизонтальных градиентов

Iy — карту вертикальных градиентов

```python
def draw_keypoints(image, keypoints, Ix=None, Iy=None, show_vectors=False, vector_scale=5):

    img = image.copy()

    plt.figure(figsize=(8,6))
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')  # Явно указываем grayscale
    else:
        plt.imshow(img)

    for (r,c) in keypoints:
        plt.scatter(c, r, s=20)

        if show_vectors and Ix is not None and Iy is not None:
            gx = Ix[r][c]
            gy = Iy[r][c]

            plt.arrow(c, r,
                      gx*vector_scale,
                      gy*vector_scale,
                      head_width=3,
                      length_includes_head=True)

    plt.axis("off")
    plt.show()
    fig = plt.gcf()
    fig.canvas.draw()
    
    buf = fig.canvas.buffer_rgba()
    data = np.asarray(buf)
    data = data[:, :, :3]
    
    return data
```
Функция draw_keypoints рисует найденные ключевые точки на изображении. Сначала создаётся копия изображения. Если изображение чёрно-белое, оно превращается в трёхканальное (RGB), чтобы на нём можно было рисовать цветные элементы.


Для каждой найденной точки:

на изображении рисуется маркер (точка)

если включён параметр show_vectors, дополнительно рисуется стрелка

Стрелка показывает направление градиента в этой точке. Она берётся из Ix и Iy и показывает, в каком направлении яркость изменяется сильнее всего.

Параметр vector_scale просто увеличивает длину стрелок, чтобы их было лучше видно.


```python
gray = images_temp[1]

keypoints, Ix, Iy = harris_keypoints(gray, minor_size=9, k=0.04, threshold_ratio=0.01)

draw_keypoints(gray, keypoints)
```
<img width="636" height="482" alt="image" src="https://github.com/user-attachments/assets/ad982e42-6306-4730-aab8-2a464e3ecb0f" />
<Figure size 640x480 with 0 Axes>

Как видим, ключевые точки отлично отобразились. Но здесь, думаю, сыграло хорошее качество изображения и правильно подобранный наугад размер апертуры. Попробуем с более быстрым вариантом, с апертурой поменьше:

```python
gray = images_temp[1]

keypoints, Ix, Iy = harris_keypoints(gray, minor_size=5)

draw_keypoints(gray, keypoints)
```
<img width="636" height="482" alt="image" src="https://github.com/user-attachments/assets/bdc5ffe2-17ca-4893-9dcb-f574d2b31c5e" />
Точек меньше. Для анализа была написана функция рисующая стрелки направлений для обнаруженных градиентов, посмотрим:
```python
draw_keypoints(gray, keypoints, Ix, Iy, show_vectors=True)
```
<img width="551" height="482" alt="image" src="https://github.com/user-attachments/assets/087319d1-fa6e-4278-8436-bfc09a974923" />

Для наглядности отобразим градиенты и сравним с изначальным изображением.

<img width="950" height="247" alt="image" src="https://github.com/user-attachments/assets/d61702a6-b5eb-486a-ac14-8e4d2202b3af" />

# 2.4 Фильтрация точек
Перед поиском точек мы заранее позаботились о том, чтобы скорректировать освещение изображений и избавиться от жестких бликов и прочих плохих факторов, которые могли негативно повлиять обнаружение. Тем самым был опережён план этой работы.

Хоть на полученном результате поиска точек мы не видим ошибочных (вне исследуемых объектов), но возможно очень много отличных ситуаций, в которых они могут быть найдены. Например, если дверца холодильника, послужившая фоном будет загрязнена или поцарапана - это может дать сильный скачок градиента и такой расклад сделает даже повторное размытие бесполезным. Более того, с дополнительным размытиеммы уменьшаем эффективность поиска градиентов. Нужен альтернативный метод. Для таких задач используются разные методы. Некоторые из них: DBSCAN, RANSAC.

RANSAC - популярный, но сложен для текущего уровня погружения в теорию. DBSCAN - тоже популярен, но понятнее. Применяется кластеризация, а потом фильтрация по соседям. Возьмем как основу этот метод, но уберём из него класетиразацию для упрощения. Будем проверять по количеству соседей точки напрямую. Применительно к даннй работы - это должно стать отличным решением. Для очень загруженных пятнами изображений пригодится уже упомянутая кластеризация.

Пока оставим в следующем виде:
```python
def filter_isolated_points(keypoints, radius=10, min_neighbors=5):

    filtered = []

    for i, p in enumerate(keypoints):

        neighbors = 0

        for j, q in enumerate(keypoints):

            if i == j:
                continue

            dist = np.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2)

            if dist < radius:
                neighbors += 1

        if neighbors >= min_neighbors:
            filtered.append(p)

    return filtered
```
Что происходит:

берём точку

смотрим сколько других точек ближе чем radius

если их меньше min_neighbors, считаем её шумом

Таким образом исчезнут, возможные точки ошибок, потому что рядом с ней нет соседей.

```python
keypoints_filtered = filter_isolated_points(keypoints, 300)
draw_keypoints(gray, keypoints_filtered)
```
<img width="636" height="482" alt="image" src="https://github.com/user-attachments/assets/1d400c54-4913-4f8b-aee2-455e6dfefef2" />

Мы избавились от вброса но с этим потеряли много нужных точек! Настраиваем наш фильтр:
```python
keypoints_filtered = filter_isolated_points(keypoints, 300, 3)
draw_keypoints(gray, keypoints_filtered)
```
<img width="636" height="482" alt="image" src="https://github.com/user-attachments/assets/1345dc6d-c4af-4b5f-be24-5346ec51bae6" />

Теперь перейдём к дескрипторам.

 # 2.5 Построение дескрипторов (SIFT)

 Дескриптор — это числовое описание окрестности ключевой точки. Он должен быть устойчив к изменению освещения, небольшому повороту и сдвигу — чтобы одна и та же точка на двух разных кадрах давала похожий дескриптор, а разные точки — непохожие.
<p>SIFT (Scale-Invariant Feature Transform) — один из самых известных алгоритмов для этого. Он строит дескриптор из гистограмм градиентов в окрестности точки. Алгоритм состоит из четырёх этапов, разберём каждый.

Подготовим материалы для дальнейшей работы по аналогии с прошлыми шагами:
```python
working_images = []
for i in images_hist_equalized:
    working_images.append(lb1.gaussian_2d(i, 1.2, 21))
    
show_images(working_images)
```
limits for Z kernel 1.0 1.0
limits for Z kernel after normalization 7.658057422279819e-32 0.11052426603583844
limits for gaussian output 3.163609320933093e-13 253.00000870842632
limits for Z kernel 1.0 1.0
limits for Z kernel after normalization 7.658057422279819e-32 0.11052426603583844
limits for gaussian output 3.458141693931284e-21 252.9663474994394
limits for Z kernel 1.0 1.0
limits for Z kernel after normalization 7.658057422279819e-32 0.11052426603583844
limits for gaussian output 5.474673108273487e-14 252.8718107877355
limits for Z kernel 1.0 1.0
limits for Z kernel after normalization 7.658057422279819e-32 0.11052426603583844
limits for gaussian output 4.816821895670379e-18 252.99999744799152
limits for Z kernel 1.0 1.0
limits for Z kernel after normalization 7.658057422279819e-32 0.11052426603583844
limits for gaussian output 1.996054377706734e-09 252.99999997289632
limits for Z kernel 1.0 1.0
limits for Z kernel after normalization 7.658057422279819e-32 0.11052426603583844
limits for gaussian output 7.050926399277544e-11 252.99999970335162
limits for Z kernel 1.0 1.0
limits for Z kernel after normalization 7.658057422279819e-32 0.11052426603583844
limits for gaussian output 2.630634622012462e-13 252.9713101122748
limits for Z kernel 1.0 1.0
limits for Z kernel after normalization 7.658057422279819e-32 0.11052426603583844
limits for gaussian output 3.7849109486005766e-13 252.99994819945033

<img width="1189" height="494" alt="image" src="https://github.com/user-attachments/assets/db80fd6b-7165-44cc-8a49-0690021b8129" />

На всех изображениях было применено размытие. Это подавит мелкий шум перед вычислением градиентов. Теперь найдём и отобразим ключевые точки для всей серии:

```python
working_images_keypoints = []
working_images_visualised = []
for i in working_images:
    keypoints, Ix, Iy = harris_keypoints(i, minor_size=3)
    keypoints = filter_isolated_points(keypoints, 300, 3)
    working_images_keypoints.append(keypoints)
    working_images_visualised.append(draw_keypoints(i, keypoints))
```
<img width="636" height="482" alt="image" src="https://github.com/user-attachments/assets/9bb2e26a-4e8f-4113-81e8-46f692b80a1a" />
<img width="636" height="482" alt="image" src="https://github.com/user-attachments/assets/1b0cf687-cef5-4c5a-88e2-69b7c4b2da9c" />
<img width="636" height="482" alt="image" src="https://github.com/user-attachments/assets/53096c90-91ea-4798-98d8-a219ee3889ad" />
<img width="636" height="482" alt="image" src="https://github.com/user-attachments/assets/bed34dd1-f3e0-4a28-bd79-2d46b55b0935" />
<img width="636" height="482" alt="image" src="https://github.com/user-attachments/assets/2f511bd6-1c4a-4fde-9913-5ce533d6c456" />
<img width="636" height="482" alt="image" src="https://github.com/user-attachments/assets/5bed5d51-3e93-45ea-904b-b8c9f8e82c89" />
<img width="636" height="482" alt="image" src="https://github.com/user-attachments/assets/d4b570a5-280a-4f42-9332-a1287d0a2b53" />
<img width="636" height="482" alt="image" src="https://github.com/user-attachments/assets/46666825-9f82-4d7b-997c-cbd2ce644d36" />

Результат приемлемый. Ключевые точки найдены на всех кадрах. Заметно, что точки кластеризуются вокруг объектов — магнитов и вокруг лица на фотографии, — что и ожидается: именно там происходят резкие изменения яркости в обоих направлениях. Переходим к теории SIFT.

Harris дал точки, но он не умеет их узнавать на другом изображении. Если повернуть картинку, изменить масштаб или освещение — координаты точек изменятся.

Именно поэтому появился алгоритм SIFT. Его задача: для каждой точки построить уникальное числовое описание (дескриптор), которое можно сравнивать между кадрами.
SIFT — идея алгоритма

Алгоритм делает две вещи: находит устойчивые точки интереса и строит для каждой точки вектор признаков, который описывает локальную структуру изображения

Этот вектор потом можно сравнивать между изображениями.

Классический SIFT состоит из 4 этапов:

## 2.5.1 Пирамиды Гаусса
Первый шаг — создать несколько размытых версий изображения.

Это нужно, чтобы точки находились независимо от масштаба.
Мелкие детали исчезают при сильном размытии, а крупные остаются.

Алгоритм строит так называемую пирамиду Гаусса.
```python
def gaussian_pyramid(image, sigmas=[1,2,4,8]):
    
    pyramid = []
    
    for sigma in sigmas:
        blurred = lb1.gaussian_2d(image, sigma, minor_size=17)
        pyramid.append(blurred)
        
    return pyramid

pyramid1 = gaussian_pyramid(working_images[0])
show_images(pyramid1, 1, 4)
<img width="1189" height="230" alt="image" src="https://github.com/user-attachments/assets/28624062-8163-4784-a2eb-9e5537b30853" />
```
На каждом следующем уровне пирамиды изображение размывается сильнее: мелкие детали пропадают, крупные структуры остаются. Это позволяет находить точки на разных масштабах. В нашей задаче камера движется примерно на одном расстоянии от объекта, поэтому масштабная инвариантность не критична: пирамида строится для полноты алгоритма.

## 2.5.2 Разница гауссиан DoG

DoG (Difference of Gaussians) — это разница между соседними уровнями пирамиды Гаусса. Математически это приближение лапласиана гауссиана (LoG), который хорошо реагирует на точки и края.
<p>В оригинальном SIFT именно в DoG-пространстве ищутся экстремумы — точки, которые являются максимумом или минимумом среди 26 соседей (8 в своём слое + 9 выше + 9 ниже). Функция будет реализована, но использовать для детектирования не будет, так как уже есть Харис.


```python
def difference_of_gaussians(pyramid):
    
    dogs = []
    
    for i in range(len(pyramid)-1):
        dog = pyramid[i+1] - pyramid[i]
        dogs.append(dog)
        
    return dogs
```
Обычно SIFT ищет точки в DoG, но мы уже реализовали алгоритм Хариса, поэтому будем использовать его. Пирамида тоже не нужна для детектирования: только для масштабной инвариантности, которая в рамках данной лабораторной не имеет приоритета на другими задачами, так как расстояние от камеры до холодильника колеблется несильно.

## 2.5.3 Экстремумы

Шаг 2.5.3 важен — он нужен не для поиска новых точек, а для того чтобы назначить каждой точке доминирующий угол по локальной гистограмме градиентов. Без этого дескриптор не будет инвариантен к повороту.

```python
def compute_keypoint_orientations(keypoints, Ix, Iy,
                                  orientation_window_size=16,
                                  num_bins=36):

    height, width = Ix.shape
    half = orientation_window_size // 2

    sigma = half
    oriented_keypoints = []

    for (r, c) in keypoints:
        if r < half or r >= height - half or c < half or c >= width - half:
            continue

        hist = [0.0] * num_bins
        bin_width = 360.0 / num_bins

        for i in range(-half, half):
            for j in range(-half, half):

                gx = Ix[r + i][c + j]
                gy = Iy[r + i][c + j]


                magnitude = math.sqrt(gx * gx + gy * gy)
                angle_deg = math.degrees(math.atan2(gy, gx)) % 360
                gauss_weight = math.exp(-(i * i + j * j) / (2 * sigma * sigma))

                bin_idx = int(angle_deg / bin_width) % num_bins
                hist[bin_idx] += magnitude * gauss_weight

        max_val = max(hist)
        peak_bin = hist.index(max_val)

        dominant_angle = math.radians((peak_bin + 0.5) * bin_width)

        oriented_keypoints.append((r, c, dominant_angle))

    return oriented_keypoints
```
Что происходит внутри:
<p>Для каждой точки берётся окно orientation_window_size * orientation_window_size пикселей.
В каждом пикселе считается магнитуда и угол градиента. Вклад каждого пикселя взвешивается на магнитуду и на гауссов вес — пиксели в центре окна важнее, чем на краях.
<p>Вклады накапливаются в гистограмму из 36 бинов (шаг 10°, покрывают 0..360°). Бин с максимальным значением даёт доминирующую ориентацию точки.
<p>На выходе каждая точка (r, c) превращается в тройку (r, c, angle_rad) — теперь дескриптор будет строиться относительно этого угла и станет инвариантен к повороту камеры.

```python
oriented_kp = compute_keypoint_orientations(keypoints, Ix, Iy)
print(f"Точек после ориентации: {len(oriented_kp)}")
```
Точек после ориентации: 53

Точки получили ориентацию. Переходим к построению самого дескриптора.

## 2.5.4 Построение дескриптора

Для каждой ориентированной точки берём 16x16 пикселей и делим его на сетку 4x4 блока (каждый 4x4 пикселя).
<p>В каждом блоке строится гистограмма градиентов по 8 направлениям (бины по 45 градусов). Ключевой момент: угол каждого пикселя считается относительно доминирующей ориентации точки — это и даёт инвариантность к повороту.
<p>16 блоков x 8 бинов = вектор из 128 чисел. Он нормализуется, затем значения обрезаются на уровне 0.2 (это стандартный трюк SIFT для подавления нелинейностей освещения) и нормализуются снова
         
```python
def compute_sift_descriptors(oriented_keypoints, Ix, Iy,
                              patch_size=16,
                              num_spatial_bins=4,
                              num_orientation_bins=8):

    height, width = Ix.shape
    half = patch_size // 2
    cell_size = patch_size // num_spatial_bins
    bin_width = 360.0 / num_orientation_bins

    valid_keypoints = []
    descriptors = []

    for (r, c, dominant_angle) in oriented_keypoints:
        if r < half or r >= height - half or c < half or c >= width - half:
            continue

        histograms = []
        for bi in range(num_spatial_bins):
            row_hists = []
            for bj in range(num_spatial_bins):
                row_hists.append([0.0] * num_orientation_bins)
            histograms.append(row_hists)

        for i in range(-half, half):
            for j in range(-half, half):

                gx = Ix[r + i][c + j]
                gy = Iy[r + i][c + j]

                magnitude = math.sqrt(gx * gx + gy * gy)

                raw_angle = math.degrees(math.atan2(gy, gx))
                relative_angle = (raw_angle - math.degrees(dominant_angle)) % 360

                bi = (i + half) // cell_size
                bj = (j + half) // cell_size

                bi = min(bi, num_spatial_bins - 1)
                bj = min(bj, num_spatial_bins - 1)

                bin_idx = int(relative_angle / bin_width) % num_orientation_bins

                histograms[bi][bj][bin_idx] += magnitude

        descriptor = []
        for bi in range(num_spatial_bins):
            for bj in range(num_spatial_bins):
                for val in histograms[bi][bj]:
                    descriptor.append(val)

        descriptor = np.array(descriptor, dtype=float)

        norm = np.sqrt(np.sum(descriptor * descriptor))
        if norm > 1e-6:
            descriptor = descriptor / norm

        descriptor = np.clip(descriptor, 0, 0.2)

        norm2 = np.sqrt(np.sum(descriptor * descriptor))
        if norm2 > 1e-6:
            descriptor = descriptor / norm2

        valid_keypoints.append((r, c, dominant_angle))
        descriptors.append(descriptor)

    return valid_keypoints, np.array(descriptors)
```
Дескриптор готов. Проверим на одном изображении:

valid_kp, descs = compute_sift_descriptors(oriented_kp, Ix, Iy)
print(f"Дескрипторов: {len(descs)}, форма вектора: {descs[0].shape}")

Дескрипторов: 53, форма вектора: (128,)

128-мерный вектор для каждой точки получен. Теперь применим пайплайн ко всей серии изображений:
```python
all_keypoints = []
all_descriptors = []

for img in working_images:
    kp, Ix, Iy = harris_keypoints(img, minor_size=3)
    kp = filter_isolated_points(kp, 30, 3)
    oriented = compute_keypoint_orientations(kp, Ix, Iy)
    valid_kp, descs = compute_sift_descriptors(oriented, Ix, Iy)
    all_keypoints.append(valid_kp)
    all_descriptors.append(descs)
```
Дескрипторы построены для всех кадров. Количество точек может отличаться между кадрами — это нормально, часть точек отсеивается у края изображения.

Теперь приступаем к основному этапу работы - сопоставим соседние кадры и визуализируем это.

Оформим функцию show_images_any, которая уже и с grayscale и с rgb справится.

```python
def show_images_any(images, rows=2, cols=4, figsize=(12, 6), titles=None):
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
 
    for i, ax in enumerate(axes):
        if i < len(images):
            img = images[i]
            if len(img.shape) == 2:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
            if titles and i < len(titles):
                ax.set_title(titles[i], fontsize=9)
            ax.axis('off')
        else:
            ax.axis('off')
 
    plt.tight_layout()
    plt.show()
```
# 2.6 Сопоставить точки между соседними кадрами
Идея: Для каждого дескриптора из кадра A ищем ближайший дескриптор в кадре B по евклидовому расстоянию.

Тест Лоу (Lowe's ratio test): берём два ближайших соседа (best и second_best). Если dist(best) < ratio * dist(second_best), матч считается надёжным. Стандартное значение ratio = 0.75. Смысл: если лучший матч явно лучше второго — он скорее всего правильный. Если они близки по расстоянию — скорее всего оба неправильные.

P.S. Далее "матч" = match. С англ. совпадение.

```python
def match_descriptors(kp_a, desc_a, kp_b, desc_b, ratio=0.75):
    matches = []
 
    for i in range(len(desc_a)):
        best_dist = float('inf')
        second_dist = float('inf')
        best_j = -1
 
        for j in range(len(desc_b)):
            dist = euclidean_distance(desc_a[i], desc_b[j])
 
            if dist < best_dist:
                second_dist = best_dist
                best_dist = dist
                best_j = j
            elif dist < second_dist:
                second_dist = dist
 
        if second_dist > 1e-6 and best_dist / second_dist < ratio:
            r_a, c_a = kp_a[i][0], kp_a[i][1]
            r_b, c_b = kp_b[best_j][0], kp_b[best_j][1]
            matches.append(((r_a, c_a), (r_b, c_b)))
 
    return matches
```
Функция euclidean_distance считает расстояние между двумя дескрипторами-векторами.
<p>match_descriptors для каждой точки кадра A перебирает все точки кадра B и находит два ближайших дескриптора. Тест Лоу отсеивает неоднозначные матчи: если лучший и второй по качеству кандидат похожи по расстоянию — значит точка неуникальная и лучше её отбросить. Остаются только чёткие, уверенные совпадения.

```python
def draw_matches(img_a, img_b, matches, max_display=50):
    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]
 

    h_combined = max(h_a, h_b)
    combined = np.zeros((h_combined, w_a + w_b, 3), dtype=np.uint8)
 

    if len(img_a.shape) == 2:
        combined[:h_a, :w_a] = np.dstack((img_a, img_a, img_a))
    else:
        combined[:h_a, :w_a] = img_a
 

    if len(img_b.shape) == 2:
        combined[:h_b, w_a:w_a + w_b] = np.dstack((img_b, img_b, img_b))
    else:
        combined[:h_b, w_a:w_a + w_b] = img_b
 
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(combined)
 
    np.random.seed(42)
 
    displayed = matches[:max_display]
    for ((r_a, c_a), (r_b, c_b)) in displayed:
        color = (np.random.random(), np.random.random(), np.random.random())
        ax.scatter(c_a, r_a, s=15, color=color, zorder=3)
        ax.scatter(c_b + w_a, r_b, s=15, color=color, zorder=3)
        ax.plot([c_a, c_b + w_a], [r_a, r_b], color=color, linewidth=0.8, alpha=0.7)
 
    ax.axis('off')
    ax.set_title(f'Матчей показано: {len(displayed)} из {len(matches)}')
    plt.tight_layout()
    plt.show()
```
draw_matches соединяет два кадра горизонтально и рисует цветные линии между сопоставленными точками. Каждая пара обозначена своим цветом для наглядности.
<p>Запускаем матчинг для всех соседних пар кадров:
         
```python
all_matches = []

for i in range(len(working_images) - 1):
    print(f"\nКадр {i} → Кадр {i+1}")
    matches = match_descriptors(
        all_keypoints[i], all_descriptors[i],
        all_keypoints[i+1], all_descriptors[i+1]
    )
    print(f"  Найдено матчей: {len(matches)}")
    all_matches.append(matches)

    draw_matches(working_images[i], working_images[i+1], matches)
```

Кадр 0 → Кадр 1
  Найдено матчей: 7
<img width="1389" height="556" alt="image" src="https://github.com/user-attachments/assets/a942727e-4f68-4ccf-b38a-7ebc8f6c2da1" />

Кадр 1 → Кадр 2
  Найдено матчей: 13
<img width="1389" height="556" alt="image" src="https://github.com/user-attachments/assets/590701b0-68cd-4fdd-98f4-e10246263382" />

Кадр 2 → Кадр 3
  Найдено матчей: 12
<img width="1389" height="556" alt="image" src="https://github.com/user-attachments/assets/d4338d11-76e8-4f51-95e3-996659ec599f" />

Кадр 3 → Кадр 4
  Найдено матчей: 13
<img width="1389" height="556" alt="image" src="https://github.com/user-attachments/assets/efcf0f58-6e35-468f-b991-bcdfb8513801" />

Кадр 4 → Кадр 5
  Найдено матчей: 9
<img width="1389" height="556" alt="image" src="https://github.com/user-attachments/assets/02f11c8d-9132-4d9e-bcbc-cac36fdbc698" />

Кадр 5 → Кадр 6
  Найдено матчей: 11
<img width="1389" height="556" alt="image" src="https://github.com/user-attachments/assets/66068f15-bdbd-450e-8c30-855afcbf94e2" />

Кадр 6 → Кадр 7
  Найдено матчей: 11
<img width="1389" height="556" alt="image" src="https://github.com/user-attachments/assets/7000487f-13cc-45e5-8cdf-8a133258d622" />

Матчи найдены. Видно, что точки на магнитах уверенно сопоставляются между кадрами.

# 2.7 Вычисление преобразования (поворот и сдвиг)
Модель преобразования:
   У нас только поворот и сдвиг (без масштаба и проективных искажений).
   Это называется rigid body transformation (жёсткое тело):
   <p>
   [x']   [cos θ  -sin θ] [x]   [tx]
   <p>
   [y'] = [sin θ   cos θ] [y] + [ty]
   <p>
   Как считаем? По парам матчей вычисляем угол поворота и вектор сдвига.
   Шаги:
   1. Центрируем обе группы точек (вычитаем центроид)
   2. Для каждой пары считаем угол: atan2(y_b - y_centroid_b, x_b - ...) и т.д.
      Точнее — используем cross и dot product между центрированными векторами,
      это даёт угол поворота по каждой паре.
   3. Усредняем углы (через sin/cos, иначе проблемы с переходом через 0/360).
   4. Вычисляем сдвиг: tx, ty = centroid_b - R @ centroid_a

Функция также реализует упрощённый RANSAC — повторяем выборку случайных пар N раз, берём модель с наибольшим консенсусом. Это защищает от неправильных матчей (outliers), которые всегда есть.

```python
def estimate_rotation_translation(matches, ransac_iterations=500, inlier_threshold=5.0):
    if len(matches) < 2:
        print("Недостаточно матчей")
        return 0.0, 0.0, 0.0, []
 
    def fit_model(sample):
        n = len(sample)
 
        cax = sum(m[0][1] for m in sample) / n
        cay = sum(m[0][0] for m in sample) / n
        cbx = sum(m[1][1] for m in sample) / n
        cby = sum(m[1][0] for m in sample) / n
 
        dot   = 0.0
        cross = 0.0
        for ((r_a, c_a), (r_b, c_b)) in sample:
            ax = c_a - cax;  ay = r_a - cay
            bx = c_b - cbx;  by = r_b - cby
            dot   += ax * bx + ay * by
            cross += ax * by - ay * bx
 
        angle = math.atan2(cross, dot)
 
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        tx = cbx - (cos_a * cax - sin_a * cay)
        ty = cby - (sin_a * cax + cos_a * cay)
 
        return angle, tx, ty
 
    def count_inliers(all_matches, angle, tx, ty):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        inliers = []
        for ((r_a, c_a), (r_b, c_b)) in all_matches:
            xp = cos_a * c_a - sin_a * r_a + tx
            yp = sin_a * c_a + cos_a * r_a + ty
            err = math.sqrt((xp - c_b)**2 + (yp - r_b)**2)
            if err < inlier_threshold:
                inliers.append(((r_a, c_a), (r_b, c_b)))
        return inliers
 
    best_angle = 0.0
    best_tx    = 0.0
    best_ty    = 0.0
    best_inliers = []
 
    for _ in range(ransac_iterations):
        sample = random.sample(matches, min(4, len(matches)))
        angle, tx, ty = fit_model(sample)
        inliers = count_inliers(matches, angle, tx, ty)
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_angle   = angle
            best_tx      = tx
            best_ty      = ty
 
    if len(best_inliers) >= 2:
        best_angle, best_tx, best_ty = fit_model(best_inliers)
 
    print(f"  Угол поворота: {math.degrees(best_angle):.2f}°")
    print(f"  Сдвиг: tx={best_tx:.1f}px, ty={best_ty:.1f}px")
    print(f"  Inliers: {len(best_inliers)} из {len(matches)} матчей")
 
    return best_angle, best_tx, best_ty, best_inliers
```
Супер, смотрим трансформации для всей выборки.

```python
transforms = []

for i, matches in enumerate(all_matches):
    print(f"\nТрансформация {i} → {i+1}:")
    angle, tx, ty, inliers = estimate_rotation_translation(matches)
    transforms.append((angle, tx, ty))

    draw_matches(working_images[i], working_images[i+1], inliers)
```
Трансформация 0 → 1:
  Угол поворота: 0.54°
  Сдвиг: tx=261.0px, ty=-4.2px
  Inliers: 7 из 7 матчей
<img width="1389" height="556" alt="image" src="https://github.com/user-attachments/assets/804bcb10-ebff-4261-a956-c17482d27edb" />

Трансформация 1 → 2:
  Угол поворота: -0.05°
  Сдвиг: tx=18.3px, ty=124.3px
  Inliers: 13 из 13 матчей
<img width="1389" height="556" alt="image" src="https://github.com/user-attachments/assets/5a8fae4c-fcce-4a4c-ac50-5de003db32b9" />

Трансформация 2 → 3:
  Угол поворота: 0.11°
  Сдвиг: tx=3.7px, ty=68.1px
  Inliers: 11 из 12 матчей
<img width="1389" height="556" alt="image" src="https://github.com/user-attachments/assets/2c285ce1-be91-4724-8bd7-b144bda86085" />

Трансформация 3 → 4:
  Угол поворота: -0.27°
  Сдвиг: tx=-255.4px, ty=-5.8px
  Inliers: 12 из 13 матчей
<img width="1389" height="556" alt="image" src="https://github.com/user-attachments/assets/6879dd0d-05e0-4cc5-a92a-eec08eb9e362" />

Трансформация 4 → 5:
  Угол поворота: -0.69°
  Сдвиг: tx=-257.2px, ty=12.0px
  Inliers: 9 из 9 матчей
<img width="1389" height="556" alt="image" src="https://github.com/user-attachments/assets/537bde0e-41a2-40d7-b394-991859e7a7ec" />

Трансформация 5 → 6:
  Угол поворота: 0.50°
  Сдвиг: tx=-8.3px, ty=-100.4px
  Inliers: 10 из 11 матчей
<img width="1389" height="556" alt="image" src="https://github.com/user-attachments/assets/afb5c165-c855-4c29-862d-a4143366761f" />

Трансформация 6 → 7:
  Угол поворота: -0.56°
  Сдвиг: tx=-9.5px, ty=-81.8px
  Inliers: 10 из 11 матчей
<img width="1389" height="556" alt="image" src="https://github.com/user-attachments/assets/f887c482-7bff-4d70-aba4-e569f6cd5304" />

После RANSAC остались только матчи, согласующиеся с моделью поворот+сдвиг. Количество inliers относительно общего числа матчей показывает качество сопоставления: чем выше доля тем лучше.

# 2.8 Накопить преобразования и построить траекторию камеры
Теперь у нас есть список трансформаций для каждой соседней пары кадров. Осталось накопить их и получить траекторию.

<p>Первый подход — накапливать (angle, tx, ty) последовательно. Позиция камеры на шаге i+1 вычисляется через позицию на шаге i с учётом накопленного угла. Этот метод работает, но ошибки накапливаются от кадра к кадру.
         
```python
def build_trajectory(transforms):
    positions = [(0.0, 0.0)]
    angles    = [0.0]
 
    cam_x = 0.0
    cam_y = 0.0
    global_angle = 0.0
 
    for (local_angle, tx, ty) in transforms:
        global_angle += local_angle
 
        cam_x += -tx
        cam_y += ty
 
        positions.append((cam_x, cam_y))
        angles.append(global_angle)
 
    return positions, angles
```
build_trajectory идёт по списку трансформаций и на каждом шаге прибавляет к позиции камеры инвертированный сдвиг: если объект уехал вправо на tx камера уехала влево. Угол накапливается отдельно и используется только для отрисовки стрелок направления.

Визуализируем:

```python
def draw_trajectory(positions, angles=None, image_labels=None):
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
 
    fig, ax = plt.subplots(figsize=(8, 8))
 
    ax.plot(xs, ys, color='steelblue', linewidth=1.5, zorder=1)
 
    ax.scatter(xs, ys, color='steelblue', s=40, zorder=2)
 
    if angles is not None:
        span = max(max(xs) - min(xs), max(ys) - min(ys))
        arrow_len = span * 0.06 + 5
        for (x, y), a in zip(positions, angles):
            dx = math.cos(a) * arrow_len
            dy = math.sin(a) * arrow_len
            ax.annotate('', xy=(x + dx, y + dy), xytext=(x, y),
                        arrowprops=dict(arrowstyle='->', color='tomato', lw=1.5))
 
    labels = image_labels if image_labels else [str(i) for i in range(len(positions))]
    for i, (x, y) in enumerate(positions):
        ax.annotate(labels[i], (x, y),
                    textcoords='offset points', xytext=(6, 6),
                    fontsize=9, color='dimgray')

    ax.scatter([xs[0]], [ys[0]], color='green', s=100, zorder=3, label='Старт')
    ax.scatter([xs[-1]], [ys[-1]], color='red',   s=100, zorder=3, label='Финиш')

    closure_error = math.sqrt((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2)
    ax.plot([xs[-1], xs[0]], [ys[-1], ys[0]],
            color='gray', linewidth=1.0, linestyle='--', alpha=0.6,
            label=f'Ошибка замыкания: {closure_error:.1f}px')
 
    ax.invert_yaxis()
 
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title('Траектория камеры')
    ax.set_xlabel('X (пиксели)')
    ax.set_ylabel('Y (пиксели)')
 
    plt.tight_layout()
    plt.show()
```

```python
positions, angles = build_trajectory(transforms)

labels = [f'img{i+1}' for i in range(len(positions))]
draw_trajectory(positions, angles, image_labels=labels)

print("\nСводная таблица трансформаций:")
print(f"{'Пара':<12} {'Угол (°)':<12} {'tx (px)':<12} {'ty (px)':<12}")
for i, (angle, tx, ty) in enumerate(transforms):
    print(f"{i}→{i+1:<9} {math.degrees(angle):<12.2f} {tx:<12.1f} {ty:<12.1f}")
```

<img width="786" height="348" alt="image" src="https://github.com/user-attachments/assets/7fc94aa4-f538-4587-99e5-73088616371e" />

Сводная таблица трансформаций:
Пара         Угол (°)     tx (px)      ty (px)     
0→1         0.54         261.0        -4.2        
1→2         -0.05        18.3         124.3       
2→3         0.11         3.7          68.1        
3→4         -0.27        -255.4       -5.8        
4→5         -0.69        -257.2       12.0        
5→6         0.50         -8.3         -100.4      
6→7         -0.56        -9.5         -81.8

Траектория выглядит разумно по форме, однако при сравнении с реальным маршрутом камеры видно, что ошибка замыкания велика: накопленные неточности в трансформациях уводят финишную точку далеко от старта. Попробуем более точный метод — через центроиды ключевых точек.

<p>Идея: вместо того чтобы накапливать трансформации, мы для каждого кадра независимо считаем центр масс всех ключевых точек. Это позиция объекта в пикселях кадра. Сдвиг объекта между кадрами, это и есть движение в системе координат изображения. Ошибки не накапливаются, каждый кадр независим.

```python
def build_trajectories_from_keypoints(all_keypoints):
    #считаем центроид каждого кадра
    centroids = []
    for kps in all_keypoints:
        if len(kps) == 0:
            centroids.append(None)
            continue
        mean_c = sum(kp[1] for kp in kps) / len(kps)
        mean_r = sum(kp[0] for kp in kps) / len(kps)
        centroids.append((mean_c, mean_r))
 
    origin = next((c for c in centroids if c is not None), (0.0, 0.0))
    x0, y0 = origin
 
    obj_positions = []
    cam_positions = []
 
    for c in centroids:
        if c is None:
            obj_positions.append(None)
            cam_positions.append(None)
        else:
            dx = c[0] - x0
            dy = c[1] - y0
            obj_positions.append(( dx,  dy))   #объект
            cam_positions.append((-dx, -dy))   #камера
 
    return obj_positions, cam_positions, centroids
```
build_trajectories_from_keypoints считает центроид всех ключевых точек для каждого кадра.
Смещение относительно первого кадра даёт траекторию объекта. Камера движется строго противоположно: если объект уехал на (dx, dy) в пикселях кадра, камера переместилась на (-dx, -dy) в мировых координатах.

```python
def draw_trajectory_generic(positions, image_labels=None,
                             title='Траектория', color='steelblue'):

    valid = [(i, p) for i, p in enumerate(positions) if p is not None]
    idxs  = [v[0] for v in valid]
    xs    = [v[1][0] for v in valid]
    ys    = [v[1][1] for v in valid]
    labels = image_labels if image_labels else [str(i) for i in range(len(positions))]
 
    fig, ax = plt.subplots(figsize=(8, 8))
 
    ax.plot(xs, ys, color=color, linewidth=1.5, zorder=1)
    ax.scatter(xs, ys, color=color, s=40, zorder=2)
 
    for idx, x, y in zip(idxs, xs, ys):
        ax.annotate(labels[idx], (x, y),
                    textcoords='offset points', xytext=(6, 6),
                    fontsize=9, color='dimgray')
 
    ax.scatter([xs[0]], [ys[0]], color='green', s=100, zorder=3, label='Старт')
    ax.scatter([xs[-1]], [ys[-1]], color='red',  s=100, zorder=3, label='Финиш')
 
    closure = math.sqrt((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2)
    ax.plot([xs[-1], xs[0]], [ys[-1], ys[0]],
            color='gray', linewidth=1.0, linestyle='--', alpha=0.6,
            label=f'Ошибка замыкания: {closure:.1f}px')
 
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(title)
    ax.set_xlabel('X (пиксели)')
    ax.set_ylabel('Y (пиксели)')
 
    plt.tight_layout()
    plt.show()
```

draw_trajectory_generic это универсальная функция отрисовки. Принимает любой список позиций, заголовок и цвет. Рисует траекторию, подписывает точки, отмечает старт и финиш, показывает пунктиром ошибку замыкания.

```python
def draw_both_trajectories(obj_positions, cam_positions, image_labels=None):
    labels = image_labels if image_labels else \
             [str(i) for i in range(len(obj_positions))]
 
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
 
    configs = [
        (obj_positions, 'darkorange', 'Траектория объекта'),
        (cam_positions, 'steelblue',  'Траектория камеры'),
    ]
 
    for ax, (positions, color, title) in zip(axes, configs):
        valid = [(i, p) for i, p in enumerate(positions) if p is not None]
        idxs  = [v[0] for v in valid]
        xs    = [v[1][0] for v in valid]
        ys    = [v[1][1] for v in valid]
 
        ax.plot(xs, ys, color=color, linewidth=1.5, zorder=1)
        ax.scatter(xs, ys, color=color, s=40, zorder=2)
 
        for idx, x, y in zip(idxs, xs, ys):
            ax.annotate(labels[idx], (x, y),
                        textcoords='offset points', xytext=(6, 6),
                        fontsize=9, color='dimgray')
 
        ax.scatter([xs[0]], [ys[0]], color='green', s=100, zorder=3, label='Старт')
        ax.scatter([xs[-1]], [ys[-1]], color='red',  s=100, zorder=3, label='Финиш')
 
        closure = math.sqrt((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2)
        ax.plot([xs[-1], xs[0]], [ys[-1], ys[0]],
                color='gray', linewidth=1.0, linestyle='--', alpha=0.6,
                label=f'Ошибка замыкания: {closure:.1f}px')
 
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_title(title)
        ax.set_xlabel('X (пиксели)')
        ax.set_ylabel('Y (пиксели)')
 
    plt.tight_layout()
    plt.show()
```
draw_both_trajectories выводит обе траектории рядом для сравнения. Теперь запустим:

```python
labels = [f'img{i+1}' for i in range(len(working_images))]
obj_positions, cam_positions, centroids = build_trajectories_from_keypoints(all_keypoints)
```

Центроиды посчитаны. Рисуем траектории по отдельности:

```python
draw_trajectory_generic(obj_positions, labels, 'Траектория объекта', 'darkorange')
draw_trajectory_generic(cam_positions, labels, 'Траектория камеры',  'steelblue')
```
<img width="786" height="467" alt="image" src="https://github.com/user-attachments/assets/4ea40612-a934-4541-a20c-5c0a7e401d56" />
<img width="786" height="460" alt="image" src="https://github.com/user-attachments/assets/a61a40fa-f560-4378-bfaf-5b26a47e7e9b" />




# Лабораторная работа 4
## Определения количества пальцев одной руки на изображении
### Выполнил: студент группы 8Е21 Плотников Юрий Андреевич

Импорт библиотек

```python
import os
import copy
import time
import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import classification_report, confusion_matrix

import cv2

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)

torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Настройка устройства


```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```
Device: cuda

GPU: NVIDIA GeForce RTX 3060 Laptop GPU


Параметры
```python
IMAGE_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 20
LR = 3e-4

DATASET_PATH = r"D:\Libraries\Documents\VScode\CV_labs_Plotnikov_8E21\Lab_4\dataset\Finger_Count_Images_Dataset"
```

Для RTX 3060 6GB batch=32 обычно безопасен.

Аугментации и preprocessing

```python
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(15),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

Загрузка датасета и split

```python
full_train_dataset = datasets.ImageFolder(
    root=DATASET_PATH,
    transform=train_transforms
)

full_eval_dataset = datasets.ImageFolder(
    root=DATASET_PATH,
    transform=test_transforms
)

indices = torch.randperm(len(full_train_dataset)).tolist()

train_size = int(0.7 * len(indices))
val_size = int(0.15 * len(indices))

train_idx = indices[:train_size]

val_idx = indices[
    train_size:train_size + val_size
]

test_idx = indices[
    train_size + val_size:
]

from torch.utils.data import Subset

train_dataset = Subset(
    full_train_dataset,
    train_idx
)

val_dataset = Subset(
    full_eval_dataset,
    val_idx
)

test_dataset = Subset(
    full_eval_dataset,
    test_idx
)

class_names = full_train_dataset.classes

print("Classes:", class_names)

print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))
```

Classes: ['0', '1', '2', '3', '4', '5']

Train: 4443

Validation: 952

Test: 953

Разделение:
- 70% train
- 15% validation
- 15% test

DataLoader

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)
```

Модель EfficientNet-B0

```python
model = models.efficientnet_b0(weights="IMAGENET1K_V1")
```

Меняем classifier под 6 классов:

```python
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 6)
model = model.to(device)
```

Loss и optimizer

```python
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LR
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    patience=2,
    factor=0.5
)

scaler = torch.amp.GradScaler('cuda')
```

Проверка количества классов

```python
print(class_names)
```

['0', '1', '2', '3', '4', '5']

```python

```

Функция обучения

```python
def train_model(model, train_loader, val_loader):

    os.makedirs("saved_epochs", exist_ok=True)

    train_losses = []
    val_losses = []

    best_acc = 0.0

    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        # =========================
        # TRAIN
        # =========================
        model.train()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_loader):

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):

                outputs = model(inputs)

                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            running_loss += loss.item() * inputs.size(0)

            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)

        epoch_acc = (
            running_corrects.double() /
            len(train_loader.dataset)
        ).item()

        train_losses.append(epoch_loss)

        # =========================
        # VALIDATION
        # =========================
        model.eval()

        val_running_loss = 0.0
        val_running_corrects = 0

        with torch.no_grad():

            for inputs, labels in val_loader:

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                _, preds = torch.max(outputs, 1)

                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)

                val_running_corrects += torch.sum(
                    preds == labels.data
                )

        val_loss = (
            val_running_loss /
            len(val_loader.dataset)
        )

        val_acc = (
            val_running_corrects.double() /
            len(val_loader.dataset)
        ).item()

        val_losses.append(val_loss)

        scheduler.step(val_loss)

        print(f"Train Loss: {epoch_loss:.4f}")
        print(f"Train Acc : {epoch_acc:.4f}")

        print(f"Val Loss  : {val_loss:.4f}")
        print(f"Val Acc   : {val_acc:.4f}")

        # ====================================
        # СОХРАНЯЕМ ТЕКУЩУЮ ЭПОХУ
        # ====================================
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': epoch_loss,
            'val_loss': val_loss,
            'train_acc': epoch_acc,
            'val_acc': val_acc
        }

        epoch_path = (
            f"saved_epochs/"
            f"epoch_{epoch+1}_"
            f"valacc_{val_acc:.4f}.pth"
        )

        torch.save(checkpoint, epoch_path)

        print(f"Saved: {epoch_path}")

        # ====================================
        # СОХРАНЯЕМ ЛУЧШУЮ МОДЕЛЬ
        # ====================================
        if val_acc > best_acc:

            best_acc = val_acc

            best_model_path = "saved_epochs/best_model.pth"

            torch.save(checkpoint, best_model_path)

            print(
                f"New best model saved "
                f"(val_acc={val_acc:.4f})"
            )

    print(f"\nBest validation accuracy: {best_acc:.4f}")

    return model, train_losses, val_losses
```

Сделаем финальную проверку перед обучением

```python
images, labels = next(iter(train_loader))
print(images.shape)
print(labels.shape)
print(labels[:10])
```

torch.Size([64, 3, 128, 128])

torch.Size([64])

tensor([2, 4, 3, 2, 2, 4, 0, 3, 2, 4])

```python
sample_img, sample_label = train_dataset[0]
print(sample_img.shape)
print(sample_label)
```

torch.Size([3, 128, 128])

3

Обучение

```python
model, train_losses, val_losses = train_model(
    model,
    train_loader,
    val_loader
)
```
Epoch 1/20
100%|██████████| 70/70 [01:30<00:00,  1.29s/it]

Train Loss: 1.4809

Train Acc : 0.3907

Val Loss  : 1.0349

Val Acc   : 0.5746

Saved: saved_epochs/epoch_1_valacc_0.5746.pth

New best model saved (val_acc=0.5746)

Epoch 2/20

100%|██████████| 70/70 [01:17<00:00,  1.11s/it]

Train Loss: 0.9238

Train Acc : 0.6415

Val Loss  : 0.7682

Val Acc   : 0.6954

Saved: saved_epochs/epoch_2_valacc_0.6954.pth

New best model saved (val_acc=0.6954)

Epoch 3/20

100%|██████████| 70/70 [01:21<00:00,  1.16s/it]

Train Loss: 0.6799

Train Acc : 0.7463

Val Loss  : 0.7029

Val Acc   : 0.7363

Saved: saved_epochs/epoch_3_valacc_0.7363.pth

New best model saved (val_acc=0.7363)



Epoch 4/20

100%|██████████| 70/70 [01:25<00:00,  1.22s/it]

Train Loss: 0.5430

Train Acc : 0.7943

Val Loss  : 0.6254

Val Acc   : 0.7805

Saved: saved_epochs/epoch_4_valacc_0.7805.pth

New best model saved (val_acc=0.7805)



Epoch 5/20

100%|██████████| 70/70 [01:28<00:00,  1.27s/it]

Train Loss: 0.4358

Train Acc : 0.8386

Val Loss  : 0.6230

Val Acc   : 0.7826

Saved: saved_epochs/epoch_5_valacc_0.7826.pth

New best model saved (val_acc=0.7826)



Epoch 6/20

100%|██████████| 70/70 [01:27<00:00,  1.24s/it]

Train Loss: 0.3802

Train Acc : 0.8623

Val Loss  : 0.6166

Val Acc   : 0.7868

Saved: saved_epochs/epoch_6_valacc_0.7868.pth

New best model saved (val_acc=0.7868)



Epoch 7/20

100%|██████████| 70/70 [01:21<00:00,  1.17s/it]

Train Loss: 0.3062

Train Acc : 0.8843

Val Loss  : 0.5949

Val Acc   : 0.7920

Saved: saved_epochs/epoch_7_valacc_0.7920.pth

New best model saved (val_acc=0.7920)



Epoch 8/20

100%|██████████| 70/70 [01:24<00:00,  1.21s/it]

Train Loss: 0.2720

Train Acc : 0.8974

Val Loss  : 0.6632

Val Acc   : 0.7920

Saved: saved_epochs/epoch_8_valacc_0.7920.pth



Epoch 9/20

100%|██████████| 70/70 [01:29<00:00,  1.28s/it]

Train Loss: 0.2300

Train Acc : 0.9187

Val Loss  : 0.6067

Val Acc   : 0.8193

Saved: saved_epochs/epoch_9_valacc_0.8193.pth

New best model saved (val_acc=0.8193)



Epoch 10/20

100%|██████████| 70/70 [01:24<00:00,  1.21s/it]

Train Loss: 0.1989

Train Acc : 0.9264

Val Loss  : 0.6069

Val Acc   : 0.8088

Saved: saved_epochs/epoch_10_valacc_0.8088.pth



Epoch 11/20

100%|██████████| 70/70 [01:25<00:00,  1.23s/it]

Train Loss: 0.1647

Train Acc : 0.9390

Val Loss  : 0.5538

Val Acc   : 0.8319

Saved: saved_epochs/epoch_11_valacc_0.8319.pth

New best model saved (val_acc=0.8319)



Epoch 12/20

100%|██████████| 70/70 [01:24<00:00,  1.21s/it]

Train Loss: 0.1338

Train Acc : 0.9525

Val Loss  : 0.5589

Val Acc   : 0.8309

Saved: saved_epochs/epoch_12_valacc_0.8309.pth



Epoch 13/20

100%|██████████| 70/70 [01:26<00:00,  1.24s/it]

Train Loss: 0.1085

Train Acc : 0.9617

Val Loss  : 0.5627

Val Acc   : 0.8277

Saved: saved_epochs/epoch_13_valacc_0.8277.pth



Epoch 14/20

100%|██████████| 70/70 [01:28<00:00,  1.27s/it]

Train Loss: 0.1004

Train Acc : 0.9667

Val Loss  : 0.6036

Val Acc   : 0.8277

Saved: saved_epochs/epoch_14_valacc_0.8277.pth



Epoch 15/20

100%|██████████| 70/70 [01:22<00:00,  1.18s/it]

Train Loss: 0.0973

Train Acc : 0.9669

Val Loss  : 0.5782

Val Acc   : 0.8330

Saved: saved_epochs/epoch_15_valacc_0.8330.pth

New best model saved (val_acc=0.8330)



Epoch 16/20

100%|██████████| 70/70 [01:24<00:00,  1.21s/it]

Train Loss: 0.0729

Train Acc : 0.9737

Val Loss  : 0.5570

Val Acc   : 0.8372

Saved: saved_epochs/epoch_16_valacc_0.8372.pth

New best model saved (val_acc=0.8372)



Epoch 17/20

100%|██████████| 70/70 [01:29<00:00,  1.28s/it]

Train Loss: 0.0757

Train Acc : 0.9750

Val Loss  : 0.5694

Val Acc   : 0.8403

Saved: saved_epochs/epoch_17_valacc_0.8403.pth

New best model saved (val_acc=0.8403)



Epoch 18/20

100%|██████████| 70/70 [01:27<00:00,  1.25s/it]

Train Loss: 0.0645

Train Acc : 0.9795

Val Loss  : 0.5817

Val Acc   : 0.8351

Saved: saved_epochs/epoch_18_valacc_0.8351.pth



Epoch 19/20

100%|██████████| 70/70 [01:18<00:00,  1.12s/it]

Train Loss: 0.0553

Train Acc : 0.9842

Val Loss  : 0.5723

Val Acc   : 0.8445

Saved: saved_epochs/epoch_19_valacc_0.8445.pth

New best model saved (val_acc=0.8445)



Epoch 20/20

100%|██████████| 70/70 [01:23<00:00,  1.19s/it]

Train Loss: 0.0600

Train Acc : 0.9829

Val Loss  : 0.5697

Val Acc   : 0.8414

Saved: saved_epochs/epoch_20_valacc_0.8414.pth


Best validation accuracy: 0.8445


```python
best_checkpoint = torch.load(
    "saved_epochs/best_model.pth",
    map_location=device
)

model.load_state_dict(
    best_checkpoint['model_state_dict']
)

model = model.to(device)

model.eval()

print("Loaded best model")

print("Best epoch:", best_checkpoint['epoch'])

print("Best validation accuracy:",
      best_checkpoint['val_acc'])
```

Loaded best model

Best epoch: 19

Best validation accuracy: 0.8445378151260504

Тестирование

```python
model.eval()

all_preds = []

all_labels = []

test_correct = 0

test_total = 0

with torch.no_grad(), torch.amp.autocast('cuda'):

    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        test_correct += torch.sum(
            preds == labels
        ).item()

        test_total += labels.size(0)

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

test_accuracy = test_correct / test_total

print(f"\nTest Accuracy: {test_accuracy:.4f}")
```

Test Accuracy: 0.8478

```python
print(classification_report(
    all_labels,
    all_preds,
    target_names=class_names
))
```

<img width="533" height="290" alt="image" src="https://github.com/user-attachments/assets/73729aaf-759f-4bbb-af83-02c4bf568c54" />

```python
cm = confusion_matrix(
    all_labels,
    all_preds
)

plt.figure(figsize=(8, 6))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks(
    range(len(class_names)),
    class_names
)

plt.yticks(
    range(len(class_names)),
    class_names
)

plt.xlabel("Predicted")

plt.ylabel("True")

for i in range(len(class_names)):
    for j in range(len(class_names)):

        plt.text(
            j,
            i,
            cm[i, j],
            ha='center',
            va='center'
        )

plt.show()
```

<img width="610" height="547" alt="image" src="https://github.com/user-attachments/assets/8c6a742b-ecc1-4b4c-b507-e0c285a274fe" />

Webcam inference

```python
model.eval()

camera_transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # =================================
    # ROI
    # =================================
    x1, y1 = 100, 100
    x2, y2 = 400, 400

    roi = frame[y1:y2, x1:x2]

    # =================================
    # PREPROCESS
    # =================================
    roi_rgb = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2RGB
    )

    input_tensor = camera_transform(
        roi_rgb
    )

    input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.to(device)

    # =================================
    # INFERENCE
    # =================================
    with torch.no_grad(), torch.amp.autocast('cuda'):

        outputs = model(input_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, pred = torch.max(
            probabilities,
            1
        )

        predicted_class = class_names[
            pred.item()
        ]

        confidence = confidence.item()

    # =================================
    # DRAW ROI
    # =================================
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # =================================
    # DRAW TEXT
    # =================================
    cv2.putText(
        frame,
        f"Finger Count: {predicted_class}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Finger Counter",
        frame
    )

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()

cv2.destroyAllWindows()
```

## Результаты работы 
<img width="797" height="634" alt="Screenshot 2026-05-08 175844" src="https://github.com/user-attachments/assets/1d0b766d-b2ad-46c5-8b8b-3b350d218f0f" />
<img width="788" height="644" alt="Screenshot 2026-05-08 175646" src="https://github.com/user-attachments/assets/e2c9e026-8859-476e-ad1a-53921310aa07" />
<img width="793" height="626" alt="Screenshot 2026-05-08 175618" src="https://github.com/user-attachments/assets/67b24b5f-e5c4-4657-989c-a326a4667df9" />
<img width="796" height="637" alt="Screenshot 2026-05-08 175529" src="https://github.com/user-attachments/assets/54f6b6dc-0a90-4e23-85fe-baee358b868a" />
<img width="788" height="630" alt="Screenshot 2026-05-08 175522" src="https://github.com/user-attachments/assets/08576210-aac4-4296-a0c8-c27c034ea81c" />
<img width="793" height="627" alt="Screenshot 2026-05-08 175459" src="https://github.com/user-attachments/assets/78998fdc-59bd-41c1-88a9-adbde86696da" />
<img width="791" height="624" alt="Screenshot 2026-05-08 175443" src="https://github.com/user-attachments/assets/c4730639-236c-49f2-9a0b-1ce28e4f6dcf" />
<img width="795" height="636" alt="Screenshot 2026-05-08 175430" src="https://github.com/user-attachments/assets/cbe7deae-ee8b-46a2-bde1-39e3c4f2a3ba" />
<img width="790" height="633" alt="Screenshot 2026-05-08 175424" src="https://github.com/user-attachments/assets/372dfa4b-9df5-4aa7-bba1-53a6291a6055" />
<img width="792" height="631" alt="Screenshot 2026-05-08 175358" src="https://github.com/user-attachments/assets/041dba86-9da4-4f68-8580-7acebe78a4c2" />
<img width="795" height="629" alt="Screenshot 2026-05-08 175340" src="https://github.com/user-attachments/assets/0d38ad69-fbfc-4562-afdf-0ba656c479b2" />
<img width="793" height="632" alt="Screenshot 2026-05-08 175329" src="https://github.com/user-attachments/assets/51f5d388-2d16-4eaf-b1f1-2b66d24da164" />
<img width="798" height="631" alt="Screenshot 2026-05-08 175307" src="https://github.com/user-attachments/assets/ef422ea3-e821-4077-b878-ba39e5b10f8e" />
<img width="794" height="638" alt="Screenshot 2026-05-08 175219" src="https://github.com/user-attachments/assets/4f12c9c0-a376-403b-9b94-0d183ae55452" />
<img width="796" height="645" alt="Screenshot 2026-05-08 175200" src="https://github.com/user-attachments/assets/ce429ed2-b0b0-4360-ae17-e3890a39ee33" />
<img width="791" height="629" alt="Screenshot 2026-05-08 175101" src="https://github.com/user-attachments/assets/d48d78fb-81d2-47a9-a1f9-1369ee81bc22" />
<img width="798" height="633" alt="Screenshot 2026-05-08 175041" src="https://github.com/user-attachments/assets/94d61714-0952-4146-a02a-fe895f53cc1b" />


## Вывод
В результате выполнения лабораторной работы был реализован классификатор количества пальцев одной руки.
