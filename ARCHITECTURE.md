# Архітектура Feature Matcher Studio

## Загальна структура

```
Feature Matcher Studio
│
├── UI Layer (PyQt5)
│   ├── main_window.py
│   │   ├── MainWindow (3 вкладки)
│   │   ├── VideoWorker (QThread для обробки)
│   │   └── QDoubleSpinBoxCompat
│   │
│   └── image_viewer.py
│       └── ImageViewer (QGraphicsView + zoom/pan)
│
├── Vision Core
│   ├── feature_strategies.py
│   │   ├── FeatureStrategy (ABC)
│   │   ├── ORBStrategy
│   │   ├── AKAZEStrategy
│   │   ├── SIFTStrategy
│   │   └── get_available_strategies()
│   │
│   ├── matcher.py
│   │   ├── MatchResult (dataclass)
│   │   └── ImageMatcher
│   │       ├── detect_and_match()
│   │       ├── _preprocess()
│   │       ├── _ratio_filter()
│   │       ├── _reprojection_error()
│   │       ├── _adaptive_threshold()
│   │       ├── _heatmap()
│   │       └── _draw_matches()
│   │
│   └── video_processor.py
│       └── VideoProcessor
│           └── process_video()
│
└── main.py (Entry point)
```

---

## Дизайн-паттерни

### 1. Strategy Pattern (feature_strategies.py)

Дозволяє динамічно вибирати алгоритм ознак без зміни основного коду:

```python
class FeatureStrategy(ABC):
    """Abstract base for feature detection strategies."""
    @abstractmethod
    def create_detector(self):
        pass

class ORBStrategy(FeatureStrategy):
    def create_detector(self):
        return cv2.ORB_create(...)

class SIFTStrategy(FeatureStrategy):
    def create_detector(self):
        return cv2.SIFT_create(...)

# Usage
matcher = ImageMatcher(strategy=ORBStrategy())
matcher.set_strategy(SIFTStrategy())  # Switch at runtime
```

**Переваги:**

- Легко додавати нові алгоритми (BRISK, AKAZE, тощо)
- Без змін в основній логіці
- Клієнт не залежить від конкретної реалізації

### 2. Dataclass for Results (matcher.py)

```python
@dataclass
class MatchResult:
    matched_image: Optional[np.ndarray]
    heatmap: Optional[np.ndarray]
    homography: Optional[np.ndarray]
    mask: Optional[np.ndarray]
    similarity: float
    similarity_good: float
    inliers: int
    reprojection_error: float
    adaptive_threshold: float
    stability: float
    keypoints1: int
    keypoints2: int
    good_matches: int
    method: str
```

**Переваги:**

- Типова безпека
- Автоматично генерує `__init__`, `__repr__`, тощо
- Чиста структура даних

### 3. Threading for Video (main_window.py)

```python
class VideoWorker(QThread):
    progress = pyqtSignal(int)
    match_found = pyqtSignal(int, object)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def run(self):
        # Довга операція в фоновому потоці
        for frame_idx, total_frames, result in processor.process_video(...):
            self.progress.emit(progress_pct)
            self.match_found.emit(frame_idx, result)
```

**Переваги:**

- UI не блокується під час обробки видео
- Можна паузувати/стопити процес
- Прогрес-бар оновлюється рідко

---

## Основні алгоритми

### Image Matching Pipeline

```
Вхід: img1, img2
│
├─ Preprocessing
│  └─ Grayscale + optional Gaussian blur
│
├─ Feature Detection & Description
│  └─ detectAndCompute() → keypoints1, descriptors1
│
├─ Matching
│  ├─ KNN: matcher.knnMatch(k=2) + Lowe ratio test
│  └─ OR crossCheck: matcher.match()
│
├─ Homography Estimation
│  ├─ Collect good matches
│  ├─ findHomography() with RANSAC
│  └─ Get mask (inliers)
│
├─ Metrics Calculation
│  ├─ similarity = inliers / min(kp)
│  ├─ reprojection_error = perspectiveTransform() error
│  ├─ stability = 1 / (1 + error)
│  └─ adaptive_threshold = weighted score
│
├─ Visualization
│  ├─ drawMatches() with mask
│  ├─ Project corners + polylines
│  └─ Heatmap generation
│
└─ Вихід: MatchResult
```

### Heatmap Generation

```python
def _heatmap(img, keypoints):
    heat = np.zeros((h, w))
    for kp in keypoints:
        heat[int(kp.y), int(kp.x)] += 1.0
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=10)
    colored = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 0.6, colored, 0.4, 0)
```

**Результат:** Красная область = густа масса ключевых точек

### Adaptive Threshold

```python
def _adaptive_threshold(similarity, similarity_good, good_count):
    score = 0.6 * similarity + 0.3 * similarity_good + 0.1 * good_count
    return np.clip(score, 5.0, 90.0)
```

**Логіка:**

- 60% від основної метрики (inliers/min)
- 30% від добрих збігів (good/min)
- 10% від кількості збігів
- Результат в діапазоні [5%, 90%]

### Video Frame Selection

```python
def process_video(video_path, reference_image, matcher):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        match_result = matcher.detect_and_match(reference_image, frame)
        dynamic_threshold = max(user_threshold, adaptive_threshold)

        if match_result.similarity >= dynamic_threshold:
            yield frame_idx, result

        frame_idx += 1
```

---

## UI Architecture (PyQt5)

### Вкладка структура

```python
MainWindow
├─ Tab 1: Compare Images
│  ├─ Left Panel (Controls)
│  │  ├─ Load section (buttons)
│  │  ├─ Analysis section (buttons)
│  │  └─ Settings section (combo/spin)
│  └─ Right Panel (Results)
│     ├─ Image viewer + similarity label
│     ├─ Heatmap viewer
│     └─ Details label
│
├─ Tab 2: Video Search
│  ├─ Left Panel (Controls)
│  │  ├─ Load section
│  │  ├─ Playback section
│  │  ├─ Video params
│  │  └─ Algorithm section
│  └─ Right Panel (Results)
│     ├─ Progress bar
│     ├─ Timeline (matched frames)
│     ├─ Video viewer
│     └─ Frame details
│
└─ Tab 3: Help
   ├─ Algorithm descriptions
   ├─ Matcher modes info
   ├─ Metrics explanation
   └─ Tips & tricks
```

### Dock Widget (Log Panel)

Динамічно показує/сховує лог внизу вікна:

- Success (✓ зелено)
- Warning (⚠ жовто)
- Error (✗ червоно)

### Dark Theme (QSS)

```css
QWidget {
    background-color: #1e1f29;
    color: #f5f5f5;
}
QPushButton:hover {
    background-color: #3a3d4d;
}
QPushButton:disabled {
    color: #777;
}
QProgressBar::chunk {
    background-color: #5b8def;
}
```

---

## Обробка помилок

```
try:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        emit error("Не вдалося відкрити відео")
        return
except Exception as e:
    emit error(f"Помилка: {str(e)}")
finally:
    cap.release()
```

---

## Тестування

### Unit Test Example

```python
def test_orb_matching():
    img1 = cv2.imread("test1.jpg")
    img2 = cv2.imread("test2.jpg")
    matcher = ImageMatcher(strategy=ORBStrategy())
    result = matcher.detect_and_match(img1, img2)

    assert result.homography is not None
    assert result.similarity > 20
    assert len(result.matched_image.shape) == 3
```

---

## Performance Optimization

| Операція               | Час        | Оптимізація                 |
| ---------------------- | ---------- | --------------------------- |
| ORB detection          | ~50ms      | Швидко за замовчуванням     |
| AKAZE detection        | ~100ms     | Збалансована                |
| SIFT detection         | ~500ms     | Спеціалізована              |
| RANSAC                 | ~20ms      | Вбудована в OpenCV          |
| Heatmap                | ~10ms      | Gaussian blur оптимізований |
| Video (30fps, skip=10) | ~3fps eff. | Потік в фоні                |

---

## Майбутні розширення

1. **FLANN Matcher** — для великих датасетів
2. **GPU Acceleration** — CUDA для faster matching
3. **Feature-preserving Blur** — Bilateral filter
4. **Multi-scale Detection** — Pyramid matching
5. **Template Matching** — cv2.matchTemplate()
6. **Batch Processing** — Обробка множини кадрів
7. **Export Results** — Збереження результатів в JSON/CSV
8. **Real-time Camera Feed** — Фід з вебкамери

---

## Залежності

```
PyQt5      — UI framework
opencv-python — Computer vision
numpy      — Numerical operations
```

Ніяких deep learning бібліотек (TensorFlow, PyTorch) не використовується.

---

## Висновок

**Feature Matcher Studio** демонструє:

- ✓ Clean architecture з поділом на слої (UI / Vision)
- ✓ Design patterns (Strategy, Dataclass, Threading)
- ✓ Сучасні OpenCV методи (ORB, AKAZE, SIFT, RANSAC)
- ✓ User-friendly PyQt5 interface
- ✓ Готовність до розширення
- ✓ Коментований і структурований код
