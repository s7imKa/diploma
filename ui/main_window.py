import os
from typing import Optional, List
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QIcon, QColor, QTextCursor, QFont
from PyQt5.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QSpinBox,
    QGroupBox,
    QFormLayout,
    QTabWidget,
    QComboBox,
    QCheckBox,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QDockWidget,
    QPlainTextEdit,
    QFrame,
    QSplitter,
    QDoubleSpinBox,
)

import cv2
import numpy as np

from vision.matcher import ImageMatcher, MatchResult
from vision.video_processor import VideoProcessor
from vision.feature_strategies import get_available_strategies, FeatureStrategy
from ui.image_viewer import ImageViewer


class VideoWorker(QThread):
    progress = pyqtSignal(int)
    match_found = pyqtSignal(int, object)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, processor: VideoProcessor, matcher: ImageMatcher, video_path: str, ref_img: np.ndarray):
        super().__init__()
        self.processor = processor
        self.matcher = matcher
        self.video_path = video_path
        self.ref_img = ref_img

    def run(self):
        """Run video processing in background thread."""
        try:
            gen = self.processor.process_video(self.video_path, self.ref_img, self.matcher)
            if gen is None:
                self.error.emit("Не вдалося відкрити відео")
                return
            for frame_idx, total_frames, match_result in gen:
                pct = int(frame_idx / max(1, total_frames) * 100)
                self.progress.emit(min(pct, 100))
                self.match_found.emit(frame_idx, match_result)
                if self.processor.stopped:
                    break
            self.progress.emit(100)
            self.finished.emit()
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit()


class MainWindow(QMainWindow):
    """Main application window with image/video matching interface."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Feature Matcher Studio")
        self.resize(1400, 950)

        # Core components
        self.strategies: List[FeatureStrategy] = get_available_strategies()
        self.matcher = ImageMatcher(strategy=self.strategies[0])
        self.video_processor = VideoProcessor()
        self.video_worker: Optional[VideoWorker] = None

        # State
        self.image1: Optional[np.ndarray] = None
        self.image2: Optional[np.ndarray] = None
        self.video_path: Optional[str] = None
        self.video_matches: List[tuple[int, MatchResult]] = []
        self.current_tab = 0  # 0=Images, 1=Video
        self.last_result: Optional[MatchResult] = None  # Store for demo/save

        # UI components (initialized later)
        self.tabs: Optional[QTabWidget] = None
        self.log_widget: Optional[QPlainTextEdit] = None
        self.status_indicator: Optional[QLabel] = None

        # Button states for enable/disable
        self.compare_btn: Optional[QPushButton] = None
        self.start_video_btn: Optional[QPushButton] = None
        
        # New: verdict and explanation labels
        self.verdict_label: Optional[QLabel] = None
        self.explanation_label: Optional[QLabel] = None

        self._init_ui()
        self._apply_dark_theme()

    def _init_ui(self):
        """Build main UI structure: tabs with compare/video tabs."""
        # Initialize log dock FIRST so _on_tab_changed() can use it
        self._init_log_dock()
        self._init_status_bar()

        central = QWidget()
        self.setCentralWidget(central)

        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self._init_compare_tab()
        self._init_video_tab()
        self._init_settings_tab()

        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        layout.setContentsMargins(0, 0, 0, 0)
        central.setLayout(layout)

    # UI pieces
    def _init_log_dock(self):
        """Initialize log panel as dockable widget at bottom."""
        self.log_widget = QPlainTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setMaximumHeight(150)
        self.log_widget.setMinimumHeight(80)

        dock = QDockWidget("📋 Лог операцій", self)
        dock.setWidget(self.log_widget)
        dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)

    def _init_status_bar(self):
        """Initialize status bar with indicator and progress info."""
        self.status_indicator = QLabel("✓ Готово")
        self.status_indicator.setStyleSheet(self._status_color("green"))
        self.status_indicator.setMinimumWidth(120)

        self.statusBar().addPermanentWidget(self.status_indicator)

    def _status_color(self, color: str) -> str:
        """Return QSS for colored status indicator."""
        return (
            f"QLabel {{ background-color:{color}; color:#fff; padding:6px 12px; "
            f"border-radius:4px; font-weight:bold; }}"
        )

    def _init_compare_tab(self):
        """Tab 1: Image comparison with controls, results, heatmap, and verdict."""
        tab = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(12, 12, 12, 12)

        # ==================== LEFT PANEL: Controls ====================
        left_panel = self._build_compare_controls()

        # ==================== CENTER: Results ====================
        center_panel = QWidget()
        center_layout = QVBoxLayout()
        center_layout.setSpacing(8)

        # VERDICT & EXPLANATION (UX improvement #1)
        verdict_group = QGroupBox("📊 Результат аналізу")
        verdict_layout = QVBoxLayout()
        self.verdict_label = QLabel("🟡 Очікування...")
        self.verdict_label.setStyleSheet("font-weight: bold; font-size: 13pt; color: #FFA500;")
        self.verdict_label.setAlignment(Qt.AlignCenter)
        self.explanation_label = QLabel("Завантажте два зображення та натисніть «Порівняти»")
        self.explanation_label.setStyleSheet("font-size: 10pt; color: #CCCCCC;")
        self.explanation_label.setAlignment(Qt.AlignCenter)
        self.explanation_label.setWordWrap(True)
        verdict_layout.addWidget(self.verdict_label)
        verdict_layout.addWidget(self.explanation_label)
        verdict_group.setLayout(verdict_layout)
        center_layout.addWidget(verdict_group, 0)

        # Result viewer with label
        result_group = QGroupBox("🖼️ Результат збігу")
        result_layout = QVBoxLayout()
        self.compare_viewer = ImageViewer()
        self.compare_viewer.setMinimumHeight(360)
        result_layout.addWidget(self.compare_viewer)
        result_group.setLayout(result_layout)
        center_layout.addWidget(result_group, 3)

        # Metrics display
        self.similarity_label = QLabel("Схожість: —")
        self.similarity_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        center_layout.addWidget(self.similarity_label)

        # Heatmap viewer
        heatmap_group = QGroupBox("🔥 Теплова карта ключових точок")
        heatmap_layout = QVBoxLayout()
        self.heatmap_viewer = ImageViewer()
        self.heatmap_viewer.setMinimumHeight(200)
        heatmap_layout.addWidget(self.heatmap_viewer)
        heatmap_group.setLayout(heatmap_layout)
        center_layout.addWidget(heatmap_group, 1)

        center_panel.setLayout(center_layout)

        # ==================== Details panel (bottom) ====================
        self.details_label = QLabel("Деталі: —")
        self.details_label.setWordWrap(True)
        self.details_label.setStyleSheet("background:#2b2d37; padding:8px; border-radius:4px;")

        # Combine center + details
        right_layout = QVBoxLayout()
        right_layout.addWidget(center_panel, 1)
        right_layout.addWidget(self.details_label, 0)

        right_panel = QWidget()
        right_panel.setLayout(right_layout)

        # ==================== Assemble ====================
        main_layout.addWidget(left_panel, 0)
        main_layout.addWidget(right_panel, 1)
        tab.setLayout(main_layout)

        self.tabs.addTab(tab, "🖼️ Порівняння зображень")

    def _build_compare_controls(self) -> QWidget:
        """Build left control panel for image comparison."""
        panel = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(12)

        # === SECTION 1: Load Images ===
        load_group = QGroupBox("1️⃣ Завантажити зображення")
        load_layout = QVBoxLayout()
        load_layout.setSpacing(8)

        btn_img1 = self._make_button("📂 Перше зображення", self.load_first_image, "Завантажте еталон для пошуку")
        btn_img1.setMinimumHeight(40)
        load_layout.addWidget(btn_img1)

        btn_img2 = self._make_button("📂 Друге зображення", self.load_second_image, "Завантажте зображення для порівняння")
        btn_img2.setMinimumHeight(40)
        load_layout.addWidget(btn_img2)

        load_group.setLayout(load_layout)
        layout.addWidget(load_group)

        # === SECTION 2: Action ===
        action_group = QGroupBox("2️⃣ Аналіз")
        action_layout = QVBoxLayout()
        action_layout.setSpacing(8)

        self.compare_btn = self._make_button("▶️ Порівняти", self.compare, "Знайти збіги між зображеннями (RANSAC + гомографія)")
        self.compare_btn.setMinimumHeight(48)
        self.compare_btn.setStyleSheet(self.compare_btn.styleSheet() + "\nfont-size: 11pt; font-weight: bold;")
        self.compare_btn.setEnabled(False)
        action_layout.addWidget(self.compare_btn)

        btn_heatmap = self._make_button("🔥 Heatmap", self.show_heatmap, "Показати щільність ключових точок")
        action_layout.addWidget(btn_heatmap)

        action_group.setLayout(action_layout)
        layout.addWidget(action_group)

        # === SECTION 3: Settings (Advanced) ===
        settings_group = QGroupBox("⚙️ Налаштування (Image)")
        settings_layout = QFormLayout()
        settings_layout.setSpacing(8)

        self.feature_combo = QComboBox()
        for strat in self.strategies:
            self.feature_combo.addItem(strat.name)
        self.feature_combo.currentIndexChanged.connect(self.on_feature_change)
        settings_layout.addRow("Алгоритм:", self.feature_combo)

        self.matcher_mode_combo = QComboBox()
        self.matcher_mode_combo.addItems(["KNN + Lowe ratio", "BF crossCheck"])
        self.matcher_mode_combo.currentIndexChanged.connect(self.on_matcher_mode_change)
        self.matcher_mode_combo.setToolTip("KNN: два найкращі кандидати + ratio test\nCrossCheck: взаємна відповідність")
        settings_layout.addRow("Matcher:", self.matcher_mode_combo)

        self.ratio_spin = QDoubleSpinBox()
        self.ratio_spin.setRange(0.1, 1.0)
        self.ratio_spin.setSingleStep(0.05)
        self.ratio_spin.setValue(0.75)
        self.ratio_spin.setToolTip("Поріг для фільтру Lowe (менше = строгіше)")
        self.ratio_spin.valueChanged.connect(self.on_ratio_change)
        settings_layout.addRow("Lowe ratio:", self.ratio_spin)

        self.blur_check = QCheckBox("Gaussian blur")
        self.blur_check.stateChanged.connect(self.on_blur_change)
        self.blur_check.setToolTip("Згладити зображення перед пошуком")
        settings_layout.addRow("Обробка:", self.blur_check)

        self.blur_kernel_spin = QSpinBox()
        self.blur_kernel_spin.setRange(3, 21)
        self.blur_kernel_spin.setSingleStep(2)
        self.blur_kernel_spin.setValue(5)
        self.blur_kernel_spin.setToolTip("Розмір ядра фільтра (непарне число)")
        self.blur_kernel_spin.valueChanged.connect(self.on_blur_change)
        settings_layout.addRow("Kernel size:", self.blur_kernel_spin)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # === SECTION 4: Utilities (NEW UX) ===
        utils_group = QGroupBox("🛠️ Утиліти")
        utils_layout = QVBoxLayout()
        utils_layout.setSpacing(8)

        btn_demo = self._make_button("📺 Demo", self.run_demo, "Запустити демонстрацію зі вбудованими зображеннями")
        utils_layout.addWidget(btn_demo)

        btn_save_img = self._make_button("💾 Зберегти результат (PNG)", self.save_result_image, "Зберегти результат аналізу як PNG")
        utils_layout.addWidget(btn_save_img)

        btn_save_log = self._make_button("📝 Зберегти лог (TXT)", self.save_log_text, "Експортувати лог операцій")
        utils_layout.addWidget(btn_save_log)

        btn_help = self._make_button("❓ Як це працює?", self.show_help_dialog, "Довідка про алгоритми та обмеження")
        utils_layout.addWidget(btn_help)

        utils_group.setLayout(utils_layout)
        layout.addWidget(utils_group)

        # Stretch to fill space
        layout.addStretch()

        panel.setLayout(layout)
        return panel

    def _make_button(self, text: str, callback, tooltip: str = "") -> QPushButton:
        """Create styled button with tooltip."""
        btn = QPushButton(text)
        btn.clicked.connect(callback)
        if tooltip:
            btn.setToolTip(tooltip)
        return btn

    def _init_video_tab(self):
        """Tab 2: Video search with controls, timeline, and preview."""
        tab = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(12, 12, 12, 12)

        # ==================== LEFT PANEL: Controls ====================
        left_panel = self._build_video_controls()

        # ==================== CENTER: Results ====================
        center_layout = QVBoxLayout()
        center_layout.setSpacing(8)

        # Progress
        progress_group = QGroupBox("Прогрес")
        progress_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        progress_layout.addWidget(self.progress_bar)
        progress_group.setLayout(progress_layout)
        center_layout.addWidget(progress_group, 0)

        # Timeline: list of matched frames
        timeline_group = QGroupBox("📹 Кадри зі збігами")
        timeline_layout = QVBoxLayout()
        self.timeline = QListWidget()
        self.timeline.itemSelectionChanged.connect(self.on_timeline_select)
        self.timeline.setMaximumHeight(150)
        timeline_layout.addWidget(self.timeline)
        timeline_group.setLayout(timeline_layout)
        center_layout.addWidget(timeline_group, 0)

        # Video frame viewer
        video_group = QGroupBox("📽️ Поточний кадр")
        video_layout = QVBoxLayout()
        self.video_viewer = ImageViewer()
        self.video_viewer.setMinimumHeight(300)
        video_layout.addWidget(self.video_viewer)
        video_group.setLayout(video_layout)
        center_layout.addWidget(video_group, 1)

        # Details
        self.video_details_label = QLabel("Деталі: —")
        self.video_details_label.setWordWrap(True)
        self.video_details_label.setStyleSheet("background:#2b2d37; padding:8px; border-radius:4px;")
        center_layout.addWidget(self.video_details_label, 0)

        center_panel = QWidget()
        center_panel.setLayout(center_layout)

        # ==================== Assemble ====================
        main_layout.addWidget(left_panel, 0)
        main_layout.addWidget(center_panel, 1)
        tab.setLayout(main_layout)

        self.tabs.addTab(tab, "📹 Пошук у відео")

    def _build_video_controls(self) -> QWidget:
        """Build left control panel for video search."""
        panel = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(12)

        # === SECTION 1: Load Video ===
        load_group = QGroupBox("1️⃣ Завантажити")
        load_layout = QVBoxLayout()
        load_layout.setSpacing(8)

        btn_video = self._make_button("📽️ Завантажити відео", self.load_video, "MP4, AVI, MKV")
        btn_video.setMinimumHeight(40)
        load_layout.addWidget(btn_video)
        
        # Debug button: save first frame for comparison
        btn_debug = self._make_button("🔍 Зберегти перший кадр", self.save_first_frame, "Зберегти перший кадр відео для порівняння з еталоном")
        load_layout.addWidget(btn_debug)

        load_group.setLayout(load_layout)
        layout.addWidget(load_group)

        # === SECTION 2: Playback Control ===
        control_group = QGroupBox("2️⃣ Управління")
        control_layout = QVBoxLayout()
        control_layout.setSpacing(8)

        self.start_video_btn = self._make_button("▶️ Запустити пошук", self.start_video_search, "Почати пошук об'єкта у відео")
        self.start_video_btn.setMinimumHeight(48)
        self.start_video_btn.setStyleSheet(self.start_video_btn.styleSheet() + "\nfont-size: 11pt; font-weight: bold;")
        self.start_video_btn.setEnabled(False)
        control_layout.addWidget(self.start_video_btn)

        pause_layout = QHBoxLayout()
        btn_pause = self._make_button("⏸ Пауза", self.toggle_pause, "Зупинити/продовжити обробку")
        btn_step = self._make_button("⏭ Крок", self.step_timeline, "Перейти до наступного кадру")
        pause_layout.addWidget(btn_pause)
        pause_layout.addWidget(btn_step)
        control_layout.addLayout(pause_layout)

        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        # === SECTION 3: Parameters ===
        params_group = QGroupBox("⚙️ Параметри відео")
        params_layout = QFormLayout()
        params_layout.setSpacing(8)

        self.frame_skip_spin = QSpinBox()
        self.frame_skip_spin.setRange(1, 200)
        self.frame_skip_spin.setValue(10)
        self.frame_skip_spin.setToolTip("Перевіряти кожен N-й кадр (більше = швидше)")
        params_layout.addRow("Крок кадрів:", self.frame_skip_spin)

        self.sim_threshold_spin = QSpinBox()
        self.sim_threshold_spin.setRange(1, 100)
        self.sim_threshold_spin.setValue(20)
        self.sim_threshold_spin.setToolTip("Мінімальна схожість для запису (% від min(kp))")
        params_layout.addRow("Мін. схожість:", self.sim_threshold_spin)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # === SECTION 4: Algorithm (reuse from settings) ===
        algo_group = QGroupBox("⚙️ Алгоритм (як Image tab)")
        algo_layout = QFormLayout()
        algo_layout.setSpacing(8)

        self.feature_combo_video = QComboBox()
        for strat in self.strategies:
            self.feature_combo_video.addItem(strat.name)
        # Link to main matcher
        self.feature_combo_video.currentIndexChanged.connect(self.on_feature_change)
        algo_layout.addRow("Алгоритм:", self.feature_combo_video)

        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group)

        layout.addStretch()

        panel.setLayout(layout)
        return panel

    def _init_settings_tab(self):
        """Tab 3: Global settings and algorithm reference."""
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)

        # Title
        title_label = QLabel("⚙️ Глобальні налаштування")
        title_font = title_label.font()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)

        # === MATCHING ALGORITHMS ===
        algo_info = QGroupBox("🔍 Алгоритми ознак")
        algo_layout = QFormLayout()

        algo_desc = QLabel(
            "<b>ORB:</b> Fast, rotation invariant; binary descriptors.<br/>"
            "<b>AKAZE:</b> Similar to ORB; often more stable.<br/>"
            "<b>SIFT:</b> Slow but most accurate; requires OpenCV built with SIFT support."
        )
        algo_desc.setWordWrap(True)
        algo_layout.addRow("Опис:", algo_desc)

        algo_info.setLayout(algo_layout)
        layout.addWidget(algo_info)

        # === MATCHER INFO ===
        matcher_info = QGroupBox("🎯 Режими Matcher")
        matcher_layout = QFormLayout()

        matcher_desc = QLabel(
            "<b>KNN + Lowe ratio:</b> K-Nearest Neighbors + ratio test для фільтрації.<br/>"
            "<b>BF crossCheck:</b> Brute Force з взаємною відповідністю (більш строго)."
        )
        matcher_desc.setWordWrap(True)
        matcher_layout.addRow("Опис:", matcher_desc)

        matcher_info.setLayout(matcher_layout)
        layout.addWidget(matcher_info)

        # === METRICS INFO ===
        metrics_info = QGroupBox("📊 Метрики схожості")
        metrics_layout = QFormLayout()

        metrics_desc = QLabel(
            "<b>inliers / min(kp):</b> Основна метрика (частка потвердженого збігу).<br/>"
            "<b>good / min(kp):</b> Доля добрих збігів до фільтрації RANSAC.<br/>"
            "<b>Reproj error:</b> Середня помилка переспроєкції для інлайєрів.<br/>"
            "<b>Stability:</b> Міра стійкості гомографії (1/(1+error)).<br/>"
            "<b>Adaptive threshold:</b> Динамічний поріг для відео."
        )
        metrics_desc.setWordWrap(True)
        metrics_layout.addRow("Опис:", metrics_desc)

        metrics_info.setLayout(metrics_layout)
        layout.addWidget(metrics_info)

        # === TIPS ===
        tips_info = QGroupBox("💡 Поради")
        tips_layout = QVBoxLayout()

        tips_text = QLabel(
            "• Завтоаження чітких еталонів краще за нечіткі.\n"
            "• Більш обмежувальний Lowe ratio (0.5–0.6) зменшує помилки.\n"
            "• Gaussian blur допомагає при шумі, але сповільнює пошук.\n"
            "• Адаптивний поріг автоматично коригується за статистикою.\n"
            "• Покрок кадрів у відео регулює швидкість vs. точність."
        )
        tips_text.setWordWrap(True)
        tips_layout.addWidget(tips_text)

        tips_info.setLayout(tips_layout)
        layout.addWidget(tips_info)

        layout.addStretch()

        tab.setLayout(layout)
        self.tabs.addTab(tab, "⚙️ Навігація")

    # Event handlers
    def load_first_image(self):
        """Load first reference image."""
        path, _ = QFileDialog.getOpenFileName(self, "Обрати перше зображення", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            img = cv2.imread(path)
            if img is None:
                QMessageBox.warning(self, "Помилка", "Не вдалося завантажити зображення")
                return
            self.image1 = img
            self._log_success(f"✓ Перше зображення завантажено: {path.split('/')[-1]}")
            if self.image2 is not None:
                self.compare_btn.setEnabled(True)
            if self.video_path:
                self.start_video_btn.setEnabled(True)

    def load_second_image(self):
        """Load second image for comparison."""
        path, _ = QFileDialog.getOpenFileName(self, "Обрати друге зображення", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            img = cv2.imread(path)
            if img is None:
                QMessageBox.warning(self, "Помилка", "Не вдалося завантажити зображення")
                return
            self.image2 = img
            self._log_success(f"✓ Друге зображення завантажено: {path.split('/')[-1]}")
            if self.image1 is not None:
                self.compare_btn.setEnabled(True)

    def load_video(self):
        """Load video file."""
        path, _ = QFileDialog.getOpenFileName(self, "Обрати відео", "", "Video (*.mp4 *.avi *.mkv)")
        if path:
            self.video_path = path
            self._log_success(f"✓ Відео завантажено: {path.split('/')[-1]}")
            if self.image1 is not None:
                self.start_video_btn.setEnabled(True)

    def compare(self):
        """Compare two loaded images."""
        if self.image1 is None or self.image2 is None:
            self._log_error("✗ Завантажте два зображення для порівняння")
            QMessageBox.warning(self, "Помилка", "Завантажте два зображення")
            return
        self.compare_btn.setEnabled(False)
        self._set_status("Обробка...", "#d4b106")
        result = self.matcher.detect_and_match(self.image1, self.image2)
        self._display_match_result(result)
        self._set_status("Готово", "green")
        self.compare_btn.setEnabled(True)

    def show_heatmap(self):
        """Show heatmap in detail view."""
        self.tabs.setCurrentIndex(0)
        self.heatmap_viewer.setFocus()

    def start_video_search(self):
        """Start video search for matching frames."""
        if self.image1 is None:
            QMessageBox.warning(self, "Нема еталона", "Завантажте перше (еталонне) зображення на вкладці Image.")
            return
        if not self.video_path:
            QMessageBox.warning(self, "Нема відео", "Завантажте відео.")
            return
        if self.video_worker and self.video_worker.isRunning():
            self._log_warn("⚠ Пошук вже виконується")
            return

        self.video_processor.frame_skip = self.frame_skip_spin.value()
        self.video_processor.similarity_threshold = float(self.sim_threshold_spin.value())
        self.video_processor.stopped = False
        self.video_processor.paused = False
        self.video_matches.clear()
        self.timeline.clear()
        self.progress_bar.setValue(0)

        self._set_status("Обробка відео...", "#d4b106")
        self._log(f"⏱ Запуск пошуку: крок={self.frame_skip_spin.value()}, поріг={self.sim_threshold_spin.value()}%")

        self.video_worker = VideoWorker(self.video_processor, self.matcher, self.video_path, self.image1)
        self.video_worker.progress.connect(self.progress_bar.setValue)
        self.video_worker.match_found.connect(self._on_video_match_found)
        self.video_worker.finished.connect(self._on_video_finished)
        self.video_worker.error.connect(self._on_video_error)
        self.video_worker.start()

    def toggle_pause(self):
        """Toggle pause/resume for video processing."""
        if self.video_worker and self.video_worker.isRunning():
            self.video_processor.toggle_pause()
            state = "ПАУЗА" if self.video_processor.paused else "ПРОДОВЖЕНО"
            self._log_warn(f"⏸ {state}")

    def step_timeline(self):
        """Navigate to next matched frame in timeline."""
        current_row = self.timeline.currentRow()
        next_row = current_row + 1 if current_row >= 0 else 0
        if next_row < self.timeline.count():
            self.timeline.setCurrentRow(next_row)
        else:
            self._log_warn("Більше збігів немає")

    def on_timeline_select(self):
        """Display selected matched frame from timeline."""
        item = self.timeline.currentItem()
        if not item:
            return
        frame_idx = item.data(Qt.UserRole)
        for idx, result in self.video_matches:
            if idx == frame_idx:
                self.video_viewer.set_image(result.matched_image)
                self.video_details_label.setText(
                    f"📹 Кадр #{frame_idx} | {result.similarity:.1f}% | Inliers: {result.inliers} | Stability: {result.stability:.3f} | "
                    f"⏱ Час метчінгу: {result.match_time_ms:.1f} мс | "
                    f"Тіки: start={result.start_tick:.6f}, end={result.end_tick:.6f}, Δ={result.tick_diff:.6f}"
                )
                break

    def _on_video_match_found(self, frame_idx: int, result: MatchResult):
        """Callback when a match is found in video."""
        self.video_matches.append((frame_idx, result))
        item = QListWidgetItem(f"Кадр #{frame_idx} | {result.similarity:.1f}% | ⏱ {result.match_time_ms:.1f} мс")
        item.setData(Qt.UserRole, frame_idx)
        self.timeline.addItem(item)
        self.video_viewer.set_image(result.matched_image)
        self.video_details_label.setText(
            f"📹 Кадр #{frame_idx} | {result.similarity:.1f}% | "
            f"Inliers: {result.inliers} | Stability: {result.stability:.3f} | "
            f"⏱ Час метчінгу: {result.match_time_ms:.1f} мс | "
            f"Тіки: start={result.start_tick:.6f}, end={result.end_tick:.6f}, Δ={result.tick_diff:.6f}"
        )
        self._log_success(f"✓ Кадр #{frame_idx}: {result.similarity:.1f}% (адаптивний {result.adaptive_threshold:.1f}%, час={result.match_time_ms:.1f} мс)")

    def _on_video_finished(self):
        """Callback when video processing finishes."""
        count = len(self.video_matches)
        self._set_status("✓ Готово", "green")
        
        if count == 0:
            # No frames found - provide helpful suggestions
            self._log_error(
                "✗ Пошук завершено. Знайдено 0 кадрів.\n"
                "💡 Рекомендації:\n"
                "  • Зменшіть крок кадрів (наприклад, на 1-3)\n"
                "  • Зменшіть мінімальну схожість (5-10%)\n"
                "  • Спробуйте SIFT замість ORB/AKAZE\n"
                "  • Переконайтесь, що еталон містить текстуру\n"
                "  • Впімкніть Gaussian blur на вкладці Image"
            )
            QMessageBox.information(
                self, "Об'єкт не знайдено",
                "У відео не знайдено жодного збігу.\n\n"
                "Спробуйте:\n"
                "1. Зменшити крок кадрів (більше кадрів для обробки)\n"
                "2. Зменшити поріг схожості\n"
                "3. Використати SIFT алгоритм\n"
                "4. Впімкніти фільтр Gaussian Blur"
            )
        else:
            self._log_success(f"✓ Пошук завершено. Знайдено {count} кадрів")

    def _on_video_error(self, message: str):
        """Callback for video processing error."""
        self._set_status("✗ Помилка", "#ff4d4f")
        self._log_error(f"✗ Помилка: {message}")
        QMessageBox.critical(self, "Помилка відео", message)

    def _on_tab_changed(self, idx: int):
        """Callback when tab changes."""
        self.current_tab = idx
        tab_names = ["Image", "Video", "Settings"]
        self._log(f"Переключення на вкладку: {tab_names[idx]}")

    def closeEvent(self, event):  # noqa: N802
        """Handle window close event."""
        if self.video_worker and self.video_worker.isRunning():
            self.video_processor.stop()
            self.video_worker.wait(500)
        super().closeEvent(event)

    # Settings handlers
    def on_feature_change(self, idx: int):
        """Change feature detection algorithm."""
        if 0 <= idx < len(self.strategies):
            self.matcher.set_strategy(self.strategies[idx])
            # sync combo on video tab
            self.feature_combo_video.blockSignals(True)
            self.feature_combo_video.setCurrentIndex(idx)
            self.feature_combo_video.blockSignals(False)
            self._log(f"🔍 Алгоритм ознак: {self.strategies[idx].name}")

    def on_matcher_mode_change(self, idx: int):
        """Change matcher mode."""
        use_knn = idx == 0
        cross_check = not use_knn
        self.matcher.set_matcher_mode(use_knn=use_knn, ratio_thresh=self.ratio_spin.value(), cross_check=cross_check)
        mode_text = "KNN + Lowe ratio" if use_knn else "BF crossCheck"
        self._log(f"🎯 Matcher режим: {mode_text}")

    def on_ratio_change(self, value: float):
        """Update Lowe ratio threshold."""
        self.matcher.set_matcher_mode(use_knn=self.matcher.use_knn, ratio_thresh=value, cross_check=self.matcher.cross_check)
        self._log(f"📊 Lowe ratio: {value:.2f}")

    def on_blur_change(self):
        """Update blur preprocessing."""
        self.matcher.set_preprocessing(self.blur_check.isChecked(), self.blur_kernel_spin.value())
        state = "вм." if self.blur_check.isChecked() else "вим."
        self._log(f"🔧 Gaussian blur: {state} (kernel={self.blur_kernel_spin.value()})")

    def _display_match_result(self, result: MatchResult, title: str = "Результат"):
        """Display image matching result with metrics, verdict, and explanation."""
        self.last_result = result
        self._update_verdict(result)

        if result.matched_image is None:
            self.compare_viewer.clear()
            self.heatmap_viewer.clear()
            self.similarity_label.setText("❌ Недостатньо збігів для гомографії")
            self.details_label.setText("—")
            self._log_warn(result.explanation)
            return

        # Show results
        self.compare_viewer.set_image(result.matched_image)
        if result.heatmap is not None:
            self.heatmap_viewer.set_image(result.heatmap)

        # Main similarity metric
        similarity_color = "green" if result.similarity >= result.adaptive_threshold else "#ff9800"
        sim_style = f"font-weight: bold; font-size: 12pt; color:{similarity_color};"
        self.similarity_label.setStyleSheet(sim_style)
        self.similarity_label.setText(
            f"✓ Схожість (inliers/min): {result.similarity:.1f}% | Inliers: {result.inliers} | Good: {result.good_matches} | "
            f"⏱ Час метчінгу: {result.match_time_ms:.1f} мс | "
            f"Тіки: start={result.start_tick:.6f}, end={result.end_tick:.6f}, Δ={result.tick_diff:.6f}"
        )

        # Details
        detail_text = (
            f"Метод: <b>{result.method}</b> | "
            f"Good/min: {result.similarity_good:.1f}% | "
            f"Reproj error: {result.reprojection_error:.4f} | "
            f"Stability: {result.stability:.3f} | "
            f"Adaptive: {result.adaptive_threshold:.1f}%<br>"
            f"Тіки: start={result.start_tick:.6f}, end={result.end_tick:.6f}, Δ={result.tick_diff:.6f}"
        )
        self.details_label.setText(detail_text)
        self.setWindowTitle(title)

        self._log_success(
            f"✓ Збіг знайдено: {result.similarity:.1f}% (inliers={result.inliers}, "
            f"good={result.good_matches}, stability={result.stability:.3f}, "
            f"час={result.match_time_ms:.1f} мс)"
        )

    def _update_verdict(self, result: MatchResult):
        """Update verdict and explanation labels based on result."""
        if result.verdict.startswith("🟢"):
            color = "#4CAF50"
        elif result.verdict.startswith("🟡"):
            color = "#FFA500"
        else:
            color = "#FF4444"
        self.verdict_label.setText(result.verdict)
        self.verdict_label.setStyleSheet(f"font-weight: bold; font-size: 13pt; color: {color};")
        self.explanation_label.setText(result.explanation)
        self.explanation_label.setStyleSheet("font-size: 10pt; color: #CCCCCC;")

    def _set_status(self, text: str, color: str):
        """Update status indicator."""
        self.status_indicator.setText(text)
        self.status_indicator.setStyleSheet(self._status_color(color))

    def _apply_dark_theme(self):
        """Apply dark theme stylesheet."""
        palette = """
        QWidget {
            background-color: #1e1f29;
            color: #f5f5f5;
        }
        QPushButton {
            background-color: #2d2f3a;
            border: 1px solid #3a3d4d;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 10pt;
            color: #f5f5f5;
        }
        QPushButton:hover {
            background-color: #3a3d4d;
        }
        QPushButton:pressed {
            background-color: #2d2f3a;
        }
        QPushButton:disabled {
            background-color: #1a1c24;
            color: #777;
            border-color: #1a1c24;
        }
        QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget {
            background-color: #2b2d37;
            border: 1px solid #3a3d4d;
            color: #f5f5f5;
            padding: 4px;
            border-radius: 3px;
        }
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
            border: 2px solid #5b8def;
        }
        QTabBar::tab {
            background: #2d2f3a;
            border: 1px solid #3a3d4d;
            padding: 8px 20px;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background: #3a3d4d;
            border-bottom: 2px solid #5b8def;
        }
        QTabWidget::pane {
            border: 1px solid #3a3d4d;
        }
        QProgressBar {
            border: 1px solid #3a3d4d;
            border-radius: 4px;
            text-align: center;
            background-color: #2b2d37;
        }
        QProgressBar::chunk {
            background-color: #5b8def;
            border-radius: 3px;
        }
        QGroupBox {
            border: 1px solid #3a3d4d;
            border-radius: 4px;
            margin-top: 10px;
            padding-top: 10px;
            color: #f5f5f5;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px 0 3px;
        }
        QCheckBox {
            spacing: 6px;
        }
        QCheckBox::indicator {
            width: 16px;
            height: 16px;
            background-color: #2b2d37;
            border: 1px solid #3a3d4d;
            border-radius: 2px;
        }
        QCheckBox::indicator:checked {
            background-color: #5b8def;
        }
        QLabel {
            color: #f5f5f5;
        }
        """
        self.setStyleSheet(palette)

    def _log(self, text: str):
        """Log plain message."""
        self.log_widget.appendPlainText(text)
        self.log_widget.moveCursor(self.log_widget.textCursor().End)

    def _log_success(self, text: str):
        """Log success message (green)."""
        self._log(text)

    def _log_warn(self, text: str):
        """Log warning message (yellow)."""
        self._log(text)

    def _log_error(self, text: str):
        """Log error message (red)."""
        self._log(text)

    # ==================== NEW: Demo, Save, Help functionality ====================

    def run_demo(self):
        """Run demo with built-in test images (generates simple test patterns)."""
        self._log("▶️ Запуск Demo режиму...")
        
        # Generate simple test images: original + rotated/scaled version
        size = (400, 300)
        # Image 1: Rectangle with text-like pattern
        img1 = np.ones((size[1], size[0], 3), dtype=np.uint8) * 50
        cv2.rectangle(img1, (50, 50), (200, 150), (100, 200, 100), 2)
        cv2.rectangle(img1, (60, 60), (190, 140), (150, 250, 150), -1)
        cv2.circle(img1, (100, 100), 30, (0, 255, 255), 2)
        cv2.putText(img1, "Demo", (120, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Image 2: Same object slightly rotated and scaled
        img2 = img1.copy()
        M = cv2.getRotationMatrix2D((size[0]//2, size[1]//2), 15, 1.1)
        img2 = cv2.warpAffine(img2, M, (size[0], size[1]))
        
        self.image1 = img1
        self.image2 = img2
        self._log_success("✓ Demo зображення завантажено (синтетичні тестові зображення)")
        
        # Automatically run comparison
        self.compare()
        self.show_heatmap()

    def save_result_image(self):
        """Save matching result as PNG file."""
        if self.last_result is None or self.last_result.matched_image is None:
            QMessageBox.information(self, "Немає результату", "Спершу виконайте порівняння.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Зберегти результат", "", "PNG (*.png)")
        if path:
            cv2.imwrite(path, self.last_result.matched_image)
            self._log_success(f"💾 Збережено результат: {path}")

    def save_log_text(self):
        """Export log as TXT file."""
        path, _ = QFileDialog.getSaveFileName(self, "Зберегти лог", "", "Text (*.txt)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.log_widget.toPlainText())
            self._log_success(f"📝 Лог збережено: {path}")

    def show_help_dialog(self):
        """Show help dialog explaining algorithms and limitations."""
        msg = (
            "Як це працює:\n"
            "1) ORB/AKAZE/SIFT → ключові точки та дескриптори\n"
            "2) BFMatcher: KNN + Lowe ratio або crossCheck\n"
            "3) RANSAC findHomography: відсіює викиди, дає інлайєри\n"
            "4) Якщо інлайєрів ≥ 8 — будуємо гомографію, малюємо рамку\n"
            "\nОбмеження: слабка текстура, мультфільми, сильна оклюзія/розмиття."
        )
        QMessageBox.information(self, "Як це працює?", msg)
    
    def save_first_frame(self):
        """Save first frame of video for diagnostic comparison with reference image."""
        if not self.video_path:
            QMessageBox.warning(self, "Помилка", "Спочатку завантажте відео")
            return
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            QMessageBox.critical(self, "Помилка", "Не вдалося відкрити відео")
            return
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            QMessageBox.critical(self, "Помилка", "Не вдалося прочитати кадр з відео")
            return
        
        path, _ = QFileDialog.getSaveFileName(self, "Зберегти перший кадр", "first_frame.jpg", "Images (*.jpg *.png)")
        if not path:
            return
        
        cv2.imwrite(path, frame)
        self._log_success(f"Перший кадр збережено: {path}")
        
        # Also show comparison info
        if self.img1 is not None:
            ref_size = f"{self.img1.shape[1]}x{self.img1.shape[0]}"
            frame_size = f"{frame.shape[1]}x{frame.shape[0]}"
            info = f"Порівняння розмірів:\n\nЕталон: {ref_size}\nКадр відео: {frame_size}\n\n"
            info += "Рекомендація: еталон і кадри повинні бути схожого розміру та якості"
            QMessageBox.information(self, "Діагностика", info)


class QDoubleSpinBoxCompat(QDoubleSpinBox):
    """Custom QDoubleSpinBox with cleaner integration."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setRange(0.1, 1.0)
        self.setSingleStep(0.05)
        self.setDecimals(2)
