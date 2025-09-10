import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGridLayout, QTextEdit, QLineEdit,
                             QPushButton, QLabel, QSpinBox, QFileDialog, 
                             QMessageBox, QTabWidget, QSplitter, QGroupBox,
                             QComboBox, QDoubleSpinBox, QCheckBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QTextCursor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import random

class MatrixInputWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.matrix_size = 3
        self.matrix_inputs = []
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Размер матрицы
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Размер матрицы:"))
        self.size_spinbox = QSpinBox()
        self.size_spinbox.setMinimum(2)
        self.size_spinbox.setMaximum(50)
        self.size_spinbox.setValue(3)
        self.size_spinbox.valueChanged.connect(self.change_matrix_size)
        size_layout.addWidget(self.size_spinbox)
        size_layout.addStretch()
        layout.addLayout(size_layout)
        
        # Поле для ввода матрицы
        self.matrix_widget = QWidget()
        self.matrix_layout = QGridLayout()
        self.matrix_widget.setLayout(self.matrix_layout)
        layout.addWidget(self.matrix_widget)
        
        self.create_matrix_inputs()
        self.setLayout(layout)
    
    def create_matrix_inputs(self):
        # Очищаем предыдущие поля
        for i in reversed(range(self.matrix_layout.count())): 
            self.matrix_layout.itemAt(i).widget().setParent(None)
        
        self.matrix_inputs = []
        for i in range(self.matrix_size):
            row = []
            for j in range(self.matrix_size + 1):  # +1 для вектора b
                input_field = QLineEdit()
                input_field.setMaximumWidth(60)
                if j < self.matrix_size:
                    input_field.setPlaceholderText(f"a{i+1}{j+1}")
                else:
                    input_field.setPlaceholderText(f"b{i+1}")
                row.append(input_field)
                self.matrix_layout.addWidget(input_field, i, j)
            self.matrix_inputs.append(row)
    
    def change_matrix_size(self, size):
        self.matrix_size = size
        self.create_matrix_inputs()
    
    def get_matrix(self):
        try:
            A = []
            b = []
            for i in range(self.matrix_size):
                row = []
                for j in range(self.matrix_size + 1):
                    text = self.matrix_inputs[i][j].text().strip()
                    if not text:
                        raise ValueError(f"Пустое поле в позиции ({i+1}, {j+1})")
                    value = float(text)
                    if j < self.matrix_size:
                        row.append(value)
                    else:
                        b.append(value)
                A.append(row)
            return np.array(A), np.array(b)
        except ValueError as e:
            raise ValueError(f"Ошибка ввода данных: {str(e)}")
    
    def set_matrix(self, A, b):
        """Заполняет поля ввода матрицей A и вектором b"""
        n = len(A)
        self.size_spinbox.setValue(n)
        self.create_matrix_inputs()
        
        for i in range(n):
            for j in range(n):
                self.matrix_inputs[i][j].setText(f"{A[i, j]:.6f}")
            self.matrix_inputs[i][n].setText(f"{b[i]:.6f}")

class MatrixDisplayWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFont("Courier", 10))
        
        layout.addWidget(QLabel("Результаты:"))
        layout.addWidget(self.text_edit)
        self.setLayout(layout)
    
    def display_results(self, A, b, solution, iterations, residuals_history, 
                       convergence_info, method_name="Сопряженных градиентов"):
        text = f"=== МЕТОД {method_name.upper()} ===\n\n"
        
        text += "Исходная матрица A:\n"
        text += self.matrix_to_str(A) + "\n"
        
        text += "Вектор b:\n"
        text += " ".join([f"{x:8.4f}" for x in b]) + "\n\n"
        
        text += f"Количество итераций: {iterations}\n"
        text += f"Сходимость: {convergence_info}\n\n"
        
        text += "Решение системы:\n"
        for i, x in enumerate(solution):
            text += f"x{i+1} = {x:.12f}\n"
        
        if residuals_history is not None and len(residuals_history) > 0:
            text += f"\nФинальная невязка: {residuals_history[-1]:.2e}\n"
            
            text += "\nИстория невязок:\n"
            if len(residuals_history) <= 20:
                for i, residual in enumerate(residuals_history):
                    text += f"Итерация {i+1}: {residual:.2e}\n"
            else:
                for i in range(10):
                    text += f"Итерация {i+1}: {residuals_history[i]:.2e}\n"
                text += "...\n"
                for i in range(len(residuals_history)-10, len(residuals_history)):
                    text += f"Итерация {i+1}: {residuals_history[i]:.2e}\n"
        
        self.text_edit.setPlainText(text)
    
    def matrix_to_str(self, matrix):
        text = ""
        for row in matrix:
            text += " ".join([f"{x:8.4f}" for x in row]) + "\n"
        return text

class PlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        
        layout.addWidget(QLabel("Графики сходимости:"))
        layout.addWidget(self.canvas)
        self.setLayout(layout)
    
    def plot_convergence(self, residuals_history, energy_norm_history=None):
        self.figure.clear()
        
        if energy_norm_history:
            # Два подграфика
            ax1 = self.figure.add_subplot(211)
            ax2 = self.figure.add_subplot(212)
            
            # График невязки
            ax1.semilogy(range(1, len(residuals_history) + 1), residuals_history, 
                        'b-', linewidth=2, marker='o', markersize=4)
            ax1.set_xlabel('Итерация')
            ax1.set_ylabel('Невязка (log scale)')
            ax1.set_title('Сходимость метода: Невязка')
            ax1.grid(True, alpha=0.3)
            
            # График энергетической нормы
            ax2.semilogy(range(1, len(energy_norm_history) + 1), energy_norm_history,
                        'r-', linewidth=2, marker='s', markersize=4)
            ax2.set_xlabel('Итерация')
            ax2.set_ylabel('Энергетическая норма ошибки (log scale)')
            ax2.set_title('Сходимость метода: Энергетическая норма')
            ax2.grid(True, alpha=0.3)
            
        else:
            # Один график невязки
            ax = self.figure.add_subplot(111)
            ax.semilogy(range(1, len(residuals_history) + 1), residuals_history,
                       'b-', linewidth=2, marker='o', markersize=4)
            ax.set_xlabel('Итерация')
            ax.set_ylabel('Невязка (log scale)')
            ax.set_title('Сходимость метода сопряженных градиентов')
            ax.grid(True, alpha=0.3)
        
        self.figure.tight_layout()
        self.canvas.draw()

class MatrixGeneratorDialog(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent, Qt.Window)
        self.parent = parent
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Генератор матриц")
        self.setGeometry(200, 200, 400, 400)
        
        layout = QVBoxLayout()
        
        # Размер матрицы
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Размер матрицы:"))
        self.size_spinbox = QSpinBox()
        self.size_spinbox.setMinimum(2)
        self.size_spinbox.setMaximum(50)
        self.size_spinbox.setValue(10)
        size_layout.addWidget(self.size_spinbox)
        layout.addLayout(size_layout)
        
        # Тип матрицы
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Тип матрицы:"))
        self.matrix_type = QComboBox()
        self.matrix_type.addItems(["Симметричная положительно определенная", 
                                  "Диагональная", "Трёхдиагональная", 
                                  "Ленточная", "Случайная SPD"])
        type_layout.addWidget(self.matrix_type)
        layout.addLayout(type_layout)
        
        # Диапазон значений
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Диапазон значений:"))
        self.min_val = QDoubleSpinBox()
        self.min_val.setRange(0.1, 1000)
        self.min_val.setValue(1)
        self.max_val = QDoubleSpinBox()
        self.max_val.setRange(0.1, 1000)
        self.max_val.setValue(10)
        range_layout.addWidget(QLabel("от"))
        range_layout.addWidget(self.min_val)
        range_layout.addWidget(QLabel("до"))
        range_layout.addWidget(self.max_val)
        layout.addLayout(range_layout)
        
        # Параметры для ленточных матриц
        band_layout = QHBoxLayout()
        band_layout.addWidget(QLabel("Ширина ленты:"))
        self.bandwidth = QSpinBox()
        self.bandwidth.setRange(1, 20)
        self.bandwidth.setValue(3)
        band_layout.addWidget(self.bandwidth)
        layout.addLayout(band_layout)
        
        # Число обусловленности
        cond_layout = QHBoxLayout()
        cond_layout.addWidget(QLabel("Число обусловленности:"))
        self.condition_number = QDoubleSpinBox()
        self.condition_number.setRange(1, 10000)
        self.condition_number.setValue(100)
        cond_layout.addWidget(self.condition_number)
        layout.addLayout(cond_layout)
        
        # Кнопки
        button_layout = QHBoxLayout()
        self.generate_btn = QPushButton("Сгенерировать")
        self.generate_btn.clicked.connect(self.generate_matrix)
        self.cancel_btn = QPushButton("Отмена")
        self.cancel_btn.clicked.connect(self.close)
        
        button_layout.addWidget(self.generate_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def generate_matrix(self):
        try:
            n = self.size_spinbox.value()
            min_val = self.min_val.value()
            max_val = self.max_val.value()
            matrix_type = self.matrix_type.currentText()
            bandwidth = self.bandwidth.value()
            cond_number = self.condition_number.value()
            
            A = self.generate_matrix_type(n, min_val, max_val, matrix_type, bandwidth, cond_number)
            
            # Генерируем вектор b
            x_exact = np.random.uniform(-10, 10, n)
            b = A @ x_exact
            
            if self.parent:
                self.parent.matrix_input.set_matrix(A, b)
            
            self.close()
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))
    
    def generate_matrix_type(self, n, min_val, max_val, matrix_type, bandwidth=3, cond_number=100):
        if matrix_type == "Симметричная положительно определенная":
            # Генерируем SPD матрицу через собственные значения
            A = np.random.uniform(min_val, max_val, (n, n))
            A = (A + A.T) / 2  # Делаем симметричной
            
            # Устанавливаем заданное число обусловленности
            eigenvalues = np.linspace(1, cond_number, n)
            D = np.diag(eigenvalues)
            Q, _ = np.linalg.qr(A)  # Случайная ортогональная матрица
            return Q @ D @ Q.T
            
        elif matrix_type == "Диагональная":
            # Диагональная матрица с положительными элементами
            A = np.diag(np.random.uniform(min_val, max_val, n))
            return A
            
        elif matrix_type == "Трёхдиагональная":
            # Трёхдиагональная SPD матрица
            A = np.zeros((n, n))
            for i in range(n):
                if i > 0:
                    A[i, i-1] = random.uniform(0, min_val/2)  # Субдиагональ
                A[i, i] = random.uniform(min_val, max_val)    # Главная диагональ
                if i < n-1:
                    A[i, i+1] = A[i+1, i]  # Симметрия
            return A
            
        elif matrix_type == "Ленточная":
            # Ленточная SPD матрица
            A = np.zeros((n, n))
            for i in range(n):
                for j in range(max(0, i-bandwidth+1), min(n, i+bandwidth)):
                    if i == j:
                        A[i, j] = random.uniform(min_val, max_val)
                    else:
                        A[i, j] = random.uniform(0, min_val/2)
                        A[j, i] = A[i, j]  # Симметрия
            return A
            
        elif matrix_type == "Случайная SPD":
            # Случайная SPD матрица
            A = np.random.uniform(min_val, max_val, (n, n))
            A = (A + A.T) / 2
            # Делаем положительно определенной
            A += n * np.eye(n) * max_val
            return A
        
        return np.eye(n)

class ConjugateGradientSolver:
    @staticmethod
    def solve(A, b, tolerance=1e-8, max_iterations=1000, preconditioner=None):
        """
        Решение системы Ax = b методом сопряженных градиентов
        """
        n = len(b)
        x = np.zeros(n)  # Начальное приближение
        r = b - A @ x    # Вектор невязки
        p = r.copy()     # Направление поиска
        
        residuals_history = []
        energy_norm_history = []
        
        # Предобуславливание (если используется)
        if preconditioner is not None:
            z = preconditioner @ r
        else:
            z = r.copy()
        
        rz_old = np.dot(r, z)
        
        for iteration in range(max_iterations):
            Ap = A @ p
            alpha = rz_old / np.dot(p, Ap)
            
            x = x + alpha * p
            r = r - alpha * Ap
            
            # Сохраняем невязку
            residual = np.linalg.norm(r)
            residuals_history.append(residual)
            
            # Сохраняем энергетическую норму ошибки
            if iteration > 0:
                energy_norm = np.sqrt(np.abs(alpha * rz_old))
                energy_norm_history.append(energy_norm)
            
            # Проверка условия остановки
            if residual < tolerance:
                break
            
            # Предобуславливание
            if preconditioner is not None:
                z = preconditioner @ r
            else:
                z = r.copy()
            
            rz_new = np.dot(r, z)
            beta = rz_new / rz_old
            
            p = z + beta * p
            rz_old = rz_new
        
        convergence_info = "Сходимость достигнута" if residual < tolerance else "Максимум итераций"
        
        return x, iteration + 1, residuals_history, energy_norm_history, convergence_info
    
    @staticmethod
    def check_matrix_properties(A):
        """
        Проверка свойств матрицы для метода сопряженных градиентов
        """
        n = len(A)
        
        # 1. Проверка симметричности
        symmetric = np.allclose(A, A.T)
        
        # 2. Проверка положительной определенности
        positive_definite = True
        try:
            # Попытка разложения Холецкого
            np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            positive_definite = False
        
        # 3. Число обусловленности
        try:
            cond_number = np.linalg.cond(A)
        except:
            cond_number = float('inf')
        
        # 4. Спектральный радиус
        eigenvalues = np.linalg.eigvals(A)
        spectral_radius = np.max(np.abs(eigenvalues))
        
        return {
            'symmetric': symmetric,
            'positive_definite': positive_definite,
            'condition_number': cond_number,
            'spectral_radius': spectral_radius,
            'suitable_for_cg': symmetric and positive_definite
        }
    
    @staticmethod
    def create_preconditioner(A, type='jacobi'):
        """
        Создание предобуславливателя
        """
        if type == 'jacobi':
            # Диагональный предобуславливатель (Якоби)
            D_inv = np.diag(1.0 / np.diag(A))
            return D_inv
        elif type == 'none':
            return None
        else:
            return None

class CGCalculator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Калькулятор метода сопряженных градиентов")
        self.setGeometry(100, 100, 1400, 900)
        
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Левая панель - ввод данных
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        
        input_group = QGroupBox("Ввод матрицы")
        input_layout = QVBoxLayout()
        self.matrix_input = MatrixInputWidget()
        input_layout.addWidget(self.matrix_input)
        input_group.setLayout(input_layout)
        left_layout.addWidget(input_group)
        
        # Параметры метода
        params_group = QGroupBox("Параметры метода")
        params_layout = QGridLayout()
        
        params_layout.addWidget(QLabel("Точность:"), 0, 0)
        self.tolerance = QLineEdit("1e-8")
        self.tolerance.setMaximumWidth(100)
        params_layout.addWidget(self.tolerance, 0, 1)
        
        params_layout.addWidget(QLabel("Макс. итераций:"), 1, 0)
        self.max_iter = QSpinBox()
        self.max_iter.setRange(10, 100000)
        self.max_iter.setValue(1000)
        self.max_iter.setMaximumWidth(100)
        params_layout.addWidget(self.max_iter, 1, 1)
        
        params_layout.addWidget(QLabel("Предобуславливатель:"), 2, 0)
        self.preconditioner = QComboBox()
        self.preconditioner.addItems(["Без предобуславливания", "Диагональный (Якоби)"])
        self.preconditioner.setMaximumWidth(150)
        params_layout.addWidget(self.preconditioner, 2, 1)
        
        params_group.setLayout(params_layout)
        left_layout.addWidget(params_group)
        
        # Кнопки действий
        button_layout = QGridLayout()
        
        self.solve_btn = QPushButton("Решить")
        self.solve_btn.clicked.connect(self.solve)
        button_layout.addWidget(self.solve_btn, 0, 0)
        
        self.check_btn = QPushButton("Проверить матрицу")
        self.check_btn.clicked.connect(self.check_matrix)
        button_layout.addWidget(self.check_btn, 0, 1)
        
        self.clear_btn = QPushButton("Очистить")
        self.clear_btn.clicked.connect(self.clear)
        button_layout.addWidget(self.clear_btn, 1, 0)
        
        self.generate_btn = QPushButton("Сгенерировать")
        self.generate_btn.clicked.connect(self.show_generator)
        button_layout.addWidget(self.generate_btn, 1, 1)
        
        self.import_btn = QPushButton("Импорт матрицы")
        self.import_btn.clicked.connect(self.import_matrix)
        button_layout.addWidget(self.import_btn, 2, 0)
        
        self.export_matrix_btn = QPushButton("Экспорт матрицы")
        self.export_matrix_btn.clicked.connect(self.export_matrix)
        button_layout.addWidget(self.export_matrix_btn, 2, 1)
        
        self.export_results_btn = QPushButton("Экспорт результатов")
        self.export_results_btn.clicked.connect(self.export_results)
        button_layout.addWidget(self.export_results_btn, 3, 0, 1, 2)
        
        left_layout.addLayout(button_layout)
        left_panel.setLayout(left_layout)
        
        # Правая панель - результаты и графики
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        
        # Вкладки для результатов и графиков
        self.tabs = QTabWidget()
        
        # Вкладка с результатами
        self.results_display = MatrixDisplayWidget()
        self.tabs.addTab(self.results_display, "Результаты")
        
        # Вкладка с графиками
        self.plot_widget = PlotWidget()
        self.tabs.addTab(self.plot_widget, "Графики сходимости")
        
        right_layout.addWidget(self.tabs)
        right_panel.setLayout(right_layout)
        
        # Разделитель
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 1000])
        
        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        self.generator_dialog = None
        self.current_energy_norms = None
    
    def show_generator(self):
        if self.generator_dialog is None:
            self.generator_dialog = MatrixGeneratorDialog(self)
        self.generator_dialog.show()
    
    def solve(self):
        try:
            A, b = self.matrix_input.get_matrix()
            tolerance = float(self.tolerance.text())
            max_iterations = self.max_iter.value()
            
            # Выбор предобуславливателя
            preconditioner_type = self.preconditioner.currentText()
            if preconditioner_type == "Диагональный (Якоби)":
                preconditioner = ConjugateGradientSolver.create_preconditioner(A, 'jacobi')
            else:
                preconditioner = None
            
            solver = ConjugateGradientSolver()
            solution, iterations, residuals_history, energy_norm_history, convergence_info = solver.solve(
                A, b, tolerance, max_iterations, preconditioner
            )
            
            self.current_energy_norms = energy_norm_history
            
            if iterations >= max_iterations:
                QMessageBox.warning(self, "Предупреждение", 
                                  f"Достигнуто максимальное количество итераций ({max_iterations}).")
            
            self.results_display.display_results(
                A, b, solution, iterations, residuals_history, convergence_info, "Сопряженных градиентов"
            )
            
            # Построение графиков
            self.plot_widget.plot_convergence(residuals_history, energy_norm_history)
            self.tabs.setCurrentIndex(1)  # Переключаемся на вкладку с графиками
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))
    
    def check_matrix(self):
        try:
            A, b = self.matrix_input.get_matrix()
            solver = ConjugateGradientSolver()
            properties = solver.check_matrix_properties(A)
            
            text = "=== АНАЛИЗ МАТРИЦЫ ДЛЯ МЕТОДА СОПРЯЖЕННЫХ ГРАДИЕНТОВ ===\n\n"
            
            text += "Симметричность: "
            text += "Да" if properties['symmetric'] else "Нет"
            text += "\n"
            
            text += "Положительная определенность: "
            text += "Да" if properties['positive_definite'] else "Нет"
            text += "\n"
            
            text += f"Число обусловленности: {properties['condition_number']:.2e}\n"
            text += f"Спектральный радиус: {properties['spectral_radius']:.6f}\n\n"
            
            text += "Пригодность для метода сопряженных градиентов: "
            if properties['suitable_for_cg']:
                text += "✓ ОТЛИЧНО - матрица подходит\n"
                text += "Метод гарантированно сойдется за n шатков (в точной арифметике)\n"
            else:
                text += "⚠ ПРЕДУПРЕЖДЕНИЕ - матрица не подходит\n"
                if not properties['symmetric']:
                    text += "- Матрица не симметрична\n"
                if not properties['positive_definite']:
                    text += "- Матрица не положительно определена\n"
                text += "Метод может не сходиться или давать неверные результаты\n"
            
            text += "\nРекомендации:\n"
            if properties['condition_number'] > 1000:
                text += "- Число обусловленности велико, рассмотрите предобуславливание\n"
            if not properties['symmetric']:
                text += "- Для несимметричных матриц используйте другие методы (GMRES, BiCGSTAB)\n"
            
            self.results_display.text_edit.setPlainText(text)
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))
    
    def clear(self):
        for row in self.matrix_input.matrix_inputs:
            for input_field in row:
                input_field.clear()
        self.results_display.text_edit.clear()
        self.tolerance.setText("1e-8")
        self.max_iter.setValue(1000)
        self.current_energy_norms = None
        
        # Очищаем графики
        self.plot_widget.figure.clear()
        self.plot_widget.canvas.draw()
    
    def import_matrix(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Открыть матрицу", "", "Текстовые файлы (*.txt);;Все файлы (*)"
            )
            
            if file_path:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                # Поддерживаем два формата: построчный и матричный
                if '[' in content and ']' in content:
                    lines = content.split('\n')
                    matrix_data = []
                    for line in lines:
                        if '[' in line and ']' in line:
                            numbers = line.replace('[', '').replace(']', '').split()
                            row = [float(x) for x in numbers if x.strip()]
                            matrix_data.append(row)
                else:
                    lines = content.split('\n')
                    matrix_data = []
                    for line in lines:
                        if line.strip():
                            row = [float(x) for x in line.split()]
                            matrix_data.append(row)
                
                if not matrix_data:
                    raise ValueError("Файл пуст или содержит неверные данные")
                
                n = len(matrix_data)
                if any(len(row) != n + 1 for row in matrix_data):
                    raise ValueError("Неверный формат матрицы. Ожидается n x (n+1)")
                
                self.matrix_input.size_spinbox.setValue(n)
                self.matrix_input.create_matrix_inputs()
                
                for i in range(n):
                    for j in range(n + 1):
                        self.matrix_input.matrix_inputs[i][j].setText(f"{matrix_data[i][j]:.6f}")
                
                QMessageBox.information(self, "Успех", "Матрица успешно загружена!")
                
        except Exception as e:
            QMessageBox.critical(self, "Ошибка импорта", f"Ошибка при импорте: {str(e)}")
    
    def export_matrix(self):
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Сохранить матрицу", "matrix.txt", "Текстовые файлы (*.txt);;Все файлы (*)"
            )
            
            if file_path:
                A, b = self.matrix_input.get_matrix()
                n = len(A)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    for i in range(n):
                        row_str = " ".join([f"{x:.6f}" for x in A[i]])
                        f.write(f"{row_str} {b[i]:.6f}\n")
                
                QMessageBox.information(self, "Успех", "Матрица успешно экспортирована!")
                
        except Exception as e:
            QMessageBox.critical(self, "Ошибка экспорта", f"Ошибка при экспорте матрицы: {str(e)}")
    
    def export_results(self):
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Сохранить результаты", "cg_results.txt", "Текстовые файлы (*.txt);;Все файлы (*)"
            )
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.results_display.text_edit.toPlainText())
                
                QMessageBox.information(self, "Успех", "Результаты успешно сохранены!")
                
        except Exception as e:
            QMessageBox.critical(self, "Ошибка экспорта", f"Ошибка при экспорте результатов: {str(e)}")

def main():
    app = QApplication(sys.argv)
    calculator = CGCalculator()
    calculator.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()