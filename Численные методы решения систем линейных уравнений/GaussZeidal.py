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
        self.size_spinbox.setMaximum(20)
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
                       convergence_info, method_name="Гаусса-Зейделя"):
        text = f"=== МЕТОД {method_name.upper()} ===\n\n"
        
        text += "Исходная матрица A:\n"
        text += self.matrix_to_str(A) + "\n"
        
        text += "Вектор b:\n"
        text += " ".join([f"{x:8.4f}" for x in b]) + "\n\n"
        
        text += f"Количество итераций: {iterations}\n"
        text += f"Сходимость: {convergence_info}\n\n"
        
        text += "Решение системы:\n"
        for i, x in enumerate(solution):
            text += f"x{i+1} = {x:.10f}\n"
        
        if residuals_history is not None and len(residuals_history) > 0:
            text += f"\nФинальная невязка: {residuals_history[-1]:.2e}\n"
            
            text += "\nИстория невязок (первые 10 и последние 10 итераций):\n"
            if len(residuals_history) <= 20:
                for i, residual in enumerate(residuals_history):
                    text += f"Итерация {i+1}: {residual:.2e}\n"
            else:
                for i in range(10):
                    text += f"Итерация {i+1}: {residuals_history[i]:.2e}\n"
                text += "...\n"
                for i in range(len(residuals_history)-10, len(residuals_history)):
                    text += f"Итерация {i+1}: {residuals_history[i]:.2e}\n"
        
        # Проверка решения
        text += "\nПроверка решения (A*x - b):\n"
        residual_check = A @ solution - b
        for i, res in enumerate(residual_check):
            text += f"Уравнение {i+1}: {res:.2e}\n"
        
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
    
    def plot_convergence(self, residuals_history, errors_history=None, title="Сходимость метода Гаусса-Зейделя"):
        self.figure.clear()
        
        if errors_history:
            # Два подграфика: невязка и ошибка
            ax1 = self.figure.add_subplot(211)
            ax2 = self.figure.add_subplot(212)
            
            # График невязки
            ax1.semilogy(range(1, len(residuals_history) + 1), residuals_history, 
                        'b-', linewidth=2, marker='o', markersize=4, label='Невязка')
            ax1.set_xlabel('Итерация')
            ax1.set_ylabel('Невязка (log scale)')
            ax1.set_title('Сходимость по невязке')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # График ошибки
            ax2.semilogy(range(1, len(errors_history) + 1), errors_history,
                        'r-', linewidth=2, marker='s', markersize=4, label='Ошибка')
            ax2.set_xlabel('Итерация')
            ax2.set_ylabel('Ошибка (log scale)')
            ax2.set_title('Сходимость по ошибке')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
        else:
            # Один график невязки
            ax = self.figure.add_subplot(111)
            ax.semilogy(range(1, len(residuals_history) + 1), residuals_history,
                       'b-', linewidth=2, marker='o', markersize=4)
            ax.set_xlabel('Итерация')
            ax.set_ylabel('Невязка (log scale)')
            ax.set_title(title)
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
        self.size_spinbox.setMaximum(20)
        self.size_spinbox.setValue(5)
        size_layout.addWidget(self.size_spinbox)
        layout.addLayout(size_layout)
        
        # Тип матрицы
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Тип матрицы:"))
        self.matrix_type = QComboBox()
        self.matrix_type.addItems(["Диагонально-преобладающая", 
                                  "Симметричная положительно определенная",
                                  "Трёхдиагональная",
                                  "Случайная",
                                  "Строго диагонально-преобладающая"])
        type_layout.addWidget(self.matrix_type)
        layout.addLayout(type_layout)
        
        # Диапазон значений
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Диапазон значений:"))
        self.min_val = QDoubleSpinBox()
        self.min_val.setRange(-100, 100)
        self.min_val.setValue(-5)
        self.max_val = QDoubleSpinBox()
        self.max_val.setRange(-100, 100)
        self.max_val.setValue(5)
        range_layout.addWidget(QLabel("от"))
        range_layout.addWidget(self.min_val)
        range_layout.addWidget(QLabel("до"))
        range_layout.addWidget(self.max_val)
        layout.addLayout(range_layout)
        
        # Параметр диагонального преобладания
        self.strong_diag = QCheckBox("Усиленное диагональное преобладание")
        self.strong_diag.setChecked(True)
        layout.addWidget(self.strong_diag)
        
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
            strong_diag = self.strong_diag.isChecked()
            
            A = self.generate_matrix_type(n, min_val, max_val, matrix_type, strong_diag)
            
            # Генерируем вектор b
            x_exact = np.random.uniform(-10, 10, n)
            b = A @ x_exact
            
            if self.parent:
                self.parent.matrix_input.set_matrix(A, b)
            
            self.close()
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))
    
    def generate_matrix_type(self, n, min_val, max_val, matrix_type, strong_diag=True):
        if matrix_type == "Диагонально-преобладающая":
            A = np.random.uniform(min_val, max_val, (n, n))
            # Делаем матрицу диагонально преобладающей
            for i in range(n):
                row_sum = np.sum(np.abs(A[i])) - np.abs(A[i, i])
                if strong_diag:
                    A[i, i] = row_sum + random.uniform(2, 5)  # Сильное преобладание
                else:
                    A[i, i] = row_sum + random.uniform(0.1, 1)  # Слабое преобладание
            return A
        
        elif matrix_type == "Строго диагонально-преобладающая":
            A = np.random.uniform(min_val, max_val, (n, n))
            # Строгое диагональное преобладание
            for i in range(n):
                row_sum = np.sum(np.abs(A[i])) - np.abs(A[i, i])
                A[i, i] = row_sum + random.uniform(5, 10)  # Строгое преобладание
            return A
        
        elif matrix_type == "Симметричная положительно определенная":
            A = np.random.uniform(min_val, max_val, (n, n))
            A = (A + A.T) / 2  # Делаем симметричной
            # Делаем положительно определенной
            A += n * np.eye(n) * max_val
            return A
        
        elif matrix_type == "Трёхдиагональная":
            # Трёхдиагональная матрица
            A = np.zeros((n, n))
            for i in range(n):
                if i > 0:
                    A[i, i-1] = random.uniform(min_val, max_val)
                A[i, i] = random.uniform(min_val + 5, max_val + 5)  # Главная диагональ
                if i < n-1:
                    A[i, i+1] = random.uniform(min_val, max_val)
            return A
        
        elif matrix_type == "Случайная":
            return np.random.uniform(min_val, max_val, (n, n))
        
        return np.eye(n)

class GaussSeidelSolver:
    @staticmethod
    def solve(A, b, tolerance=1e-6, max_iterations=1000, omega=1.0):
        """
        Решение системы Ax = b методом Гаусса-Зейделя
        omega - параметр релаксации (1.0 - обычный метод, <1.0 - нижняя релаксация, >1.0 - верхняя релаксация)
        """
        n = len(b)
        x = np.zeros(n)  # Начальное приближение
        x_prev = x.copy()
        residuals_history = []
        errors_history = []
        
        # Проверка сходимости
        convergence_info = GaussSeidelSolver.check_convergence(A)
        
        # Итерационный процесс
        for iteration in range(max_iterations):
            x_prev = x.copy()
            
            for i in range(n):
                sigma = 0.0
                for j in range(n):
                    if j != i:
                        sigma += A[i, j] * x[j]
                
                # Метод Гаусса-Зейделя с релаксацией
                x[i] = (1 - omega) * x_prev[i] + omega * (b[i] - sigma) / A[i, i]
            
            # Вычисляем невязку
            residual = np.linalg.norm(A @ x - b)
            residuals_history.append(residual)
            
            # Вычисляем изменение решения
            error = np.linalg.norm(x - x_prev)
            errors_history.append(error)
            
            # Проверка условия остановки
            if residual < tolerance and error < tolerance:
                break
        
        convergence_status = "Сходимость достигнута" if residual < tolerance else "Максимум итераций"
        
        return x, iteration + 1, residuals_history, errors_history, convergence_info
    
    @staticmethod
    def check_convergence(A):
        """
        Проверка условий сходимости метода Гаусса-Зейделя
        """
        n = len(A)
        
        # 1. Проверка диагонального преобладания
        strict_diag_dom = True
        weak_diag_dom = True
        
        for i in range(n):
            diag = np.abs(A[i, i])
            off_diag_sum = np.sum(np.abs(A[i])) - diag
            
            if diag <= off_diag_sum:
                strict_diag_dom = False
            if diag < off_diag_sum:
                weak_diag_dom = False
        
        # 2. Проверка симметрии и положительной определенности
        symmetric = np.allclose(A, A.T)
        positive_definite = False
        try:
            np.linalg.cholesky(A)
            positive_definite = True
        except:
            pass
        
        # 3. Проверка спектрального радиуса
        D = np.diag(np.diag(A))
        L = np.tril(A, -1)
        U = np.triu(A, 1)
        
        # Матрица итерации Гаусса-Зейделя: G = -(D + L)^(-1) * U
        try:
            DL_inv = np.linalg.inv(D + L)
            G = -DL_inv @ U
            spectral_radius = np.max(np.abs(np.linalg.eigvals(G)))
        except:
            spectral_radius = float('inf')
        
        # Анализ сходимости
        if strict_diag_dom:
            convergence = "Гарантированная сходимость (строгое диагональное преобладание)"
        elif weak_diag_dom and symmetric and positive_definite:
            convergence = "Гарантированная сходимость (симметричная положительно определенная)"
        elif spectral_radius < 1:
            convergence = f"Сходимость (спектральный радиус: {spectral_radius:.4f} < 1)"
        else:
            convergence = "Сходимость не гарантирована"
        
        return {
            'strict_diagonal_dominance': strict_diag_dom,
            'weak_diagonal_dominance': weak_diag_dom,
            'symmetric': symmetric,
            'positive_definite': positive_definite,
            'spectral_radius': spectral_radius,
            'convergence_info': convergence
        }
    
    @staticmethod
    def calculate_optimal_omega(A):
        """
        Оценка оптимального параметра релаксации для метода SOR
        """
        try:
            # Простая оценка на основе спектрального радиуса
            D = np.diag(np.diag(A))
            L = np.tril(A, -1)
            U = np.triu(A, 1)
            
            # Матрица итерации Якоби
            D_inv = np.linalg.inv(D)
            J = -D_inv @ (L + U)
            rho_j = np.max(np.abs(np.linalg.eigvals(J)))
            
            # Оптимальный параметр для SOR
            omega_opt = 2 / (1 + np.sqrt(1 - rho_j**2)) if rho_j < 1 else 1.0
            return min(omega_opt, 1.9)  # Ограничиваем сверху
        except:
            return 1.0

class GaussSeidelCalculator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Калькулятор метода Гаусса-Зейделя")
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
        self.tolerance = QLineEdit("1e-6")
        self.tolerance.setMaximumWidth(100)
        params_layout.addWidget(self.tolerance, 0, 1)
        
        params_layout.addWidget(QLabel("Макс. итераций:"), 1, 0)
        self.max_iter = QSpinBox()
        self.max_iter.setRange(10, 100000)
        self.max_iter.setValue(1000)
        self.max_iter.setMaximumWidth(100)
        params_layout.addWidget(self.max_iter, 1, 1)
        
        params_layout.addWidget(QLabel("Параметр релаксации (ω):"), 2, 0)
        self.omega = QDoubleSpinBox()
        self.omega.setRange(0.1, 2.0)
        self.omega.setValue(1.0)
        self.omega.setSingleStep(0.1)
        self.omega.setMaximumWidth(100)
        params_layout.addWidget(self.omega, 2, 1)
        
        self.auto_omega = QCheckBox("Автоподбор ω")
        self.auto_omega.setChecked(False)
        params_layout.addWidget(self.auto_omega, 3, 0, 1, 2)
        
        params_group.setLayout(params_layout)
        left_layout.addWidget(params_group)
        
        # Кнопки действий
        button_layout = QGridLayout()
        
        self.solve_btn = QPushButton("Решить")
        self.solve_btn.clicked.connect(self.solve)
        button_layout.addWidget(self.solve_btn, 0, 0)
        
        self.check_btn = QPushButton("Проверить сходимость")
        self.check_btn.clicked.connect(self.check_convergence)
        button_layout.addWidget(self.check_btn, 0, 1)
        
        self.optimize_btn = QPushButton("Оптимизировать ω")
        self.optimize_btn.clicked.connect(self.optimize_omega)
        button_layout.addWidget(self.optimize_btn, 1, 0)
        
        self.clear_btn = QPushButton("Очистить")
        self.clear_btn.clicked.connect(self.clear)
        button_layout.addWidget(self.clear_btn, 1, 1)
        
        self.generate_btn = QPushButton("Сгенерировать")
        self.generate_btn.clicked.connect(self.show_generator)
        button_layout.addWidget(self.generate_btn, 2, 0)
        
        self.import_btn = QPushButton("Импорт матрицы")
        self.import_btn.clicked.connect(self.import_matrix)
        button_layout.addWidget(self.import_btn, 2, 1)
        
        self.export_matrix_btn = QPushButton("Экспорт матрицы")
        self.export_matrix_btn.clicked.connect(self.export_matrix)
        button_layout.addWidget(self.export_matrix_btn, 3, 0)
        
        self.export_results_btn = QPushButton("Экспорт результатов")
        self.export_results_btn.clicked.connect(self.export_results)
        button_layout.addWidget(self.export_results_btn, 3, 1)
        
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
    
    def show_generator(self):
        if self.generator_dialog is None:
            self.generator_dialog = MatrixGeneratorDialog(self)
        self.generator_dialog.show()
    
    def solve(self):
        try:
            A, b = self.matrix_input.get_matrix()
            tolerance = float(self.tolerance.text())
            max_iterations = self.max_iter.value()
            
            # Проверка диагональных элементов
            for i in range(len(A)):
                if A[i, i] == 0:
                    QMessageBox.warning(self, "Ошибка", 
                                      f"Нулевой диагональный элемент a[{i+1},{i+1}]!")
                    return
            
            # Выбор параметра релаксации
            if self.auto_omega.isChecked():
                omega = GaussSeidelSolver.calculate_optimal_omega(A)
                self.omega.setValue(omega)
            else:
                omega = self.omega.value()
            
            solver = GaussSeidelSolver()
            solution, iterations, residuals_history, errors_history, convergence_info = solver.solve(
                A, b, tolerance, max_iterations, omega
            )
            
            if iterations >= max_iterations:
                QMessageBox.warning(self, "Предупреждение", 
                                  f"Достигнуто максимальное количество итераций ({max_iterations}).")
            
            convergence_text = f"{convergence_info['convergence_info']}"
            if omega != 1.0:
                convergence_text += f" (ω = {omega:.3f})"
            
            self.results_display.display_results(
                A, b, solution, iterations, residuals_history, convergence_text, "Гаусса-Зейделя"
            )
            
            # Построение графиков
            self.plot_widget.plot_convergence(residuals_history, errors_history, 
                                            f"Сходимость метода Гаусса-Зейделя (ω = {omega:.3f})")
            self.tabs.setCurrentIndex(1)  # Переключаемся на вкладку с графиками
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))
    
    def check_convergence(self):
        try:
            A, b = self.matrix_input.get_matrix()
            convergence_info = GaussSeidelSolver.check_convergence(A)
            
            text = "=== АНАЛИЗ СХОДИМОСТИ МЕТОДА ГАУССА-ЗЕЙДЕЛЯ ===\n\n"
            
            text += "Строгое диагональное преобладание: "
            text += "Да" if convergence_info['strict_diagonal_dominance'] else "Нет"
            text += "\n"
            
            text += "Нестрогое диагональное преобладание: "
            text += "Да" if convergence_info['weak_diagonal_dominance'] else "Нет"
            text += "\n"
            
            text += "Симметричность: "
            text += "Да" if convergence_info['symmetric'] else "Нет"
            text += "\n"
            
            text += "Положительная определенность: "
            text += "Да" if convergence_info['positive_definite'] else "Нет"
            text += "\n"
            
            text += f"Спектральный радиус: {convergence_info['spectral_radius']:.6f}\n\n"
            
            text += f"Оценка сходимости: {convergence_info['convergence_info']}\n\n"
            
            # Рекомендации
            text += "Рекомендации:\n"
            if not convergence_info['strict_diagonal_dominance']:
                text += "- Рассмотрите перестановку строк для достижения диагонального преобладания\n"
            if convergence_info['spectral_radius'] >= 1:
                text += "- Метод может не сходиться, рассмотрите другие методы\n"
            if convergence_info['symmetric'] and convergence_info['positive_definite']:
                text += "- Матрица SPD, метод должен хорошо сходиться\n"
            
            # Рекомендация по параметру релаксации
            optimal_omega = GaussSeidelSolver.calculate_optimal_omega(A)
            text += f"\nРекомендуемый параметр релаксации ω: {optimal_omega:.3f}\n"
            if optimal_omega > 1.0:
                text += "(Метод верхней релаксации - SOR)\n"
            elif optimal_omega < 1.0:
                text += "(Метод нижней релаксации)\n"
            else:
                text += "(Обычный метод Гаусса-Зейделя)\n"
            
            self.results_display.text_edit.setPlainText(text)
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))
    
    def optimize_omega(self):
        try:
            A, b = self.matrix_input.get_matrix()
            optimal_omega = GaussSeidelSolver.calculate_optimal_omega(A)
            self.omega.setValue(optimal_omega)
            QMessageBox.information(self, "Оптимизация", 
                                  f"Оптимальный параметр релаксации: ω = {optimal_omega:.3f}")
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))
    
    def clear(self):
        for row in self.matrix_input.matrix_inputs:
            for input_field in row:
                input_field.clear()
        self.results_display.text_edit.clear()
        self.tolerance.setText("1e-6")
        self.max_iter.setValue(1000)
        self.omega.setValue(1.0)
        self.auto_omega.setChecked(False)
        
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
                self, "Сохранить результаты", "gauss_seidel_results.txt", "Текстовые файлы (*.txt);;Все файлы (*)"
            )
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.results_display.text_edit.toPlainText())
                
                QMessageBox.information(self, "Успех", "Результаты успешно сохранены!")
                
        except Exception as e:
            QMessageBox.critical(self, "Ошибка экспорта", f"Ошибка при экспорте результатов: {str(e)}")

def main():
    app = QApplication(sys.argv)
    calculator = GaussSeidelCalculator()
    calculator.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()