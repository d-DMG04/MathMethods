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
import sympy as sp

class FunctionInputWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Ввод функции
        function_layout = QHBoxLayout()
        function_layout.addWidget(QLabel("Функция f(x):"))
        self.function_input = QLineEdit()
        self.function_input.setPlaceholderText("sin(x) + x^2")
        self.function_input.setText("sin(x) + x^2")
        function_layout.addWidget(self.function_input)
        layout.addLayout(function_layout)
        
        # Пределы интегрирования
        limits_layout = QHBoxLayout()
        limits_layout.addWidget(QLabel("Пределы интегрирования:"))
        
        limits_layout.addWidget(QLabel("от"))
        self.a_input = QLineEdit()
        self.a_input.setMaximumWidth(80)
        self.a_input.setText("0")
        limits_layout.addWidget(self.a_input)
        
        limits_layout.addWidget(QLabel("до"))
        self.b_input = QLineEdit()
        self.b_input.setMaximumWidth(80)
        self.b_input.setText("3.14159")
        limits_layout.addWidget(self.b_input)
        
        layout.addLayout(limits_layout)
        
        # Количество разбиений
        partitions_layout = QHBoxLayout()
        partitions_layout.addWidget(QLabel("Количество разбиений:"))
        self.n_input = QSpinBox()
        self.n_input.setMinimum(1)
        self.n_input.setMaximum(10000)
        self.n_input.setValue(100)
        partitions_layout.addWidget(self.n_input)
        layout.addLayout(partitions_layout)
        
        # Тип метода
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Метод прямоугольников:"))
        self.method_type = QComboBox()
        self.method_type.addItems(["Левых прямоугольников", 
                                  "Правых прямоугольников", 
                                  "Средних прямоугольников"])
        method_layout.addWidget(self.method_type)
        layout.addLayout(method_layout)
        
        self.setLayout(layout)
    
    def get_parameters(self):
        try:
            function_str = self.function_input.text().strip()
            if not function_str:
                raise ValueError("Функция не может быть пустой")
            
            a = float(self.a_input.text())
            b = float(self.b_input.text())
            if a >= b:
                raise ValueError("Верхний предел должен быть больше нижнего")
            
            n = self.n_input.value()
            method = self.method_type.currentText()
            
            return function_str, a, b, n, method
            
        except ValueError as e:
            raise ValueError(f"Ошибка ввода данных: {str(e)}")

class ResultDisplayWidget(QWidget):
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
    
    def display_results(self, function_str, a, b, n, method_type, result, error_estimate, exact_result=None):
        text = "=== МЕТОД ПРЯМОУГОЛЬНИКОВ ===\n\n"
        
        text += f"Функция: f(x) = {function_str}\n"
        text += f"Пределы интегрирования: [{a:.6f}, {b:.6f}]\n"
        text += f"Количество разбиений: {n}\n"
        text += f"Метод: {method_type}\n\n"
        
        text += f"Приближенное значение интеграла: {result:.10f}\n"
        
        if exact_result is not None:
            text += f"Точное значение интеграла: {exact_result:.10f}\n"
            text += f"Абсолютная погрешность: {abs(result - exact_result):.2e}\n"
            text += f"Относительная погрешность: {abs((result - exact_result)/exact_result)*100:.6f}%\n"
        
        text += f"Оценка погрешности: {error_estimate:.2e}\n\n"
        
        text += f"Шаг интегрирования: h = {(b - a)/n:.6f}\n"
        
        self.text_edit.setPlainText(text)

class PlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        
        layout.addWidget(QLabel("Визуализация интегрирования:"))
        layout.addWidget(self.canvas)
        self.setLayout(layout)
    
    def plot_integration(self, x, y, rectangles, a, b, method_type, function_str):
        self.figure.clear()
        
        ax = self.figure.add_subplot(111)
        
        # График функции
        ax.plot(x, y, 'b-', linewidth=2, label=f'f(x) = {function_str}')
        
        # Прямоугольники
        for rect in rectangles:
            x_rect, y_rect, height = rect
            ax.fill_between(x_rect, y_rect, alpha=0.3, color='red')
            ax.plot(x_rect, y_rect, 'r-', linewidth=1)
        
        # Настройки графика
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title(f'Метод {method_type.lower()}\nИнтеграл ≈ {np.sum([h for _, _, h in rectangles]):.6f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(a - 0.1*(b-a), b + 0.1*(b-a))
        
        self.figure.tight_layout()
        self.canvas.draw()

class FunctionGeneratorDialog(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent, Qt.Window)
        self.parent = parent
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Генератор функций")
        self.setGeometry(200, 200, 400, 300)
        
        layout = QVBoxLayout()
        
        # Тип функции
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Тип функции:"))
        self.function_type = QComboBox()
        self.function_type.addItems(["Полиномиальная", 
                                   "Тригонометрическая", 
                                   "Экспоненциальная",
                                   "Логарифмическая",
                                   "Сложная"])
        type_layout.addWidget(self.function_type)
        layout.addLayout(type_layout)
        
        # Пределы интегрирования
        limits_layout = QHBoxLayout()
        limits_layout.addWidget(QLabel("Пределы интегрирования:"))
        
        self.a_min = QDoubleSpinBox()
        self.a_min.setRange(-100, 100)
        self.a_min.setValue(0)
        limits_layout.addWidget(QLabel("a min"))
        limits_layout.addWidget(self.a_min)
        
        self.a_max = QDoubleSpinBox()
        self.a_max.setRange(-100, 100)
        self.a_max.setValue(5)
        limits_layout.addWidget(QLabel("a max"))
        limits_layout.addWidget(self.a_max)
        
        self.b_min = QDoubleSpinBox()
        self.b_min.setRange(-100, 100)
        self.b_min.setValue(1)
        limits_layout.addWidget(QLabel("b min"))
        limits_layout.addWidget(self.b_min)
        
        self.b_max = QDoubleSpinBox()
        self.b_max.setRange(-100, 100)
        self.b_max.setValue(10)
        limits_layout.addWidget(QLabel("b max"))
        limits_layout.addWidget(self.b_max)
        
        layout.addLayout(limits_layout)
        
        # Сложность
        complexity_layout = QHBoxLayout()
        complexity_layout.addWidget(QLabel("Сложность:"))
        self.complexity = QSpinBox()
        self.complexity.setRange(1, 5)
        self.complexity.setValue(2)
        complexity_layout.addWidget(self.complexity)
        layout.addLayout(complexity_layout)
        
        # Кнопки
        button_layout = QHBoxLayout()
        self.generate_btn = QPushButton("Сгенерировать")
        self.generate_btn.clicked.connect(self.generate_function)
        self.cancel_btn = QPushButton("Отмена")
        self.cancel_btn.clicked.connect(self.close)
        
        button_layout.addWidget(self.generate_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def generate_function(self):
        try:
            func_type = self.function_type.currentText()
            complexity = self.complexity.value()
            
            function_str = self.generate_function_type(func_type, complexity)
            
            # Генерируем пределы интегрирования
            a = random.uniform(self.a_min.value(), self.a_max.value())
            b = random.uniform(max(a + 0.1, self.b_min.value()), self.b_max.value())
            
            if self.parent:
                self.parent.function_input.function_input.setText(function_str)
                self.parent.function_input.a_input.setText(f"{a:.4f}")
                self.parent.function_input.b_input.setText(f"{b:.4f}")
            
            self.close()
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))
    
    def generate_function_type(self, func_type, complexity):
        functions = {
            "Полиномиальная": [
                "x", "x^2", "x^3", "2*x + 1", "x^2 - 3*x + 2",
                "x^4 - 2*x^2 + 1", "3*x^3 - 2*x^2 + x - 5"
            ],
            "Тригонометрическая": [
                "sin(x)", "cos(x)", "sin(x) + cos(x)", "sin(2*x)", 
                "cos(x)^2", "sin(x)*cos(x)", "sin(x) + 2*cos(x)"
            ],
            "Экспоненциальная": [
                "exp(x)", "exp(-x)", "exp(2*x)", "exp(x) + exp(-x)",
                "exp(x)*sin(x)", "exp(-x^2)", "exp(x)/(1 + exp(x))"
            ],
            "Логарифмическая": [
                "log(x + 1)", "log(x)", "log(x^2 + 1)", "x*log(x + 1)",
                "log(x + 1)/(x + 1)", "sqrt(x)*log(x + 1)"
            ],
            "Сложная": [
                "sin(x) + x^2", "exp(-x)*cos(x)", "x*sin(x) + cos(x)",
                "log(x + 1) + sqrt(x)", "sin(x)^2 + cos(x)^2",
                "exp(-x^2/2)", "x/(1 + x^2)"
            ]
        }
        
        func_list = functions.get(func_type, functions["Сложная"])
        return random.choice(func_list)

class RectangleMethodSolver:
    @staticmethod
    def solve(function_str, a, b, n, method_type):
        """
        Решение интеграла методом прямоугольников
        """
        # Создаем lambda-функцию из строки
        x = sp.symbols('x')
        try:
            expr = sp.sympify(function_str)
            f = sp.lambdify(x, expr, 'numpy')
        except:
            raise ValueError("Неверный формат функции")
        
        h = (b - a) / n
        x_points = np.linspace(a, b, n + 1)
        
        rectangles = []
        result = 0.0
        
        if method_type == "Левых прямоугольников":
            for i in range(n):
                x_left = x_points[i]
                y_left = f(x_left)
                result += y_left * h
                
                # Для визуализации
                x_rect = [x_points[i], x_points[i+1], x_points[i+1], x_points[i]]
                y_rect = [0, 0, y_left, y_left]
                rectangles.append((x_rect, y_rect, y_left * h))
        
        elif method_type == "Правых прямоугольников":
            for i in range(n):
                x_right = x_points[i+1]
                y_right = f(x_right)
                result += y_right * h
                
                x_rect = [x_points[i], x_points[i+1], x_points[i+1], x_points[i]]
                y_rect = [0, 0, y_right, y_right]
                rectangles.append((x_rect, y_rect, y_right * h))
        
        elif method_type == "Средних прямоугольников":
            for i in range(n):
                x_mid = (x_points[i] + x_points[i+1]) / 2
                y_mid = f(x_mid)
                result += y_mid * h
                
                x_rect = [x_points[i], x_points[i+1], x_points[i+1], x_points[i]]
                y_rect = [0, 0, y_mid, y_mid]
                rectangles.append((x_rect, y_rect, y_mid * h))
        
        # Оценка погрешности
        error_estimate = RectangleMethodSolver.estimate_error(f, a, b, n, method_type)
        
        # Пытаемся вычислить точное значение
        exact_result = None
        try:
            exact_integral = sp.integrate(expr, (x, a, b))
            exact_result = float(exact_integral)
        except:
            pass
        
        # Данные для графика
        x_dense = np.linspace(a, b, 1000)
        y_dense = f(x_dense)
        
        return result, error_estimate, exact_result, x_dense, y_dense, rectangles
    
    @staticmethod
    def estimate_error(f, a, b, n, method_type):
        """
        Оценка погрешности метода прямоугольников
        """
        h = (b - a) / n
        
        if method_type == "Средних прямоугольников":
            # Погрешность O(h^2)
            return (b - a) * h**2 / 24 * np.max(np.abs([f(a), f(b)]))
        else:
            # Погрешность O(h)
            return (b - a) * h / 2 * np.max(np.abs([f(a), f(b)]))
    
    @staticmethod
    def calculate_exact_integral(function_str, a, b):
        """Вычисление точного значения интеграла"""
        x = sp.symbols('x')
        try:
            expr = sp.sympify(function_str)
            integral = sp.integrate(expr, (x, a, b))
            return float(integral)
        except:
            return None

class RectangleMethodCalculator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Калькулятор метода прямоугольников")
        self.setGeometry(100, 100, 1400, 900)
        
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Левая панель - ввод данных
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        
        input_group = QGroupBox("Ввод параметров интегрирования")
        input_layout = QVBoxLayout()
        self.function_input = FunctionInputWidget()
        input_layout.addWidget(self.function_input)
        input_group.setLayout(input_layout)
        left_layout.addWidget(input_group)
        
        # Кнопки действий
        button_layout = QGridLayout()
        
        self.solve_btn = QPushButton("Вычислить интеграл")
        self.solve_btn.clicked.connect(self.solve)
        button_layout.addWidget(self.solve_btn, 0, 0)
        
        self.exact_btn = QPushButton("Точное значение")
        self.exact_btn.clicked.connect(self.calculate_exact)
        button_layout.addWidget(self.exact_btn, 0, 1)
        
        self.clear_btn = QPushButton("Очистить")
        self.clear_btn.clicked.connect(self.clear)
        button_layout.addWidget(self.clear_btn, 1, 0)
        
        self.generate_btn = QPushButton("Сгенерировать функцию")
        self.generate_btn.clicked.connect(self.show_generator)
        button_layout.addWidget(self.generate_btn, 1, 1)
        
        self.export_results_btn = QPushButton("Экспорт результатов")
        self.export_results_btn.clicked.connect(self.export_results)
        button_layout.addWidget(self.export_results_btn, 2, 0, 1, 2)
        
        left_layout.addLayout(button_layout)
        left_panel.setLayout(left_layout)
        
        # Правая панель - результаты и графики
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        
        # Вкладки для результатов и графиков
        self.tabs = QTabWidget()
        
        # Вкладка с результатами
        self.results_display = ResultDisplayWidget()
        self.tabs.addTab(self.results_display, "Результаты")
        
        # Вкладка с графиками
        self.plot_widget = PlotWidget()
        self.tabs.addTab(self.plot_widget, "Визуализация")
        
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
            self.generator_dialog = FunctionGeneratorDialog(self)
        self.generator_dialog.show()
    
    def solve(self):
        try:
            function_str, a, b, n, method_type = self.function_input.get_parameters()
            
            solver = RectangleMethodSolver()
            result, error_estimate, exact_result, x_dense, y_dense, rectangles = solver.solve(
                function_str, a, b, n, method_type
            )
            
            self.results_display.display_results(
                function_str, a, b, n, method_type, result, error_estimate, exact_result
            )
            
            # Построение графика
            self.plot_widget.plot_integration(x_dense, y_dense, rectangles, a, b, method_type, function_str)
            self.tabs.setCurrentIndex(1)  # Переключаемся на вкладку с графиками
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))
    
    def calculate_exact(self):
        try:
            function_str, a, b, _, _ = self.function_input.get_parameters()
            
            exact_result = RectangleMethodSolver.calculate_exact_integral(function_str, a, b)
            
            if exact_result is not None:
                text = f"Точное значение интеграла ∫[{a:.6f}, {b:.6f}] {function_str} dx = {exact_result:.10f}"
                QMessageBox.information(self, "Точное значение", text)
            else:
                QMessageBox.warning(self, "Ошибка", "Не удалось вычислить точное значение интеграла")
                
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))
    
    def clear(self):
        self.function_input.function_input.clear()
        self.function_input.a_input.clear()
        self.function_input.b_input.clear()
        self.function_input.n_input.setValue(100)
        self.results_display.text_edit.clear()
        
        # Очищаем графики
        self.plot_widget.figure.clear()
        self.plot_widget.canvas.draw()
    
    def export_results(self):
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Сохранить результаты", "integration_results.txt", "Текстовые файлы (*.txt);;Все файлы (*)"
            )
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.results_display.text_edit.toPlainText())
                
                QMessageBox.information(self, "Успех", "Результаты успешно сохранены!")
                
        except Exception as e:
            QMessageBox.critical(self, "Ошибка экспорта", f"Ошибка при экспорте результатов: {str(e)}")

def main():
    app = QApplication(sys.argv)
    calculator = RectangleMethodCalculator()
    calculator.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()