import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGridLayout, QTextEdit, QLineEdit,
                             QPushButton, QLabel, QSpinBox, QFileDialog, 
                             QMessageBox, QTabWidget, QSplitter, QGroupBox,
                             QComboBox, QDoubleSpinBox, QCheckBox, QTableWidget,
                             QTableWidgetItem, QHeaderView)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QTextCursor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import random
import sympy as sp
from scipy import integrate

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
        self.function_input.setText("sin(x)")
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
        
        # Параметры интегрирования
        params_layout = QGridLayout()
        
        params_layout.addWidget(QLabel("Количество разбиений:"), 0, 0)
        self.n_input = QSpinBox()
        self.n_input.setMinimum(1)
        self.n_input.setMaximum(100000)
        self.n_input.setValue(10)
        self.n_input.valueChanged.connect(self.update_step_display)
        params_layout.addWidget(self.n_input, 0, 1)
        
        params_layout.addWidget(QLabel("Точность:"), 1, 0)
        self.tolerance_input = QLineEdit()
        self.tolerance_input.setText("1e-6")
        self.tolerance_input.setMaximumWidth(80)
        params_layout.addWidget(self.tolerance_input, 1, 1)
        
        params_layout.addWidget(QLabel("Шаг (h):"), 2, 0)
        self.step_label = QLabel("0.314159")
        params_layout.addWidget(self.step_label, 2, 1)
        
        # Автоматический подбор разбиений
        self.auto_n = QCheckBox("Автоподбор разбиений по точности")
        self.auto_n.setChecked(False)
        params_layout.addWidget(self.auto_n, 3, 0, 1, 2)
        
        layout.addLayout(params_layout)
        
        self.setLayout(layout)
    
    def update_step_display(self):
        try:
            a = float(self.a_input.text())
            b = float(self.b_input.text())
            n = self.n_input.value()
            h = (b - a) / n
            self.step_label.setText(f"{h:.6f}")
        except:
            self.step_label.setText("N/A")
    
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
            tolerance = float(self.tolerance_input.text())
            auto_n = self.auto_n.isChecked()
            
            return function_str, a, b, n, tolerance, auto_n
            
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
    
    def display_results(self, function_str, a, b, n, result, error_estimate, 
                       exact_result=None, iterations_info=None, richardson_result=None):
        text = "=== МЕТОД ТРАПЕЦИЙ ===\n\n"
        
        text += f"Функция: f(x) = {function_str}\n"
        text += f"Пределы интегрирования: [{a:.6f}, {b:.6f}]\n"
        text += f"Количество разбиений: {n}\n"
        text += f"Шаг интегрирования: h = {(b - a)/n:.6f}\n\n"
        
        text += f"Приближенное значение интеграла: {result:.12f}\n"
        
        if richardson_result is not None:
            text += f"Уточненное значение (Ричардсон): {richardson_result:.12f}\n"
        
        if exact_result is not None:
            text += f"Точное значение интеграла: {exact_result:.12f}\n"
            abs_error = abs(result - exact_result)
            rel_error = abs(abs_error / exact_result) * 100 if exact_result != 0 else float('inf')
            text += f"Абсолютная погрешность: {abs_error:.2e}\n"
            text += f"Относительная погрешность: {rel_error:.6f}%\n"
        
        text += f"Теоретическая оценка погрешности: {error_estimate:.2e}\n"
        
        if iterations_info:
            text += f"\nИнформация об итерациях: {iterations_info}\n"
        
        # Дополнительная информация
        text += f"\nДополнительно:\n"
        text += f"Общая площадь трапеций: {result:.6f}\n"
        text += f"Длина интервала: {b - a:.6f}\n"
        text += f"Среднее значение функции: {result / (b - a):.6f}\n"
        
        self.text_edit.setPlainText(text)

class TableWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        layout.addWidget(QLabel("Таблица вычислений:"))
        layout.addWidget(self.table)
        self.setLayout(layout)
    
    def display_table(self, data):
        """Отображение таблицы с данными вычислений"""
        n = len(data)
        self.table.setRowCount(n)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["i", "x_i", "x_{i+1}", "f(x_i)", "f(x_{i+1})", "Площадь"])
        
        for i, row_data in enumerate(data):
            self.table.setItem(i, 0, QTableWidgetItem(str(i)))
            for j, value in enumerate(row_data, 1):
                self.table.setItem(i, j, QTableWidgetItem(f"{value:.6f}"))
        
        # Автоподбор размера столбцов
        self.table.resizeColumnsToContents()

class PlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        self.figure = Figure(figsize=(12, 10))
        self.canvas = FigureCanvas(self.figure)
        
        layout.addWidget(QLabel("Визуализация интегрирования:"))
        layout.addWidget(self.canvas)
        self.setLayout(layout)
    
    def plot_integration(self, x, y, trapezoids, a, b, function_str, result):
        self.figure.clear()
        
        # Создаем 2x2 grid подграфиков
        ax1 = self.figure.add_subplot(221)  # Основной график
        ax2 = self.figure.add_subplot(222)  # Крупный план
        ax3 = self.figure.add_subplot(223)  # Ошибка по интервалам
        ax4 = self.figure.add_subplot(224)  # Сходимость
        
        # Основной график
        ax1.plot(x, y, 'b-', linewidth=2, label=f'f(x) = {function_str}', zorder=5)
        
        # Трапеции
        total_area = 0
        areas = []
        for i, trapezoid in enumerate(trapezoids):
            x_trap, y_trap, area = trapezoid
            ax1.fill_between(x_trap, y_trap, alpha=0.3, color='red', zorder=1)
            ax1.plot(x_trap, y_trap, 'r-', linewidth=1, zorder=2)
            total_area += area
            areas.append(area)
        
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        ax1.set_title(f'Метод трапеций\nИнтеграл ≈ {result:.6f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(a - 0.1*(b-a), b + 0.1*(b-a))
        
        # Крупный план (первые 3 трапеции)
        if len(trapezoids) > 3:
            zoom_end = min(4, len(trapezoids))
            zoom_x = [a, trapezoids[zoom_end-1][0][2]]  # x координаты первых трапеций
            
            ax2.plot(x, y, 'b-', linewidth=2, zorder=5)
            for i in range(zoom_end):
                x_trap, y_trap, area = trapezoids[i]
                ax2.fill_between(x_trap, y_trap, alpha=0.5, color='green', zorder=1)
                ax2.plot(x_trap, y_trap, 'g-', linewidth=1, zorder=2)
            
            ax2.set_xlabel('x')
            ax2.set_ylabel('f(x)')
            ax2.set_title('Крупный план (первые трапеции)')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(zoom_x[0], zoom_x[1])
        
        # График площадей трапеций
        if areas:
            x_areas = np.arange(len(areas))
            ax3.bar(x_areas, areas, alpha=0.7, color='orange')
            ax3.set_xlabel('Номер трапеции')
            ax3.set_ylabel('Площадь')
            ax3.set_title('Площади отдельных трапеций')
            ax3.grid(True, alpha=0.3)
        
        # Заглушка для графика сходимости (заполнится позже)
        ax4.text(0.5, 0.5, 'График сходимости\nбудет построен при\nадаптивном интегрировании', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Сходимость метода')
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def plot_convergence(self, n_values, errors, exact_result):
        """График сходимости метода"""
        ax = self.figure.add_subplot(224)
        ax.loglog(n_values, errors, 'bo-', linewidth=2, markersize=4)
        ax.set_xlabel('Количество разбиений (n)')
        ax.set_ylabel('Абсолютная погрешность')
        ax.set_title('Сходимость метода трапеций')
        ax.grid(True, alpha=0.3)
        
        # Теоретическая сходимость O(h^2)
        if len(n_values) > 1:
            h_values = [(n_values[0]/n) for n in n_values]
            theoretical = [errors[0] * (h/h_values[0])**2 for h in h_values]
            ax.loglog(n_values, theoretical, 'r--', label='Теоретическая O(h²)')
            ax.legend()

class FunctionGeneratorDialog(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent, Qt.Window)
        self.parent = parent
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Генератор функций")
        self.setGeometry(200, 200, 500, 400)
        
        layout = QVBoxLayout()
        
        # Категория функции
        category_layout = QHBoxLayout()
        category_layout.addWidget(QLabel("Категория:"))
        self.category = QComboBox()
        self.category.addItems(["Тригонометрические", "Полиномы", "Экспоненциальные", 
                               "Логарифмические", "Специальные", "Случайные"])
        self.category.currentTextChanged.connect(self.update_function_list)
        category_layout.addWidget(self.category)
        layout.addLayout(category_layout)
        
        # Выбор конкретной функции
        function_layout = QHBoxLayout()
        function_layout.addWidget(QLabel("Функция:"))
        self.function_list = QComboBox()
        function_layout.addWidget(self.function_list)
        layout.addLayout(function_layout)
        
        # Пределы интегрирования
        limits_group = QGroupBox("Пределы интегрирования")
        limits_layout = QGridLayout()
        
        limits_layout.addWidget(QLabel("Нижний предел a:"), 0, 0)
        self.a_min = QDoubleSpinBox()
        self.a_min.setRange(-100, 100)
        self.a_min.setValue(0)
        limits_layout.addWidget(self.a_min, 0, 1)
        
        limits_layout.addWidget(QLabel("Верхний предел b:"), 1, 0)
        self.b_min = QDoubleSpinBox()
        self.b_min.setRange(-100, 100)
        self.b_min.setValue(1)
        limits_layout.addWidget(self.b_min, 1, 1)
        
        limits_layout.addWidget(QLabel("Диапазон a:"), 2, 0)
        self.a_range = QDoubleSpinBox()
        self.a_range.setRange(0.1, 20)
        self.a_range.setValue(5)
        limits_layout.addWidget(self.a_range, 2, 1)
        
        limits_layout.addWidget(QLabel("Диапазон b:"), 3, 0)
        self.b_range = QDoubleSpinBox()
        self.b_range.setRange(0.1, 20)
        self.b_range.setValue(5)
        limits_layout.addWidget(self.b_range, 3, 1)
        
        limits_group.setLayout(limits_layout)
        layout.addWidget(limits_group)
        
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
        self.update_function_list()
    
    def update_function_list(self):
        category = self.category.currentText()
        functions = {
            "Тригонометрические": ["sin(x)", "cos(x)", "sin(x)^2", "cos(2*x)", "sin(x) + cos(x)", "tan(x)"],
            "Полиномы": ["x", "x^2", "x^3", "x^2 + 2*x + 1", "x^4 - 3*x^2", "2*x^3 - x^2 + x - 5"],
            "Экспоненциальные": ["exp(x)", "exp(-x)", "exp(x^2)", "exp(-x^2)", "x*exp(-x)", "exp(x)*sin(x)"],
            "Логарифмические": ["log(x+1)", "log(x^2+1)", "x*log(x+1)", "log(1 + x^2)", "sqrt(x)*log(x+1)"],
            "Специальные": ["1/(1 + x^2)", "sqrt(1 - x^2)", "sin(x)/x", "exp(-x^2/2)", "1/sqrt(1 - x^2)"],
            "Случайные": ["случайный выбор"]
        }
        
        self.function_list.clear()
        self.function_list.addItems(functions.get(category, ["sin(x)"]))
    
    def generate_function(self):
        try:
            function_str = self.function_list.currentText()
            if function_str == "случайный выбор":
                # Случайный выбор из всех категорий
                all_functions = []
                for cat in ["Тригонометрические", "Полиномы", "Экспоненциальные", "Логарифмические", "Специальные"]:
                    self.category.setCurrentText(cat)
                    self.update_function_list()
                    all_functions.extend([self.function_list.itemText(i) for i in range(self.function_list.count())])
                function_str = random.choice(all_functions)
            
            # Генерация пределов интегрирования
            a_base = self.a_min.value()
            b_base = self.b_min.value()
            a_range = self.a_range.value()
            b_range = self.b_range.value()
            
            a = random.uniform(a_base, a_base + a_range)
            b = random.uniform(max(a + 0.1, b_base), b_base + b_range)
            
            if self.parent:
                self.parent.function_input.function_input.setText(function_str)
                self.parent.function_input.a_input.setText(f"{a:.4f}")
                self.parent.function_input.b_input.setText(f"{b:.4f}")
                self.parent.function_input.n_input.setValue(random.randint(5, 50))
            
            self.close()
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

class TrapezoidalSolver:
    @staticmethod
    def solve(function_str, a, b, n, tolerance=1e-6, auto_n=False):
        """
        Решение интеграла методом трапеций с расширенным функционалом
        """
        # Создаем lambda-функцию из строки
        x = sp.symbols('x')
        try:
            expr = sp.sympify(function_str)
            f = sp.lambdify(x, expr, 'numpy')
        except:
            raise ValueError("Неверный формат функции")
        
        if auto_n:
            # Адаптивное интегрирование с автоматическим подбором n
            return TrapezoidalSolver.adaptive_solve(f, a, b, tolerance)
        else:
            # Обычный метод трапеций с фиксированным n
            return TrapezoidalSolver.fixed_step_solve(f, a, b, n)
    
    @staticmethod
    def fixed_step_solve(f, a, b, n):
        """Метод трапеций с фиксированным шагом"""
        h = (b - a) / n
        x_points = np.linspace(a, b, n + 1)
        y_points = f(x_points)
        
        # Основная формула трапеций
        result = (h / 2) * (y_points[0] + y_points[-1] + 2 * np.sum(y_points[1:-1]))
        
        # Данные для визуализации
        trapezoids = []
        table_data = []
        
        for i in range(n):
            x_left = x_points[i]
            x_right = x_points[i+1]
            y_left = y_points[i]
            y_right = y_points[i+1]
            
            area = (h / 2) * (y_left + y_right)
            
            # Для визуализации
            x_trap = [x_left, x_right, x_right, x_left]
            y_trap = [0, 0, y_right, y_left]
            trapezoids.append((x_trap, y_trap, area))
            
            # Для таблицы
            table_data.append([x_left, x_right, y_left, y_right, area])
        
        # Оценка погрешности
        error_estimate = TrapezoidalSolver.estimate_error(f, a, b, n)
        
        # Точное значение (если возможно)
        exact_result = TrapezoidalSolver.calculate_exact_integral(f, a, b)
        
        # Данные для графика
        x_dense = np.linspace(a, b, 1000)
        y_dense = f(x_dense)
        
        return {
            'result': result,
            'error_estimate': error_estimate,
            'exact_result': exact_result,
            'x_dense': x_dense,
            'y_dense': y_dense,
            'trapezoids': trapezoids,
            'table_data': table_data,
            'iterations_info': f"Фиксированный шаг, n = {n}",
            'n_values': [n],
            'errors': [abs(result - exact_result)] if exact_result is not None else []
        }
    
    @staticmethod
    def adaptive_solve(f, a, b, tolerance):
        """Адаптивный метод трапеций с автоматическим подбором n"""
        n = 4  # Начальное количество разбиений
        max_iterations = 20
        results = []
        n_values = []
        errors = []
        
        for iteration in range(max_iterations):
            result = TrapezoidalSolver.fixed_step_solve(f, a, b, n)
            results.append(result['result'])
            n_values.append(n)
            
            if result['exact_result'] is not None:
                error = abs(result['result'] - result['exact_result'])
                errors.append(error)
                
                if error < tolerance:
                    break
            
            n *= 2  # Удваиваем количество разбиений
        
        # Экстраполяция Ричардсона
        richardson_result = None
        if len(results) >= 2:
            richardson_result = (4 * results[-1] - results[-2]) / 3
        
        final_result = results[-1] if results else 0
        
        return {
            'result': final_result,
            'error_estimate': tolerance,
            'exact_result': result.get('exact_result'),
            'x_dense': result.get('x_dense', np.linspace(a, b, 1000)),
            'y_dense': result.get('y_dense', f(np.linspace(a, b, 1000))),
            'trapezoids': result.get('trapezoids', []),
            'table_data': result.get('table_data', []),
            'iterations_info': f"Адаптивный метод, {iteration+1} итераций, n = {n}",
            'richardson_result': richardson_result,
            'n_values': n_values,
            'errors': errors
        }
    
    @staticmethod
    def estimate_error(f, a, b, n):
        """Оценка погрешности метода трапеций"""
        h = (b - a) / n
        
        # Оценка второй производной (упрощенная)
        x_mid = (a + b) / 2
        try:
            # Численная оценка второй производной
            dx = 1e-5
            f2 = (f(x_mid + dx) - 2*f(x_mid) + f(x_mid - dx)) / (dx**2)
            error = abs((b - a) * h**2 / 12 * f2)
        except:
            # Упрощенная оценка
            error = (b - a) * h**2 / 12 * max(abs(f(a)), abs(f(b)))
        
        return error
    
    @staticmethod
    def calculate_exact_integral(f, a, b):
        """Вычисление точного значения интеграла"""
        x = sp.symbols('x')
        try:
            # Пытаемся получить строку функции для символьного интегрирования
            # Это упрощенный подход - в реальности нужно парсить f
            return integrate.quad(f, a, b)[0]
        except:
            try:
                # Альтернативный подход через sympy
                func_str = str(f)
                if 'lambda' in func_str:
                    return None
                expr = sp.sympify(func_str.split(':')[-1].strip())
                integral = sp.integrate(expr, (x, a, b))
                return float(integral)
            except:
                return None

class TrapezoidalCalculator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Калькулятор метода трапеций")
        self.setGeometry(100, 100, 1600, 1000)
        
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Левая панель - ввод данных
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        
        input_group = QGroupBox("Параметры интегрирования")
        input_layout = QVBoxLayout()
        self.function_input = FunctionInputWidget()
        input_layout.addWidget(self.function_input)
        input_group.setLayout(input_layout)
        left_layout.addWidget(input_group)
        
        # Дополнительные методы
        methods_group = QGroupBox("Дополнительные методы")
        methods_layout = QVBoxLayout()
        
        self.richardson_cb = QCheckBox("Экстраполяция Ричардсона")
        self.richardson_cb.setChecked(True)
        methods_layout.addWidget(self.richardson_cb)
        
        self.compare_cb = QCheckBox("Сравнение с другими методами")
        self.compare_cb.setChecked(True)
        methods_layout.addWidget(self.compare_cb)
        
        methods_group.setLayout(methods_layout)
        left_layout.addWidget(methods_group)
        
        # Кнопки действий
        button_layout = QGridLayout()
        
        self.solve_btn = QPushButton("Вычислить интеграл")
        self.solve_btn.clicked.connect(self.solve)
        button_layout.addWidget(self.solve_btn, 0, 0)
        
        self.adaptive_btn = QPushButton("Адаптивное интегрирование")
        self.adaptive_btn.clicked.connect(self.adaptive_solve)
        button_layout.addWidget(self.adaptive_btn, 0, 1)
        
        self.exact_btn = QPushButton("Точное значение")
        self.exact_btn.clicked.connect(self.calculate_exact)
        button_layout.addWidget(self.exact_btn, 1, 0)
        
        self.clear_btn = QPushButton("Очистить")
        self.clear_btn.clicked.connect(self.clear)
        button_layout.addWidget(self.clear_btn, 1, 1)
        
        self.generate_btn = QPushButton("Сгенерировать функцию")
        self.generate_btn.clicked.connect(self.show_generator)
        button_layout.addWidget(self.generate_btn, 2, 0)
        
        self.export_btn = QPushButton("Экспорт результатов")
        self.export_btn.clicked.connect(self.export_results)
        button_layout.addWidget(self.export_btn, 2, 1)
        
        left_layout.addLayout(button_layout)
        left_panel.setLayout(left_layout)
        
        # Правая панель - результаты и графики
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        
        # Вкладки
        self.tabs = QTabWidget()
        
        # Вкладка с результатами
        self.results_display = ResultDisplayWidget()
        self.tabs.addTab(self.results_display, "Результаты")
        
        # Вкладка с таблицей
        self.table_widget = TableWidget()
        self.tabs.addTab(self.table_widget, "Таблица вычислений")
        
        # Вкладка с графиками
        self.plot_widget = PlotWidget()
        self.tabs.addTab(self.plot_widget, "Визуализация")
        
        right_layout.addWidget(self.tabs)
        right_panel.setLayout(right_layout)
        
        # Разделитель
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([500, 1100])
        
        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        self.generator_dialog = None
        self.current_result = None
    
    def show_generator(self):
        if self.generator_dialog is None:
            self.generator_dialog = FunctionGeneratorDialog(self)
        self.generator_dialog.show()
    
    def solve(self):
        try:
            function_str, a, b, n, tolerance, auto_n = self.function_input.get_parameters()
            
            solver = TrapezoidalSolver()
            result = solver.solve(function_str, a, b, n, tolerance, auto_n)
            self.current_result = result
            
            # Применяем экстраполяцию Ричардсона если нужно
            richardson_result = result.get('richardson_result') if self.richardson_cb.isChecked() else None
            
            self.results_display.display_results(
                function_str, a, b, n, 
                result['result'], result['error_estimate'], 
                result['exact_result'], result['iterations_info'],
                richardson_result
            )
            
            # Отображаем таблицу данных
            if result['table_data']:
                self.table_widget.display_table(result['table_data'])
            
            # Строим графики
            self.plot_widget.plot_integration(
                result['x_dense'], result['y_dense'], 
                result['trapezoids'], a, b, function_str, result['result']
            )
            
            # Если есть данные о сходимости, строим график
            if result.get('n_values') and result.get('errors'):
                self.plot_widget.plot_convergence(
                    result['n_values'], result['errors'], result['exact_result']
                )
            
            self.tabs.setCurrentIndex(2)  # Переключаемся на вкладку с графиками
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))
    
    def adaptive_solve(self):
        try:
            function_str, a, b, n, tolerance, _ = self.function_input.get_parameters()
            
            solver = TrapezoidalSolver()
            result = solver.solve(function_str, a, b, n, tolerance, True)
            self.current_result = result
            
            self.results_display.display_results(
                function_str, a, b, result['n_values'][-1] if result['n_values'] else n,
                result['result'], result['error_estimate'], 
                result['exact_result'], result['iterations_info'],
                result.get('richardson_result')
            )
            
            # Строим график сходимости
            if result.get('n_values') and result.get('errors'):
                self.plot_widget.plot_convergence(
                    result['n_values'], result['errors'], result['exact_result']
                )
            
            self.tabs.setCurrentIndex(2)
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))
    
    def calculate_exact(self):
        try:
            function_str, a, b, _, _, _ = self.function_input.get_parameters()
            
            x = sp.symbols('x')
            expr = sp.sympify(function_str)
            exact_integral = sp.integrate(expr, (x, a, b))
            
            text = f"Точное значение интеграла:\n"
            text += f"∫[{a:.6f}, {b:.6f}] {function_str} dx = {exact_integral}\n"
            text += f"Численное значение: {float(exact_integral):.12f}"
            
            QMessageBox.information(self, "Точное значение", text)
                
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", "Не удалось вычислить точное значение интеграла")
    
    def clear(self):
        self.function_input.function_input.clear()
        self.function_input.a_input.clear()
        self.function_input.b_input.clear()
        self.function_input.n_input.setValue(10)
        self.results_display.text_edit.clear()
        self.table_widget.table.setRowCount(0)
        
        # Очищаем графики
        self.plot_widget.figure.clear()
        self.plot_widget.canvas.draw()
        
        self.current_result = None
    
    def export_results(self):
        try:
            if not self.current_result:
                QMessageBox.warning(self, "Предупреждение", "Нет данных для экспорта")
                return
                
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Сохранить результаты", "trapezoidal_integration.txt", "Текстовые файлы (*.txt);;Все файлы (*)"
            )
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.results_display.text_edit.toPlainText())
                    f.write("\n\n=== ДАННЫЕ ТРАПЕЦИЙ ===\n")
                    
                    if self.current_result.get('table_data'):
                        f.write("i\tx_i\t\tx_{i+1}\t\tf(x_i)\t\tf(x_{i+1})\tПлощадь\n")
                        for i, row in enumerate(self.current_result['table_data']):
                            f.write(f"{i}\t{row[0]:.6f}\t{row[1]:.6f}\t{row[2]:.6f}\t{row[3]:.6f}\t{row[4]:.6f}\n")
                
                QMessageBox.information(self, "Успех", "Результаты успешно сохранены!")
                
        except Exception as e:
            QMessageBox.critical(self, "Ошибка экспорта", f"Ошибка при экспорте результатов: {str(e)}")

def main():
    app = QApplication(sys.argv)
    calculator = TrapezoidalCalculator()
    calculator.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()