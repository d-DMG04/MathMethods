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

class TridiagonalInputWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.matrix_size = 3
        self.a_inputs = []  # нижняя диагональ (a1...an-1)
        self.b_inputs = []  # главная диагональ (b1...bn)
        self.c_inputs = []  # верхняя диагональ (c1...cn-1)
        self.d_inputs = []  # вектор правой части (d1...dn)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Размер матрицы
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Размер матрицы:"))
        self.size_spinbox = QSpinBox()
        self.size_spinbox.setMinimum(2)
        self.size_spinbox.setMaximum(100)
        self.size_spinbox.setValue(3)
        self.size_spinbox.valueChanged.connect(self.change_matrix_size)
        size_layout.addWidget(self.size_spinbox)
        size_layout.addStretch()
        layout.addLayout(size_layout)
        
        # Заголовки столбцов
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("a (нижняя)"))
        header_layout.addWidget(QLabel("b (главная)"))
        header_layout.addWidget(QLabel("c (верхняя)"))
        header_layout.addWidget(QLabel("d (правая часть)"))
        layout.addLayout(header_layout)
        
        # Поле для ввода трехдиагональной матрицы
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
        
        self.a_inputs = []
        self.b_inputs = []
        self.c_inputs = []
        self.d_inputs = []
        
        n = self.matrix_size
        
        for i in range(n):
            # Нижняя диагональ (a)
            a_input = QLineEdit()
            a_input.setMaximumWidth(60)
            if i > 0:
                a_input.setPlaceholderText(f"a{i+1}")
            else:
                a_input.setPlaceholderText("a1 (0)")
                a_input.setText("0")
                a_input.setReadOnly(True)
            self.a_inputs.append(a_input)
            self.matrix_layout.addWidget(a_input, i, 0)
            
            # Главная диагональ (b)
            b_input = QLineEdit()
            b_input.setMaximumWidth(60)
            b_input.setPlaceholderText(f"b{i+1}")
            self.b_inputs.append(b_input)
            self.matrix_layout.addWidget(b_input, i, 1)
            
            # Верхняя диагональ (c)
            c_input = QLineEdit()
            c_input.setMaximumWidth(60)
            if i < n - 1:
                c_input.setPlaceholderText(f"c{i+1}")
            else:
                c_input.setPlaceholderText("cn (0)")
                c_input.setText("0")
                c_input.setReadOnly(True)
            self.c_inputs.append(c_input)
            self.matrix_layout.addWidget(c_input, i, 2)
            
            # Правая часть (d)
            d_input = QLineEdit()
            d_input.setMaximumWidth(60)
            d_input.setPlaceholderText(f"d{i+1}")
            self.d_inputs.append(d_input)
            self.matrix_layout.addWidget(d_input, i, 3)
    
    def change_matrix_size(self, size):
        self.matrix_size = size
        self.create_matrix_inputs()
    
    def get_matrix(self):
        try:
            n = self.matrix_size
            a = np.zeros(n)
            b = np.zeros(n)
            c = np.zeros(n)
            d = np.zeros(n)
            
            # Чтение нижней диагонали (a)
            for i in range(1, n):  # a[0] всегда 0
                text = self.a_inputs[i].text().strip()
                if not text:
                    raise ValueError(f"Пустое поле a{i+1}")
                a[i] = float(text)
            
            # Чтение главной диагонали (b)
            for i in range(n):
                text = self.b_inputs[i].text().strip()
                if not text:
                    raise ValueError(f"Пустое поле b{i+1}")
                b[i] = float(text)
            
            # Чтение верхней диагонали (c)
            for i in range(n-1):  # c[n-1] всегда 0
                text = self.c_inputs[i].text().strip()
                if not text:
                    raise ValueError(f"Пустое поле c{i+1}")
                c[i] = float(text)
            
            # Чтение правой части (d)
            for i in range(n):
                text = self.d_inputs[i].text().strip()
                if not text:
                    raise ValueError(f"Пустое поле d{i+1}")
                d[i] = float(text)
            
            return a, b, c, d
            
        except ValueError as e:
            raise ValueError(f"Ошибка ввода данных: {str(e)}")
    
    def set_matrix(self, a, b, c, d):
        """Заполняет поля ввода трехдиагональной матрицей"""
        n = len(b)
        self.size_spinbox.setValue(n)
        self.create_matrix_inputs()
        
        # Устанавливаем значения
        for i in range(n):
            if i > 0:
                self.a_inputs[i].setText(f"{a[i]:.6f}")
            self.b_inputs[i].setText(f"{b[i]:.6f}")
            if i < n - 1:
                self.c_inputs[i].setText(f"{c[i]:.6f}")
            self.d_inputs[i].setText(f"{d[i]:.6f}")

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
    
    def display_results(self, a, b, c, d, solution, residuals=None, method_name="Прогонки"):
        text = f"=== МЕТОД {method_name.upper()} ===\n\n"
        
        text += "Трехдиагональная матрица:\n"
        text += self.tridiagonal_matrix_to_str(a, b, c, d) + "\n"
        
        text += "Решение системы:\n"
        for i, x in enumerate(solution):
            text += f"x{i+1} = {x:.8f}\n"
        
        if residuals is not None:
            text += f"\nНевязка: {residuals:.2e}\n"
        
        text += "\nПроверка решения (A*x - d):\n"
        n = len(solution)
        check = np.zeros(n)
        for i in range(n):
            if i == 0:
                check[i] = b[0] * solution[0] + c[0] * solution[1] - d[0]
            elif i == n-1:
                check[i] = a[n-1] * solution[n-2] + b[n-1] * solution[n-1] - d[n-1]
            else:
                check[i] = a[i] * solution[i-1] + b[i] * solution[i] + c[i] * solution[i+1] - d[i]
            text += f"Уравнение {i+1}: {check[i]:.2e}\n"
        
        self.text_edit.setPlainText(text)
    
    def tridiagonal_matrix_to_str(self, a, b, c, d):
        n = len(b)
        text = ""
        for i in range(n):
            row = []
            for j in range(n):
                if j == i-1:
                    row.append(f"{a[i]:8.4f}")
                elif j == i:
                    row.append(f"{b[i]:8.4f}")
                elif j == i+1:
                    row.append(f"{c[i]:8.4f}")
                else:
                    row.append("   0.0000")
            row.append(f" | {d[i]:8.4f}")
            text += " ".join(row) + "\n"
        return text

class PlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        
        layout.addWidget(QLabel("Визуализация решения:"))
        layout.addWidget(self.canvas)
        self.setLayout(layout)
    
    def plot_solution(self, solution, d, title="Решение системы"):
        self.figure.clear()
        
        ax = self.figure.add_subplot(111)
        n = len(solution)
        
        x = np.arange(1, n + 1)
        
        ax.plot(x, solution, 'bo-', linewidth=2, markersize=6, label='Решение x')
        ax.plot(x, d, 'r--', linewidth=1, markersize=4, label='Правая часть d (масштабированная)')
        
        ax.set_xlabel('Индекс переменной')
        ax.set_ylabel('Значение')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.figure.tight_layout()
        self.canvas.draw()

class MatrixGeneratorDialog(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent, Qt.Window)
        self.parent = parent
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Генератор трехдиагональных матриц")
        self.setGeometry(200, 200, 400, 350)
        
        layout = QVBoxLayout()
        
        # Размер матрицы
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Размер матрицы:"))
        self.size_spinbox = QSpinBox()
        self.size_spinbox.setMinimum(2)
        self.size_spinbox.setMaximum(100)
        self.size_spinbox.setValue(10)
        size_layout.addWidget(self.size_spinbox)
        layout.addLayout(size_layout)
        
        # Тип матрицы
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Тип матрицы:"))
        self.matrix_type = QComboBox()
        self.matrix_type.addItems(["С диагональным преобладанием", 
                                  "Симметричная", 
                                  "С постоянными коэффициентами",
                                  "Случайная"])
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
        
        # Параметр диагонального преобладания
        self.diag_dominance = QCheckBox("Строгое диагональное преобладание")
        self.diag_dominance.setChecked(True)
        layout.addWidget(self.diag_dominance)
        
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
            strict_diag = self.diag_dominance.isChecked()
            
            a, b, c, d = self.generate_matrix_type(n, min_val, max_val, matrix_type, strict_diag)
            
            if self.parent:
                self.parent.matrix_input.set_matrix(a, b, c, d)
            
            self.close()
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))
    
    def generate_matrix_type(self, n, min_val, max_val, matrix_type, strict_diag=True):
        a = np.zeros(n)  # нижняя диагональ
        b = np.zeros(n)  # главная диагональ
        c = np.zeros(n)  # верхняя диагональ
        d = np.zeros(n)  # правая часть
        
        if matrix_type == "С диагональным преобладанием":
            for i in range(n):
                if i > 0:
                    a[i] = random.uniform(0, min_val/2)
                if i < n-1:
                    c[i] = random.uniform(0, min_val/2)
                
                # Главная диагональ с преобладанием
                off_diag_sum = (a[i] if i > 0 else 0) + (c[i] if i < n-1 else 0)
                if strict_diag:
                    b[i] = off_diag_sum + random.uniform(min_val, max_val)
                else:
                    b[i] = off_diag_sum + random.uniform(0.1, max_val)
                
                d[i] = random.uniform(min_val, max_val)
                
        elif matrix_type == "Симметричная":
            for i in range(n):
                if i > 0:
                    a[i] = random.uniform(0, max_val/2)
                if i < n-1:
                    c[i] = a[i+1]  # Симметрия: c[i] = a[i+1]
                
                b[i] = random.uniform(min_val, max_val)
                d[i] = random.uniform(min_val, max_val)
                
        elif matrix_type == "С постоянными коэффициентами":
            a_val = random.uniform(0, min_val/2)
            b_val = random.uniform(min_val, max_val)
            c_val = random.uniform(0, min_val/2)
            
            for i in range(n):
                if i > 0:
                    a[i] = a_val
                if i < n-1:
                    c[i] = c_val
                b[i] = b_val
                d[i] = random.uniform(min_val, max_val)
                
        elif matrix_type == "Случайная":
            for i in range(n):
                if i > 0:
                    a[i] = random.uniform(0, max_val)
                if i < n-1:
                    c[i] = random.uniform(0, max_val)
                b[i] = random.uniform(min_val, max_val)
                d[i] = random.uniform(min_val, max_val)
        
        return a, b, c, d

class ThomasSolver:
    @staticmethod
    def solve(a, b, c, d):
        """
        Решение трехдиагональной системы методом прогонки (Томаса)
        """
        n = len(d)
        
        # Проверка условия применимости метода
        if not ThomasSolver.check_conditions(a, b, c):
            raise ValueError("Матрица не удовлетворяет условиям применимости метода прогонки")
        
        # Прямой ход прогонки
        alpha = np.zeros(n-1)
        beta = np.zeros(n-1)
        
        # Первое уравнение
        alpha[0] = -c[0] / b[0]
        beta[0] = d[0] / b[0]
        
        # Промежуточные уравнения
        for i in range(1, n-1):
            denominator = b[i] + a[i] * alpha[i-1]
            alpha[i] = -c[i] / denominator
            beta[i] = (d[i] - a[i] * beta[i-1]) / denominator
        
        # Обратный ход прогонки
        x = np.zeros(n)
        
        # Последнее уравнение
        x[n-1] = (d[n-1] - a[n-1] * beta[n-2]) / (b[n-1] + a[n-1] * alpha[n-2])
        
        # Обратная подстановка
        for i in range(n-2, -1, -1):
            x[i] = alpha[i] * x[i+1] + beta[i]
        
        # Вычисление невязки
        residual = ThomasSolver.calculate_residual(a, b, c, d, x)
        
        return x, residual
    
    @staticmethod
    def calculate_residual(a, b, c, d, x):
        """Вычисление невязки"""
        n = len(x)
        residual = 0
        for i in range(n):
            if i == 0:
                res_i = b[0] * x[0] + c[0] * x[1] - d[0]
            elif i == n-1:
                res_i = a[n-1] * x[n-2] + b[n-1] * x[n-1] - d[n-1]
            else:
                res_i = a[i] * x[i-1] + b[i] * x[i] + c[i] * x[i+1] - d[i]
            residual += res_i ** 2
        return np.sqrt(residual)
    
    @staticmethod
    def check_conditions(a, b, c):
        """
        Проверка условий применимости метода прогонки
        """
        n = len(b)
        
        # 1. Условие достаточное: диагональное преобладание
        diag_dom = True
        for i in range(n):
            diag = abs(b[i])
            off_diag = (abs(a[i]) if i > 0 else 0) + (abs(c[i]) if i < n-1 else 0)
            if diag <= off_diag:
                diag_dom = False
                break
        
        # 2. Проверка на вырожденность
        try:
            # Строим полную матрицу для проверки
            A = np.zeros((n, n))
            for i in range(n):
                if i > 0:
                    A[i, i-1] = a[i]
                A[i, i] = b[i]
                if i < n-1:
                    A[i, i+1] = c[i]
            
            det = np.linalg.det(A)
            if abs(det) < 1e-10:
                return False
                
        except:
            pass
        
        return diag_dom
    
    @staticmethod
    def analyze_matrix(a, b, c):
        """Анализ трехдиагональной матрицы"""
        n = len(b)
        
        analysis = {
            'diagonal_dominance': True,
            'strict_diagonal_dominance': True,
            'condition_number': None,
            'determinant': None
        }
        
        # Проверка диагонального преобладания
        for i in range(n):
            diag = abs(b[i])
            off_diag = (abs(a[i]) if i > 0 else 0) + (abs(c[i]) if i < n-1 else 0)
            
            if diag < off_diag:
                analysis['strict_diagonal_dominance'] = False
            if diag <= off_diag:
                analysis['diagonal_dominance'] = False
        
        # Вычисление числа обусловленности и определителя
        try:
            A = np.zeros((n, n))
            for i in range(n):
                if i > 0:
                    A[i, i-1] = a[i]
                A[i, i] = b[i]
                if i < n-1:
                    A[i, i+1] = c[i]
            
            analysis['condition_number'] = np.linalg.cond(A)
            analysis['determinant'] = np.linalg.det(A)
            
        except:
            analysis['condition_number'] = float('inf')
            analysis['determinant'] = 0
        
        return analysis

class ThomasCalculator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Калькулятор метода прогонки (Томаса)")
        self.setGeometry(100, 100, 1200, 800)
        
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Левая панель - ввод данных
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        
        input_group = QGroupBox("Ввод трехдиагональной матрицы")
        input_layout = QVBoxLayout()
        self.matrix_input = TridiagonalInputWidget()
        input_layout.addWidget(self.matrix_input)
        input_group.setLayout(input_layout)
        left_layout.addWidget(input_group)
        
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
        self.tabs.addTab(self.plot_widget, "График решения")
        
        right_layout.addWidget(self.tabs)
        right_panel.setLayout(right_layout)
        
        # Разделитель
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([500, 700])
        
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
            a, b, c, d = self.matrix_input.get_matrix()
            
            solver = ThomasSolver()
            solution, residual = solver.solve(a, b, c, d)
            
            self.results_display.display_results(
                a, b, c, d, solution, residual, "Прогонки"
            )
            
            # Построение графика решения
            self.plot_widget.plot_solution(solution, d/np.max(np.abs(d)), "Решение трехдиагональной системы")
            self.tabs.setCurrentIndex(1)  # Переключаемся на вкладку с графиками
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))
    
    def check_matrix(self):
        try:
            a, b, c, d = self.matrix_input.get_matrix()
            analysis = ThomasSolver.analyze_matrix(a, b, c)
            
            text = "=== АНАЛИЗ ТРЕХДИАГОНАЛЬНОЙ МАТРИЦЫ ===\n\n"
            
            text += "Диагональное преобладание: "
            text += "Да" if analysis['diagonal_dominance'] else "Нет"
            text += "\n"
            
            text += "Строгое диагональное преобладание: "
            text += "Да" if analysis['strict_diagonal_dominance'] else "Нет"
            text += "\n"
            
            if analysis['condition_number'] is not None:
                text += f"Число обусловленности: {analysis['condition_number']:.2e}\n"
            
            if analysis['determinant'] is not None:
                text += f"Определитель: {analysis['determinant']:.6e}\n"
            
            text += "\nПригодность для метода прогонки: "
            if analysis['strict_diagonal_dominance']:
                text += "✓ ОТЛИЧНО - метод гарантированно сходится\n"
            elif analysis['diagonal_dominance']:
                text += "✓ ХОРОШО - метод должен сойтись\n"
            else:
                text += "⚠ ВНИМАНИЕ - метод может не сойтись\n"
            
            text += "\nРекомендации:\n"
            if not analysis['diagonal_dominance']:
                text += "- Матрица не имеет диагонального преобладания\n"
                text += "- Рассмотрите возможность перестановки строк\n"
            
            if analysis['condition_number'] > 1e10:
                text += "- Матрица плохо обусловлена, возможны ошибки округления\n"
            
            self.results_display.text_edit.setPlainText(text)
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))
    
    def clear(self):
        n = self.matrix_input.matrix_size
        for i in range(n):
            if i > 0:
                self.matrix_input.a_inputs[i].clear()
            self.matrix_input.b_inputs[i].clear()
            if i < n - 1:
                self.matrix_input.c_inputs[i].clear()
            self.matrix_input.d_inputs[i].clear()
        
        self.results_display.text_edit.clear()
        
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
                
                lines = content.split('\n')
                matrix_data = []
                
                for line in lines:
                    if line.strip():
                        row = [float(x) for x in line.split()]
                        matrix_data.append(row)
                
                if not matrix_data:
                    raise ValueError("Файл пуст или содержит неверные данные")
                
                n = len(matrix_data)
                if len(matrix_data[0]) != 4:
                    raise ValueError("Неверный формат. Ожидается 4 столбца: a, b, c, d")
                
                a = np.zeros(n)
                b = np.zeros(n)
                c = np.zeros(n)
                d = np.zeros(n)
                
                for i in range(n):
                    a[i] = matrix_data[i][0]
                    b[i] = matrix_data[i][1]
                    c[i] = matrix_data[i][2]
                    d[i] = matrix_data[i][3]
                
                self.matrix_input.set_matrix(a, b, c, d)
                QMessageBox.information(self, "Успех", "Матрица успешно загружена!")
                
        except Exception as e:
            QMessageBox.critical(self, "Ошибка импорта", f"Ошибка при импорте: {str(e)}")
    
    def export_matrix(self):
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Сохранить матрицу", "tridiagonal_matrix.txt", "Текстовые файлы (*.txt);;Все файлы (*)"
            )
            
            if file_path:
                a, b, c, d = self.matrix_input.get_matrix()
                n = len(b)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    for i in range(n):
                        f.write(f"{a[i]:.6f} {b[i]:.6f} {c[i]:.6f} {d[i]:.6f}\n")
                
                QMessageBox.information(self, "Успех", "Матрица успешно экспортирована!")
                
        except Exception as e:
            QMessageBox.critical(self, "Ошибка экспорта", f"Ошибка при экспорте матрицы: {str(e)}")
    
    def export_results(self):
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Сохранить результаты", "thomas_results.txt", "Текстовые файлы (*.txt);;Все файлы (*)"
            )
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.results_display.text_edit.toPlainText())
                
                QMessageBox.information(self, "Успех", "Результаты успешно сохранены!")
                
        except Exception as e:
            QMessageBox.critical(self, "Ошибка экспорта", f"Ошибка при экспорте результатов: {str(e)}")

def main():
    app = QApplication(sys.argv)
    calculator = ThomasCalculator()
    calculator.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()