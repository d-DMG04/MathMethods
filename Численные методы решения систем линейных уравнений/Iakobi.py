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
                       convergence_info, method_name="Якоби"):
        text = f"=== МЕТОД {method_name.upper()} ===\n\n"
        
        text += "Исходная матрица A:\n"
        text += self.matrix_to_str(A) + "\n"
        
        text += "Вектор b:\n"
        text += " ".join([f"{x:8.4f}" for x in b]) + "\n\n"
        
        text += f"Количество итераций: {iterations}\n"
        text += f"Сходимость: {convergence_info}\n\n"
        
        text += "Решение системы:\n"
        for i, x in enumerate(solution):
            text += f"x{i+1} = {x:.8f}\n"
        
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
        
        self.text_edit.setPlainText(text)
    
    def matrix_to_str(self, matrix):
        text = ""
        for row in matrix:
            text += " ".join([f"{x:8.4f}" for x in row]) + "\n"
        return text

class MatrixGeneratorDialog(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent, Qt.Window)
        self.parent = parent
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Генератор матриц")
        self.setGeometry(200, 200, 400, 350)
        
        layout = QVBoxLayout()
        
        # Размер матрицы
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Размер матрицы:"))
        self.size_spinbox = QSpinBox()
        self.size_spinbox.setMinimum(2)
        self.size_spinbox.setMaximum(20)
        self.size_spinbox.setValue(3)
        size_layout.addWidget(self.size_spinbox)
        layout.addLayout(size_layout)
        
        # Тип матрицы
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Тип матрицы:"))
        self.matrix_type = QComboBox()
        self.matrix_type.addItems(["Случайная", "Диагонально-преобладающая", 
                                  "Симметричная", "Тёплицева", "Трёхдиагональная"])
        type_layout.addWidget(self.matrix_type)
        layout.addLayout(type_layout)
        
        # Диапазон значений
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Диапазон значений:"))
        self.min_val = QDoubleSpinBox()
        self.min_val.setRange(-1000, 1000)
        self.min_val.setValue(-10)
        self.max_val = QDoubleSpinBox()
        self.max_val.setRange(-1000, 1000)
        self.max_val.setValue(10)
        range_layout.addWidget(QLabel("от"))
        range_layout.addWidget(self.min_val)
        range_layout.addWidget(QLabel("до"))
        range_layout.addWidget(self.max_val)
        layout.addLayout(range_layout)
        
        # Параметр диагонального преобладания
        self.diag_dominance = QCheckBox("Усиленное диагональное преобладание")
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
            strong_diag = self.diag_dominance.isChecked()
            
            A = self.generate_matrix_type(n, min_val, max_val, matrix_type, strong_diag)
            
            # Генерируем вектор b
            b = np.random.uniform(min_val, max_val, n)
            
            if self.parent:
                self.parent.matrix_input.set_matrix(A, b)
            
            self.close()
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))
    
    def generate_matrix_type(self, n, min_val, max_val, matrix_type, strong_diag=True):
        if matrix_type == "Случайная":
            return np.random.uniform(min_val, max_val, (n, n))
        
        elif matrix_type == "Диагонально-преобладающая":
            A = np.random.uniform(min_val, max_val, (n, n))
            # Делаем матрицу диагонально преобладающей
            for i in range(n):
                row_sum = np.sum(np.abs(A[i])) - np.abs(A[i, i])
                if strong_diag:
                    A[i, i] = row_sum + random.uniform(5, 10)  # Сильное преобладание
                else:
                    A[i, i] = row_sum + random.uniform(1, 3)   # Слабое преобладание
            return A
        
        elif matrix_type == "Симметричная":
            A = np.random.uniform(min_val, max_val, (n, n))
            return (A + A.T) / 2  # Делаем симметричной
        
        elif matrix_type == "Тёплицева":
            # Тёплицева матрица имеет постоянные диагонали
            A = np.zeros((n, n))
            diag_values = np.random.uniform(min_val, max_val, 2*n-1)
            for i in range(n):
                for j in range(n):
                    A[i, j] = diag_values[n-1 + j - i]
            return A
        
        elif matrix_type == "Трёхдиагональная":
            # Трёхдиагональная матрица
            A = np.zeros((n, n))
            for i in range(n):
                if i > 0:
                    A[i, i-1] = random.uniform(min_val, max_val)  # Нижняя диагональ
                A[i, i] = random.uniform(min_val + 5, max_val + 5)  # Главная диагональ (усиленная)
                if i < n-1:
                    A[i, i+1] = random.uniform(min_val, max_val)  # Верхняя диагональ
            return A
        
        return np.random.uniform(min_val, max_val, (n, n))

class JacobiSolver:
    @staticmethod
    def solve(A, b, tolerance=1e-6, max_iterations=1000):
        """
        Решение системы Ax = b методом Якоби
        """
        n = len(A)
        x = np.zeros(n)  # Начальное приближение
        x_new = np.zeros(n)
        residuals_history = []
        
        # Проверка сходимости (диагональное преобладание)
        convergence = True
        for i in range(n):
            row_sum = np.sum(np.abs(A[i])) - np.abs(A[i, i])
            if np.abs(A[i, i]) <= row_sum:
                convergence = False
                break
        
        convergence_info = "Сходится (диагональное преобладание)" if convergence else "Может не сходиться"
        
        # Итерационный процесс
        for iteration in range(max_iterations):
            for i in range(n):
                sigma = 0
                for j in range(n):
                    if j != i:
                        sigma += A[i, j] * x[j]
                x_new[i] = (b[i] - sigma) / A[i, i]
            
            # Вычисляем невязку
            residual = np.linalg.norm(A @ x_new - b)
            residuals_history.append(residual)
            
            # Проверка условия остановки
            if residual < tolerance:
                break
            
            x = x_new.copy()
        
        return x_new, iteration + 1, residuals_history, convergence_info
    
    @staticmethod
    def check_convergence(A):
        """
        Проверка условий сходимости метода Якоби
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
        
        # 2. Проверка спектрального радиуса
        D = np.diag(np.diag(A))
        D_inv = np.linalg.inv(D)
        R = A - D
        T = -D_inv @ R
        spectral_radius = np.max(np.abs(np.linalg.eigvals(T)))
        
        return {
            'strict_diagonal_dominance': strict_diag_dom,
            'weak_diagonal_dominance': weak_diag_dom,
            'spectral_radius': spectral_radius,
            'converges': spectral_radius < 1
        }

class JacobiCalculator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Калькулятор метода Якоби")
        self.setGeometry(100, 100, 1200, 800)
        
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
        
        params_layout.addWidget(QLabel("Начальное приближение:"), 2, 0)
        self.initial_guess = QComboBox()
        self.initial_guess.addItems(["Нулевой вектор", "Случайный вектор", "Вектор b"])
        self.initial_guess.setMaximumWidth(100)
        params_layout.addWidget(self.initial_guess, 2, 1)
        
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
        
        # Правая панель - результаты
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        self.results_display = MatrixDisplayWidget()
        right_layout.addWidget(self.results_display)
        right_panel.setLayout(right_layout)
        
        # Разделитель
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 800])
        
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
            
            # Проверка на диагональные элементы
            for i in range(len(A)):
                if A[i, i] == 0:
                    QMessageBox.warning(self, "Ошибка", 
                                      f"Нулевой диагональный элемент a[{i+1},{i+1}]! Метод Якоби не применим.")
                    return
            
            solver = JacobiSolver()
            solution, iterations, residuals_history, convergence_info = solver.solve(
                A, b, tolerance, max_iterations
            )
            
            if iterations >= max_iterations:
                QMessageBox.warning(self, "Предупреждение", 
                                  f"Достигнуто максимальное количество итераций ({max_iterations}). Решение может быть неточным.")
            
            self.results_display.display_results(
                A, b, solution, iterations, residuals_history, convergence_info, "Якоби"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))
    
    def check_convergence(self):
        try:
            A, b = self.matrix_input.get_matrix()
            solver = JacobiSolver()
            convergence_info = solver.check_convergence(A)
            
            text = "=== АНАЛИЗ СХОДИМОСТИ МЕТОДА ЯКОБИ ===\n\n"
            
            text += "Строгое диагональное преобладание: "
            text += "Да" if convergence_info['strict_diagonal_dominance'] else "Нет"
            text += "\n"
            
            text += "Нестрогое диагональное преобладание: "
            text += "Да" if convergence_info['weak_diagonal_dominance'] else "Нет"
            text += "\n"
            
            text += f"Спектральный радиус: {convergence_info['spectral_radius']:.6f}\n"
            text += "Сходимость гарантирована: "
            text += "Да" if convergence_info['converges'] else "Нет"
            text += "\n\n"
            
            if convergence_info['converges']:
                text += "✓ Метод Якоби сходится для данной матрицы\n"
            else:
                text += "⚠ Метод Якоби может не сходиться для данной матрицы\n"
                text += "Рекомендации:\n"
                text += "- Убедитесь в диагональном преобладании\n"
                text += "- Попробуйте переставить строки/столбцы\n"
                text += "- Рассмотрите другие методы (Гаусса-Зейделя, LU-разложение)\n"
            
            self.results_display.text_edit.setPlainText(text)
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))
    
    def clear(self):
        for row in self.matrix_input.matrix_inputs:
            for input_field in row:
                input_field.clear()
        self.results_display.text_edit.clear()
        self.tolerance.setText("1e-6")
        self.max_iter.setValue(1000)
    
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
                    # Матричный формат с квадратными скобками
                    lines = content.split('\n')
                    matrix_data = []
                    for line in lines:
                        if '[' in line and ']' in line:
                            # Извлекаем числа из строки типа [1 2 3]
                            numbers = line.replace('[', '').replace(']', '').split()
                            row = [float(x) for x in numbers if x.strip()]
                            matrix_data.append(row)
                else:
                    # Простой построчный формат
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
                
                # Устанавливаем размер и заполняем поля
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
                        # Записываем строку матрицы A
                        row_str = " ".join([f"{x:.6f}" for x in A[i]])
                        # Добавляем элемент вектора b
                        f.write(f"{row_str} {b[i]:.6f}\n")
                
                QMessageBox.information(self, "Успех", "Матрица успешно экспортирована!")
                
        except Exception as e:
            QMessageBox.critical(self, "Ошибка экспорта", f"Ошибка при экспорте матрицы: {str(e)}")
    
    def export_results(self):
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Сохранить результаты", "jacobi_results.txt", "Текстовые файлы (*.txt);;Все файлы (*)"
            )
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.results_display.text_edit.toPlainText())
                
                QMessageBox.information(self, "Успех", "Результаты успешно сохранены!")
                
        except Exception as e:
            QMessageBox.critical(self, "Ошибка экспорта", f"Ошибка при экспорте результатов: {str(e)}")

def main():
    app = QApplication(sys.argv)
    calculator = JacobiCalculator()
    calculator.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()