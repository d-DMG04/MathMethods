import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGridLayout, QTextEdit, QLineEdit,
                             QPushButton, QLabel, QSpinBox, QFileDialog, 
                             QMessageBox, QTabWidget, QSplitter, QGroupBox,
                             QComboBox, QDoubleSpinBox)
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
    
    def display_results(self, A, L, U, P, solution, residuals=None):
        text = "=== LU-РАЗЛОЖЕНИЕ ===\n\n"
        
        text += "Исходная матрица A:\n"
        text += self.matrix_to_str(A) + "\n"
        
        if P is not None:
            text += "Матрица перестановок P:\n"
            text += self.matrix_to_str(P) + "\n"
        
        text += "Нижняя треугольная матрица L:\n"
        text += self.matrix_to_str(L) + "\n"
        
        text += "Верхняя треугольная матрица U:\n"
        text += self.matrix_to_str(U) + "\n"
        
        text += "\nРешение системы:\n"
        for i, x in enumerate(solution):
            text += f"x{i+1} = {x:.6f}\n"
        
        if residuals is not None:
            text += f"\nНевязка: {residuals:.2e}"
        
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
        self.setGeometry(200, 200, 400, 300)
        
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
        self.matrix_type.addItems(["Случайная", "Диагонально-преобладающая", "Симметричная", "Тёплицева"])
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
            
            A = self.generate_matrix_type(n, min_val, max_val, matrix_type)
            
            # Генерируем вектор b
            b = np.random.uniform(min_val, max_val, n)
            
            if self.parent:
                self.parent.matrix_input.set_matrix(A, b)
            
            self.close()
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))
    
    def generate_matrix_type(self, n, min_val, max_val, matrix_type):
        if matrix_type == "Случайная":
            return np.random.uniform(min_val, max_val, (n, n))
        
        elif matrix_type == "Диагонально-преобладающая":
            A = np.random.uniform(min_val, max_val, (n, n))
            # Делаем матрицу диагонально преобладающей
            for i in range(n):
                row_sum = np.sum(np.abs(A[i])) - np.abs(A[i, i])
                A[i, i] = row_sum + random.uniform(1, 5)  # Гарантируем преобладание
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
        
        return np.random.uniform(min_val, max_val, (n, n))

class LUSolver:
    @staticmethod
    def lu_decomposition(A):
        n = len(A)
        L = np.zeros((n, n))
        U = np.zeros((n, n))
        P = np.eye(n)
        
        # LU-разложение с частичным выбором
        for k in range(n):
            # Поиск максимального элемента в столбце
            max_row = np.argmax(np.abs(A[k:n, k])) + k
            if max_row != k:
                # Перестановка строк
                A[[k, max_row]] = A[[max_row, k]]
                P[[k, max_row]] = P[[max_row, k]]
                if k > 0:
                    L[[k, max_row], :k] = L[[max_row, k], :k]
            
            L[k, k] = 1.0
            for j in range(k, n):
                U[k, j] = A[k, j] - np.dot(L[k, :k], U[:k, j])
            for i in range(k+1, n):
                L[i, k] = (A[i, k] - np.dot(L[i, :k], U[:k, k])) / U[k, k]
        
        return L, U, P
    
    @staticmethod
    def solve_lu(L, U, b):
        # Решение Ly = b
        n = len(L)
        y = np.zeros(n)
        for i in range(n):
            y[i] = b[i] - np.dot(L[i, :i], y[:i])
        
        # Решение Ux = y
        x = np.zeros(n)
        for i in range(n-1, -1, -1):
            x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
        
        return x

class LUCalculator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Калькулятор LU-разложения")
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
        
        # Кнопки действий
        button_layout = QGridLayout()
        
        self.solve_btn = QPushButton("Решить")
        self.solve_btn.clicked.connect(self.solve)
        button_layout.addWidget(self.solve_btn, 0, 0)
        
        self.clear_btn = QPushButton("Очистить")
        self.clear_btn.clicked.connect(self.clear)
        button_layout.addWidget(self.clear_btn, 0, 1)
        
        self.generate_btn = QPushButton("Сгенерировать")
        self.generate_btn.clicked.connect(self.show_generator)
        button_layout.addWidget(self.generate_btn, 1, 0)
        
        self.import_btn = QPushButton("Импорт матрицы")
        self.import_btn.clicked.connect(self.import_matrix)
        button_layout.addWidget(self.import_btn, 1, 1)
        
        self.export_matrix_btn = QPushButton("Экспорт матрицы")
        self.export_matrix_btn.clicked.connect(self.export_matrix)
        button_layout.addWidget(self.export_matrix_btn, 2, 0)
        
        self.export_results_btn = QPushButton("Экспорт результатов")
        self.export_results_btn.clicked.connect(self.export_results)
        button_layout.addWidget(self.export_results_btn, 2, 1)
        
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
            
            # Проверка на вырожденность
            if np.linalg.det(A) == 0:
                QMessageBox.warning(self, "Ошибка", "Матрица вырождена!")
                return
            
            solver = LUSolver()
            L, U, P = solver.lu_decomposition(A.copy())
            
            # Применяем перестановки к вектору b
            b_permuted = P @ b
            
            solution = solver.solve_lu(L, U, b_permuted)
            
            # Вычисляем невязку
            residuals = np.linalg.norm(A @ solution - b)
            
            self.results_display.display_results(A, L, U, P, solution, residuals)
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))
    
    def clear(self):
        for row in self.matrix_input.matrix_inputs:
            for input_field in row:
                input_field.clear()
        self.results_display.text_edit.clear()
    
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
                self, "Сохранить результаты", "lu_results.txt", "Текстовые файлы (*.txt);;Все файлы (*)"
            )
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.results_display.text_edit.toPlainText())
                
                QMessageBox.information(self, "Успех", "Результаты успешно сохранены!")
                
        except Exception as e:
            QMessageBox.critical(self, "Ошибка экспорта", f"Ошибка при экспорте результатов: {str(e)}")

def main():
    app = QApplication(sys.argv)
    calculator = LUCalculator()
    calculator.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()