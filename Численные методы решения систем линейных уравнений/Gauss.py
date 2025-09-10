import numpy as np
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D

class GaussianEliminationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Метод Гаусса для решения систем линейных уравнений")
        self.root.geometry("900x900")
        
        style = ttk.Style()
        style.configure("TLabel", font=("Arial", 10))
        style.configure("TButton", font=("Arial", 10))
        style.configure("TEntry", font=("Arial", 10))
        
        # Вкладки для организации
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill='both')
        
        # Вкладка ввода
        self.input_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.input_tab, text="Ввод данных")
        
        # Вкладка решения
        self.solution_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.solution_tab, text="Решение")
        
        # Вкладка графика
        self.graph_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.graph_tab, text="График")
        
        # Вкладка ввода
        ttk.Label(self.input_tab, text="Количество уравнений/переменных (n):").grid(row=0, column=0, padx=10, pady=10)
        self.n_entry = ttk.Entry(self.input_tab)
        self.n_entry.grid(row=0, column=1, padx=10, pady=10)
        
        self.generate_matrix_button = ttk.Button(self.input_tab, text="Создать матрицу", command=self.generate_inputs)
        self.generate_matrix_button.grid(row=0, column=2, padx=10, pady=10)
        
        self.random_button = ttk.Button(self.input_tab, text="Генерировать случайную", command=self.generate_random)
        self.random_button.grid(row=0, column=3, padx=10, pady=10)
        
        self.import_button = ttk.Button(self.input_tab, text="Импорт из TXT", command=self.import_from_txt)
        self.import_button.grid(row=0, column=4, padx=10, pady=10)
        
        self.input_frame = ttk.Frame(self.input_tab)
        self.input_frame.grid(row=1, column=0, columnspan=5, padx=10, pady=10)
        
        self.solve_button = ttk.Button(self.input_tab, text="Решить систему", command=self.solve_system, state=tk.DISABLED)
        self.solve_button.grid(row=2, column=1, padx=10, pady=10)
        
        self.export_button = ttk.Button(self.input_tab, text="Экспорт в TXT", command=self.export_to_txt, state=tk.DISABLED)
        self.export_button.grid(row=2, column=2, padx=10, pady=10)
        
        self.clear_button = ttk.Button(self.input_tab, text="Очистить", command=self.clear_inputs)
        self.clear_button.grid(row=2, column=3, padx=10, pady=10)
        
        # Инструкции
        ttk.Label(self.input_tab, text="Введите коэффициенты. Для n>3 график - бар-чарт решений. Случайная матрица: невырожденная с числами [-10,10].").grid(row=3, column=0, columnspan=5, padx=10, pady=10)
        
        # --- Вкладка решения ---
        self.steps_text = scrolledtext.ScrolledText(self.solution_tab, width=80, height=35, font=("Courier", 10))
        self.steps_text.pack(padx=10, pady=10, expand=True, fill='both')
        
        #Вкладка графика
        self.graph_canvas = None
        
        self.entries_A = []
        self.entries_b = []
        self.n = 0
        self.A = None
        self.b = None

    def generate_inputs(self):
        try:
            self.n = int(self.n_entry.get())
            if self.n < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Ошибка", "Введите положительное целое число для n.")
            return
        
        self._create_input_fields()
        self.solve_button.config(state=tk.NORMAL)
        self.export_button.config(state=tk.NORMAL)

    def _create_input_fields(self, fill_values=False):
        # Очистка предыдущих полей
        for widget in self.input_frame.winfo_children():
            widget.destroy()
        self.entries_A = []
        self.entries_b = []
        
        # Заголовки столбцов
        for j in range(self.n):
            ttk.Label(self.input_frame, text=f"x{j+1}").grid(row=0, column=j, padx=5, pady=5)
        ttk.Label(self.input_frame, text="b").grid(row=0, column=self.n, padx=5, pady=5)
        
        # Поля ввода
        for i in range(self.n):
            ttk.Label(self.input_frame, text=f"Ур. {i+1}:").grid(row=i+1, column=self.n+1, padx=5, pady=5)
            row_entries = []
            for j in range(self.n):
                entry = ttk.Entry(self.input_frame, width=8)
                entry.grid(row=i+1, column=j, padx=5, pady=5)
                if fill_values and self.A is not None:
                    entry.insert(0, str(self.A[i, j]))
                row_entries.append(entry)
            self.entries_A.append(row_entries)
            
            b_entry = ttk.Entry(self.input_frame, width=8)
            b_entry.grid(row=i+1, column=self.n, padx=5, pady=5)
            if fill_values and self.b is not None:
                b_entry.insert(0, str(self.b[i]))
            self.entries_b.append(b_entry)

    def generate_random(self):
        try:
            self.n = int(self.n_entry.get())
            if self.n < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Ошибка", "Введите положительное целое число для n.")
            return
        
        # Генерация невырожденной матрицы
        while True:
            self.A = np.random.uniform(-10, 10, (self.n, self.n))
            self.b = np.random.uniform(-10, 10, self.n)
            if np.linalg.det(self.A) != 0:
                break
        
        self._create_input_fields(fill_values=True)
        self.solve_button.config(state=tk.NORMAL)
        self.export_button.config(state=tk.NORMAL)

    def import_from_txt(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if not file_path:
            return
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                self.n = int(lines[0].strip())
                self.A = np.zeros((self.n, self.n))
                self.b = np.zeros(self.n)
                for i, line in enumerate(lines[1:]):
                    values = list(map(float, line.strip().split()))
                    self.A[i] = values[:-1]
                    self.b[i] = values[-1]
            
            self.n_entry.delete(0, tk.END)
            self.n_entry.insert(0, str(self.n))
            self._create_input_fields(fill_values=True)
            self.solve_button.config(state=tk.NORMAL)
            self.export_button.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Ошибка импорта", f"Неверный формат файла: {str(e)}")

    def export_to_txt(self):
        if not self._collect_matrices():
            return
        
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if not file_path:
            return
        
        try:
            with open(file_path, 'w') as f:
                f.write(f"{self.n}\n")
                for i in range(self.n):
                    row = " ".join([str(self.A[i, j]) for j in range(self.n)]) + " " + str(self.b[i]) + "\n"
                    f.write(row)
            messagebox.showinfo("Экспорт", "Система экспортирована успешно.")
        except Exception as e:
            messagebox.showerror("Ошибка экспорта", str(e))

    def clear_inputs(self):
        self.n_entry.delete(0, tk.END)
        for widget in self.input_frame.winfo_children():
            widget.destroy()
        self.steps_text.delete(1.0, tk.END)
        self.clear_graph()
        self.solve_button.config(state=tk.DISABLED)
        self.export_button.config(state=tk.DISABLED)
        self.A = None
        self.b = None
        self.n = 0

    def _collect_matrices(self):
        try:
            self.A = np.zeros((self.n, self.n))
            self.b = np.zeros(self.n)
            for i in range(self.n):
                for j in range(self.n):
                    self.A[i, j] = float(self.entries_A[i][j].get())
                self.b[i] = float(self.entries_b[i].get())
            return True
        except ValueError:
            messagebox.showerror("Ошибка", "Все поля должны содержать числа.")
            return False

    def solve_system(self):
        if not self._collect_matrices():
            return
        
        try:
            steps, x = self.gaussian_elimination(self.A.copy(), self.b.copy())
            
            self.steps_text.delete(1.0, tk.END)
            for step in steps:
                self.steps_text.insert(tk.END, step + "\n\n")
            self.clear_graph()
            self.plot_graph(x)
            self.notebook.select(1)
        except np.linalg.LinAlgError as e:
            messagebox.showerror("Ошибка", str(e))

    def gaussian_elimination(self, A, b):
        steps = []
        n = len(b)
        Ab = np.hstack([A, b.reshape(-1, 1)])
        steps.append("Исходная расширенная матрица:\n" + self.pretty_matrix(Ab))
        
        for i in range(n):
            if Ab[i, i] == 0:
                for k in range(i+1, n):
                    if Ab[k, i] != 0:
                        Ab[[i, k]] = Ab[[k, i]]
                        steps.append(f"Перестановка строк {i+1} и {k+1}:\n" + self.pretty_matrix(Ab))
                        break
                else:
                    raise np.linalg.LinAlgError("Матрица вырождена: нет уникального решения.")
            
            pivot = Ab[i, i]
            Ab[i] /= pivot
            steps.append(f"Нормализация строки {i+1} (деление на {pivot:.4f}):\n" + self.pretty_matrix(Ab))
            
            for k in range(i+1, n):
                factor = Ab[k, i]
                Ab[k] -= factor * Ab[i]
                steps.append(f"Элиминация в строке {k+1} (вычитание {factor:.4f} * строка {i+1}):\n" + self.pretty_matrix(Ab))
        
        for i in range(n-1, 0, -1):
            for k in range(i-1, -1, -1):
                factor = Ab[k, i]
                Ab[k] -= factor * Ab[i]
                steps.append(f"Обратный ход: элиминация в строке {k+1} (вычитание {factor:.4f} * строка {i+1}):\n" + self.pretty_matrix(Ab))
        
        x = Ab[:, -1]
        steps.append("Решение:\n" + "\n".join([f"x{i+1} = {x[i]:.6f}" for i in range(n)]))
        return steps, x

    def pretty_matrix(self, mat):
        return "\n".join([" ".join([f"{val:8.4f}" for val in row]) for row in mat])

    def plot_graph(self, x):
        fig = plt.figure(figsize=(7, 7))
        
        if self.n == 2:
            ax = fig.add_subplot(111)
            x_vals = np.linspace(min(x[0]-10, -10), max(x[0]+10, 10), 400)
            for i in range(2):
                if self.A[i, 1] != 0:
                    y_vals = (self.b[i] - self.A[i, 0] * x_vals) / self.A[i, 1]
                    ax.plot(x_vals, y_vals, label=f"Ур. {i+1}: {self.A[i,0]:.2f}x + {self.A[i,1]:.2f}y = {self.b[i]:.2f}")
                else:
                    ax.axvline(x=self.b[i]/self.A[i,0] if self.A[i,0] != 0 else 0, label=f"Ур. {i+1}")
            ax.scatter(x[0], x[1], color='red', s=50, label=f"Решение: ({x[0]:.2f}, {x[1]:.2f})")
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.legend()
            ax.grid(True)
        
        elif self.n == 3:
            ax = fig.add_subplot(111, projection='3d')
            xx, yy = np.meshgrid(np.linspace(-10, 10, 10), np.linspace(-10, 10, 10))
            for i in range(3):
                if self.A[i, 2] != 0:
                    zz = (self.b[i] - self.A[i, 0]*xx - self.A[i, 1]*yy) / self.A[i, 2]
                    ax.plot_surface(xx, yy, zz, alpha=0.5, label=f"Плоскость {i+1}")
            ax.scatter(x[0], x[1], x[2], color='red', s=50, label=f"Решение: ({x[0]:.2f}, {x[1]:.2f}, {x[2]:.2f})")
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_zlabel('x3')
            ax.legend()
        
        else:
            ax = fig.add_subplot(111)
            vars = [f'x{i+1}' for i in range(self.n)]
            ax.bar(vars, x)
            ax.set_ylabel('Значения')
            ax.set_title('Значения переменных в решении')
            ax.grid(True)
        self.graph_canvas = FigureCanvasTkAgg(fig, master=self.graph_tab)
        self.graph_canvas.draw()
        self.graph_canvas.get_tk_widget().pack(expand=True, fill='both')

    def clear_graph(self):
        if self.graph_canvas:
            self.graph_canvas.get_tk_widget().destroy()
            self.graph_canvas = None

if __name__ == "__main__":
    root = tk.Tk()
    app = GaussianEliminationApp(root)
    root.mainloop()