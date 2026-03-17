import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
import time
import threading

def fitness(grid):
    n_rows, n_cols = grid.shape
    score = 0
    for i in range(n_rows):
        for j in range(n_cols - 1):
            if grid[i, j] != grid[i, j+1]:
                score += 1
    for i in range(n_rows - 1):
        for j in range(n_cols):
            if grid[i, j] != grid[i+1, j]:
                score += 1
    return score

def create_checkerboard(N):
    grid = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            grid[i, j] = (i + j) % 2
    return grid

def random_grid(N):
    return np.random.randint(0, 2, size=(N, N))

def genetic_algorithm(N, pop_size=50, generations=100, mutation_rate=0.05,
                      crossover_rate=0.8, tournament_size=3):
    start_time = time.perf_counter()
    population = [random_grid(N) for _ in range(pop_size)]
    best_fitness_per_gen = []

    for gen in range(generations):
        fitnesses = [fitness(ind) for ind in population]
        best_fitness = max(fitnesses)
        best_fitness_per_gen.append(best_fitness)

        new_population = []
        while len(new_population) < pop_size:
            # Selection (tournament)
            idx = np.random.choice(len(population), tournament_size, replace=False)
            parent1 = population[idx[np.argmax([fitnesses[i] for i in idx])]].copy()
            idx = np.random.choice(len(population), tournament_size, replace=False)
            parent2 = population[idx[np.argmax([fitnesses[i] for i in idx])]].copy()

            # Crossover
            if np.random.rand() < crossover_rate:
                p1 = parent1.flatten()
                p2 = parent2.flatten()
                point = np.random.randint(1, len(p1))
                c1 = np.concatenate([p1[:point], p2[point:]])
                c2 = np.concatenate([p2[:point], p1[point:]])
                child1 = c1.reshape(parent1.shape)
                child2 = c2.reshape(parent2.shape)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Mutation
            for child in (child1, child2):
                flat = child.flatten()
                for i in range(len(flat)):
                    if np.random.rand() < mutation_rate:
                        flat[i] = 1 - flat[i]
                # No need to reshape back because we modify flat in place, but we must assign
                child[:] = flat.reshape(child.shape)

            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        population = new_population

    fitnesses = [fitness(ind) for ind in population]
    best_idx = np.argmax(fitnesses)
    best_grid = population[best_idx]
    best_fit = fitnesses[best_idx]
    elapsed = time.perf_counter() - start_time
    return best_grid, best_fit, elapsed, best_fitness_per_gen

def hill_climbing(N, max_iters=5000, restarts=5):
    start_time = time.perf_counter()
    best_global_grid = None
    best_global_fit = -1

    for _ in range(restarts):
        current = random_grid(N)
        current_fit = fitness(current)

        for _ in range(max_iters):
            i = np.random.randint(N)
            j = np.random.randint(N)
            candidate = current.copy()
            candidate[i, j] = 1 - candidate[i, j]
            cand_fit = fitness(candidate)
            if cand_fit > current_fit:
                current = candidate
                current_fit = cand_fit
            max_possible = 2 * N * (N-1)
            if current_fit == max_possible:
                break

        if current_fit > best_global_fit:
            best_global_grid = current
            best_global_fit = current_fit

    elapsed = time.perf_counter() - start_time
    return best_global_grid, best_global_fit, elapsed, None

def random_search(N, iterations=10000):
    start_time = time.perf_counter()
    best_grid = None
    best_fit = -1
    for _ in range(iterations):
        cand = random_grid(N)
        cand_fit = fitness(cand)
        if cand_fit > best_fit:
            best_fit = cand_fit
            best_grid = cand
    elapsed = time.perf_counter() - start_time
    return best_grid, best_fit, elapsed, None

class MetasurfaceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("1‑bit Coding Metasurface Designer")
        self.root.geometry("1200x700")
        self.root.resizable(True, True)

        # Control frame (left side)
        control_frame = ttk.LabelFrame(root, text="Settings", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Grid size
        ttk.Label(control_frame, text="Grid size N:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.size_var = tk.IntVar(value=8)
        size_spin = ttk.Spinbox(control_frame, from_=2, to=20, textvariable=self.size_var, width=5)
        size_spin.grid(row=0, column=1, sticky=tk.W, pady=5)

        # Algorithm choice
        ttk.Label(control_frame, text="Algorithm:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.algo_var = tk.StringVar(value="GA")
        algo_frame = ttk.Frame(control_frame)
        algo_frame.grid(row=1, column=1, sticky=tk.W, pady=5)
        ttk.Radiobutton(algo_frame, text="Genetic Algorithm (GA)", variable=self.algo_var,
                        value="GA").pack(anchor=tk.W)
        ttk.Radiobutton(algo_frame, text="Hill Climbing (HC)", variable=self.algo_var,
                        value="HC").pack(anchor=tk.W)
        ttk.Radiobutton(algo_frame, text="Random Search (RS)", variable=self.algo_var,
                        value="RS").pack(anchor=tk.W)

        # Run button
        self.run_btn = ttk.Button(control_frame, text="Run Optimisation", command=self.run_optimisation)
        self.run_btn.grid(row=2, column=0, columnspan=2, pady=10)

        # Exit button
        ttk.Button(control_frame, text="Exit", command=root.quit).grid(row=3, column=0, columnspan=2, pady=5)

        # Results display (text)
        ttk.Label(control_frame, text="Results:", font=('Arial', 10, 'bold')).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(10,0))
        self.result_text = tk.Text(control_frame, height=6, width=30, state=tk.DISABLED)
        self.result_text.grid(row=5, column=0, columnspan=2, pady=5)

        # Progress / log
        ttk.Label(control_frame, text="Log:", font=('Arial', 10, 'bold')).grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=(10,0))
        self.log_text = tk.Text(control_frame, height=8, width=30, state=tk.DISABLED)
        self.log_text.grid(row=7, column=0, columnspan=2, pady=5)

        # Plot frame (right side)
        plot_frame = ttk.LabelFrame(root, text="Visualisation", padding=10)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create matplotlib figure with three subplots
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(10, 4))
        self.fig.tight_layout(pad=4.0)

        # Embed figure in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initialise plots with empty data
        self.ax1.set_title("Best Design")
        self.ax1.set_xticks([])
        self.ax1.set_yticks([])
        self.ax2.set_title("Checkerboard Baseline")
        self.ax2.set_xticks([])
        self.ax2.set_yticks([])
        self.ax3.set_title("Convergence (if available)")
        self.ax3.set_xlabel("Generation")
        self.ax3.set_ylabel("Best Fitness")
        self.canvas.draw()

        # For thread safety
        self.running = False

    def log(self, message):
        """Append message to log text widget."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.root.update_idletasks()

    def set_result(self, text):
        """Set result text widget."""
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, text)
        self.result_text.config(state=tk.DISABLED)

    def run_optimisation(self):
        if self.running:
            messagebox.showinfo("Info", "Already running an optimisation. Please wait.")
            return

        # Get parameters
        try:
            N = self.size_var.get()
            if N < 2:
                raise ValueError
        except:
            messagebox.showerror("Error", "Grid size must be an integer >= 2.")
            return

        algo = self.algo_var.get()
        self.log(f"Starting {algo} with N={N}...")

        # Disable run button during execution
        self.run_btn.config(state=tk.DISABLED)
        self.running = True

        # Run algorithm in a separate thread to keep GUI responsive
        thread = threading.Thread(target=self._run_algorithm, args=(N, algo))
        thread.daemon = True
        thread.start()

    def _run_algorithm(self, N, algo):
        try:
            if algo == "GA":
                best_grid, best_fit, elapsed, history = genetic_algorithm(N)
            elif algo == "HC":
                best_grid, best_fit, elapsed, history = hill_climbing(N)
            else:  # RS
                best_grid, best_fit, elapsed, history = random_search(N)

            # Compute checkerboard baseline
            checker = create_checkerboard(N)
            checker_fit = fitness(checker)
            max_possible = 2 * N * (N-1)

            # Prepare result text
            result = f"Algorithm: {algo}\nGrid size: {N}x{N}\nBest fitness: {best_fit} / {max_possible}\nCheckerboard: {checker_fit}\nTime: {elapsed:.4f} s\n"
            if best_fit == max_possible:
                result += "✓ Global optimum reached!"
            else:
                result += f"✗ Gap to optimum: {max_possible - best_fit}"

            # Update plots (must be done in main thread)
            self.root.after(0, self._update_plots, best_grid, checker, history, result)

        except Exception as e:
            self.root.after(0, messagebox.showerror, "Error", str(e))
        finally:
            self.running = False
            self.root.after(0, lambda: self.run_btn.config(state=tk.NORMAL))

    def _update_plots(self, best_grid, checker, history, result_text):
        # Clear axes
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        # Plot best design
        im1 = self.ax1.imshow(best_grid, cmap='coolwarm', interpolation='none', vmin=0, vmax=1)
        self.ax1.set_title("Best Design")
        self.ax1.set_xticks([])
        self.ax1.set_yticks([])
        plt.colorbar(im1, ax=self.ax1, ticks=[0,1], label='Phase Bit')

        # Plot checkerboard
        im2 = self.ax2.imshow(checker, cmap='coolwarm', interpolation='none', vmin=0, vmax=1)
        self.ax2.set_title("Checkerboard Baseline")
        self.ax2.set_xticks([])
        self.ax2.set_yticks([])
        plt.colorbar(im2, ax=self.ax2, ticks=[0,1], label='Phase Bit')

        # Plot convergence if available
        if history is not None:
            self.ax3.plot(history, linewidth=2)
            self.ax3.set_title("Convergence")
            self.ax3.set_xlabel("Generation")
            self.ax3.set_ylabel("Best Fitness")
            self.ax3.grid(True)
        else:
            self.ax3.text(0.5, 0.5, "No convergence data\nfor this algorithm",
                          ha='center', va='center', transform=self.ax3.transAxes)
            self.ax3.set_xticks([])
            self.ax3.set_yticks([])

        self.fig.tight_layout(pad=4.0)
        self.canvas.draw()

        # Update result text
        self.set_result(result_text)
        self.log("Optimisation finished.")

if __name__ == "__main__":
    root = tk.Tk()
    app = MetasurfaceGUI(root)
    root.mainloop()
