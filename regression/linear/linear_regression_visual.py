import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


class LinearRegressionVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Linear Regression Visualizer")
        self.root.geometry("1200x700")
        self.root.configure(bg="#1e1e1e")

        self.bg = "#1e1e1e"
        self.panel = "#252525"
        self.text = "#eeeeee"
        self.secondary = "#bdbdbd"
        self.accent = "#4caf50"
        self.point = "#5b9bd5"
        self.grid = "#383838"

        self.focus = "Residuals"

        self.x = np.array([
            0.7, 1.6, 2.6, 3.4, 4.3,
            5.2, 6.1, 7.2, 8.1, 9.0
        ])

        self.y = np.array([
            3.0, 1.4, 3.5, 2.8, 4.8,
            3.4, 5.9, 4.5, 6.7, 5.7
        ])

        self.dragging_point = None

        self.create_ui()
        self.fit_model()
        self.draw()

    def create_ui(self):
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        self.left = tk.Frame(
            self.root,
            bg=self.panel,
            padx=30,
            pady=25
        )

        self.left.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        self.right = tk.Frame(
            self.root,
            bg=self.panel
        )

        self.right.grid(
            row=0,
            column=1,
            sticky="nsew"
        )

        self.create_equation()
        self.create_tabs()
        self.create_button()
        self.create_plot()

    def create_equation(self):
        self.equation = tk.Label(
            self.left,
            text="y = b0 + b1x",
            font=("Times New Roman", 28, "italic"),
            fg=self.text,
            bg=self.panel
        )

        self.equation.pack(
            pady=(35, 25)
        )

        self.parameters = tk.Label(
            self.left,
            text="",
            font=("Times New Roman", 22),
            fg=self.text,
            bg=self.panel
        )

        self.parameters.pack(
            pady=(0, 20)
        )

        self.description = tk.Label(
            self.left,
            text="",
            font=("Times New Roman", 15),
            fg=self.secondary,
            bg=self.panel,
            justify="center",
            wraplength=500
        )

        self.description.pack()

    def create_tabs(self):
        self.tab_frame = tk.Frame(
            self.left,
            bg=self.panel
        )

        self.tab_frame.pack(
            fill="x",
            pady=(35, 15)
        )

        self.tabs = {}

        for name in [
            "Data",
            "Line",
            "Predictions",
            "Residuals"
        ]:
            button = tk.Button(
                self.tab_frame,
                text=name,
                font=("Arial", 13),
                bd=0,
                relief="flat",
                padx=15,
                pady=8,
                fg=self.text,
                bg="#414141",
                activeforeground=self.text,
                activebackground="#414141",
                command=lambda n=name: self.change_focus(n)
            )

            button.pack(
                side="left",
                expand=True
            )

            self.tabs[name] = button

    def create_button(self):
        self.generate_button = tk.Button(
            self.left,
            text="Generate new data",
            font=("Arial", 14),
            fg=self.text,
            bg=self.panel,
            activeforeground=self.text,
            activebackground="#333333",
            bd=1,
            relief="solid",
            command=self.generate_data
        )

        self.generate_button.pack(
            fill="x",
            pady=(15, 10),
            ipady=8
        )

    def create_plot(self):
        self.figure = Figure(
            figsize=(7, 6),
            dpi=100,
            facecolor=self.panel
        )

        self.ax = self.figure.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(
            self.figure,
            master=self.right
        )

        self.canvas.get_tk_widget().pack(
            fill="both",
            expand=True,
            padx=15,
            pady=15
        )

        self.canvas.mpl_connect(
            "button_press_event",
            self.on_press
        )

        self.canvas.mpl_connect(
            "button_release_event",
            self.on_release
        )

        self.canvas.mpl_connect(
            "motion_notify_event",
            self.on_motion
        )

    def fit_model(self):
        self.model = LinearRegression()

        X = self.x.reshape(-1, 1)

        self.model.fit(
            X,
            self.y
        )

        self.predictions = self.model.predict(X)

        self.slope = self.model.coef_[0]
        self.intercept = self.model.intercept_

        self.r2 = r2_score(
            self.y,
            self.predictions
        )

        self.residuals = self.y - self.predictions

    def draw(self):
        self.fit_model()

        self.ax.clear()

        self.ax.set_facecolor(self.panel)

        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)

        self.ax.grid(
            True,
            color=self.grid,
            linewidth=0.8,
            alpha=0.8
        )

        self.ax.tick_params(
            colors="#999999",
            labelsize=9
        )

        for spine in self.ax.spines.values():
            spine.set_color("#555555")

        line_x = np.linspace(0, 10, 200)
        line_y = (
            self.intercept
            + self.slope * line_x
        )

        if self.focus == "Data":
            self.draw_data()

        elif self.focus == "Line":
            self.draw_line(
                line_x,
                line_y
            )

        elif self.focus == "Predictions":
            self.draw_predictions()

        elif self.focus == "Residuals":
            self.draw_residuals()

        self.canvas.draw_idle()

        self.update_text()

    def draw_data(self):
        self.ax.scatter(
            self.x,
            self.y,
            s=70,
            facecolors=self.point,
            edgecolors="#3675ad",
            linewidths=2,
            zorder=5
        )

    def draw_line(self, line_x, line_y):
        self.ax.plot(
            line_x,
            line_y,
            color=self.accent,
            linewidth=2.5
        )

        self.ax.scatter(
            self.x,
            self.y,
            s=65,
            facecolors=self.point,
            edgecolors="#3675ad",
            linewidths=2
        )

    def draw_predictions(self):
        line_x = np.linspace(0, 10, 200)

        line_y = (
            self.intercept
            + self.slope * line_x
        )

        self.ax.plot(
            line_x,
            line_y,
            color=self.accent,
            linewidth=2.5
        )

        self.ax.scatter(
            self.x,
            self.y,
            s=60,
            facecolors=self.point,
            edgecolors="#3675ad",
            linewidths=2
        )

        self.ax.scatter(
            self.x,
            self.predictions,
            s=60,
            facecolors="none",
            edgecolors=self.accent,
            linewidths=2
        )

        for xi, yi, pi in zip(
            self.x,
            self.y,
            self.predictions
        ):
            self.ax.plot(
                [xi, xi],
                [yi, pi],
                linestyle="--",
                color="#777777",
                linewidth=1.5
            )

    def draw_residuals(self):
        line_x = np.linspace(0, 10, 200)

        line_y = (
            self.intercept
            + self.slope * line_x
        )

        self.ax.plot(
            line_x,
            line_y,
            color=self.accent,
            linewidth=2.5
        )

        self.ax.scatter(
            self.x,
            self.y,
            s=60,
            facecolors="none",
            edgecolors=self.accent,
            linewidths=2
        )

        for xi, yi, pi in zip(
            self.x,
            self.y,
            self.predictions
        ):
            self.ax.plot(
                [xi, xi],
                [yi, pi],
                linestyle="--",
                color="#4f7ea8",
                linewidth=2
            )

        self.ax.scatter(
            self.x,
            self.predictions,
            s=50,
            color=self.point
        )

    def update_text(self):
        sign = "+" if self.intercept >= 0 else "-"

        self.parameters.config(
            text=(
                f"y = {self.intercept:.2f} "
                f"{sign} "
                f"{abs(self.slope):.2f}x"
            )
        )

        descriptions = {
            "Data": (
                f"R2 = {self.r2:.2f} | "
                "b0 = intercept | b1 = slope\n"
                "The dataset contains the observed values "
                "used to fit the regression model."
            ),
            "Line": (
                f"R2 = {self.r2:.2f} | "
                "b0 = intercept | b1 = slope\n"
                "The regression line is fitted by minimizing "
                "the squared vertical residuals."
            ),
            "Predictions": (
                f"R2 = {self.r2:.2f} | "
                "b0 = intercept | b1 = slope\n"
                "The model predicts y for each observed x."
            ),
            "Residuals": (
                f"R2 = {self.r2:.2f} | "
                "b0 = intercept | b1 = slope\n"
                "Residuals are the vertical differences "
                "between observations and predictions."
            )
        }

        self.description.config(
            text=descriptions[self.focus]
        )

        for name, button in self.tabs.items():
            if name == self.focus:
                button.config(
                    bg="#eeeeee",
                    fg="#222222"
                )
            else:
                button.config(
                    bg="#414141",
                    fg=self.text
                )

    def change_focus(self, focus):
        self.focus = focus
        self.draw()

    def generate_data(self):
        rng = np.random.default_rng()

        self.x = np.linspace(
            0.7,
            9.0,
            10
        )

        self.y = (
            1.5
            + 0.5 * self.x
            + rng.normal(
                0,
                0.8,
                len(self.x)
            )
        )

        self.draw()

    def on_press(self, event):
        if event.inaxes != self.ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        distances = np.sqrt(
            (self.x - event.xdata) ** 2
            + (self.y - event.ydata) ** 2
        )

        index = np.argmin(distances)

        if distances[index] < 0.5:
            self.dragging_point = index

    def on_release(self, event):
        self.dragging_point = None

    def on_motion(self, event):
        if self.dragging_point is None:
            return

        if event.inaxes != self.ax:
            return

        if event.ydata is None:
            return

        index = self.dragging_point

        self.y[index] = np.clip(
            event.ydata,
            0.1,
            9.9
        )

        self.draw()


if __name__ == "__main__":
    root = tk.Tk()

    app = LinearRegressionVisualizer(root)

    root.mainloop()