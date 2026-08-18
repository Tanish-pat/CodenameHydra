import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np

FUNCTIONS = [
    {
        "name": "Concave: f(x) = -6 * (x - 2)² + 5 * x - 10",
        "func": lambda x: -6 * ((x - 2) ** 2) + 5 * x - 10,
        "x_range": (0, 4),
    },
    {
        "name": "Convex: f(x) = e^x",
        "func": lambda x: np.exp(x),
        "x_range": (0, 2),
    },
    {
        "name": "Concave: f(x) = ln(x)",
        "func": lambda x: np.log(x),
        "x_range": (0.5, 4),
    },
    {
        "name": "Inflection: f(x) = x³ - 3x",
        "func": lambda x: x**3 - 3 * x,
        "x_range": (-2, 2),
    },
]

plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(10, 7))
plt.subplots_adjust(left=0.1, bottom=0.25, top=0.9, right=0.9)

current_func_idx = 0
x_val = 0.8
y_val = 3.2
lambda_val = 0.5

def get_current_params():
    f_info = FUNCTIONS[current_func_idx]
    f = f_info["func"]
    x_min, x_max = f_info["x_range"]
    return f_info, f, x_min, x_max

def update(val=None):
    ax.clear()

    global lambda_val
    lambda_val = slider_lambda.val
    f_info, f, x_min, x_max = get_current_params()
    x_grid = np.linspace(x_min, x_max, 300)
    y_grid = f(x_grid)
    fx = f(x_val)
    fy = f(y_val)
    x_combo = lambda_val * x_val + (1 - lambda_val) * y_val
    f_xcombo = f(x_combo)  
    secant_val = lambda_val * fx + (1 - lambda_val) * fy  
    ax.plot(x_grid, y_grid, color="white", lw=2.5, label="f(x)")
    ax.plot([x_val, y_val],[fx, fy],color="#2ecc71",linestyle="--",lw=2,label="Secant Line",)

    ax.plot(x_val, fx, "go", ms=8)
    ax.plot(y_val, fy, "go", ms=8)

    ax.plot(x_combo, f_xcombo, "go", ms=9, zorder=5)

    ax.plot(x_combo, secant_val, "ro", ms=8, zorder=5)

    ax.axvline(x=x_val, color="gray", linestyle=":", alpha=0.6, label="x & y bounds")
    ax.axvline(x=y_val, color="gray", linestyle=":", alpha=0.6)
    ax.axvline(x=x_combo,color="#f39c12",linestyle="--",alpha=0.8,label=r"$\lambda x + (1-\lambda)y$",)

    ax.axhline(y=fx, color="gray", linestyle=":", alpha=0.4, label="f(x) & f(y)")
    ax.axhline(y=fy, color="gray", linestyle=":", alpha=0.4)
    ax.axhline(y=f_xcombo,color="#2ecc71",linestyle=":",alpha=0.7,label=r"$f(\lambda x + (1-\lambda)y)$",)
    ax.axhline(y=secant_val,color="#e74c3c",linestyle=":",alpha=0.7,label=r"$\lambda f(x) + (1-\lambda)f(y)$",)

    ax.set_xticks([x_val, x_combo, y_val])
    ax.set_xticklabels(["x", r"$\lambda x + (1-\lambda)y$", "y"], fontsize=11)

    ax.set_yticks([fx, fy, secant_val, f_xcombo])
    ax.set_yticklabels(["f(x)","f(y)",r"$\lambda f(x) + (1-\lambda)f(y)$",r"$f(\lambda x + (1-\lambda)y)$",],fontsize=10,)

    if f_xcombo > secant_val + 1e-4:
        ineq_symbol = "≥"
        func_class = "CONCAVE Region"
        color_class = "#2ecc71"
    elif f_xcombo < secant_val - 1e-4:
        ineq_symbol = "≤"
        func_class = "CONVEX Region"
        color_class = "#e74c3c"
    else:
        ineq_symbol = "="
        func_class = "LINEAR / EQUAL"
        color_class = "#f1c40f"

    formula_text = (r"$f(\lambda x + (1-\lambda)y)$ "+ f" {ineq_symbol} "+ r" $\lambda f(x) + (1-\lambda)f(y)$")

    ax.set_title(
        f"{f_info['name']}\nClass: {func_class}  [{formula_text}]",
        fontsize=13,
        color=color_class,
        pad=15,
    )

    ax.grid(True, linestyle="--", alpha=0.2)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.3)
    fig.canvas.draw_idle()


ax_lambda = plt.axes([0.25, 0.12, 0.5, 0.03], facecolor="#2c3e50")
slider_lambda = Slider(
    ax=ax_lambda,
    label=r"$\lambda$ (weight)",
    valmin=0.0,
    valmax=1.0,
    valinit=0.5,
    valfmt="%.2f",
    color="#e74c3c",
)
slider_lambda.on_changed(update)


def next_function(event):
    global current_func_idx, x_val, y_val
    current_func_idx = (current_func_idx + 1) % len(FUNCTIONS)
    _, _, x_min, x_max = get_current_params()
    span = x_max - x_min
    x_val = x_min + 0.15 * span
    y_val = x_min + 0.85 * span
    update()

ax_button = plt.axes([0.38, 0.03, 0.24, 0.05])
btn_next = Button(ax_button, "Next Function", color="#34495e", hovercolor="#2980b9")
btn_next.on_clicked(next_function)

update()
plt.show()