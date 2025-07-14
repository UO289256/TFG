import numpy as np
import matplotlib.pyplot as plt

# Generamos los datos de ejemplo
np.random.seed(42)
x = np.linspace(0, 10, 20)
y = 2 * x + 1 + np.random.normal(scale=4.0, size=x.shape)

# Ajustamos la recta de regresión y calculamos las predicciones
coeficientes = np.polyfit(x, y, 1)
y_pred = np.polyval(coeficientes, x)

# Seleccionamos un punto para ilustrar el residual
indice = 7
x_ind, y_ind = x[indice], y[indice]
y_pred_ind = np.polyval(coeficientes, x_ind)

# Creamos la figura y el eje con tamaño y resolución definidos
figura, eje = plt.subplots(figsize=(12, 6), dpi=300)

# Configuramos el color y grosor de los bordes de los ejes
color_ejes = '#BBBBBB'
for borde in ['top', 'bottom', 'left', 'right']:
    eje.spines[borde].set_color(color_ejes)
    eje.spines[borde].set_linewidth(1.0)

# Representamos los puntos y la línea de regresión
eje.scatter(
    x, y,
    color="#AC7A8A",
    edgecolor=color_ejes,
    linewidth=0.5,
    alpha=0.85,
    s=50,
    zorder=2
)
eje.plot(
    x, y_pred,
    'k--',
    linewidth=1.0,
    zorder=1
)

# Dibujamos el residual como flecha de doble punta para validarlo
eje.annotate(
    '',
    xy=(x_ind, y_ind),
    xytext=(x_ind, y_pred_ind),
    arrowprops=dict(arrowstyle='<->', color='black', lw=1),
    zorder=3
)

# Anotamos el error residual
eje.text(
    x_ind - 0.8, (y_ind + y_pred_ind) / 2,
    "Error\nresidual",
    color='black', fontsize=12,
    va='center', ha='left'
)

# Indicamos la posición de la línea de regresión con anotación
x_media = x.mean()
y_media = np.polyval(coeficientes, x_media)
eje.annotate(
    "Línea de regresión",
    xy=(x_media, y_media),
    xytext=(x_media + 2, y_media + 5),
    arrowprops=dict(arrowstyle='->', color='black', lw=1),
    fontsize=12, ha='left', va='bottom'
)

# Etiquetamos ejes, activamos cuadrícula y ajustamos ticks
eje.set_xlabel('X', fontsize=12)
eje.set_ylabel('Y', fontsize=12)
eje.yaxis.grid(True, color=color_ejes, linestyle='--', linewidth=0.7, alpha=0.6)
eje.set_axisbelow(True)
eje.tick_params(axis='both', which='both', length=0)

# Ajustamos el diseño, guardamos la figura en PDF y la mostramos
plt.tight_layout()
plt.savefig("grafico_regresion.pdf")
plt.show()
