from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle

# параметры волновода
a = 23e-3
b = 10e-3
f = 17e9
l = 8e-3
x0 = 10e-3
z0 = 15e-3
I = 2


# сетка для расчётов
_x = np.linspace(0, a, 21)
_y = np.linspace(0, b, 47)
x, y = np.meshgrid(_x, _y)


# постоянные
c = 3e8
o = 2 * np.pi * f
mu = 4 * np.pi * 1e-7
eps = 8.85e-12
pi = np.pi


def g2(m, n):
    return np.pi**2 * ((m/a)**2 + (n/b)**2)


def h(m, n):
    return ((o/c)**2 - g2(m, n))**.5


def E(m, n, z):
    h1 = h(m, n)
    ex = -1j * (h1 * m * pi) / (g2(m, n)*a) * np.cos(m * pi * x / a) * \
        np.sin(n * pi * y / b) * np.exp(-1j * h1 * z)
    ey = -1j * (h1 * n * pi) / (g2(m, n)*b) * \
        np.sin(m * pi * x / a) * np.cos(n*pi*y/b) * np.exp(-1j * h1 * z)
    hx = -h1 / (o * mu) * ey
    hy = h1 / (o * mu) * ex
    return np.array([ex, ey, hx, hy])


def H(m, n, z):
    h1 = h(m, n)
    ex = 1j * (o * mu * n * pi) / (g2(m, n) * b) * \
        np.cos(m * pi * x / a) * np.sin(n * pi * y / b) * \
        np.exp(-1j * h1 * z)
    ey = -1j * (o * mu * m * pi) / (g2(m, n)*a) * \
        np.sin(m * pi * x / a) * np.cos(n * pi * y / b) * \
        np.exp(-1j * h1 * z)
    hx = -o * eps / h1 * ey
    hy = o * eps / h1 * ex
    return np.array([ex, ey, hx, hy])

# расчёт амплитуд
h1 = (2 * I) / (a * b * h(1, 0)) * pi * l / \
    a * np.sin(pi * x0 / a) * np.sin(h(1, 0) * z0)
h2 = (2 * I) / (a * b * h(2, 0)) * 2 * pi * l / \
    a * np.sin(2 * pi * x0 / a) * np.sin(h(2, 0) * z0)
h4 = (4 * I) / (a * b * h(1, 1)) * b / a * \
    np.sin(pi * x0 / a) * np.sin(pi * l / b) * np.sin(h(1, 1) * z0)
e5 = (4 * I) / (a * b * o * eps) * np.sin(pi * x0 / a) * \
    np.sin(pi * l / b) * np.sin(h(1, 1) * z0)

# print(h1, h2, h4, e5)


def f(z):
    return h1 * H(1, 0, z) + \
        h2 * H(2, 0, z) + \
        h4 * H(1, 1, z) + \
        e5 * E(1, 1, z)


def p(z):
    f1 = f(z)
    return (np.real(f1[0] * np.conj(f1[3]) - f1[1] * np.conj(f1[2]))) / 2

# пробегая по оси z
for i in range(15, 71):
    z = i * 1e-3
    plt.clf()
    plt.title("$z = %d$ мм" % i)
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        top='off',
        left='off',
        right='off',
        labelbottom='off',
        labeltop='off',
        labelleft='off',
        labelright='off')
    plt.rc('text', usetex=True)
    plt.rc('text.latex', unicode=True)
    plt.rc('text.latex', preamble='\\usepackage[russian]{babel}')
    plt.axes().set_aspect("equal")
    cs = plt.contourf(x, y, p(z), np.linspace(0, 1.2e7, 61), cmap=plt.cm.Greys)
    ax = plt.axes()

    # внезапно: добавляем штырь
    ax.add_patch(Rectangle((x0-a/400, 0), a/200, l, facecolor="white"))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cs, cax)
    plt.savefig("%d.png" % i)

# расчёт мощностей
# p1 = o*mu / (a*b*h(1, 0)) * I**2 * \
#     (l * np.sin(np.pi*x0/a) * np.sin(h(1, 0)*z0))**2
# print(h(1, 0), p1)
# p2 = o*mu / (a*b*h(2, 0)) * I**2 * \
#     (l * np.sin(2*np.pi*x0/a) * np.sin(h(2, 0)*z0))**2
# print(h(2, 0), p2)
# p4 = 2*o*mu / (a*b*h(1, 1)*g2(1, 1)) * I**2 * \
#     (b/a*np.sin(np.pi*x0/a) * np.sin(np.pi*l/b) * np.sin(h(1, 1)*z0))**2
# print(h(1, 1), p4)
# p5 = h(1, 1)/g2(1, 1) * 2*I**2/(o*eps*a*b) * \
#     (np.sin(np.pi*x0/a) * np.sin(np.pi*l/b) * np.sin(h(1, 1)*z0))**2
# print(h(1, 1), p5)
