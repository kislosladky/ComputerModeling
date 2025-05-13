import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


G = 6.67430e-11  # гравитационная постоянная


# Класс тела
class Body:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)


def compute_accelerations(bodies):
    accelerations = [np.zeros(2) for _ in bodies]
    for i, body_i in enumerate(bodies):
        for j, body_j in enumerate(bodies):
            if i != j:
                r = body_j.position - body_i.position
                dist = np.linalg.norm(r)
                if dist != 0:
                    accelerations[i] += G * body_j.mass * r / dist ** 3 #закон всемирного тяготения
    return accelerations


def runge_kutta_4_step(bodies, dt):
    n = len(bodies)

    orig_pos = [body.position.copy() for body in bodies]
    orig_velo = [body.velocity.copy() for body in bodies]

    # k1
    accel1 = compute_accelerations(bodies)
    k1velo = [a * dt for a in accel1]
    k1r = [v * dt for v in orig_velo]

    # k2
    for i in range(n):
        bodies[i].position = orig_pos[i] + 0.5 * k1r[i]
        bodies[i].velocity = orig_velo[i] + 0.5 * k1velo[i]
    accel2 = compute_accelerations(bodies)
    k2velo = [a * dt for a in accel2]
    k2r = [(orig_velo[i] + 0.5 * k1velo[i]) * dt for i in range(n)]

    # k3
    for i in range(n):
        bodies[i].position = orig_pos[i] + 0.5 * k2r[i]
        bodies[i].velocity = orig_velo[i] + 0.5 * k2velo[i]
    accel3 = compute_accelerations(bodies)
    k3velo = [a * dt for a in accel3]
    k3r = [(orig_velo[i] + 0.5 * k2velo[i]) * dt for i in range(n)]

    # k4
    for i in range(n):
        bodies[i].position = orig_pos[i] + k3r[i]
        bodies[i].velocity = orig_velo[i] + k3velo[i]
    accel4 = compute_accelerations(bodies)
    k4velo = [a * dt for a in accel4]
    k4r = [(orig_velo[i] + k3velo[i]) * dt for i in range(n)]

    # Обновление состояний
    for i in range(n):
        bodies[i].position = orig_pos[i] + (k1r[i] + 2 * k2r[i] + 2 * k3r[i] + k4r[i]) / 6
        bodies[i].velocity = orig_velo[i] + (k1velo[i] + 2 * k2velo[i] + 2 * k3velo[i] + k4velo[i]) / 6


# Создание тел
bodies = [
    Body(1.989e30, [0, 0], [10e2, 12e2]),  # звезда
    Body(5.972e24, [1.496e11, 0], [0, 29.78e3]),  # планета
    Body(7.35e22, [1.7e11, 1.7e11], [-23.78e3, 10.78e3])  # спутник
]

# Время симуляции
simulation_time = 6.154e7
dt = 1000
positions = [[] for _ in bodies]

# Основной цикл
time = 0
while time < simulation_time:
    for i, b in enumerate(bodies):
        positions[i].append(b.position.copy())
    runge_kutta_4_step(bodies, dt)
    time += dt

# График
plt.figure(figsize=(8, 8))
for i, traj in enumerate(positions):
    traj = np.array(traj)
    plt.plot(traj[:, 0], traj[:, 1], label=f'Тело {i + 1}')
plt.xlabel('x (м)')
plt.ylabel('y (м)')
plt.title('Гравитационное взаимодействие тел')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()


