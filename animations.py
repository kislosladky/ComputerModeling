import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

G = 6.67430e-11  # гравитационная постоянная

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
                    accelerations[i] += G * body_j.mass * r / dist ** 3
    return accelerations

def runge_kutta_4_step(bodies, dt):
    n = len(bodies)
    orig_pos = [body.position.copy() for body in bodies]
    orig_velo = [body.velocity.copy() for body in bodies]

    accel1 = compute_accelerations(bodies)
    k1velo = [a * dt for a in accel1]
    k1r = [v * dt for v in orig_velo]

    for i in range(n):
        bodies[i].position = orig_pos[i] + 0.5 * k1r[i]
        bodies[i].velocity = orig_velo[i] + 0.5 * k1velo[i]
    accel2 = compute_accelerations(bodies)
    k2velo = [a * dt for a in accel2]
    k2r = [(orig_velo[i] + 0.5 * k1velo[i]) * dt for i in range(n)]

    for i in range(n):
        bodies[i].position = orig_pos[i] + 0.5 * k2r[i]
        bodies[i].velocity = orig_velo[i] + 0.5 * k2velo[i]
    accel3 = compute_accelerations(bodies)
    k3velo = [a * dt for a in accel3]
    k3r = [(orig_velo[i] + 0.5 * k2velo[i]) * dt for i in range(n)]

    for i in range(n):
        bodies[i].position = orig_pos[i] + k3r[i]
        bodies[i].velocity = orig_velo[i] + k3velo[i]
    accel4 = compute_accelerations(bodies)
    k4velo = [a * dt for a in accel4]
    k4r = [(orig_velo[i] + k3velo[i]) * dt for i in range(n)]

    for i in range(n):
        bodies[i].position = orig_pos[i] + (k1r[i] + 2*k2r[i] + 2*k3r[i] + k4r[i]) / 6
        bodies[i].velocity = orig_velo[i] + (k1velo[i] + 2*k2velo[i] + 2*k3velo[i] + k4velo[i]) / 6

# Параметры
bodies = [
    Body(1.989e30, [0, 0], [10e2, 12e2]),
    Body(5.972e24, [1.496e11, 0], [0, 29.78e3]),
    Body(7.35e22, [1.7e11, 0], [0, 29.78e3])
]

simulation_time = 6.154e7
dt = 5000  # можно увеличить для ускорения
frame_interval = 20  # интервал между кадрами анимации в мс
max_frames = int(simulation_time // dt)

positions = [[] for _ in bodies]

# Фигура и ось
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlabel('x (м)')
ax.set_ylabel('y (м)')
ax.set_title('Анимация гравитационного взаимодействия тел')
ax.axis('equal')
ax.grid(True)

colors = ['yellow', 'blue', 'gray']
lines = [ax.plot([], [], '-', lw=1, color=c)[0] for c in colors]
dots = [ax.plot([], [], 'o', color=c)[0] for c in colors]

def init():
    ax.set_xlim(-2e11, 2e11)
    ax.set_ylim(-2e11, 2e11)
    for line in lines:
        line.set_data([], [])
    for dot in dots:
        dot.set_data([], [])
    return lines + dots

def update(frame):
    runge_kutta_4_step(bodies, dt)
    for i, body in enumerate(bodies):
        positions[i].append(body.position.copy())
        traj = np.array(positions[i])
        lines[i].set_data(traj[:, 0], traj[:, 1])
        dots[i].set_data(body.position[0], body.position[1])
    return lines + dots

ani = FuncAnimation(fig, update, frames=max_frames, init_func=init,
                    blit=True, interval=frame_interval, repeat=False)

plt.show()
