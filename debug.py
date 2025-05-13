import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib

matplotlib.use('Qt5Agg')

fig, ax = plt.subplots()
point, = ax.plot([], [], 'ro')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

def init():
    point.set_data([], [])
    return point,

def update(frame):
    print(f"Update frame {frame}")
    point.set_data(frame, frame)
    return point,

ani = animation.FuncAnimation(fig, update, frames=10, init_func=init, blit=False)
plt.show()
# ani.save("orbits.mp4", fps=30, dpi=200)