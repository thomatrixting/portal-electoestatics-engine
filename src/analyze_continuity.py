import matplotlib.pyplot as plt
from plotting import plot_field, plot_trajectories, plot_velocities

recording_sor = "output/recording_20260723_204411.npz"
recording_mom = "output/recording_20260723_204148.npz" 

recording = recording_sor

# Ver la trayectoria de la carga sobre el campo
plot_trajectories(recording, every_n_frames=1,
                  field="potential", scheme="Potential",
                  show_isolines=True)

# Velocidades
plot_velocities(recording, every_n_frames=1, max_value=1)

axes = plot_velocities(recording,
                       every_n_frames=1, show=False, max_value=1)

# marcar el frame aproximado de teletransportación
for ax in axes:
    ax.axvline(x=161, color="red", linestyle="--",
               linewidth=1.2, label="Teleportation event")
    ax.legend()

plt.suptitle("Continuidad de velocidad (SOR)",
             fontsize=12)
plt.tight_layout()
plt.savefig("results_continuity/axiom_continuity_sor.png", dpi=150)
plt.show()