"""
analyze_axiom.py
=================
Comparación SOR vs MOM para el axioma de continuidad del potencial.

"""
from plotting import plot_field
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

snapshot_sor = "output/snapshot_20260723_204431.npz"
snapshot_mom = "output/snapshot_20260723_204029.npz" 

# npz_sor = np.load(snapshot_sor, allow_pickle=True)
# print("Keys SOR:", list(npz_sor.keys()))   # verificar nombre del campo

# vmin = float(npz_sor["potential"].min())
# vmax = float(npz_sor["potential"].max())


# Potencial
plot_field(snapshot_sor,
           field="potential", scheme="Potential",
           show_isolines=True, isoline_count=20,
           ax=axes[0], show=False)
axes[0].set_title("Potencial — SOR")

plot_field(snapshot_mom,
           field="potential", scheme="Potential",
           show_isolines=True, isoline_count=20,
           ax=axes[1], show=False)
axes[1].set_title("Potencial — MoM")

# Campo
# plot_field(snapshot_sor,
#            field="gradient", scheme="Default",
#            show_isolines=False, 
#            show_vectors=True,      
#            vector_step=8,          
#            ax=axes[0], show=False)
# axes[0].set_title("Campo eléctrico — SOR")

# plot_field(snapshot_mom,
#            field="gradient", scheme="Default",
#            show_isolines=False,
#            show_vectors=True,      
#            vector_step=8,          
#            ax=axes[1], show=False)
# axes[1].set_title("Campo eléctrico — MoM")

plt.tight_layout()
plt.savefig("results_continuity/comparassion_potential.png", dpi=150)
print("File Saved!")
plt.show()