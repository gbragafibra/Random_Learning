import numpy as np
import matplotlib.pyplot as plt



n = np.zeros((50,50)) #Initial random concentrations
n[15,20] = 0.5
τ = 0.1 
τ_steps = 500
D = 1.5 #Diffusion coef
α = 1.2 #proliferation rate

def FS(n, τ, τ_steps, D, α):
    """
    Fisher-Kolmogorov eq
    """
    for t in range(int(τ*τ_steps)):
        Δ_nx, Δ_ny = np.gradient(n) #∇
        Δ2_nxx = np.gradient(Δ_nx, axis = 0) #∇²
        Δ2_nyy = np.gradient(Δ_ny, axis = 1) #∇²
        Δ2_n = Δ2_nxx + Δ2_nyy #∇²

        n += τ * (D * Δ2_n + α * n * (1 - n))
        n = np.maximum(0, n) #Lower bounded >0
        n /= np.sum(n) #Normalize
    
    plt.imshow(n, cmap = "viridis")
    plt.colorbar(label = "n")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    pass

FS(n, τ, τ_steps, D, α)