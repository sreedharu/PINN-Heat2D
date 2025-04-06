import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.autograd import grad

# Set random seed for reproducibility
torch.manual_seed(0)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sample and laser parameters
L = 0.05  # domain size in meters
T_max = 0.05  # max time in seconds
alpha = 1e-5  # thermal diffusivity

x0, y0 = 0.025, 0.025  # laser center
sigma = 0.00125  # spatial Gaussian width
tau = 0.001  # temporal Gaussian width

# Neural Network model
class PINN2D(nn.Module):
    def __init__(self):
        super(PINN2D, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, xyt):
        return self.net(xyt)

# Heat source function Q(x,y,t)
def Q(x, y, t):
    return torch.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2)) * \
           torch.exp(-((t - 0.0025) ** 2) / (2 * tau ** 2))

# Compute PDE residual
def pde_residual(model, xyt):
    xyt.requires_grad_(True)
    T = model(xyt)

    grads = grad(T, xyt, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    Tx, Ty, Tt = grads[:, 0], grads[:, 1], grads[:, 2]

    Txx = grad(Tx, xyt, grad_outputs=torch.ones_like(Tx), create_graph=True)[0][:, 0]
    Tyy = grad(Ty, xyt, grad_outputs=torch.ones_like(Ty), create_graph=True)[0][:, 1]

    q = Q(xyt[:, 0], xyt[:, 1], xyt[:, 2])
    return Tt - alpha * (Txx + Tyy) - q

# Generate training points
def generate_collocation_points(N):
    x = torch.rand(N, 1) * L
    y = torch.rand(N, 1) * L
    t = torch.rand(N, 1) * T_max
    return torch.cat([x, y, t], dim=1).to(device)

def generate_initial_points(N):
    x = torch.rand(N, 1) * L
    y = torch.rand(N, 1) * L
    t = torch.zeros_like(x)
    return torch.cat([x, y, t], dim=1).to(device)

def generate_boundary_points(N):
    t = torch.rand(N, 1) * T_max
    sides = []
    for val in [0.0, L]:
        x = torch.full((N, 1), val)
        y = torch.rand(N, 1) * L
        sides.append(torch.cat([x, y, t], dim=1))
        x = torch.rand(N, 1) * L
        y = torch.full((N, 1), val)
        sides.append(torch.cat([x, y, t], dim=1))
    return torch.cat(sides, dim=0).to(device)

# Initialize model and optimizer
model = PINN2D().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(5000):
    optimizer.zero_grad()

    colloc = generate_collocation_points(500)
    init = generate_initial_points(200)
    bc = generate_boundary_points(100)

    loss_pde = torch.mean(pde_residual(model, colloc) ** 2)
    loss_init = torch.mean(model(init) ** 2)
    loss_bc = torch.mean(model(bc) ** 2)

    loss = loss_pde + loss_init + loss_bc
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# Create surface plot as a gif
x = torch.linspace(0, L, 100)
y = torch.linspace(0, L, 100)
X, Y = torch.meshgrid(x, y, indexing='ij')
frames = []

fig, ax = plt.subplots()
for t_val in np.linspace(0, T_max, 50):
    T_plot = model(torch.stack([X.flatten(), Y.flatten(),
                                 torch.full_like(X.flatten(), t_val)], dim=1).to(device))
    T_img = T_plot.view(100, 100).detach().cpu().numpy()
    frame = ax.imshow(T_img, cmap='hot', origin='lower', extent=[0, L, 0, L], animated=True)
    frames.append([frame])

ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True)
ani.save("heat_spread.gif", writer="pillow")
print("✅ GIF saved as heat_spread.gif")

plt.show()
