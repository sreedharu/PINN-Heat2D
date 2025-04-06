import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Neural Network Architecture
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(2, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.hidden(inputs)

# Define PDE residual
def heat_residual(model, x, t, alpha=0.01):
    x.requires_grad_(True)
    t.requires_grad_(True)

    T = model(x, t)
    T_t = torch.autograd.grad(T, t, grad_outputs=torch.ones_like(T), retain_graph=True, create_graph=True)[0]
    T_x = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T), retain_graph=True, create_graph=True)[0]
    T_xx = torch.autograd.grad(T_x, x, grad_outputs=torch.ones_like(T), retain_graph=True, create_graph=True)[0]

    return T_t - alpha * T_xx

# Generate training data
def generate_training_data(nf=10000, nb=200, ni=200):
    # Collocation points inside the domain
    x_f = torch.rand((nf, 1), device=device)
    t_f = torch.rand((nf, 1), device=device)

    # Boundary points (x=0 or x=1)
    t_b = torch.rand((nb, 1), device=device)
    x_b0 = torch.zeros_like(t_b)
    x_b1 = torch.ones_like(t_b)

    # Initial condition (t=0)
    x_i = torch.rand((ni, 1), device=device)
    t_i = torch.zeros_like(x_i)
    T_i = torch.sin(np.pi * x_i).to(device)

    return x_f, t_f, x_b0, t_b, x_b1, t_b, x_i, t_i, T_i

# Train PINN
def train(model, optimizer, epochs=5000):
    x_f, t_f, x_b0, t_b0, x_b1, t_b1, x_i, t_i, T_i = generate_training_data()

    for epoch in range(epochs):
        optimizer.zero_grad()

        # PDE residual
        res = heat_residual(model, x_f, t_f)
        loss_pde = torch.mean(res**2)

        # Boundary loss
        T_b0 = model(x_b0, t_b0)
        T_b1 = model(x_b1, t_b1)
        loss_bc = torch.mean(T_b0**2) + torch.mean(T_b1**2)

        # Initial condition loss
        T_pred_i = model(x_i, t_i)
        loss_ic = torch.mean((T_pred_i - T_i)**2)

        loss = loss_pde + loss_bc + loss_ic
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.6f}')

# Plotting the results
def plot_solution(model):
    x = torch.linspace(0, 1, 100).reshape(-1, 1).to(device)
    t = torch.tensor([0.5]*100).reshape(-1, 1).to(device)

    with torch.no_grad():
        T = model(x, t).cpu().numpy()

    plt.plot(x.cpu().numpy(), T)
    plt.xlabel('x')
    plt.ylabel('Temperature T(x, t=0.5)')
    plt.title('Predicted Temperature at t = 0.5')
    plt.grid()
    plt.show()

# Main
if __name__ == "__main__":
    model = PINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train(model, optimizer)
    plot_solution(model)
