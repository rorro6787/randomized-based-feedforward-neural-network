import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import numpy as np

# Configuración inicial
torch.manual_seed(0)
np.random.seed(0)

# Hiperparámetros iniciales
gamma = 0.01
batch_size = 128

# Función para cargar y particionar datos
def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = MNIST(root='./data', train=False, transform=transform, download=True)

    train_indices, val_indices = train_test_split(range(len(train_dataset)), test_size=0.25, random_state=42)
    train_data = Subset(train_dataset, train_indices)
    val_data = Subset(train_dataset, val_indices)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("\nDatos cargados y particionados.\n")
    return train_loader, val_loader, test_loader

# Función para obtener tensores de datos y etiquetas de un DataLoader
def loader_to_tensors(loader):
    X, Y = [], []
    for data, target in loader:
        X.append(data.view(data.size(0), -1))  # Aplanar las imágenes
        Y.append(torch.nn.functional.one_hot(target, num_classes=10))
    return torch.cat(X), torch.cat(Y)

# Definir función del kernel RBF
def rbf_kernel(X, W):
    dist_matrix = torch.cdist(X, W) ** 2
    return torch.exp(-gamma * dist_matrix)

# Búsqueda de hiperparámetros
def hyperparameter_search(X_train, Y_train, X_val, Y_val, C_values, L_values):
    performance = torch.zeros(len(C_values), len(L_values))
    num_iterations = len(C_values) * len(L_values)

    print("Comenzando la búsqueda de hiperparámetros...\n")
    for i, C in enumerate(C_values):
        for j, L in enumerate(L_values):
            print(f"Iteración {i * len(L_values) + j + 1}/{num_iterations}")
            print(f"Evaluando para C = {C} y L = {L}")
            
            W_hidden = torch.randn(X_train.size(1), L, dtype=torch.float32)
            H = rbf_kernel(X_train, W_hidden.T)

            I = torch.eye(L, dtype=torch.float32)
            W_output = torch.linalg.solve(H.T @ H + (1/C) * I, H.T @ Y_train.float())

            H_val = rbf_kernel(X_val, W_hidden.T)
            Y_pred_val = H_val @ W_output

            Y_pred_val_labels = Y_pred_val.argmax(dim=1)
            Y_val_labels = Y_val.argmax(dim=1)
            CCR_val = (Y_pred_val_labels == Y_val_labels).float().mean().item()
            
            performance[i, j] = CCR_val
            print(f"CCR en validación para C = {C}, L = {L}: {CCR_val}\n")

    performance_np = performance.numpy()
    best_idx = performance_np.argmax()
    i_opt, j_opt = np.unravel_index(best_idx, performance_np.shape)
    C_opt, L_opt = C_values[i_opt], L_values[j_opt]

    print("Búsqueda de hiperparámetros finalizada.\n")
    print(f"Mejor CCR en validación: {performance.flatten()[best_idx]}")
    print(f"Mejor C encontrado en validación: {C_opt}")
    print(f"Mejor L encontrado en validación: {L_opt}\n")

    return C_opt, L_opt

# Entrenamiento final y evaluación en el conjunto de prueba
def train_and_evaluate(X_train, Y_train, X_test, Y_test, C_opt, L_opt):
    W_hidden_opt = torch.randn(X_train.size(1), L_opt, dtype=torch.float32)
    H_opt = rbf_kernel(X_train, W_hidden_opt.T)
    W_output_opt = torch.linalg.solve(H_opt.T @ H_opt + (1/C_opt) * torch.eye(L_opt, dtype=torch.float32), H_opt.T @ Y_train.float())

    H_test = rbf_kernel(X_test, W_hidden_opt.T)
    Y_pred_test = H_test @ W_output_opt

    Y_pred_test_labels = Y_pred_test.argmax(dim=1)
    Y_test_labels = Y_test.argmax(dim=1)
    
    CCR_test = (Y_pred_test_labels == Y_test_labels).float().mean().item()
    MSE_test = torch.mean((Y_pred_test - Y_test.float()) ** 2).item()

    return CCR_test, MSE_test

# Función principal
def main():
    # Cargar datos
    train_loader, val_loader, test_loader = load_data()

    # Obtener tensores para entrenamiento, validación y prueba
    X_train, Y_train = loader_to_tensors(train_loader)
    X_val, Y_val = loader_to_tensors(val_loader)
    X_test, Y_test = loader_to_tensors(test_loader)

    # Definir valores de C y L para la búsqueda de hiperparámetros
    C_values = [10**(-3), 10**(-2), 10**(-1), 1, 10, 100, 1000]
    L_values = [50, 100, 500, 1000, 1500, 2000]

    # Búsqueda de hiperparámetros
    C_opt, L_opt = hyperparameter_search(X_train, Y_train, X_val, Y_val, C_values, L_values)

    # Entrenamiento final y evaluación en el conjunto de prueba
    CCR_test, MSE_test = train_and_evaluate(X_train, Y_train, X_test, Y_test, C_opt, L_opt)
    print(f"CCR en conjunto de prueba: {CCR_test}")
    print(f"MSE en conjunto de prueba: {MSE_test}")

# Ejecutar el flujo principal
if __name__ == "__main__":
    main()
