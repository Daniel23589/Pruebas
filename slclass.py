import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import os
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
#print(val_set)
#print(os.listdir(train_set))
#train_set = datasets.ImageFolder(f"{image_path}", transform=transform) print(train_set)

# Establece las rutas
data_path = "C:/Users/danie/Desktop/Food_Ontology-main/Food_Ontology-main/MAFood121"
image_path = f"{data_path}/images"
annotation_path = f"{data_path}/annotations"
dish_list_path = f"{data_path}/annotations/dishes.txt"

# Crea transformaciones de imagen
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Carga el conjunto de datos de imagenes
train_set = datasets.ImageFolder(f"{image_path}", transform=transform)
val_set = datasets.ImageFolder(f"{image_path}", transform=transform)
test_set = datasets.ImageFolder(f"{image_path}", transform=transform)

# Carga las etiquetas
f = open(data_path + '/annotations/train.txt', "r")
train_labels = f.read().split('\n')
f.close()

f = open(data_path + '/annotations/val.txt', "r")
val_labels = f.read().split('\n')
f.close()

# Carga las etiquetas
f = open(data_path + '/annotations/test.txt', "r")
test_labels = f.read().split('\n')
f.close()

# Carga la lista de platos de comida
with open(dish_list_path, 'r') as f:
    class_list = f.read().splitlines()

# Carga la lista de etiquetas numéricas de los platos de comida
train_label_list_path = f"{annotation_path}/train_lbls_d.txt"
val_label_list_path = f"{annotation_path}/val_lbls_d.txt"
test_label_list_path = f"{annotation_path}/test_lbls_d.txt"

with open(train_label_list_path, 'r') as f:
    train_labels_list = [int(label) for label in f.read().splitlines()]
with open(val_label_list_path, 'r') as f:
    val_labels_list = [int(label) for label in f.read().splitlines()]
with open(test_label_list_path, 'r') as f:
    test_labels_list = [int(label) for label in f.read().splitlines()]

# Crea diccionarios de etiquetas
train_dict = {i: train_labels_list[i] for i in range(len(train_labels_list))}
val_dict = {i: val_labels_list[i] for i in range(len(val_labels_list))}
test_dict = {i: test_labels_list[i] for i in range(len(test_labels_list))}

# Crea un DataLoader
batch_size = 64

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Imprime el número de clases y el tamaño del conjunto de datos
num_classes = len(class_list)
print(f"Number of classes: {num_classes}")
print("######################################################################################################")
print(f"Train set size: {len(train_set)}")
print("######################################################################################################")
print(f"Validation set size: {len(val_set)}")
print("######################################################################################################")
print(f"Test set size: {len(test_set)}")
print("######################################################################################################")

#Imprime un ejemplo de la imagen y la etiqueta correspondiente
example_img, example_label = next(iter(train_loader))
print(f"Example image shape: {example_img.shape}")
print("######################################################################################################")
print(f"Example label: {example_label}")
print("######################################################################################################")

#Imprime la lista de clases
print(f"List of classes: {class_list}")
print("######################################################################################################")
#-------------------------------------------------------------------------------------------------------------

# Conversión de etiquetas numéricas en codificación one-hot.
import torch.nn.functional as F
train_one_hot = F.one_hot(torch.tensor(train_labels_list), num_classes)

# modelo de la red
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt

# Cargar la arquitectura ResNet50 pre-entrenada en ImageNet
resnet = models.resnet50(pretrained=True)

# Congelar los parámetros de la red
for param in resnet.parameters():
    param.requires_grad = False

# Reemplazar la última capa completamente conectada
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, num_classes)

# Definir la función de pérdida y el optimizador
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=0.001)

# Definir el número de épocas y la variable total_step
num_epochs = 100
total_step = len(train_loader)

# Definir listas para almacenar los valores de pérdida y precisión
losses = []
accuracies = []

# Entrenamiento de la red
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        #print("wait!!")
        # Mover los datos a la GPU si está disponible
        images = images.to(device)
        labels = labels.to(device)

        # Vaciar los gradientes acumulados
        optimizer.zero_grad()

        # Hacer una propagación hacia adelante y obtener la pérdida
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        resnet.to(device)
        images = images.to(device)
        outputs = resnet(images)
        loss = criterion(outputs, labels)

        # Realizar una propagación hacia atrás y actualizar los parámetros
        loss.backward()
        optimizer.step()

        # Calcular la precisión en el conjunto de entrenamiento
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        train_accuracy = correct / total

        # Agregar los valores de pérdida y precisión a las listas
        losses.append(loss.item())
        accuracies.append(train_accuracy)

        # Imprimir los resultados después de cada 100 iteraciones
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch+1, num_epochs, i+1, total_step, loss.item(),
                          train_accuracy*100))
            
# Guardar el modelo entrenado
torch.save(resnet.state_dict(), 'model_train.pth')

# Guardar el mejor modelo
torch.save({
    'epoch': epoch,
    'model_state_dict': resnet.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'accuracy': train_accuracy
}, 'best_model.pth')

# Graficar las pérdidas y las precisión (accuracy)
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()

plt.plot(accuracies)
plt.title('Training Accuracy')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.show()

# F1 Score
from sklearn.metrics import f1_score
resnet.eval()  # Establecer el modelo en modo de evaluación
y_true = []
y_pred = []

with torch.no_grad():  # Deshabilitar el cálculo de gradientes
    for images, labels in test_loader:
        # Mover los datos a la GPU si está disponible
        images = images.to(device)
        labels = labels.to(device)

        # Obtener las predicciones del modelo
        outputs = resnet(images)
        _, predicted = torch.max(outputs.data, 1)

        # Agregar las etiquetas verdaderas y las predicciones a las listas
        y_true += labels.cpu().tolist()
        y_pred += predicted.cpu().tolist()

f1 = f1_score(y_true, y_pred, average='weighted')
print('F1 Score: {:.2f}%'.format(f1*100))

# Graficar los resultados de losses y accuracies
fig, ax = plt.subplots()
ax.plot(losses, label='Train Loss')
ax.plot(accuracies, label='Train Accuracy')
ax.legend()
ax.set(title='Training Results', xlabel='Iterations', ylabel='Value')

# Guardar la imagen de los resultados obtenidos
fig.savefig('training_results.png')

