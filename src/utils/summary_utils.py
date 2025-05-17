from typing import Dict, List, Tuple
import torch
import os
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from datetime import datetime
from src.utils.engine import train_func, test_func
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc, recall_score
import seaborn as sns

from torch.optim import lr_scheduler


def plot_confusion_matrix(cm, class_names,title):
    """
    Plots the confusion matrix using seaborn.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.close(fig)
    return fig

def plot_roc_curve(fpr, tpr, roc_auc):
    fig = plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.close()
    return fig

def train_func_profiler(model: nn.Module, data: DataLoader, loss_fn:nn.Module,
               optimizer: torch.optim.Optimizer, device: torch.device
               , log_dir: str) -> Tuple[float, float]:
    """
    Trains a model for a single epoch

    Args
    ----
        model: A Pytorch model
        data: a DataLoader object with the train data
        loss_fn: loss function to minimized
        optimizer: A Pytorch optimizer
        device: A target device to perform the operations ("cuda" or "cpu")

    Returns
    ------
        A tuple with the loss and accuracy of the training epoch like
        (train_loss, train_acc)
    """
    def trace_handler(prof):
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))

    model.train()
    train_loss, train_acc = 0, 0
    all_preds = []
    all_labels = []
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=1,
            repeat=1),
        #on_trace_ready=trace_handler,
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
        with_stack=True
    ) as profiler:
        for _ , (x, y) in enumerate(data):
            x, y = x.to(device), y.to(device)
            y_logit = model(x)
            loss = loss_fn(y_logit, y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            profiler.step()
            y_pred = torch.softmax(y_logit, 1).argmax(1)
            train_acc += (y_pred == y).sum().item() / len (y_pred)
            all_preds.extend(y_pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            #We can use another function if we want
    train_loss = train_loss / len(data)
    train_acc = train_acc / len(data)
    return train_loss, train_acc, all_preds, all_labels

def train(model: nn.Module, test_data: DataLoader, train_data: DataLoader, loss_fn:nn.Module,
        optimizer: torch.optim.Optimizer, device: torch.device, epochs: int
        , writer: SummaryWriter, title: str) -> Dict[str, List]:
    """
    Trains and test a Pytorch model

    Args:
    -----
        model: A Pytorch model
        train_data: a DataLoader object with the train data
        test_data: a DataLoader object with the train data
        loss_fn: loss function to minimized
        optimizer: A Pytorch optimizer
        device: A target device to perform the operations ("cuda" or "cpu")
        epochs: A integre with the number of epochs that the model will be train
    Returns:
    --------
        A dictionary with the train and test loss and accuracy for every epoch
        in the form of 
        {"train_loss": [...],
        "train_acc": [...],
        "test_loss": [...],
        "test_acc": [...]}
    """

    best_test_acc = 0
    best_train_loss = 1000
    best_model = None
    patience_counter = 0

    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc,train_preds, train_labes = train_func(
            model,train_data,loss_fn,optimizer, device)
        test_loss, test_acc, test_preds, test_labels = test_func(
            model, test_data, loss_fn, device)

        scheduler.step()

        train_f1 = f1_score(train_labes, train_preds, average="weighted")
        test_f1 = f1_score(test_labels, test_preds, average="weighted")
        train_recall = recall_score(train_labes, train_preds, average="weighted")
        test_recall = recall_score(test_labels, test_preds, average="weighted")


        print(
            "\n"
            f"Epoch {epoch+1} |"
            f"train_loss :{train_loss: .4f} |"
            f"train_acc :{train_acc: .4f} |"
            f"test_loss :{test_loss: .4f} |"
            f"test_acc :{test_acc: .4f} |"
            f"train_f1_w :{train_f1: .4f} |"
            f"test_f1_w :{test_f1: .4f} |"
            f"train_recall :{train_recall: .4f} |"
            f"test_recall :{test_recall: .4f} "
        )
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        

        ##Add of the writer
        if writer:
            writer.add_scalars(main_tag="Loss", 
            tag_scalar_dict={"train_loss": train_loss, "tests_loss": test_loss},
            global_step=epoch
            )

            writer.add_scalars(main_tag="Accuracy", 
            tag_scalar_dict={"train_acc": train_acc, "tests_acc": test_acc},
            global_step=epoch
            )

            writer.add_scalars(main_tag="F1_weighted", 
            tag_scalar_dict={"train_f1_w": train_f1, "tests_f1_w": test_f1},
            global_step=epoch
            )

            writer.add_scalars(main_tag="Recall_weighted",
            tag_scalar_dict={"train_recall": train_recall, "tests_recall": test_recall},
            global_step=epoch
            )

            #Confusion matrix
            if epoch == epochs - 1:
                cm = confusion_matrix(test_preds, test_labels)
                cm_fig = plot_confusion_matrix(cm, train_data.dataset.dataset.classes, title)
                writer.add_figure("Confusion Matrix", cm_fig, global_step=epoch)


        # Para guardar el mejor modelo
        if test_acc > best_test_acc and train_loss < best_train_loss:
            best_test_acc = train_acc
            best_train_loss = train_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter > 5:
            print(f"[INFO] Early stopping at epoch {epoch+1}")
            cm = confusion_matrix(test_preds, test_labels)
            cm_fig = plot_confusion_matrix(cm, train_data.dataset.dataset.classes, title)
            writer.add_figure("Confusion Matrix", cm_fig, global_step=epoch)
            break

    return results, best_model, best_test_acc, cm_fig, test_loss

def create_write(name: str, model: str, experiment_name: str,extra: str=None) -> SummaryWriter():

    timestamp = datetime.now().strftime("%Y-%m-%d")
    if extra:
        log_dir = os.path.join("runs", timestamp, experiment_name, name, model, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, name, model)
    print(f"[INFO] create summary writer, saving to: {log_dir}")
    return SummaryWriter(log_dir=log_dir)

def select_optimizer(model: nn.Module, optimizer: str):
    if optimizer == "Adam":
        return torch.optim.Adam(model.parameters(), lr=0.01)
    if optimizer == "SGD":
        return torch.optim.SGD(model.parameters(), lr=0.01)
    if optimizer == "Adamw":
        return torch.optim.AdamW(model.parameters(), lr=0.01)
