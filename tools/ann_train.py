import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, 
    f1_score
)
from utils.ann_plot_utils import (
    plot_metric_contour,
    plot_single_epoch_vs_metric
)
from dataset.BH_scene_dataset import BHSceneDataset
from models.ann import ANN_base
from typing_extensions import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import json
import yaml
import os


def test(
        model: nn.Module, 
        test_loader: DataLoader, 
        device: torch.device
    ) -> Tuple[float, float]:
    # return the result of the test accuracy and f1 score
    model.eval()    ## set the model to eval mode

    correct = 0     ## correct prediction
    all_preds = []  ## all predictions
    all_targets = []    ## all targets
    
    with torch.no_grad():       ## no gradient
        model.eval()    ## set the model to eval mode

    correct = 0     ## correct prediction
    all_preds = []  ## all predictions
    all_targets = []    ## all targets
    
    with torch.no_grad():       ## no gradient
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data, target = data.to(device), target.to(device)
            output = model(data)        ## get logit
            preds = output.argmax(dim=1)    ## get prediction
            correct += preds.eq(target).sum().item()    ## get correct prediction
            
            all_preds.append(preds.cpu().numpy())   ## save the prediction
            all_targets.append(target.cpu().numpy())    ## save the target
    
    accuracy = 100. * correct / len(test_loader.dataset)
    
    all_preds = np.concatenate(all_preds)       ## concatenate all the predictions
    all_targets = np.concatenate(all_targets)   ## concatenate all the targets

    f1 = f1_score(all_targets, all_preds, average="macro")  ## get f1 score
    model.train()   ## set the model to train mode
    return accuracy, f1


def train(
        model: nn.Module, 
        train_loader: DataLoader, 
        test_loader: DataLoader, 
        optimizer: torch.optim.Optimizer, 
        device: torch.device, 
        epochs: int
    ) -> Tuple[nn.Module, torch.optim.Optimizer, List[Dict[str, float]]]:
    # return the result of the training accuracy and f1 score

    results =[]
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):     ## for each epoch
        model.train()
        all_preds = []
        all_targets = []
        all_preds = []
        all_targets = []
        for data, target in tqdm(train_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)        ## get logit
            loss = loss_fn(output, target)    ## get loss
            loss = loss_fn(output, target)    ## get loss
            optimizer.zero_grad()    ## zero the gradient
            loss.backward()    ## backpropagation
            optimizer.step()    ## update the weight
            preds = output.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            preds = output.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(target.cpu().numpy())

        # train_acc, train_f1 = test(model, train_loader, device)   ## get train accuracy and f1 score
        train_acc = accuracy_score(np.concatenate(all_targets), np.concatenate(all_preds)) * 100
        train_f1 = f1_score(np.concatenate(all_targets), np.concatenate(all_preds), average="macro")
        # train_acc, train_f1 = test(model, train_loader, device)   ## get train accuracy and f1 score
        train_acc = accuracy_score(np.concatenate(all_targets), np.concatenate(all_preds)) * 100
        train_f1 = f1_score(np.concatenate(all_targets), np.concatenate(all_preds), average="macro")
        test_acc, test_f1 = test(model, test_loader, device)    ## get test accuracy and f1 score
        ## save the result
        results.append( 
            {
                "train_accuracy": train_acc, 
                "train_f1": train_f1, 
                "test_accuracy": test_acc, 
                "test_f1": 
                test_f1
            }
        )       ## save the result

        ## print train and test result
        print(f"Epoch: {epoch}, Train Accuracy: {train_acc}, Test Accuracy: {test_acc}")
        print(f"Epoch: {epoch}, Train F1: {train_f1}, Test F1: {test_f1}")

    print(f"Final Train Accuracy: {results[-1]['train_accuracy']}, Final Test Accuracy: {results[-1]['test_accuracy']}")
    print(f"Final Train F1: {results[-1]['train_f1']}, Final Test F1: {results[-1]['test_f1']}")

    return model, optimizer, results
    

def run_experiment(
        config: Dict[str, any],
        train_dataset: BHSceneDataset, 
        test_dataset: BHSceneDataset
    ) -> List[Dict[str, any]]:
    # return the result of the experiment
    hyperparameters = config["training_params"]["hyperparameter_range"]
    learing_rate = hyperparameters["learning_rate"]
    batch_size = hyperparameters["batch_size"]
    epochs = config["training_params"]["default_param"]["num_epochs"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []

    for lr in learing_rate:     ## for each learning rate
        for bs in batch_size:   ## for each batch size
            model = ANN_base(config["training_params"]["default_param"]["model"])
            model.to(device)        ## move model to device
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
            model, optimizer, result = train(model, train_loader, test_loader, optimizer, device, epochs)
            ## save the result
            results.append(
                {
                    "learning_rate": lr,
                    "batch_size": bs,
                    "result": result
                }
            )
            del model
            del optimizer
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
    
    return results

                    
def main():
    with open("./conifg/ann_train.yaml") as f:
        config = yaml.safe_load(f)
    train_dataset = BHSceneDataset(**config["train_dataset"])   ## get train dataset
    test_dataset = BHSceneDataset(**config["test_dataset"])    ## get test dataset
    if config["training_params"]["run_experiments"]:
        results = run_experiment(config, train_dataset, test_dataset)
        save_dir = os.path.join(        ## save dir
            './plots/ann',
            config["training_params"]["exp_name"]
        )
        os.makedirs(save_dir, exist_ok=True)
        plot_metric_contour(
            results, 
            "test_accuracy", 
            os.path.join(save_dir, "test_accuracy.png") if config["training_params"]["save_plots"] else None
        )
        plot_metric_contour(
            results, 
            "test_f1", 
            os.path.join(save_dir, "test_f1.png") if config["training_params"]["save_plots"] else None
        )
        plot_single_epoch_vs_metric(
            results, 
            "test_accuracy", 
            os.path.join(save_dir, "test_accuracy_vs_epoch.png") if config["training_params"]["save_plots"] else None
        )
        plot_single_epoch_vs_metric(
            results, 
            "test_f1", 
            os.path.join(save_dir, "test_f1_vs_epoch.png") if config["training_params"]["save_plots"] else None
        )
        if config["training_params"]["save_plots"]:
            ## save the results
            with open(os.path.join(save_dir, "results.json"), "w") as f:
                json.dump(results, f)
    else:
        default_param = config["training_params"]["default_param"]
        model = ANN_base(default_param["model"])        ## get model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)        ## move model to device
        optimizer = torch.optim.Adam(model.parameters(), lr=default_param["learning_rate"])
        train_loader = DataLoader(train_dataset, batch_size=default_param["batch_size"], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=default_param["batch_size"], shuffle=True)
        ## train the model
        model, optimizer, result = train(model, train_loader, test_loader, optimizer, device, default_param["num_epochs"])
        print(result)
        

if __name__=="__main__":
    main()
