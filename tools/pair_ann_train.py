import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score
from utils.ann_plot_utils import (
    plot_metric_contour,
    plot_single_epoch_vs_metric
)
from dataset.BH_scene_dataset_concatenate import PairedLanguageDataset
from models.pair_ann import PAIR_ANN
from typing_extensions import List, Dict, Tuple
from loguru import logger
import numpy as np
from tqdm import tqdm
import json
import yaml
import os


def test_model(
        model: nn.Module, 
        loader: DataLoader, 
        device: torch.device
    ) -> float:
        # return the result of the test accuracy and f1 score
    model.eval()    ## set the model to eval mode

    correct = 0     ## correct prediction
    total = 0

    all_preds = []
    all_targets = []
    
    with torch.no_grad():       ## no gradient
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            labels = labels.unsqueeze(1)   ## squeeze the labels
            outputs = model(data)        ## get logit
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.append(predicted.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
    
    accuracy = 100. * correct / total

    loss = nn.BCELoss()
    # test_loss = loss(torch.cat(all_preds), torch.cat(all_targets)).item()
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    test_loss = loss(torch.from_numpy(all_preds), torch.from_numpy(all_targets)).item()
    
    model.train()   ## set the model to train mode
    return accuracy, test_loss

    
def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epochs: int
    ) -> Tuple[nn.Module, torch.optim.Optimizer, List[Dict]]:
    
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True)
    criterion = nn.BCELoss()
    results = []
    
    for epoch in range(epochs):
        model.train()
        all_preds = []
        all_targets = []
        
        for data, labels in (pbar := tqdm(train_loader)):
            data, labels = data.to(device), labels.to(device)
            labels = labels.unsqueeze(1)   ## squeeze the labels
            
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            predicted = (outputs > 0.5).float()

            pbar.set_postfix({"loss": loss.item()})
            
            all_preds.append(predicted.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
        
        train_acc = accuracy_score(np.concatenate(all_targets), np.concatenate(all_preds)) * 100
        test_acc, test_loss = test_model(model, test_loader, device)    ## get test accuracy and f1 score

        lr_scheduler.step(test_loss)   ## step the learning rate scheduler
        # train_loss = criterion(torch.cat(all_preds), torch.cat(all_targets)).item()
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        train_loss = criterion(torch.from_numpy(all_preds), torch.from_numpy(all_targets)).item()

        results.append( 
            {
                "train_accuracy": train_acc, 
                "test_accuracy": test_acc, 
                "train_loss": train_loss,
                "test_loss": test_loss,
            }
        )       ## save the result
        
        
        ## print train and test result
        logger.debug(f"Epoch: {epoch}, Train Accuracy: {train_acc}, Test Accuracy: {test_acc}")

    
    logger.info(f"Final Train Accuracy: {results[-1]['train_accuracy']}, Final Test Accuracy: {results[-1]['test_accuracy']}")
    # logger.info(f"Final Train F1: {results[-1]['train_f1']}, Final Test F1: {results[-1]['test_f1']}")
    
    return model, optimizer, results


def run_experiment(
        config: Dict[str, any],
        train_dataset: PairedLanguageDataset, 
        test_dataset: PairedLanguageDataset
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
            model = PAIR_ANN(config["training_params"]["default_param"]["model"])
            model.to(device)        ## move model to device
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=True)
            test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
            model, optimizer, result = train_model(model, train_loader, test_loader, optimizer, device, epochs)
            model_name = "{}_{}_{}.pt".format(
                config["training_params"]["exp_name"],
                lr,
                bs
            )
            model_dir = os.path.join(        ## save dir
                './plots/ann_pair_',
                config["training_params"]["exp_name"]
            )
            model_path = os.path.join(model_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            logger.info(f"Model saved at {model_path}")
            logger.info(f"Model Name: {model_name}")
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
    with open("./conifg/pair_ann.yaml") as f:
        config = yaml.safe_load(f)
    train_dataset = PairedLanguageDataset(**config["train_dataset"])   ## get train dataset
    test_dataset = PairedLanguageDataset(**config["test_dataset"])    ## get test dataset
    if config["training_params"]["run_experiments"]:
        save_dir = os.path.join(        ## save dir
            './plots/ann_pair_',
            config["training_params"]["exp_name"]
        )
        os.makedirs(save_dir, exist_ok=True)
        results = run_experiment(config, train_dataset, test_dataset)
        plot_metric_contour(
            results, 
            "test_accuracy", 
            os.path.join(save_dir, "test_accuracy.png") if config["training_params"]["save_plots"] else None
        )
        plot_single_epoch_vs_metric(
            results, 
            "test_accuracy", 
            os.path.join(save_dir, "test_accuracy_vs_epoch.png") if config["training_params"]["save_plots"] else None
        )
        plot_metric_contour(
            results, 
            "train_accuracy", 
            os.path.join(save_dir, "train_accuracy.png") if config["training_params"]["save_plots"] else None
        )
        plot_single_epoch_vs_metric(
            results, 
            "train_accuracy", 
            os.path.join(save_dir, "train_accuracy_vs_epoch.png") if config["training_params"]["save_plots"] else None
        )
        plot_single_epoch_vs_metric(
            results, 
            "test_loss", 
            os.path.join(save_dir, "test_loss_vs_epoch.png") if config["training_params"]["save_plots"] else None
        )
        plot_single_epoch_vs_metric(
            results, 
            "train_loss", 
            os.path.join(save_dir, "train_loss_vs_epoch.png") if config["training_params"]["save_plots"] else None
        )
        if config["training_params"]["save_plots"]:
            ## save the results
            with open(os.path.join(save_dir, "results.json"), "w") as f:
                json.dump(results, f)
    else:
        default_param = config["training_params"]["default_param"]
        model = PAIR_ANN(default_param["model"])        ## get model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)        ## move model to device
        optimizer = torch.optim.Adam(model.parameters(), lr=default_param["learning_rate"])
        train_loader = DataLoader(train_dataset, batch_size=default_param["batch_size"], shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=default_param["batch_size"], shuffle=True)
        ## train the model
        model, optimizer, result = train_model(model, train_loader, test_loader, optimizer, device, default_param["num_epochs"])
        logger.info(result)
        

if __name__=="__main__":
    main()
