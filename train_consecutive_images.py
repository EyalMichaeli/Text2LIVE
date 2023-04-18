import os
import random
from argparse import ArgumentParser
import datetime
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
import shutil
import imageio
import math
import numpy as np
import torch
import yaml
from tqdm import tqdm
import cv2
from PIL import Image
import glob
import logging
from datasets.image_dataset import SingleImageDataset
from models.clip_extractor import ClipExtractor
from models.image_model import Model
from util.losses import LossG
from util.util import tensor2im, get_optimizer
from train_image import train_model, save_locally

from image_text_tools.gpt_methods import get_gpt_response
from util.logging import init_logging


def visualize_output(results_folder: str, title: str = ""):
    """
    Visualize the output of the training process.
    "uses only the "composite" image.
    if it's more than 3 images, it will be split into multiple rows.
    """
    folders_names = [name for name in os.listdir(results_folder) if os.path.isdir(os.path.join(results_folder, name))]
    # sort by epoch
    folders_names = sorted(folders_names, key=lambda x: int(x))
    # init plot
    num_rows = math.ceil(len(folders_names)/3)
    fig, axs = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
    # set axis off for all images
    for ax in axs.flat:
        ax.axis("off")
    # iterate over all folders
    for i, folder_name in enumerate(folders_names):
        epoch_folder = os.path.join(results_folder, folder_name)
        # read the image
        img = np.array(Image.open(os.path.join(epoch_folder, "composite.png")))
        # add it to the plot
        axs[i//3, i%3].imshow(img)
        axs[i//3, i%3].set_title(f"epoch: {Path(epoch_folder).name}")
    plt.show()

    


def main(args, config_path, complete_text: str = None, n_epochs=400, visualize=False, device="cuda:0", verbose=False):
    if verbose:
        logging.info(f"device: {device}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config["n_epochs"] = n_epochs
    with open(f"./configs/image_example_configs/{args.example_config}", "r") as f:
        example_config = yaml.safe_load(f)
    config["example_config"] = args.example_config
    config.update(example_config)

    run_name = f"-{config['image_path'].split('/')[-1]}"
    if config["use_wandb"]:
        import wandb

        wandb.init(project=config["wandb_project"], entity=config["wandb_entity"], config=config, name=run_name)
        wandb.run.name = str(wandb.run.id) + wandb.run.name
        config = dict(wandb.config)
    else:
        now = datetime.datetime.now()
        run_name = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}{run_name}"
        path = Path(f"{config['results_folder']}/{run_name}")
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "config.yaml", "w") as f:
            yaml.dump(config, f)

        # added, save example config too.
        with open(path / "text_config.yaml", "w") as f:
            yaml.dump(example_config, f)

        config["results_folder"] = str(path)

    train_model(config, device)
    if config["use_wandb"]:
        wandb.finish()
    

    if visualize:
        visualize_output(config["results_folder"], title=complete_text)

    return config["results_folder"]


class Args:
    def __init__(self, config_path: str, example_config_path: str):
        self.config_path = config_path
        self.example_config = example_config_path


def change_config_yaml(output_path, config_dict):
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f)


def pick_random_text_aug():
    """
    picks a random text augmentation (that was used in the original paper) to later be added ot the text.
    """    
    text_augs = ["a photo of ", "the photo of ", "image of ", "an image of ", "high quality image of ", 
                "a high quality image of ", "the ", "a ", "an "]
    text_aug = random.choice(text_augs)
    return text_aug



def wrapped_main_func(config_path, example_config_name, text_config_dict, n_epochs=400, visualize=False, device=torch.device("cuda:0"), verbose=False):
    args = Args(config_path, example_config_name)
    change_config_yaml("/mnt/raid/home/eyal_michaeli/git/Text2LIVE/configs/image_example_configs/template.yaml", text_config_dict)
    if visualize:
        # plot the original image given in the text config dict
        plt.imshow(Image.open(text_config_dict["image_path"]))
        # find out title using image_path, and add to plot
        title = Path(text_config_dict["image_path"]).parent.name
        plt.title(f"original image class: {title}")
        plt.show()
    # get the complete text from the config dict
    complete_text = text_config_dict["comp_text"]
    output_folder = main(args, config_path, complete_text, visualize=visualize, n_epochs=n_epochs, device=device, verbose=verbose)
    return output_folder


def run_on_imagenet_image(image_path, config_path, example_config_name, n_epochs, use_gpt=True, text_config_dict=None, visualze=False, device="cuda:0", verbose=False):
    """
    Run the model on a single image.
    Identifies what class is it using the path (its the parent of the image path)
    And uses it to adjust the text config dict
    """
    class_name = Path(image_path).parent.name
    if use_gpt:
        extra_string_for_model = get_gpt_response(text=class_name)
    else:
        extra_string_for_model = random.choice(["old", "new", "pretty", "good looking", "ugly", "bad", "nice", "beautiful"])
    logging.info(f"class_name: {class_name}, extra_string_for_model: {extra_string_for_model}")
    target_text = f'{extra_string_for_model} {class_name}'
    if text_config_dict is None:
        text_config_dict = {
                    'image_path': image_path,
                    'screen_text': target_text, 
                    'comp_text': f'{pick_random_text_aug()} {target_text}',
                    'src_text': class_name,
                    'bootstrap_text': class_name,
                    'bootstrap_epoch': random.choice([200, 300, 400, 500])
                }
    if verbose:
        logging.info(f"text_config_dict: {text_config_dict}")

    output_folder = wrapped_main_func(config_path, example_config_name, text_config_dict, n_epochs, visualize=visualze, device=device, verbose=verbose)
    return output_folder, f"{extra_string_for_model}_{class_name}"





if __name__ == "__main__":    
    
    NUM_RUNS_FOR_EACH_IMAGE = 1
    RUN_NAME = "imagenet_10k_first_half"

    config_path = "./configs/image_config.yaml"
    example_config_name = f"template.yaml"  
    device = torch.device("cuda:3")  # change gpu ID and index_to_split if u want to run parallely

    init_logging(run_name=f"{RUN_NAME}")

    logging.info(f"using device: {device}")

    imagenet_image_paths = sorted(list(glob.glob("/mnt/raid/home/eyal_michaeli/datasets/imagenet_10k/imagenet_images/*/*.jpg")))
    # take only half 
    index_to_split = len(imagenet_image_paths)//2
    imagenet_image_paths = imagenet_image_paths[: index_to_split]
    logging.info(f"len(imagenet_image_paths): {len(imagenet_image_paths)}")
    logging.info(f"index_to_split: {index_to_split}")

    # run once on each imagenet image:
    for image_path in tqdm(imagenet_image_paths):
        for _ in range(NUM_RUNS_FOR_EACH_IMAGE):
            # pick a random number of epochs
            n_epochs = random.choice([200, 300, 400])
            output_folder, edited_class_string = run_on_imagenet_image(image_path, config_path, example_config_name, n_epochs=n_epochs, use_gpt=True, visualze=False, device=device)

            # copy the resulting image (the latest epoch) to the same imagenet folder
            # so we can later use it to train the classifier
            # (we want to train the classifier on the images that were generated by the model)

            # take the *image_name* from the result folder
            image_name = Path(output_folder).name[20:-4]
            file_path_to_copy = Path(output_folder) / f"{n_epochs}" / "composite.png"
            # copy the file to the imagenet class folder, with the name consisting of the epoch number and the image name
            target_file_name = f"{image_name}_text2live_{edited_class_string}_epochs_{n_epochs}.png"
            target_path = Path(image_path).parent / target_file_name
            logging.info(f"copying {file_path_to_copy} to {target_path}")
            shutil.copy(file_path_to_copy, target_path)

        

