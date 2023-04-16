import os
import random
from argparse import ArgumentParser
import datetime
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
import imageio
import numpy as np
import torch
import yaml
from tqdm import tqdm
import cv2
from PIL import Image
import glob
from datasets.image_dataset import SingleImageDataset
from models.clip_extractor import ClipExtractor
from models.image_model import Model
from util.losses import LossG
from util.util import tensor2im, get_optimizer


def train_model(config, device):

    # set seed
    seed = config["seed"]
    if seed == -1:
        seed = np.random.randint(2 ** 32)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    print(f"running with seed: {seed}.")

    # create dataset, loader
    dataset = SingleImageDataset(config)

    # define model
    model = Model(config).to(device)

    # define loss function
    clip_extractor = ClipExtractor(config, device)
    criterion = LossG(config, clip_extractor, device)

    # define optimizer, scheduler
    optimizer = get_optimizer(config, model.parameters())

    for epoch in tqdm(range(1, config["n_epochs"] + 1)):
        inputs = dataset[0]
        for key in inputs:
            if key != "step":
                inputs[key] = inputs[key].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        for key in inputs:
            if key != "step":
                inputs[key] = [inputs[key][0]]
        losses = criterion(outputs, inputs)
        loss_G = losses["loss"]
        log_data = losses
        log_data["epoch"] = epoch

        # log current generated image to wandb
        if epoch % config["log_images_freq"] == 0:
            src_img = dataset.get_img().to(device)
            with torch.no_grad():
                output = model.render(model.netG(src_img), bg_image=src_img)
            for layer_name, layer_img in output.items():
                image_numpy_output = tensor2im(layer_img)
                log_data[layer_name] = [wandb.Image(image_numpy_output)] if config["use_wandb"] else image_numpy_output

        loss_G.backward()
        optimizer.step()

        # update learning rate
        if config["scheduler_policy"] == "exponential":
            optimizer.param_groups[0]["lr"] = max(config["min_lr"], config["gamma"] * optimizer.param_groups[0]["lr"])
        lr = optimizer.param_groups[0]["lr"]
        log_data["lr"] = lr

        if config["use_wandb"]:
            wandb.log(log_data)
        else:
            if epoch % config["log_images_freq"] == 0:
                save_locally(config["results_folder"], log_data)


def save_locally(results_folder, log_data):
    path = Path(results_folder, str(log_data["epoch"]))
    path.mkdir(parents=True, exist_ok=True)
    for key in log_data.keys():
        if key in ["composite", "alpha", "edit_on_greenscreen", "edit"]:
            imageio.imwrite(f"{path}/{key}.png", log_data[key])


def visualize_output(results_folder: str):
    # count how many numeric folders are in results_folder
    num_folders = len([name for name in os.listdir(results_folder) if os.path.isdir(os.path.join(results_folder, name))])
    # init plot
    fig, axs = plt.subplots(1, num_folders, figsize=(num_folders * 5, 5))
    # loop over all folders
    images = []
    for img_path in glob.glob(f"{results_folder}/*/composite.png"):
        # append image to list
        images.append(np.array(Image.open(img_path)))
    for i, img in enumerate(images):
        axs[i].imshow(img)
        axs[i].axis("off")
        plt.tight_layout()
        plt.title(f"epoch: {(i + 1) * 50}")
    plt.show()


def main(args, config_path, n_epochs=400, visualize=False, device="cuda:0"):
    print(f"device: {device}")

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
        visualize_output(config["results_folder"])


class Args:
    def __init__(self, config_path: str, example_config_path: str):
        self.config_path = config_path
        self.example_config = example_config_path


def change_config_yaml(output_path, config_dict):
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f)


def wrapped_main_func(config_path, example_config_name, text_config_dict, visualize=False, device=torch.device("cuda:0")):
    args = Args(config_path, example_config_name)
    change_config_yaml("/mnt/raid/home/eyal_michaeli/git/Text2LIVE/configs/image_example_configs/template.yaml", text_config_dict)
    if visualize:
        # plot the original image given in the text config dict
        plt.imshow(Image.open(text_config_dict["image_path"]))
        plt.show()
    main(args, config_path, visualize=visualize, device=device)


def run_on_imagenet_image(image_path, config_path, example_config_name, device):
    """
    Run the model on a single image.
    Identifies what class is it using the path (its the parent of the image path)
    And uses it to adjust the text config dict
    """
    class_name = Path(image_path).parent.name
    extra_string_for_model = random.choice(["old", "new", "pretty", "good looking", "ugly", "bad", "nice", "beautiful"])
    print(f"class_name: {class_name}, extra_string_for_model: {extra_string_for_model}")
    text_config_dict = {
                'image_path': image_path,
                'screen_text': f'{extra_string_for_model} {class_name}', 
                'comp_text': f'{extra_string_for_model} {class_name}',
                'src_text': class_name,
                'bootstrap_text': class_name,
                'bootstrap_epoch': random.choice([200, 300, 400, 500])
            }
    print(text_config_dict)
    wrapped_main_func(config_path, example_config_name, text_config_dict, visualize=True, device=device)




if __name__ == "__main__":    
    
    config_path = "./configs/image_config.yaml"
    example_config_name = f"cs2cs.yaml"
    example_config_path = f"/mnt/raid/home/eyal_michaeli/git/Text2LIVE/configs/image_example_configs/cs2cs.yaml"



    image_path = "/mnt/raid/home/eyal_michaeli/datasets/imagenet_example/ILSVRC2010_val_00020374.JPEG" # path to the input image
    text_config_dict = {
                'image_path': image_path,
                'screen_text': 'old tiger', 
                'comp_text': 'old tiger',
                'src_text': 'tiger',
                'bootstrap_text': 'tiger',
                'bootstrap_epoch': 500
            }

    args = Args(config_path=config_path,
                example_config_path=example_config_name)
    change_config_yaml(example_config_path, text_config_dict)
    main(args, config_path)
    
    exit(0)

    # the config from the yaml file (saved as text_config.yaml)
    image_path = "/mnt/raid/home/eyal_michaeli/datasets/cityscapes_flattened/images_a/test_berlin_berlin_000000_000019_leftImg8bit.png" # path to the input image
    text_config_dict = {
                'image_path': image_path,
                'screen_text': 'night', 
                'comp_text': 'cityscape at nighttime',
                'src_text': 'cityscapes',
                'bootstrap_text': 'sky',
                'bootstrap_epoch': 1000
            }

    args = Args(config_path=config_path,
                example_config_path=example_config_name)
    change_config_yaml(example_config_path, text_config_dict)
    main(args, config_path)



    # the config from the yaml file (saved as text_config.yaml)
    image_path = "/mnt/raid/home/eyal_michaeli/datasets/cityscapes_flattened/images_a/test_berlin_berlin_000000_000019_leftImg8bit.png" # path to the input image
    text_config_dict = {
                'image_path': image_path,
                'screen_text': 'snow',  # edit layer
                'comp_text': 'snowy cityscape',  # full edited image
                'src_text': 'cityscape',  # describe original
                'bootstrap_text': '',  # what to change?
                'bootstrap_epoch': 1000
            }

    args = Args(config_path=config_path,
                example_config_path=example_config_name)
    change_config_yaml(example_config_path, text_config_dict)
    main(args, config_path)



    # the config from the yaml file (saved as text_config.yaml)
    image_path = "/mnt/raid/home/eyal_michaeli/datasets/cityscapes_flattened/images_a/test_berlin_berlin_000000_000019_leftImg8bit.png" # path to the input image
    text_config_dict = {
                'image_path': image_path,
                'screen_text': 'snow',  # edit layer
                'comp_text': 'snowy cityscape',  # full edited image
                'src_text': 'cityscape',  # describe original
                'bootstrap_text': 'weather',  # what to change?
                'bootstrap_epoch': 1000
            }

    args = Args(config_path=config_path,
                example_config_path=example_config_name)
    change_config_yaml(example_config_path, text_config_dict)
    main(args, config_path)



    # the config from the yaml file (saved as text_config.yaml)
    image_path = "/mnt/raid/home/eyal_michaeli/datasets/cityscapes_flattened/images_a/test_berlin_berlin_000000_000019_leftImg8bit.png" # path to the input image
    text_config_dict = {
                'image_path': image_path,
                'screen_text': 'snow',  # edit layer
                'comp_text': 'snowy cityscape',  # full edited image
                'src_text': 'cityscape',  # describe original
                'bootstrap_text': 'trees',  # what to change?
                'bootstrap_epoch': 1000
            }

    args = Args(config_path=config_path,
                example_config_path=example_config_name)
    change_config_yaml(example_config_path, text_config_dict)
    main(args, config_path)



    # the config from the yaml file (saved as text_config.yaml)
    image_path = "/mnt/raid/home/eyal_michaeli/datasets/cityscapes_flattened/images_a/test_berlin_berlin_000000_000019_leftImg8bit.png" # path to the input image
    text_config_dict = {
                'image_path': image_path,
                'screen_text': 'autumn', 
                'comp_text': 'autumn cityscape scene',
                'src_text': 'cityscape',
                'bootstrap_text': '',
                'bootstrap_epoch': 1000
            }

    args = Args(config_path=config_path,
                example_config_path=example_config_name)
    change_config_yaml(example_config_path, text_config_dict)
    main(args, config_path)



    # the config from the yaml file (saved as text_config.yaml)
    image_path = "/mnt/raid/home/eyal_michaeli/datasets/cityscapes_flattened/images_a/test_berlin_berlin_000000_000019_leftImg8bit.png" # path to the input image
    text_config_dict = {
                'image_path': image_path,
                'screen_text': 'autumn', 
                'comp_text': 'autumn cityscape scene',
                'src_text': 'cityscape',
                'bootstrap_text': 'weather',
                'bootstrap_epoch': 1000
            }

    args = Args(config_path=config_path,
                example_config_path=example_config_name)
    change_config_yaml(example_config_path, text_config_dict)
    main(args, config_path)



    image_path = "/mnt/raid/home/eyal_michaeli/datasets/cityscapes_flattened/images_a/test_berlin_berlin_000000_000019_leftImg8bit.png" # path to the input image
    text_config_dict = {
                'image_path': image_path,
                'screen_text': 'autumn', 
                'comp_text': 'autumn cityscape scene',
                'src_text': 'cityscape',
                'bootstrap_text': 'trees',
                'bootstrap_epoch': 1000
            }

    args = Args(config_path=config_path,
                example_config_path=example_config_name)
    change_config_yaml(example_config_path, text_config_dict)
    main(args, config_path)



    image_path = "/mnt/raid/home/eyal_michaeli/datasets/cityscapes_flattened/images_a/test_berlin_berlin_000000_000019_leftImg8bit.png" # path to the input image
    text_config_dict = {
                'image_path': image_path,
                'screen_text': 'black cars', 
                'comp_text': 'cityscape with black cars',
                'src_text': 'cityscape with cars',
                'bootstrap_text': 'cars',
                'bootstrap_epoch': 1000
            }

    args = Args(config_path=config_path,
                example_config_path=example_config_name)
    change_config_yaml(example_config_path, text_config_dict)
    main(args, config_path)



    image_path = "/mnt/raid/home/eyal_michaeli/datasets/cityscapes_flattened/images_a/test_berlin_berlin_000000_000019_leftImg8bit.png" # path to the input image
    text_config_dict = {
                'image_path': image_path,
                'screen_text': 'black cars', 
                'comp_text': 'cityscape with white cars',
                'src_text': 'cityscape with cars',
                'bootstrap_text': 'cars',
                'bootstrap_epoch': 1000
            }

    args = Args(config_path=config_path,
                example_config_path=example_config_name)
    change_config_yaml(example_config_path, text_config_dict)
    main(args, config_path)




    image_path = "/mnt/raid/home/eyal_michaeli/datasets/cityscapes_flattened/images_a/test_berlin_berlin_000000_000019_leftImg8bit.png" # path to the input image
    text_config_dict = {
                'image_path': image_path,
                'screen_text': 'black cars', 
                'comp_text': 'cityscape with old cars',
                'src_text': 'cityscape with cars',
                'bootstrap_text': 'cars',
                'bootstrap_epoch': 1000
            }

    args = Args(config_path=config_path,
                example_config_path=example_config_name)
    change_config_yaml(example_config_path, text_config_dict)
    main(args, config_path)



    image_path = "/mnt/raid/home/eyal_michaeli/datasets/cityscapes_flattened/images_a/test_berlin_berlin_000000_000019_leftImg8bit.png" # path to the input image
    text_config_dict = {
                'image_path': image_path,
                'screen_text': 'black cars', 
                'comp_text': 'cityscape with black cars',
                'src_text': 'cityscape',
                'bootstrap_text': 'cars',
                'bootstrap_epoch': 1000
            }

    args = Args(config_path=config_path,
                example_config_path=example_config_name)
    change_config_yaml(example_config_path, text_config_dict)
    main(args, config_path)


    """Different IMAGE"""



    # the config from the yaml file (saved as text_config.yaml)
    image_path = "/mnt/raid/home/eyal_michaeli/datasets/cityscapes_flattened/images_a/train_extra_erlangen_erlangen_000000_000049_leftImg8bit.png" # path to the input image
    text_config_dict = {
                'image_path': image_path,
                'screen_text': 'shadow', 
                'comp_text': 'cityscape with shadow on the road',
                'src_text': 'cityscape',
                'bootstrap_text': 'road',
                'bootstrap_epoch': 1000
            }

    args = Args(config_path=config_path,
                example_config_path=example_config_name)
    change_config_yaml(example_config_path, text_config_dict)
    main(args, config_path)



    # the config from the yaml file (saved as text_config.yaml)
    image_path = "/mnt/raid/home/eyal_michaeli/datasets/cityscapes_flattened/images_a/test_berlin_berlin_000000_000019_leftImg8bit.png" # path to the input image
    text_config_dict = {
                'image_path': image_path,
                'screen_text': 'shadow', 
                'comp_text': 'road with shadow',
                'src_text': 'road',
                'bootstrap_text': 'road',
                'bootstrap_epoch': 1000
            }

    args = Args(config_path=config_path,
                example_config_path=example_config_name)
    change_config_yaml(example_config_path, text_config_dict)
    main(args, config_path)



    # the config from the yaml file (saved as text_config.yaml)
    image_path = "/mnt/raid/home/eyal_michaeli/datasets/cityscapes_flattened/images_a/train_extra_erlangen_erlangen_000000_000049_leftImg8bit.png" # path to the input image
    text_config_dict = {
                'image_path': image_path,
                'screen_text': 'buildings', 
                'comp_text': 'cityscape with old buildings',
                'src_text': 'cityscape',
                'bootstrap_text': 'buildings',
                'bootstrap_epoch': 1000
            }

    args = Args(config_path=config_path,
                example_config_path=example_config_name)
    change_config_yaml(example_config_path, text_config_dict)
    main(args, config_path)



    """
    New image
    """
        # the config from the yaml file (saved as text_config.yaml)
    image_path = "/mnt/raid/home/eyal_michaeli/datasets/cityscapes_flattened/images_a/test_berlin_berlin_000262_000019_leftImg8bit.png" # path to the input image
    text_config_dict = {
                'image_path': image_path,
                'screen_text': 'road made out of bricks', 
                'comp_text': 'cityscape with a brick road',
                'src_text': 'cityscape with a road',
                'bootstrap_text': 'road',
                'bootstrap_epoch': 1000
            }


    args = Args(config_path=config_path,
                example_config_path=example_config_name)
    change_config_yaml(example_config_path, text_config_dict)
    main(args, config_path)



       # the config from the yaml file (saved as text_config.yaml)
    image_path = "/mnt/raid/home/eyal_michaeli/datasets/cityscapes_flattened/images_a/test_berlin_berlin_000262_000019_leftImg8bit.png" # path to the input image
    text_config_dict = {
                'image_path': image_path,
                'screen_text': 'brick road', 
                'comp_text': 'cityscape with a brick road',
                'src_text': 'cityscape with a road',
                'bootstrap_text': 'road',
                'bootstrap_epoch': 1000
            }


    args = Args(config_path=config_path,
                example_config_path=example_config_name)
    change_config_yaml(example_config_path, text_config_dict)
    main(args, config_path)



       # the config from the yaml file (saved as text_config.yaml)
    image_path = "/mnt/raid/home/eyal_michaeli/datasets/cityscapes_flattened/images_a/test_berlin_berlin_000262_000019_leftImg8bit.png" # path to the input image
    text_config_dict = {
                'image_path': image_path,
                'screen_text': 'brick road', 
                'comp_text': 'cityscape with a brick road',
                'src_text': 'cityscape',
                'bootstrap_text': 'road',
                'bootstrap_epoch': 1000
            }


    args = Args(config_path=config_path,
                example_config_path=example_config_name)
    change_config_yaml(example_config_path, text_config_dict)
    main(args, config_path)



       # the config from the yaml file (saved as text_config.yaml)
    image_path = "/mnt/raid/home/eyal_michaeli/datasets/cityscapes_flattened/images_a/test_berlin_berlin_000262_000019_leftImg8bit.png" # path to the input image
    text_config_dict = {
                'image_path': image_path,
                'screen_text': 'yellow trees', 
                'comp_text': 'cityscape with yellow trees',
                'src_text': 'cityscape with trees',
                'bootstrap_text': 'trees',
                'bootstrap_epoch': 1000
            }


    args = Args(config_path=config_path,
                example_config_path=example_config_name)
    change_config_yaml(example_config_path, text_config_dict)
    main(args, config_path)



       # the config from the yaml file (saved as text_config.yaml)
    image_path = "/mnt/raid/home/eyal_michaeli/datasets/cityscapes_flattened/images_a/test_berlin_berlin_000262_000019_leftImg8bit.png" # path to the input image
    text_config_dict = {
                'image_path': image_path,
                'screen_text': 'yellow vegetation', 
                'comp_text': 'cityscape with yellow vegetation',
                'src_text': 'cityscape with vegetation',
                'bootstrap_text': 'vegetation',
                'bootstrap_epoch': 1000
            }


    args = Args(config_path=config_path,
                example_config_path=example_config_name)
    change_config_yaml(example_config_path, text_config_dict)
    main(args, config_path)



       # the config from the yaml file (saved as text_config.yaml)
    image_path = "/mnt/raid/home/eyal_michaeli/datasets/cityscapes_flattened/images_a/test_berlin_berlin_000262_000019_leftImg8bit.png" # path to the input image
    text_config_dict = {
                'image_path': image_path,
                'screen_text': 'black cars', 
                'comp_text': 'cityscape with black cars',
                'src_text': 'cityscape with cars',
                'bootstrap_text': 'cars',
                'bootstrap_epoch': 1000
            }


    args = Args(config_path=config_path,
                example_config_path=example_config_name)
    change_config_yaml(example_config_path, text_config_dict)
    main(args, config_path)



       # the config from the yaml file (saved as text_config.yaml)
    image_path = "/mnt/raid/home/eyal_michaeli/datasets/cityscapes_flattened/images_a/test_berlin_berlin_000262_000019_leftImg8bit.png" # path to the input image
    text_config_dict = {
                'image_path': image_path,
                'screen_text': 'snowy road', 
                'comp_text': 'cityscape with a road convered in snow',
                'src_text': 'cityscape with a road',
                'bootstrap_text': 'road',
                'bootstrap_epoch': 1000
            }


    args = Args(config_path=config_path,
                example_config_path=example_config_name)
    change_config_yaml(example_config_path, text_config_dict)
    main(args, config_path)




       # the config from the yaml file (saved as text_config.yaml)
    image_path = "/mnt/raid/home/eyal_michaeli/datasets/cityscapes_flattened/images_a/test_berlin_berlin_000262_000019_leftImg8bit.png" # path to the input image
    text_config_dict = {
                'image_path': image_path,
                'screen_text': 'snowy road', 
                'comp_text': 'a road convered in snow',
                'src_text': 'a road',
                'bootstrap_text': 'road',
                'bootstrap_epoch': 1000
            }


    args = Args(config_path=config_path,
                example_config_path=example_config_name)
    change_config_yaml(example_config_path, text_config_dict)
    main(args, config_path)



       # the config from the yaml file (saved as text_config.yaml)
    image_path = "/mnt/raid/home/eyal_michaeli/datasets/cityscapes_flattened/images_a/test_berlin_berlin_000262_000019_leftImg8bit.png" # path to the input image
    text_config_dict = {
                'image_path': image_path,
                'screen_text': 'a gravel road', 
                'comp_text': 'a gravel road',
                'src_text': 'a road',
                'bootstrap_text': 'road',
                'bootstrap_epoch': 1000
            }


    args = Args(config_path=config_path,
                example_config_path=example_config_name)
    change_config_yaml(example_config_path, text_config_dict)
    main(args, config_path)



       # the config from the yaml file (saved as text_config.yaml)
    image_path = "/mnt/raid/home/eyal_michaeli/datasets/cityscapes_flattened/images_a/test_berlin_berlin_000262_000019_leftImg8bit.png" # path to the input image
    text_config_dict = {
                'image_path': image_path,
                'screen_text': 'dirty road', 
                'comp_text': 'a road convered in dirt',
                'src_text': 'a road',
                'bootstrap_text': 'road',
                'bootstrap_epoch': 1000
            }


    args = Args(config_path=config_path,
                example_config_path=example_config_name)
    change_config_yaml(example_config_path, text_config_dict)
    main(args, config_path)




       # the config from the yaml file (saved as text_config.yaml)
    image_path = "/mnt/raid/home/eyal_michaeli/datasets/cityscapes_flattened/images_a/test_berlin_berlin_000262_000019_leftImg8bit.png" # path to the input image
    text_config_dict = {
                'image_path': image_path,
                'screen_text': 'clean road', 
                'comp_text': 'a clean road',
                'src_text': 'a road',
                'bootstrap_text': 'road',
                'bootstrap_epoch': 1000
            }


    args = Args(config_path=config_path,
                example_config_path=example_config_name)
    change_config_yaml(example_config_path, text_config_dict)
    main(args, config_path)

