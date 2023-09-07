import re
import time as tm
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel
from tqdm import tqdm


class ImageDataset(Dataset):
    def __init__(self, image_dir, image_classes, extractor, resolution=224) -> None:
        super(ImageDataset, self).__init__()
        self.image_root = image_dir
        self.labels = image_classes
        # self.names = [i.strip('\n').split(': ')[1] for i in self.id_name_pairs]
        self.extractor = extractor
        self.MAX_SIZE = 50
        self.RESOLUTION_HEIGHT = resolution
        self.RESOLUTION_WIDTH = resolution
        self.CHANNELS = 3

    def __len__(self):
        # return len(self.id_name_pairs)
        return len(self.labels)
    
    def __getitem__(self, index):
        images = []
        category_path = self.image_root / self.labels[index]
        for filename in category_path.iterdir():
            try:
                images.append(Image.open(category_path / filename).convert('RGB'))
            except:
                print('Failed to pil', filename)

        category_size = len(images)
        values = self.extractor(images=images, return_tensors="pt")
        inputs = torch.zeros(self.MAX_SIZE, self.CHANNELS, self.RESOLUTION_HEIGHT, self.RESOLUTION_WIDTH)
        with torch.no_grad():
            inputs[:category_size,:,:,:].copy_(values.pixel_values)

        return inputs, (self.labels[index], category_size)

class VMEmbedding:
    def __init__(self, args, labels):
        self.args = args
        self.labels = labels
        self.pretrained_model = args.model.pretrained_model
        self.image_dir = Path(args.data.image_dir)
        self.model_name = args.model.model_name
        self.bs = 8
        self.alias_emb_dir = Path(args.data.alias_emb_dir)/ "averaged" / self.model_name
        # self.is_average = args.model.i/s_avg
        self.device = [i for i in range(torch.cuda.device_count())] if torch.cuda.device_count() >= 1 else ["cpu"]

    def get_vm_layer_representations(self):
        cache_path = Path.home() / ".cache/huggingface/transformers/models" / self.pretrained_model
        feature_extractor = AutoFeatureExtractor.from_pretrained(self.pretrained_model, cache_dir=cache_path)
        if not self.model_name.startwith("resnet"):
            model = AutoModel.from_pretrained(self.pretrained_model, cache_dir=cache_path, output_hidden_states=True, return_dict=True)
            model = model.to(self.device[0])

        model.eval()

        imageset = ImageDataset(self.image_dir, self.labels, feature_extractor )
        image_dataloader = torch.utils.data.DataLoader(imageset, batch_size=self.bs, num_workers=4, pin_memory=True)

        # images_name = []
        categories_encode = []
        image_categories = []

        for inputs, (names, category_size) in tqdm(image_dataloader, mininterval=60.0, maxinterval=360.0):
            inputs_shape = inputs.shape
            inputs = inputs.reshape(-1, inputs_shape[2],inputs_shape[3],inputs_shape[4]).to(self.device[0])
            
            with torch.no_grad():
                outputs = model(pixel_values=inputs)
                # chunks = torch.chunk(outputs.last_hidden_state[:,0,:].cpu(), inputs_shape[0], dim=0)
                chunks = torch.chunk(outputs.last_hidden_state[:,1:,:].cpu(), inputs_shape[0], dim=0)

                for idx, chip in enumerate(chunks):
                    # features for every image
                    # images_features = np.mean(chip[:category_size[idx]].numpy(), axis=(2,3), keepdims=True).squeeze()
                    images_features = chip[:category_size[idx]].numpy()
                    # features for categories
                    category_feature = np.expand_dims(images_features.mean(axis=0), 0)
                    image_categories.append(names[idx])
                    # images_name = [f"{names[idx]}_{i}" for i in range(category_size[idx])]

                    categories_encode.append(category_feature)


        embeddings = np.vstack(categories_encode)
    #   categories_encode = np.concatenate(categories_encode)
        dim_size = embeddings.shape[1]

        torch.save({"dico": image_categories, "vectors": torch.from_numpy(embeddings).float()}, 
                    str(self.alias_emb_dir / f"imagenet_{self.model_name}_dim_{dim_size}.pth"))
            