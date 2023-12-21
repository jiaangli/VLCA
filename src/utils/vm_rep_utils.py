from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel
from tqdm import tqdm
from img2vec_pytorch import Img2Vec


class ImageDataset(Dataset):
    def __init__(self, image_dir, image_classes, extractor, resolution=224) -> None:
        super(ImageDataset, self).__init__()
        self.image_root = image_dir
        self.labels = image_classes
        # self.names = [i.strip('\n').split(': ')[1] for i in self.id_name_pairs]
        self.extractor = extractor
        self.MAX_SIZE = 200
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
        inputs = torch.zeros(self.MAX_SIZE, self.CHANNELS, self.RESOLUTION_HEIGHT, self.RESOLUTION_WIDTH)
        try:
            values = self.extractor(images=images, return_tensors="pt")
            with torch.no_grad():
                inputs[:category_size,:,:,:].copy_(values.pixel_values)
        except:
            print("*"*20 + 'Failed to extract' + "*"*20, category_path)

        return inputs, (self.labels[index], category_size)

class VMEmbedding:
    def __init__(self, args, labels):
        self.args = args
        self.labels = labels
        self.pretrained_model = args.model.pretrained_model
        self.image_dir = Path(args.data.image_dir)
        self.model_name = args.model.model_name
        self.bs = 1
        self.per_image = args.data.emb_per_object if "huge" in self.model_name else False
        self.alias_emb_dir = Path(args.data.alias_emb_dir) / args.model.model_type / args.data.dataset_name
        # self.is_average = args.model.i/s_avg
        self.device = [i for i in range(torch.cuda.device_count())] if torch.cuda.device_count() >= 1 else ["cpu"]

    def get_vm_layer_representations(self):

        if self.model_name.startswith("seg"):
            resolution = int(self.model_name[-3:])
        else:
            resolution = 224
        if self.model_name.startswith("res"):
            img2vec = Img2Vec(model=self.model_name, cuda=torch.cuda.is_available())
            categories_encode = []
            image_categories = []
            with torch.no_grad():
                for i in range(len(self.labels)):
                    images = []
                    category_path = self.image_dir / self.labels[i]
                    for filename in category_path.iterdir():
                        try:
                            images.append(Image.open(category_path / filename).convert('RGB'))
                        except:
                            print('Failed to pil', filename)
                    
                    categories_encode.append(img2vec.get_vec(images, tensor=True).to('cpu').numpy().squeeze().mean(axis=0))
                    image_categories.append(self.labels[i])
                        # print(features[0].shape)
                embeddings = np.vstack(categories_encode)
                dim_size = embeddings.shape[1]
                save_category_path = self.alias_emb_dir / f"{self.model_name}_{dim_size}.pth"
                torch.save({"dico": image_categories, "vectors": torch.from_numpy(embeddings).float()}, 
                            str(save_category_path))
                print("save to ", str(save_category_path))
            
        else:
            cache_path = Path.home() / ".cache/huggingface/transformers/models" / self.pretrained_model
            feature_extractor = AutoFeatureExtractor.from_pretrained(self.pretrained_model, cache_dir=cache_path)
            if not self.model_name.startswith("resnet"):
                model = AutoModel.from_pretrained(self.pretrained_model, cache_dir=cache_path, output_hidden_states=True, return_dict=True)
                model = model.to(self.device[0])

            model = model.eval()

            imageset = ImageDataset(self.image_dir, self.labels, feature_extractor, resolution)
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
                    if self.model_name.startswith("vit"):
                        chunks = torch.chunk(outputs.hidden_states[-1][:,1:,:].cpu(), inputs_shape[0], dim=0)
                    else:
                        chunks = torch.chunk(outputs.last_hidden_state.cpu(), inputs_shape[0], dim=0)


                    for idx, chip in enumerate(chunks):
                        # features for every image
                        if self.model_name.startswith("vit"):
                            images_features = np.mean(chip[:category_size[idx]].numpy(), axis=1, keepdims=True).squeeze()
                        else:
                            images_features = np.mean(chip[:category_size[idx]].numpy(), axis=(2,3), keepdims=True).squeeze()
                        # features for categories
                        category_feature = np.expand_dims(images_features.mean(axis=0), 0)
                        image_categories.append(names[idx])
                        categories_encode.append(category_feature)
                        
                        if self.per_image:
                            images_name = [f"{names[idx]}_{i}" for i in range(category_size[idx])]
                            save_per_path = Path("/projects/nlp/people/kfb818/Dir/datasets/dispersions") / self.args.data.dataset_name / self.model_name 
                            if not save_per_path.exists():
                                save_per_path.mkdir(parents=True, exist_ok=True)
                            torch.save({"dico": images_name, "vectors": torch.from_numpy(images_features).float()}, 
                                    str(save_per_path / f"{names[idx]}.pth"))


            embeddings = np.vstack(categories_encode)
        #   categories_encode = np.concatenate(categories_encode)
            dim_size = embeddings.shape[1]
            save_category_path = self.alias_emb_dir / f"{self.model_name}_{dim_size}.pth"
            torch.save({"dico": image_categories, "vectors": torch.from_numpy(embeddings).float()}, 
                        str(save_category_path))
            print("save to ", str(save_category_path))


            