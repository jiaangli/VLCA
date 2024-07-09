from dataclasses import dataclass, field
from enum import Enum

from omegaconf import II, MISSING


class ModelType(Enum):
    LM = "LM"
    VM = "VM"


class ExperimentsType(Enum):
    IMAGE_DISP = "image_disp"
    LANG_DISP = "lang_disp"
    FREQ = "freq"
    POLY = "poly"
    BASE = "base"


@dataclass
class ModelInfo:
    model_type: ModelType = field(
        default=ModelType.LM, metadata={"help": "Model types: LM/VM"}
    )
    model_id: str = field(
        default="bert-base-uncased", metadata={"help": "Model to use."}
    )
    dim: int = field(default=768, metadata={"help": "size of dimension."})
    model_size: float = field(
        default=MISSING, metadata={"help": "Millions of Parameters in the model."}
    )
    model_name: str = field(init=False)

    def __post_init__(self):
        if self.model_id in ["ViT-B/32", "ViT-L/14", "RN50", "RN101", "RN50x64"]:
            self.model_name = f"clip-{self.model_id.replace('/', '-')}"
        else:
            self.model_name = self.model_id.split("/")[-1]


MODEL_CONFIGS = {
    "opt-125m": ModelInfo(
        model_id="facebook/opt-125m",
        model_size=125,
        dim=768,
        model_type=ModelType.LM,
    ),
    # "opt-1.3b": ModelInfo(
    #     model_id="facebook/opt-1.3b",
    #     model_size=1300,
    #     dim=2048,
    #     model_type=ModelType.LM,
    # ),
    "opt-6.7b": ModelInfo(
        model_id="facebook/opt-6.7b",
        model_size=6700,
        dim=4096,
        model_type=ModelType.LM,
    ),
    "opt-30b": ModelInfo(
        model_id="facebook/opt-30b",
        model_size=30000,
        dim=7168,
        model_type=ModelType.LM,
    ),
    "Llama-2-7b-hf": ModelInfo(
        model_id="meta/Llama-2-7b-hf",
        model_size=7000,
        dim=4096,
        model_type=ModelType.LM,
    ),
    "Llama-2-13b-hf": ModelInfo(
        model_id="meta/Llama-2-13b-hf",
        model_size=13000,
        dim=5120,
        model_type=ModelType.LM,
    ),
    "bert_uncased_L-2_H-128_A-2": ModelInfo(
        model_id="google/bert_uncased_L-2_H-128_A-2",
        model_size=4.4,
        dim=128,
        model_type=ModelType.LM,
    ),
    "bert_uncased_L-4_H-256_A-4": ModelInfo(
        model_id="google/bert_uncased_L-4_H-256_A-4",
        model_size=11.3,
        dim=256,
        model_type=ModelType.LM,
    ),
    "bert_uncased_L-4_H-512_A-8": ModelInfo(
        model_id="google/bert_uncased_L-4_H-512_A-8",
        model_size=29.1,
        dim=512,
        model_type=ModelType.LM,
    ),
    "bert_uncased_L-8_H-512_A-8": ModelInfo(
        model_id="google/bert_uncased_L-8_H-512_A-8",
        model_size=41.7,
        dim=512,
        model_type=ModelType.LM,
    ),
    "bert-base-uncased": ModelInfo(
        model_id="bert-base-uncased",
        model_size=110,
        dim=768,
        model_type=ModelType.LM,
    ),
    "bert-large-uncased": ModelInfo(
        model_id="bert-large-uncased",
        model_size=340,
        dim=1024,
        model_type=ModelType.LM,
    ),
    "gpt2": ModelInfo(
        model_id="gpt2",
        model_size=117,
        dim=768,
        model_type=ModelType.LM,
    ),
    # "gpt2-medium": ModelInfo(
    #     model_id="gpt2-medium",
    #     model_size=345,
    #     dim=1024,
    #     model_type=ModelType.LM,
    # ),
    "gpt2-large": ModelInfo(
        model_id="gpt2-large",
        model_size=762,
        dim=1280,
        model_type=ModelType.LM,
    ),
    "gpt2-xl": ModelInfo(
        model_id="gpt2-xl",
        model_size=1542,
        dim=1600,
        model_type=ModelType.LM,
    ),
    "resnet18": ModelInfo(
        model_id="resnet18",
        dim=512,
        model_size=512,
        model_type=ModelType.VM,
    ),
    "resnet34": ModelInfo(
        model_id="resnet34",
        dim=512,
        model_type=ModelType.VM,
    ),
    "resnet50": ModelInfo(
        model_id="resnet50",
        dim=2048,
        model_type=ModelType.VM,
    ),
    "resnet101": ModelInfo(
        model_id="resnet101",
        dim=2048,
        model_type=ModelType.VM,
    ),
    "resnet152": ModelInfo(
        model_id="resnet152",
        dim=2048,
        model_type=ModelType.VM,
    ),
    "segformer-b0-finetuned-ade-512-512": ModelInfo(
        model_id="nvidia/segformer-b0-finetuned-ade-512-512",
        dim=256,
        model_type=ModelType.VM,
    ),
    "segformer-b1-finetuned-ade-512-512": ModelInfo(
        model_id="nvidia/segformer-b1-finetuned-ade-512-512",
        dim=512,
        model_type=ModelType.VM,
    ),
    "segformer-b2-finetuned-ade-512-512": ModelInfo(
        model_id="nvidia/segformer-b2-finetuned-ade-512-512",
        dim=512,
        model_type=ModelType.VM,
    ),
    "segformer-b3-finetuned-ade-512-512": ModelInfo(
        model_id="nvidia/segformer-b3-finetuned-ade-512-512",
        dim=512,
        model_type=ModelType.VM,
    ),
    "segformer-b4-finetuned-ade-512-512": ModelInfo(
        model_id="nvidia/segformer-b4-finetuned-ade-512-512",
        dim=512,
        model_type=ModelType.VM,
    ),
    "segformer-b5-finetuned-ade-640-640": ModelInfo(
        model_id="nvidia/segformer-b5-finetuned-ade-640-640",
        dim=512,
        model_type=ModelType.VM,
    ),
    "vit-mae-base": ModelInfo(
        model_id="facebook/vit-mae-base",
        dim=768,
        model_type=ModelType.VM,
    ),
    "vit-mae-large": ModelInfo(
        model_id="facebook/vit-mae-large",
        dim=1024,
        model_type=ModelType.VM,
    ),
    "vit-mae-huge": ModelInfo(
        model_id="facebook/vit-mae-huge",
        dim=1280,
        model_type=ModelType.VM,
    ),
    "clip-rn50x64": ModelInfo(
        model_id="RN50x64",
        dim=1024,
        model_type=ModelType.VM,
    ),
    "clip-rn50": ModelInfo(
        model_id="RN50",
        dim=1024,
        model_type=ModelType.VM,
    ),
    "clip-rn101": ModelInfo(
        model_id="RN101",
        dim=512,
        model_type=ModelType.VM,
    ),
    "clip-vit-b-32": ModelInfo(
        model_id="ViT-B/32",
        dim=512,
        model_type=ModelType.VM,
    ),
    "clip-vit-l-14": ModelInfo(
        model_id="ViT-L/14",
        dim=768,
        model_type=ModelType.VM,
    ),
}


@dataclass
class CommonConfig:
    seed: int = field(default=0, metadata={"help": "Seed for reproducibility."})
    # sentences_file: str = field(
    #     default="./data/sentences.json",
    #     metadata={"help": "Path to save the sentences."},
    # )
    # wordlist_file: str = field(
    #     default="./data/all_words.json",
    #     metadata={"help": "Path to save the wordlist."},
    # )
    alias_emb_dir: str = field(
        default="./data/emb",
        metadata={"help": "Path to save word embeddings (decontextualized)"},
    )
    emb_per_object: bool = field(
        default=False,
        metadata={"help": "Path to save one embedding per image or sentence"},
    )
    num_classes: int = field(
        default=100000,
        metadata={"help": "Number of classes in the dataset."},
    )
    dictionary_path: str = field(
        default="./data/dicts",
        metadata={"help": "Path to save the dictionary."},
    )


@dataclass
class DataConfig:
    dataset_name: str = field(
        default="imagenet", metadata={"help": "Name of the image dataset."}
    )
    image_dir: str = field(
        default=MISSING,
        metadata={"help": "Path to save the raw image data."},
    )
    image_id_pairs: str = field(
        default=MISSING,
        metadata={"help": "Path to save the image id pairs."},
    )


@dataclass
class MuseConfig:
    seed: int = field(
        default=II("common.seed"), metadata={"help": "Seed for reproducibility."}
    )
    exp_type: ExperimentsType = field(
        default=ExperimentsType.BASE,
        metadata={
            "help": "Different experiments settings: BASE, image_disp, lang_disp, freq, poly."
        },
    )
    lm: str = field(default="lm", metadata={"help": "Language model name."})
    vm: str = field(default="vm", metadata={"help": "Vision model name."})
    dim: int = field(default=768, metadata={"help": "Dimension of the embeddings."})
    fold: int = field(default=1, metadata={"help": "Fold number."})
    bin_name: str = field(
        default="",
        metadata={"help": "Various Bins name only for non-original experiments."},
    )
    data_type: str = field(default="cleaned", metadata={"help": "Data types: cleaned."})
    n_refinement: int = field(default=0, metadata={"help": "Number of refinements."})
    normalize_embeddings: str = field(
        default="center", metadata={"help": "Normalization."}
    )
    more_exp: bool = field(init=False)
    tgt_lang: str = field(init=False, metadata={"help": "Target language."})
    src_lang: str = field(init=False, metadata={"help": "Source language."})
    emb_dim: int = field(init=False, metadata={"help": "Dimension of the embeddings."})
    dico_eval: str = field(
        init=False, metadata={"help": "Path to load evaluation dictionary."}
    )
    dico_train: str = field(
        init=False, metadata={"help": "Path to load training dictionary."}
    )
    src_emb: str = field(
        init=False, metadata={"help": "Path to load source representations."}
    )
    tgt_emb: str = field(
        init=False, metadata={"help": "Path to load target representations."}
    )
    exp_name: str = field(
        default="", metadata={"help": "Path to log and store experiments."}
    )
    exp_path: str = field(
        default="", metadata={"help": "Path to log and store experiments."}
    )
    cuda: bool = field(default=True, metadata={"help": "Use GPU."})
    export: str = field(
        default="", metadata={"help": "Export embeddings after training (txt / pth)"}
    )

    # No need to change the following parameters
    exp_id: str = field(default="", metadata={"help": "Experiment ID."})
    max_vocab: int = field(
        default=200000, metadata={"help": "Maximum vocabulary size (-1 to disable)"}
    )
    dico_method: str = field(
        default="csls_knn_100",
        metadata={
            "help": "Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)"
        },
    )
    dico_build: str = field(
        default="S2T&T2S", metadata={"help": "S2T,T2S,S2T|T2S,S2T&T2S"}
    )
    dico_threshold: float = field(
        default=0, metadata={"help": "Threshold confidence for dictionary generation"}
    )
    dico_max_rank: int = field(
        default=10000, metadata={"help": "Maximum dictionary words rank (0 to disable)"}
    )
    dico_min_size: int = field(
        default=0, metadata={"help": "Minimum dictionary size (0 to disable)"}
    )
    dico_max_size: int = field(
        default=0, metadata={"help": "Maximum dictionary size (0 to disable)"}
    )
    verbose: int = field(default=2, metadata={"help": "Verbosity level."})
    load_optim: bool = field(
        default=False, metadata={"help": "Load optimized results."}
    )
    dico_root: str = field(
        default=f"{II('common.dictionary_path')}/{II('dataset.dataset_name')}",
        metadata={"help": "Path to save the dictionary."},
    )
    vm_emb_root: str = field(
        default=f"{II('common.alias_emb_dir')}/{ModelType.VM.value}/{II('dataset.dataset_name')}",
        metadata={"help": "Path to save the vision model embeddings."},
    )
    lm_emb_root: str = field(
        default=f"{II('common.alias_emb_dir')}/{ModelType.LM.value}",
        metadata={"help": "Path to save the language model embeddings."},
    )

    def __post_init__(self):
        self.more_exp = True if self.exp_type != ExperimentsType.BASE else False
        exp_dict_folders = {
            ExperimentsType.IMAGE_DISP: f"{self.vm}_disp",
            ExperimentsType.LANG_DISP: f"{self.vm}_disp",
            ExperimentsType.POLY: "poly",
            ExperimentsType.FREQ: "freq",
            ExperimentsType.BASE: "base",
        }

        test_dict_folder = exp_dict_folders.get(self.exp_type, "base")
        self.src_lang = self.vm
        self.tgt_lang = self.lm
        self.emb_dim = self.dim
        self.dico_train = f"{self.dico_root}/{self.exp_type.value}/train_{self.fold}_{self.data_type}.txt"
        self.dico_eval = (
            f"{self.dico_root}/{test_dict_folder}/test_{self.fold}_{self.data_type}.txt"
        )
        self.src_emb = f"{self.vm_emb_root}/{self.vm}_{self.emb_dim}.pth"
        self.tgt_emb = f"{self.lm_emb_root}/{self.lm}_{self.emb_dim}.pth"


@dataclass
class RunConfig:
    model: ModelInfo = field(default_factory=lambda: MODEL_CONFIGS["opt-125m"])
    common: CommonConfig = field(default_factory=CommonConfig)
    dataset: DataConfig = field(default_factory=DataConfig)
    muse: MuseConfig = field(default_factory=MuseConfig)
    run_muse: bool = field(default=False, metadata={"help": "Run MUSE part."})
