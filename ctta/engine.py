import torch

from ctta.config import get_train_data_path, resolve_drift_tag
from ctta.constants import TASK_TO_EXTRACTOR
from ctta.data import DataMixin
from ctta.drift import DriftMixin
from ctta.inference import InferenceMixin
from ctta.modeling import ModelMixin
from ctta.outputs import OutputMixin
from ctta.pipeline import PipelineMixin
from ctta.training import TrainingMixin


class CTTAEngine(
    DataMixin,
    DriftMixin,
    TrainingMixin,
    InferenceMixin,
    ModelMixin,
    OutputMixin,
    PipelineMixin,
):
    def __init__(self, args):
        self.args = args
        self.quantized_loading = args.load_in_8bit or args.load_in_4bit
        self.train_data_path = get_train_data_path(args)
        self.drift_tag = resolve_drift_tag(args, self.train_data_path)
        self.infer_batch_size = args.infer_batch_size or args.batch_size
        self.add_eos_token = False
        self.tokenizer = None
        self.base_model = None
        self.peft_config = None
        self.training_arguments = None
        self.lora_probe_training_arguments = None
        self.prompt_template = None
        self.test_data = None
        self.test_profile = None
        self.task_adapter_mode = None
        self.semantic_encoder = None
        self.semantic_embedding_cache = {}
        self.semantic_entry_embedding_cache = {}
        self.format_flag = args.task_name != "tweet_paraphrase"
        self.extract_article = TASK_TO_EXTRACTOR.get(args.task_name)

    def cleanup_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
