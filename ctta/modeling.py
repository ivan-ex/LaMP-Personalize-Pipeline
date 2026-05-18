import torch
import transformers
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils import print_trainable_parameters


class ModelMixin:
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name,
            padding_side="left",
            token=self.args.access_token,
        )
        self.tokenizer.eos_token = "</s>"
        self.tokenizer.pad_token = "[PAD]"
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        quantization_config = None
        if self.quantized_loading:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            local_files_only=False,
            device_map=self.args.device_map,
            trust_remote_code=True,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
        )

        self.base_model.config.use_cache = False
        self.base_model.config.pad_token_id = self.tokenizer.pad_token_id
        self.base_model.config.eos_token_id = self.tokenizer.eos_token_id
        self.base_model.config.bos_token_id = self.tokenizer.bos_token_id

        self.base_model.gradient_checkpointing_enable()
        if getattr(self.base_model, "is_loaded_in_4bit", False) or getattr(self.base_model, "is_loaded_in_8bit", False):
            self.base_model = prepare_model_for_kbit_training(self.base_model)
        self.base_model.enable_input_require_grads()

        self.peft_config = LoraConfig(
            r=8,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.training_arguments = transformers.TrainingArguments(
            output_dir="output/",
            per_device_train_batch_size=self.args.batch_size,
            gradient_accumulation_steps=1,
            optim="adamw_torch",
            num_train_epochs=self.args.max_epoch,
            save_strategy="no",
            save_steps=10**9,
            logging_steps=50,
            learning_rate=1e-4,
            weight_decay=1e-2,
            bf16=True,
            max_grad_norm=0.3,
            warmup_ratio=0.1,
            group_by_length=False,
            lr_scheduler_type="linear",
            report_to="none",
        )
        self.lora_probe_training_arguments = transformers.TrainingArguments(
            output_dir="output/lora_drift_probe/",
            per_device_train_batch_size=self.args.batch_size,
            gradient_accumulation_steps=1,
            optim="adamw_torch",
            num_train_epochs=self.args.lora_drift_probe_epochs,
            save_strategy="no",
            save_steps=10**9,
            logging_steps=10**9,
            learning_rate=self.args.lora_drift_probe_lr or self.training_arguments.learning_rate,
            weight_decay=1e-2,
            bf16=True,
            max_grad_norm=0.3,
            warmup_ratio=0.0,
            group_by_length=False,
            lr_scheduler_type="linear",
            report_to="none",
        )

        task_lora_model = PeftModel.from_pretrained(
            model=self.base_model,
            model_id=self.args.task_lora,
            is_trainable=False,
        )
        self.base_model = task_lora_model.merge_and_unload()
        print_trainable_parameters(task_lora_model)
        del task_lora_model
        self.cleanup_memory()
        self.task_adapter_mode = "merge_task_adapter_then_add_user_adapter"

    def create_user_lora_model(self):
        model = get_peft_model(self.base_model, self.peft_config)
        print_trainable_parameters(model)
        model.gradient_checkpointing_enable()
        return model

    def unload_user_lora_model(self, model):
        peft_base = getattr(model, "base_model", None)
        if peft_base is not None and hasattr(peft_base, "unload"):
            return peft_base.unload()
        if hasattr(model, "unload"):
            return model.unload()
        return model
