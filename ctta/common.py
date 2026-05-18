import torch
import torch.nn.functional as F
import transformers


def normalize_text(text):
    text = str(text).strip().lower()
    return " ".join(text.split())


def cast_norm_modules_to_float32(model):
    for name, module in model.named_modules():
        if "norm" in name:
            module.to(torch.float32)


def snapshot_trainable_params(model):
    return {
        name: param.detach().clone().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def restore_trainable_params(model, params):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad and name in params:
                param.copy_(params[name].to(device=param.device, dtype=param.dtype, non_blocking=True))


def compute_trainable_param_distance(old_params, new_params):
    squared_delta = 0.0
    squared_old = 0.0
    squared_new = 0.0
    dot_product = 0.0
    abs_delta_sum = 0.0
    param_count = 0

    for name, old_param in old_params.items():
        new_param = new_params.get(name)
        if new_param is None:
            continue

        old_flat = old_param.detach().cpu().to(torch.float32).reshape(-1)
        new_flat = new_param.detach().cpu().to(torch.float32).reshape(-1)
        delta = new_flat - old_flat
        squared_delta += float(torch.dot(delta, delta))
        squared_old += float(torch.dot(old_flat, old_flat))
        squared_new += float(torch.dot(new_flat, new_flat))
        dot_product += float(torch.dot(old_flat, new_flat))
        abs_delta_sum += float(torch.abs(delta).sum())
        param_count += delta.numel()

    l2_delta = squared_delta ** 0.5
    old_norm = squared_old ** 0.5
    new_norm = squared_new ** 0.5
    relative_l2 = l2_delta / max(old_norm, 1e-12)
    if old_norm <= 1e-12 or new_norm <= 1e-12:
        cosine_distance = 0.0
    else:
        cosine_sim = dot_product / max(old_norm * new_norm, 1e-12)
        cosine_sim = max(min(cosine_sim, 1.0), -1.0)
        cosine_distance = 1.0 - cosine_sim

    return {
        "relative_l2": relative_l2,
        "l2": l2_delta,
        "mean_abs_delta": abs_delta_sum / param_count if param_count else 0.0,
        "cosine_distance": cosine_distance,
        "param_count": param_count,
    }


class CTTAContinualTrainer(transformers.Trainer):
    def __init__(
        self,
        old_params=None,
        anchor_lambda=0.0,
        lwf_lambda=0.0,
        distill_temperature=2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.old_params = old_params
        self.anchor_lambda = anchor_lambda
        self.lwf_lambda = lwf_lambda
        self.distill_temperature = distill_temperature
        self._old_param_device_cache = {}

    def _old_param_on_device(self, name, param):
        cache_key = (name, str(param.device), param.dtype)
        old_param = self._old_param_device_cache.get(cache_key)
        if old_param is None or old_param.shape != param.shape:
            old_param = self.old_params[name].to(device=param.device, dtype=param.dtype, non_blocking=True)
            self._old_param_device_cache[cache_key] = old_param
        return old_param

    def _compute_teacher_logits(self, model, inputs):
        cached_params = {}
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.requires_grad and name in self.old_params:
                        cached_params[name] = param.detach().clone()
                        param.copy_(self._old_param_on_device(name, param))
                return model(**inputs).logits.detach()
        finally:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in cached_params:
                        param.copy_(cached_params[name])
            if was_training:
                model.train()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss

        if self.old_params and self.anchor_lambda > 0:
            anchor_loss = torch.zeros((), device=loss.device, dtype=loss.dtype)
            anchor_count = 0
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.old_params:
                    old_param = self._old_param_on_device(name, param)
                    anchor_term = torch.sum((param - old_param) ** 2).to(device=loss.device, dtype=loss.dtype)
                    anchor_loss = anchor_loss + anchor_term
                    anchor_count += param.numel()
            if anchor_count > 0:
                loss = loss + self.anchor_lambda * (anchor_loss / anchor_count)

        if self.old_params and self.lwf_lambda > 0:
            teacher_logits = self._compute_teacher_logits(model, inputs)
            temperature = self.distill_temperature
            student_log_probs = F.log_softmax(outputs.logits / temperature, dim=-1)
            teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
            token_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)

            labels = inputs.get("labels")
            if labels is not None:
                mask = labels.ne(-100)
            else:
                mask = inputs.get("attention_mask")

            if mask is not None and mask.sum() > 0:
                distill_loss = (token_kl * mask.to(token_kl.dtype)).sum() / mask.sum()
            else:
                distill_loss = token_kl.mean()
            loss = loss + self.lwf_lambda * distill_loss * (temperature ** 2)

        return (loss, outputs) if return_outputs else loss
