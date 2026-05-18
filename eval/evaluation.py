import glob
import json
import os
import shutil
import zipfile

os.environ.setdefault("HF_EVALUATE_OFFLINE", "1")

import evaluate


METRICS_DIR_CANDIDATES = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "metrics")),
    "/home/xuyifan/metrics",
]


def load_local_metric(metric_name):
    errors = []
    for metrics_dir in METRICS_DIR_CANDIDATES:
        metric_path = os.path.join(metrics_dir, metric_name)
        if not os.path.exists(metric_path):
            continue
        try:
            return evaluate.load(metric_path)
        except Exception as exc:
            errors.append(f"{metric_path}: {exc}")

    checked_paths = [os.path.join(metrics_dir, metric_name) for metrics_dir in METRICS_DIR_CANDIDATES]
    error_suffix = f" Load errors: {'; '.join(errors)}" if errors else ""
    raise FileNotFoundError(
        f"Cannot load local metric '{metric_name}'. Checked: {checked_paths}.{error_suffix}"
    )

def postprocess_text_classification(preds, labels):
    preds = [str(pred).strip() for pred in preds]
    labels = [str(label).strip() for label in labels]
    return preds, labels

def postprocess_text_generation(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def create_metric_f1_accuracy(all_labels):
    f1_metric = load_local_metric("f1")
    accuracy_metric = load_local_metric("accuracy")

    def create_mapping(x):
        try:
            return all_labels.index(x)
        except ValueError:
            return -1
    def compute_metrics(decoded_preds, decoded_labels):
        decoded_preds, decoded_labels = postprocess_text_classification(decoded_preds, decoded_labels)
        decoded_preds = [create_mapping(x) for x in decoded_preds]
        decoded_labels = [create_mapping(x) for x in decoded_labels]
        result_acc = accuracy_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result_f1 = f1_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            labels=list(range(len(all_labels))),
            average="macro",
        )
        return {"accuracy": result_acc["accuracy"], "f1": result_f1["f1"]}
    return compute_metrics

def create_metric_mae_rmse():
    mse_metric = load_local_metric("mse")
    mae_metric = load_local_metric("mae")

    def create_mapping(x, y):
        try:
            return float(x)
        except (TypeError, ValueError):
            print(x)
            y = float(y)
            if abs(1 - y) > abs(5 - y):
                return 1.0
            else:
                return 5.0
    def compute_metrics(decoded_preds, decoded_labels):
        decoded_preds, decoded_labels = postprocess_text_classification(decoded_preds, decoded_labels)
        decoded_preds = [create_mapping(x,y) for x,y in zip(decoded_preds, decoded_labels)]
        decoded_labels = [create_mapping(x,x) for x in decoded_labels]
        result_mae = mae_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result_rmse = mse_metric.compute(predictions=decoded_preds, references=decoded_labels, squared=False)
        return {"MAE": result_mae["mae"], "RMSE": result_rmse["mse"]}
    return compute_metrics

def create_metric_rouge():
    rouge_metric = load_local_metric("rouge")

    def compute_metrics(decoded_preds, decoded_labels):
        decoded_preds, decoded_labels = postprocess_text_generation(decoded_preds, decoded_labels)
        result_rouge = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"rouge-1": result_rouge["rouge1"], "rouge-L": result_rouge["rougeL"]}
    return compute_metrics


def compute_metrics_for_task(task_name, predictions, golds):
    if task_name in ["LaMP_1", "LaMP_2N", "LaMP_2M"]:
        metric = create_metric_f1_accuracy(get_labels_for_task(task_name))
    elif task_name == "LaMP_3":
        metric = create_metric_mae_rmse()
    else:
        metric = create_metric_rouge()
    return metric(predictions, golds)


def get_labels_for_task(task_name):
    if task_name == "LaMP_1":
        return ["[1]", "[2]"]
    if task_name == "LaMP_2N":
        return [
            "food & drink",
            "sports",
            "education",
            "parents",
            "religion",
            "travel",
            "business",
            "crime",
            "science & technology",
            "culture & arts",
            "entertainment",
            "politics",
            "women",
            "style & beauty",
            "healthy living",
        ]
    if task_name == "LaMP_2M":
        return [
            "sci-fi",
            "based on a book",
            "comedy",
            "action",
            "twist ending",
            "dystopia",
            "dark comedy",
            "classic",
            "psychology",
            "fantasy",
            "romance",
            "thought-provoking",
            "social commentary",
            "violence",
            "true story",
        ]
    if task_name == "LaMP_3":
        return ["1", "2", "3", "4", "5"]
    raise ValueError("Invalid task_name")

class LaMPEvaluation(object):
    
    def __init__(self, all_golds_zip_file_addr = None, single_gold_json_file_addr = None, extract_addr = "./tmp", skip_missing_gold_ids = False) -> None:
        assert all_golds_zip_file_addr or single_gold_json_file_addr, "The golds should be provided for all datasets or at least one."
        assert not (all_golds_zip_file_addr and single_gold_json_file_addr), "The golds should be provided using zip file or json file not both."
        self.tasks_golds = dict()
        self.extract_addr = extract_addr
        self.evaluate_all_is_possible = False
        self.skip_missing_gold_ids = skip_missing_gold_ids
        if all_golds_zip_file_addr:
            os.makedirs(self.extract_addr, exist_ok=True)
            with zipfile.ZipFile(all_golds_zip_file_addr, 'r') as zobj:
                zobj.extractall(path = extract_addr)
            for file_addr in glob.glob(os.path.join(self.extract_addr, "**/*.json"), recursive=True):
                with open(file_addr) as file:
                    task = json.load(file)
                    self.tasks_golds[task['task']] = task['golds']
            self._empty_dir(self.extract_addr)
            self.evaluate_all_is_possible = True
        if single_gold_json_file_addr:
            with open(single_gold_json_file_addr) as file:
                    task = json.load(file)
                    self.tasks_golds[task['task']] = task['golds']
    
    def _empty_dir(self, directory_path):
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

    def _get_all_gold_ids(self, task_name):
        return set([sample['id'] for sample in self.tasks_golds[task_name]])
    
    def _get_all_ids(self, input):
        return set([sample['id'] for sample in input])
    
    def evaluate_all(self, predicts_zipfile_addr):
        assert self.evaluate_all_is_possible, "You did not provide golds for all tasks."
        with zipfile.ZipFile(predicts_zipfile_addr, 'r') as zobj:
            zobj.extractall(path = self.extract_addr)
        results_raw = dict()
        all_task_names = set()
        for file_addr in glob.glob(os.path.join(self.extract_addr, "**/*.json"), recursive=True):
            with open(file_addr) as file:
                preds = json.load(file)
            all_task_names.add(preds['task'])
            results_raw[preds['task']] = self._evaluate_task(preds['golds'], preds['task'])
        self._empty_dir(self.extract_addr)
        assert len(all_task_names) == 7, "The provided results do not cover all the tasks in the benchmark."
        return results_raw

    def evaluate_task(self, predicts_json_addr, task_name):
        with open(predicts_json_addr) as file:
            preds = json.load(file)
        assert preds['task'] == task_name, "The provided task_name and the results do not match."
        assert preds['task'] in self.tasks_golds.keys(), "The provided golds cannot be used to evaluate this task."
        return self._evaluate_task(preds['golds'], task_name)

    def _evaluate_task(self, predictions, task_name):
        golds_dict = {str(sample['id']): sample['output'] for sample in self.tasks_golds[task_name]}
        pred_ids = [str(sample['id']) for sample in predictions]

        missing_gold_ids = [sample_id for sample_id in pred_ids if sample_id not in golds_dict]
        if missing_gold_ids and not self.skip_missing_gold_ids:
            raise AssertionError(
                "Some prediction ids cannot be found in the gold label table: {}".format(missing_gold_ids[:10])
            )

        if self.skip_missing_gold_ids:
            filtered_predictions = [sample for sample in predictions if str(sample['id']) in golds_dict]
            skipped_count = len(predictions) - len(filtered_predictions)
            if skipped_count:
                print(
                    f"Warning: skipped {skipped_count} predictions whose ids were not found in the gold label table."
                )
            predictions = filtered_predictions
            if not predictions:
                raise ValueError("No prediction ids matched the gold label table after filtering.")

        # Align labels by prediction ids so evaluation no longer depends on
        # prediction order or on providing the full gold subset.
        golds = [golds_dict[str(sample['id'])] for sample in predictions]
        preds = [sample['output'] for sample in predictions]
        return compute_metrics_for_task(task_name, preds, golds)
    
    def _get_labels(self, task_name):
        return get_labels_for_task(task_name)
