import json

from datasets import Dataset

from utils import get_first_k_tokens


class DataMixin:
    def load_data(self):
        with open(self.train_data_path, "r") as f:
            self.test_data = json.load(f)

        with open("./prompt/prompt.json", "r") as f:
            self.prompt_template = json.load(f)

        if self.args.add_profile:
            with open(f"./data/{self.args.task_name}/profile_user_100.json", "r") as f:
                self.test_profile = json.load(f)

    def get_memory_slice(self, entries):
        if self.args.memory_size is None or self.args.memory_size <= 0:
            return entries
        return entries[-self.args.memory_size:]

    def get_bm25_class(self):
        from rank_bm25 import BM25Okapi

        return BM25Okapi

    def tokenize(self, prompt, add_eos_token=True):
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.args.cut_off,
            padding=False,
            return_tensors=None,
        )

        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.args.cut_off
            and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(self, data_point):
        tokenized_full_prompt = self.tokenize(data_point["full_prompt"])
        tokenized_user_prompt = self.tokenize(data_point["prompt"], add_eos_token=self.add_eos_token)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        if self.add_eos_token:
            user_prompt_len -= 1

        tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
        return tokenized_full_prompt

    def build_train_data_for_entries(self, profile_entries, profile_prefix=None, start_idx=0, end_idx=None):
        train_data = []
        if end_idx is None:
            end_idx = len(profile_entries)

        for idx in range(start_idx, end_idx):
            q = {key: get_first_k_tokens(value, 768) for key, value in profile_entries[idx].items()}

            prompt = self.prompt_template[self.args.task_name]["OPPU_input"].format(**q)
            full_prompt = self.prompt_template[self.args.task_name]["OPPU_full"].format(**q)

            if self.args.k > 0 and idx != 0 and self.format_flag:
                visible_history_list = []
                memory_history = self.get_memory_slice(profile_entries[:idx])
                for history_item in memory_history:
                    visible_history_list.append(
                        {key: get_first_k_tokens(value, 768) for key, value in history_item.items()}
                    )

                history_list = [
                    self.prompt_template[self.args.task_name]["retrieval_history"].format(**p)
                    for p in visible_history_list
                ]
                tokenized_corpus = [doc.split(" ") for doc in history_list]
                bm25 = self.get_bm25_class()(tokenized_corpus)

                tokenized_query = self.prompt_template[self.args.task_name]["retrieval_query"].format(**q).split(" ")
                retrieved_history = bm25.get_top_n(tokenized_query, history_list, n=self.args.k)
                history_string = "".join(retrieved_history)
                prompt = history_string + "\n" + prompt
                full_prompt = history_string + "\n" + full_prompt

            if self.args.add_profile and self.format_flag and profile_prefix:
                prompt = profile_prefix + "\n" + prompt
                full_prompt = profile_prefix + "\n" + full_prompt

            train_data.append({"prompt": prompt, "full_prompt": full_prompt})

        return train_data

    def build_train_data_for_indices(self, profile_entries, indices, profile_prefix=None):
        train_data = []
        for idx in indices:
            train_data.extend(
                self.build_train_data_for_entries(
                    profile_entries,
                    profile_prefix=profile_prefix,
                    start_idx=idx,
                    end_idx=idx + 1,
                )
            )
        return train_data

    def build_tokenized_train_dataset(self, train_data, shuffle=False):
        train_dataset = Dataset.from_list(train_data)
        tokenized_dataset = train_dataset.map(
            self.generate_and_tokenize_prompt,
            remove_columns=train_dataset.column_names,
            load_from_cache_file=False,
        )
        if shuffle:
            tokenized_dataset = tokenized_dataset.shuffle(seed=42)
        return tokenized_dataset

