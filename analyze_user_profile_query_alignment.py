import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_request_description(text: str) -> str:
    match = re.search(r"description:\s*(.*)$", text, re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


def tokenize(text: str):
    return set(re.findall(r"[a-zA-Z]+", text.lower()))


def cosine(counter_a: Counter, counter_b: Counter) -> float:
    keys = set(counter_a) | set(counter_b)
    dot = sum(counter_a[k] * counter_b[k] for k in keys)
    norm_a = math.sqrt(sum(v * v for v in counter_a.values()))
    norm_b = math.sqrt(sum(v * v for v in counter_b.values()))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def summarize_user(user: dict):
    profile_tags = Counter(item["tag"] for item in user.get("profile", []))
    query_golds = Counter(item["gold"] for item in user.get("query", []))
    profile_vocab = tokenize(" ".join(item["description"] for item in user.get("profile", [])))
    request_descs = [extract_request_description(item["input"]) for item in user.get("query", [])]

    lexical_scores = []
    for desc in request_descs:
        query_vocab = tokenize(desc)
        union = profile_vocab | query_vocab
        lexical_scores.append(len(profile_vocab & query_vocab) / len(union) if union else 0.0)

    dominant_profile_tag = profile_tags.most_common(1)[0][0] if profile_tags else None
    dominant_hit_rate = (
        sum(gold == dominant_profile_tag for gold in query_golds.elements()) / sum(query_golds.values())
        if query_golds
        else 0.0
    )
    profile_tag_coverage = (
        sum(gold in profile_tags for gold in query_golds.elements()) / sum(query_golds.values())
        if query_golds
        else 0.0
    )

    return {
        "profile_count": len(user.get("profile", [])),
        "query_count": len(user.get("query", [])),
        "profile_tags": profile_tags,
        "query_golds": query_golds,
        "tag_cosine": cosine(profile_tags, query_golds),
        "dominant_hit_rate": dominant_hit_rate,
        "profile_tag_coverage": profile_tag_coverage,
        "mean_lexical_jaccard": mean(lexical_scores) if lexical_scores else 0.0,
    }


def print_dataset_metrics(name: str, summaries: list[dict]):
    print(f"\n[{name}]")
    print(f"user_count: {len(summaries)}")
    print(
        "query_count mean/median:",
        round(mean(item["query_count"] for item in summaries), 2),
        median(item["query_count"] for item in summaries),
    )
    print(
        "tag_cosine mean/median:",
        round(mean(item["tag_cosine"] for item in summaries), 4),
        round(median(item["tag_cosine"] for item in summaries), 4),
    )
    print(
        "dominant_tag_hit mean/median:",
        round(mean(item["dominant_hit_rate"] for item in summaries), 4),
        round(median(item["dominant_hit_rate"] for item in summaries), 4),
    )
    print(
        "profile_tag_coverage mean/median:",
        round(mean(item["profile_tag_coverage"] for item in summaries), 4),
        round(median(item["profile_tag_coverage"] for item in summaries), 4),
    )
    print(
        "lexical_jaccard mean/median:",
        round(mean(item["mean_lexical_jaccard"] for item in summaries), 4),
        round(median(item["mean_lexical_jaccard"] for item in summaries), 4),
    )


def print_examples(name: str, rows: list[dict], top_n: int):
    print(f"\n[{name} examples: high alignment]")
    for row in sorted(rows, key=lambda item: item["summary"]["tag_cosine"], reverse=True)[:top_n]:
        summary = row["summary"]
        print(
            row["user_id"],
            "cosine=", round(summary["tag_cosine"], 4),
            "profile_tags=", summary["profile_tags"].most_common(3),
            "query_golds=", summary["query_golds"].most_common(3),
            "profile/query=", (summary["profile_count"], summary["query_count"]),
        )

    print(f"\n[{name} examples: low alignment]")
    for row in sorted(rows, key=lambda item: item["summary"]["tag_cosine"])[:top_n]:
        summary = row["summary"]
        print(
            row["user_id"],
            "cosine=", round(summary["tag_cosine"], 4),
            "profile_tags=", summary["profile_tags"].most_common(3),
            "query_golds=", summary["query_golds"].most_common(3),
            "profile/query=", (summary["profile_count"], summary["query_count"]),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        default="/data1/xuyifan/OPPU/data/movie_tagging",
        help="Directory containing all_user.json and dev/dev_user.json",
    )
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    all_users = load_json(dataset_dir / "all_user.json")
    dev_users = load_json(dataset_dir / "dev" / "dev_user.json")

    all_map = {user["user_id"]: user for user in all_users}
    dev_map = {user["user_id"]: user for user in dev_users}
    shared_ids = sorted(set(all_map) & set(dev_map))

    print("[dataset overlap]")
    print("all_user_count:", len(all_users))
    print("dev_user_count:", len(dev_users))
    print("shared_user_count:", len(shared_ids))
    print("all_only_user_count:", len(set(all_map) - set(dev_map)))
    print("dev_only_user_count:", len(set(dev_map) - set(all_map)))
    print("same_profile_count:", sum(all_map[uid]["profile"] == dev_map[uid]["profile"] for uid in shared_ids))

    all_rows = [{"user_id": uid, "summary": summarize_user(all_map[uid])} for uid in shared_ids]
    dev_rows = [{"user_id": uid, "summary": summarize_user(dev_map[uid])} for uid in shared_ids]

    print_dataset_metrics("all_user(shared ids)", [row["summary"] for row in all_rows])
    print_dataset_metrics("dev_user", [row["summary"] for row in dev_rows])
    print_examples("all_user(shared ids)", all_rows, args.top_n)
    print_examples("dev_user", dev_rows, args.top_n)


if __name__ == "__main__":
    main()
