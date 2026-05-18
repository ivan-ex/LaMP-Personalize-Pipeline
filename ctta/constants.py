from utils import (
    extract_movie,
    extract_news_cat,
    extract_news_headline,
    extract_product_review,
    extract_scholarly_title,
    extract_tweet_paraphrasing,
)


TASK_TO_EXTRACTOR = {
    "movie_tagging": extract_movie,
    "news_categorize": extract_news_cat,
    "news_headline": extract_news_headline,
    "product_rating": extract_product_review,
    "scholarly_title": extract_scholarly_title,
    "tweet_paraphrase": extract_tweet_paraphrasing,
}


TASK_DRIFT_CONFIG = {
    "movie_tagging": {
        "semantic_fields": [
            {"field": "tag", "prefix": "tag", "weight_arg": "drift_tag_weight"},
            {"field": "description", "prefix": "description", "weight_arg": "drift_text_weight"},
        ]
    },
    "citation": {
        "semantic_fields": [
            {"field": "citation", "prefix": "citation", "weight_arg": "drift_tag_weight"},
            {"field": "title", "prefix": "title", "weight_arg": "drift_text_weight"},
        ]
    },
    "news_categorize": {
        "semantic_fields": [
            {"field": "category", "prefix": "category", "weight_arg": "drift_tag_weight"},
            {"field": "text", "prefix": "text", "weight_arg": "drift_text_weight"},
        ]
    },
    "news_headline": {
        "semantic_fields": [
            {"field": "title", "prefix": "title", "weight_arg": "drift_tag_weight"},
            {"field": "text", "prefix": "text", "weight_arg": "drift_text_weight"},
        ]
    },
    "product_rating": {
        "semantic_fields": [
            {"field": "score", "prefix": "score", "weight_arg": "drift_tag_weight"},
            {"field": "text", "prefix": "review", "weight_arg": "drift_text_weight"},
        ]
    },
    "scholarly_title": {
        "semantic_fields": [
            {"field": "title", "prefix": "title", "weight_arg": "drift_tag_weight"},
            {"field": "abstract", "prefix": "abstract", "weight_arg": "drift_text_weight"},
        ]
    },
    "tweet_paraphrase": {
        "semantic_fields": [
            {"field": "text", "prefix": "text", "weight_arg": "drift_text_weight"},
        ]
    },
}


TASK_LABEL_FIELD = {
    "movie_tagging": "tag",
    "citation": "citation",
    "news_categorize": "category",
    "news_headline": "title",
    "product_rating": "score",
    "scholarly_title": "title",
    "tweet_paraphrase": "text",
}


DISCRETE_LABELS = {
    "movie_tagging": [
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
    ],
    "news_categorize": [
        "travel",
        "education",
        "parents",
        "style & beauty",
        "entertainment",
        "food & drink",
        "science & technology",
        "business",
        "sports",
        "healthy living",
        "women",
        "politics",
        "crime",
        "culture & arts",
        "religion",
    ],
    "product_rating": ["1", "2", "3", "4", "5"],
}

