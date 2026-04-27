# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


def get_adanos_feature_config():
    fields = [
        "Ref($retail_buzz_avg, 1)",
        "Ref($retail_sentiment_avg, 1)",
        "Ref($retail_coverage, 1)",
        "Ref($retail_alignment_score, 1)",
        "Ref($reddit_buzz, 1)",
        "Ref($x_buzz, 1)",
        "Ref($news_buzz, 1)",
        "Ref($polymarket_buzz, 1)",
        "Ref($reddit_mentions, 1)",
        "Ref($x_mentions, 1)",
        "Ref($news_mentions, 1)",
        "Ref($polymarket_trade_count, 1)",
        "Ref($retail_buzz_avg, 1)/(Mean($retail_buzz_avg, 5)+1e-12)",
        "Ref($retail_buzz_avg, 1)/(Mean($retail_buzz_avg, 10)+1e-12)",
        "Ref($retail_sentiment_avg, 1)-Ref($retail_sentiment_avg, 5)",
        "Mean($retail_coverage, 5)",
        "Mean($retail_alignment_score, 5)",
        "Abs(Ref($reddit_sentiment, 1)-Ref($news_sentiment, 1))",
        "Abs(Ref($x_sentiment, 1)-Ref($news_sentiment, 1))",
        "Ref($polymarket_trade_count, 1)/(Mean($polymarket_trade_count, 5)+1e-12)",
    ]
    names = [
        "RETAIL_BUZZ_L1",
        "RETAIL_SENTIMENT_L1",
        "RETAIL_COVERAGE_L1",
        "RETAIL_ALIGNMENT_L1",
        "REDDIT_BUZZ_L1",
        "X_BUZZ_L1",
        "NEWS_BUZZ_L1",
        "POLYMARKET_BUZZ_L1",
        "REDDIT_MENTIONS_L1",
        "X_MENTIONS_L1",
        "NEWS_MENTIONS_L1",
        "POLYMARKET_TRADES_L1",
        "RETAIL_BUZZ_RATIO5",
        "RETAIL_BUZZ_RATIO10",
        "RETAIL_SENTIMENT_DELTA5",
        "RETAIL_COVERAGE_MEAN5",
        "RETAIL_ALIGNMENT_MEAN5",
        "REDDIT_NEWS_DISAGREE_L1",
        "X_NEWS_DISAGREE_L1",
        "POLYMARKET_TRADES_RATIO5",
    ]
    return fields, names
