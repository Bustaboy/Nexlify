#!/usr/bin/env python3
"""
Nexlify Sentiment Analysis System

Multi-source sentiment analysis for cryptocurrency trading:
- Crypto Fear & Greed Index
- CryptoPanic news sentiment
- Social media sentiment (Twitter/Reddit)
- Whale alerts
- On-chain metrics

All sources are cached and rate-limited for efficiency
"""

import logging
import time
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SentimentScore:
    """Sentiment score from a source"""

    source: str
    value: float  # -1 to 1 (negative to positive)
    confidence: float  # 0 to 1
    timestamp: float
    raw_data: Optional[Dict] = None


@dataclass
class AggregateSentiment:
    """Aggregate sentiment from all sources"""

    overall_score: float  # -1 to 1
    fear_greed_index: Optional[float]  # 0-100
    news_sentiment: Optional[float]  # -1 to 1
    social_sentiment: Optional[float]  # -1 to 1
    whale_activity: Optional[float]  # -1 to 1 (negative = selling, positive = buying)
    on_chain_sentiment: Optional[float]  # -1 to 1
    confidence: float  # 0 to 1
    timestamp: float


class SentimentAnalyzer:
    """
    Multi-source sentiment analysis with caching and rate limiting

    Free APIs:
    - Fear & Greed Index (no key needed)
    - CryptoPanic (free tier: 3000 req/month)

    Optional (require API keys):
    - Twitter API
    - Reddit API
    - Whale Alert
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize sentiment analyzer

        Args:
            config: Configuration dict with API keys
                {
                    'cryptopanic_key': 'xxx',  # Optional but recommended
                    'twitter_bearer': 'xxx',   # Optional
                    'reddit_client_id': 'xxx', # Optional
                    'reddit_secret': 'xxx',    # Optional
                    'whale_alert_key': 'xxx'   # Optional
                }
        """
        self.config = config or {}

        # API endpoints
        self.fear_greed_url = "https://api.alternative.me/fng/"
        self.cryptopanic_url = "https://cryptopanic.com/api/v1/posts/"

        # Cache (5 minute TTL)
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes

        # Rate limiting
        self.last_request_time = {}
        self.min_request_interval = {
            "fear_greed": 60,  # 1 minute
            "cryptopanic": 20,  # 3 per minute (free tier)
            "twitter": 1,  # 1 per second
            "reddit": 2,  # 30 per minute
            "whale_alert": 5,  # 12 per minute
        }

        # History for trend analysis
        self.history = deque(maxlen=100)

        logger.info("📊 Sentiment Analyzer initialized")
        logger.info(f"   Fear & Greed Index: ✓ Enabled (no key needed)")

        if self.config.get("cryptopanic_key"):
            logger.info(f"   CryptoPanic: ✓ Enabled")
        else:
            logger.info(f"   CryptoPanic: ⚠️  No API key (limited access)")

        if self.config.get("twitter_bearer"):
            logger.info(f"   Twitter: ✓ Enabled")
        else:
            logger.info(f"   Twitter: ✗ Disabled (no API key)")

        if self.config.get("reddit_client_id"):
            logger.info(f"   Reddit: ✓ Enabled")
        else:
            logger.info(f"   Reddit: ✗ Disabled (no API key)")

        if self.config.get("whale_alert_key"):
            logger.info(f"   Whale Alert: ✓ Enabled")
        else:
            logger.info(f"   Whale Alert: ✗ Disabled (no API key)")

    async def get_sentiment(self, symbol: str = "BTC") -> AggregateSentiment:
        """
        Get aggregate sentiment from all sources

        Args:
            symbol: Cryptocurrency symbol (e.g., "BTC", "ETH")

        Returns:
            AggregateSentiment with scores from all sources
        """
        # Check cache
        cache_key = f"sentiment_{symbol}"
        if cache_key in self.cache:
            cached_data, cache_time = self.cache[cache_key]
            if time.time() - cache_time < self.cache_ttl:
                logger.debug(f"Using cached sentiment for {symbol}")
                return cached_data

        logger.info(f"📊 Fetching sentiment for {symbol}...")

        # Gather sentiment from all sources in parallel
        tasks = [
            self._get_fear_greed_index(),
            self._get_cryptopanic_sentiment(symbol),
            self._get_social_sentiment(symbol),
            self._get_whale_activity(symbol),
            self._get_onchain_sentiment(symbol),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Unpack results
        fear_greed = results[0] if not isinstance(results[0], Exception) else None
        news_sentiment = results[1] if not isinstance(results[1], Exception) else None
        social_sentiment = results[2] if not isinstance(results[2], Exception) else None
        whale_activity = results[3] if not isinstance(results[3], Exception) else None
        onchain_sentiment = (
            results[4] if not isinstance(results[4], Exception) else None
        )

        # Calculate aggregate sentiment
        aggregate = self._calculate_aggregate(
            fear_greed,
            news_sentiment,
            social_sentiment,
            whale_activity,
            onchain_sentiment,
        )

        # Cache result
        self.cache[cache_key] = (aggregate, time.time())

        # Add to history
        self.history.append(aggregate)

        logger.info(
            f"✅ Sentiment for {symbol}: {aggregate.overall_score:.2f} "
            f"(confidence: {aggregate.confidence:.0%})"
        )

        return aggregate

    async def _get_fear_greed_index(self) -> Optional[float]:
        """
        Get Crypto Fear & Greed Index (0-100)

        Source: https://alternative.me/crypto/fear-and-greed-index/
        Free, no API key needed
        """
        # Rate limiting
        if not self._can_make_request("fear_greed"):
            logger.debug("Fear & Greed: rate limited, using cached value")
            return None

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.fear_greed_url, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        value = int(data["data"][0]["value"])

                        self.last_request_time["fear_greed"] = time.time()

                        logger.debug(f"Fear & Greed Index: {value}/100")
                        return value
                    else:
                        logger.warning(f"Fear & Greed API error: {response.status}")
                        return None

        except Exception as e:
            logger.error(f"Fear & Greed fetch failed: {e}")
            return None

    async def _get_cryptopanic_sentiment(self, symbol: str) -> Optional[float]:
        """
        Get news sentiment from CryptoPanic

        Free tier: 3000 requests/month
        Requires API key for full access
        """
        # Rate limiting
        if not self._can_make_request("cryptopanic"):
            logger.debug("CryptoPanic: rate limited")
            return None

        try:
            # Convert symbol (BTC -> bitcoin)
            currencies = {
                "BTC": "bitcoin",
                "ETH": "ethereum",
                "BNB": "binance-coin",
                "SOL": "solana",
                "ADA": "cardano",
                "XRP": "ripple",
            }
            currency = currencies.get(symbol.upper(), symbol.lower())

            params = {"currencies": currency, "filter": "hot", "kind": "news"}

            # Add API key if available
            if self.config.get("cryptopanic_key"):
                params["auth_token"] = self.config["cryptopanic_key"]

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.cryptopanic_url, params=params, timeout=5
                ) as response:
                    if response.status == 200:
                        data = await response.json()

                        self.last_request_time["cryptopanic"] = time.time()

                        # Analyze sentiment from news
                        sentiment = self._analyze_cryptopanic_news(data)
                        logger.debug(f"CryptoPanic sentiment: {sentiment:.2f}")
                        return sentiment
                    else:
                        logger.warning(f"CryptoPanic API error: {response.status}")
                        return None

        except Exception as e:
            logger.error(f"CryptoPanic fetch failed: {e}")
            return None

    def _analyze_cryptopanic_news(self, data: Dict) -> float:
        """Analyze sentiment from CryptoPanic news data"""
        if "results" not in data or not data["results"]:
            return 0.0

        sentiments = []
        for post in data["results"][:20]:  # Analyze last 20 posts
            # CryptoPanic provides votes
            votes = post.get("votes", {})
            positive = votes.get("positive", 0)
            negative = votes.get("negative", 0)
            important = votes.get("important", 0)
            liked = votes.get("liked", 0)
            disliked = votes.get("disliked", 0)

            # Calculate sentiment score
            total_votes = positive + negative + important + liked + disliked
            if total_votes > 0:
                sentiment = (
                    positive + important + liked - negative - disliked
                ) / total_votes
                sentiments.append(sentiment)

            # Also check title for keywords
            title = post.get("title", "").lower()
            if any(
                word in title for word in ["bullish", "surge", "rally", "pump", "moon"]
            ):
                sentiments.append(0.5)
            elif any(
                word in title for word in ["bearish", "crash", "dump", "plunge", "fear"]
            ):
                sentiments.append(-0.5)

        if sentiments:
            return sum(sentiments) / len(sentiments)
        return 0.0

    async def _get_social_sentiment(self, symbol: str) -> Optional[float]:
        """
        Get social media sentiment (Twitter + Reddit)

        Requires API keys
        """
        sentiments = []

        # Twitter sentiment
        if self.config.get("twitter_bearer"):
            twitter_sentiment = await self._get_twitter_sentiment(symbol)
            if twitter_sentiment is not None:
                sentiments.append(twitter_sentiment)

        # Reddit sentiment
        if self.config.get("reddit_client_id"):
            reddit_sentiment = await self._get_reddit_sentiment(symbol)
            if reddit_sentiment is not None:
                sentiments.append(reddit_sentiment)

        if sentiments:
            return sum(sentiments) / len(sentiments)
        return None

    async def _get_twitter_sentiment(self, symbol: str) -> Optional[float]:
        """Get Twitter sentiment (requires API key)"""
        # Rate limiting
        if not self._can_make_request("twitter"):
            return None

        try:
            # Twitter API v2 recent search
            url = "https://api.twitter.com/2/tweets/search/recent"
            headers = {"Authorization": f"Bearer {self.config['twitter_bearer']}"}
            params = {
                "query": f"${symbol} OR #{symbol} lang:en -is:retweet",
                "max_results": 100,
                "tweet.fields": "public_metrics",
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, headers=headers, params=params, timeout=5
                ) as response:
                    if response.status == 200:
                        data = await response.json()

                        self.last_request_time["twitter"] = time.time()

                        # Simple sentiment: likes/retweets ratio
                        sentiment = self._analyze_twitter_data(data)
                        logger.debug(f"Twitter sentiment: {sentiment:.2f}")
                        return sentiment

        except Exception as e:
            logger.error(f"Twitter fetch failed: {e}")

        return None

    def _analyze_twitter_data(self, data: Dict) -> float:
        """Analyze sentiment from Twitter data"""
        if "data" not in data:
            return 0.0

        sentiments = []
        for tweet in data["data"]:
            text = tweet.get("text", "").lower()
            metrics = tweet.get("public_metrics", {})

            # Keyword-based sentiment
            positive_keywords = ["bullish", "moon", "pump", "buy", "hodl", "long"]
            negative_keywords = ["bearish", "dump", "sell", "short", "crash", "rip"]

            sentiment = 0.0
            for word in positive_keywords:
                if word in text:
                    sentiment += 0.2
            for word in negative_keywords:
                if word in text:
                    sentiment -= 0.2

            # Weight by engagement
            likes = metrics.get("like_count", 0)
            retweets = metrics.get("retweet_count", 0)
            weight = 1 + (likes + retweets * 2) / 100

            sentiments.append(sentiment * weight)

        if sentiments:
            return max(-1.0, min(1.0, sum(sentiments) / len(sentiments)))
        return 0.0

    async def _get_reddit_sentiment(self, symbol: str) -> Optional[float]:
        """Get Reddit sentiment (requires API key)"""
        # Rate limiting
        if not self._can_make_request("reddit"):
            return None

        try:
            # Reddit API (OAuth required for full access)
            # For simplicity, using public JSON endpoints
            subreddits = ["CryptoCurrency", "Bitcoin", "ethtrader", "CryptoMarkets"]

            sentiments = []

            for subreddit in subreddits:
                url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=25"

                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        url, headers={"User-Agent": "Nexlify/1.0"}, timeout=5
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            sentiment = self._analyze_reddit_data(data, symbol)
                            if sentiment != 0.0:
                                sentiments.append(sentiment)

            self.last_request_time["reddit"] = time.time()

            if sentiments:
                return sum(sentiments) / len(sentiments)

        except Exception as e:
            logger.error(f"Reddit fetch failed: {e}")

        return None

    def _analyze_reddit_data(self, data: Dict, symbol: str) -> float:
        """Analyze sentiment from Reddit data"""
        if "data" not in data or "children" not in data["data"]:
            return 0.0

        sentiments = []
        for post in data["data"]["children"]:
            post_data = post["data"]
            title = post_data.get("title", "").lower()

            # Check if post is about the symbol
            if symbol.lower() not in title and f"${symbol.lower()}" not in title:
                continue

            # Sentiment from upvote ratio
            upvote_ratio = post_data.get("upvote_ratio", 0.5)
            sentiment = (upvote_ratio - 0.5) * 2  # Convert 0-1 to -1 to 1

            # Keyword boost
            positive_keywords = ["bullish", "moon", "pump", "buy", "long"]
            negative_keywords = ["bearish", "dump", "sell", "short", "crash"]

            for word in positive_keywords:
                if word in title:
                    sentiment += 0.2
            for word in negative_keywords:
                if word in title:
                    sentiment -= 0.2

            sentiments.append(max(-1.0, min(1.0, sentiment)))

        if sentiments:
            return sum(sentiments) / len(sentiments)
        return 0.0

    async def _get_whale_activity(self, symbol: str) -> Optional[float]:
        """
        Get whale activity sentiment

        Large transactions can indicate institutional movement
        Requires Whale Alert API key
        """
        if not self.config.get("whale_alert_key"):
            return None

        # Rate limiting
        if not self._can_make_request("whale_alert"):
            return None

        try:
            url = "https://api.whale-alert.io/v1/transactions"
            params = {
                "api_key": self.config["whale_alert_key"],
                "currency": symbol.lower(),
                "min_value": 1000000,  # $1M+
                "start": int(time.time()) - 3600,  # Last hour
                "limit": 100,
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()

                        self.last_request_time["whale_alert"] = time.time()

                        sentiment = self._analyze_whale_data(data)
                        logger.debug(f"Whale activity sentiment: {sentiment:.2f}")
                        return sentiment

        except Exception as e:
            logger.error(f"Whale Alert fetch failed: {e}")

        return None

    def _analyze_whale_data(self, data: Dict) -> float:
        """Analyze sentiment from whale transactions"""
        if "transactions" not in data or not data["transactions"]:
            return 0.0

        buy_volume = 0
        sell_volume = 0

        for tx in data["transactions"]:
            amount_usd = tx.get("amount_usd", 0)
            to_type = tx.get("to", {}).get("owner_type", "")
            from_type = tx.get("from", {}).get("owner_type", "")

            # Exchange inflow = selling pressure
            if to_type == "exchange":
                sell_volume += amount_usd

            # Exchange outflow = buying pressure
            if from_type == "exchange":
                buy_volume += amount_usd

        total_volume = buy_volume + sell_volume
        if total_volume > 0:
            # -1 (all selling) to 1 (all buying)
            sentiment = (buy_volume - sell_volume) / total_volume
            return sentiment

        return 0.0

    async def _get_onchain_sentiment(self, symbol: str) -> Optional[float]:
        """
        Get on-chain metrics sentiment

        For now, returns None - can be extended with:
        - Glassnode API
        - IntoTheBlock API
        - CoinMetrics API
        """
        # TODO: Implement on-chain metrics
        # - Active addresses
        # - Exchange netflow
        # - SOPR (Spent Output Profit Ratio)
        # - NVT ratio
        return None

    def _calculate_aggregate(
        self,
        fear_greed: Optional[float],
        news_sentiment: Optional[float],
        social_sentiment: Optional[float],
        whale_activity: Optional[float],
        onchain_sentiment: Optional[float],
    ) -> AggregateSentiment:
        """Calculate weighted aggregate sentiment"""

        scores = []
        weights = []

        # Fear & Greed (normalize 0-100 to -1 to 1)
        if fear_greed is not None:
            normalized = (fear_greed - 50) / 50  # 0 -> -1, 100 -> 1
            scores.append(normalized)
            weights.append(0.30)  # 30% weight

        # News sentiment
        if news_sentiment is not None:
            scores.append(news_sentiment)
            weights.append(0.25)  # 25% weight

        # Social sentiment
        if social_sentiment is not None:
            scores.append(social_sentiment)
            weights.append(0.20)  # 20% weight

        # Whale activity
        if whale_activity is not None:
            scores.append(whale_activity)
            weights.append(0.15)  # 15% weight

        # On-chain sentiment
        if onchain_sentiment is not None:
            scores.append(onchain_sentiment)
            weights.append(0.10)  # 10% weight

        # Calculate weighted average
        if scores:
            # Normalize weights
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]

            overall_score = sum(s * w for s, w in zip(scores, weights))
            confidence = len(scores) / 5.0  # Max 5 sources
        else:
            overall_score = 0.0
            confidence = 0.0

        return AggregateSentiment(
            overall_score=overall_score,
            fear_greed_index=fear_greed,
            news_sentiment=news_sentiment,
            social_sentiment=social_sentiment,
            whale_activity=whale_activity,
            on_chain_sentiment=onchain_sentiment,
            confidence=confidence,
            timestamp=time.time(),
        )

    def _can_make_request(self, source: str) -> bool:
        """Check if we can make a request (rate limiting)"""
        if source not in self.last_request_time:
            return True

        elapsed = time.time() - self.last_request_time[source]
        min_interval = self.min_request_interval.get(source, 1)

        return elapsed >= min_interval

    def get_sentiment_features(self, sentiment: AggregateSentiment) -> Dict[str, float]:
        """
        Convert sentiment to features for ML models

        Returns:
            Dict of sentiment features
        """
        features = {
            "sentiment_overall": sentiment.overall_score,
            "sentiment_confidence": sentiment.confidence,
        }

        if sentiment.fear_greed_index is not None:
            features["fear_greed_index"] = sentiment.fear_greed_index / 100
            features["fear_greed_extreme_fear"] = (
                1.0 if sentiment.fear_greed_index < 20 else 0.0
            )
            features["fear_greed_extreme_greed"] = (
                1.0 if sentiment.fear_greed_index > 80 else 0.0
            )

        if sentiment.news_sentiment is not None:
            features["news_sentiment"] = sentiment.news_sentiment

        if sentiment.social_sentiment is not None:
            features["social_sentiment"] = sentiment.social_sentiment

        if sentiment.whale_activity is not None:
            features["whale_activity"] = sentiment.whale_activity

        # Trend analysis if we have history
        if len(self.history) >= 2:
            recent_scores = [s.overall_score for s in list(self.history)[-10:]]
            features["sentiment_trend"] = recent_scores[-1] - recent_scores[0]
            features["sentiment_volatility"] = sum(
                abs(recent_scores[i] - recent_scores[i - 1])
                for i in range(1, len(recent_scores))
            ) / len(recent_scores)

        return features


# Convenience functions
async def get_sentiment(
    symbol: str = "BTC", config: Optional[Dict] = None
) -> AggregateSentiment:
    """Quick sentiment fetch"""
    analyzer = SentimentAnalyzer(config)
    return await analyzer.get_sentiment(symbol)


# Export
__all__ = ["SentimentScore", "AggregateSentiment", "SentimentAnalyzer", "get_sentiment"]
