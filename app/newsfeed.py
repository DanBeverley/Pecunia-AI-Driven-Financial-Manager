#!/usr/bin/env python3
"""
Pecunia AI - Financial Newsfeed Module
AI-powered real-time financial news aggregation, summarization, and display
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import logging
from dataclasses import dataclass
import re
from urllib.parse import urlparse
import openai
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import yfinance as yf
from textblob import TextBlob
import plotly.graph_objects as go
import plotly.express as px
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NewsArticle:
    """Data class for news articles"""
    title: str
    summary: str
    content: str
    url: str
    source: str
    published_at: datetime
    category: str
    sentiment_score: float
    relevance_score: float
    symbols: List[str]
    image_url: Optional[str] = None
    read_time: Optional[int] = None

class NewsAPIManager:
    """Enhanced News API management with multiple sources"""
    
    def __init__(self):
        # Primary APIs - Production configuration
        self.api_keys = {
            'newsapi': os.getenv('NEWSAPI_KEY', 'your_newsapi_key_here'),  # Get from newsapi.org
            'alpha_vantage': os.getenv('ALPHA_VANTAGE_KEY', 'IIFJKUOT0B0M7OV6'),  # Already provided
            'polygon': os.getenv('POLYGON_API_KEY', 'your_polygon_key_here'),   # Get from polygon.io
            'finnhub': os.getenv('FINNHUB_API_KEY', 'your_finnhub_key_here')    # Get from finnhub.io
        }
        
        # Production GPU settings
        use_gpu = os.getenv('ENABLE_GPU', 'False').lower() == 'true'
        
        # Initialize AI summarizer
        try:
            self.summarizer = pipeline(
                "summarization", 
                model="facebook/bart-large-cnn",
                device=0 if use_gpu else -1
            )
            logger.info("✅ BART summarizer loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load BART summarizer: {e}")
            self.summarizer = None
        
        # Initialize sentiment analyzer
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=0 if use_gpu else -1
            )
            logger.info("✅ FinBERT sentiment analyzer loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load FinBERT sentiment analyzer: {e}")
            self.sentiment_analyzer = None
            
        # Cache for articles - Production settings
        self.article_cache = {}
        self.cache_duration = int(os.getenv('CACHE_DURATION', '300'))  # 5 minutes default
        
    def get_financial_keywords(self) -> List[str]:
        """Get relevant financial keywords for news filtering"""
        return [
            'stocks', 'market', 'trading', 'investment', 'crypto', 'bitcoin',
            'ethereum', 'finance', 'economy', 'fed', 'inflation', 'rates',
            'earnings', 'dividend', 'ipo', 'merger', 'acquisition', 'nasdaq',
            'sp500', 'dow', 'oil', 'gold', 'forex', 'bonds', 'recession',
            'gdp', 'unemployment', 'consumer', 'retail', 'tech', 'bank'
        ]
    
    def fetch_newsapi_articles(self, query: str = 'finance', limit: int = 20) -> List[Dict]:
        """Fetch articles from NewsAPI"""
        if self.api_keys['newsapi'] == 'your_newsapi_key_here':
            return self._get_mock_articles(limit)
        
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': query,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': limit,
                'apiKey': self.api_keys['newsapi']
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return data.get('articles', [])
            
        except Exception as e:
            logger.error(f"Error fetching NewsAPI articles: {e}")
            return self._get_mock_articles(limit)
    
    def fetch_alpha_vantage_news(self, symbols: List[str] = None, limit: int = 10) -> List[Dict]:
        """Fetch financial news from Alpha Vantage"""
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'apikey': self.api_keys['alpha_vantage'],
                'limit': limit
            }
            
            if symbols:
                params['tickers'] = ','.join(symbols[:5])  # Limit to 5 symbols
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            return data.get('feed', [])
            
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage news: {e}")
            return []
    
    def _get_mock_articles(self, limit: int = 20) -> List[Dict]:
        """Generate mock financial news articles for demo purposes"""
        mock_articles = [
            {
                'title': 'Federal Reserve Signals Potential Rate Changes Amid Economic Uncertainty',
                'description': 'The Federal Reserve is considering adjustments to interest rates as economic indicators show mixed signals.',
                'url': 'https://example.com/fed-rates',
                'urlToImage': 'https://via.placeholder.com/400x200?text=Fed+News',
                'publishedAt': (datetime.now() - timedelta(hours=2)).isoformat(),
                'source': {'name': 'Financial Times'},
                'content': 'Federal Reserve officials are closely monitoring economic data as they consider potential changes to interest rates. Recent inflation data and employment figures are being carefully analyzed...'
            },
            {
                'title': 'Tech Giants Report Strong Q4 Earnings Despite Market Volatility',
                'description': 'Major technology companies exceed analyst expectations in their latest quarterly reports.',
                'url': 'https://example.com/tech-earnings',
                'urlToImage': 'https://via.placeholder.com/400x200?text=Tech+Earnings',
                'publishedAt': (datetime.now() - timedelta(hours=4)).isoformat(),
                'source': {'name': 'MarketWatch'},
                'content': 'Technology sector leaders have reported earnings that surpassed Wall Street expectations, demonstrating resilience in a challenging market environment...'
            },
            {
                'title': 'Cryptocurrency Market Shows Signs of Recovery After Recent Decline',
                'description': 'Bitcoin and other major cryptocurrencies are gaining momentum as institutional interest returns.',
                'url': 'https://example.com/crypto-recovery',
                'urlToImage': 'https://via.placeholder.com/400x200?text=Crypto+News',
                'publishedAt': (datetime.now() - timedelta(hours=6)).isoformat(),
                'source': {'name': 'CoinDesk'},
                'content': 'The cryptocurrency market is experiencing renewed optimism as Bitcoin crosses key resistance levels and institutional investors show increased interest...'
            },
            {
                'title': 'ESG Investing Trends Shape Corporate Strategies for 2024',
                'description': 'Environmental, Social, and Governance factors increasingly influence investment decisions and corporate policies.',
                'url': 'https://example.com/esg-trends',
                'urlToImage': 'https://via.placeholder.com/400x200?text=ESG+Investing',
                'publishedAt': (datetime.now() - timedelta(hours=8)).isoformat(),
                'source': {'name': 'Bloomberg'},
                'content': 'Companies are adapting their strategies to meet growing ESG expectations from investors, with sustainability becoming a key performance indicator...'
            },
            {
                'title': 'Emerging Markets Present New Opportunities for Global Investors',
                'description': 'Developing economies show promise as alternative investment destinations amid global uncertainty.',
                'url': 'https://example.com/emerging-markets',
                'urlToImage': 'https://via.placeholder.com/400x200?text=Emerging+Markets',
                'publishedAt': (datetime.now() - timedelta(hours=10)).isoformat(),
                'source': {'name': 'Reuters'},
                'content': 'Investors are increasingly looking to emerging markets for diversification and growth opportunities as traditional markets face headwinds...'
            }
        ]
        
        return mock_articles[:limit]
    
    def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of financial text"""
        try:
            if self.sentiment_analyzer:
                result = self.sentiment_analyzer(text[:512])  # Limit input length
                
                # FinBERT returns positive, negative, neutral
                label = result[0]['label'].lower()
                score = result[0]['score']
                
                if label == 'positive':
                    return score
                elif label == 'negative':
                    return -score
                else:
                    return 0.0
            else:
                # Fallback to TextBlob
                blob = TextBlob(text)
                return blob.sentiment.polarity
                
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return 0.0
    
    def summarize_article(self, content: str, max_length: int = 100) -> str:
        """Generate AI summary of article content"""
        try:
            if self.summarizer and len(content) > 200:
                # Clean and prepare text
                clean_content = re.sub(r'\s+', ' ', content.strip())
                
                # Summarize using BART
                summary = self.summarizer(
                    clean_content[:1024],  # Limit input
                    max_length=max_length,
                    min_length=30,
                    do_sample=False
                )
                
                return summary[0]['summary_text']
            else:
                # Fallback: Extract first few sentences
                sentences = content.split('. ')
                summary = '. '.join(sentences[:3]) + '.'
                return summary[:max_length] + '...' if len(summary) > max_length else summary
                
        except Exception as e:
            logger.error(f"Error in summarization: {e}")
            # Extract first few sentences as fallback
            sentences = content.split('. ')
            summary = '. '.join(sentences[:2]) + '.'
            return summary[:max_length] + '...' if len(summary) > max_length else summary
    
    def extract_stock_symbols(self, text: str) -> List[str]:
        """Extract stock symbols from text"""
        # Common patterns for stock symbols
        patterns = [
            r'\b([A-Z]{1,5})\b(?=\s|$)',  # 1-5 uppercase letters
            r'\$([A-Z]{1,5})\b',          # $ prefix
            r'\b([A-Z]{1,5})\.([A-Z]{1,3})\b'  # Exchange notation
        ]
        
        symbols = set()
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if isinstance(matches[0], tuple) if matches else False:
                symbols.update([match[0] for match in matches])
            else:
                symbols.update(matches)
        
        # Filter out common false positives
        false_positives = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HAD', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW', 'ITS', 'MAY', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WHO', 'BOY', 'DID', 'ITS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE'}
        
        return [symbol for symbol in symbols if symbol not in false_positives and len(symbol) <= 5]
    
    def calculate_relevance_score(self, article: Dict, user_interests: List[str] = None) -> float:
        """Calculate relevance score based on content and user interests"""
        score = 0.0
        
        # Base financial relevance
        financial_keywords = self.get_financial_keywords()
        title_text = (article.get('title', '') + ' ' + article.get('description', '')).lower()
        
        keyword_matches = sum(1 for keyword in financial_keywords if keyword in title_text)
        score += (keyword_matches / len(financial_keywords)) * 0.6
        
        # User interest matching
        if user_interests:
            interest_matches = sum(1 for interest in user_interests if interest.lower() in title_text)
            score += (interest_matches / len(user_interests)) * 0.4
        
        # Recency bonus
        try:
            pub_date = datetime.fromisoformat(article.get('publishedAt', '').replace('Z', '+00:00'))
            hours_old = (datetime.now().replace(tzinfo=pub_date.tzinfo) - pub_date).total_seconds() / 3600
            recency_score = max(0, 1 - (hours_old / 24))  # Decay over 24 hours
            score += recency_score * 0.2
        except:
            pass
        
        return min(score, 1.0)
    
    def process_articles(self, raw_articles: List[Dict], user_interests: List[str] = None) -> List[NewsArticle]:
        """Process raw articles into structured NewsArticle objects"""
        processed_articles = []
        
        for article in raw_articles:
            try:
                # Extract content
                title = article.get('title', 'Untitled')
                content = article.get('content', article.get('description', ''))
                
                if not content or len(content.strip()) < 50:
                    continue
                
                # Generate summary
                summary = self.summarize_article(content)
                
                # Analyze sentiment
                sentiment_score = self.analyze_sentiment(title + ' ' + content)
                
                # Calculate relevance
                relevance_score = self.calculate_relevance_score(article, user_interests)
                
                # Extract symbols
                symbols = self.extract_stock_symbols(title + ' ' + content)
                
                # Parse date
                try:
                    published_at = datetime.fromisoformat(
                        article.get('publishedAt', '').replace('Z', '+00:00')
                    )
                except:
                    published_at = datetime.now()
                
                # Estimate read time
                word_count = len(content.split())
                read_time = max(1, word_count // 200)  # ~200 WPM
                
                # Determine category
                category = self._categorize_article(title + ' ' + content)
                
                news_article = NewsArticle(
                    title=title,
                    summary=summary,
                    content=content,
                    url=article.get('url', ''),
                    source=article.get('source', {}).get('name', 'Unknown'),
                    published_at=published_at,
                    category=category,
                    sentiment_score=sentiment_score,
                    relevance_score=relevance_score,
                    symbols=symbols,
                    image_url=article.get('urlToImage'),
                    read_time=read_time
                )
                
                processed_articles.append(news_article)
                
            except Exception as e:
                logger.error(f"Error processing article: {e}")
                continue
        
        # Sort by relevance and recency
        processed_articles.sort(
            key=lambda x: (x.relevance_score * 0.7 + (1 - min(1, (datetime.now() - x.published_at).total_seconds() / 86400)) * 0.3),
            reverse=True
        )
        
        return processed_articles
    
    def _categorize_article(self, text: str) -> str:
        """Categorize article based on content"""
        text_lower = text.lower()
        
        categories = {
            'cryptocurrency': ['crypto', 'bitcoin', 'ethereum', 'blockchain', 'defi', 'nft'],
            'stocks': ['stock', 'equity', 'share', 'nasdaq', 'nyse', 'dow'],
            'economy': ['economy', 'gdp', 'inflation', 'recession', 'unemployment'],
            'central_banks': ['fed', 'federal reserve', 'ecb', 'interest rate', 'monetary policy'],
            'commodities': ['oil', 'gold', 'silver', 'commodity', 'crude'],
            'technology': ['tech', 'ai', 'artificial intelligence', 'software', 'hardware'],
            'earnings': ['earnings', 'revenue', 'profit', 'quarterly', 'q1', 'q2', 'q3', 'q4'],
            'mergers': ['merger', 'acquisition', 'deal', 'buyout', 'takeover']
        }
        
        for category, keywords in categories.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        
        return 'general'

class NewsDisplay:
    """Enhanced news display with modern UI components"""
    
    def __init__(self):
        self.news_manager = NewsAPIManager()
        
    def render_newsfeed_header(self):
        """Render the newsfeed header with controls"""
        st.markdown("""
        <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem;'>
            <h1 style='color: white; text-align: center; margin: 0;'>
                📰 AI-Powered Financial Newsfeed
            </h1>
            <p style='color: white; text-align: center; margin: 0.5rem 0 0 0; opacity: 0.9;'>
                Real-time financial news with AI summarization and sentiment analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Control panel
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        
        with col1:
            news_sources = st.multiselect(
                "📡 News Sources",
                ["NewsAPI", "Alpha Vantage", "Financial Times", "Bloomberg", "Reuters"],
                default=["NewsAPI", "Alpha Vantage"]
            )
        
        with col2:
            categories = st.multiselect(
                "🏷️ Categories",
                ["All", "Stocks", "Crypto", "Economy", "Technology", "Earnings"],
                default=["All"]
            )
        
        with col3:
            user_symbols = st.text_input(
                "📈 Track Symbols (comma-separated)",
                placeholder="AAPL, TSLA, BTC-USD"
            )
        
        with col4:
            auto_refresh = st.checkbox("🔄 Auto Refresh", value=True)
            
        return {
            'sources': news_sources,
            'categories': categories,
            'symbols': [s.strip().upper() for s in user_symbols.split(',') if s.strip()],
            'auto_refresh': auto_refresh
        }
    
    def render_article_card(self, article: NewsArticle, show_full: bool = False):
        """Render an individual article card"""
        # Sentiment color
        sentiment_color = "#28a745" if article.sentiment_score > 0.1 else "#dc3545" if article.sentiment_score < -0.1 else "#6c757d"
        sentiment_text = "Positive" if article.sentiment_score > 0.1 else "Negative" if article.sentiment_score < -0.1 else "Neutral"
        
        # Time ago
        time_diff = datetime.now() - article.published_at
        if time_diff.days > 0:
            time_ago = f"{time_diff.days}d ago"
        elif time_diff.seconds > 3600:
            time_ago = f"{time_diff.seconds // 3600}h ago"
        else:
            time_ago = f"{time_diff.seconds // 60}m ago"
        
        # Article card HTML
        card_html = f"""
        <div style='border: 1px solid #e0e0e0; border-radius: 10px; padding: 1.5rem; margin-bottom: 1.5rem; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <div style='display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem;'>
                <div style='flex-grow: 1;'>
                    <h3 style='margin: 0 0 0.5rem 0; color: #333; line-height: 1.3;'>{article.title}</h3>
                    <div style='display: flex; align-items: center; gap: 1rem; font-size: 0.85rem; color: #666;'>
                        <span>📰 {article.source}</span>
                        <span>🕐 {time_ago}</span>
                        <span>📖 {article.read_time} min read</span>
                        <span style='color: {sentiment_color}; font-weight: bold;'>😊 {sentiment_text}</span>
                        <span>🎯 {article.relevance_score:.1%} relevant</span>
                    </div>
                </div>
            </div>
            
            <div style='margin-bottom: 1rem;'>
                <p style='margin: 0; color: #555; line-height: 1.5;'>{article.summary}</p>
            </div>
            
            {f"<div style='margin-bottom: 1rem;'><strong>📈 Symbols:</strong> {', '.join(article.symbols)}</div>" if article.symbols else ""}
            
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div style='display: flex; gap: 0.5rem;'>
                    <span style='background: #e7f3ff; color: #0066cc; padding: 0.25rem 0.5rem; border-radius: 15px; font-size: 0.8rem;'>
                        {article.category.replace('_', ' ').title()}
                    </span>
                </div>
                <div>
                    <a href='{article.url}' target='_blank' style='color: #0066cc; text-decoration: none;'>
                        Read Full Article →
                    </a>
                </div>
            </div>
        </div>
        """
        
        st.markdown(card_html, unsafe_allow_html=True)
        
        # Show full content if requested
        if show_full:
            with st.expander("📖 Full Content", expanded=False):
                st.markdown(f"**Source:** {article.source}")
                st.markdown(f"**Published:** {article.published_at.strftime('%Y-%m-%d %H:%M')}")
                st.markdown("---")
                st.write(article.content)
    
    def render_market_overview(self, symbols: List[str]):
        """Render market overview for tracked symbols"""
        if not symbols:
            return
        
        st.markdown("### 📊 Market Overview")
        
        try:
            # Fetch market data
            market_data = []
            for symbol in symbols[:5]:  # Limit to 5 symbols
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    hist = ticker.history(period="1d")
                    
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        prev_close = info.get('previousClose', current_price)
                        change = current_price - prev_close
                        change_pct = (change / prev_close) * 100 if prev_close else 0
                        
                        market_data.append({
                            'Symbol': symbol,
                            'Price': f"${current_price:.2f}",
                            'Change': f"${change:+.2f}",
                            'Change %': f"{change_pct:+.2f}%",
                            'Volume': f"{info.get('volume', 0):,}"
                        })
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {e}")
            
            if market_data:
                df = pd.DataFrame(market_data)
                st.dataframe(df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading market data: {e}")
    
    def render_sentiment_analysis(self, articles: List[NewsArticle]):
        """Render sentiment analysis visualization"""
        if not articles:
            return
        
        st.markdown("### 📊 Market Sentiment Analysis")
        
        # Calculate sentiment distribution
        positive = sum(1 for a in articles if a.sentiment_score > 0.1)
        negative = sum(1 for a in articles if a.sentiment_score < -0.1)
        neutral = len(articles) - positive - negative
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=['Positive', 'Neutral', 'Negative'],
            values=[positive, neutral, negative],
            hole=.3,
            marker_colors=['#28a745', '#6c757d', '#dc3545']
        )])
        
        fig.update_layout(title="News Sentiment Distribution", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment over time
        if len(articles) > 1:
            sentiment_df = pd.DataFrame([
                {'time': a.published_at, 'sentiment': a.sentiment_score, 'title': a.title}
                for a in articles
            ]).sort_values('time')
            
            fig_line = px.line(
                sentiment_df, x='time', y='sentiment',
                title='Sentiment Over Time',
                hover_data=['title']
            )
            fig_line.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_line, use_container_width=True)

def fetch_and_update_news(config: Dict = None) -> List[NewsArticle]:
    """
    Periodically fetch news and update the feed
    Main function called by the app
    """
    news_manager = NewsAPIManager()
    
    # Configuration from UI
    symbols = config.get('symbols', []) if config else []
    sources = config.get('sources', ['NewsAPI']) if config else ['NewsAPI']
    
    all_articles = []
    
    # Fetch from different sources
    try:
        if 'NewsAPI' in sources:
            raw_articles = news_manager.fetch_newsapi_articles('finance OR stocks OR crypto', limit=15)
            processed = news_manager.process_articles(raw_articles, user_interests=['finance', 'stocks'])
            all_articles.extend(processed)
        
        if 'Alpha Vantage' in sources and symbols:
            raw_av_articles = news_manager.fetch_alpha_vantage_news(symbols[:3], limit=10)
            # Convert Alpha Vantage format to standard format
            av_articles = []
            for av_article in raw_av_articles:
                av_articles.append({
                    'title': av_article.get('title', 'Untitled'),
                    'description': av_article.get('summary', ''),
                    'content': av_article.get('summary', ''),
                    'url': av_article.get('url', ''),
                    'source': {'name': 'Alpha Vantage'},
                    'publishedAt': av_article.get('time_published', datetime.now().isoformat()),
                    'urlToImage': None
                })
            
            processed_av = news_manager.process_articles(av_articles, user_interests=symbols)
            all_articles.extend(processed_av)
            
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        # Return mock articles on error
        mock_articles = news_manager._get_mock_articles(10)
        all_articles = news_manager.process_articles(mock_articles)
    
    # Remove duplicates based on title similarity
    unique_articles = []
    seen_titles = set()
    
    for article in all_articles:
        title_key = re.sub(r'\W+', '', article.title.lower())[:50]
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            unique_articles.append(article)
    
    return unique_articles[:20]  # Return top 20 articles

def display_newsfeed() -> None:
    """
    Main display function for the newsfeed tab
    Shows a tab with summarized headlines
    """
    st.set_page_config(
        page_title="Pecunia AI - Financial Newsfeed",
        page_icon="📰",
        layout="wide"
    )
    
    display = NewsDisplay()
    
    # Render header and get configuration
    config = display.render_newsfeed_header()
    
    # Initialize session state
    if 'articles' not in st.session_state:
        st.session_state.articles = []
    if 'last_update' not in st.session_state:
        st.session_state.last_update = datetime.min
    
    # Auto-refresh logic
    should_refresh = (
        config['auto_refresh'] and 
        (datetime.now() - st.session_state.last_update).total_seconds() > 300  # 5 minutes
    ) or len(st.session_state.articles) == 0
    
    # Manual refresh button
    col_refresh, col_info = st.columns([1, 4])
    with col_refresh:
        if st.button("🔄 Refresh News", type="primary"):
            should_refresh = True
    
    with col_info:
        if st.session_state.last_update != datetime.min:
            st.info(f"Last updated: {st.session_state.last_update.strftime('%H:%M:%S')}")
    
    # Fetch news if needed
    if should_refresh:
        with st.spinner("🔄 Fetching latest financial news..."):
            try:
                st.session_state.articles = fetch_and_update_news(config)
                st.session_state.last_update = datetime.now()
                st.success(f"✅ Loaded {len(st.session_state.articles)} articles")
            except Exception as e:
                st.error(f"❌ Error loading news: {e}")
                st.session_state.articles = []
    
    # Display content
    if st.session_state.articles:
        # Market overview for tracked symbols
        if config['symbols']:
            display.render_market_overview(config['symbols'])
            st.markdown("---")
        
        # Sentiment analysis
        display.render_sentiment_analysis(st.session_state.articles)
        st.markdown("---")
        
        # Filter articles by category
        filtered_articles = st.session_state.articles
        if 'All' not in config['categories']:
            category_map = {
                'Stocks': ['stocks', 'equity'],
                'Crypto': ['cryptocurrency'],
                'Economy': ['economy', 'central_banks'],
                'Technology': ['technology'],
                'Earnings': ['earnings']
            }
            
            selected_cats = []
            for cat in config['categories']:
                selected_cats.extend(category_map.get(cat, [cat.lower()]))
            
            filtered_articles = [
                a for a in st.session_state.articles 
                if a.category in selected_cats
            ]
        
        # Display articles
        st.markdown(f"### 📰 Latest Financial News ({len(filtered_articles)} articles)")
        
        if filtered_articles:
            for i, article in enumerate(filtered_articles):
                display.render_article_card(article)
                
                # Add separator every 5 articles
                if (i + 1) % 5 == 0 and i < len(filtered_articles) - 1:
                    st.markdown("---")
        else:
            st.info("No articles match your selected categories. Try selecting 'All' or different categories.")
    
    else:
        st.info("No news articles available. Click 'Refresh News' to load the latest financial news.")

def summarize_and_display(article: str) -> None:
    """
    Offers users a choice between summarized key points or the full article
    """
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📝 Show Summary", type="primary"):
            st.session_state.show_summary = True
            st.session_state.show_full = False
    
    with col2:
        if st.button("📖 Show Full Article"):
            st.session_state.show_full = True  
            st.session_state.show_summary = False
    
    # Display based on selection
    if st.session_state.get('show_summary', False):
        news_manager = NewsAPIManager()
        summary = news_manager.summarize_article(article)
        
        st.markdown("### 📝 AI Summary")
        st.markdown(f"> {summary}")
        
        # Key points extraction
        sentences = article.split('.')
        if len(sentences) > 3:
            st.markdown("### 🎯 Key Points")
            for i, sentence in enumerate(sentences[:5], 1):
                if sentence.strip():
                    st.markdown(f"{i}. {sentence.strip()}")
    
    elif st.session_state.get('show_full', False):
        st.markdown("### 📖 Full Article")
        st.markdown(article)

# Main execution
if __name__ == "__main__":
    display_newsfeed() 