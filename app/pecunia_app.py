#!/usr/bin/env python3
"""
Pecunia AI - Main Streamlit Application
Unified interface integrating newsfeed, education, and community modules
"""

import streamlit as st
import sys
from pathlib import Path

# Add the app directory to Python path for imports
app_dir = Path(__file__).parent
sys.path.insert(0, str(app_dir))

# Import our modules
from newsfeed import display_newsfeed, fetch_and_update_news
from education import display_tutorial
from community import display_forum

def main():
    """Main Pecunia AI application"""
    
    # Configure the page
    st.set_page_config(
        page_title="Pecunia AI - Personal Financial Manager",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .nav-button {
        width: 100%;
        margin-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 1rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem;'>
            <h2 style='color: white; margin: 0;'>💰 Pecunia AI</h2>
            <p style='color: white; margin: 0; opacity: 0.9; font-size: 0.9rem;'>Your AI Financial Assistant</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation menu
        page = st.selectbox(
            "🧭 Navigate to:",
            ["🏠 Dashboard", "📰 Financial Newsfeed", "🎓 Education Center", "💬 Community Forum"],
            index=0
        )
        
        st.markdown("---")
        
        # Quick stats
        st.markdown("### 📊 Quick Stats")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Portfolio", "$125.4K", "+2.3%")
        with col2:
            st.metric("Monthly Savings", "$2,850", "+12%")
        
        st.metric("Investment Return", "8.7%", "+1.2%")
        st.metric("Risk Score", "Moderate", "")
        
        st.markdown("---")
        
        # User profile section
        st.markdown("### 👤 Profile")
        st.markdown("**NewInvestor2024**")
        st.markdown("📍 Chicago, IL")
        st.markdown("🎯 Goal: $1M by 2035")
        st.progress(0.125, text="12.5% to goal")
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        if st.button("📊 View Portfolio", key="quick_portfolio"):
            st.info("Portfolio view would open here")
        
        if st.button("💰 Add Transaction", key="quick_transaction"):
            st.info("Transaction form would open here")
        
        if st.button("🎯 Set New Goal", key="quick_goal"):
            st.info("Goal setting would open here")
    
    # Main content area
    if page == "🏠 Dashboard":
        render_dashboard()
    elif page == "📰 Financial Newsfeed":
        display_newsfeed()
    elif page == "🎓 Education Center":
        display_tutorial("financial_education")
    elif page == "💬 Community Forum":
        display_forum()

def render_dashboard():
    """Render the main dashboard"""
    
    # Main header
    st.markdown("""
    <div class='main-header'>
        <h1 style='margin: 0; font-size: 2.5rem;'>💰 Welcome to Pecunia AI</h1>
        <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;'>
            Your comprehensive AI-driven financial management platform
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics overview
    st.markdown("## 📊 Financial Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='margin: 0; font-size: 1.5rem;'>$125,432</h3>
            <p style='margin: 0; opacity: 0.9;'>Total Portfolio Value</p>
            <p style='margin: 0; color: #90EE90;'>↗️ +2.3% this month</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='margin: 0; font-size: 1.5rem;'>$2,850</h3>
            <p style='margin: 0; opacity: 0.9;'>Monthly Savings</p>
            <p style='margin: 0; color: #90EE90;'>↗️ +12% vs target</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='margin: 0; font-size: 1.5rem;'>8.7%</h3>
            <p style='margin: 0; opacity: 0.9;'>YTD Return</p>
            <p style='margin: 0; color: #90EE90;'>↗️ +1.2% vs S&P 500</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='margin: 0; font-size: 1.5rem;'>Moderate</h3>
            <p style='margin: 0; opacity: 0.9;'>Risk Level</p>
            <p style='margin: 0; opacity: 0.8;'>📊 Balanced portfolio</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Feature highlights
    st.markdown("## 🚀 Platform Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='feature-card'>
            <h3>📰 AI-Powered Newsfeed</h3>
            <p>Stay updated with personalized financial news, market analysis, and AI-generated summaries tailored to your investment interests.</p>
            <ul>
                <li>Real-time market news</li>
                <li>AI sentiment analysis</li>
                <li>Personalized content</li>
                <li>Multiple news sources</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📰 Explore Newsfeed", key="nav_newsfeed"):
            st.session_state.page = "📰 Financial Newsfeed"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class='feature-card'>
            <h3>🎓 Financial Education</h3>
            <p>Master personal finance with interactive courses, tutorials, and hands-on learning experiences designed for all skill levels.</p>
            <ul>
                <li>Interactive courses</li>
                <li>Progress tracking</li>
                <li>Practical tools</li>
                <li>Certificates</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🎓 Start Learning", key="nav_education"):
            st.session_state.page = "🎓 Education Center"
            st.rerun()
    
    with col3:
        st.markdown("""
        <div class='feature-card'>
            <h3>💬 Community Forum</h3>
            <p>Connect with fellow investors, ask questions, share insights, and learn from experienced financial experts in our community.</p>
            <ul>
                <li>Expert discussions</li>
                <li>Q&A platform</li>
                <li>Peer learning</li>
                <li>Success stories</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("💬 Join Community", key="nav_community"):
            st.session_state.page = "💬 Community Forum"
            st.rerun()
    
    st.markdown("---")
    
    # Recent activity and quick insights
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 📈 Recent Market Insights")
        
        # Sample insights
        insights = [
            {
                "title": "S&P 500 Reaches New High",
                "summary": "The market continues its upward trend with strong earnings reports from tech giants.",
                "sentiment": "Positive",
                "impact": "Your portfolio is well-positioned to benefit from this trend."
            },
            {
                "title": "Federal Reserve Policy Update",
                "summary": "Latest Fed meeting suggests stable interest rates for the remainder of the quarter.",
                "sentiment": "Neutral",
                "impact": "Consider maintaining current bond allocation."
            },
            {
                "title": "Cryptocurrency Market Recovery",
                "summary": "Bitcoin and major altcoins show signs of recovery after recent volatility.",
                "sentiment": "Positive",
                "impact": "Your 5% crypto allocation is performing well."
            }
        ]
        
        for insight in insights:
            sentiment_color = "#28a745" if insight["sentiment"] == "Positive" else "#ffc107" if insight["sentiment"] == "Neutral" else "#dc3545"
            
            st.markdown(f"""
            <div style='border-left: 4px solid {sentiment_color}; padding: 1rem; margin-bottom: 1rem; background: #f8f9fa; border-radius: 0 10px 10px 0;'>
                <h4 style='margin: 0 0 0.5rem 0; color: #333;'>{insight['title']}</h4>
                <p style='margin: 0 0 0.5rem 0; color: #666;'>{insight['summary']}</p>
                <p style='margin: 0; color: {sentiment_color}; font-weight: bold;'>💡 {insight['impact']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("## 🎯 Recommendations")
        
        recommendations = [
            "📚 Complete 'Portfolio Diversification' course",
            "💰 Increase emergency fund by $500",
            "📊 Rebalance portfolio (overweight in tech)",
            "🎓 Learn about tax-loss harvesting",
            "💬 Join retirement planning discussion"
        ]
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"""
            <div style='background: white; padding: 0.8rem; margin-bottom: 0.5rem; border-radius: 8px; border: 1px solid #e0e0e0;'>
                <div style='font-size: 0.9rem; color: #333;'>{i}. {rec}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Achievement badges
        st.markdown("### 🏆 Recent Achievements")
        
        badges = [
            {"name": "First Investment", "icon": "🚀"},
            {"name": "Consistent Saver", "icon": "💰"},
            {"name": "Course Completer", "icon": "🎓"},
            {"name": "Community Helper", "icon": "💬"}
        ]
        
        for badge in badges:
            st.markdown(f"""
            <div style='background: linear-gradient(90deg, #FFD700, #FFA500); color: white; padding: 0.5rem; margin-bottom: 0.3rem; border-radius: 20px; text-align: center; font-size: 0.8rem;'>
                {badge['icon']} {badge['name']}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Call-to-action section
    st.markdown("## 🚀 Ready to Take Control of Your Financial Future?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 View Detailed Analytics", type="primary", key="cta_analytics"):
            st.info("Analytics dashboard would open here")
    
    with col2:
        if st.button("🤖 Get AI Recommendations", type="primary", key="cta_ai"):
            st.info("AI recommendation engine would start here")
    
    with col3:
        if st.button("📱 Download Mobile App", type="primary", key="cta_mobile"):
            st.info("Mobile app download links would appear here")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem; padding: 2rem 0;'>
        <p>💰 <strong>Pecunia AI</strong> - Empowering Your Financial Journey with Artificial Intelligence</p>
        <p>📧 Contact: support@pecunia.ai | 📞 1-800-PECUNIA | 🌐 www.pecunia.ai</p>
        <p style='font-size: 0.8rem; opacity: 0.7;'>© 2024 Pecunia AI. All rights reserved. Investment advice is for educational purposes only.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 