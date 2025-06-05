#!/usr/bin/env python3
"""
Pecunia AI - Financial Education Module
Interactive financial education system with tutorials, courses, and personalized learning
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from enum import Enum
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DifficultyLevel(Enum):
    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"

class CourseCategory(Enum):
    INVESTING = "Investing & Trading"
    PERSONAL_FINANCE = "Personal Finance"
    CRYPTOCURRENCY = "Cryptocurrency"
    RETIREMENT = "Retirement Planning"
    TAXES = "Tax Planning"
    REAL_ESTATE = "Real Estate"
    BUSINESS = "Business & Entrepreneurship"
    RISK_MANAGEMENT = "Risk Management"

@dataclass
class Quiz:
    """Quiz data structure"""
    questions: List[Dict]
    passing_score: int = 70
    time_limit: Optional[int] = None

@dataclass
class Lesson:
    """Individual lesson structure"""
    title: str
    content: str
    duration: int  # in minutes
    video_url: Optional[str] = None
    interactive_elements: List[Dict] = None
    quiz: Optional[Quiz] = None

@dataclass
class Course:
    """Course structure"""
    id: str
    title: str
    description: str
    category: CourseCategory
    difficulty: DifficultyLevel
    lessons: List[Lesson]
    estimated_hours: int
    prerequisites: List[str] = None
    learning_objectives: List[str] = None
    instructor: str = "Pecunia AI"

@dataclass
class UserProgress:
    """Track user learning progress"""
    course_id: str
    completed_lessons: List[int]
    quiz_scores: Dict[int, int]
    total_time_spent: int  # in minutes
    last_accessed: datetime
    completion_percentage: float
    certificates_earned: List[str] = None

class FinancialEducationSystem:
    """Main education system with courses and tutorials"""
    
    def __init__(self):
        self.courses = self._initialize_courses()
        self.user_progress = {}
        
    def _initialize_courses(self) -> Dict[str, Course]:
        """Initialize the course catalog"""
        courses = {}
        
        # Beginner Investing Course
        investing_basics = Course(
            id="investing_basics",
            title="Investing Fundamentals",
            description="Learn the basics of investing, from stocks and bonds to portfolio diversification.",
            category=CourseCategory.INVESTING,
            difficulty=DifficultyLevel.BEGINNER,
            estimated_hours=8,
            learning_objectives=[
                "Understand different types of investments",
                "Learn about risk and return",
                "Master portfolio diversification",
                "Understand market basics"
            ],
            lessons=[
                Lesson(
                    title="Introduction to Investing",
                    content="""
# Introduction to Investing

## What is Investing?
Investing is the process of putting money to work in order to grow it over time. When you invest, you're essentially buying an asset that you expect will increase in value or generate income.

## Key Investment Types

### 1. Stocks
- Represent ownership in a company
- Potential for capital appreciation and dividends
- Higher risk, higher potential return

### 2. Bonds
- Loans to companies or governments
- Regular interest payments
- Generally lower risk than stocks

### 3. Mutual Funds & ETFs
- Diversified portfolios managed by professionals
- Spread risk across many investments
- Good for beginners

### 4. Real Estate
- Physical property investments
- Potential for rental income and appreciation
- Requires more capital and management

## Why Should You Invest?

1. **Beat Inflation**: Money loses purchasing power over time
2. **Build Wealth**: Compound growth can significantly increase your money
3. **Reach Financial Goals**: Retirement, house down payment, etc.
4. **Generate Income**: Dividends and interest payments

## Getting Started

The key is to start early and be consistent. Even small amounts can grow significantly over time thanks to compound interest.
                    """,
                    duration=30,
                    interactive_elements=[
                        {
                            "type": "calculator",
                            "title": "Compound Interest Calculator",
                            "description": "See how your money can grow over time"
                        }
                    ],
                    quiz=Quiz(
                        questions=[
                            {
                                "question": "What is the main goal of investing?",
                                "options": ["To have fun", "To grow money over time", "To show off", "To gamble"],
                                "correct": 1,
                                "explanation": "The primary goal of investing is to grow your money over time to build wealth and reach financial goals."
                            },
                            {
                                "question": "Which investment type represents ownership in a company?",
                                "options": ["Bonds", "Stocks", "CDs", "Savings Account"],
                                "correct": 1,
                                "explanation": "Stocks represent partial ownership (equity) in a company."
                            }
                        ]
                    )
                ),
                Lesson(
                    title="Understanding Risk and Return",
                    content="""
# Understanding Risk and Return

## The Risk-Return Relationship

One of the most fundamental concepts in investing is that **higher potential returns typically come with higher risk**.

## Types of Investment Risk

### 1. Market Risk
- Overall market movements affect your investments
- Economic conditions, interest rates, political events
- Cannot be eliminated through diversification

### 2. Company-Specific Risk
- Risk specific to individual companies
- Poor management, product failures, scandals
- Can be reduced through diversification

### 3. Inflation Risk
- Risk that inflation will erode purchasing power
- Particularly affects fixed-income investments
- Stocks historically provide inflation protection

### 4. Liquidity Risk
- Risk that you cannot sell your investment quickly
- Real estate and some bonds have higher liquidity risk
- Stocks are generally very liquid

## Measuring Risk

### Standard Deviation
- Measures how much an investment's returns vary
- Higher standard deviation = higher volatility = higher risk

### Beta
- Measures how much an investment moves relative to the market
- Beta of 1 = moves with the market
- Beta > 1 = more volatile than market
- Beta < 1 = less volatile than market

## Risk Tolerance

Your risk tolerance depends on:
- **Time Horizon**: Longer = can take more risk
- **Financial Situation**: More savings = can take more risk
- **Emotional Comfort**: How well you sleep when markets are down
- **Age**: Younger = typically can take more risk

## Risk Management Strategies

1. **Diversification**: Don't put all eggs in one basket
2. **Asset Allocation**: Mix of stocks, bonds, other investments
3. **Dollar-Cost Averaging**: Invest fixed amounts regularly
4. **Rebalancing**: Maintain your target allocation
                    """,
                    duration=25,
                    quiz=Quiz(
                        questions=[
                            {
                                "question": "What is the relationship between risk and return?",
                                "options": ["No relationship", "Higher risk = lower return", "Higher risk = higher potential return", "Risk doesn't matter"],
                                "correct": 2,
                                "explanation": "Generally, investments with higher risk offer the potential for higher returns to compensate investors for taking on that risk."
                            }
                        ]
                    )
                ),
                Lesson(
                    title="Building Your First Portfolio",
                    content="""
# Building Your First Portfolio

## What is a Portfolio?

A portfolio is your collection of investments. Think of it as your investment "basket" containing different types of assets.

## Asset Allocation

This is how you divide your money among different investment types:

### Age-Based Rule of Thumb
- **Stock Percentage = 120 - Your Age**
- If you're 30: 120 - 30 = 90% stocks, 10% bonds
- If you're 50: 120 - 50 = 70% stocks, 30% bonds

### Target Date Funds
- Automatically adjust allocation as you age
- Start aggressive (more stocks) when young
- Become conservative (more bonds) as you near retirement

## Diversification Strategies

### Geographic Diversification
- U.S. stocks: 60-70%
- International developed markets: 20-30%
- Emerging markets: 5-10%

### Sector Diversification
- Technology, Healthcare, Finance, Consumer goods, etc.
- Don't put all money in one industry

### Market Cap Diversification
- Large-cap stocks (stable, established companies)
- Mid-cap stocks (growing companies)
- Small-cap stocks (smaller, potentially high-growth companies)

## Sample Beginner Portfolios

### Conservative (Low Risk)
- 40% U.S. Total Stock Market
- 20% International Stocks
- 35% Bonds
- 5% REITs

### Moderate (Medium Risk)
- 60% U.S. Total Stock Market
- 25% International Stocks
- 10% Bonds
- 5% REITs

### Aggressive (High Risk)
- 70% U.S. Total Stock Market
- 25% International Stocks
- 5% Bonds

## Getting Started

1. **Determine your risk tolerance**
2. **Choose your asset allocation**
3. **Select low-cost index funds or ETFs**
4. **Invest regularly (dollar-cost averaging)**
5. **Rebalance annually**
6. **Stay the course - don't panic sell**

## Common Beginner Mistakes

- Trying to time the market
- Picking individual stocks without research
- Panic selling during market downturns
- Not diversifying enough
- Paying high fees
- Not starting early enough
                    """,
                    duration=35,
                    interactive_elements=[
                        {
                            "type": "portfolio_builder",
                            "title": "Portfolio Allocation Tool",
                            "description": "Build and visualize your investment portfolio"
                        }
                    ]
                )
            ]
        )
        courses["investing_basics"] = investing_basics
        
        # Personal Finance Course
        personal_finance = Course(
            id="personal_finance",
            title="Personal Finance Mastery",
            description="Master budgeting, saving, debt management, and financial planning.",
            category=CourseCategory.PERSONAL_FINANCE,
            difficulty=DifficultyLevel.BEGINNER,
            estimated_hours=6,
            learning_objectives=[
                "Create and maintain a budget",
                "Build an emergency fund",
                "Manage and eliminate debt",
                "Set and achieve financial goals"
            ],
            lessons=[
                Lesson(
                    title="Budgeting Basics",
                    content="""
# Budgeting Basics

## Why Budget?

Budgeting is the foundation of financial success. It helps you:
- Track where your money goes
- Ensure you're saving for goals
- Avoid overspending
- Build wealth over time

## The 50/30/20 Rule

A simple budgeting framework:
- **50% Needs**: Housing, utilities, groceries, transportation
- **30% Wants**: Entertainment, dining out, hobbies
- **20% Savings & Debt Payment**: Emergency fund, retirement, extra debt payments

## Zero-Based Budgeting

Every dollar gets assigned a job:
- Income - Expenses - Savings = $0
- More detailed than 50/30/20
- Requires tracking every expense category

## Popular Budgeting Methods

### Envelope Method
- Cash for each spending category
- When envelope is empty, you're done spending
- Works well for discretionary spending

### Pay Yourself First
- Save/invest before paying other expenses
- Automate savings transfers
- Live on what's left

## Budgeting Tools

- **Spreadsheets**: Excel, Google Sheets
- **Apps**: Mint, YNAB, Personal Capital
- **Bank Tools**: Many banks offer budgeting features
- **Cash Envelopes**: Physical cash method

## Getting Started

1. **Track current spending** for 2-4 weeks
2. **Categorize expenses** (needs vs. wants)
3. **Set spending limits** for each category
4. **Automate savings** and bill payments
5. **Review and adjust** monthly
                    """,
                    duration=25
                ),
                Lesson(
                    title="Emergency Fund Essentials",
                    content="""
# Emergency Fund Essentials

## What is an Emergency Fund?

An emergency fund is money set aside for unexpected expenses or financial emergencies. It's your financial safety net.

## Why You Need One

- **Job loss** or reduced income
- **Medical emergencies** not covered by insurance
- **Car repairs** or major breakdowns
- **Home repairs** (roof, plumbing, etc.)
- **Unexpected travel** (family emergencies)

## How Much to Save

### Starter Emergency Fund
- **$1,000 minimum** for beginners
- Covers small emergencies while building wealth

### Full Emergency Fund
- **3-6 months of expenses** for most people
- **6-12 months** if you're self-employed or have irregular income
- **3 months** if you have very stable employment

## Where to Keep It

### High-Yield Savings Account
- Easy access to funds
- FDIC insured
- Earns some interest

### Money Market Account
- Slightly higher interest than savings
- May have minimum balance requirements
- Still liquid and accessible

### Short-term CDs
- For part of emergency fund
- Higher interest rates
- Less liquid but still accessible

## Building Your Emergency Fund

### Step-by-Step Process
1. **Start small**: Even $25/month helps
2. **Automate transfers**: Set up automatic savings
3. **Use windfalls**: Tax refunds, bonuses, gifts
4. **Sell unused items**: Declutter and save the money
5. **Temporary sacrifices**: Cut expenses temporarily

### Ways to Boost Your Fund
- **Side hustle**: Freelancing, part-time work
- **Cashback credit cards**: Put cashback into emergency fund
- **Spare change apps**: Round up purchases and save
- **Challenge yourself**: No-spend weeks or months

## Don't Touch It!

Emergency fund is NOT for:
- Vacations
- Christmas gifts
- Car down payments
- Investment opportunities
- Paying off debt (use separate money for this)

## Replenish After Use

If you use your emergency fund:
1. **Don't feel guilty** - that's what it's for!
2. **Make replenishing a priority**
3. **Temporarily reduce other savings** to rebuild it
4. **Learn from the experience** - could it have been prevented?
                    """,
                    duration=20
                )
            ]
        )
        courses["personal_finance"] = personal_finance
        
        # Cryptocurrency Course
        crypto_course = Course(
            id="crypto_basics",
            title="Cryptocurrency Fundamentals",
            description="Understand blockchain technology, Bitcoin, Ethereum, and crypto investing.",
            category=CourseCategory.CRYPTOCURRENCY,
            difficulty=DifficultyLevel.INTERMEDIATE,
            estimated_hours=10,
            learning_objectives=[
                "Understand blockchain technology",
                "Learn about major cryptocurrencies",
                "Understand crypto trading and investing",
                "Learn about DeFi and NFTs"
            ],
            lessons=[
                Lesson(
                    title="Introduction to Blockchain",
                    content="""
# Introduction to Blockchain

## What is Blockchain?

Blockchain is a distributed ledger technology that maintains a continuously growing list of records (blocks) that are linked and secured using cryptography.

## Key Characteristics

### Decentralization
- No single point of control
- Network of computers (nodes) maintains the system
- Reduces reliance on central authorities

### Transparency
- All transactions are visible on the network
- Public ledger that anyone can verify
- Promotes trust and accountability

### Immutability
- Once data is recorded, it's extremely difficult to change
- Each block contains a cryptographic hash of the previous block
- Provides security and prevents fraud

### Security
- Cryptographic protection
- Consensus mechanisms prevent malicious attacks
- Distributed nature makes it resilient

## How Blockchain Works

1. **Transaction Initiated**: User sends cryptocurrency to another user
2. **Broadcasting**: Transaction is broadcast to the network
3. **Validation**: Network nodes validate the transaction
4. **Block Creation**: Valid transactions are grouped into a block
5. **Consensus**: Network agrees on the new block
6. **Block Addition**: New block is added to the chain
7. **Completion**: Transaction is complete and immutable

## Types of Blockchain

### Public Blockchain
- Open to everyone
- Fully decentralized
- Bitcoin, Ethereum

### Private Blockchain
- Restricted access
- Controlled by organization
- Used by companies for internal processes

### Consortium Blockchain
- Semi-decentralized
- Controlled by a group of organizations
- Faster than public, more open than private

## Blockchain Applications

- **Cryptocurrencies**: Digital money
- **Smart Contracts**: Self-executing contracts
- **Supply Chain**: Track products from origin to consumer
- **Identity Verification**: Secure digital identities
- **Voting**: Transparent and secure elections
- **Real Estate**: Property records and transfers
                    """,
                    duration=30
                ),
                Lesson(
                    title="Understanding Bitcoin",
                    content="""
# Understanding Bitcoin

## What is Bitcoin?

Bitcoin is the first and most well-known cryptocurrency, created in 2009 by the pseudonymous Satoshi Nakamoto. It's often called "digital gold."

## Key Features

### Limited Supply
- Maximum 21 million bitcoins will ever exist
- Currently about 19+ million in circulation
- Scarcity creates potential for value appreciation

### Peer-to-Peer
- Direct transactions without intermediaries
- No banks or payment processors needed
- Lower fees for international transfers

### Store of Value
- Often compared to gold
- Hedge against inflation
- Digital asset that maintains purchasing power

## How Bitcoin Works

### Mining
- Computers solve complex mathematical problems
- Miners verify transactions and secure the network
- Rewarded with new bitcoins and transaction fees

### Wallets
- Software that stores your private keys
- Hot wallets (online) vs. Cold wallets (offline)
- Control your private keys = control your bitcoin

### Transactions
- Sent from one address to another
- Verified by the network in ~10 minutes
- Recorded permanently on the blockchain

## Bitcoin Investing

### Ways to Buy Bitcoin
- **Cryptocurrency exchanges**: Coinbase, Binance, Kraken
- **Traditional brokers**: Some now offer crypto
- **Bitcoin ATMs**: Physical machines in many cities
- **Peer-to-peer**: Direct from other users

### Storage Options
- **Exchange wallets**: Convenient but less secure
- **Software wallets**: Apps on your phone/computer
- **Hardware wallets**: Physical devices (most secure)
- **Paper wallets**: Private keys printed on paper

### Investment Strategies
- **Dollar-cost averaging**: Buy fixed amounts regularly
- **HODLing**: Buy and hold long-term
- **Trading**: Active buying and selling (risky)

## Risks and Considerations

### Volatility
- Bitcoin prices can swing dramatically
- Not suitable for short-term needs
- Can lose significant value quickly

### Regulatory Risk
- Government regulations can affect price
- Some countries have banned or restricted Bitcoin
- Tax implications vary by jurisdiction

### Technical Risk
- Lost private keys = lost bitcoin
- Scams and fraudulent exchanges
- User error can result in permanent loss

### Environmental Concerns
- Bitcoin mining consumes significant energy
- Some argue it's wasteful
- Industry working on renewable energy solutions
                    """,
                    duration=35
                )
            ]
        )
        courses["crypto_basics"] = crypto_course
        
        return courses
    
    def get_user_progress(self, user_id: str, course_id: str) -> UserProgress:
        """Get user progress for a specific course"""
        key = f"{user_id}_{course_id}"
        if key not in self.user_progress:
            self.user_progress[key] = UserProgress(
                course_id=course_id,
                completed_lessons=[],
                quiz_scores={},
                total_time_spent=0,
                last_accessed=datetime.now(),
                completion_percentage=0.0
            )
        return self.user_progress[key]
    
    def update_progress(self, user_id: str, course_id: str, lesson_index: int, time_spent: int = 0, quiz_score: int = None):
        """Update user progress"""
        progress = self.get_user_progress(user_id, course_id)
        
        if lesson_index not in progress.completed_lessons:
            progress.completed_lessons.append(lesson_index)
        
        progress.total_time_spent += time_spent
        progress.last_accessed = datetime.now()
        
        if quiz_score is not None:
            progress.quiz_scores[lesson_index] = quiz_score
        
        # Calculate completion percentage
        total_lessons = len(self.courses[course_id].lessons)
        progress.completion_percentage = (len(progress.completed_lessons) / total_lessons) * 100
    
    def get_recommended_courses(self, user_level: DifficultyLevel, interests: List[str] = None) -> List[Course]:
        """Get recommended courses based on user level and interests"""
        recommended = []
        
        for course in self.courses.values():
            if course.difficulty == user_level:
                if interests:
                    if any(interest.lower() in course.title.lower() or 
                          interest.lower() in course.description.lower() for interest in interests):
                        recommended.append(course)
                else:
                    recommended.append(course)
        
        return recommended

class EducationDisplay:
    """Handle the display and interaction for the education system"""
    
    def __init__(self):
        self.education_system = FinancialEducationSystem()
        
    def render_education_header(self):
        """Render the education module header"""
        st.markdown("""
        <div style='background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem;'>
            <h1 style='color: white; text-align: center; margin: 0;'>
                🎓 Financial Education Academy
            </h1>
            <p style='color: white; text-align: center; margin: 0.5rem 0 0 0; opacity: 0.9;'>
                Master personal finance with interactive courses and tutorials
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_course_catalog(self):
        """Display the course catalog"""
        st.markdown("## 📚 Course Catalog")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            difficulty_filter = st.selectbox(
                "🎯 Difficulty Level",
                ["All"] + [level.value for level in DifficultyLevel]
            )
        
        with col2:
            category_filter = st.selectbox(
                "🏷️ Category",
                ["All"] + [cat.value for cat in CourseCategory]
            )
        
        with col3:
            sort_by = st.selectbox(
                "📊 Sort By",
                ["Recommended", "Duration", "Difficulty", "Category"]
            )
        
        # Filter courses
        courses = list(self.education_system.courses.values())
        
        if difficulty_filter != "All":
            courses = [c for c in courses if c.difficulty.value == difficulty_filter]
        
        if category_filter != "All":
            courses = [c for c in courses if c.category.value == category_filter]
        
        # Display courses
        for course in courses:
            self.render_course_card(course)
    
    def render_course_card(self, course: Course):
        """Render an individual course card"""
        # Progress for current user (demo)
        progress = self.education_system.get_user_progress("demo_user", course.id)
        
        # Difficulty color
        diff_colors = {
            "Beginner": "#28a745",
            "Intermediate": "#ffc107", 
            "Advanced": "#dc3545"
        }
        diff_color = diff_colors.get(course.difficulty.value, "#6c757d")
        
        card_html = f"""
        <div style='border: 1px solid #e0e0e0; border-radius: 10px; padding: 1.5rem; margin-bottom: 1.5rem; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <div style='display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem;'>
                <div style='flex-grow: 1;'>
                    <h3 style='margin: 0 0 0.5rem 0; color: #333;'>{course.title}</h3>
                    <p style='margin: 0 0 1rem 0; color: #666; line-height: 1.5;'>{course.description}</p>
                    
                    <div style='display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;'>
                        <span style='background: {diff_color}; color: white; padding: 0.25rem 0.5rem; border-radius: 15px; font-size: 0.8rem;'>
                            {course.difficulty.value}
                        </span>
                        <span style='color: #666; font-size: 0.9rem;'>
                            ⏱️ {course.estimated_hours} hours
                        </span>
                        <span style='color: #666; font-size: 0.9rem;'>
                            📖 {len(course.lessons)} lessons
                        </span>
                        <span style='color: #666; font-size: 0.9rem;'>
                            🎯 {progress.completion_percentage:.0f}% complete
                        </span>
                    </div>
                    
                    <div style='margin-bottom: 1rem;'>
                        <div style='background: #f8f9fa; border-radius: 5px; padding: 0.5rem;'>
                            <strong>Learning Objectives:</strong>
                            <ul style='margin: 0.5rem 0 0 0; padding-left: 1.5rem;'>
                                {''.join(f'<li>{obj}</li>' for obj in (course.learning_objectives or [])[:3])}
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
            
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div style='display: flex; gap: 0.5rem;'>
                    <span style='background: #e7f3ff; color: #0066cc; padding: 0.25rem 0.5rem; border-radius: 15px; font-size: 0.8rem;'>
                        {course.category.value}
                    </span>
                </div>
            </div>
        </div>
        """
        
        st.markdown(card_html, unsafe_allow_html=True)
        
        # Course action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button(f"📖 Start Course", key=f"start_{course.id}"):
                st.session_state.current_course = course.id
                st.session_state.current_lesson = 0
                st.rerun()
        
        with col2:
            if st.button(f"📊 View Progress", key=f"progress_{course.id}"):
                st.session_state.show_progress = course.id
                st.rerun()
        
        with col3:
            if progress.completion_percentage > 0:
                if st.button(f"▶️ Continue", key=f"continue_{course.id}"):
                    st.session_state.current_course = course.id
                    st.session_state.current_lesson = len(progress.completed_lessons)
                    st.rerun()
    
    def render_lesson_content(self, course: Course, lesson_index: int):
        """Render individual lesson content"""
        if lesson_index >= len(course.lessons):
            st.error("Lesson not found!")
            return
        
        lesson = course.lessons[lesson_index]
        
        # Lesson header
        st.markdown(f"""
        <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem; color: white;'>
            <h2 style='margin: 0; color: white;'>{lesson.title}</h2>
            <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>
                📖 Lesson {lesson_index + 1} of {len(course.lessons)} • ⏱️ {lesson.duration} minutes
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if lesson_index > 0:
                if st.button("⬅️ Previous Lesson"):
                    st.session_state.current_lesson = lesson_index - 1
                    st.rerun()
        
        with col3:
            if lesson_index < len(course.lessons) - 1:
                if st.button("Next Lesson ➡️"):
                    st.session_state.current_lesson = lesson_index + 1
                    # Mark current lesson as completed
                    self.education_system.update_progress("demo_user", course.id, lesson_index, lesson.duration)
                    st.rerun()
        
        # Lesson content
        st.markdown(lesson.content)
        
        # Interactive elements
        if lesson.interactive_elements:
            st.markdown("---")
            st.markdown("## 🛠️ Interactive Elements")
            
            for element in lesson.interactive_elements:
                self.render_interactive_element(element)
        
        # Quiz
        if lesson.quiz:
            st.markdown("---")
            self.render_quiz(lesson.quiz, course.id, lesson_index)
        
        # Mark as completed button
        if st.button("✅ Mark Lesson as Completed", type="primary"):
            self.education_system.update_progress("demo_user", course.id, lesson_index, lesson.duration)
            st.success("Lesson marked as completed!")
            time.sleep(1)
            if lesson_index < len(course.lessons) - 1:
                st.session_state.current_lesson = lesson_index + 1
                st.rerun()
    
    def render_interactive_element(self, element: Dict):
        """Render interactive learning elements"""
        if element["type"] == "calculator":
            self.render_compound_interest_calculator()
        elif element["type"] == "portfolio_builder":
            self.render_portfolio_builder()
    
    def render_compound_interest_calculator(self):
        """Interactive compound interest calculator"""
        st.markdown("### 📈 Compound Interest Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            initial_amount = st.number_input("Initial Investment ($)", min_value=0, value=1000, step=100)
            monthly_contribution = st.number_input("Monthly Contribution ($)", min_value=0, value=200, step=50)
            annual_return = st.slider("Expected Annual Return (%)", min_value=1, max_value=15, value=7, step=1)
            years = st.slider("Investment Period (Years)", min_value=1, max_value=50, value=20, step=1)
        
        # Calculate compound interest
        monthly_rate = annual_return / 100 / 12
        months = years * 12
        
        # Future value calculation
        fv_initial = initial_amount * (1 + annual_return/100) ** years
        fv_monthly = monthly_contribution * (((1 + monthly_rate) ** months - 1) / monthly_rate) if monthly_rate > 0 else monthly_contribution * months
        total_future_value = fv_initial + fv_monthly
        total_invested = initial_amount + (monthly_contribution * months)
        total_earnings = total_future_value - total_invested
        
        with col2:
            st.markdown(f"""
            **Results:**
            - **Total Invested:** ${total_invested:,.0f}
            - **Total Earnings:** ${total_earnings:,.0f}
            - **Final Amount:** ${total_future_value:,.0f}
            - **Return Multiple:** {total_future_value/total_invested:.1f}x
            """)
        
        # Create chart
        years_list = list(range(0, years + 1))
        balances = []
        
        for year in years_list:
            if year == 0:
                balance = initial_amount
            else:
                months_elapsed = year * 12
                fv_initial_year = initial_amount * (1 + annual_return/100) ** year
                fv_monthly_year = monthly_contribution * (((1 + monthly_rate) ** months_elapsed - 1) / monthly_rate) if monthly_rate > 0 else monthly_contribution * months_elapsed
                balance = fv_initial_year + fv_monthly_year
            balances.append(balance)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years_list, y=balances, mode='lines+markers', name='Portfolio Value'))
        fig.update_layout(title="Portfolio Growth Over Time", xaxis_title="Years", yaxis_title="Portfolio Value ($)")
        st.plotly_chart(fig, use_container_width=True)
    
    def render_portfolio_builder(self):
        """Interactive portfolio allocation tool"""
        st.markdown("### 🥧 Portfolio Allocation Builder")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Adjust your allocation:**")
            stocks = st.slider("U.S. Stocks (%)", 0, 100, 60)
            intl_stocks = st.slider("International Stocks (%)", 0, 100-stocks, 20)
            bonds = st.slider("Bonds (%)", 0, 100-stocks-intl_stocks, 15)
            reits = 100 - stocks - intl_stocks - bonds
            
            st.write(f"REITs: {reits}% (auto-calculated)")
            
            if stocks + intl_stocks + bonds + reits != 100:
                st.warning("Allocation must total 100%")
        
        with col2:
            # Create pie chart
            labels = ['U.S. Stocks', 'International Stocks', 'Bonds', 'REITs']
            values = [stocks, intl_stocks, bonds, reits]
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
            fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                            marker=dict(colors=colors, line=dict(color='#000000', width=2)))
            fig.update_layout(title="Your Portfolio Allocation")
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk assessment
            risk_score = (stocks * 1.0 + intl_stocks * 0.9 + reits * 0.8 + bonds * 0.3) / 100
            risk_level = "Conservative" if risk_score < 0.4 else "Moderate" if risk_score < 0.7 else "Aggressive"
            
            st.markdown(f"""
            **Portfolio Analysis:**
            - **Risk Level:** {risk_level}
            - **Risk Score:** {risk_score:.2f}/1.0
            - **Expected Annual Return:** {5 + risk_score * 5:.1f}%
            - **Volatility:** {8 + risk_score * 12:.1f}%
            """)
    
    def render_quiz(self, quiz: Quiz, course_id: str, lesson_index: int):
        """Render interactive quiz"""
        st.markdown("## 🧠 Knowledge Check")
        
        # Initialize quiz state
        if f"quiz_answers_{course_id}_{lesson_index}" not in st.session_state:
            st.session_state[f"quiz_answers_{course_id}_{lesson_index}"] = {}
        
        quiz_key = f"quiz_answers_{course_id}_{lesson_index}"
        
        # Display questions
        for i, question in enumerate(quiz.questions):
            st.markdown(f"**Question {i+1}:** {question['question']}")
            
            answer = st.radio(
                f"Select your answer:",
                question["options"],
                key=f"quiz_{course_id}_{lesson_index}_{i}"
            )
            
            st.session_state[quiz_key][i] = question["options"].index(answer)
        
        # Submit quiz
        if st.button("Submit Quiz", type="primary"):
            score = 0
            total = len(quiz.questions)
            
            # Calculate score
            for i, question in enumerate(quiz.questions):
                if st.session_state[quiz_key].get(i) == question["correct"]:
                    score += 1
            
            percentage = (score / total) * 100
            
            # Update progress
            self.education_system.update_progress("demo_user", course_id, lesson_index, quiz_score=percentage)
            
            # Show results
            if percentage >= quiz.passing_score:
                st.success(f"🎉 Congratulations! You scored {percentage:.0f}% ({score}/{total})")
            else:
                st.error(f"📚 You scored {percentage:.0f}% ({score}/{total}). You need {quiz.passing_score}% to pass. Please review the material and try again.")
            
            # Show explanations
            st.markdown("### 📝 Answer Explanations")
            for i, question in enumerate(quiz.questions):
                user_answer = st.session_state[quiz_key].get(i, -1)
                correct_answer = question["correct"]
                
                if user_answer == correct_answer:
                    st.success(f"Question {i+1}: ✅ Correct!")
                else:
                    st.error(f"Question {i+1}: ❌ Incorrect. Correct answer: {question['options'][correct_answer]}")
                
                if "explanation" in question:
                    st.info(f"💡 {question['explanation']}")

def display_tutorial(topic: str) -> None:
    """
    Renders interactive tutorials with Streamlit
    Main function for tutorial display
    """
    display = EducationDisplay()
    
    # Check if we're in lesson view
    if hasattr(st.session_state, 'current_course') and st.session_state.current_course:
        course = display.education_system.courses.get(st.session_state.current_course)
        if course:
            lesson_index = getattr(st.session_state, 'current_lesson', 0)
            
            # Course header
            st.markdown(f"# 📚 {course.title}")
            
            # Progress bar
            progress = display.education_system.get_user_progress("demo_user", course.id)
            st.progress(progress.completion_percentage / 100, text=f"Course Progress: {progress.completion_percentage:.0f}%")
            
            # Back to catalog button
            if st.button("⬅️ Back to Course Catalog"):
                del st.session_state.current_course
                if hasattr(st.session_state, 'current_lesson'):
                    del st.session_state.current_lesson
                st.rerun()
            
            # Display lesson
            display.render_lesson_content(course, lesson_index)
            return
    
    # Show progress for specific course
    if hasattr(st.session_state, 'show_progress') and st.session_state.show_progress:
        course_id = st.session_state.show_progress
        course = display.education_system.courses.get(course_id)
        progress = display.education_system.get_user_progress("demo_user", course_id)
        
        st.markdown(f"# 📊 Progress Report: {course.title}")
        
        # Back button
        if st.button("⬅️ Back to Course Catalog"):
            del st.session_state.show_progress
            st.rerun()
        
        # Progress metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Completion", f"{progress.completion_percentage:.0f}%")
        with col2:
            st.metric("Lessons Completed", f"{len(progress.completed_lessons)}/{len(course.lessons)}")
        with col3:
            st.metric("Time Spent", f"{progress.total_time_spent} min")
        with col4:
            avg_quiz_score = np.mean(list(progress.quiz_scores.values())) if progress.quiz_scores else 0
            st.metric("Avg Quiz Score", f"{avg_quiz_score:.0f}%")
        
        # Detailed progress
        st.markdown("### 📖 Lesson Progress")
        for i, lesson in enumerate(course.lessons):
            completed = i in progress.completed_lessons
            quiz_score = progress.quiz_scores.get(i, None)
            
            status = "✅ Completed" if completed else "⏳ Not Started"
            quiz_info = f"Quiz: {quiz_score:.0f}%" if quiz_score else "Quiz: Not Taken"
            
            st.markdown(f"**Lesson {i+1}: {lesson.title}**")
            st.markdown(f"Status: {status} | {quiz_info}")
        
        return
    
    # Default view - course catalog
    st.set_page_config(
        page_title="Pecunia AI - Financial Education",
        page_icon="🎓",
        layout="wide"
    )
    
    display.render_education_header()
    
    # Navigation tabs
    tab1, tab2, tab3 = st.tabs(["📚 Course Catalog", "🎯 My Progress", "🏆 Achievements"])
    
    with tab1:
        display.render_course_catalog()
    
    with tab2:
        st.markdown("## 📊 Your Learning Progress")
        
        # Overall stats
        total_courses = len(display.education_system.courses)
        completed_courses = 0
        total_time = 0
        
        for course_id in display.education_system.courses.keys():
            progress = display.education_system.get_user_progress("demo_user", course_id)
            if progress.completion_percentage == 100:
                completed_courses += 1
            total_time += progress.total_time_spent
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Courses Completed", f"{completed_courses}/{total_courses}")
        with col2:
            st.metric("Total Study Time", f"{total_time} minutes")
        with col3:
            completion_rate = (completed_courses / total_courses) * 100 if total_courses > 0 else 0
            st.metric("Overall Progress", f"{completion_rate:.0f}%")
        
        # Course-specific progress
        st.markdown("### 📚 Course Progress")
        for course in display.education_system.courses.values():
            progress = display.education_system.get_user_progress("demo_user", course.id)
            if progress.total_time_spent > 0:  # Only show started courses
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"**{course.title}**")
                    st.progress(progress.completion_percentage / 100)
                with col2:
                    st.write(f"{progress.completion_percentage:.0f}%")
                with col3:
                    if st.button("Continue", key=f"continue_prog_{course.id}"):
                        st.session_state.current_course = course.id
                        st.session_state.current_lesson = len(progress.completed_lessons)
                        st.rerun()
    
    with tab3:
        st.markdown("## 🏆 Achievements & Certificates")
        
        # Sample achievements
        achievements = [
            {"title": "First Steps", "description": "Completed your first lesson", "icon": "🚀", "unlocked": True},
            {"title": "Quiz Master", "description": "Scored 100% on a quiz", "icon": "🧠", "unlocked": False},
            {"title": "Course Completer", "description": "Completed your first course", "icon": "🎓", "unlocked": False},
            {"title": "Investment Guru", "description": "Completed all investing courses", "icon": "📈", "unlocked": False},
            {"title": "Financial Planner", "description": "Completed personal finance course", "icon": "💰", "unlocked": False}
        ]
        
        for achievement in achievements:
            status = "🔓 Unlocked" if achievement["unlocked"] else "🔒 Locked"
            color = "#d4edda" if achievement["unlocked"] else "#f8d7da"
            
            st.markdown(f"""
            <div style='background: {color}; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
                <h4 style='margin: 0;'>{achievement['icon']} {achievement['title']} - {status}</h4>
                <p style='margin: 0.5rem 0 0 0;'>{achievement['description']}</p>
            </div>
            """, unsafe_allow_html=True)

# Main execution
if __name__ == "__main__":
    display_tutorial("financial_education") 