#!/usr/bin/env python3
"""
Pecunia AI - Community Forum Module
Social platform for financial discussions, Q&A, and peer-to-peer learning
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from enum import Enum
import random
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThreadCategory(Enum):
    GENERAL = "General Discussion"
    INVESTING = "Investing & Trading"
    PERSONAL_FINANCE = "Personal Finance"
    CRYPTOCURRENCY = "Cryptocurrency"
    REAL_ESTATE = "Real Estate"
    RETIREMENT = "Retirement Planning"
    MARKET_ANALYSIS = "Market Analysis"
    BEGINNER = "Beginner Questions"
    SUCCESS_STORIES = "Success Stories"
    TOOLS_RESOURCES = "Tools & Resources"

class UserRole(Enum):
    MEMBER = "Member"
    CONTRIBUTOR = "Contributor"
    EXPERT = "Expert"
    MODERATOR = "Moderator"
    ADMIN = "Admin"

@dataclass
class User:
    """User profile"""
    user_id: str
    username: str
    role: UserRole
    join_date: datetime
    reputation_score: int = 0
    posts_count: int = 0
    helpful_answers: int = 0
    avatar_url: Optional[str] = None
    bio: Optional[str] = None
    location: Optional[str] = None
    website: Optional[str] = None
    badges: List[str] = None

@dataclass
class Post:
    """Individual post/comment"""
    post_id: str
    thread_id: str
    user_id: str
    content: str
    created_at: datetime
    upvotes: int = 0
    downvotes: int = 0
    is_solution: bool = False
    parent_post_id: Optional[str] = None  # For replies
    edited_at: Optional[datetime] = None

@dataclass
class Thread:
    """Discussion thread"""
    thread_id: str
    title: str
    category: ThreadCategory
    user_id: str  # Thread creator
    created_at: datetime
    last_activity: datetime
    view_count: int = 0
    is_pinned: bool = False
    is_locked: bool = False
    is_solved: bool = False
    tags: List[str] = None
    posts: List[Post] = None

class CommunityManager:
    """Manage community data and interactions"""
    
    def __init__(self):
        self.users = {}
        self.threads = {}
        self.posts = {}
        self.user_votes = {}  # Track user voting history
        
        # Initialize with sample data
        self._initialize_sample_data()
    
    def _initialize_sample_data(self):
        """Create sample community data"""
        # Sample users
        sample_users = [
            User(
                user_id="user_1",
                username="InvestorPro",
                role=UserRole.EXPERT,
                join_date=datetime.now() - timedelta(days=365),
                reputation_score=2450,
                posts_count=156,
                helpful_answers=89,
                bio="20+ years investing experience. Focus on value investing and dividend growth.",
                location="New York, NY",
                badges=["Expert Contributor", "Top Advisor", "Helpful Member"]
            ),
            User(
                user_id="user_2",
                username="CryptoEnthusiast",
                role=UserRole.CONTRIBUTOR,
                join_date=datetime.now() - timedelta(days=180),
                reputation_score=1250,
                posts_count=78,
                helpful_answers=34,
                bio="Blockchain developer and crypto investor. Always happy to help beginners.",
                location="San Francisco, CA",
                badges=["Crypto Expert", "Helpful Member"]
            ),
            User(
                user_id="user_3",
                username="BudgetMaster",
                role=UserRole.CONTRIBUTOR,
                join_date=datetime.now() - timedelta(days=90),
                reputation_score=890,
                posts_count=45,
                helpful_answers=28,
                bio="Personal finance coach specializing in budgeting and debt elimination.",
                location="Austin, TX",
                badges=["Budget Guru", "Debt Free"]
            ),
            User(
                user_id="user_4",
                username="NewInvestor2024",
                role=UserRole.MEMBER,
                join_date=datetime.now() - timedelta(days=30),
                reputation_score=120,
                posts_count=15,
                helpful_answers=2,
                bio="Just started my investment journey. Learning as much as I can!",
                location="Chicago, IL",
                badges=["New Member"]
            ),
            User(
                user_id="user_5",
                username="RetirementPlanner",
                role=UserRole.EXPERT,
                join_date=datetime.now() - timedelta(days=500),
                reputation_score=3200,
                posts_count=203,
                helpful_answers=145,
                bio="Certified Financial Planner with expertise in retirement planning and tax strategies.",
                location="Denver, CO",
                badges=["CFP Professional", "Retirement Expert", "Tax Specialist"]
            )
        ]
        
        for user in sample_users:
            self.users[user.user_id] = user
        
        # Sample threads
        sample_threads = [
            Thread(
                thread_id="thread_1",
                title="Best brokerage for beginners in 2024?",
                category=ThreadCategory.BEGINNER,
                user_id="user_4",
                created_at=datetime.now() - timedelta(hours=2),
                last_activity=datetime.now() - timedelta(minutes=15),
                view_count=45,
                tags=["brokerage", "beginner", "2024"],
                posts=[]
            ),
            Thread(
                thread_id="thread_2",
                title="Is Bitcoin a good hedge against inflation?",
                category=ThreadCategory.CRYPTOCURRENCY,
                user_id="user_2",
                created_at=datetime.now() - timedelta(hours=8),
                last_activity=datetime.now() - timedelta(hours=1),
                view_count=123,
                is_pinned=True,
                tags=["bitcoin", "inflation", "hedge"],
                posts=[]
            ),
            Thread(
                thread_id="thread_3",
                title="How I paid off $50k in debt in 2 years",
                category=ThreadCategory.SUCCESS_STORIES,
                user_id="user_3",
                created_at=datetime.now() - timedelta(days=1),
                last_activity=datetime.now() - timedelta(hours=3),
                view_count=289,
                tags=["debt", "success", "budgeting"],
                posts=[]
            ),
            Thread(
                thread_id="thread_4",
                title="Roth IRA vs Traditional IRA for young professionals",
                category=ThreadCategory.RETIREMENT,
                user_id="user_4",
                created_at=datetime.now() - timedelta(days=2),
                last_activity=datetime.now() - timedelta(hours=5),
                view_count=67,
                is_solved=True,
                tags=["IRA", "retirement", "tax"],
                posts=[]
            ),
            Thread(
                thread_id="thread_5",
                title="Market volatility: Time to buy or wait?",
                category=ThreadCategory.MARKET_ANALYSIS,
                user_id="user_1",
                created_at=datetime.now() - timedelta(days=3),
                last_activity=datetime.now() - timedelta(hours=2),
                view_count=156,
                tags=["market", "volatility", "timing"],
                posts=[]
            )
        ]
        
        for thread in sample_threads:
            self.threads[thread.thread_id] = thread
        
        # Sample posts
        sample_posts = [
            # Thread 1 posts
            Post(
                post_id="post_1",
                thread_id="thread_1",
                user_id="user_4",
                content="I'm just starting to invest and wondering which brokerage would be best for a beginner. I've heard about Fidelity, Schwab, and Vanguard. What are the pros and cons of each? I'm planning to start with index fund investing.",
                created_at=datetime.now() - timedelta(hours=2),
                upvotes=5,
                downvotes=0
            ),
            Post(
                post_id="post_2",
                thread_id="thread_1",
                user_id="user_1",
                content="Great question! All three are excellent choices for beginners. Here's my take:\n\n**Fidelity**: Zero expense ratio index funds, great mobile app, excellent customer service.\n\n**Schwab**: Low costs, good research tools, easy-to-use interface.\n\n**Vanguard**: Pioneer in low-cost investing, particularly strong in index funds.\n\nFor a beginner focusing on index funds, you really can't go wrong with any of these. I'd suggest looking at their specific fund offerings and see which aligns best with your investment goals.",
                created_at=datetime.now() - timedelta(hours=1, minutes=30),
                upvotes=12,
                downvotes=0,
                is_solution=True
            ),
            Post(
                post_id="post_3",
                thread_id="thread_1",
                user_id="user_3",
                content="I second what @InvestorPro said. I personally use Fidelity and love their ZERO expense ratio funds. The FZROX (Total Market) and FXNAX (US Bond) are great starting points. Their app is also very user-friendly for beginners.",
                created_at=datetime.now() - timedelta(minutes=45),
                upvotes=8,
                downvotes=0,
                parent_post_id="post_2"
            ),
            
            # Thread 2 posts
            Post(
                post_id="post_4",
                thread_id="thread_2",
                user_id="user_2",
                content="With inflation hitting multi-decade highs, I've been thinking about Bitcoin's role as an inflation hedge. Traditional assets like gold and TIPS are obvious choices, but what are your thoughts on Bitcoin?\n\nHistorically, Bitcoin has shown some correlation with inflation expectations, but it's also incredibly volatile. What's everyone's take on this?",
                created_at=datetime.now() - timedelta(hours=8),
                upvotes=15,
                downvotes=3
            ),
            Post(
                post_id="post_5",
                thread_id="thread_2",
                user_id="user_1",
                content="Bitcoin is still too young and volatile to be considered a reliable inflation hedge, in my opinion. While it has performed well during some inflationary periods, it's also crashed during others.\n\nI think a small allocation (5-10% max) could be reasonable as part of a diversified portfolio, but I wouldn't rely on it as my primary inflation hedge. Stick with proven assets like I-bonds, TIPS, and real estate for the bulk of your inflation protection.",
                created_at=datetime.now() - timedelta(hours=6),
                upvotes=22,
                downvotes=1
            ),
            Post(
                post_id="post_6",
                thread_id="thread_2",
                user_id="user_5",
                content="From a financial planning perspective, I agree with @InvestorPro. Bitcoin is still speculative. For clients concerned about inflation, I typically recommend:\n\n1. I-bonds (guaranteed inflation protection)\n2. TIPS\n3. Real estate (REITs for liquidity)\n4. Commodities\n5. Some international exposure\n\nBitcoin can be a small speculative position, but not a core inflation hedge.",
                created_at=datetime.now() - timedelta(hours=3),
                upvotes=18,
                downvotes=0
            ),
            
            # Thread 3 posts
            Post(
                post_id="post_7",
                thread_id="thread_3",
                user_id="user_3",
                content="Two years ago, I was drowning in debt - $50,000 between credit cards, student loans, and a car loan. Today, I'm completely debt-free! Here's how I did it:\n\n**1. Created a detailed budget**\n- Tracked every expense for a month\n- Used the envelope method for discretionary spending\n\n**2. Increased income**\n- Took on freelance work evenings/weekends\n- Sold items I didn't need\n\n**3. Used the debt avalanche method**\n- Paid minimums on all debts\n- Put extra money toward highest interest rate debt first\n\n**4. Cut expenses ruthlessly**\n- Moved to a cheaper apartment\n- Cooked at home instead of eating out\n- Canceled subscriptions I didn't use\n\nIt wasn't easy, but seeing that debt disappear was incredible motivation. Happy to answer any questions!",
                created_at=datetime.now() - timedelta(days=1),
                upvotes=45,
                downvotes=0
            ),
            Post(
                post_id="post_8",
                thread_id="thread_3",
                user_id="user_4",
                content="This is so inspiring! I'm dealing with about $25k in credit card debt right now. How did you stay motivated when it felt overwhelming? And how much extra income were you able to generate with freelancing?",
                created_at=datetime.now() - timedelta(hours=20),
                upvotes=8,
                downvotes=0
            ),
            Post(
                post_id="post_9",
                thread_id="thread_3",
                user_id="user_3",
                content="@NewInvestor2024 The key was celebrating small wins! Every $1,000 paid off was a victory. I created a visual tracker that I updated weekly.\n\nFor freelancing, I was making an extra $800-1,200 per month doing graphic design work. It was tough working evenings, but knowing every dollar was going toward freedom kept me going.\n\nFor $25k in CC debt, focus on that first before investing. The guaranteed 'return' of paying off high-interest debt beats any investment. You've got this!",
                created_at=datetime.now() - timedelta(hours=18),
                upvotes=15,
                downvotes=0,
                parent_post_id="post_8"
            )
        ]
        
        for post in sample_posts:
            self.posts[post.post_id] = post
            # Add posts to their respective threads
            if post.thread_id in self.threads:
                if self.threads[post.thread_id].posts is None:
                    self.threads[post.thread_id].posts = []
                self.threads[post.thread_id].posts.append(post)
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.users.get(user_id)
    
    def get_thread(self, thread_id: str) -> Optional[Thread]:
        """Get thread by ID"""
        return self.threads.get(thread_id)
    
    def get_threads_by_category(self, category: ThreadCategory) -> List[Thread]:
        """Get threads filtered by category"""
        return [t for t in self.threads.values() if t.category == category]
    
    def get_popular_threads(self, limit: int = 10) -> List[Thread]:
        """Get most popular threads by view count"""
        sorted_threads = sorted(self.threads.values(), key=lambda x: x.view_count, reverse=True)
        return sorted_threads[:limit]
    
    def get_recent_threads(self, limit: int = 10) -> List[Thread]:
        """Get most recent threads"""
        sorted_threads = sorted(self.threads.values(), key=lambda x: x.last_activity, reverse=True)
        return sorted_threads[:limit]
    
    def search_threads(self, query: str) -> List[Thread]:
        """Search threads by title and content"""
        query_lower = query.lower()
        results = []
        
        for thread in self.threads.values():
            if (query_lower in thread.title.lower() or 
                any(query_lower in tag.lower() for tag in (thread.tags or []))):
                results.append(thread)
        
        return results
    
    def vote_post(self, user_id: str, post_id: str, vote_type: str) -> bool:
        """Vote on a post (upvote/downvote)"""
        if post_id not in self.posts:
            return False
        
        # Check if user already voted
        vote_key = f"{user_id}_{post_id}"
        previous_vote = self.user_votes.get(vote_key)
        
        post = self.posts[post_id]
        
        # Remove previous vote if exists
        if previous_vote == "upvote":
            post.upvotes -= 1
        elif previous_vote == "downvote":
            post.downvotes -= 1
        
        # Add new vote
        if vote_type == "upvote" and previous_vote != "upvote":
            post.upvotes += 1
            self.user_votes[vote_key] = "upvote"
        elif vote_type == "downvote" and previous_vote != "downvote":
            post.downvotes += 1
            self.user_votes[vote_key] = "downvote"
        else:
            # Remove vote if clicking same vote type
            if vote_key in self.user_votes:
                del self.user_votes[vote_key]
        
        return True
    
    def create_thread(self, title: str, category: ThreadCategory, user_id: str, content: str, tags: List[str] = None) -> str:
        """Create a new thread"""
        thread_id = str(uuid.uuid4())
        post_id = str(uuid.uuid4())
        now = datetime.now()
        
        # Create initial post
        initial_post = Post(
            post_id=post_id,
            thread_id=thread_id,
            user_id=user_id,
            content=content,
            created_at=now
        )
        
        # Create thread
        thread = Thread(
            thread_id=thread_id,
            title=title,
            category=category,
            user_id=user_id,
            created_at=now,
            last_activity=now,
            tags=tags or [],
            posts=[initial_post]
        )
        
        self.threads[thread_id] = thread
        self.posts[post_id] = initial_post
        
        # Update user stats
        if user_id in self.users:
            self.users[user_id].posts_count += 1
        
        return thread_id
    
    def reply_to_thread(self, thread_id: str, user_id: str, content: str, parent_post_id: str = None) -> str:
        """Add a reply to a thread"""
        if thread_id not in self.threads:
            return None
        
        post_id = str(uuid.uuid4())
        now = datetime.now()
        
        # Create reply post
        reply_post = Post(
            post_id=post_id,
            thread_id=thread_id,
            user_id=user_id,
            content=content,
            created_at=now,
            parent_post_id=parent_post_id
        )
        
        self.posts[post_id] = reply_post
        self.threads[thread_id].posts.append(reply_post)
        self.threads[thread_id].last_activity = now
        
        # Update user stats
        if user_id in self.users:
            self.users[user_id].posts_count += 1
        
        return post_id

class CommunityDisplay:
    """Handle community forum UI and interactions"""
    
    def __init__(self):
        self.community_manager = CommunityManager()
        
        # Initialize current user (demo)
        if "current_user_id" not in st.session_state:
            st.session_state.current_user_id = "user_4"  # Default to NewInvestor2024
    
    def render_community_header(self):
        """Render the community forum header"""
        st.markdown("""
        <div style='background: linear-gradient(90deg, #ff6b6b 0%, #ffa500 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem;'>
            <h1 style='color: white; text-align: center; margin: 0;'>
                💬 Pecunia Community Forum
            </h1>
            <p style='color: white; text-align: center; margin: 0.5rem 0 0 0; opacity: 0.9;'>
                Connect, learn, and share financial knowledge with fellow investors
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_forum_navigation(self):
        """Render forum navigation and stats"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_threads = len(self.community_manager.threads)
            st.metric("Total Discussions", total_threads)
        
        with col2:
            total_posts = len(self.community_manager.posts)
            st.metric("Total Posts", total_posts)
        
        with col3:
            active_users = len(self.community_manager.users)
            st.metric("Active Members", active_users)
        
        with col4:
            if st.button("🆕 New Discussion", type="primary"):
                st.session_state.show_new_thread_form = True
                st.rerun()
    
    def render_category_filter(self):
        """Render category filtering options"""
        st.markdown("### 📂 Browse by Category")
        
        # Category buttons
        categories = list(ThreadCategory)
        cols = st.columns(5)
        
        for i, category in enumerate(categories):
            with cols[i % 5]:
                if st.button(f"{category.value}", key=f"cat_{category.name}"):
                    st.session_state.selected_category = category
                    st.rerun()
        
        # Search
        search_query = st.text_input("🔍 Search discussions", placeholder="Search for topics, keywords...")
        if search_query:
            st.session_state.search_query = search_query
            st.rerun()
    
    def render_thread_list(self, threads: List[Thread], title: str):
        """Render a list of threads"""
        st.markdown(f"### {title}")
        
        if not threads:
            st.info("No discussions found. Be the first to start a conversation!")
            return
        
        for thread in threads:
            self.render_thread_card(thread)
    
    def render_thread_card(self, thread: Thread):
        """Render an individual thread card"""
        user = self.community_manager.get_user(thread.user_id)
        username = user.username if user else "Unknown User"
        
        # Calculate time ago
        time_diff = datetime.now() - thread.last_activity
        if time_diff.days > 0:
            time_ago = f"{time_diff.days}d ago"
        elif time_diff.seconds > 3600:
            time_ago = f"{time_diff.seconds // 3600}h ago"
        else:
            time_ago = f"{time_diff.seconds // 60}m ago"
        
        # Status indicators
        status_icons = []
        if thread.is_pinned:
            status_icons.append("📌")
        if thread.is_solved:
            status_icons.append("✅")
        if thread.is_locked:
            status_icons.append("🔒")
        
        status_str = " ".join(status_icons)
        
        # Thread card
        with st.container():
            col1, col2, col3 = st.columns([6, 2, 1])
            
            with col1:
                # Title with status icons
                title_display = f"{status_str} {thread.title}" if status_icons else thread.title
                if st.button(title_display, key=f"thread_{thread.thread_id}", help="Click to view discussion"):
                    st.session_state.current_thread_id = thread.thread_id
                    st.rerun()
                
                # Thread info
                st.markdown(f"""
                <div style='font-size: 0.85rem; color: #666; margin-top: -0.5rem;'>
                    👤 <strong>{username}</strong> • 
                    🏷️ {thread.category.value} • 
                    🕐 {time_ago} • 
                    👁️ {thread.view_count} views
                </div>
                """, unsafe_allow_html=True)
                
                # Tags
                if thread.tags:
                    tag_html = " ".join([f"<span style='background: #e7f3ff; color: #0066cc; padding: 0.2rem 0.4rem; border-radius: 10px; font-size: 0.75rem; margin-right: 0.25rem;'>{tag}</span>" for tag in thread.tags[:3]])
                    st.markdown(tag_html, unsafe_allow_html=True)
            
            with col2:
                post_count = len(thread.posts) if thread.posts else 0
                st.markdown(f"""
                <div style='text-align: center; font-size: 0.9rem; color: #666;'>
                    💬 {post_count} replies
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                # User avatar/reputation
                if user:
                    role_colors = {
                        UserRole.MEMBER: "#6c757d",
                        UserRole.CONTRIBUTOR: "#28a745",
                        UserRole.EXPERT: "#007bff",
                        UserRole.MODERATOR: "#ffc107",
                        UserRole.ADMIN: "#dc3545"
                    }
                    role_color = role_colors.get(user.role, "#6c757d")
                    
                    st.markdown(f"""
                    <div style='text-align: center; font-size: 0.8rem;'>
                        <div style='color: {role_color}; font-weight: bold;'>{user.role.value}</div>
                        <div style='color: #666;'>⭐ {user.reputation_score}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
    
    def render_thread_detail(self, thread_id: str):
        """Render detailed thread view with all posts"""
        thread = self.community_manager.get_thread(thread_id)
        if not thread:
            st.error("Thread not found!")
            return
        
        # Update view count
        thread.view_count += 1
        
        # Back button
        if st.button("⬅️ Back to Forum"):
            if "current_thread_id" in st.session_state:
                del st.session_state.current_thread_id
            st.rerun()
        
        # Thread header
        user = self.community_manager.get_user(thread.user_id)
        username = user.username if user else "Unknown User"
        
        st.markdown(f"# {thread.title}")
        
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.markdown(f"**Category:** {thread.category.value}")
        with col2:
            st.markdown(f"**Started by:** {username}")
        with col3:
            st.markdown(f"**Views:** {thread.view_count}")
        
        if thread.tags:
            tag_html = " ".join([f"<span style='background: #e7f3ff; color: #0066cc; padding: 0.3rem 0.6rem; border-radius: 15px; font-size: 0.8rem; margin-right: 0.5rem;'>{tag}</span>" for tag in thread.tags])
            st.markdown(f"**Tags:** {tag_html}", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Render posts
        if thread.posts:
            for i, post in enumerate(thread.posts):
                self.render_post_detail(post, is_original=i==0)
        
        # Reply form (if not locked)
        if not thread.is_locked:
            st.markdown("### 💬 Reply to this discussion")
            
            with st.form(f"reply_form_{thread_id}"):
                reply_content = st.text_area(
                    "Your reply",
                    placeholder="Share your thoughts, ask questions, or provide helpful information...",
                    height=150
                )
                
                col1, col2 = st.columns([1, 4])
                with col1:
                    submit_reply = st.form_submit_button("Post Reply", type="primary")
                
                if submit_reply and reply_content.strip():
                    # Add reply
                    post_id = self.community_manager.reply_to_thread(
                        thread_id,
                        st.session_state.current_user_id,
                        reply_content.strip()
                    )
                    
                    if post_id:
                        st.success("Reply posted successfully!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Error posting reply. Please try again.")
        else:
            st.info("🔒 This discussion is locked and no longer accepting replies.")
    
    def render_post_detail(self, post: Post, is_original: bool = False):
        """Render a detailed post view"""
        user = self.community_manager.get_user(post.user_id)
        username = user.username if user else "Unknown User"
        
        # Time formatting
        time_diff = datetime.now() - post.created_at
        if time_diff.days > 0:
            time_ago = f"{time_diff.days}d ago"
        elif time_diff.seconds > 3600:
            time_ago = f"{time_diff.seconds // 3600}h ago"
        else:
            time_ago = f"{time_diff.seconds // 60}m ago"
        
        # Post container
        border_color = "#28a745" if is_original else "#e0e0e0"
        bg_color = "#f8f9fa" if is_original else "white"
        
        with st.container():
            st.markdown(f"""
            <div style='border: 2px solid {border_color}; border-radius: 10px; padding: 1.5rem; margin-bottom: 1.5rem; background: {bg_color};'>
            """, unsafe_allow_html=True)
            
            # Post header
            col1, col2, col3 = st.columns([3, 2, 1])
            
            with col1:
                role_colors = {
                    UserRole.MEMBER: "#6c757d",
                    UserRole.CONTRIBUTOR: "#28a745", 
                    UserRole.EXPERT: "#007bff",
                    UserRole.MODERATOR: "#ffc107",
                    UserRole.ADMIN: "#dc3545"
                }
                role_color = role_colors.get(user.role, "#6c757d") if user else "#6c757d"
                
                st.markdown(f"""
                <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;'>
                    <strong style='color: {role_color};'>{username}</strong>
                    <span style='background: {role_color}; color: white; padding: 0.2rem 0.4rem; border-radius: 10px; font-size: 0.7rem;'>
                        {user.role.value if user else 'Member'}
                    </span>
                    {'<span style="background: #28a745; color: white; padding: 0.2rem 0.4rem; border-radius: 10px; font-size: 0.7rem;">✓ Solution</span>' if post.is_solution else ''}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if user:
                    st.markdown(f"""
                    <div style='font-size: 0.8rem; color: #666;'>
                        ⭐ {user.reputation_score} reputation<br>
                        💬 {user.posts_count} posts
                    </div>
                    """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div style='font-size: 0.8rem; color: #666; text-align: right;'>
                    🕐 {time_ago}
                </div>
                """, unsafe_allow_html=True)
            
            # Post content
            st.markdown(post.content)
            
            # Post actions
            col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 2])
            
            with col1:
                if st.button(f"👍 {post.upvotes}", key=f"upvote_{post.post_id}"):
                    self.community_manager.vote_post(st.session_state.current_user_id, post.post_id, "upvote")
                    st.rerun()
            
            with col2:
                if st.button(f"👎 {post.downvotes}", key=f"downvote_{post.post_id}"):
                    self.community_manager.vote_post(st.session_state.current_user_id, post.post_id, "downvote")
                    st.rerun()
            
            with col3:
                if st.button("💬 Reply", key=f"reply_{post.post_id}"):
                    st.session_state[f"show_reply_{post.post_id}"] = True
                    st.rerun()
            
            with col4:
                if not post.is_solution and post.user_id != st.session_state.current_user_id:
                    if st.button("✅ Solution", key=f"solution_{post.post_id}"):
                        post.is_solution = True
                        # Update user reputation
                        if user:
                            user.helpful_answers += 1
                            user.reputation_score += 10
                        st.success("Marked as solution!")
                        st.rerun()
            
            # Inline reply form
            if st.session_state.get(f"show_reply_{post.post_id}", False):
                with st.form(f"inline_reply_{post.post_id}"):
                    reply_text = st.text_area(f"Reply to {username}", key=f"reply_text_{post.post_id}")
                    
                    col_submit, col_cancel = st.columns([1, 1])
                    with col_submit:
                        submit_inline = st.form_submit_button("Post Reply")
                    with col_cancel:
                        cancel_inline = st.form_submit_button("Cancel")
                    
                    if submit_inline and reply_text.strip():
                        # Add inline reply
                        thread = self.community_manager.get_thread(post.thread_id)
                        if thread:
                            self.community_manager.reply_to_thread(
                                post.thread_id,
                                st.session_state.current_user_id,
                                f"@{username} {reply_text.strip()}",
                                post.post_id
                            )
                            st.session_state[f"show_reply_{post.post_id}"] = False
                            st.success("Reply posted!")
                            time.sleep(1)
                            st.rerun()
                    
                    if cancel_inline:
                        st.session_state[f"show_reply_{post.post_id}"] = False
                        st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    def render_new_thread_form(self):
        """Render form to create new thread"""
        st.markdown("## 🆕 Start a New Discussion")
        
        with st.form("new_thread_form"):
            title = st.text_input("Discussion Title", placeholder="What would you like to discuss?")
            
            category = st.selectbox(
                "Category",
                options=list(ThreadCategory),
                format_func=lambda x: x.value
            )
            
            content = st.text_area(
                "Your message",
                placeholder="Provide details, context, or ask your question here...",
                height=200
            )
            
            tags_input = st.text_input(
                "Tags (comma-separated)",
                placeholder="investing, beginner, stocks, etc."
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                submit_thread = st.form_submit_button("Create Discussion", type="primary")
            with col2:
                cancel_thread = st.form_submit_button("Cancel")
            
            if submit_thread and title.strip() and content.strip():
                # Parse tags
                tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()] if tags_input else []
                
                # Create thread
                thread_id = self.community_manager.create_thread(
                    title.strip(),
                    category,
                    st.session_state.current_user_id,
                    content.strip(),
                    tags
                )
                
                if thread_id:
                    st.success("Discussion created successfully!")
                    st.session_state.show_new_thread_form = False
                    st.session_state.current_thread_id = thread_id
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Error creating discussion. Please try again.")
            
            if cancel_thread:
                st.session_state.show_new_thread_form = False
                st.rerun()
    
    def render_user_profile(self, user_id: str):
        """Render user profile sidebar"""
        user = self.community_manager.get_user(user_id)
        if not user:
            return
        
        with st.sidebar:
            st.markdown("### 👤 Your Profile")
            
            st.markdown(f"**{user.username}**")
            
            role_colors = {
                UserRole.MEMBER: "#6c757d",
                UserRole.CONTRIBUTOR: "#28a745",
                UserRole.EXPERT: "#007bff", 
                UserRole.MODERATOR: "#ffc107",
                UserRole.ADMIN: "#dc3545"
            }
            role_color = role_colors.get(user.role, "#6c757d")
            
            st.markdown(f"""
            <div style='margin-bottom: 1rem;'>
                <span style='background: {role_color}; color: white; padding: 0.3rem 0.6rem; border-radius: 15px; font-size: 0.8rem;'>
                    {user.role.value}
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            # Stats
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Reputation", user.reputation_score)
            with col2:
                st.metric("Posts", user.posts_count)
            
            st.metric("Helpful Answers", user.helpful_answers)
            
            # Bio
            if user.bio:
                st.markdown("**About:**")
                st.markdown(user.bio)
            
            # Badges
            if user.badges:
                st.markdown("**Badges:**")
                for badge in user.badges:
                    st.markdown(f"🏆 {badge}")
            
            # Member since
            join_days = (datetime.now() - user.join_date).days
            st.markdown(f"**Member since:** {join_days} days ago")

def display_forum() -> None:
    """
    Main function to display the community forum
    Shows community threads and posts
    """
    st.set_page_config(
        page_title="Pecunia AI - Community Forum",
        page_icon="💬",
        layout="wide"
    )
    
    display = CommunityDisplay()
    
    # Render user profile in sidebar
    display.render_user_profile(st.session_state.current_user_id)
    
    # Check if we're in thread detail view
    if hasattr(st.session_state, 'current_thread_id') and st.session_state.current_thread_id:
        display.render_thread_detail(st.session_state.current_thread_id)
        return
    
    # Check if we're showing new thread form
    if st.session_state.get('show_new_thread_form', False):
        display.render_community_header()
        display.render_new_thread_form()
        return
    
    # Main forum view
    display.render_community_header()
    display.render_forum_navigation()
    
    st.markdown("---")
    
    # Category filter and search
    display.render_category_filter()
    
    st.markdown("---")
    
    # Determine which threads to show
    if hasattr(st.session_state, 'selected_category') and st.session_state.selected_category:
        threads = display.community_manager.get_threads_by_category(st.session_state.selected_category)
        title = f"📂 {st.session_state.selected_category.value} Discussions"
        display.render_thread_list(threads, title)
        
        # Clear category filter button
        if st.button("🔄 Show All Discussions"):
            del st.session_state.selected_category
            st.rerun()
    
    elif hasattr(st.session_state, 'search_query') and st.session_state.search_query:
        threads = display.community_manager.search_threads(st.session_state.search_query)
        title = f"🔍 Search Results for '{st.session_state.search_query}'"
        display.render_thread_list(threads, title)
        
        # Clear search button
        if st.button("🔄 Clear Search"):
            del st.session_state.search_query
            st.rerun()
    
    else:
        # Default view - show recent and popular threads
        col1, col2 = st.columns(2)
        
        with col1:
            recent_threads = display.community_manager.get_recent_threads(5)
            display.render_thread_list(recent_threads, "🕐 Recent Discussions")
        
        with col2:
            popular_threads = display.community_manager.get_popular_threads(5)
            display.render_thread_list(popular_threads, "🔥 Popular Discussions")

# Main execution
if __name__ == "__main__":
    display_forum() 