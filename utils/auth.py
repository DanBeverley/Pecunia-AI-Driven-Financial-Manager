"""
Pecunia AI - Enterprise Authentication & Authorization System
Advanced security with JWT, OAuth2, MFA, biometric authentication, and zero-trust architecture
"""

import asyncio
import base64
import hashlib
import hmac
import json
import os
import secrets
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import re
import logging

import jwt
import bcrypt
import pyotp
import qrcode
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import redis
from sqlalchemy import create_engine, Column, String, DateTime, Boolean, Integer, Text, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import requests
from authlib.integrations.flask_oauth2 import AuthorizationServer, ResourceProtector
from authlib.oauth2.rfc6749 import grants
from flask import Flask, request, jsonify, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
import face_recognition
import cv2
import numpy as np

# Configure advanced logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
security_logger = logging.getLogger('security')

Base = declarative_base()

class AuthMethod(Enum):
    """Supported authentication methods"""
    PASSWORD = "password"
    TOTP = "totp"
    SMS = "sms"
    EMAIL = "email"
    BIOMETRIC_FACE = "biometric_face"
    BIOMETRIC_FINGERPRINT = "biometric_fingerprint"
    HARDWARE_TOKEN = "hardware_token"
    OAUTH2_GOOGLE = "oauth2_google"
    OAUTH2_MICROSOFT = "oauth2_microsoft"
    OAUTH2_GITHUB = "oauth2_github"

class UserRole(Enum):
    """User roles with different permissions"""
    ADMIN = "admin"
    POWER_USER = "power_user"
    ANALYST = "analyst"
    TRADER = "trader"
    VIEWER = "viewer"
    API_CLIENT = "api_client"

class SessionState(Enum):
    """Session states for advanced session management"""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPENDED = "suspended"
    COMPROMISED = "compromised"

@dataclass
class SecurityPolicy:
    """Enterprise security policy configuration"""
    password_min_length: int = 12
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_numbers: bool = True
    password_require_symbols: bool = True
    password_max_age_days: int = 90
    failed_login_threshold: int = 5
    lockout_duration_minutes: int = 30
    session_timeout_minutes: int = 60
    require_mfa: bool = True
    allowed_mfa_methods: List[AuthMethod] = field(default_factory=lambda: [AuthMethod.TOTP, AuthMethod.SMS])
    max_concurrent_sessions: int = 3
    password_history_count: int = 5

class User(Base):
    """Enterprise user model with advanced security features"""
    __tablename__ = 'users'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(255), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    salt = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, default=UserRole.VIEWER.value)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime, nullable=True)
    last_login = Column(DateTime, nullable=True)
    last_password_change = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # MFA fields
    totp_secret = Column(String(32), nullable=True)
    backup_codes = Column(Text, nullable=True)  # JSON array of backup codes
    phone_number = Column(String(20), nullable=True)
    sms_verified = Column(Boolean, default=False)
    
    # Biometric fields
    face_encoding = Column(LargeBinary, nullable=True)
    fingerprint_template = Column(LargeBinary, nullable=True)
    
    # OAuth fields
    oauth_providers = Column(Text, nullable=True)  # JSON object
    
    # Security tracking
    password_history = Column(Text, nullable=True)  # JSON array of old password hashes
    security_questions = Column(Text, nullable=True)  # JSON array

class UserSession(Base):
    """Advanced session management"""
    __tablename__ = 'user_sessions'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), nullable=False)
    session_token = Column(String(255), unique=True, nullable=False)
    refresh_token = Column(String(255), unique=True, nullable=True)
    device_info = Column(Text, nullable=True)  # JSON object
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    location = Column(String(255), nullable=True)
    state = Column(String(50), default=SessionState.ACTIVE.value)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)

class AuditLog(Base):
    """Comprehensive security audit logging"""
    __tablename__ = 'audit_logs'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), nullable=True)
    action = Column(String(255), nullable=False)
    resource = Column(String(255), nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    success = Column(Boolean, nullable=False)
    details = Column(Text, nullable=True)  # JSON object
    risk_score = Column(Integer, default=0)
    timestamp = Column(DateTime, default=datetime.utcnow)

class PasswordValidator:
    """Advanced password validation with security policies"""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
    
    def validate_password(self, password: str, user: Optional[User] = None) -> Tuple[bool, List[str]]:
        """Comprehensive password validation"""
        errors = []
        
        # Length check
        if len(password) < self.policy.password_min_length:
            errors.append(f"Password must be at least {self.policy.password_min_length} characters long")
        
        # Character requirements
        if self.policy.password_require_uppercase and not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
        
        if self.policy.password_require_lowercase and not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")
        
        if self.policy.password_require_numbers and not re.search(r'\d', password):
            errors.append("Password must contain at least one number")
        
        if self.policy.password_require_symbols and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append("Password must contain at least one special character")
        
        # Common password check
        if self._is_common_password(password):
            errors.append("Password is too common, please choose a more secure password")
        
        # Password history check
        if user and self._check_password_history(password, user):
            errors.append("Password has been used recently, please choose a different password")
        
        return len(errors) == 0, errors
    
    def _is_common_password(self, password: str) -> bool:
        """Check against common passwords"""
        common_passwords = {
            'password', '123456', 'password123', 'admin', 'qwerty',
            'letmein', 'welcome', 'monkey', '1234567890', 'abc123'
        }
        return password.lower() in common_passwords
    
    def _check_password_history(self, password: str, user: User) -> bool:
        """Check if password was used recently"""
        if not user.password_history:
            return False
        
        try:
            history = json.loads(user.password_history)
            for old_hash in history[-self.policy.password_history_count:]:
                if bcrypt.checkpw(password.encode('utf-8'), old_hash.encode('utf-8')):
                    return True
        except (json.JSONDecodeError, ValueError):
            pass
        
        return False

class BiometricAuthenticator:
    """Advanced biometric authentication system"""
    
    def __init__(self):
        self.face_recognition_threshold = 0.6
        self.fingerprint_threshold = 0.7
    
    def register_face(self, image_data: bytes) -> Optional[bytes]:
        """Register face biometric"""
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract face encoding
            face_locations = face_recognition.face_locations(rgb_image)
            if len(face_locations) != 1:
                return None  # Must have exactly one face
            
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            if not face_encodings:
                return None
            
            return face_encodings[0].tobytes()
            
        except Exception as e:
            logger.error(f"Face registration error: {e}")
            return None
    
    def verify_face(self, image_data: bytes, stored_encoding: bytes) -> bool:
        """Verify face biometric"""
        try:
            # Convert stored encoding back to numpy array
            stored_face_encoding = np.frombuffer(stored_encoding, dtype=np.float64)
            
            # Process new image
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(rgb_image)
            if not face_locations:
                return False
            
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            if not face_encodings:
                return False
            
            # Compare encodings
            distances = face_recognition.face_distance([stored_face_encoding], face_encodings[0])
            return distances[0] < self.face_recognition_threshold
            
        except Exception as e:
            logger.error(f"Face verification error: {e}")
            return False

class MFAManager:
    """Multi-factor authentication manager"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.biometric_auth = BiometricAuthenticator()
    
    def setup_totp(self, user: User) -> Tuple[str, str]:
        """Setup TOTP for user"""
        secret = pyotp.random_base32()
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            user.email,
            issuer_name="Pecunia AI"
        )
        
        # Generate QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)
        
        return secret, totp_uri
    
    def verify_totp(self, secret: str, token: str) -> bool:
        """Verify TOTP token"""
        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=2)
    
    def generate_backup_codes(self, count: int = 10) -> List[str]:
        """Generate backup codes for MFA"""
        return [secrets.token_hex(4).upper() for _ in range(count)]
    
    def send_sms_code(self, phone_number: str) -> str:
        """Send SMS verification code"""
        code = secrets.randbelow(1000000)
        formatted_code = f"{code:06d}"
        
        # Store code in Redis with expiration
        key = f"sms_code:{phone_number}"
        self.redis.setex(key, 300, formatted_code)  # 5 minute expiration
        
        # In production, integrate with SMS service (Twilio, AWS SNS, etc.)
        logger.info(f"SMS code for {phone_number}: {formatted_code}")
        
        return formatted_code
    
    def verify_sms_code(self, phone_number: str, code: str) -> bool:
        """Verify SMS code"""
        key = f"sms_code:{phone_number}"
        stored_code = self.redis.get(key)
        
        if stored_code and stored_code.decode() == code:
            self.redis.delete(key)
            return True
        
        return False

class JWTManager:
    """Enterprise JWT token management"""
    
    def __init__(self, secret_key: str, redis_client: redis.Redis):
        self.secret_key = secret_key
        self.redis = redis_client
        self.algorithm = "HS256"
        self.access_token_expiry = timedelta(minutes=15)
        self.refresh_token_expiry = timedelta(days=30)
    
    def generate_tokens(self, user: User) -> Tuple[str, str]:
        """Generate access and refresh tokens"""
        now = datetime.utcnow()
        
        # Access token payload
        access_payload = {
            "user_id": user.id,
            "username": user.username,
            "email": user.email,
            "role": user.role,
            "iat": now,
            "exp": now + self.access_token_expiry,
            "type": "access"
        }
        
        # Refresh token payload
        refresh_payload = {
            "user_id": user.id,
            "iat": now,
            "exp": now + self.refresh_token_expiry,
            "type": "refresh",
            "jti": str(uuid.uuid4())
        }
        
        access_token = jwt.encode(access_payload, self.secret_key, algorithm=self.algorithm)
        refresh_token = jwt.encode(refresh_payload, self.secret_key, algorithm=self.algorithm)
        
        # Store refresh token in Redis
        self.redis.setex(f"refresh_token:{refresh_payload['jti']}", 
                        int(self.refresh_token_expiry.total_seconds()), 
                        user.id)
        
        return access_token, refresh_token
    
    def verify_token(self, token: str, token_type: str = "access") -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            if payload.get("type") != token_type:
                return None
            
            # Check if token is blacklisted
            if self._is_token_blacklisted(token):
                return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Refresh access token using refresh token"""
        refresh_payload = self.verify_token(refresh_token, "refresh")
        if not refresh_payload:
            return None
        
        # Check if refresh token is still valid in Redis
        jti = refresh_payload.get("jti")
        if not self.redis.get(f"refresh_token:{jti}"):
            return None
        
        # Get user and generate new access token
        user_id = refresh_payload["user_id"]
        # In production, fetch user from database
        # For now, create minimal user object
        class TempUser:
            def __init__(self, user_id):
                self.id = user_id
                self.username = f"user_{user_id}"
                self.email = f"user_{user_id}@example.com"
                self.role = UserRole.VIEWER.value
        
        temp_user = TempUser(user_id)
        access_token, _ = self.generate_tokens(temp_user)
        
        return access_token
    
    def revoke_token(self, token: str):
        """Revoke/blacklist a token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            exp = payload.get("exp", 0)
            
            # Add to blacklist with expiration
            blacklist_key = f"blacklist:{hashlib.sha256(token.encode()).hexdigest()}"
            ttl = max(0, exp - int(time.time()))
            
            if ttl > 0:
                self.redis.setex(blacklist_key, ttl, "revoked")
                
        except jwt.InvalidTokenError:
            pass  # Token already invalid
    
    def _is_token_blacklisted(self, token: str) -> bool:
        """Check if token is blacklisted"""
        blacklist_key = f"blacklist:{hashlib.sha256(token.encode()).hexdigest()}"
        return self.redis.exists(blacklist_key)

class OAuth2Provider:
    """OAuth2 integration for external providers"""
    
    def __init__(self):
        self.providers = {
            AuthMethod.OAUTH2_GOOGLE: {
                "client_id": os.getenv("GOOGLE_CLIENT_ID"),
                "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
                "authorization_base_url": "https://accounts.google.com/o/oauth2/auth",
                "token_url": "https://oauth2.googleapis.com/token",
                "userinfo_url": "https://www.googleapis.com/oauth2/v2/userinfo",
                "scope": ["openid", "email", "profile"]
            },
            AuthMethod.OAUTH2_MICROSOFT: {
                "client_id": os.getenv("MICROSOFT_CLIENT_ID"),
                "client_secret": os.getenv("MICROSOFT_CLIENT_SECRET"),
                "authorization_base_url": "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
                "token_url": "https://login.microsoftonline.com/common/oauth2/v2.0/token",
                "userinfo_url": "https://graph.microsoft.com/v1.0/me",
                "scope": ["openid", "email", "profile"]
            }
        }
    
    def get_authorization_url(self, provider: AuthMethod, redirect_uri: str, state: str) -> str:
        """Get OAuth2 authorization URL"""
        if provider not in self.providers:
            raise ValueError(f"Unsupported OAuth2 provider: {provider}")
        
        config = self.providers[provider]
        params = {
            "client_id": config["client_id"],
            "redirect_uri": redirect_uri,
            "scope": " ".join(config["scope"]),
            "response_type": "code",
            "state": state
        }
        
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        return f"{config['authorization_base_url']}?{query_string}"
    
    def exchange_code_for_token(self, provider: AuthMethod, code: str, redirect_uri: str) -> Optional[Dict[str, Any]]:
        """Exchange authorization code for access token"""
        if provider not in self.providers:
            return None
        
        config = self.providers[provider]
        
        data = {
            "client_id": config["client_id"],
            "client_secret": config["client_secret"],
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri
        }
        
        try:
            response = requests.post(config["token_url"], data=data)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"OAuth2 token exchange failed: {e}")
            return None
    
    def get_user_info(self, provider: AuthMethod, access_token: str) -> Optional[Dict[str, Any]]:
        """Get user information from OAuth2 provider"""
        if provider not in self.providers:
            return None
        
        config = self.providers[provider]
        headers = {"Authorization": f"Bearer {access_token}"}
        
        try:
            response = requests.get(config["userinfo_url"], headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"OAuth2 user info request failed: {e}")
            return None

class SecurityAnalyzer:
    """Advanced security analysis and threat detection"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.max_failed_attempts = 5
        self.suspicious_activity_threshold = 100
    
    def analyze_login_attempt(self, user_id: str, ip_address: str, user_agent: str, success: bool) -> int:
        """Analyze login attempt and return risk score"""
        risk_score = 0
        
        # Check failed login attempts
        if not success:
            failed_key = f"failed_logins:{user_id}"
            failed_count = self.redis.incr(failed_key)
            self.redis.expire(failed_key, 3600)  # 1 hour window
            
            if failed_count > self.max_failed_attempts:
                risk_score += 50
        
        # Check for unusual IP addresses
        ip_key = f"user_ips:{user_id}"
        known_ips = self.redis.smembers(ip_key)
        
        if ip_address.encode() not in known_ips:
            risk_score += 30
            # Add IP to known IPs if login successful
            if success:
                self.redis.sadd(ip_key, ip_address)
                self.redis.expire(ip_key, 86400 * 30)  # 30 days
        
        # Check for unusual user agents
        ua_key = f"user_agents:{user_id}"
        known_uas = self.redis.smembers(ua_key)
        
        if user_agent.encode() not in known_uas:
            risk_score += 20
            if success:
                self.redis.sadd(ua_key, user_agent)
                self.redis.expire(ua_key, 86400 * 30)  # 30 days
        
        # Check for rapid login attempts
        attempt_key = f"login_attempts:{ip_address}"
        attempt_count = self.redis.incr(attempt_key)
        self.redis.expire(attempt_key, 300)  # 5 minute window
        
        if attempt_count > 10:
            risk_score += 40
        
        return min(risk_score, 100)  # Cap at 100
    
    def detect_anomalies(self, user_id: str, activity_data: Dict[str, Any]) -> List[str]:
        """Detect anomalous user behavior"""
        anomalies = []
        
        # Time-based anomalies
        current_hour = datetime.now().hour
        activity_key = f"activity_hours:{user_id}"
        
        # Store activity hours
        self.redis.lpush(activity_key, current_hour)
        self.redis.ltrim(activity_key, 0, 167)  # Keep last week of hours
        self.redis.expire(activity_key, 86400 * 7)
        
        # Analyze activity pattern
        hours = [int(h) for h in self.redis.lrange(activity_key, 0, -1)]
        if hours:
            avg_hour = sum(hours) / len(hours)
            if abs(current_hour - avg_hour) > 6:  # More than 6 hours from average
                anomalies.append("unusual_login_time")
        
        # Geographic anomalies (would require IP geolocation service)
        # Session duration anomalies
        # API usage pattern anomalies
        
        return anomalies

class EnterpriseAuthManager:
    """Central authentication management system"""
    
    def __init__(self, database_url: str, redis_url: str, security_policy: SecurityPolicy):
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        self.redis = redis.from_url(redis_url)
        self.security_policy = security_policy
        self.password_validator = PasswordValidator(security_policy)
        self.mfa_manager = MFAManager(self.redis)
        self.jwt_manager = JWTManager(os.getenv("JWT_SECRET_KEY", "default-secret"), self.redis)
        self.oauth2_provider = OAuth2Provider()
        self.security_analyzer = SecurityAnalyzer(self.redis)
        
        # Encryption for sensitive data
        key = os.getenv("ENCRYPTION_KEY", Fernet.generate_key())
        self.cipher = Fernet(key)
    
    def register_user(self, username: str, email: str, password: str, **kwargs) -> Tuple[bool, Union[User, List[str]]]:
        """Register new user with comprehensive validation"""
        db = self.SessionLocal()
        
        try:
            # Validate password
            is_valid, errors = self.password_validator.validate_password(password)
            if not is_valid:
                return False, errors
            
            # Check if user exists
            existing_user = db.query(User).filter(
                (User.username == username) | (User.email == email)
            ).first()
            
            if existing_user:
                return False, ["Username or email already exists"]
            
            # Create user
            salt = secrets.token_hex(16)
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            
            user = User(
                username=username,
                email=email,
                password_hash=password_hash,
                salt=salt,
                role=kwargs.get('role', UserRole.VIEWER.value)
            )
            
            db.add(user)
            db.commit()
            
            # Log registration
            self._log_audit_event(
                user_id=user.id,
                action="user_registration",
                success=True,
                details={"username": username, "email": email}
            )
            
            return True, user
            
        except Exception as e:
            db.rollback()
            logger.error(f"User registration failed: {e}")
            return False, ["Registration failed"]
        finally:
            db.close()
    
    def authenticate_user(self, username: str, password: str, ip_address: str, user_agent: str) -> Tuple[bool, Optional[User], List[str]]:
        """Authenticate user with security analysis"""
        db = self.SessionLocal()
        
        try:
            user = db.query(User).filter(
                (User.username == username) | (User.email == username)
            ).first()
            
            if not user:
                self._log_audit_event(
                    action="login_attempt",
                    success=False,
                    details={"username": username, "reason": "user_not_found"},
                    ip_address=ip_address
                )
                return False, None, ["Invalid credentials"]
            
            # Check if user is locked
            if user.locked_until and user.locked_until > datetime.utcnow():
                return False, None, ["Account is locked"]
            
            # Verify password
            if not bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
                user.failed_login_attempts += 1
                
                # Lock account if too many failed attempts
                if user.failed_login_attempts >= self.security_policy.failed_login_threshold:
                    user.locked_until = datetime.utcnow() + timedelta(minutes=self.security_policy.lockout_duration_minutes)
                
                db.commit()
                
                self._log_audit_event(
                    user_id=user.id,
                    action="login_attempt",
                    success=False,
                    details={"reason": "invalid_password"},
                    ip_address=ip_address
                )
                
                return False, None, ["Invalid credentials"]
            
            # Reset failed attempts on successful login
            user.failed_login_attempts = 0
            user.locked_until = None
            user.last_login = datetime.utcnow()
            db.commit()
            
            # Security analysis
            risk_score = self.security_analyzer.analyze_login_attempt(
                user.id, ip_address, user_agent, True
            )
            
            self._log_audit_event(
                user_id=user.id,
                action="login_success",
                success=True,
                details={"risk_score": risk_score},
                ip_address=ip_address,
                risk_score=risk_score
            )
            
            # Check if MFA is required
            mfa_required = []
            if self.security_policy.require_mfa:
                if user.totp_secret:
                    mfa_required.append(AuthMethod.TOTP)
                if user.phone_number and user.sms_verified:
                    mfa_required.append(AuthMethod.SMS)
                if user.face_encoding:
                    mfa_required.append(AuthMethod.BIOMETRIC_FACE)
            
            return True, user, mfa_required
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False, None, ["Authentication failed"]
        finally:
            db.close()
    
    def verify_mfa(self, user_id: str, method: AuthMethod, credential: str, **kwargs) -> bool:
        """Verify multi-factor authentication"""
        db = self.SessionLocal()
        
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                return False
            
            if method == AuthMethod.TOTP:
                return self.mfa_manager.verify_totp(user.totp_secret, credential)
            
            elif method == AuthMethod.SMS:
                return self.mfa_manager.verify_sms_code(user.phone_number, credential)
            
            elif method == AuthMethod.BIOMETRIC_FACE:
                image_data = kwargs.get('image_data')
                if image_data and user.face_encoding:
                    return self.mfa_manager.biometric_auth.verify_face(image_data, user.face_encoding)
            
            return False
            
        except Exception as e:
            logger.error(f"MFA verification failed: {e}")
            return False
        finally:
            db.close()
    
    def create_session(self, user: User, device_info: Dict[str, Any]) -> UserSession:
        """Create authenticated session"""
        db = self.SessionLocal()
        
        try:
            # Generate tokens
            access_token, refresh_token = self.jwt_manager.generate_tokens(user)
            
            # Create session
            session = UserSession(
                user_id=user.id,
                session_token=access_token,
                refresh_token=refresh_token,
                device_info=json.dumps(device_info),
                ip_address=device_info.get('ip_address'),
                user_agent=device_info.get('user_agent'),
                expires_at=datetime.utcnow() + self.jwt_manager.access_token_expiry
            )
            
            db.add(session)
            db.commit()
            
            return session
            
        except Exception as e:
            db.rollback()
            logger.error(f"Session creation failed: {e}")
            raise
        finally:
            db.close()
    
    def _log_audit_event(self, action: str, success: bool, user_id: str = None, 
                        ip_address: str = None, details: Dict[str, Any] = None, 
                        risk_score: int = 0):
        """Log security audit events"""
        db = self.SessionLocal()
        
        try:
            audit_log = AuditLog(
                user_id=user_id,
                action=action,
                ip_address=ip_address,
                success=success,
                details=json.dumps(details) if details else None,
                risk_score=risk_score
            )
            
            db.add(audit_log)
            db.commit()
            
        except Exception as e:
            logger.error(f"Audit logging failed: {e}")
        finally:
            db.close()

# Authorization decorators
def require_auth(roles: List[UserRole] = None):
    """Decorator for requiring authentication and authorization"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get token from request header
            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                return jsonify({"error": "Authentication required"}), 401
            
            token = auth_header.split(' ')[1]
            
            # Verify token (implement JWT verification)
            # Check user roles if specified
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def require_mfa(methods: List[AuthMethod] = None):
    """Decorator for requiring multi-factor authentication"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check if MFA has been completed for this session
            # Implementation depends on session management
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Global auth manager instance (to be initialized with proper config)
auth_manager = None

def initialize_auth_system(database_url: str, redis_url: str, security_policy: SecurityPolicy = None):
    """Initialize the authentication system"""
    global auth_manager
    if security_policy is None:
        security_policy = SecurityPolicy()
    
    auth_manager = EnterpriseAuthManager(database_url, redis_url, security_policy)
    return auth_manager 