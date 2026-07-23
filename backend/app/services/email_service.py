import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import secrets
from sqlalchemy.orm import Session
from app.models.user import User
from app.db.database import settings


def generate_verification_token():
    """Generate a random verification token"""
    return secrets.token_urlsafe(32)


def send_email(to_email: str, subject: str, body: str):
    """Send an email using the configured SMTP settings"""
    try:
        # Create message container
        msg = MIMEMultipart()
        msg['From'] = settings.sender_email
        msg['To'] = to_email
        msg['Subject'] = subject

        # Add HTML body
        msg.attach(MIMEText(body, 'html'))

        # Create SMTP session
        if settings.smtp_use_tls:
            server = smtplib.SMTP_SSL(settings.smtp_server, settings.smtp_port)
        else:
            server = smtplib.SMTP(settings.smtp_server, settings.smtp_port)
            server.starttls()

        # Login to SMTP server
        server.login(settings.smtp_username, settings.smtp_password)

        # Send email
        text = msg.as_string()
        server.sendmail(settings.sender_email, to_email, text)

        # Close connection
        server.quit()

        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False


def send_verification_email(user: User, base_url: str = "http://localhost:3000"):
    """Send a verification email to the user"""
    # Generate verification token
    token = generate_verification_token()
    
    # Set token expiry (24 hours)
    expiry = datetime.utcnow() + timedelta(minutes=settings.verification_token_expire_minutes)
    
    # Update user with token and expiry
    user.verification_token = token
    user.verification_token_expiry = expiry
    
    # Create verification link
    verify_url = f"{base_url}/verify-email?token={token}"
    
    # Email subject and body
    subject = "Verify Your Email Address for QLib AI"
    body = f"""<html>
<body style="font-family: Arial, sans-serif; line-height: 1.6;">
    <h2>Welcome to QLib AI!</h2>
    <p>Thank you for registering with QLib AI. Please click the button below to verify your email address:</p>
    <p style="margin: 20px 0;">
        <a href="{verify_url}" style="display: inline-block; padding: 12px 24px; background-color: #4CAF50; color: white; text-decoration: none; border-radius: 4px; font-weight: bold;">
            Verify Email Address
        </a>
    </p>
    <p>If you didn't create an account with QLib AI, you can safely ignore this email.</p>
    <p>Best regards,<br>QLib AI Team</p>
</body>
</html>"""
    
    # Send email
    return send_email(user.email, subject, body)


def verify_email_token(db: Session, token: str):
    """Verify an email verification token"""
    # Find user with this token
    user = db.query(User).filter(User.verification_token == token).first()
    
    if not user:
        return None, "Invalid verification token"
    
    # Check if token is expired
    if user.verification_token_expiry and user.verification_token_expiry < datetime.utcnow():
        return None, "Verification token has expired"
    
    # Mark email as verified and clear token
    user.email_verified = True
    user.verification_token = None
    user.verification_token_expiry = None
    
    return user, "Email verified successfully"


def resend_verification_email(db: Session, email: str, base_url: str = "http://localhost:3000"):
    """Resend verification email to user"""
    # Find user by email
    user = db.query(User).filter(User.email == email).first()
    
    if not user:
        return False, "User not found"
    
    if user.email_verified:
        return False, "Email already verified"
    
    # Send new verification email
    if send_verification_email(user, base_url):
        return True, "Verification email resent successfully"
    else:
        return False, "Failed to send verification email"


def send_password_reset_email(user: User, base_url: str = "http://localhost:3000"):
    """Send a password reset email to the user"""
    # Generate password reset token
    token = generate_verification_token()
    
    # Set token expiry (1 hour)
    expiry = datetime.utcnow() + timedelta(minutes=60)
    
    # Update user with password reset token and expiry
    user.password_reset_token = token
    user.password_reset_expiry = expiry
    
    # Create password reset link
    reset_url = f"{base_url}/reset-password?token={token}"
    
    # Email subject and body
    subject = "Reset Your Password for QLib AI"
    body = f"""<html>
<body style="font-family: Arial, sans-serif; line-height: 1.6;">
    <h2>Password Reset Request</h2>
    <p>We received a request to reset your password for your QLib AI account.</p>
    <p style="margin: 20px 0;">
        <a href="{reset_url}" style="display: inline-block; padding: 12px 24px; background-color: #4CAF50; color: white; text-decoration: none; border-radius: 4px; font-weight: bold;">
            Reset Password
        </a>
    </p>
    <p>This link will expire in 1 hour. If you didn't request a password reset, you can safely ignore this email.</p>
    <p>Best regards,<br>QLib AI Team</p>
</body>
</html>"""
    
    return send_email(user.email, subject, body)


def verify_password_reset_token(db: Session, token: str):
    """Verify a password reset token"""
    # Find user with this token
    user = db.query(User).filter(User.password_reset_token == token).first()
    
    if not user:
        return None, "Invalid password reset token"
    
    # Check if token is expired
    if user.password_reset_expiry and user.password_reset_expiry < datetime.utcnow():
        return None, "Password reset token has expired"
    
    return user, "Password reset token is valid"


def reset_user_password(user: User, new_password: str):
    """Reset user password"""
    from app.services.auth import get_password_hash
    
    # Update password and clear reset token
    user.password_hash = get_password_hash(new_password)
    user.password_reset_token = None
    user.password_reset_expiry = None
    
    return user, "Password reset successfully"
