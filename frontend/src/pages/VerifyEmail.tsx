import React, { useState, useEffect } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { verifyEmail, resendVerification } from '../services/auth'

const VerifyEmail: React.FC = () => {
  const [isVerifying, setIsVerifying] = useState(true)
  const [verificationStatus, setVerificationStatus] = useState<'success' | 'error' | null>(null)
  const [message, setMessage] = useState('')
  const [email, setEmail] = useState('')
  const [isResending, setIsResending] = useState(false)
  const [resendMessage, setResendMessage] = useState('')
  const [resendError, setResendError] = useState('')
  
  const navigate = useNavigate()
  const location = useLocation()

  // Extract token from URL query parameters
  useEffect(() => {
    const params = new URLSearchParams(location.search)
    const token = params.get('token')
    
    if (token) {
      handleVerifyEmail(token)
    } else {
      setIsVerifying(false)
      setVerificationStatus('error')
      setMessage('No verification token found in URL')
    }
  }, [location.search])

  const handleVerifyEmail = async (token: string) => {
    try {
      const result = await verifyEmail(token)
      setVerificationStatus('success')
      setMessage(result.message || 'Email verified successfully!')
    } catch (err: any) {
      setVerificationStatus('error')
      setMessage(err.message || 'Failed to verify email')
    } finally {
      setIsVerifying(false)
    }
  }

  const handleResendVerification = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!email) {
      setResendError('Please enter your email address')
      return
    }
    
    setIsResending(true)
    setResendMessage('')
    setResendError('')
    
    try {
      const result = await resendVerification(email)
      setResendMessage(result.message || 'Verification email resent successfully')
    } catch (err: any) {
      setResendError(err.message || 'Failed to resend verification email')
    } finally {
      setIsResending(false)
    }
  }

  const handleLogin = () => {
    navigate('/login')
  }

  return (
    <div className="login-container">
      <div className="login-header">
        <h1>QLib AI</h1>
        <p className="login-subtitle">Email Verification</p>
      </div>
      
      {isVerifying ? (
        <div className="verification-loading">
          <p>Verifying your email...</p>
        </div>
      ) : (
        <div className="verification-result">
          <div className={`verification-status ${verificationStatus}`}>
            <h2>
              {verificationStatus === 'success' ? '✓ Email Verified Successfully' : '✗ Verification Failed'}
            </h2>
            <p>{message}</p>
          </div>
          
          {verificationStatus === 'success' && (
            <div className="verification-actions">
              <button 
                className="btn btn-primary"
                onClick={handleLogin}
              >
                Go to Login
              </button>
            </div>
          )}
          
          {verificationStatus === 'error' && (
            <div className="resend-verification">
              <h3>Resend Verification Email</h3>
              <form onSubmit={handleResendVerification}>
                {resendMessage && <div className="success">{resendMessage}</div>}
                {resendError && <div className="error">{resendError}</div>}
                
                <div className="form-group">
                  <label htmlFor="email" className="form-label">Your Email Address</label>
                  <input
                    type="email"
                    id="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="Enter your email"
                    className="form-input"
                    disabled={isResending}
                  />
                </div>
                
                <button 
                  type="submit" 
                  className="btn btn-primary"
                  disabled={isResending}
                >
                  {isResending ? (
                    <span className="loading">Sending...</span>
                  ) : (
                    'Resend Verification Email'
                  )}
                </button>
              </form>
            </div>
          )}
          
          <div className="login-footer">
            <p className="login-help">
              <a href="/login">Back to Login</a> | <a href="/register">Create New Account</a>
            </p>
          </div>
        </div>
      )}
    </div>
  )
}

export default VerifyEmail
