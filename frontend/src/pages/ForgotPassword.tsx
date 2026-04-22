import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { forgotPassword } from '../services/auth'

const ForgotPassword: React.FC = () => {
  const [email, setEmail] = useState('')
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [touched, setTouched] = useState(false)
  const navigate = useNavigate()

  // Form validation
  const validate = () => {
    let isValid = true
    let errorMessage = ''

    if (!email.trim()) {
      isValid = false
      errorMessage = 'Please enter your email address'
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      isValid = false
      errorMessage = 'Please enter a valid email address'
    }

    return { isValid, errorMessage }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setTouched(true)
    
    const validation = validate()
    if (!validation.isValid) {
      setError(validation.errorMessage)
      setSuccess('')
      return
    }
    
    setError('')
    setIsLoading(true)
    
    try {
      const result = await forgotPassword(email)
      setSuccess(result.message || 'Password reset link sent successfully! Please check your email.')
      // Clear form after successful submission
      setEmail('')
      setTouched(false)
    } catch (err: any) {
      setError(err.message || 'Failed to send password reset link. Please try again.')
      setSuccess('')
    } finally {
      setIsLoading(false)
    }
  }

  const handleLogin = () => {
    navigate('/login')
  }

  return (
    <div className="login-container">
      <div className="login-header">
        <h1>QLib AI</h1>
        <p className="login-subtitle">Forgot Password</p>
      </div>
      <form className="login-form" onSubmit={handleSubmit}>
        {success && <div className="success login-success">{success}</div>}
        {error && <div className="error login-error">{error}</div>}
        
        <div className="form-group">
          <label htmlFor="email" className="form-label">Email Address</label>
          <input
            type="email"
            id="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            onBlur={() => setTouched(true)}
            placeholder="Enter your email address"
            className={`form-input ${touched && (!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) ? 'input-error' : ''}`}
            required
            disabled={isLoading}
          />
          {touched && !email && (
            <div className="error">Email is required</div>
          )}
          {touched && email && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email) && (
            <div className="error">Please enter a valid email address</div>
          )}
        </div>
        
        <button 
          type="submit" 
          className="btn btn-primary login-btn"
          disabled={isLoading}
        >
          {isLoading ? (
            <span className="loading">Sending Reset Link</span>
          ) : (
            'Send Reset Link'
          )}
        </button>
        
        <div className="login-footer">
          <p className="login-help">
            <a href="/login" onClick={handleLogin}>Back to Login</a>
          </p>
        </div>
      </form>
    </div>
  )
}

export default ForgotPassword
