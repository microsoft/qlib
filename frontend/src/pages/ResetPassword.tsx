import React, { useState, useEffect } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { resetPassword } from '../services/auth'

const ResetPassword: React.FC = () => {
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [touched, setTouched] = useState({
    password: false,
    confirmPassword: false
  })
  const [token, setToken] = useState<string | null>(null)
  
  const navigate = useNavigate()
  const location = useLocation()

  // Extract token from URL query parameters
  useEffect(() => {
    const params = new URLSearchParams(location.search)
    const resetToken = params.get('token')
    
    if (resetToken) {
      setToken(resetToken)
    } else {
      setError('Invalid or missing password reset token')
    }
  }, [location.search])

  // Form validation
  const validate = () => {
    let isValid = true
    let errorMessage = ''

    if (!password.trim()) {
      isValid = false
      errorMessage = 'Please enter your new password'
    } else if (password.length < 6) {
      isValid = false
      errorMessage = 'Password must be at least 6 characters long'
    } else if (password !== confirmPassword) {
      isValid = false
      errorMessage = 'Passwords do not match'
    }

    return { isValid, errorMessage }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setTouched({
      password: true,
      confirmPassword: true
    })
    
    if (!token) {
      setError('Invalid or missing password reset token')
      return
    }
    
    const validation = validate()
    if (!validation.isValid) {
      setError(validation.errorMessage)
      setSuccess('')
      return
    }
    
    setError('')
    setIsLoading(true)
    
    try {
      const result = await resetPassword(token, password)
      setSuccess(result.message || 'Password reset successfully! You can now login with your new password.')
      // Clear form after successful submission
      setPassword('')
      setConfirmPassword('')
      setTouched({
        password: false,
        confirmPassword: false
      })
      
      // Redirect to login page after 3 seconds
      setTimeout(() => {
        navigate('/login')
      }, 3000)
    } catch (err: any) {
      setError(err.message || 'Failed to reset password. Please try again.')
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
        <p className="login-subtitle">Reset Your Password</p>
      </div>
      <form className="login-form" onSubmit={handleSubmit}>
        {success && <div className="success login-success">{success}</div>}
        {error && <div className="error login-error">{error}</div>}
        
        <div className="form-group">
          <label htmlFor="password" className="form-label">New Password</label>
          <input
            type="password"
            id="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            onBlur={() => setTouched(prev => ({ ...prev, password: true }))}
            placeholder="Enter your new password"
            className={`form-input ${touched.password && (!password || password.length < 6) ? 'input-error' : ''}`}
            required
            disabled={isLoading}
          />
          {touched.password && !password && (
            <div className="error">Password is required</div>
          )}
          {touched.password && password.length > 0 && password.length < 6 && (
            <div className="error">Password must be at least 6 characters</div>
          )}
        </div>
        
        <div className="form-group">
          <label htmlFor="confirmPassword" className="form-label">Confirm Password</label>
          <input
            type="password"
            id="confirmPassword"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            onBlur={() => setTouched(prev => ({ ...prev, confirmPassword: true }))}
            placeholder="Confirm your new password"
            className={`form-input ${touched.confirmPassword && (password !== confirmPassword) ? 'input-error' : ''}`}
            required
            disabled={isLoading}
          />
          {touched.confirmPassword && password !== confirmPassword && (
            <div className="error">Passwords do not match</div>
          )}
        </div>
        
        <button 
          type="submit" 
          className="btn btn-primary login-btn"
          disabled={isLoading || !token}
        >
          {isLoading ? (
            <span className="loading">Resetting Password</span>
          ) : (
            'Reset Password'
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

export default ResetPassword
