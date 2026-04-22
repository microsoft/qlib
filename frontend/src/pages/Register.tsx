import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { register } from '../services/auth'

const Register: React.FC = () => {
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [fullName, setFullName] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [touched, setTouched] = useState({
    username: false,
    email: false,
    fullName: false,
    password: false,
    confirmPassword: false
  })
  const navigate = useNavigate()

  // Form validation with detailed feedback
  const validateField = (field: string, value: string) => {
    switch (field) {
      case 'username':
        if (!value.trim()) return 'Username is required'
        if (value.length < 3) return 'Username must be at least 3 characters long'
        return ''
      case 'email':
        if (!value.trim()) return 'Email is required'
        if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value)) return 'Please enter a valid email address'
        return ''
      case 'password':
        if (!value.trim()) return 'Password is required'
        if (value.length < 6) return 'Password must be at least 6 characters long'
        if (!/[A-Za-z]/.test(value) || !/[0-9]/.test(value)) return 'Password must contain both letters and numbers'
        return ''
      case 'confirmPassword':
        if (!value.trim()) return 'Please confirm your password'
        if (value !== password) return 'Passwords do not match'
        return ''
      default:
        return ''
    }
  }

  // Get field-specific error
  const getFieldError = (field: keyof typeof touched) => {
    if (!touched[field]) return ''
    switch (field) {
      case 'username':
        return validateField('username', username)
      case 'email':
        return validateField('email', email)
      case 'password':
        return validateField('password', password)
      case 'confirmPassword':
        return validateField('confirmPassword', confirmPassword)
      default:
        return ''
    }
  }

  // Form validation for submission
  const validateForm = () => {
    const usernameError = validateField('username', username)
    const emailError = validateField('email', email)
    const passwordError = validateField('password', password)
    const confirmPasswordError = validateField('confirmPassword', confirmPassword)
    
    return !usernameError && !emailError && !passwordError && !confirmPasswordError
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    // Set all fields as touched to show validation errors
    setTouched({
      username: true,
      email: true,
      fullName: true,
      password: true,
      confirmPassword: true
    })
    
    if (!validateForm()) {
      // Don't set a general error, field-specific errors will be shown
      setSuccess('')
      return
    }
    
    setError('')
    setIsLoading(true)
    
    try {
      await register(username, email, fullName, password)
      setSuccess('Registration successful! Please check your email to verify your account.')
      // Clear form after successful registration
      setUsername('')
      setEmail('')
      setFullName('')
      setPassword('')
      setConfirmPassword('')
      setTouched({
        username: false,
        email: false,
        fullName: false,
        password: false,
        confirmPassword: false
      })
    } catch (err: any) {
      setError(err.message || 'Registration failed. Please try again.')
      setSuccess('')
    } finally {
      setIsLoading(false)
    }
  }

  const handleInputChange = (field: keyof typeof touched, value: string) => {
    // Update the field value
    switch (field) {
      case 'username':
        setUsername(value)
        break
      case 'email':
        setEmail(value)
        break
      case 'fullName':
        setFullName(value)
        break
      case 'password':
        setPassword(value)
        break
      case 'confirmPassword':
        setConfirmPassword(value)
        break
    }
    
    // Clear any general error when user starts typing
    if (error) {
      setError('')
    }
  }

  const handleLogin = () => {
    navigate('/login')
  }

  return (
    <div className="login-container">
      <div className="login-header">
        <h1>QLib AI</h1>
        <p className="login-subtitle">Create your account</p>
      </div>
      <form className="login-form" onSubmit={handleSubmit}>
        {success && <div className="success login-success">{success}</div>}
        {error && <div className="error login-error">{error}</div>}
        
        <div className="form-group">
          <label htmlFor="username" className="form-label">Username</label>
          <input
            type="text"
            id="username"
            value={username}
            onChange={(e) => handleInputChange('username', e.target.value)}
            onBlur={() => setTouched(prev => ({ ...prev, username: true }))}
            placeholder="Enter your username (min 3 chars)"
            className={`form-input ${touched.username && getFieldError('username') ? 'input-error' : ''}`}
            required
            disabled={isLoading}
          />
          {touched.username && getFieldError('username') && (
            <div className="error">{getFieldError('username')}</div>
          )}
        </div>
        
        <div className="form-group">
          <label htmlFor="email" className="form-label">Email Address</label>
          <input
            type="email"
            id="email"
            value={email}
            onChange={(e) => handleInputChange('email', e.target.value)}
            onBlur={() => setTouched(prev => ({ ...prev, email: true }))}
            placeholder="Enter your email"
            className={`form-input ${touched.email && getFieldError('email') ? 'input-error' : ''}`}
            required
            disabled={isLoading}
          />
          {touched.email && getFieldError('email') && (
            <div className="error">{getFieldError('email')}</div>
          )}
        </div>
        
        <div className="form-group">
          <label htmlFor="fullName" className="form-label">Full Name (Optional)</label>
          <input
            type="text"
            id="fullName"
            value={fullName}
            onChange={(e) => handleInputChange('fullName', e.target.value)}
            placeholder="Enter your full name"
            className="form-input"
            disabled={isLoading}
          />
        </div>
        
        <div className="form-group">
          <label htmlFor="password" className="form-label">Password</label>
          <input
            type="password"
            id="password"
            value={password}
            onChange={(e) => handleInputChange('password', e.target.value)}
            onBlur={() => setTouched(prev => ({ ...prev, password: true }))}
            placeholder="Enter your password (min 6 chars, letters + numbers)"
            className={`form-input ${touched.password && getFieldError('password') ? 'input-error' : ''}`}
            required
            disabled={isLoading}
          />
          {touched.password && getFieldError('password') && (
            <div className="error">{getFieldError('password')}</div>
          )}
        </div>
        
        <div className="form-group">
          <label htmlFor="confirmPassword" className="form-label">Confirm Password</label>
          <input
            type="password"
            id="confirmPassword"
            value={confirmPassword}
            onChange={(e) => handleInputChange('confirmPassword', e.target.value)}
            onBlur={() => setTouched(prev => ({ ...prev, confirmPassword: true }))}
            placeholder="Confirm your password"
            className={`form-input ${touched.confirmPassword && getFieldError('confirmPassword') ? 'input-error' : ''}`}
            required
            disabled={isLoading}
          />
          {touched.confirmPassword && getFieldError('confirmPassword') && (
            <div className="error">{getFieldError('confirmPassword')}</div>
          )}
        </div>
        
        <button 
          type="submit" 
          className="btn btn-primary login-btn"
          disabled={isLoading}
        >
          {isLoading ? (
            <span className="loading">Registering...</span>
          ) : (
            'Register'
          )}
        </button>
        
        <div className="login-footer">
          <p className="login-help">
            Already have an account? <a href="/login" onClick={handleLogin}>Login here</a>
          </p>
        </div>
      </form>
    </div>
  )
}

export default Register
