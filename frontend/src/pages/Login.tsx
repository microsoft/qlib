import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { login } from '../services/auth'

const Login: React.FC = () => {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [touched, setTouched] = useState({
    username: false,
    password: false
  })
  const navigate = useNavigate()

  // 表单验证
  const validate = () => {
    let isValid = true
    let errorMessage = ''

    if (!username.trim()) {
      isValid = false
      errorMessage = 'Please enter your username'
    } else if (!password.trim()) {
      isValid = false
      errorMessage = 'Please enter your password'
    } else if (password.length < 6) {
      isValid = false
      errorMessage = 'Password must be at least 6 characters long'
    }

    return { isValid, errorMessage }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setTouched({ username: true, password: true })
    
    const validation = validate()
    if (!validation.isValid) {
      setError(validation.errorMessage)
      return
    }
    
    setError('')
    setIsLoading(true)
    
    try {
      await login(username, password)
      navigate('/dashboard')
    } catch (_err) {
      setError('Invalid username or password')
    } finally {
      setIsLoading(false)
    }
  }

  const handleInputChange = (field: 'username' | 'password', value: string) => {
    if (field === 'username') {
      setUsername(value)
    } else {
      setPassword(value)
    }
    
    // 清除错误信息当用户开始输入
    if (error) {
      setError('')
    }
  }

  return (
    <div className="login-container">
      <div className="login-header">
        <h1>QLib AI</h1>
        <p className="login-subtitle">Sign in to your account</p>
      </div>
      <form className="login-form" onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="username" className="form-label">Username</label>
          <input
            type="text"
            id="username"
            value={username}
            onChange={(e) => handleInputChange('username', e.target.value)}
            onBlur={() => setTouched(prev => ({ ...prev, username: true }))}
            placeholder="Enter your username"
            className={`form-input ${touched.username && !username ? 'input-error' : ''}`}
            required
            disabled={isLoading}
          />
          {touched.username && !username && (
            <div className="error">Username is required</div>
          )}
        </div>
        <div className="form-group">
          <label htmlFor="password" className="form-label">Password</label>
          <input
            type="password"
            id="password"
            value={password}
            onChange={(e) => handleInputChange('password', e.target.value)}
            onBlur={() => setTouched(prev => ({ ...prev, password: true }))}
            placeholder="Enter your password"
            className={`form-input ${touched.password && !password ? 'input-error' : ''}`}
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
        {error && <div className="error login-error">{error}</div>}
        <div className="login-options">
          <a href="/forgot-password" className="forgot-password-link">
            Forgot Password?
          </a>
        </div>
        
        <button 
          type="submit" 
          className="btn btn-primary login-btn"
          disabled={isLoading}
        >
          {isLoading ? (
            <span className="loading">Logging in</span>
          ) : (
            'Login'
          )}
        </button>
        <div className="login-footer">
          <p className="login-help">
            <span>Default credentials: </span>
            <strong>username: admin, password: admin123</strong>
          </p>
          <p className="login-register">
            Don't have an account? <a href="/register">Register here</a>
          </p>
        </div>
      </form>
    </div>
  )
}

export default Login
