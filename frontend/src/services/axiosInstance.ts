import axios from 'axios'

// Create an axios instance with default configuration
const axiosInstance = axios.create({
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor to add auth token
axiosInstance.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor to handle 401 errors
axiosInstance.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      // Clear user data and redirect to login
      localStorage.removeItem('token')
      localStorage.removeItem('token_type')
      localStorage.removeItem('username')
      localStorage.removeItem('userInfo')
      
      // Redirect to login page
      window.location.href = '/'
    }
    return Promise.reject(error)
  }
)

export default axiosInstance
