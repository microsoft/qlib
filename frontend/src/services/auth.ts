import axiosInstance from './axiosInstance'

const API_URL = '/api/auth/'

interface LoginResponse {
  access_token: string
  token_type: string
}

export interface UserInfo {
  id?: number
  username: string
  email?: string
  full_name?: string
  role?: string
  disabled?: boolean
  created_at?: string
  updated_at?: string
  last_login?: string
}

export const login = async (username: string, password: string): Promise<void> => {
  try {
    // Call actual login API to get token
    const response = await axiosInstance.post<LoginResponse>(`${API_URL}token`, {
      username,
      password
    }, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded'
      },
      transformRequest: [(data) => {
        // Transform data to form-urlencoded format
        return Object.entries(data).map(([key, value]) => `${encodeURIComponent(key)}=${encodeURIComponent(value as string)}`).join('&');
      }]
    });
    
    const { access_token, token_type } = response.data;
    localStorage.setItem('token', access_token);
    localStorage.setItem('token_type', token_type);
    localStorage.setItem('username', username);
    
    // 获取并存储用户信息
    const userInfo = await getUserInfo();
    if (userInfo) {
      localStorage.setItem('userInfo', JSON.stringify(userInfo));
    }
  } catch (error) {
    console.error('Login failed:', error);
    throw error;
  }
}

export const logout = (): void => {
  localStorage.removeItem('token')
  localStorage.removeItem('username')
  localStorage.removeItem('userInfo')
}

export const getToken = (): string | null => {
  return localStorage.getItem('token')
}

export const isAuthenticated = (): boolean => {
  return !!getToken()
}

export const getUserInfo = async (): Promise<UserInfo | null> => {
  try {
    const response = await axiosInstance.get<UserInfo>(`${API_URL}users/me`)
    // Store user info in localStorage for quick access
    localStorage.setItem('userInfo', JSON.stringify(response.data))
    return response.data
  } catch (error) {
    console.error('Failed to get user info:', error)
    return null
  }
}

// User management functions (admin only)
export const getUsers = async (): Promise<UserInfo[]> => {
  try {
    const response = await axiosInstance.get<UserInfo[]>(`${API_URL}users`)
    return response.data
  } catch (error) {
    console.error('Failed to get users:', error)
    throw error
  }
}

export const createUser = async (userData: {
  username: string
  email: string
  full_name: string
  password: string
  role: string
  disabled: boolean
}): Promise<void> => {
  try {
    await axiosInstance.post(`${API_URL}users`, userData)
  } catch (error) {
    console.error('Failed to create user:', error)
    throw error
  }
}

export const updateUser = async (
  userId: number,
  userData: {
    username: string
    email: string
    full_name: string
    password?: string
    role: string
    disabled: boolean
  }
): Promise<void> => {
  try {
    // Remove password if empty
    const updateData = { ...userData }
    if (!updateData.password) {
      delete updateData.password
    }
    
    await axiosInstance.put(`${API_URL}users/${userId}`, updateData)
  } catch (error) {
    console.error('Failed to update user:', error)
    throw error
  }
}

export const deleteUser = async (userId: number): Promise<void> => {
  try {
    await axiosInstance.delete(`${API_URL}users/${userId}`)
  } catch (error) {
    console.error('Failed to delete user:', error)
    throw error
  }
}

// User registration function
export const register = async (
  username: string,
  email: string,
  fullName: string,
  password: string
): Promise<void> => {
  try {
    await axiosInstance.post(`${API_URL}register`, {
      username,
      email,
      full_name: fullName,
      password,
      role: 'viewer' // Default role for new users
    })
  } catch (error) {
    console.error('Registration failed:', error)
    throw error
  }
}

// Email verification function
export const verifyEmail = async (token: string): Promise<{ message: string }> => {
  try {
    const response = await axiosInstance.get(`${API_URL}verify-email`, {
      params: { token }
    })
    return response.data
  } catch (error) {
    console.error('Email verification failed:', error)
    throw error
  }
}

// Resend verification email function
export const resendVerification = async (email: string): Promise<{ message: string }> => {
  try {
    const response = await axiosInstance.post(`${API_URL}resend-verification`, {
      email
    })
    return response.data
  } catch (error) {
    console.error('Failed to resend verification email:', error)
    throw error
  }
}

// Forgot password function
export const forgotPassword = async (email: string): Promise<{ message: string }> => {
  try {
    const response = await axiosInstance.post(`${API_URL}forgot-password`, {
      email
    })
    return response.data
  } catch (error) {
    console.error('Failed to send password reset link:', error)
    throw error
  }
}

// Reset password function
export const resetPassword = async (token: string, newPassword: string): Promise<{ message: string }> => {
  try {
    const response = await axiosInstance.post(`${API_URL}reset-password`, {
      token,
      new_password: newPassword
    })
    return response.data
  } catch (error) {
    console.error('Failed to reset password:', error)
    throw error
  }
}
