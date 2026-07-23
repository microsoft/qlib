import React from 'react'
import { Navigate, Outlet } from 'react-router-dom'
import { isAuthenticated } from '../../services/auth'

interface PrivateRouteProps {
  allowedRoles?: string[]
}

const PrivateRoute: React.FC<PrivateRouteProps> = ({ allowedRoles }) => {
  const isAuth = isAuthenticated()
  const userInfo = localStorage.getItem('userInfo') ? JSON.parse(localStorage.getItem('userInfo') || '{}') : null

  if (!isAuth) {
    // Not authenticated, redirect to login
    return <Navigate to="/" replace />
  }

  if (allowedRoles && userInfo && !allowedRoles.includes(userInfo.role)) {
    // Authenticated but not authorized, redirect to dashboard
    return <Navigate to="/dashboard" replace />
  }

  // Authenticated and authorized, render the component
  return <Outlet />
}

export default PrivateRoute
