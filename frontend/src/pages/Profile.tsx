import React, { useState, useEffect } from 'react'
import { getUserInfo } from '../services/auth'

const Profile: React.FC = () => {
  const [userInfo, setUserInfo] = useState<any>(null)
  const [loading, setLoading] = useState<boolean>(true)

  useEffect(() => {
    const fetchUserInfo = async () => {
      try {
        const info = await getUserInfo()
        setUserInfo(info)
      } catch (error) {
        console.error('Failed to fetch user info:', error)
      } finally {
        setLoading(false)
      }
    }
    fetchUserInfo()
  }, [])

  if (loading) {
    return <div className="loading">加载用户信息...</div>
  }

  return (
    <div className="container page-transition">
      <div className="page-header">
        <h1>用户资料</h1>
      </div>
      <div className="card">
        <h2>个人信息</h2>
        <div className="profile-info">
          <div className="info-item">
            <label>用户名:</label>
            <span>{userInfo.username}</span>
          </div>
          <div className="info-item">
            <label>邮箱:</label>
            <span>{userInfo.email || '未提供'}</span>
          </div>
          <div className="info-item">
            <label>全名:</label>
            <span>{userInfo.full_name || '未提供'}</span>
          </div>
          <div className="info-item">
            <label>账户状态:</label>
            <span>{userInfo.is_active ? '活跃' : '非活跃'}</span>
          </div>
          <div className="info-item">
            <label>加入时间:</label>
            <span>{new Date(userInfo.created_at).toLocaleString()}</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Profile