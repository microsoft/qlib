import React, { useState, useEffect } from 'react'
import { getUsers, createUser, updateUser, deleteUser } from '../services/auth'
import { getServiceStatus } from '../services/monitoring'
import type { UserInfo } from '../services/auth'
import type { ServiceStatusResponse } from '../services/monitoring'

interface UserFormData {
  username: string
  email: string
  full_name: string
  password: string
  role: string
  disabled: boolean
}

const Admin: React.FC = () => {
  const [users, setUsers] = useState<UserInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [showAddModal, setShowAddModal] = useState(false)
  const [showEditModal, setShowEditModal] = useState(false)
  const [currentUser, setCurrentUser] = useState<UserInfo | null>(null)
  const [searchTerm, setSearchTerm] = useState('')
  const [sortField, setSortField] = useState<keyof UserInfo>('id')
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc')
  const [currentPage, setCurrentPage] = useState(1)
  const [usersPerPage] = useState(10)
  const [activeTab, setActiveTab] = useState<'users' | 'configs' | 'logs' | 'monitoring'>('users')
  const [monitoringSubTab, setMonitoringSubTab] = useState<'overview' | 'service-status'>('overview')
  const [serviceStatus, setServiceStatus] = useState<ServiceStatusResponse | null>(null)
  const [serviceStatusLoading, setServiceStatusLoading] = useState(false)
  const [serviceStatusError, setServiceStatusError] = useState<string | null>(null)
  const [formData, setFormData] = useState<UserFormData>({
    username: '',
    email: '',
    full_name: '',
    password: '',
    role: 'viewer',
    disabled: false
  })

  useEffect(() => {
    fetchUsers()
  }, [])

  // 当监控标签页激活或子标签页切换到服务状态时，获取服务状态
  useEffect(() => {
    if (activeTab === 'monitoring' && monitoringSubTab === 'service-status') {
      fetchServiceStatus()
    }
  }, [activeTab, monitoringSubTab])

  // 获取服务状态
  const fetchServiceStatus = async () => {
    setServiceStatusLoading(true)
    setServiceStatusError(null)
    try {
      const status = await getServiceStatus()
      setServiceStatus(status)
    } catch (err) {
      setServiceStatusError('获取服务状态失败')
      console.error('Failed to fetch service status:', err)
    } finally {
      setServiceStatusLoading(false)
    }
  }

  // 手动刷新服务状态
  const handleRefreshServiceStatus = () => {
    fetchServiceStatus()
  }

  const fetchUsers = async () => {
    try {
      setLoading(true)
      const usersData = await getUsers()
      setUsers(usersData)
    } catch (err) {
      setError('Failed to fetch users')
      console.error('Error fetching users:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
    const target = e.target as HTMLInputElement;
    const { name, value, type } = target;
    const checked = target.checked;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }))
  }

  const handleAddUser = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      await createUser(formData)
      setShowAddModal(false)
      resetForm()
      fetchUsers()
    } catch (err) {
      setError('Failed to create user')
      console.error('Error creating user:', err)
    }
  }

  const handleEditUser = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!currentUser || !currentUser.id) return
    
    try {
      await updateUser(currentUser.id, formData)
      setShowEditModal(false)
      resetForm()
      fetchUsers()
    } catch (err) {
      setError('Failed to update user')
      console.error('Error updating user:', err)
    }
  }

  const handleDeleteUser = async (userId: number) => {
    if (window.confirm('Are you sure you want to delete this user?')) {
      try {
        await deleteUser(userId)
        fetchUsers()
      } catch (err) {
        setError('Failed to delete user')
        console.error('Error deleting user:', err)
      }
    }
  }

  const openEditModal = (user: UserInfo) => {
    setCurrentUser(user)
    setFormData({
      username: user.username,
      email: user.email || '',
      full_name: user.full_name || '',
      password: '',
      role: user.role || 'viewer',
      disabled: user.disabled || false
    })
    setShowEditModal(true)
  }

  const handleSort = (field: keyof UserInfo) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')
    } else {
      setSortField(field as keyof UserInfo)
      setSortDirection('asc')
    }
  }

  const resetForm = () => {
    setFormData({
      username: '',
      email: '',
      full_name: '',
      password: '',
      role: 'viewer',
      disabled: false
    })
    setCurrentUser(null)
  }

  if (loading) {
    return <div className="loading">Loading...</div>
  }

  // 过滤和排序用户
  const filteredAndSortedUsers = users
    .filter(user => {
      const searchLower = searchTerm.toLowerCase()
      return (
        user.username.toLowerCase().includes(searchLower) ||
        (user.email && user.email.toLowerCase().includes(searchLower)) ||
        (user.full_name && user.full_name.toLowerCase().includes(searchLower))
      )
    })
    .sort((a, b) => {
      const aVal = a[sortField]
      const bVal = b[sortField]
      
      if (aVal === undefined || aVal === null) return sortDirection === 'asc' ? -1 : 1
      if (bVal === undefined || bVal === null) return sortDirection === 'asc' ? 1 : -1
      
      if (typeof aVal === 'string' && typeof bVal === 'string') {
        return sortDirection === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal)
      } else if (typeof aVal === 'number' && typeof bVal === 'number') {
        return sortDirection === 'asc' ? aVal - bVal : bVal - aVal
      } else if (typeof aVal === 'boolean' && typeof bVal === 'boolean') {
        return sortDirection === 'asc' ? (aVal === bVal ? 0 : aVal ? 1 : -1) : (aVal === bVal ? 0 : aVal ? -1 : 1)
      } else {
        return 0
      }
    })

  // 分页逻辑
  const indexOfLastUser = currentPage * usersPerPage
  const indexOfFirstUser = indexOfLastUser - usersPerPage
  const currentUsers = filteredAndSortedUsers.slice(indexOfFirstUser, indexOfLastUser)
  const totalPages = Math.ceil(filteredAndSortedUsers.length / usersPerPage)

  // 分页控制函数
  const handlePageChange = (pageNumber: number) => {
    setCurrentPage(pageNumber)
  }

  const handlePrevPage = () => {
    if (currentPage > 1) {
      setCurrentPage(currentPage - 1)
    }
  }

  const handleNextPage = () => {
    if (currentPage < totalPages) {
      setCurrentPage(currentPage + 1)
    }
  }

  return (
    <div className="container">
      <div className="page-header">
        <h1>系统管理</h1>
      </div>
      
      {error && (
        <div className="error-message">
          {error}
          <button onClick={() => setError(null)}>×</button>
        </div>
      )}

      {/* 标签页导航 */}
      <div className="admin-tabs">
        <button 
          className={`tab-btn ${activeTab === 'users' ? 'active' : ''}`}
          onClick={() => setActiveTab('users')}
        >
          用户管理
        </button>
        <button 
          className={`tab-btn ${activeTab === 'configs' ? 'active' : ''}`}
          onClick={() => setActiveTab('configs')}
        >
          系统配置
        </button>
        <button 
          className={`tab-btn ${activeTab === 'logs' ? 'active' : ''}`}
          onClick={() => setActiveTab('logs')}
        >
          日志管理
        </button>
        <button 
          className={`tab-btn ${activeTab === 'monitoring' ? 'active' : ''}`}
          onClick={() => setActiveTab('monitoring')}
        >
          系统监控
        </button>
      </div>

      {/* 用户管理标签页内容 */}
      {activeTab === 'users' && (
        <>
          <div className="admin-actions">
            <button className="btn btn-primary" onClick={() => setShowAddModal(true)}>
              添加用户
            </button>
          </div>

          <div className="search-filter">
            <input
              type="text"
              className="search-input"
              placeholder="搜索用户名、邮箱或全名..."
              value={searchTerm}
              onChange={(e) => {
                setSearchTerm(e.target.value)
                setCurrentPage(1) // 搜索时重置到第一页
              }}
            />
          </div>

          <div className="card">
            <table className="users-table">
              <thead>
                <tr>
                  <th onClick={() => handleSort('id')} className="sortable">
                    ID {sortField === 'id' && (sortDirection === 'asc' ? '↑' : '↓')}
                  </th>
                  <th onClick={() => handleSort('username')} className="sortable">
                    用户名 {sortField === 'username' && (sortDirection === 'asc' ? '↑' : '↓')}
                  </th>
                  <th onClick={() => handleSort('email')} className="sortable">
                    邮箱 {sortField === 'email' && (sortDirection === 'asc' ? '↑' : '↓')}
                  </th>
                  <th onClick={() => handleSort('full_name')} className="sortable">
                    全名 {sortField === 'full_name' && (sortDirection === 'asc' ? '↑' : '↓')}
                  </th>
                  <th onClick={() => handleSort('role')} className="sortable">
                    角色 {sortField === 'role' && (sortDirection === 'asc' ? '↑' : '↓')}
                  </th>
                  <th onClick={() => handleSort('disabled')} className="sortable">
                    状态 {sortField === 'disabled' && (sortDirection === 'asc' ? '↑' : '↓')}
                  </th>
                  <th onClick={() => handleSort('created_at')} className="sortable">
                    创建时间 {sortField === 'created_at' && (sortDirection === 'asc' ? '↑' : '↓')}
                  </th>
                  <th onClick={() => handleSort('last_login')} className="sortable">
                    最后登录 {sortField === 'last_login' && (sortDirection === 'asc' ? '↑' : '↓')}
                  </th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                {currentUsers.length > 0 ? (
                  currentUsers.map(user => (
                    <tr key={user.id}>
                      <td>{user.id}</td>
                      <td>{user.username}</td>
                      <td>{user.email || '-'}</td>
                      <td>{user.full_name || '-'}</td>
                      <td>{user.role}</td>
                      <td>
                        <span className={user.disabled ? 'status-disabled' : 'status-active'}>
                          {user.disabled ? '禁用' : '活跃'}
                        </span>
                      </td>
                      <td>{user.created_at ? new Date(user.created_at).toLocaleDateString() : '-'}</td>
                      <td>{user.last_login ? new Date(user.last_login).toLocaleString() : '-'}</td>
                      <td>
                        <button 
                          className="btn btn-sm btn-primary"
                          onClick={() => openEditModal(user)}
                        >
                          编辑
                        </button>
                        <button 
                          className="btn btn-sm btn-danger"
                          onClick={() => user.id && handleDeleteUser(user.id)}
                        >
                          删除
                        </button>
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={9} style={{ textAlign: 'center', padding: '20px', color: '#666' }}>
                      没有找到匹配的用户
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          {/* 分页控件 */}
          {totalPages > 1 && (
            <div className="pagination">
              <button 
                onClick={handlePrevPage} 
                disabled={currentPage === 1}
              >
                上一页
              </button>
              
              {Array.from({ length: totalPages }, (_, i) => i + 1).map(pageNumber => (
                <button
                  key={pageNumber}
                  onClick={() => handlePageChange(pageNumber)}
                  className={currentPage === pageNumber ? 'active' : ''}
                >
                  {pageNumber}
                </button>
              ))}
              
              <button 
                onClick={handleNextPage} 
                disabled={currentPage === totalPages}
              >
                下一页
              </button>
            </div>
          )}
        </>
      )}

      {/* 系统配置标签页内容 */}
      {activeTab === 'configs' && (
        <div className="card">
          <h2>系统配置</h2>
          <div className="configs-content">
            <p>系统配置管理功能正在开发中...</p>
            <div className="configs-placeholder">
              <div className="config-item">
                <h3>API配置</h3>
                <p>管理系统API的相关配置</p>
              </div>
              <div className="config-item">
                <h3>数据库配置</h3>
                <p>管理数据库连接和设置</p>
              </div>
              <div className="config-item">
                <h3>安全配置</h3>
                <p>管理系统安全相关设置</p>
              </div>
              <div className="config-item">
                <h3>日志配置</h3>
                <p>管理系统日志记录设置</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 日志管理标签页内容 */}
      {activeTab === 'logs' && (
        <div className="card">
          <h2>日志管理</h2>
          <div className="logs-content">
            <p>日志管理功能正在开发中...</p>
            <div className="logs-placeholder">
              <div className="log-item">
                <h3>系统日志</h3>
                <p>查看系统运行日志</p>
              </div>
              <div className="log-item">
                <h3>用户日志</h3>
                <p>查看用户操作日志</p>
              </div>
              <div className="log-item">
                <h3>错误日志</h3>
                <p>查看系统错误日志</p>
              </div>
              <div className="log-item">
                <h3>性能日志</h3>
                <p>查看系统性能日志</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 系统监控标签页内容 */}
      {activeTab === 'monitoring' && (
        <div className="card">
          <h2>系统监控</h2>
          
          {/* 监控子标签页导航 */}
          <div className="admin-tabs">
            <button 
              className={`tab-btn ${monitoringSubTab === 'overview' ? 'active' : ''}`}
              onClick={() => setMonitoringSubTab('overview')}
            >
              系统概览
            </button>
            <button 
              className={`tab-btn ${monitoringSubTab === 'service-status' ? 'active' : ''}`}
              onClick={() => setMonitoringSubTab('service-status')}
            >
              服务状态监控
            </button>
          </div>
          
          {/* 系统概览 */}
          {monitoringSubTab === 'overview' && (
            <div className="monitoring-content">
              <p>系统监控功能正在开发中...</p>
              <div className="monitoring-stats">
                <div className="stat-card">
                  <h3>系统状态</h3>
                  <div className="stat-value status-active">运行中</div>
                </div>
                <div className="stat-card">
                  <h3>在线用户</h3>
                  <div className="stat-value">1</div>
                </div>
                <div className="stat-card">
                  <h3>CPU使用率</h3>
                  <div className="stat-value">25%</div>
                </div>
                <div className="stat-card">
                  <h3>内存使用率</h3>
                  <div className="stat-value">45%</div>
                </div>
              </div>
              <div className="monitoring-placeholder">
                <div className="monitoring-item">
                    <h3>系统资源监控</h3>
                    <p>实时监控CPU、内存、磁盘等系统资源使用情况</p>
                </div>
                <div className="monitoring-item">
                  <h3>API性能监控</h3>
                  <p>监控API请求响应时间和成功率</p>
                </div>
                <div className="monitoring-item">
                  <h3>数据库监控</h3>
                  <p>监控数据库连接数和查询性能</p>
                </div>
                <div className="monitoring-item">
                  <h3>服务状态监控</h3>
                  <p>监控系统各服务的运行状态</p>
                </div>
              </div>
            </div>
          )}
          
          {/* 服务状态监控 */}
          {monitoringSubTab === 'service-status' && (
            <div className="service-status-content">
              <div className="service-status-header">
                <h3>服务状态监控</h3>
                <button 
                  className="btn btn-sm btn-primary"
                  onClick={handleRefreshServiceStatus}
                  disabled={serviceStatusLoading}
                >
                  {serviceStatusLoading ? '刷新中...' : '刷新'}
                </button>
              </div>
              
              {serviceStatusError && (
                <div className="error-message">
                  {serviceStatusError}
                  <button onClick={() => setServiceStatusError(null)}>×</button>
                </div>
              )}
              
              {serviceStatusLoading ? (
                <div className="loading">加载中...</div>
              ) : (
                <div className="service-status-grid">
                  {serviceStatus && Object.entries(serviceStatus.services).map(([serviceName, service]) => (
                    <div className="service-card" key={serviceName}>
                      <div className="service-header">
                        <h4>{serviceName === 'local_api' ? '本地API服务' : 'DDNS模型训练服务'}</h4>
                        <div className="status-container">
                          <span className={`status-badge status-${service.status}`}>
                            {service.status === 'healthy' ? '正常' : 
                             service.status === 'unhealthy' ? '异常' : 
                             service.status === 'running' ? '运行中' : 
                             '不可达'}
                          </span>
                          <span className={`running-indicator ${service.is_running ? 'running' : 'not-running'}`}>
                            {service.is_running ? '● 运行中' : '● 已停止'}
                          </span>
                        </div>
                      </div>
                      <div className="service-details">
                        <p>{service.details}</p>
                        {service.server_url && (
                          <p className="server-url">服务器地址: {service.server_url}</p>
                        )}
                        {service.server_status && (
                          <div className="server-status-details">
                            <h5>服务器详细状态:</h5>
                            <pre>{JSON.stringify(service.server_status, null, 2)}</pre>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
              
              {serviceStatus && (
                <div className="last-updated">
                  最后更新: {new Date(serviceStatus.timestamp).toLocaleString()}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Add User Modal */}
      {showAddModal && (
        <div className="modal-overlay">
          <div className="modal">
            <div className="modal-header">
              <h2>添加新用户</h2>
              <button className="close-btn" onClick={() => setShowAddModal(false)}>×</button>
            </div>
            <form onSubmit={handleAddUser} className="modal-body">
              <div className="form-group">
                <label className="form-label">用户名</label>
                <input
                  type="text"
                  name="username"
                  className="form-input"
                  value={formData.username}
                  onChange={handleInputChange}
                  required
                />
              </div>
              <div className="form-group">
                <label className="form-label">邮箱</label>
                <input
                  type="email"
                  name="email"
                  className="form-input"
                  value={formData.email}
                  onChange={handleInputChange}
                />
              </div>
              <div className="form-group">
                <label className="form-label">全名</label>
                <input
                  type="text"
                  name="full_name"
                  className="form-input"
                  value={formData.full_name}
                  onChange={handleInputChange}
                />
              </div>
              <div className="form-group">
                <label className="form-label">密码</label>
                <input
                  type="password"
                  name="password"
                  className="form-input"
                  value={formData.password}
                  onChange={handleInputChange}
                  required
                />
              </div>
              <div className="form-group">
                <label className="form-label">角色</label>
                <select
                  name="role"
                  className="form-input"
                  value={formData.role}
                  onChange={handleInputChange}
                >
                  <option value="viewer">查看者</option>
                  <option value="developer">开发者</option>
                  <option value="admin">管理员</option>
                </select>
              </div>
              <div className="form-group checkbox-group">
                <input
                  type="checkbox"
                  id="disabled"
                  name="disabled"
                  checked={formData.disabled}
                  onChange={handleInputChange}
                />
                <label htmlFor="disabled">禁用</label>
              </div>
              <div className="modal-footer">
                <button type="button" className="btn btn-secondary" onClick={() => setShowAddModal(false)}>
                  取消
                </button>
                <button type="submit" className="btn btn-primary">
                  创建用户
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Edit User Modal */}
      {showEditModal && currentUser && (
        <div className="modal-overlay">
          <div className="modal">
            <div className="modal-header">
              <h2>编辑用户: {currentUser.username}</h2>
              <button className="close-btn" onClick={() => setShowEditModal(false)}>×</button>
            </div>
            <form onSubmit={handleEditUser} className="modal-body">
              <div className="form-group">
                <label className="form-label">用户名</label>
                <input
                  type="text"
                  name="username"
                  className="form-input"
                  value={formData.username}
                  onChange={handleInputChange}
                  required
                />
              </div>
              <div className="form-group">
                <label className="form-label">邮箱</label>
                <input
                  type="email"
                  name="email"
                  className="form-input"
                  value={formData.email}
                  onChange={handleInputChange}
                />
              </div>
              <div className="form-group">
                <label className="form-label">全名</label>
                <input
                  type="text"
                  name="full_name"
                  className="form-input"
                  value={formData.full_name}
                  onChange={handleInputChange}
                />
              </div>
              <div className="form-group">
                <label className="form-label">密码 (留空保持当前密码)</label>
                <input
                  type="password"
                  name="password"
                  className="form-input"
                  value={formData.password}
                  onChange={handleInputChange}
                />
              </div>
              <div className="form-group">
                <label className="form-label">角色</label>
                <select
                  name="role"
                  className="form-input"
                  value={formData.role}
                  onChange={handleInputChange}
                >
                  <option value="viewer">查看者</option>
                  <option value="developer">开发者</option>
                  <option value="admin">管理员</option>
                </select>
              </div>
              <div className="form-group checkbox-group">
                <input
                  type="checkbox"
                  id="disabled"
                  name="disabled"
                  checked={formData.disabled}
                  onChange={handleInputChange}
                />
                <label htmlFor="disabled">禁用</label>
              </div>
              <div className="modal-footer">
                <button type="button" className="btn btn-secondary" onClick={() => setShowEditModal(false)}>
                  取消
                </button>
                <button type="submit" className="btn btn-primary">
                  更新用户
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}

export default Admin
