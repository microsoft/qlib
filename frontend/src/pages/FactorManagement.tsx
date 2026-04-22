import React, { useState, useEffect } from 'react'
import { 
  getFactors, createFactor, updateFactor, deleteFactor,
  getFactorGroups, getFactorGroupWithFactors
} from '../services/factors'
import './FactorManagement.css'

// Factor Group interfaces
interface FactorGroup {
  id: number
  name: string
  description: string
  factor_count: number
  status: string
  created_at: string
  updated_at: string
}

// Factor interfaces
interface Factor {
  id: number
  name: string
  description: string
  formula: string
  type: string
  status: string
  group_id?: number
  group?: FactorGroup
  created_at: string
  updated_at: string
}

const FactorManagement: React.FC = () => {
  // Factor states
  const [factors, setFactors] = useState<Factor[]>([])
  const [factorGroups, setFactorGroups] = useState<FactorGroup[]>([])
  const [expandedGroups, setExpandedGroups] = useState<Set<number>>(new Set())
  const [loading, setLoading] = useState(true)
  const [showForm, setShowForm] = useState(false)
  const [editingFactor, setEditingFactor] = useState<Factor | null>(null)
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [formula, setFormula] = useState('')
  const [type, setType] = useState('')
  const [group_id, setGroupId] = useState<number | undefined>(undefined)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')

  useEffect(() => {
    fetchData()
  }, [])

  const fetchData = async () => {
    try {
      setLoading(true)
      const [factorsData, groupsData] = await Promise.all([
        getFactors(),
        getFactorGroups()
      ])
      setFactors(factorsData)
      setFactorGroups(groupsData)
    } catch (err) {
      setError('Failed to fetch data')
      console.error('Error fetching data:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setSuccess('')

    try {
      if (editingFactor) {
        // Update existing factor
        await updateFactor(editingFactor.id, {
          name,
          description,
          formula,
          type,
          group_id
        })
        setSuccess('Factor updated successfully')
      } else {
        // Create new factor
        await createFactor({
          name,
          description,
          formula,
          type,
          group_id
        })
        setSuccess('Factor created successfully')
      }
      
      // Reset form and fetch updated data
      resetForm()
      fetchData()
    } catch (err) {
      setError(editingFactor ? 'Failed to update factor' : 'Failed to create factor')
      console.error('Error submitting factor:', err)
    }
  }

  const handleEdit = (factor: Factor) => {
    setEditingFactor(factor)
    setName(factor.name)
    setDescription(factor.description || '')
    setFormula(factor.formula)
    setType(factor.type)
    setGroupId(factor.group_id)
    setShowForm(true)
  }

  const handleDelete = async (id: number) => {
    if (window.confirm('Are you sure you want to delete this factor?')) {
      try {
        await deleteFactor(id)
        setSuccess('Factor deleted successfully')
        fetchData()
      } catch (err) {
        setError('Failed to delete factor')
        console.error('Error deleting factor:', err)
      }
    }
  }

  const resetForm = () => {
    setEditingFactor(null)
    setName('')
    setDescription('')
    setFormula('')
    setType('')
    setGroupId(undefined)
    setShowForm(false)
  }

  const toggleGroupExpansion = async (groupId: number) => {
    const newExpandedGroups = new Set(expandedGroups)
    if (newExpandedGroups.has(groupId)) {
      newExpandedGroups.delete(groupId)
    } else {
      newExpandedGroups.add(groupId)
      // Fetch group with factors if not already loaded
      const groupWithFactors = await getFactorGroupWithFactors(groupId)
      // Update factors with the group's factors
      setFactors(prevFactors => {
        // Remove existing factors from this group
        const filteredFactors = prevFactors.filter(f => f.group_id !== groupId)
        // Add the new factors
        return [...filteredFactors, ...groupWithFactors.factors]
      })
    }
    setExpandedGroups(newExpandedGroups)
  }

  // Group factors by their group_id
  const groupFactors = () => {
    const grouped: Record<number | string, Factor[]> = {}
    
    // Initialize groups
    factorGroups.forEach(group => {
      grouped[group.id] = []
    })
    grouped['ungrouped'] = []
    
    // Assign factors to groups
    factors.forEach(factor => {
      if (factor.group_id && grouped[factor.group_id] !== undefined) {
        grouped[factor.group_id].push(factor)
      } else {
        grouped['ungrouped'].push(factor)
      }
    })
    
    return grouped
  }

  const groupedFactors = groupFactors()

  return (
    <div className="container">
      {error && <div className="alert alert-error">{error}</div>}
      {success && <div className="alert alert-success">{success}</div>}
      
      <div className="page-header">
        <h1>因子管理</h1>
        <button 
          className="btn btn-primary" 
          onClick={() => setShowForm(true)}
        >
          创建因子
        </button>
      </div>
      
      {showForm && (
        <div className="card" style={{ marginBottom: '20px' }}>
          <div className="form-header">
            <h2>{editingFactor ? '编辑因子' : '创建因子'}</h2>
            <button className="btn btn-secondary" onClick={resetForm}>
              取消
            </button>
          </div>
          <form onSubmit={handleSubmit} className="factor-form">
            <div className="form-group">
              <label htmlFor="name">因子名称</label>
              <input
                type="text"
                id="name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
                className="form-control"
              />
            </div>
            <div className="form-group">
              <label htmlFor="description">描述</label>
              <textarea
                id="description"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                className="form-control"
                rows={3}
              />
            </div>
            <div className="form-group">
              <label htmlFor="formula">计算公式</label>
              <textarea
                id="formula"
                value={formula}
                onChange={(e) => setFormula(e.target.value)}
                required
                className="form-control"
                rows={3}
              />
            </div>
            <div className="form-group">
              <label htmlFor="type">因子类型</label>
              <input
                type="text"
                id="type"
                value={type}
                onChange={(e) => setType(e.target.value)}
                required
                className="form-control"
              />
            </div>
            <div className="form-group">
              <label htmlFor="group_id">因子组</label>
              <select
                id="group_id"
                value={group_id || ''}
                onChange={(e) => setGroupId(e.target.value ? parseInt(e.target.value) : undefined)}
                className="form-control"
              >
                <option value="">无分组</option>
                {factorGroups.map(group => (
                  <option key={group.id} value={group.id}>
                    {group.name} ({group.factor_count}个因子)
                  </option>
                ))}
              </select>
            </div>
            <div className="form-actions">
              <button type="submit" className="btn btn-primary">
                {editingFactor ? '更新因子' : '创建因子'}
              </button>
            </div>
          </form>
        </div>
      )}
      
      <div className="card">
        <h2>因子组列表</h2>
        {loading ? (
          <div className="loading">Loading...</div>
        ) : (
          <div className="factor-groups-list">
            {/* Display factor groups */}
            {factorGroups.map(group => {
              const isExpanded = expandedGroups.has(group.id)
              const groupFactorsList = groupedFactors[group.id] || []
              
              return (
                <div key={group.id} className="factor-group-card">
                  <div 
                    className="factor-group-header"
                    onClick={() => toggleGroupExpansion(group.id)}
                  >
                    <div className="group-info">
                      <h3>{group.name}</h3>
                      <p className="group-description">{group.description || '无描述'}</p>
                    </div>
                    <div className="group-stats">
                      <span className={`status-badge ${group.status}`}>
                        {group.status}
                      </span>
                      <span className="factor-count">
                        {group.factor_count}个因子
                      </span>
                      <span className="expand-icon">
                        {isExpanded ? '▼' : '▶'}
                      </span>
                    </div>
                  </div>
                  
                  {isExpanded && (
                    <div className="factor-group-content">
                      {groupFactorsList.length > 0 ? (
                        <table className="factors-table">
                          <thead>
                            <tr>
                              <th>名称</th>
                              <th>描述</th>
                              <th>类型</th>
                              <th>状态</th>
                              <th>创建时间</th>
                              <th>操作</th>
                            </tr>
                          </thead>
                          <tbody>
                            {groupFactorsList.map((factor) => (
                              <tr key={factor.id}>
                                <td>{factor.name}</td>
                                <td>{factor.description || '-'}</td>
                                <td>{factor.type}</td>
                                <td>
                                  <span className={`status-badge ${factor.status}`}>
                                    {factor.status}
                                  </span>
                                </td>
                                <td>{new Date(factor.created_at).toLocaleString()}</td>
                                <td className="actions">
                                  <button 
                                    className="btn btn-sm btn-primary" 
                                    onClick={(e) => {
                                      e.stopPropagation()
                                      handleEdit(factor)
                                    }}
                                  >
                                    编辑
                                  </button>
                                  <button 
                                    className="btn btn-sm btn-danger" 
                                    onClick={(e) => {
                                      e.stopPropagation()
                                      handleDelete(factor.id)
                                    }}
                                  >
                                    删除
                                  </button>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      ) : (
                        <div className="empty-group">
                          该因子组暂无因子
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )
            })}
            
            {/* Display ungrouped factors */}
            {groupedFactors['ungrouped'].length > 0 && (
              <div className="factor-group-card">
                <div className="factor-group-header">
                  <div className="group-info">
                    <h3>未分组因子</h3>
                  </div>
                  <div className="group-stats">
                    <span className="factor-count">
                      {groupedFactors['ungrouped'].length}个因子
                    </span>
                  </div>
                </div>
                <div className="factor-group-content">
                  <table className="factors-table">
                    <thead>
                      <tr>
                        <th>名称</th>
                        <th>描述</th>
                        <th>类型</th>
                        <th>状态</th>
                        <th>创建时间</th>
                        <th>操作</th>
                      </tr>
                    </thead>
                    <tbody>
                      {groupedFactors['ungrouped'].map((factor) => (
                        <tr key={factor.id}>
                          <td>{factor.name}</td>
                          <td>{factor.description || '-'}</td>
                          <td>{factor.type}</td>
                          <td>
                            <span className={`status-badge ${factor.status}`}>
                              {factor.status}
                            </span>
                          </td>
                          <td>{new Date(factor.created_at).toLocaleString()}</td>
                          <td className="actions">
                            <button 
                              className="btn btn-sm btn-primary" 
                              onClick={() => handleEdit(factor)}
                            >
                              编辑
                            </button>
                            <button 
                              className="btn btn-sm btn-danger" 
                              onClick={() => handleDelete(factor.id)}
                            >
                              删除
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default FactorManagement
