import React, { useState, useEffect } from 'react'
import { getConfigs, createConfig, updateConfig, deleteConfig } from '../services/configs'
import type { ConfigType } from '../services/configs'
import YAMLEditor from '../components/YAMLEditor/YAMLEditor'

interface Config {
  id: number
  name: string
  description: string
  content: string
  type: ConfigType
  created_at: string
  updated_at: string
}

const Configs: React.FC = () => {
  const [configs, setConfigs] = useState<Config[]>([])
  const [loading, setLoading] = useState(true)
  const [showForm, setShowForm] = useState(false)
  const [editingConfig, setEditingConfig] = useState<Config | null>(null)
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [content, setContent] = useState('')
  const [type, setType] = useState<ConfigType>('normal')
  const [error, setError] = useState('')

  useEffect(() => {
    const fetchConfigs = async () => {
      try {
        const data = await getConfigs()
        setConfigs(data)
      } catch (err) {
        console.error('Failed to fetch configs:', err)
      } finally {
        setLoading(false)
      }
    }

    fetchConfigs()
  }, [])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    try {
      if (editingConfig) {
        await updateConfig(editingConfig.id, { name, description, content, type })
      } else {
        await createConfig({ name, description, content, type })
      }
      
      // Refresh configs list
      const data = await getConfigs()
      setConfigs(data)
      
      // Reset form
      resetForm()
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to save config')
    }
  }

  const handleEdit = (config: Config) => {
    setEditingConfig(config)
    setName(config.name)
    setDescription(config.description)
    setContent(config.content)
    setType(config.type)
    setShowForm(true)
  }

  const handleDelete = async (id: number) => {
    if (window.confirm('确定要删除这个配置吗？')) {
      try {
        await deleteConfig(id)
        setConfigs(configs.filter(config => config.id !== id))
      } catch (err) {
        console.error('Failed to delete config:', err)
      }
    }
  }

  const resetForm = () => {
    setEditingConfig(null)
    setName('')
    setDescription('')
    setContent('')
    setType('normal')
    setShowForm(false)
    setError('')
  }

  if (loading) {
    return <div className="container">加载中...</div>
  }

  return (
    <div className="container page-transition">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <h1>配置管理</h1>
        <button className="btn" onClick={() => setShowForm(true)}>
          创建配置
        </button>
      </div>

      {showForm && (
        <div className="card">
          <h2>{editingConfig ? '编辑配置' : '创建配置'}</h2>
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label htmlFor="name" className="form-label">名称</label>
              <input
                type="text"
                id="name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
                className="form-input"
              />
            </div>
            <div className="form-group">
              <label htmlFor="description" className="form-label">描述</label>
              <textarea
                id="description"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                rows={3}
                className="form-input"
              />
            </div>
            <div className="form-group">
              <label htmlFor="type" className="form-label">类型</label>
              <select
                id="type"
                value={type}
                onChange={(e) => setType(e.target.value as ConfigType)}
                className="form-input"
              >
                <option value="normal">普通配置</option>
                <option value="experiment_template">实验模板</option>
              </select>
            </div>
            <div className="form-group">
              <label htmlFor="content" className="form-label">内容</label>
              <YAMLEditor
                value={content}
                onChange={setContent}
                error={error}
              />
            </div>
            <div className="form-actions">
              <button type="submit" className="btn btn-primary">
                {editingConfig ? '更新' : '创建'}
              </button>
              <button type="button" className="btn btn-secondary" onClick={resetForm}>
                取消
              </button>
            </div>
          </form>
        </div>
      )}

      <div className="card">
        <h2>配置列表</h2>
        <div className="configs-list">
          <table className="configs-table">
            <thead>
              <tr>
                <th>名称</th>
                <th>类型</th>
                <th>描述</th>
                <th>创建时间</th>
                <th>更新时间</th>
                <th>内容预览</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {configs.map(config => (
                <tr key={config.id}>
                  <td>
                    <div className="config-name">{config.name}</div>
                    <div className="config-id">ID: {config.id}</div>
                  </td>
                  <td>
                    <span className={`config-type config-type-${config.type}`}>
                      {config.type === 'experiment_template' ? '实验模板' : '普通配置'}
                    </span>
                  </td>
                  <td className="config-description">
                    {config.description || '无描述'}
                  </td>
                  <td className="config-date">
                    {new Date(config.created_at).toLocaleString()}
                  </td>
                  <td className="config-date">
                    {new Date(config.updated_at).toLocaleString()}
                  </td>
                  <td className="config-content-preview">
                    {config.content.substring(0, 100)}{config.content.length > 100 ? '...' : ''}
                  </td>
                  <td className="config-actions">
                    <button className="btn btn-sm btn-primary" onClick={() => handleEdit(config)}>
                      编辑
                    </button>
                    <button className="btn btn-sm btn-danger" onClick={() => handleDelete(config.id)}>
                      删除
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

export default Configs
