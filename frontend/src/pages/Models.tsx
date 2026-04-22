import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { getModels, deleteModel } from '../services/models'

interface ModelVersion {
  id: number
  name: string
  experiment_id: number
  version: number
  metrics: any
  path: string
  created_at: string
  performance?: any
}

interface ModelResponse {
  data: ModelVersion[]
  total: number
  page: number
  per_page: number
}

const Models: React.FC = () => {
  const [models, setModels] = useState<ModelVersion[]>([])
  const [loading, setLoading] = useState(true)
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [perPage] = useState(10)
  const navigate = useNavigate()

  useEffect(() => {
    const fetchModels = async () => {
      try {
        setLoading(true)
        const modelsData: ModelResponse = await getModels(page, perPage)
        setModels(modelsData.data)
        setTotal(modelsData.total)
      } catch (err) {
        console.error('Failed to fetch models:', err)
        setModels([])
        setTotal(0)
      } finally {
        setLoading(false)
      }
    }

    fetchModels()
  }, [page, perPage])

  const handleDeleteModel = async (id: number) => {
    if (window.confirm('确定要删除这个模型吗？')) {
      try {
        await deleteModel(id)
        // Refresh the current page after deletion
        const modelsData: ModelResponse = await getModels(page, perPage)
        setModels(modelsData.data)
        setTotal(modelsData.total)
      } catch (err) {
        console.error('Failed to delete model:', err)
      }
    }
  }

  const totalPages = Math.ceil(total / perPage)

  const handlePreviousPage = () => {
    if (page > 1) {
      setPage(page - 1)
    }
  }

  const handleNextPage = () => {
    if (page < totalPages) {
      setPage(page + 1)
    }
  }

  if (loading) {
    return <div className="container">加载中...</div>
  }

  return (
    <div className="container page-transition">
      <div className="page-header">
        <h1>模型管理</h1>
      </div>

      <div className="models-list">
        <table className="models-table">
          <thead>
            <tr>
              <th>名称</th>
              <th>版本</th>
              <th>实验ID</th>
              <th>创建时间</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            {models.map(model => (
              <tr key={model.id}>
                <td>{model.name}</td>
                <td>{model.version}</td>
                <td>{model.experiment_id}</td>
                <td>{new Date(model.created_at).toLocaleString()}</td>
                <td>
                  <button 
                    className="btn btn-sm" 
                    onClick={() => navigate(`/models/${model.id}`)}
                  >
                    查看
                  </button>
                  <button 
                    className="btn btn-sm btn-danger" 
                    onClick={() => handleDeleteModel(model.id)}
                  >
                    删除
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="pagination">
        <button 
          className="btn btn-sm" 
          onClick={handlePreviousPage} 
          disabled={page === 1}
        >
          上一页
        </button>
        <span className="pagination-info">
          第 {page} 页，共 {totalPages} 页，总计 {total} 个模型
        </span>
        <button 
          className="btn btn-sm" 
          onClick={handleNextPage} 
          disabled={page === totalPages}
        >
          下一页
        </button>
      </div>
    </div>
  )
}

export default Models
