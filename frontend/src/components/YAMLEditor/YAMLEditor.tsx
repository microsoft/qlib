import React, { useCallback, useMemo, memo } from 'react'
import Editor from '@monaco-editor/react'

interface YAMLEditorProps {
  value: string
  onChange: (value: string) => void
  error?: string
}

const YAMLEditor: React.FC<YAMLEditorProps> = memo(({ value, onChange, error }) => {
  // 使用useCallback优化事件处理函数，避免不必要的重新渲染
  const handleEditorChange = useCallback((editorValue: string | undefined) => {
    if (editorValue !== undefined) {
      onChange(editorValue)
    }
  }, [onChange])

  // 使用useMemo优化编辑器选项，避免每次渲染都创建新对象
  const editorOptions = useMemo(() => ({
    minimap: { enabled: false }, // 禁用minimap以提高性能
    scrollBeyondLastLine: false,
    automaticLayout: true,
    formatOnPaste: true,
    formatOnType: true,
    tabSize: 2,
    fontSize: 14,
    lineNumbers: 'on' as const,
    scrollbar: {
      useShadows: false,
      verticalScrollbarSize: 10,
      horizontalScrollbarSize: 10,
    },
    renderLineHighlight: 'all' as const,
    wordWrap: 'on' as const,
  }), [])

  return (
    <div className="yaml-editor-container">
      <Editor
        height="400px" // 减小默认高度，减少初始渲染时间
        defaultLanguage="yaml"
        value={value}
        onChange={handleEditorChange}
        options={editorOptions}
        loading={<div className="loading">Loading editor...</div>} // 添加加载状态
        onMount={(editor) => {
          // 编辑器挂载后的初始化操作
          editor.focus()
        }}
      />
      {error && (
        <div className="yaml-errors">
          <div className="error-item">{error}</div>
        </div>
      )}
    </div>
  )
})

YAMLEditor.displayName = 'YAMLEditor'

export default YAMLEditor
