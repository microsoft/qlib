import { BrowserRouter as Router, Routes, Route, Outlet } from 'react-router-dom'
import Login from './pages/Login'
import Register from './pages/Register'
import VerifyEmail from './pages/VerifyEmail'
import ForgotPassword from './pages/ForgotPassword'
import ResetPassword from './pages/ResetPassword'
import Dashboard from './pages/Dashboard'
import Experiments from './pages/Experiments'
import ExperimentDetail from './pages/ExperimentDetail'
import Models from './pages/Models'
import ModelDetail from './pages/ModelDetail'
import Configs from './pages/Configs'
import Profile from './pages/Profile'
import Admin from './pages/Admin'
import FactorManagement from './pages/FactorManagement'
import DataManagement from './pages/DataManagement'
import Backtest from './pages/Backtest'
import RiskManagement from './pages/RiskManagement'
import FactorAnalysis from './pages/FactorAnalysis'

import Navigation from './components/Navigation/Navigation'
import PrivateRoute from './components/PrivateRoute/PrivateRoute'
import './App.css'

// Main layout component with navigation
const MainLayout: React.FC = () => {
  return (
    <div className="app-layout">
      <Navigation />
      <main className="main-content">
        <Outlet />
      </main>
    </div>
  )
}

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<Login />} />
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          <Route path="/verify-email" element={<VerifyEmail />} />
          <Route path="/forgot-password" element={<ForgotPassword />} />
          <Route path="/reset-password" element={<ResetPassword />} />
          
          {/* Protected routes for all authenticated users */}
              <Route element={<PrivateRoute />}>
                <Route element={<MainLayout />}>
                  <Route path="/dashboard" element={<Dashboard />} />
                  <Route path="/experiments" element={<Experiments />} />
                  <Route path="/experiments/:id" element={<ExperimentDetail />} />
                  <Route path="/backtest" element={<Backtest />} />
                  <Route path="/risk" element={<RiskManagement />} />
                  <Route path="/models" element={<Models />} />
                  <Route path="/models/:id" element={<ModelDetail />} />
                  <Route path="/factors" element={<FactorManagement />} />
                  <Route path="/factor-analysis" element={<FactorAnalysis />} />
                  <Route path="/data" element={<DataManagement />} />
                  <Route path="/profile" element={<Profile />} />
                  
                  {/* Developer and admin only routes */}
                  <Route element={<PrivateRoute allowedRoles={['developer', 'admin']} />}>
                    <Route path="/configs" element={<Configs />} />
                  </Route>
                  
                  {/* Admin only routes */}
                  <Route element={<PrivateRoute allowedRoles={['admin']} />}>
                    <Route path="/admin" element={<Admin />} />
                  </Route>
                </Route>
              </Route>
        </Routes>
      </div>
    </Router>
  )
}

export default App
