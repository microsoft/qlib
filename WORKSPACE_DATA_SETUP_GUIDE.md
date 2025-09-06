# 📁 Qlib Workspace Data Setup - Complete Guide

**Updated:** 2025-09-06  
**Environment:** qlib conda environment  
**Data Location:** `/workspace/qlib/data/qlib_data/cn_data` (Workspace)

---

## ✅ **DATA MIGRATION COMPLETE**

The Qlib dataset has been successfully moved from the home directory to the workspace for better accessibility and portability:

- ✅ **Original Location:** `~/.qlib/qlib_data/cn_data` (706MB)
- ✅ **New Location:** `/workspace/qlib/data/qlib_data/cn_data` (706MB)
- ✅ **Configuration Updated:** All scripts now use workspace paths
- ✅ **Backward Compatibility:** Scripts fall back to home directory if needed
- ✅ **Testing Complete:** Full functionality verified

---

## 📂 **New Workspace Directory Structure**

```
/workspace/qlib/
├── data/                           # 📁 DATA DIRECTORY (NEW)
│   └── qlib_data/
│       └── cn_data/               # 706MB - Chinese stock market data
│           ├── calendars/         # Trading calendar files
│           │   ├── day.txt        # Regular trading days
│           │   └── day_future.txt # Future trading days  
│           ├── instruments/       # Stock universe definitions
│           │   ├── csi300.txt     # CSI 300 constituents (336 stocks)
│           │   ├── csi500.txt     # CSI 500 constituents (645 stocks)
│           │   ├── csi800.txt     # CSI 800 constituents (932 stocks)
│           │   ├── csi1000.txt    # CSI 1000 constituents (1,310 stocks)
│           │   └── all.txt        # All available stocks (5,516 stocks)
│           └── features/          # Individual stock data (5,516+ folders)
│               ├── sh600000/      # Shanghai stocks (sh prefix)
│               ├── sz000001/      # Shenzhen stocks (sz prefix)  
│               └── bj430017/      # Beijing stocks (bj prefix)
│
├── qlib_analysis_output/          # 📊 ANALYSIS OUTPUT DIRECTORY
│   ├── enhanced_trading_calendar.png
│   ├── market_overview.png
│   ├── stock_analysis_*.png
│   ├── interactive_*.html
│   ├── summary_statistics.csv
│   └── workspace_data_summary.png # NEW: Latest analysis summary
│
├── qlib_config.py                 # 🔧 CONFIGURATION MODULE (NEW)
├── workspace_data_analysis.py     # 📊 WORKSPACE ANALYSIS SCRIPT (NEW)
├── data_analysis_setup.py         # 📊 Original analysis script (UPDATED)
├── create_visualizations.py       # 📈 Visualization script (UPDATED)
└── WORKSPACE_DATA_SETUP_GUIDE.md  # 📚 This guide
```

---

## 🔧 **Configuration Management**

### **Primary Configuration File: `qlib_config.py`**

This module provides centralized configuration management:

```python
# Primary data path (workspace location)
DATA_PATH = "/workspace/qlib/data/qlib_data/cn_data"

# Backup data path (original location) 
BACKUP_DATA_PATH = "~/.qlib/qlib_data/cn_data"

# Output directory for analysis results
OUTPUT_DIR = "/workspace/qlib/qlib_analysis_output"
```

### **Key Features:**
- ✅ **Automatic Path Detection:** Falls back to backup if primary path unavailable
- ✅ **Health Monitoring:** Built-in data health checks
- ✅ **Easy Configuration:** Single file to modify all paths
- ✅ **Error Handling:** Graceful fallback and error reporting

---

## 🚀 **Updated Usage Examples**

### **1. Quick Configuration Check**
```bash
# Activate environment and check configuration
source /opt/miniforge3/etc/profile.d/conda.sh && conda activate qlib
python qlib_config.py
```

### **2. Run Workspace Data Analysis**
```bash
# Run comprehensive analysis with workspace data
python workspace_data_analysis.py
```

### **3. Use Configuration in Your Scripts**
```python
# Import configuration in your own scripts
from qlib_config import initialize_qlib, get_sample_stocks, ensure_output_dir

# Initialize Qlib (automatically uses correct path)
if initialize_qlib():
    # Get sample stocks
    stocks = get_sample_stocks('csi300', 5)
    
    # Ensure output directory exists
    output_dir = ensure_output_dir()
```

### **4. Manual Initialization**
```python
import qlib
from qlib.constant import REG_CN
from qlib.data import D

# Initialize with workspace data
qlib.init(provider_uri="/workspace/qlib/data/qlib_data/cn_data", region=REG_CN)

# Get stock data
data = D.features(
    instruments=['SZ000001'], 
    fields=['$open', '$high', '$low', '$close', '$volume'],
    start_time='2024-01-01', 
    end_time='2024-12-31'
)
```

---

## 📊 **Latest Analysis Results**

### **Data Overview (Updated):**
- **📅 Trading Days:** 5,024 days (2005-2025)
- **📈 Date Range:** 2005-01-04 to 2025-09-05
- **🏢 Total Stocks:** 5,516 instruments
- **📊 Data Size:** 706MB

### **Market Segments:**
| Index | Stocks Available | Description |
|-------|------------------|-------------|
| **CSI 300** | 336 | Large Cap (Blue-chip companies) |
| **CSI 500** | 645 | Mid Cap (Mid-sized companies) |
| **CSI 800** | 932 | Large + Mid Cap (Broad market) |
| **CSI 1000** | 1,310 | Small Cap (Small companies) |

### **Sample Stock Performance (Recent Analysis):**
| Stock | Avg Price | Daily Return | Volatility | Avg Volume | Data Points |
|-------|-----------|--------------|------------|------------|-------------|
| SZ000001 | 8.86 | 0.07% | 1.40% | 1.57M | 310 days |
| SZ000002 | 8.80 | -0.04% | 2.50% | 1.36M | 310 days |
| SZ000063 | 7.07 | 0.19% | 2.91% | 5.83M | 310 days |

---

## 🎯 **Benefits of Workspace Location**

### **✅ Advantages:**
1. **📁 Better Organization:** Data grouped with project files
2. **🔄 Portability:** Easy to backup/share entire workspace
3. **👥 Collaboration:** Clearer data location for team projects
4. **🚀 Deployment:** Simpler path management for containers/servers
5. **🔧 Maintenance:** Single location for all project assets

### **🔄 Backward Compatibility:**
- Original `~/.qlib/` data location still supported
- Scripts automatically fall back to home directory if workspace data unavailable
- No breaking changes to existing workflows

---

## 🛠️ **Script Updates Summary**

### **Updated Files:**
1. **`qlib_config.py`** - NEW: Configuration management module
2. **`workspace_data_analysis.py`** - NEW: Workspace-optimized analysis script
3. **`data_analysis_setup.py`** - UPDATED: Uses workspace path
4. **`create_visualizations.py`** - UPDATED: Uses workspace path

### **Path Changes:**
```python
# OLD PATH
mount_path = "~/.qlib/qlib_data/cn_data"

# NEW PATH  
mount_path = "/workspace/qlib/data/qlib_data/cn_data"
```

---

## 🏥 **Health Check Results**

Latest health check from workspace location:

```
✅ Status: healthy
📁 Data Path: /workspace/qlib/data/qlib_data/cn_data
📊 Calendar Days: Available
🏢 Sample Stocks: 300 (CSI 300)
📈 Sample Data: Successfully retrieved
🎯 Configuration: All systems operational
```

---

## 📈 **Performance & Statistics**

### **Data Access Performance:**
- ✅ **Initialization Time:** ~0.5 seconds
- ✅ **Sample Query Time:** ~0.1 seconds
- ✅ **Health Check Time:** ~1 second
- ✅ **Memory Usage:** Minimal overhead

### **Storage Information:**
- **📦 Raw Data Size:** 706MB
- **📁 Files Count:** 5,516+ stock folders + metadata
- **💾 Disk Usage:** Efficient binary format
- **⚡ Access Speed:** Fast local file system

---

## 🚀 **Next Steps & Advanced Usage**

### **1. Quantitative Research**
```python
from qlib_config import initialize_qlib
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH

# Initialize with workspace data
initialize_qlib()

# Create Alpha158 dataset
handler = Alpha158(
    start_time='2020-01-01',
    end_time='2024-12-31',
    instruments='csi300'  # Use our 336 large-cap stocks
)

dataset = DatasetH(
    handler=handler,
    segments={
        'train': ('2020-01-01', '2022-12-31'),
        'valid': ('2023-01-01', '2023-12-31'), 
        'test': ('2024-01-01', '2024-12-31')
    }
)
```

### **2. Strategy Development**
```python
from qlib.contrib.strategy import TopkDropoutStrategy

# Create strategy using workspace data
strategy = TopkDropoutStrategy(
    signal='<MODEL>',  # Your model predictions
    topk=50,           # Top 50 stocks from our universe
    n_drop=5           # Dynamic rebalancing
)
```

### **3. Backtesting**
```bash
# Run backtesting with workspace data
cd examples
qrun benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml
```

---

## 🔗 **File Access Quick Reference**

### **Configuration:**
- **Main Config:** `/workspace/qlib/qlib_config.py`
- **Data Location:** `/workspace/qlib/data/qlib_data/cn_data/`
- **Output Directory:** `/workspace/qlib/qlib_analysis_output/`

### **Analysis Scripts:**
- **Workspace Analysis:** `python workspace_data_analysis.py`
- **Configuration Check:** `python qlib_config.py`
- **Enhanced Visualizations:** `python create_visualizations.py`

### **Data Access:**
```python
# Quick data access example
from qlib_config import initialize_qlib
from qlib.data import D

initialize_qlib()
calendar = D.calendar(start_time='2024-01-01', end_time='2024-12-31')
instruments = D.instruments('csi300')
stocks = D.list_instruments(instruments, start_time='2024-01-01', end_time='2024-12-31')
```

---

## 🎉 **Migration Complete!**

**Your Qlib environment now features:**

- ✅ **706MB of Chinese stock data** in workspace location
- ✅ **5,516+ stocks** across all market segments  
- ✅ **20+ years** of historical data (2005-2025)
- ✅ **Updated analysis scripts** with new paths
- ✅ **Flexible configuration** with automatic fallback
- ✅ **Enhanced workspace organization**
- ✅ **Full backward compatibility** maintained

**🚀 Ready for advanced quantitative analysis with improved workspace organization!**

---

*Updated by Qlib Setup Assistant | 2025-09-06*  
*Data successfully migrated to workspace for better project organization*