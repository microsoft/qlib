# QLib项目部署文档

## 1. 文档概述

本文档详细描述了QLib项目在阿里云服务器上的部署过程，包括前端、后端和模型训练进程的部署，以及Nginx的配置和服务管理。

## 2. 环境要求

- 操作系统：CentOS 7或以上版本
- Python版本：3.12.2
- Node.js版本：20.13.0或以上版本
- Nginx版本：1.16.1或以上版本
- Supervisor版本：3.4.0或以上版本

## 3. 部署步骤

### 3.1 环境准备

1. 安装必要的系统依赖：

```bash
yum install -y gcc gcc-c++ make openssl-devel zlib-devel sqlite-devel
```

2. 安装Python 3.12.2：

```bash
# 下载Python 3.12.2源码包
wget https://www.python.org/ftp/python/3.12.2/Python-3.12.2.tgz

# 解压源码包
tar -zxvf Python-3.12.2.tgz

# 编译安装
cd Python-3.12.2
./configure --enable-optimizations
make -j4
make altinstall
```

3. 安装Node.js和npm：

```bash
# 安装Node.js 20
curl -fsSL https://rpm.nodesource.com/setup_20.x | bash -
yum install -y nodejs
```

4. 安装Nginx和Supervisor：

```bash
yum install -y nginx supervisor
```

### 3.2 代码获取

1. 克隆GitHub仓库：

```bash
git clone -b dev1201 https://github.com/zhostev/qlib_t.git /home/qlib_t
```

### 3.3 前端部署

1. 安装前端依赖：

```bash
cd /home/qlib_t/frontend
npm install --legacy-peer-deps
```

2. 构建前端项目：

```bash
npm run build
```

### 3.4 后端部署

1. 安装后端依赖：

```bash
cd /home/qlib_t/backend
/usr/local/bin/python3.12 -m pip install -r requirements.txt
```

2. 安装qlib模块：

```bash
/usr/local/bin/python3.12 -m pip install -e /home/qlib_t
```

3. 初始化数据库：

```bash
/usr/local/bin/python3.12 init_db.py
```

4. 创建管理员用户：

```bash
/usr/local/bin/python3.12 create_admin.py
```

### 3.5 配置Supervisor管理后端服务

1. 创建后端服务配置文件：

```bash
cat > /etc/supervisord.d/qlib_backend.ini << EOF
[program:qlib_backend]
command=/usr/local/bin/python3.12 -m uvicorn main:app --host 0.0.0.0 --port 8000
directory=/home/qlib_t/backend
autostart=true
autorestart=true
stderr_logfile=/var/log/qlib_backend.err.log
stdout_logfile=/var/log/qlib_backend.out.log
user=root
priority=999
numprocs=1
process_name=%(program_name)s_%(process_num)02d
EOF
```

2. 创建模型训练进程配置文件：

```bash
cat > /etc/supervisord.d/qlib_model_train.ini << EOF
[program:qlib_model_train]
command=/usr/local/bin/python3.12 -m app.tasks.task_worker
directory=/home/qlib_t/backend
autostart=true
autorestart=true
stderr_logfile=/var/log/qlib_model_train.err.log
stdout_logfile=/var/log/qlib_model_train.out.log
user=root
priority=999
numprocs=1
process_name=%(program_name)s_%(process_num)02d
EOF
```

3. 重启Supervisor服务：

```bash
systemctl restart supervisord
```

### 3.6 配置Nginx

1. 创建Nginx配置文件：

```bash
cat > /etc/nginx/conf.d/qlib.conf << EOF
server {
    listen 80;
    server_name 116.62.59.244;

    # 前端静态文件
    location / {
        root /home/qlib_t/frontend/dist;
        index index.html;
        try_files \$uri \$uri/ /index.html;
    }

    # 后端API反向代理
    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # 静态资源缓存
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
        root /home/qlib_t/frontend/dist;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
EOF
```

2. 检查Nginx配置：

```bash
nginx -t
```

3. 重启Nginx服务：

```bash
systemctl restart nginx
```

## 4. 服务管理

### 4.1 查看服务状态

1. 查看Supervisor管理的服务状态：

```bash
supervisorctl status
```

2. 查看Nginx服务状态：

```bash
systemctl status nginx
```

### 4.2 启动/停止服务

1. 启动/停止Supervisor管理的服务：

```bash
# 启动所有服务
supervisorctl start all

# 停止所有服务
supervisorctl stop all

# 启动指定服务
supervisorctl start qlib_backend

# 停止指定服务
supervisorctl stop qlib_backend
```

2. 启动/停止Nginx服务：

```bash
# 启动Nginx
systemctl start nginx

# 停止Nginx
systemctl stop nginx

# 重启Nginx
systemctl restart nginx
```

## 5. 故障排除

### 5.1 后端服务启动失败

1. 查看后端服务日志：

```bash
tail -n 50 /var/log/qlib_backend.err.log
```

2. 常见问题：
   - 缺少依赖包：使用`pip install`安装缺少的依赖包
   - 端口被占用：使用`lsof -i :8000`查看占用端口的进程，然后使用`kill`命令终止该进程
   - 配置错误：检查配置文件中的路径和参数是否正确

### 5.2 模型训练进程启动失败

1. 查看模型训练进程日志：

```bash
tail -n 50 /var/log/qlib_model_train.err.log
```

2. 常见问题：
   - 缺少qlib模块：使用`pip install -e /home/qlib_t`安装qlib模块
   - 配置错误：检查配置文件中的路径和参数是否正确

### 5.3 Nginx启动失败

1. 查看Nginx日志：

```bash
tail -n 50 /var/log/nginx/error.log
```

2. 检查Nginx配置：

```bash
nginx -t
```

3. 常见问题：
   - 配置文件语法错误：检查配置文件中的语法是否正确
   - 端口被占用：使用`lsof -i :80`查看占用端口的进程，然后使用`kill`命令终止该进程

## 6. 常见问题

### 6.1 如何访问前端页面？

前端页面可以通过以下URL访问：

```
http://116.62.59.244
```

### 6.2 如何访问后端API？

后端API可以通过以下URL访问：

```
http://116.62.59.244/api
```

### 6.3 如何查看服务日志？

1. 后端服务日志：

```bash
tail -n 50 /var/log/qlib_backend.err.log
tail -n 50 /var/log/qlib_backend.out.log
```

2. 模型训练进程日志：

```bash
tail -n 50 /var/log/qlib_model_train.err.log
tail -n 50 /var/log/qlib_model_train.out.log
```

3. Nginx日志：

```bash
tail -n 50 /var/log/nginx/error.log
tail -n 50 /var/log/nginx/access.log
```

### 6.4 如何更新代码？

1. 拉取最新代码：

```bash
cd /home/qlib_t
git pull origin dev1201
```

2. 重新构建前端：

```bash
cd /home/qlib_t/frontend
npm run build
```

3. 重启后端服务：

```bash
supervisorctl restart qlib_backend
```

4. 重启模型训练进程：

```bash
supervisorctl restart qlib_model_train
```

## 7. 安全建议

1. 定期更新系统和软件包：

```bash
yum update -y
```

2. 配置防火墙，只开放必要的端口：

```bash
# 开放80端口
firewall-cmd --zone=public --add-port=80/tcp --permanent

# 重新加载防火墙配置
firewall-cmd --reload
```

3. 使用HTTPS协议：
   - 申请SSL证书
   - 配置Nginx支持HTTPS

4. 定期备份数据：
   - 备份数据库
   - 备份配置文件
   - 备份日志文件

## 8. 联系方式

如果您在部署过程中遇到问题，请联系技术支持团队。

---

**文档版本**：v1.0
**文档日期**：2025-12-01
**文档作者**：QLib开发团队
