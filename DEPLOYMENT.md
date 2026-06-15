# 部署说明

本项目由两个部分组成：

- 前端：Vite 静态站点，目录为 `visual-model-builder/frontend`
- 后端：FastAPI 服务，目录为 `visual-model-builder/backend`

如果想通过公网链接访问项目，需要分别部署前端和后端，并让前端指向后端的公网 API 地址。

## 后端部署

后端建议以 Docker Web Service 的方式部署。项目已提供后端镜像配置：

```text
visual-model-builder/backend/Dockerfile
```

如果使用 Render Blueprint 部署，仓库根目录已经提供：

```text
render.yaml
```

后端部署完成后，需要设置允许访问后端的前端域名：

```text
VMB_CORS_ORIGINS=https://your-frontend-domain.example.com
```

如果需要临时支持预览域名，也可以设置：

```text
VMB_CORS_ORIGIN_REGEX=https://.*\.vercel\.app
```

部署完成后，可以用后端服务地址加 `/health` 检查 API 是否运行正常；用后端服务地址加 `/docs` 查看 FastAPI 接口文档。

## 前端部署

前端建议部署为 Vite 静态站点。部署目录为：

```text
visual-model-builder/frontend
```

构建前需要设置后端 API 地址：

```text
VITE_API_BASE_URL=https://your-backend-service.example.com
```

构建命令：

```text
npm run build
```

构建输出目录：

```text
dist
```

## 推荐部署顺序

1. 先部署后端，并确认 `/health` 可以正常访问。
2. 部署前端，并把 `VITE_API_BASE_URL` 设置为后端公网地址。
3. 将最终前端域名填入后端的 `VMB_CORS_ORIGINS`。
4. 修改 CORS 配置后，重新部署后端。

## 注意事项

- 不要提交本地数据集、模型权重或训练运行产物。
- MNIST 数据集会在后端首次使用时自动下载。
- 如果需要长时间 CPU 训练，建议选择资源更充足的后端实例。
