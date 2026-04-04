# Visual Model Builder

面向机器学习初学者的可视化模型搭建教学工具。用户通过拖拽基础模块在画布中搭建神经网络结构，系统实时展示张量形状变化并自动生成 PyTorch 代码。

## 项目结构

```
visual-model-builder/
├── frontend/                  # React + TypeScript 前端
│   ├── src/
│   │   ├── types/             # 共享类型定义
│   │   ├── registry/          # 节点注册表
│   │   ├── store/             # Zustand 状态管理
│   │   ├── services/          # API 通信层
│   │   ├── graph/             # 图操作工具（拓扑排序等）
│   │   ├── features/
│   │   │   ├── palette/       # 左侧组件面板
│   │   │   ├── canvas/        # 中间画布（React Flow）
│   │   │   ├── inspector/     # 右侧属性面板
│   │   │   └── codegen/       # 代码展示面板
│   │   ├── App.tsx            # 主布局组件
│   │   └── App.css            # 全局样式
│   └── package.json
├── backend/                   # Python + FastAPI 后端
│   ├── app/
│   │   ├── main.py            # FastAPI 入口
│   │   ├── api/               # API 路由
│   │   ├── schemas/           # Pydantic 数据模型
│   │   │   ├── graph.py       # 图结构 schema
│   │   │   ├── requests.py    # 请求 schema
│   │   │   └── responses.py   # 响应 schema
│   │   ├── services/
│   │   │   ├── graph_ir/      # 中间表示 (IR)
│   │   │   ├── validator/     # 图校验器
│   │   │   ├── shape_infer/   # Shape 推导规则
│   │   │   └── codegen/       # PyTorch 代码生成
│   │   └── tests/             # 后端测试
│   └── requirements.txt
└── README.md
```

## 环境要求

- **Conda 环境**: `visual-model-builder` (Python 3.11)
- **Node.js**: >= 18.x
- **npm**: >= 9.x

## 快速启动

### 1. 激活 Conda 环境

```bash
conda activate visual-model-builder
```

### 2. 启动后端 (端口 8000)

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

后端启动后，可访问：
- API 文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/health

### 3. 启动前端 (端口 5173)

```bash
cd frontend
npm run dev
```

前端启动后，访问: http://localhost:5173

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/validate-graph` | 校验图结构 |
| POST | `/infer-shapes` | 推导各节点 shape |
| POST | `/generate-code` | 生成 PyTorch 代码 |

## 技术栈

### 前端
- React + TypeScript
- React Flow (@xyflow/react) — 节点画布
- Zustand — 状态管理
- Vite — 构建工具

### 后端
- Python 3.11 + FastAPI
- Pydantic v2 — 数据校验
- Uvicorn — ASGI 服务器

## 开发说明

- 节点类型通过 `frontend/src/registry/nodeRegistry.ts` 注册
- 后端 shape 规则通过装饰器 `@register_shape_rule("OpName")` 注册
- 后端代码生成规则通过 `@register_codegen_rule("OpName")` 注册
- 前后端通过 JSON schema 对齐，类型定义在 `frontend/src/types/` 和 `backend/app/schemas/`

## 运行测试

```bash
cd backend
conda run -n visual-model-builder pytest app/tests/ -v
```

## 后续规划

- Phase 1: CNN 结构教学版（当前）
- Phase 2: 训练流程教学版
- Phase 3: 自定义模块封装版
- Phase 4: 序列模型版 (RNN/LSTM/GRU)
- Phase 5: Transformer 教学版
