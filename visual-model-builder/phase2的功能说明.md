# Phase 2 功能说明

## 1. 文档目的

本文档用于说明当前 Phase 2 已实现的功能、关键行为约束、维护注意事项，以及后续更新时容易引入错误的地方。

目标不是重复 README，而是帮助后续开发在继续迭代时不破坏现有 Phase 2 闭环。

---

## 2. 当前已实现的 Phase 2 功能

### 2.1 训练图搭建

当前前端已经支持以下 Phase 2 训练节点：

- `Dataset`
- `DataLoader`
- `Loss`
- `Optimizer`
- `Trainer`
- `Metric`

当前推荐训练链路为：

`Dataset -> DataLoader -> Input -> ...模型主体... -> Output -> Loss`

并补充：

- `Optimizer -> Trainer`
- `Loss -> Trainer`
- `Metric -> Trainer`（可选）

相关前端位置：

- `frontend/src/registry/nodeRegistry.ts`
- `frontend/src/features/palette/`
- `frontend/src/features/canvas/`

### 2.2 训练图校验

后端已经区分：

- 普通模型图校验：`/validate-graph`
- 训练图校验：`/validate-training-graph`

不要把训练图和普通模型图走同一套校验逻辑，否则会出现：

- 带训练节点时模型代码生成被错误拦截
- 训练链路缺失却没有被正确报错

相关后端位置：

- `backend/app/api/__init__.py`
- `backend/app/services/validator/`

### 2.3 模型代码与训练代码生成

当前代码生成区已经分为两种模式：

- `Model`
- `Training`

它们的缓存是分开的，切换标签会显示各自最后一次生成结果，不需要重新生成。

重要约束：

- `Model` 模式只生成模型结构代码
- `Training` 模式生成训练脚本
- 两个模式不能共用一份 `generatedCode`

相关前端位置：

- `frontend/src/features/codegen/CodePanel.tsx`
- `frontend/src/store/useAppStore.ts`

相关后端位置：

- `backend/app/services/codegen/__init__.py`
- `backend/app/services/codegen/training.py`

### 2.4 训练运行

当前后端已经支持最小训练闭环：

- 构建运行时模型
- 构建运行时数据集
- 构建 DataLoader
- 构建 Loss
- 构建 Optimizer
- 执行训练循环
- 返回每个 epoch 的 `loss` 和 `accuracy`

当前支持：

- `FakeData`
- `MNIST`（本地不存在时自动回退到 `FakeData`）

注意：

- 后端启动时不会强制导入 `torch`
- 只有真正执行 `/run-training` 时才需要 `torch` 和 `torchvision`

相关后端位置：

- `backend/app/services/training.py`

### 2.5 训练结果展示

当前 `Training Results` 面板已经实现：

- `Curves / Status / Logs` 三视图
- 训练完成后自动切到 `Curves`
- `loss` 曲线
- `accuracy` 曲线
- 单 epoch 时显示点标记，不再出现空白图
- 日志表格展示
- 训练可运行性说明

相关前端位置：

- `frontend/src/features/training/TrainingPanelV2.tsx`
- `frontend/src/App.css`

### 2.6 训练结果保存

当前训练结束后已经支持保存结果。

后端会自动为每次训练创建一个运行目录，默认位置：

- `backend/training_runs/<run_id>/`

每次训练会保存：

- 模型权重：`model_weights.pt`
- 训练日志：`training_logs.json`
- 训练摘要：`training_summary.json`

前端 `Training Results` 面板可以额外导出一份 HTML 报告，包含：

- 曲线图
- logs
- 超参数
- 训练时间
- 实际使用数据
- 权重保存地址
- 日志保存地址
- 摘要保存地址

相关位置：

- `backend/app/services/training.py`
- `backend/app/schemas/responses.py`
- `frontend/src/features/training/TrainingPanelV2.tsx`

### 2.7 快捷键

当前已实现快捷键：

- `Ctrl/Cmd + C`：复制当前选中节点
- `Ctrl/Cmd + V`：粘贴节点
- `Ctrl/Cmd + Z`：撤回图编辑
- `Ctrl/Cmd + S`：保存项目

注意：

- 如果当前焦点在输入框中，快捷键不会强行接管浏览器默认行为
- 复制节点走的是内部剪贴板，不是系统剪贴板

相关位置：

- `frontend/src/App.tsx`
- `frontend/src/store/useAppStore.ts`

### 2.8 复制粘贴与撤回稳定性

当前已经修复：

- 第一 个组件添加后无法正常撤回
- 粘贴节点时与已有节点重叠，导致无法单独拖动或修改
- React Flow 内部运行时字段污染项目状态

当前行为：

- 只有真正的图编辑会进入撤回历史
- 画布内部同步不会污染撤回栈
- 粘贴节点会自动避开已有节点
- store 中会清洗 React Flow 运行时字段，只保留项目真实数据

相关位置：

- `frontend/src/store/useAppStore.ts`
- `frontend/src/features/canvas/Canvas.tsx`

---

## 3. 当前实现中的关键约束

### 3.1 `Flatten` 参数语义是“单样本语义”

这是 Phase 2 中非常容易被改坏的一点。

当前图编辑器里的 `Flatten.start_dim` 是基于单样本张量理解的，不是基于带 batch 的真实运行时张量。

例如：

- 图上样本形状：`[C, H, W]`
- 运行时真实张量：`[B, C, H, W]`

因此运行时和代码生成时，`Flatten.start_dim` 会自动加 1，以跳过 batch 维度。

如果后续有人把这段逻辑删掉或改回 PyTorch 原始含义，会直接导致：

- `mat1 and mat2 shapes cannot be multiplied`
- `Linear.in_features` 与前端 shape 推导不一致

必须保持一致的位置：

- `backend/app/services/training.py`
- `backend/app/services/codegen/__init__.py`

### 3.2 训练输出需要做分类维度规范化

当前某些图会得到类似：

- `[B, 10, 1]`

在进入 `CrossEntropyLoss` 前，后端会做 trailing singleton 维度压缩。

如果后续移除这步，会出现：

- `loss` 输入维度不匹配
- 某些原本可跑的 Phase 2 教学案例再次报错

关键位置：

- `backend/app/services/training.py`
- `backend/app/services/codegen/training.py`

### 3.3 `Model` 和 `Training` 代码缓存必须隔离

当前前端已经修复：

- `Model` 标签显示训练代码
- `Training` 标签显示模型代码
- 切换标签后必须再生成一次才对应

所以后续不要再把两类代码合并成一个共享字符串状态。

必须保持：

- `generatedCodeByMode.model`
- `generatedCodeByMode.training`

关键位置：

- `frontend/src/store/useAppStore.ts`
- `frontend/src/features/codegen/CodePanel.tsx`

### 3.4 React Flow 运行时字段不能写回持久化项目

当前 store 会对节点和边做清洗，只保存项目真正需要的字段：

- 节点：`id/type/position/data`
- 边：`id/source/target/sourceHandle/targetHandle`

不要把 React Flow 的运行时字段直接存回项目，例如：

- `selected`
- `dragging`
- 内部尺寸字段
- 运行时布局字段

否则会导致：

- 撤回行为不稳定
- 复制粘贴后多个节点像“粘”在一起
- JSON 保存内容不干净

关键位置：

- `frontend/src/store/useAppStore.ts`

### 3.5 训练结果保存依赖后端元信息

前端导出 HTML 报告时，不是自己猜测训练超参数和文件路径，而是依赖后端返回的 `trainingMetadata`。

因此后续如果修改 `/run-training` 返回结构，必须同步更新：

- `backend/app/schemas/responses.py`
- `frontend/src/types/index.ts`
- `frontend/src/features/training/TrainingPanelV2.tsx`

否则会出现：

- 前端导出报告缺字段
- 路径显示为空
- 权重保存地址不可信

---

## 4. 后续更新时的高风险点

### 4.1 修改节点类型或参数时

如果新增或修改节点，请同时检查：

- 前端节点注册：`frontend/src/registry/nodeRegistry.ts`
- 前端参数面板：`frontend/src/features/inspector/`
- 后端图校验：`backend/app/services/validator/`
- 后端 shape 推导：`backend/app/services/shape_infer/`
- 后端代码生成：`backend/app/services/codegen/`
- 后端运行时训练构建：`backend/app/services/training.py`

只改其中一端，极容易造成：

- 图能画但不能训练
- 代码能生成但训练不能跑
- shape 推导正确但运行时错误

### 4.2 修改保存/加载格式时

项目 JSON 当前版本是：

- `2.0.0`

保存/加载逻辑会对旧项目做兼容归一化。

如果后续调整：

- 节点字段
- 参数键名
- 训练节点结构
- `metadata`

请同步检查：

- `frontend/src/App.tsx`
- `frontend/src/graph/graphUtils`

否则旧项目文件可能无法打开，或者打开后 silently 出错。

### 4.3 修改训练结果面板时

当前实际导出和展示走的是：

- `TrainingPanelV2.tsx`

仓库中还保留了旧版：

- `frontend/src/features/training/TrainingPanel.tsx`

目前对外导出已经切换到：

- `frontend/src/features/training/index.ts -> TrainingPanelV2`

后续维护时不要误改旧文件后以为功能已经生效。

### 4.4 修改后端训练目录时

当前训练结果默认写入：

- `backend/training_runs/`

也可以通过环境变量覆盖：

- `VMB_TRAINING_RUNS_DIR`

如果改动该目录逻辑，请确保：

- 权重文件仍然可保存
- logs JSON 和 summary JSON 路径仍然可返回前端
- Windows 路径可正常显示

---

## 5. 建议的回归检查清单

每次更新 Phase 2 后，至少手动验证以下事项：

1. 训练图能通过 `/validate-training-graph`
2. `Model` 和 `Training` 两个标签能分别显示对应代码
3. `Run Training` 可以返回 `loss` 和 `accuracy`
4. `Curves` 视图能显示曲线或点
5. `Save Results` 能导出 HTML 报告
6. 后端 `training_runs` 目录能看到权重和 JSON 摘要
7. `Ctrl/Cmd + C / V / Z / S` 正常工作
8. 第一个节点添加后可立即撤回
9. 粘贴节点不会与已有节点完全重叠
10. 打开旧项目 JSON 时不会崩溃

建议命令：

```powershell
cd backend
pytest -q

cd ../frontend
npx tsc -b
npm run lint
```

---

## 6. 总结

Phase 2 当前已经不是“只展示训练曲线”的原型，而是一个完整的教学闭环：

- 可搭建训练图
- 可校验
- 可生成模型代码和训练代码
- 可真实运行训练
- 可保存权重和训练结果
- 可导出训练报告
- 可通过快捷键提升编辑效率

后续更新时，最重要的是保持以下三组一致性：

- 前端节点定义 <-> 后端校验/推导/代码生成
- 图上 shape 语义 <-> 运行时 batch 语义
- 前端展示/导出字段 <-> 后端返回字段

只要这三组一致性不被破坏，Phase 2 的整体稳定性就能维持住。
