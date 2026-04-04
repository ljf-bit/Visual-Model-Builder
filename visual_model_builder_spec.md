# 可视化模型搭建教学工具：项目功能说明书与第一阶段需求文档

## 一、项目功能说明书（供代码生成工具 / 开发助手 / 协作开发者理解整体需求）

### 1. 项目名称

Visual Model Builder（暂定名）

### 2. 项目定位

这是一个面向机器学习 / 深度学习初学者的可视化教学工具。用户通过拖拽基础模块（如 Input、Conv2d、ReLU、MaxPool、Flatten、Linear、DataLoader、Loss、Optimizer 等），在可视化画布中搭建模型结构与训练流程。系统实时展示张量形状变化、参数配置、错误提示，并自动生成对应的 PyTorch 代码，帮助用户建立“图形结构—数学结构—代码实现”之间的对应关系。

该项目不是通用 AI 工作流编排器，也不是面向资深研究者的高自由度建模 IDE。它优先服务于教学、入门理解、结构演示、代码学习与实验启蒙。

### 3. 核心目标

1. 让初学者通过拖拽方式理解神经网络的组成结构。
2. 让用户能直观看到各层输入输出 shape 的变化。
3. 让用户能修改模块参数，并理解这些参数的作用。
4. 让用户能把多个基础模块组合成一个可命名的新模块。
5. 让用户能从图形结构自动得到可运行的 PyTorch 代码。
6. 让项目架构支持后续从 CNN 扩展到 RNN、LSTM、GRU、Transformer 等更复杂结构。

### 4. 非目标

以下内容不作为项目首要目标，至少不在初期版本中实现：

- 不追求与 ComfyUI 等工作流系统功能对齐。
- 不做通用大模型训练平台。
- 不做云端多用户协作平台。
- 不优先支持多框架（如 TensorFlow、JAX）同时导出。
- 不在首阶段支持复杂分支图、残差连接、动态图控制流。
- 不在首阶段处理集群调度、分布式训练、远程 GPU 编排。

### 5. 目标用户

1. 深度学习初学者
2. 需要进行课程演示的教师或助教
3. 想通过图形界面理解 PyTorch 结构的学习者
4. 需要快速构建教学示例的内容创作者

### 6. 使用场景

1. 用户拖入 Input、Conv2d、ReLU、MaxPool2d、Flatten、Linear 节点搭建一个简单 CNN。
2. 用户在右侧面板修改 Conv2d 的 out_channels、kernel_size、stride、padding。
3. 系统实时更新后续层的输出 shape。
4. 当 Linear 的 in_features 与上游 Flatten 输出不匹配时，系统给出明确错误信息。
5. 用户点击“生成代码”，系统输出对应的 PyTorch `nn.Module` 与 `forward()`。
6. 用户可将多个连续层封装成一个自定义模块并命名，例如 `FeatureExtractorBlock`。
7. 用户保存当前项目，后续继续打开编辑。

### 7. 产品原则

1. **教学优先**：解释性优于自由度。
2. **可视化优先**：结构、参数、shape 变化必须直观。
3. **低门槛优先**：术语可以保留，但界面必须给出解释。
4. **强约束优先**：前期限制图结构复杂度，以保证正确性与稳定性。
5. **代码映射清晰**：每个可视化模块都应能对应到清晰的代码片段。
6. **可扩展架构**：从第一阶段开始就为后续新增模块与新阶段预留扩展接口。

### 8. 系统范围

系统主要包含以下能力模块：

#### 8.1 可视化节点编辑器

- 节点拖拽到画布
- 节点之间连线
- 选中节点查看详情
- 删除节点/边
- 节点位置调整
- 自动保存/手动保存图结构

#### 8.2 参数编辑系统

- 根据节点类型动态展示参数表单
- 支持数字、布尔、枚举、shape 等类型参数
- 对参数值进行基础校验
- 修改后实时触发图更新与 shape 重推导

#### 8.3 图结构校验系统

- 检查图是否有 Input 和 Output
- 检查是否存在环
- 检查是否存在孤立节点
- 检查连接是否合法
- 检查层参数与输入 shape 是否兼容

#### 8.4 Shape 推导系统

- 对每个节点进行输入输出张量 shape 推导
- 将推导结果展示在节点、边或详情面板中
- 在参数变化时联动更新
- 当推导失败时给出明确原因

#### 8.5 代码生成系统

- 从图结构生成 PyTorch 模型代码
- 生成 `__init__` 与 `forward()`
- 保留节点命名与模块命名
- 后续可扩展到训练代码生成

#### 8.6 子模块封装系统（后续阶段）

- 用户选中若干节点
- 系统判断是否能封装为子图
- 用户为新模块命名
- 新模块可保存并复用

#### 8.7 项目持久化系统

- 保存为项目文件
- 打开历史项目
- 项目版本号管理
- 后续支持 schema 迁移

### 9. 总体技术架构要求

#### 9.1 前端

- React + TypeScript
- 节点画布基于 React Flow 或同类库实现
- 使用状态管理工具维护图状态、选中状态、推导结果、错误状态
- 支持节点模板注册机制

#### 9.2 后端

- Python + FastAPI（或其他轻量接口层）
- 使用 PyTorch 作为代码生成与运行目标框架
- 负责更严格的图校验、shape 推导、代码生成、训练执行

#### 9.3 桌面封装

- 优先考虑 Tauri
- 允许先做 Web MVP，再封装为桌面应用

#### 9.4 存储

- 早期可用 JSON 项目文件
- 后续可增加 SQLite 存储模板、项目索引、历史记录

### 10. 架构约束

1. 前端节点模型不能直接耦合 PyTorch 原始代码字符串。
2. 必须有独立的中间表示层（IR）。
3. shape 推导逻辑不能散落在 UI 组件内部，应抽离为独立规则层。
4. 代码生成不能通过简单字符串拼接硬编码在页面逻辑里，应通过节点类型分发机制生成。
5. 项目文件必须带版本号，便于未来迁移。
6. 模块应通过注册表扩展，而不是到处写 if-else。

### 11. 建议的领域模型

#### 11.1 节点模板

用于定义“某类节点是什么、有哪些参数、如何推导 shape、如何生成代码”。

建议字段：

- type
- displayName
- category
- inputs
- outputs
- params
- validator
- shapeInfer
- codegen

#### 11.2 项目图

用于定义用户当前在画布上构建的图。

建议字段：

- version
- metadata
- nodes
- edges

#### 11.3 中间表示（IR）

用于前后端共享的可解释结构，作为图校验、shape 推导、代码生成的统一输入。

建议字段：

- nodeId
- op
- params
- inputRefs
- outputRefs
- inferredInputShape
- inferredOutputShape
- errors

### 12. 未来阶段规划

#### Phase 1：CNN 结构教学版

聚焦基础层搭建、shape 推导、PyTorch 模型代码生成。

#### Phase 2：训练流程教学版

增加 Dataset、DataLoader、Loss、Optimizer、Trainer、Metrics 等节点。

#### Phase 3：自定义模块封装版

支持将多个节点封装为具名模块并复用。

#### Phase 4：序列模型版

支持 RNN、LSTM、GRU、Embedding、序列输入 shape 可视化。

#### Phase 5：Transformer 教学版

支持 Multi-Head Attention、LayerNorm、Residual、Positional Encoding 等。

### 13. 典型用户流程

1. 新建项目。
2. 从左侧组件栏拖入 Input 节点。
3. 继续拖入 Conv2d、ReLU、MaxPool2d、Flatten、Linear、Output。
4. 将它们按顺序连线。
5. 在属性面板中编辑参数。
6. 查看每层的输入输出 shape。
7. 如果有不匹配，查看系统错误提示并修正。
8. 点击“生成代码”。
9. 复制或导出 PyTorch 代码。
10. 保存项目文件。

### 14. 成功标准

1. 初学者能在不手写代码的情况下搭建一个简单 CNN。
2. 用户能看懂每个基础节点的用途、参数与 shape 变化。
3. 系统能稳定生成正确的 PyTorch 模型代码。
4. 项目结构足以支持后续新增节点而不重写核心架构。

---

## 二、第一阶段需求文档（Phase 1：CNN 可视化教学版）

### 1. 阶段名称

Phase 1：CNN 可视化教学版

### 2. 阶段目标

实现一个最小可用产品（MVP），让用户能够通过拖拽基础 CNN 组件，在画布上搭建简单前馈网络，配置参数，查看 shape 变化，接收错误提示，并生成对应的 PyTorch 模型代码。

### 3. 本阶段范围

本阶段仅支持简单的顺序型 CNN 模型结构，不支持复杂拓扑。

### 4. 本阶段支持的节点

#### 4.1 输入输出类

- Input
- Output

#### 4.2 网络层类

- Conv2d
- ReLU
- MaxPool2d
- Flatten
- Linear
- Dropout（可选，若开发资源允许）

### 5. 本阶段不支持内容

- RNN / LSTM / GRU
- Transformer
- 多输入多输出图
- 残差连接
- 分支合并
- 自定义模块封装
- Dataset / DataLoader / Loss / Optimizer / Trainer 节点
- 云端同步
- 多用户协作
- 插件系统

### 6. 图结构限制

1. 图必须为有向无环图（DAG）。
2. 初期只支持单一主链路，推荐按顺序连接。
3. 每个普通层节点默认只有一个输入和一个输出。
4. Input 节点不能有输入边。
5. Output 节点不能有输出边。
6. 图中必须至少有一个 Input 和一个 Output。
7. 不允许存在未连接的核心计算节点。

### 7. 用户可见功能需求

#### 7.1 项目管理

- 新建项目
- 保存项目
- 打开项目
- 项目另存为

#### 7.2 组件面板

- 左侧显示可拖拽基础节点
- 节点按类别分组显示
- 每个节点显示简要名称
- 可后续扩展节点说明文字

#### 7.3 画布编辑

- 拖拽节点到画布
- 节点之间连线
- 删除节点
- 删除边
- 移动节点位置
- 缩放与平移画布
- 选中节点显示当前参数与推导信息

#### 7.4 参数编辑面板

当用户选中节点时，右侧显示参数面板。

不同节点的参数如下：

##### Input

- inputShape，例如 `[1, 28, 28]`

##### Conv2d

- in_channels
- out_channels
- kernel_size
- stride
- padding
- dilation（可选）
- bias（可选）

##### ReLU

- inplace（可选）

##### MaxPool2d

- kernel_size
- stride
- padding

##### Flatten

- start_dim
- end_dim

##### Linear

- in_features
- out_features
- bias

##### Output

- 无核心参数，展示最终输出 shape

#### 7.5 Shape 实时推导

- 每个节点必须显示输入 shape 与输出 shape
- 当用户修改参数时，应自动触发后续节点重推导
- 如果某层推导失败，后续依赖节点应显示“待定”或“推导失败”状态
- 推导错误必须指向具体节点

#### 7.6 错误提示

系统至少应支持以下错误提示：

- 缺少 Input
- 缺少 Output
- 图中存在环
- 节点未连接
- Conv2d 的 in_channels 与上游输出通道数不匹配
- Linear 的 in_features 与上游 Flatten 输出大小不匹配
- kernel_size / stride / padding 参数非法
- 经过卷积或池化后 shape 小于等于 0
- 输入 shape 格式非法

错误提示要求：

1. 直接说明错误位置
2. 说明期望值与实际值
3. 尽量给出可操作的修复方向

示例：

- “节点 Linear_1 的 `in_features=1024`，但上游 Flatten 输出为 3136。”
- “节点 Conv2d_2 的 `in_channels=64`，但输入张量通道数为 32。”

#### 7.7 代码生成

用户点击“生成代码”后，系统应输出：

- PyTorch `import` 代码
- `class Model(nn.Module):`
- `__init__` 中各层定义
- `forward()` 中前向传播逻辑

要求：

1. 代码风格尽量清晰、易读。
2. 节点命名应映射为稳定的层变量名。
3. 代码与当前图结构保持一致。
4. 当图存在关键错误时，不允许生成最终代码，或应明确标注代码不可运行。

#### 7.8 代码展示区

- 展示自动生成的代码
- 支持复制
- 支持导出 `.py` 文件（可选）

### 8. 非功能需求

#### 8.1 可维护性

- 节点定义通过注册表维护
- shape 推导逻辑独立
- 代码生成逻辑独立
- 项目 schema 明确、可版本化

#### 8.2 可扩展性

- 后续新增 RNN、Transformer 节点时，不应重写整体画布系统
- 节点模板、shape 规则、codegen 规则可独立增加

#### 8.3 易用性

- 初学者无需知道全部 PyTorch 细节即可使用
- 界面术语可配说明
- 关键错误应可读，不应只返回技术异常

#### 8.4 性能

- 普通小图（10~30 个节点）编辑响应流畅
- 参数变更后的 shape 重推导延迟应尽量低

### 9. 建议的数据结构

#### 9.1 节点实例

```ts
type GraphNode = {
  id: string;
  type: string;
  position: { x: number; y: number };
  data: {
    label: string;
    params: Record<string, any>;
    inferredInputShape?: number[] | null;
    inferredOutputShape?: number[] | null;
    errors?: string[];
  };
};
```

#### 9.2 边实例

```ts
type GraphEdge = {
  id: string;
  source: string;
  target: string;
  sourceHandle?: string;
  targetHandle?: string;
};
```

#### 9.3 项目文件

```ts
type ProjectGraph = {
  version: string;
  metadata: {
    name: string;
    createdAt: string;
    updatedAt: string;
  };
  nodes: GraphNode[];
  edges: GraphEdge[];
};
```

### 10. 建议的内部模块划分

#### 10.1 前端模块

- `components/`：通用 UI
- `nodes/`：节点渲染组件
- `panels/`：左侧组件栏、右侧属性栏、代码面板
- `store/`：状态管理
- `registry/`：节点模板注册
- `graph/`：图操作与拓扑排序
- `shape/`：前端轻量 shape 推导或展示逻辑

#### 10.2 后端模块

- `schemas/`：请求响应与项目 schema
- `graph_ir/`：中间表示
- `validator/`：图校验
- `shape_infer/`：shape 推导规则
- `codegen/`：PyTorch 代码生成

### 11. 验收标准

当满足以下条件时，可认为第一阶段完成：

1. 用户能拖拽并连接以下节点：Input、Conv2d、ReLU、MaxPool2d、Flatten、Linear、Output。
2. 用户能在右侧面板修改各节点参数。
3. 系统能对一个典型 CNN 结构正确计算各层 shape。
4. 当参数不匹配时，系统能给出可读错误信息。
5. 系统能生成与图结构一致的 PyTorch `nn.Module` 代码。
6. 用户能保存项目并重新打开。

### 12. 推荐的典型测试样例

#### 12.1 正常样例

输入：`[1, 28, 28]`
结构：

- Conv2d(1, 16, 3, 1, 1)
- ReLU
- MaxPool2d(2, 2)
- Flatten
- Linear(16*14*14, 10)

预期：

- shape 推导正确
- 成功生成 PyTorch 代码

#### 12.2 错误样例 1

Linear.in_features 填写错误。
预期：

- 报告 in_features 与 Flatten 输出不匹配

#### 12.3 错误样例 2

Conv2d.in_channels 与输入通道不一致。
预期：

- 报告通道数不匹配

#### 12.4 错误样例 3

卷积参数导致输出空间尺寸小于等于 0。
预期：

- 报告卷积后 shape 非法

### 13. 开发优先级建议

#### P0（必须）

- 基础节点拖拽
- 连线
- 参数编辑
- shape 推导
- 错误提示
- PyTorch 模型代码生成
- 项目保存/加载

#### P1（建议）

- 节点说明提示
- 代码复制按钮
- 画布布局优化
- 导出 `.py`

#### P2（可后置）

- Dropout 节点
- 节点搜索
- 示例项目模板

### 14. 对代码生成工具的实现要求

若该文档被用于驱动代码生成工具或开发代理，生成内容应遵循以下要求：

1. 优先保证架构清晰，而不是一次性堆叠过多功能。
2. 所有节点类型应通过注册表统一管理。
3. shape 推导逻辑必须独立模块化。
4. 代码生成逻辑必须能基于图的拓扑顺序生成。
5. 项目中应有清晰的类型定义与注释。
6. 第一阶段不要提前写死后续 RNN/Transformer 逻辑，但要预留扩展接口。

---

## 三、给开发工具的简短任务摘要

请基于上述说明，优先实现一个面向初学者的可视化 CNN 模型搭建工具 MVP。第一阶段只支持顺序型 CNN 图结构，具备节点拖拽、参数编辑、shape 推导、错误提示、PyTorch 模型代码生成、项目保存与加载能力。实现时应采用可扩展架构，为未来增加训练流程节点、自定义模块封装、RNN、Transformer 等功能预留扩展能力。

---

## 四、技术设计文档（Phase 1 实现导向）

### 1. 技术设计目标

本设计文档用于指导第一阶段 MVP 的工程实现。重点不是追求功能堆叠，而是建立一套可扩展、可维护、便于后续迭代的实现结构。

设计目标如下：

1. 前端画布、节点定义、参数面板、shape 推导展示解耦。
2. 后端图校验、shape 推导、代码生成解耦。
3. 前后端通过统一 schema / IR 通信。
4. 节点能力通过注册表扩展，而不是分散硬编码。
5. 第一阶段代码结构可直接承接第二阶段与后续扩展。

### 2. 建议技术选型

#### 2.1 前端

- React
- TypeScript
- React Flow
- Zustand
- Tailwind CSS
- Monaco Editor（代码展示，可选）
- Zod（前端表单与 schema 校验，可选）

#### 2.2 后端

- Python 3.11+
- FastAPI
- Pydantic
- PyTorch
- Uvicorn

#### 2.3 桌面打包（后置）

- Tauri

#### 2.4 存储

- 首阶段：JSON 项目文件
- 后续：SQLite

### 3. 推荐仓库结构

```txt
visual-model-builder/
  frontend/
    src/
      app/
      components/
      features/
        canvas/
        inspector/
        palette/
        codegen/
        project/
      nodes/
      registry/
      store/
      graph/
      services/
      types/
      utils/
  backend/
    app/
      main.py
      api/
      schemas/
      services/
        graph_ir/
        validator/
        shape_infer/
        codegen/
      templates/
      tests/
  docs/
    spec.md
    phase1-requirements.md
    technical-design.md
    task-breakdown.md
```

### 4. 系统模块设计

#### 4.1 前端模块说明

##### 4.1.1 Palette（组件面板）

职责：

- 展示可用节点模板
- 按类别分组
- 支持拖入画布

输入：节点模板注册表
输出：创建节点事件

##### 4.1.2 Canvas（画布）

职责：

- 节点渲染
- 边渲染
- 拖拽、连线、删除
- 节点选中
- 缩放、平移

输入：图状态
输出：图编辑事件

##### 4.1.3 Inspector（属性面板）

职责：

- 展示选中节点参数
- 编辑节点参数
- 展示节点当前 shape
- 展示节点错误

输入：选中节点、节点模板
输出：参数更新事件

##### 4.1.4 Code Panel（代码面板）

职责：

- 展示生成代码
- 支持复制
- 后续支持导出

输入：后端返回代码
输出：复制 / 导出事件

##### 4.1.5 Project Service（项目管理）

职责：

- 新建项目
- 保存 JSON
- 加载 JSON
- 管理项目元数据

##### 4.1.6 Registry（节点注册表）

职责：

- 集中声明节点类型
- 集中声明参数 schema
- 集中声明 UI 展示信息
- 后续可挂接 shape 规则和 codegen 规则标识

#### 4.2 后端模块说明

##### 4.2.1 API 层

职责：

- 接收项目图
- 返回校验结果
- 返回 shape 推导结果
- 返回代码生成结果

建议接口：

- `POST /validate-graph`
- `POST /infer-shapes`
- `POST /generate-code`
- `POST /project/normalize`（可选）

##### 4.2.2 Graph IR 层

职责：

- 将前端项目图转换为统一 IR
- 提供拓扑排序
- 提供节点查找和邻接关系查询

##### 4.2.3 Validator（图校验）

职责：

- 检查拓扑合法性
- 检查输入输出节点
- 检查连接数量
- 检查参数基础合法性

##### 4.2.4 Shape Infer（shape 推导）

职责：

- 对拓扑序中的每个节点执行规则推导
- 记录每个节点输入 / 输出 shape
- 输出可读错误

##### 4.2.5 Code Generator（代码生成）

职责：

- 根据 IR 和拓扑顺序生成 `nn.Module`
- 生成 `__init__` 中的层定义
- 生成 `forward()` 前向逻辑
- 生成 import 代码

### 5. 核心数据结构设计

#### 5.1 前端节点模板

```ts
type ParamType = "int" | "float" | "bool" | "select" | "shape";

type ParamSpec = {
  key: string;
  label: string;
  type: ParamType;
  required: boolean;
  defaultValue: any;
  helpText?: string;
  options?: string[];
};

type NodeTemplate = {
  type: string;
  displayName: string;
  category: "io" | "layer" | "train";
  description?: string;
  inputPorts: number;
  outputPorts: number;
  params: ParamSpec[];
};
```

#### 5.2 前端项目图

```ts
type GraphNode = {
  id: string;
  type: string;
  position: { x: number; y: number };
  data: {
    label: string;
    params: Record<string, any>;
    inferredInputShape?: number[] | null;
    inferredOutputShape?: number[] | null;
    errors?: string[];
  };
};

type GraphEdge = {
  id: string;
  source: string;
  target: string;
  sourceHandle?: string;
  targetHandle?: string;
};

type ProjectGraph = {
  version: string;
  metadata: {
    name: string;
    createdAt: string;
    updatedAt: string;
  };
  nodes: GraphNode[];
  edges: GraphEdge[];
};
```

#### 5.3 后端 IR

```py
from pydantic import BaseModel
from typing import Any

class IRNode(BaseModel):
    node_id: str
    op: str
    params: dict[str, Any]
    input_refs: list[str]
    output_refs: list[str]
    inferred_input_shape: list[int] | None = None
    inferred_output_shape: list[int] | None = None
    errors: list[str] = []

class IRGraph(BaseModel):
    version: str
    nodes: list[IRNode]
    edges: list[dict[str, str]]
```

### 6. 前后端数据流

#### 6.1 基本编辑流

1. 用户拖入节点
2. 前端生成节点实例并写入 store
3. 用户连线
4. 前端更新图状态
5. 用户修改参数
6. 前端将当前图发送给后端
7. 后端返回校验结果与 shape 推导结果
8. 前端更新节点展示

#### 6.2 代码生成流

1. 用户点击生成代码
2. 前端发送当前图到 `/generate-code`
3. 后端先执行校验与 shape 推导
4. 如果存在阻断性错误，则返回错误列表
5. 如果通过，则返回生成代码文本
6. 前端展示代码

### 7. 状态管理建议

建议在前端 store 中管理以下状态：

```ts
type AppState = {
  project: ProjectGraph;
  selectedNodeId: string | null;
  generatedCode: string;
  globalErrors: string[];
  isDirty: boolean;
  setProject: (project: ProjectGraph) => void;
  updateNodeParams: (nodeId: string, params: Record<string, any>) => void;
  setGeneratedCode: (code: string) => void;
  setSelectedNodeId: (nodeId: string | null) => void;
};
```

### 8. 节点注册机制设计

推荐每种节点在注册表中拥有独立配置对象：

```ts
type NodeBehavior = {
  template: NodeTemplate;
  defaultData: () => Record<string, any>;
};

const nodeRegistry: Record<string, NodeBehavior> = {
  Input: ...,
  Conv2d: ...,
  ReLU: ...,
  MaxPool2d: ...,
  Flatten: ...,
  Linear: ...,
  Output: ...,
};
```

后续可扩展为：

- `frontendRenderer`
- `validatorKey`
- `shapeInferKey`
- `codegenKey`

### 9. Shape 推导规则设计

#### 9.1 规则设计原则

1. 每个节点类型拥有独立推导函数。
2. 推导函数只处理本节点逻辑，不承担全局图控制。
3. 推导结果应包含成功 / 失败状态。
4. 出错时返回可读错误，而不是抛出未处理异常。

#### 9.2 规则接口建议

```py
class ShapeResult(BaseModel):
    ok: bool
    input_shape: list[int] | None = None
    output_shape: list[int] | None = None
    errors: list[str] = []
```

```py
def infer_conv2d(input_shape: list[int], params: dict) -> ShapeResult:
    ...
```

#### 9.3 Phase 1 节点规则

- Input：直接提供 output shape
- Conv2d：校验通道数并计算空间尺寸
- ReLU：输出 shape 与输入相同
- MaxPool2d：计算空间尺寸
- Flatten：根据 `start_dim/end_dim` 折叠维度
- Linear：校验输入特征长度并输出 `[out_features]`
- Output：直接透传

### 10. 代码生成设计

#### 10.1 代码生成原则

1. 先拓扑排序，再生成代码。
2. `__init__` 和 `forward()` 分开生成。
3. 节点变量名必须稳定、可预测。
4. 避免生成难读的嵌套表达式。

#### 10.2 代码生成步骤

1. 将前端图转为 IR
2. 执行校验和 shape 推导
3. 为每个层节点分配变量名，例如 `self.conv2d_1`
4. 生成 `__init__`
5. 生成 `forward()`，按拓扑顺序依次更新 `x`
6. 拼装完整代码字符串

#### 10.3 Phase 1 简化策略

首阶段只支持主链路，可直接采用：

```py
x = self.conv2d_1(x)
x = self.relu_1(x)
x = self.maxpool2d_1(x)
...
return x
```

### 11. 错误模型设计

建议区分两类错误：

#### 11.1 全局错误

例如：

- 图中缺少 Input
- 图中缺少 Output
- 图存在环

#### 11.2 节点错误

例如：

- Conv2d 通道不匹配
- Linear 输入特征不匹配
- 参数非法

前端应支持同时展示：

- 顶部全局错误列表
- 节点局部错误标记

### 12. API 设计草案

#### 12.1 校验接口

`POST /validate-graph`

请求：

```json
{
  "project": { ... }
}
```

响应：

```json
{
  "ok": false,
  "globalErrors": ["缺少 Output 节点"],
  "nodeErrors": {
    "node-3": ["in_features 与输入不匹配"]
  }
}
```

#### 12.2 shape 推导接口

`POST /infer-shapes`

响应：

```json
{
  "ok": true,
  "nodes": {
    "node-1": {
      "inputShape": null,
      "outputShape": [1, 28, 28],
      "errors": []
    }
  }
}
```

#### 12.3 代码生成接口

`POST /generate-code`

响应：

```json
{
  "ok": true,
  "code": "import torch ...",
  "errors": []
}
```

### 13. 测试策略

#### 13.1 前端测试

- 节点创建
- 连线创建与删除
- 参数面板更新
- 保存 / 加载 JSON

#### 13.2 后端测试

- 拓扑排序
- 环检测
- Conv2d shape 推导
- MaxPool2d shape 推导
- Flatten 推导
- Linear 校验
- 简单 CNN 代码生成

#### 13.3 集成测试

- 前端构图 -> 后端校验 -> 返回 shape -> 前端渲染
- 前端构图 -> 代码生成 -> 展示

### 14. 里程碑建议

#### Milestone 1

- 前端画布可操作
- 节点拖拽、连线、删除
- 项目可保存/加载

#### Milestone 2

- 参数面板完成
- 后端图校验完成
- shape 推导闭环完成

#### Milestone 3

- 代码生成完成
- 基础错误提示优化
- 形成可演示 MVP

---

## 五、任务拆解清单（可直接分配给开发代理 / 工程成员）

### 1. 总体开发目标

在 Phase 1 中交付一个可运行的 CNN 可视化教学版 MVP，具备以下核心能力：

- 节点拖拽
- 连线编辑
- 参数面板
- shape 推导
- 错误提示
- PyTorch 模型代码生成
- 项目保存与加载

### 2. 任务拆解原则

1. 先做可用主链路，再做体验优化。
2. 优先实现架构骨架与类型定义。
3. 先闭环基础功能，再考虑增强项。
4. 每个任务尽量有明确输入输出与完成标准。

### 3. 前端任务清单

#### FE-001 初始化前端工程

- 使用 React + TypeScript 初始化项目
- 集成 React Flow
- 集成 Zustand
- 配置基础目录结构

交付标准：

- 项目可启动
- 画布页面可显示

#### FE-002 建立基础布局

- 左侧组件面板
- 中间画布
- 右侧属性面板
- 底部或右下代码面板

交付标准：

- 页面布局稳定
- 各区域可正常渲染占位内容

#### FE-003 定义前端类型与项目 schema

- 定义 `NodeTemplate`
- 定义 `GraphNode`
- 定义 `GraphEdge`
- 定义 `ProjectGraph`

交付标准：

- 类型文件可复用
- 节点创建使用统一类型

#### FE-004 实现节点注册表

- 注册 Input
- 注册 Conv2d
- 注册 ReLU
- 注册 MaxPool2d
- 注册 Flatten
- 注册 Linear
- 注册 Output

交付标准：

- 组件面板能根据注册表渲染节点列表

#### FE-005 实现拖拽创建节点

- 从组件面板拖入画布
- 自动生成唯一 node id
- 自动写入默认参数

交付标准：

- 用户能成功拖入任意基础节点

#### FE-006 实现节点连线与删除

- 创建边
- 删除边
- 删除节点时同步清理边

交付标准：

- 图编辑行为稳定

#### FE-007 实现选中态与属性面板

- 点击节点显示参数
- 根据节点类型渲染动态表单

交付标准：

- 选中不同节点时参数面板正确切换

#### FE-008 实现参数更新逻辑

- 修改参数后更新 store
- 标记项目 dirty 状态
- 触发后端推导请求

交付标准：

- 参数变更可持久反映到图状态

#### FE-009 显示 shape 与错误信息

- 节点展示 input/output shape
- 节点展示错误标识
- 顶部或侧边展示全局错误

交付标准：

- 错误和 shape 能同步显示

#### FE-010 实现项目保存/加载

- 导出为 JSON
- 导入 JSON
- 恢复节点和边

交付标准：

- 保存后可完整恢复项目

#### FE-011 实现代码面板

- 展示后端返回代码
- 复制代码

交付标准：

- 用户能查看和复制生成代码

### 4. 后端任务清单

#### BE-001 初始化后端工程

- 使用 FastAPI 初始化服务
- 配置基础目录结构
- 接入 Pydantic

交付标准：

- 服务可启动
- 提供健康检查接口

#### BE-002 定义后端 schema

- ProjectGraph schema
- API 请求响应 schema
- IR schema

交付标准：

- API 有统一数据模型

#### BE-003 实现项目图转 IR

- 将节点与边转换为内部结构
- 提供邻接表和反向索引

交付标准：

- 后续校验和 codegen 使用统一 IR

#### BE-004 实现图校验器

- 检查 Input / Output 存在
- 检查环
- 检查孤立节点
- 检查连接合法性

交付标准：

- 返回全局错误与节点错误

#### BE-005 实现拓扑排序

- 基于 DAG 输出有序节点列表

交付标准：

- 顺序图能正确排序
- 有环图能报错

#### BE-006 实现 shape 推导：Input

- 解析输入 shape
- 输出 shape

#### BE-007 实现 shape 推导：Conv2d

- 校验通道数
- 计算输出 H/W
- 检查非法尺寸

#### BE-008 实现 shape 推导：ReLU

- 透传 shape

#### BE-009 实现 shape 推导：MaxPool2d

- 计算输出 H/W
- 检查非法尺寸

#### BE-010 实现 shape 推导：Flatten

- 实现维度折叠

#### BE-011 实现 shape 推导：Linear

- 校验输入特征数
- 输出 `[out_features]`

#### BE-012 实现 shape 推导：Output

- 透传 shape

交付标准（BE-006 ~ BE-012）：

- 能对标准 CNN 主链路完成完整 shape 推导

#### BE-013 实现 `/validate-graph`

- 返回图级错误和节点级错误

#### BE-014 实现 `/infer-shapes`

- 返回每个节点的 input/output shape 和错误

#### BE-015 实现代码生成器

- 生成 import
- 生成 `class Model(nn.Module)`
- 生成 `__init__`
- 生成 `forward()`

交付标准：

- 标准 CNN 图可生成可读 PyTorch 模型代码

#### BE-016 实现 `/generate-code`

- 先校验
- 再推导
- 最后输出代码

#### BE-017 编写后端单元测试

- shape 规则测试
- 图校验测试
- codegen 测试

### 5. 联调任务清单

#### IN-001 前后端 schema 对齐

- 确认字段命名一致
- 确认 shape 字段格式一致

#### IN-002 参数修改触发推导

- 前端发送图
- 后端返回结果
- 前端更新节点展示

#### IN-003 代码生成联调

- 前端点击按钮
- 后端返回代码
- 前端展示结果

#### IN-004 错误流联调

- 制造错误样例
- 确认前后端展示一致

### 6. 建议开发顺序

#### 第 1 周

- FE-001 ~ FE-006
- BE-001 ~ BE-005

目标：

- 完成可编辑画布与后端图骨架

#### 第 2 周

- FE-007 ~ FE-009
- BE-006 ~ BE-014
- IN-001 ~ IN-002

目标：

- 打通参数编辑与 shape 推导

#### 第 3 周

- FE-010 ~ FE-011
- BE-015 ~ BE-017
- IN-003 ~ IN-004

目标：

- 打通保存 / 加载 / 代码生成 / 测试

### 7. 每项任务完成定义（Definition of Done）

每项任务完成时应满足：

1. 功能可运行
2. 类型定义齐全
3. 基础异常有处理
4. 有最少必要测试或人工验证说明
5. 不破坏既有主链路

### 8. Phase 1 最终交付物

1. 可运行的前端应用
2. 可运行的后端服务
3. 基础文档：
   - 项目功能说明书
   - 第一阶段需求文档
   - 技术设计文档
   - 任务拆解清单
4. 一个标准 CNN 示例项目
5. 一份自动生成的示例 PyTorch 模型代码

### 9. 给开发代理的直接执行说明

请按以下优先顺序实现：

1. 定义统一 schema 和节点注册表。
2. 完成画布主链路：拖拽、连线、删除、选中。
3. 完成参数面板和图状态管理。
4. 完成后端图校验与 shape 推导。
5. 完成错误展示。
6. 完成 PyTorch 模型代码生成。
7. 完成项目保存与加载。

实现过程中应保持模块解耦，不要将 shape 推导、参数定义、代码生成硬编码在单一页面组件内部。后续阶段会增加训练流程节点、自定义模块封装、RNN、Transformer 等能力，因此当前实现必须保留扩展接口。


# （Phase 2：训练流程教学版）

你现在要在已完成的 Phase 1 基础上，继续实现  **Phase 2：训练流程教学版** 。

当前项目已有可视化 CNN 搭建能力。现在请在不破坏 Phase 1 主链路的前提下，扩展系统，使用户能够在图形界面中理解并配置基本训练流程。

本阶段重点不是做复杂训练平台，而是做 **面向初学者的训练流程可视化教学功能** 。

---

## 一、Phase 2 总体目标

请基于现有项目，新增以下能力：

1. 支持训练流程相关节点
2. 支持配置训练流程参数
3. 支持生成训练相关 PyTorch 代码
4. 支持最小可运行训练闭环
5. 支持展示训练结果（至少包含 loss，建议包含 accuracy）
6. 保持架构可扩展，为后续自定义模块封装、RNN、Transformer 预留接口

本阶段不追求复杂数据集管理、远程训练、GPU 调度、实验平台化，只做一个**本地、轻量、教学导向**的 MVP。

---

## 二、本阶段功能定位

Phase 1 解决的是：

* 模型怎么搭
* 每层 shape 怎么变化
* 模型代码如何生成

Phase 2 解决的是：

* 数据从哪里来
* DataLoader 起什么作用
* Loss 怎么接入训练
* Optimizer 怎么更新参数
* 训练循环做了什么
* loss / accuracy 如何变化

你实现的功能必须强调“可理解性”和“可解释性”，而不是追求最大自由度。

---

## 三、本阶段新增节点

请新增以下节点类型，并接入节点注册表、参数面板、项目 schema、后端 IR、校验与代码生成流程。

### 1. 数据相关节点

* `Dataset`
* `DataLoader`

### 2. 训练相关节点

* `Loss`
* `Optimizer`
* `Trainer`
* `Metric`（至少支持 Accuracy；如果实现复杂，可先做单一 Metric 节点）

### 3. 保留已有模型相关节点

继续兼容：

* Input
* Conv2d
* ReLU
* MaxPool2d
* Flatten
* Linear
* Output

---

## 四、Phase 2 图结构设计要求

本阶段不再只是一条“纯模型主链路”，而是要形成一个 **训练流程图** 。

但为了控制复杂度，本阶段请采用 **强约束结构** ，不要实现完全自由图。

### 1. 建议逻辑结构

训练流程应支持表达如下关系：

* `Dataset -> DataLoader`
* `DataLoader -> Model`
* `Model -> Loss`
* `Loss -> Trainer`
* `Optimizer -> Trainer`
* `Metric -> Trainer`

其中：

* `Model` 指现有模型子图或模型主链路
* `Trainer` 是训练流程收口节点，用于组织 epoch、日志和执行

### 2. 结构约束

请实现以下约束：

1. 一个项目最多一个主训练流程
2. `Trainer` 必须且只能有一个
3. `Dataset` 至少一个
4. `DataLoader` 必须连接到 `Dataset`
5. `Loss` 必须连接模型输出
6. `Optimizer` 必须依附模型
7. `Metric` 可选，但若存在必须连接到 `Trainer`
8. 本阶段不支持多训练器、多优化器并行训练
9. 不支持复杂多分支训练图
10. 不支持自定义脚本节点

如果你认为完全用“普通边”难以表达依赖关系，可以在 IR 中对训练节点增加语义字段，但前端仍保留图形表达。

---

## 五、节点参数要求

请为新增节点提供参数定义、默认值、参数面板展示、基础校验。

### 1. Dataset

至少支持以下模式之一：

#### 优先方案：内置教学数据集

先只支持内置数据集枚举，例如：

* `MNIST`
* `FakeData`

参数建议：

* `datasetName`
* `trainSplit`（bool 或固定 train/test）
* `imageSize`
* `numClasses`

本阶段优先支持内置数据集，不要求用户上传真实数据文件。

### 2. DataLoader

参数建议：

* `batchSize`
* `shuffle`
* `numWorkers`

### 3. Loss

首阶段建议支持枚举：

* `CrossEntropyLoss`
* `MSELoss`（可选）

参数建议：

* `lossType`

### 4. Optimizer

首阶段建议支持：

* `SGD`
* `Adam`

参数建议：

* `optimizerType`
* `lr`
* `weightDecay`
* `momentum`（仅对 SGD 有效，可选）

### 5. Trainer

参数建议：

* `epochs`
* `device`（`cpu` / `auto`）
* `logInterval`
* `validateEveryEpoch`（可选）

### 6. Metric

参数建议：

* `metricType`
* 首阶段至少支持 `Accuracy`

---

## 六、前端要求

请在现有前端骨架上做增量扩展，不要推翻 Phase 1 结构。

### 1. 节点注册表扩展

请在 `registry/` 中新增：

* Dataset
* DataLoader
* Loss
* Optimizer
* Trainer
* Metric

要求：

* 统一走注册表机制
* 参数定义集中管理
* 保持与 Phase 1 节点一致的注册方式

### 2. 左侧组件面板分类扩展

组件面板按类别至少分为：

* IO
* Layer
* Training

Training 分类下展示新节点。

### 3. 属性面板扩展

选中新增节点后，右侧参数面板应能动态展示对应表单。

要求：

* 表单项与节点类型匹配
* 不同优化器 / loss 类型可联动显示不同字段（若实现成本较高，可先保留统一字段）
* 参数修改后更新 store，并触发后端校验 / 推导 / 代码生成刷新

### 4. 画布交互要求

保持 Phase 1 的：

* 拖拽
* 连线
* 删除
* 选中
* 保存 / 加载

新增要求：

* 新训练节点可正常拖入与渲染
* 能看出训练类节点与模型层节点的视觉区别
* 可通过节点颜色、标题、图标或边框区分 Training 节点和 Layer 节点

### 5. 结果展示区扩展

Phase 2 需要新增训练结果展示区，至少支持：

* 训练状态文本展示
* loss 曲线显示
* accuracy 文本或曲线显示（建议）

如果前端图表已有库可复用，可以使用；否则先用简单列表或折线占位也可以。

---

## 七、后端要求

请在现有 FastAPI 后端基础上扩展，不要破坏 Phase 1 的 shape 推导与模型代码生成逻辑。

### 1. IR 扩展

后端 IR 必须支持区分三类节点：

1. 模型结构节点
2. 数据节点
3. 训练流程节点

建议为 IRNode 增加：

* `category`
* `training_role`（可选）
* `runtime_config`（可选）

但不要让 IR 失控，字段保持清晰即可。

### 2. 校验器扩展

新增训练流程校验规则，至少包括：

* 是否存在 Trainer
* Dataset / DataLoader 是否正确连接
* Loss 是否存在且是否挂接到模型输出
* Optimizer 是否存在
* Trainer 参数是否合法
* batchSize / epochs / lr 等是否为合法数值
* 训练流程图是否满足预定义约束

请区分：

* 全局错误
* 节点错误

错误信息必须是可读文本，不要只返回布尔值。

### 3. shape 推导与训练流程分离

注意：

* `Dataset` / `DataLoader` / `Loss` / `Optimizer` / `Trainer` 不应强行复用模型 shape 推导逻辑
* 模型层仍按 Phase 1 方式推导 shape
* 训练流程节点只需要做“语义校验”和“配置完整性检查”

也就是说：

* 模型节点负责张量 shape
* 训练节点负责流程合法性与运行配置

### 4. 代码生成扩展

请在原有模型代码生成基础上，新增：

1. 数据集构造代码
2. DataLoader 构造代码
3. Loss 实例化代码
4. Optimizer 实例化代码
5. 训练循环代码
6. Metric 计算代码（至少 Accuracy）
7. `train()` 或等价训练函数

最终生成代码至少包括：

* imports
* model definition
* dataset / dataloader setup
* criterion
* optimizer
* training loop
* logging

要求：

* 代码可读性优先
* 代码风格适合教学
* 尽量少做花哨抽象
* 变量命名清晰

### 5. 运行接口

请新增一个最小训练执行接口，例如：

* `POST /run-training`

输入：

* 当前项目图或归一化后的训练配置

输出：

* 是否运行成功
* 每个 epoch 的 loss
* 可选 accuracy
* 错误信息

运行要求：

* 本阶段先用 CPU 即可
* 先支持内置数据集
* 不追求高性能
* 必须能跑通一个 toy 教学示例

如果直接在 API 中执行训练过于阻塞，可以先同步执行，后续再优化。Phase 2 不要求完整异步任务系统。

---

## 八、训练执行最小闭环要求

本阶段必须至少支持一个标准可运行示例。

推荐基准示例：

### 示例 1：MNIST / FakeData + 简单 CNN

模型结构：

* Input
* Conv2d
* ReLU
* MaxPool2d
* Flatten
* Linear
* Output

训练结构：

* Dataset(MNIST 或 FakeData)
* DataLoader(batchSize=32)
* Loss(CrossEntropyLoss)
* Optimizer(Adam, lr=0.001)
* Trainer(epochs=2)
* Metric(Accuracy)

系统应能：

1. 校验图结构
2. 生成完整训练代码
3. 调用运行接口执行训练
4. 返回每个 epoch 的 loss
5. 最好返回 accuracy

---

## 九、前后端接口建议

请至少补充并实现以下接口：

### 1. `POST /validate-training-graph`

作用：

* 校验训练流程结构是否合法
* 返回全局错误与节点错误

### 2. `POST /generate-training-code`

作用：

* 基于当前项目图生成完整训练脚本代码

### 3. `POST /run-training`

作用：

* 执行最小训练闭环
* 返回日志数据

如果你认为可以复用现有 `/generate-code`，也可以，但必须保证模型代码和训练代码的语义清晰，不要混成难维护的单层逻辑。

---

## 十、保存 / 加载兼容要求

请确保项目文件 schema 向后兼容：

1. 旧的 Phase 1 项目仍然可以打开
2. Phase 2 项目能保存新增训练节点数据
3. 项目文件必须保留版本号
4. 若 schema 有升级，请实现最小迁移逻辑或兼容读取逻辑

---

## 十一、测试要求

请补充 Phase 2 的测试，不要只做手工验证。

### 1. 后端测试

至少包含：

* 训练图结构校验测试
* Dataset / DataLoader / Loss / Optimizer / Trainer 参数校验测试
* 训练代码生成测试
* `run-training` 最小闭环测试

### 2. 前端测试或人工验证项

至少验证：

* 新节点是否可拖入
* 参数面板是否可编辑
* 训练结果区是否更新
* 保存 / 加载后训练节点是否恢复
* 代码面板是否展示完整训练脚本

### 3. 回归要求

必须验证：

* Phase 1 原有模型搭建能力没有被破坏
* shape 推导仍正常
* 模型代码生成仍可用

---

## 十二、开发顺序要求

请严格按以下顺序推进，不要一开始就同时写大量杂糅逻辑。

### Step 1：扩展 schema 与注册表

先完成：

* 前端节点注册表扩展
* 项目 schema 扩展
* 后端 IR 扩展

### Step 2：前端新增训练节点占位

完成：

* 组件面板新增节点
* 画布可放置
* 属性面板可编辑参数

### Step 3：后端训练图校验

完成：

* 训练相关校验规则
* 可读错误输出

### Step 4：训练代码生成

完成：

* 数据集代码
* DataLoader 代码
* Loss / Optimizer 代码
* 训练循环代码

### Step 5：训练运行接口

完成：

* 最小训练执行闭环
* 返回 loss / accuracy

### Step 6：前端结果展示

完成：

* 训练日志展示
* loss 曲线
* 可选 accuracy

### Step 7：保存 / 加载 / 回归测试

完成：

* Phase 1 兼容
* Phase 2 存档恢复
* 回归验证

---

## 十三、实现约束

请遵守以下约束：

1. 不要把训练逻辑硬编码在前端页面组件里
2. 不要把所有新逻辑塞进单一 `generate-code` 文件
3. 不要破坏已有节点注册机制
4. 不要破坏已有 shape 推导架构
5. 不要提前实现过多 Phase 3 内容
6. 本阶段不实现用户上传任意数据集
7. 本阶段不实现复杂实验管理系统
8. 本阶段不实现 GPU 调度平台

---

## 十四、交付结果要求

本轮完成后，应至少交付：

1. 新增训练节点的前端可视化支持
2. 后端训练图校验能力
3. 完整训练代码生成功能
4. 最小训练执行接口
5. 前端训练结果展示
6. Phase 2 文档更新
7. 回归通过的 Phase 1 能力

---

## 十五、给你的实现偏好

请按以下偏好实现：

* 优先保证结构清晰
* 优先保证教学可读性
* 优先保证最小闭环可运行
* 不追求一次做全
* 允许先用内置数据集把闭环跑通
* 允许 UI 先简单，但接口与数据流要合理

如果遇到实现复杂度较高的部分，请采用“先最小可用，再保留扩展点”的方式，而不是过度设计。

---

## 十六、建议的 Phase 2 完成标准

只有当以下条件全部满足时，才算完成本阶段：

1. 用户可以拖入 Dataset、DataLoader、Loss、Optimizer、Trainer、Metric 节点
2. 用户可以配置这些节点参数
3. 系统可以校验训练流程是否合法
4. 系统可以生成完整训练代码
5. 系统可以运行一个最小训练示例
6. 前端可以展示 loss，最好能展示 accuracy
7. Phase 1 原功能未损坏
