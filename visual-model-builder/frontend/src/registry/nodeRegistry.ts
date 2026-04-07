import type { NodeBehavior } from '../types';

const InputNode: NodeBehavior = {
  template: {
    type: 'Input',
    displayName: 'Input',
    category: 'io',
    description: 'Define the input tensor shape as [C, H, W]. In Phase 2 it can also receive data from DataLoader.',
    inputPorts: 1,
    outputPorts: 1,
    params: [
      {
        key: 'inputShape',
        label: 'Input Shape',
        type: 'shape',
        required: true,
        defaultValue: [1, 28, 28],
        helpText: 'Tensor shape in [channels, height, width] format.',
      },
    ],
  },
  defaultData: () => ({
    inputShape: [1, 28, 28],
  }),
};

const Conv2dNode: NodeBehavior = {
  template: {
    type: 'Conv2d',
    displayName: 'Conv2d',
    category: 'layer',
    description: '2D convolution layer.',
    inputPorts: 1,
    outputPorts: 1,
    params: [
      { key: 'in_channels', label: 'In Channels', type: 'int', required: true, defaultValue: 1, helpText: 'Number of input channels.' },
      { key: 'out_channels', label: 'Out Channels', type: 'int', required: true, defaultValue: 16, helpText: 'Number of output channels.' },
      { key: 'kernel_size', label: 'Kernel Size', type: 'int', required: true, defaultValue: 3, helpText: 'Convolution kernel size.' },
      { key: 'stride', label: 'Stride', type: 'int', required: true, defaultValue: 1, helpText: 'Step size of the kernel.' },
      { key: 'padding', label: 'Padding', type: 'int', required: true, defaultValue: 0, helpText: 'Zero padding size.' },
    ],
  },
  defaultData: () => ({
    in_channels: 1,
    out_channels: 16,
    kernel_size: 3,
    stride: 1,
    padding: 0,
  }),
};

const ReLUNode: NodeBehavior = {
  template: {
    type: 'ReLU',
    displayName: 'ReLU',
    category: 'activation',
    description: 'ReLU activation function.',
    inputPorts: 1,
    outputPorts: 1,
    params: [
      { key: 'inplace', label: 'Inplace', type: 'bool', required: false, defaultValue: false, helpText: 'Whether to modify the input tensor in place.' },
    ],
  },
  defaultData: () => ({
    inplace: false,
  }),
};

const MaxPool2dNode: NodeBehavior = {
  template: {
    type: 'MaxPool2d',
    displayName: 'MaxPool2d',
    category: 'layer',
    description: '2D max pooling layer.',
    inputPorts: 1,
    outputPorts: 1,
    params: [
      { key: 'kernel_size', label: 'Kernel Size', type: 'int', required: true, defaultValue: 2, helpText: 'Pooling window size.' },
      { key: 'stride', label: 'Stride', type: 'int', required: true, defaultValue: 2, helpText: 'Stride for the pooling window.' },
      { key: 'padding', label: 'Padding', type: 'int', required: true, defaultValue: 0, helpText: 'Zero padding size.' },
    ],
  },
  defaultData: () => ({
    kernel_size: 2,
    stride: 2,
    padding: 0,
  }),
};

const FlattenNode: NodeBehavior = {
  template: {
    type: 'Flatten',
    displayName: 'Flatten',
    category: 'layer',
    description: 'Flatten a tensor into a single feature vector.',
    inputPorts: 1,
    outputPorts: 1,
    params: [
      { key: 'start_dim', label: 'Start Dim', type: 'int', required: false, defaultValue: 0, helpText: 'The first dimension to flatten.' },
      { key: 'end_dim', label: 'End Dim', type: 'int', required: false, defaultValue: -1, helpText: 'The last dimension to flatten.' },
    ],
  },
  defaultData: () => ({
    start_dim: 0,
    end_dim: -1,
  }),
};

const LinearNode: NodeBehavior = {
  template: {
    type: 'Linear',
    displayName: 'Linear',
    category: 'layer',
    description: 'Fully connected layer.',
    inputPorts: 1,
    outputPorts: 1,
    params: [
      { key: 'in_features', label: 'In Features', type: 'int', required: true, defaultValue: 128, helpText: 'Input feature size.' },
      { key: 'out_features', label: 'Out Features', type: 'int', required: true, defaultValue: 10, helpText: 'Output feature size.' },
      { key: 'bias', label: 'Bias', type: 'bool', required: false, defaultValue: true, helpText: 'Whether to use a bias term.' },
    ],
  },
  defaultData: () => ({
    in_features: 128,
    out_features: 10,
    bias: true,
  }),
};

const OutputNode: NodeBehavior = {
  template: {
    type: 'Output',
    displayName: 'Output',
    category: 'io',
    description: 'Output node for the final tensor. In Phase 2 it can also forward predictions to Loss.',
    inputPorts: 1,
    outputPorts: 1,
    params: [],
  },
  defaultData: () => ({}),
};

const DatasetNode: NodeBehavior = {
  template: {
    type: 'Dataset',
    displayName: 'Dataset',
    category: 'train',
    description: 'Dataset source for builtin teaching data or real local datasets.',
    inputPorts: 0,
    outputPorts: 1,
    params: [
      {
        key: 'datasetMode',
        label: 'Dataset Mode',
        type: 'select',
        required: true,
        defaultValue: 'builtin',
        options: ['builtin', 'image_folder', 'csv'],
        helpText: 'Choose between builtin teaching data and local real datasets.',
      },
      {
        key: 'datasetName',
        label: 'Builtin Dataset',
        type: 'select',
        required: true,
        defaultValue: 'FakeData',
        options: ['FakeData', 'MNIST'],
        helpText: 'Built-in dataset used for the training loop.',
        visible: (params) => String(params.datasetMode ?? 'builtin') === 'builtin',
      },
      {
        key: 'trainSplit',
        label: 'Use Train Split',
        type: 'bool',
        required: true,
        defaultValue: true,
        helpText: 'Builtin datasets can switch between the train/test split source.',
        visible: (params) => String(params.datasetMode ?? 'builtin') === 'builtin',
      },
      {
        key: 'rootPath',
        label: 'Root Path',
        type: 'text',
        required: true,
        defaultValue: '',
        placeholder: 'E:\\datasets\\cats-vs-dogs',
        helpText: 'Root directory for an ImageFolder-compatible dataset.',
        visible: (params) => String(params.datasetMode ?? 'builtin') === 'image_folder',
      },
      {
        key: 'splitMode',
        label: 'Split Mode',
        type: 'select',
        required: true,
        defaultValue: 'predefined',
        options: ['predefined', 'ratio'],
        helpText: 'Use explicit train/val/test folders or split a flat class folder by ratio.',
        visible: (params) => String(params.datasetMode ?? 'builtin') === 'image_folder',
      },
      {
        key: 'trainRatio',
        label: 'Train Ratio',
        type: 'float',
        required: true,
        defaultValue: 0.7,
        helpText: 'Used when ratio splitting is active.',
        visible: (params) =>
          String(params.datasetMode ?? 'builtin') === 'image_folder' && String(params.splitMode ?? 'predefined') === 'ratio',
      },
      {
        key: 'valRatio',
        label: 'Val Ratio',
        type: 'float',
        required: true,
        defaultValue: 0.2,
        helpText: 'Used when ratio splitting is active.',
        visible: (params) =>
          String(params.datasetMode ?? 'builtin') === 'image_folder' && String(params.splitMode ?? 'predefined') === 'ratio',
      },
      {
        key: 'testRatio',
        label: 'Test Ratio',
        type: 'float',
        required: true,
        defaultValue: 0.1,
        helpText: 'Used when ratio splitting is active.',
        visible: (params) =>
          String(params.datasetMode ?? 'builtin') === 'image_folder' && String(params.splitMode ?? 'predefined') === 'ratio',
      },
      {
        key: 'shuffleBeforeSplit',
        label: 'Shuffle Before Split',
        type: 'bool',
        required: true,
        defaultValue: true,
        helpText: 'Shuffle flat image-folder samples before applying ratio-based splits.',
        visible: (params) =>
          String(params.datasetMode ?? 'builtin') === 'image_folder' && String(params.splitMode ?? 'predefined') === 'ratio',
      },
      {
        key: 'csvPath',
        label: 'CSV Path',
        type: 'text',
        required: true,
        defaultValue: '',
        placeholder: 'E:\\datasets\\labels.csv',
        helpText: 'CSV file path. Runtime support is reserved for a follow-up phase.',
        visible: (params) => String(params.datasetMode ?? 'builtin') === 'csv',
      },
      {
        key: 'pathColumn',
        label: 'Path Column',
        type: 'text',
        required: true,
        defaultValue: 'image_path',
        helpText: 'CSV column that points to image files.',
        visible: (params) => String(params.datasetMode ?? 'builtin') === 'csv',
      },
      {
        key: 'labelColumn',
        label: 'Label Column',
        type: 'text',
        required: true,
        defaultValue: 'label',
        helpText: 'CSV column that contains labels.',
        visible: (params) => String(params.datasetMode ?? 'builtin') === 'csv',
      },
      {
        key: 'featureColumns',
        label: 'Feature Columns',
        type: 'string_list',
        required: false,
        defaultValue: [],
        helpText: 'Reserved for future CSV/tabular support.',
        visible: (params) => String(params.datasetMode ?? 'builtin') === 'csv',
      },
      {
        key: 'taskType',
        label: 'Task Type',
        type: 'select',
        required: true,
        defaultValue: 'classification',
        options: ['classification', 'regression'],
        helpText: 'Classification is fully supported for builtin and image-folder datasets in this phase.',
      },
      {
        key: 'imageSize',
        label: 'Image Size',
        type: 'int',
        required: true,
        defaultValue: 28,
        helpText: 'Square image size after preprocessing.',
      },
      {
        key: 'colorMode',
        label: 'Color Mode',
        type: 'select',
        required: true,
        defaultValue: 'grayscale',
        options: ['grayscale', 'rgb'],
        helpText: 'Color mode after preprocessing.',
      },
      {
        key: 'normalize',
        label: 'Normalize',
        type: 'bool',
        required: true,
        defaultValue: false,
        helpText: 'Apply channel-wise normalization after ToTensor.',
      },
      {
        key: 'mean',
        label: 'Mean',
        type: 'float_list',
        required: false,
        defaultValue: [0.5],
        helpText: 'Normalization mean. A single value expands to every channel.',
        visible: (params) => Boolean(params.normalize),
      },
      {
        key: 'std',
        label: 'Std',
        type: 'float_list',
        required: false,
        defaultValue: [0.5],
        helpText: 'Normalization std. A single value expands to every channel.',
        visible: (params) => Boolean(params.normalize),
      },
      {
        key: 'augmentationEnabled',
        label: 'Enable Augmentation',
        type: 'bool',
        required: true,
        defaultValue: false,
        helpText: 'Use lightweight training-time augmentation for image datasets.',
        visible: (params) => String(params.datasetMode ?? 'builtin') !== 'csv',
      },
      {
        key: 'numClasses',
        label: 'Num Classes',
        type: 'int',
        required: true,
        defaultValue: 10,
        helpText: 'Builtin datasets use this directly; real datasets will overwrite it with inspected metadata.',
        visible: (params) => String(params.datasetMode ?? 'builtin') === 'builtin',
      },
    ],
  },
  defaultData: () => ({
    datasetMode: 'builtin',
    datasetName: 'FakeData',
    trainSplit: true,
    rootPath: '',
    splitMode: 'predefined',
    trainRatio: 0.7,
    valRatio: 0.2,
    testRatio: 0.1,
    shuffleBeforeSplit: true,
    imageSize: 28,
    colorMode: 'grayscale',
    normalize: false,
    mean: [0.5],
    std: [0.5],
    augmentationEnabled: false,
    csvPath: '',
    labelColumn: 'label',
    pathColumn: 'image_path',
    featureColumns: [],
    taskType: 'classification',
    numClasses: 10,
  }),
};

const DataLoaderNode: NodeBehavior = {
  template: {
    type: 'DataLoader',
    displayName: 'DataLoader',
    category: 'train',
    description: 'Mini-batch loader for training data.',
    inputPorts: 1,
    outputPorts: 1,
    params: [
      { key: 'batchSize', label: 'Batch Size', type: 'int', required: true, defaultValue: 32, helpText: 'Number of samples per mini-batch.' },
      { key: 'shuffle', label: 'Shuffle', type: 'bool', required: true, defaultValue: true, helpText: 'Shuffle samples every epoch.' },
      { key: 'numWorkers', label: 'Workers', type: 'int', required: true, defaultValue: 0, helpText: 'Number of worker processes for data loading.' },
      { key: 'dropLast', label: 'Drop Last', type: 'bool', required: true, defaultValue: false, helpText: 'Drop the last incomplete mini-batch.' },
      { key: 'pinMemory', label: 'Pin Memory', type: 'bool', required: true, defaultValue: false, helpText: 'Enable pinned host memory for faster device transfer.' },
      {
        key: 'persistentWorkers',
        label: 'Persistent Workers',
        type: 'bool',
        required: true,
        defaultValue: false,
        helpText: 'Keep worker processes alive between epochs. Only works when numWorkers > 0.',
        disabled: (params) => Number(params.numWorkers ?? 0) === 0,
      },
      {
        key: 'prefetchFactor',
        label: 'Prefetch Factor',
        type: 'int',
        required: false,
        defaultValue: 2,
        helpText: 'Number of batches prefetched per worker. Only works when numWorkers > 0.',
        disabled: (params) => Number(params.numWorkers ?? 0) === 0,
      },
      {
        key: 'collateFnType',
        label: 'Collate Fn',
        type: 'select',
        required: true,
        defaultValue: 'default',
        options: ['default', 'custom_placeholder'],
        helpText: 'Reserved extension point for future custom collate functions.',
      },
    ],
  },
  defaultData: () => ({
    batchSize: 32,
    shuffle: true,
    numWorkers: 0,
    dropLast: false,
    pinMemory: false,
    persistentWorkers: false,
    prefetchFactor: 2,
    collateFnType: 'default',
  }),
};

const LossNode: NodeBehavior = {
  template: {
    type: 'Loss',
    displayName: 'Loss',
    category: 'train',
    description: 'Training objective used to compare predictions and labels.',
    inputPorts: 1,
    outputPorts: 1,
    params: [
      { key: 'lossType', label: 'Loss Type', type: 'select', required: true, defaultValue: 'CrossEntropyLoss', options: ['CrossEntropyLoss', 'MSELoss'], helpText: 'Select the loss function used during training.' },
    ],
  },
  defaultData: () => ({
    lossType: 'CrossEntropyLoss',
  }),
};

const OptimizerNode: NodeBehavior = {
  template: {
    type: 'Optimizer',
    displayName: 'Optimizer',
    category: 'train',
    description: 'Updates model parameters from gradients.',
    inputPorts: 0,
    outputPorts: 1,
    params: [
      { key: 'optimizerType', label: 'Optimizer', type: 'select', required: true, defaultValue: 'Adam', options: ['Adam', 'SGD'], helpText: 'Choose the optimization algorithm.' },
      { key: 'lr', label: 'Learning Rate', type: 'float', required: true, defaultValue: 0.001, helpText: 'Step size used by the optimizer.' },
      { key: 'weightDecay', label: 'Weight Decay', type: 'float', required: true, defaultValue: 0, helpText: 'L2 regularization term.' },
      { key: 'momentum', label: 'Momentum', type: 'float', required: false, defaultValue: 0.9, helpText: 'Only used by SGD.' },
    ],
  },
  defaultData: () => ({
    optimizerType: 'Adam',
    lr: 0.001,
    weightDecay: 0,
    momentum: 0.9,
  }),
};

const TrainerNode: NodeBehavior = {
  template: {
    type: 'Trainer',
    displayName: 'Trainer',
    category: 'train',
    description: 'Collects loss, optimizer, and metric nodes into a runnable training loop.',
    inputPorts: 3,
    outputPorts: 0,
    params: [
      { key: 'epochs', label: 'Epochs', type: 'int', required: true, defaultValue: 2, helpText: 'Number of passes through the dataset.' },
      { key: 'device', label: 'Device', type: 'select', required: true, defaultValue: 'cpu', options: ['cpu', 'auto'], helpText: 'Runtime device preference.' },
      { key: 'logInterval', label: 'Log Interval', type: 'int', required: true, defaultValue: 1, helpText: 'How often to record progress during training.' },
      { key: 'validateEveryEpoch', label: 'Validate Every Epoch', type: 'bool', required: false, defaultValue: false, helpText: 'Reserved toggle for future validation runs.' },
    ],
  },
  defaultData: () => ({
    epochs: 2,
    device: 'cpu',
    logInterval: 1,
    validateEveryEpoch: false,
  }),
};

const MetricNode: NodeBehavior = {
  template: {
    type: 'Metric',
    displayName: 'Metric',
    category: 'train',
    description: 'Optional training metric shown alongside loss.',
    inputPorts: 0,
    outputPorts: 1,
    params: [
      { key: 'metricType', label: 'Metric', type: 'select', required: true, defaultValue: 'Accuracy', options: ['Accuracy'], helpText: 'Metric shown in the training panel.' },
    ],
  },
  defaultData: () => ({
    metricType: 'Accuracy',
  }),
};

export const nodeRegistry: Record<string, NodeBehavior> = {
  Input: InputNode,
  Conv2d: Conv2dNode,
  ReLU: ReLUNode,
  MaxPool2d: MaxPool2dNode,
  Flatten: FlattenNode,
  Linear: LinearNode,
  Output: OutputNode,
  Dataset: DatasetNode,
  DataLoader: DataLoaderNode,
  Loss: LossNode,
  Optimizer: OptimizerNode,
  Trainer: TrainerNode,
  Metric: MetricNode,
};

export function getNodesByCategory(): Record<string, NodeBehavior[]> {
  const grouped: Record<string, NodeBehavior[]> = {};
  for (const behavior of Object.values(nodeRegistry)) {
    const category = behavior.template.category;
    if (!grouped[category]) {
      grouped[category] = [];
    }
    grouped[category].push(behavior);
  }
  return grouped;
}

export function getNodeBehavior(type: string): NodeBehavior | undefined {
  return nodeRegistry[type];
}
