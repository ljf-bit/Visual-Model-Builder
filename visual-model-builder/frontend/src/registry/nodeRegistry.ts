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
    description: 'Built-in teaching dataset source.',
    inputPorts: 0,
    outputPorts: 1,
    params: [
      { key: 'datasetName', label: 'Dataset', type: 'select', required: true, defaultValue: 'FakeData', options: ['FakeData', 'MNIST'], helpText: 'Built-in dataset used for the training loop.' },
      { key: 'trainSplit', label: 'Use Train Split', type: 'bool', required: true, defaultValue: true, helpText: 'Choose train split when the dataset supports it.' },
      { key: 'imageSize', label: 'Image Size', type: 'int', required: true, defaultValue: 28, helpText: 'Square image size used by the dataset.' },
      { key: 'numClasses', label: 'Num Classes', type: 'int', required: true, defaultValue: 10, helpText: 'Number of target classes.' },
    ],
  },
  defaultData: () => ({
    datasetName: 'FakeData',
    trainSplit: true,
    imageSize: 28,
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
    ],
  },
  defaultData: () => ({
    batchSize: 32,
    shuffle: true,
    numWorkers: 0,
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
