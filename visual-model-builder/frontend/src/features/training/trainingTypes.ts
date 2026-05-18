export type TrainingView = 'curves' | 'status' | 'logs' | 'analysis';

export type ChartPoint = {
  x: number;
  y: number;
};

export type ReadinessItem = {
  label: string;
  ok: boolean;
};

export type SaveFilePickerOptions = {
  suggestedName?: string;
  types?: Array<{
    description: string;
    accept: Record<string, string[]>;
  }>;
};

export type SaveFileHandle = {
  createWritable: () => Promise<{
    write: (content: string) => Promise<void>;
    close: () => Promise<void>;
  }>;
};

export type SavePickerWindow = Window & {
  showSaveFilePicker?: (options?: SaveFilePickerOptions) => Promise<SaveFileHandle>;
};
