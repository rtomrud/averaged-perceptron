export type Features =
  | {
      [feature: string | number]: number;
    }
  | {
      [feature: number]: number;
    };

export type Labels = {
  [feature: string]: number;
};

export type Weights =
  | {
      [feature: string | number]: Labels;
    }
  | {
      [feature: number]: Labels;
    };

export type AveragedPerceptron = {
  predict(features: Features): string;
  update(features: Features, label: string, guess?: string): AveragedPerceptron;
  weights(): Weights;
};
