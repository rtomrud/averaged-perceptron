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

export interface AveragedPerceptron {
  /**
   * Returns the label predicted from the values in `features`, or `""` if
   * none exists.
   */
  predict(features: Features): string;

  /**
   * Returns the perceptron, updating its weights with the respective values
   * in `features` if `label` does not equal `guess`. If `guess` is not given,
   * it defaults to the output of `predict(features)`.
   */
  update(features: Features, label: string, guess?: string): AveragedPerceptron;

  /**
   * Returns an object of objects with the weight of each feature-label pair.
   */
  weights(): Weights;
}

/**
 * Returns a perceptron object. It may be initialized with `weights`, an object
 * of objects with the weight of each feature-label pair. When initialized with
 * `weights`, the number of iterations used to obtain them are  `iterations`, or
 * `0` by default.
 */
export default function averagedPerceptron(
  weights?: Weights,
  iterations?: number,
): AveragedPerceptron;
