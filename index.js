const { hasOwnProperty } = Object;

/**
 * Returns a perceptron object. It may be initialized from the given `weights`.
 * When given `weights`, the number of iterations used to obtain them are the
 * given `iterations`, or `0` by default.
 */
export default function (weights = {}, iterations = 0) {
  if (typeof weights !== "object" || weights == null) {
    throw TypeError();
  }

  if (!Number.isSafeInteger(iterations) || iterations < 0) {
    throw RangeError();
  }

  const weightsHistory = {};
  let iteration = iterations;

  const perceptron = {
    /**
     * Returns the label predicted from the given `features`, or `""` if none
     * exists. It may be given the `scores` so that it does not have to
     * recompute them.
     */
    predict(features = {}, scores = perceptron.scores(features)) {
      let bestScore = -Infinity;
      let prediction = "";
      const keys = Object.keys(scores);
      for (let i = 0; i < keys.length; i += 1) {
        const label = keys[i];
        const score = scores[label];
        if (score > bestScore) {
          bestScore = score;
          prediction = label;
        }
      }

      return prediction;
    },

    /**
     * Returns an object with the score of each label in the given `features`.
     */
    scores(features = {}) {
      const scores = {};
      const keys = Object.keys(features);
      for (let i = 0; i < keys.length; i += 1) {
        const feature = keys[i];
        const classes = weights[feature];
        const value = features[feature];
        if (classes && value !== 0) {
          const keys = Object.keys(classes);
          for (let i = 0; i < keys.length; i += 1) {
            const label = keys[i];
            scores[label] = (scores[label] || 0) + classes[label] * value;
          }
        }
      }

      return scores;
    },

    /**
     * Returns the perceptron, updating the weights with the respective value of
     * each of the given `features` if the given `label` is not predicted. It
     * may be given the `guess` so that it does not have to recompute it.
     */
    update(features = {}, label = "", guess = perceptron.predict(features)) {
      if (label !== guess) {
        Object.keys(features).forEach((feature) => {
          if (!hasOwnProperty.call(weights, feature)) {
            weights[feature] = {};
          }

          if (!hasOwnProperty.call(weightsHistory, feature)) {
            weightsHistory[feature] = {};
          }

          const value = features[feature];
          const classes = weights[feature];
          const classesHistory = weightsHistory[feature];
          const {
            [label]: labelWeight = 0,
            [guess]: guessWeight = 0,
          } = classes;
          const {
            [label]: [labelTotal, labelTimestamp] = [0, 0],
            [guess]: [guessTotal, guessTimestamp] = [0, 0],
          } = classesHistory;
          classes[label] = labelWeight + value;
          classesHistory[label] = [
            labelTotal + labelWeight * (iteration - labelTimestamp),
            iteration,
          ];
          if (guess !== "") {
            classes[guess] = guessWeight - value;
            classesHistory[guess] = [
              guessTotal + guessWeight * (iteration - guessTimestamp),
              iteration,
            ];
          }
        });
      }

      iteration += 1;
      return perceptron;
    },

    /**
     * Returns an object of features where each feature is an object of labels
     * with the weight of each feature-label pair.
     */
    weights() {
      const iterations = iteration || 1;
      const averagedWeights = {};
      Object.keys(weights).forEach((feature) => {
        const classes = weights[feature];
        const classesHistory = weightsHistory[feature] || {};
        const averagedClasses = {};
        averagedWeights[feature] = averagedClasses;
        Object.keys(classes).forEach((label) => {
          const weight = classes[label];
          const [total = 0, timestamp = 0] = classesHistory[label] || [];
          const updatedTotal = total + weight * (iterations - timestamp);
          averagedClasses[label] = updatedTotal / iterations;
        });
      });
      return averagedWeights;
    },
  };

  return perceptron;
}
