const makeWeightWrapper = weights => ({
  get: label => weights[label]
});

/**
 * Returns a perceptron object. Can be initialized from the given `weights`. If
 * given `weights`, the number of iterations used to obtain them are the given
 * `iterations`, or `0` by default.
 */
// eslint-disable-next-line max-lines-per-function
export default function(weights = makeWeightWrapper(), iterations = 0) {
  if (!Number.isSafeInteger(iterations) || iterations < 0) {
    throw RangeError();
  }

  const weightsHistory = {};
  let iteration = iterations;

  const predictFromScores = scores => {
    let bestScore = -Infinity;
    let prediction = null;
    Object.keys(scores).forEach(label => {
      const score = scores[label];
      if (score >= bestScore) {
        bestScore = score;
        prediction = label;
      }
    });
    return prediction;
  };

  const scoreFeature = features => (scoresPromise, feature) =>
    scoresPromise.then(scores => {
      const value = features[feature];
      return weights.get(feature).then(classes => {
        if (value === 0 || classes == null) {
          return scores;
        }

        Object.keys(classes).forEach(label => {
          scores[label] = (scores[label] || 0) + classes[label] * value;
        });
        return scores;
      });
    });

  const perceptron = {
    /**
     * Returns the predicted label from the given `features`, or `null` if none
     * exists. Can be given the `scores` used to predict.
     */
    predict(features = {}, scores) {
      if (scores == null) {
        return perceptron
          .scores(features)
          .then(scores => predictFromScores(scores));
      }

      return Promise.resolve(predictFromScores(scores));
    },

    /**
     * Returns an object with the scores of each label in the given `features`.
     */
    scores(features = {}) {
      return Object.keys(features).reduce(
        scoreFeature(features),
        Promise.resolve({})
      );
    },

    /**
     * Returns the perceptron, updating the weights with the respective value of
     * each of the given `features` if the given `label` is not predicted.
     * Can be given the `guess` so that it does not have to compute it.
     */
    update(features = {}, label = "", guess = perceptron.predict(features)) {
      if (label === guess) {
        iteration += 1;
        return perceptron;
      }

      Object.keys(features).forEach(feature => {
        const value = features[feature];
        if (weights[feature] == null) {
          weights[feature] = {};
        }

        if (weightsHistory[feature] == null) {
          weightsHistory[feature] = {};
        }

        const classes = weights[feature];
        const classesHistory = weightsHistory[feature];
        const { [label]: labelWeight = 0, [guess]: guessWeight = 0 } = classes;
        const {
          [label]: [labelTotal, labelTimestamp] = [0, 0],
          [guess]: [guessTotal, guessTimestamp] = [0, 0]
        } = classesHistory;
        classes[label] = labelWeight + value;
        classesHistory[label] = [
          labelTotal + labelWeight * (iteration - labelTimestamp),
          iteration
        ];
        if (guess != null) {
          classes[guess] = guessWeight - value;
          classesHistory[guess] = [
            guessTotal + guessWeight * (iteration - guessTimestamp),
            iteration
          ];
        }
      });

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
      Object.keys(weights).forEach(feature => {
        const classes = weights[feature];
        const classesHistory = weightsHistory[feature] || {};
        const averagedClasses = {};
        averagedWeights[feature] = averagedClasses;
        Object.keys(classes).forEach(label => {
          const weight = classes[label];
          const [total = 0, timestamp = 0] = classesHistory[label] || [];
          const updatedTotal = total + weight * (iterations - timestamp);
          averagedClasses[label] = updatedTotal / iterations;
        });
      });
      return averagedWeights;
    }
  };

  return perceptron;
}
