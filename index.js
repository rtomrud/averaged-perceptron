export default function (weights = {}, iterations = 0) {
  if (!Number.isSafeInteger(iterations) || iterations < 0) {
    throw RangeError(`${iterations} is not a whole number`);
  }

  let i = iterations;
  const accumulatedWeights = {};
  const perceptron = {
    predict(features) {
      const scores = {};
      for (const feature in features) {
        const classes = weights[feature];
        const value = features[feature];
        if (classes && value) {
          for (const label in classes) {
            scores[label] = (scores[label] || 0) + classes[label] * value;
          }
        }
      }

      let bestScore = -Infinity;
      let prediction = "";
      for (const label in scores) {
        const score = scores[label];
        if (score > bestScore) {
          bestScore = score;
          prediction = label;
        }
      }

      return prediction;
    },

    update(features, label, guess = perceptron.predict(features)) {
      if (label !== guess) {
        for (const feature in features) {
          if (!weights[feature]) {
            weights[feature] = {};
          }

          if (!accumulatedWeights[feature]) {
            accumulatedWeights[feature] = {};
          }

          const value = features[feature];
          const classes = weights[feature];
          const accumulatedClasses = accumulatedWeights[feature];
          const weight = classes[label] || 0;
          const { [label]: [total, timestamp] = [0, 0] } = accumulatedClasses;
          classes[label] = weight + value;
          accumulatedClasses[label] = [total + weight * (i - timestamp), i];
          if (guess) {
            const { [guess]: weight = 0 } = classes;
            const { [guess]: [total, timestamp] = [0, 0] } = accumulatedClasses;
            classes[guess] = weight - value;
            accumulatedClasses[guess] = [total + weight * (i - timestamp), i];
          }
        }
      }

      i += 1;
      return perceptron;
    },

    weights() {
      const iterations = i || 1;
      const averagedWeights = {};
      for (const feature in weights) {
        const classes = weights[feature];
        const accumulatedClasses = accumulatedWeights[feature] || {};
        const averagedClasses = {};
        averagedWeights[feature] = averagedClasses;
        for (const label in classes) {
          const weight = classes[label];
          const [total = 0, timestamp = 0] = accumulatedClasses[label] || [];
          const newTotal = total + weight * (iterations - timestamp);
          averagedClasses[label] = newTotal / iterations;
        }
      }

      return averagedWeights;
    },
  };
  return perceptron;
}
