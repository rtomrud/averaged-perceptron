import test from "./node_modules/tape/index.js";
import averagedPerceptron from "./index.js";

test("averaged-perceptron with the Iris dataset", ({ equal, end }) => {
  // Note that the features can be either an object or an array
  const trainingDataset = [
    [[5.1, 3.5, 1.4, 0.2], "Iris setosa"],
    [[4.9, 3.0, 1.4, 0.2], "Iris setosa"],
    [[4.7, 3.2, 1.3, 0.2], "Iris setosa"],
    [[4.6, 3.1, 1.5, 0.2], "Iris setosa"],
    [[5.0, 3.6, 1.4, 0.2], "Iris setosa"],
    [[5.4, 3.9, 1.7, 0.4], "Iris setosa"],
    [[4.6, 3.4, 1.4, 0.3], "Iris setosa"],
    [[5.0, 3.4, 1.5, 0.2], "Iris setosa"],
    [[4.4, 2.9, 1.4, 0.2], "Iris setosa"],
    [[4.9, 3.1, 1.5, 0.1], "Iris setosa"],
    [[5.4, 3.7, 1.5, 0.2], "Iris setosa"],
    [[4.8, 3.4, 1.6, 0.2], "Iris setosa"],
    [[4.8, 3.0, 1.4, 0.1], "Iris setosa"],
    [[4.3, 3.0, 1.1, 0.1], "Iris setosa"],
    [[5.8, 4.0, 1.2, 0.2], "Iris setosa"],
    [[5.7, 4.4, 1.5, 0.4], "Iris setosa"],
    [[5.4, 3.9, 1.3, 0.4], "Iris setosa"],
    [[5.1, 3.5, 1.4, 0.3], "Iris setosa"],
    [[5.7, 3.8, 1.7, 0.3], "Iris setosa"],
    [[5.1, 3.8, 1.5, 0.3], "Iris setosa"],
    [[5.4, 3.4, 1.7, 0.2], "Iris setosa"],
    [[5.1, 3.7, 1.5, 0.4], "Iris setosa"],
    [[4.6, 3.6, 1.0, 0.2], "Iris setosa"],
    [[5.1, 3.3, 1.7, 0.5], "Iris setosa"],
    [[4.8, 3.4, 1.9, 0.2], "Iris setosa"],
    [[5.0, 3.0, 1.6, 0.2], "Iris setosa"],
    [[5.0, 3.4, 1.6, 0.4], "Iris setosa"],
    [[5.2, 3.5, 1.5, 0.2], "Iris setosa"],
    [[5.2, 3.4, 1.4, 0.2], "Iris setosa"],
    [[4.7, 3.2, 1.6, 0.2], "Iris setosa"],
    [[4.8, 3.1, 1.6, 0.2], "Iris setosa"],
    [[5.4, 3.4, 1.5, 0.4], "Iris setosa"],
    [[5.2, 4.1, 1.5, 0.1], "Iris setosa"],
    [[7.0, 3.2, 4.7, 1.4], "Iris versicolor"],
    [[6.4, 3.2, 4.5, 1.5], "Iris versicolor"],
    [[6.9, 3.1, 4.9, 1.5], "Iris versicolor"],
    [[5.5, 2.3, 4.0, 1.3], "Iris versicolor"],
    [[6.5, 2.8, 4.6, 1.5], "Iris versicolor"],
    [[5.7, 2.8, 4.5, 1.3], "Iris versicolor"],
    [[6.3, 3.3, 4.7, 1.6], "Iris versicolor"],
    [[4.9, 2.4, 3.3, 1.0], "Iris versicolor"],
    [[6.6, 2.9, 4.6, 1.3], "Iris versicolor"],
    [[5.2, 2.7, 3.9, 1.4], "Iris versicolor"],
    [[5.0, 2.0, 3.5, 1.0], "Iris versicolor"],
    [[5.9, 3.0, 4.2, 1.5], "Iris versicolor"],
    [[6.0, 2.2, 4.0, 1.0], "Iris versicolor"],
    [[6.1, 2.9, 4.7, 1.4], "Iris versicolor"],
    [[5.6, 2.9, 3.6, 1.3], "Iris versicolor"],
    [[6.7, 3.1, 4.4, 1.4], "Iris versicolor"],
    [[5.6, 3.0, 4.5, 1.5], "Iris versicolor"],
    [[5.8, 2.7, 4.1, 1.0], "Iris versicolor"],
    [[6.2, 2.2, 4.5, 1.5], "Iris versicolor"],
    [[5.6, 2.5, 3.9, 1.1], "Iris versicolor"],
    [[5.9, 3.2, 4.8, 1.8], "Iris versicolor"],
    [[6.1, 2.8, 4.0, 1.3], "Iris versicolor"],
    [[6.3, 2.5, 4.9, 1.5], "Iris versicolor"],
    [[6.1, 2.8, 4.7, 1.2], "Iris versicolor"],
    [[6.4, 2.9, 4.3, 1.3], "Iris versicolor"],
    [[6.6, 3.0, 4.4, 1.4], "Iris versicolor"],
    [[6.8, 2.8, 4.8, 1.4], "Iris versicolor"],
    [[6.7, 3.0, 5.0, 1.7], "Iris versicolor"],
    [[6.0, 2.9, 4.5, 1.5], "Iris versicolor"],
    [[5.7, 2.6, 3.5, 1.0], "Iris versicolor"],
    [[5.5, 2.4, 3.8, 1.1], "Iris versicolor"],
    [[5.5, 2.4, 3.7, 1.0], "Iris versicolor"],
    [[5.8, 2.7, 3.9, 1.2], "Iris versicolor"],
    [[6.3, 3.3, 6.0, 2.5], "Iris virginica"],
    [[5.8, 2.7, 5.1, 1.9], "Iris virginica"],
    [[7.1, 3.0, 5.9, 2.1], "Iris virginica"],
    [[6.3, 2.9, 5.6, 1.8], "Iris virginica"],
    [[6.5, 3.0, 5.8, 2.2], "Iris virginica"],
    [[7.6, 3.0, 6.6, 2.1], "Iris virginica"],
    [[4.9, 2.5, 4.5, 1.7], "Iris virginica"],
    [[7.3, 2.9, 6.3, 1.8], "Iris virginica"],
    [[6.7, 2.5, 5.8, 1.8], "Iris virginica"],
    [[7.2, 3.6, 6.1, 2.5], "Iris virginica"],
    [[6.5, 3.2, 5.1, 2.0], "Iris virginica"],
    [[6.4, 2.7, 5.3, 1.9], "Iris virginica"],
    [[6.8, 3.0, 5.5, 2.1], "Iris virginica"],
    [[5.7, 2.5, 5.0, 2.0], "Iris virginica"],
    [[5.8, 2.8, 5.1, 2.4], "Iris virginica"],
    [[6.4, 3.2, 5.3, 2.3], "Iris virginica"],
    [[6.5, 3.0, 5.5, 1.8], "Iris virginica"],
    [[7.7, 3.8, 6.7, 2.2], "Iris virginica"],
    [[7.7, 2.6, 6.9, 2.3], "Iris virginica"],
    [[6.0, 2.2, 5.0, 1.5], "Iris virginica"],
    [[6.9, 3.2, 5.7, 2.3], "Iris virginica"],
    [[5.6, 2.8, 4.9, 2.0], "Iris virginica"],
    [[7.7, 2.8, 6.7, 2.0], "Iris virginica"],
    [[6.3, 2.7, 4.9, 1.8], "Iris virginica"],
    [[6.7, 3.3, 5.7, 2.1], "Iris virginica"],
    [[7.2, 3.2, 6.0, 1.8], "Iris virginica"],
    [[6.2, 2.8, 4.8, 1.8], "Iris virginica"],
    [[6.1, 3.0, 4.9, 1.8], "Iris virginica"],
    [[6.4, 2.8, 5.6, 2.1], "Iris virginica"],
    [[7.2, 3.0, 5.8, 1.6], "Iris virginica"],
    [[7.4, 2.8, 6.1, 1.9], "Iris virginica"],
    [[7.9, 3.8, 6.4, 2.0], "Iris virginica"],
    [[6.4, 2.8, 5.6, 2.2], "Iris virginica"]
  ];
  const testDataset = [
    [[5.5, 4.2, 1.4, 0.2], "Iris setosa"],
    [[4.9, 3.1, 1.5, 0.1], "Iris setosa"],
    [[5.0, 3.2, 1.2, 0.2], "Iris setosa"],
    [[5.5, 3.5, 1.3, 0.2], "Iris setosa"],
    [[4.9, 3.1, 1.5, 0.1], "Iris setosa"],
    [[4.4, 3.0, 1.3, 0.2], "Iris setosa"],
    [[5.1, 3.4, 1.5, 0.2], "Iris setosa"],
    [[5.0, 3.5, 1.3, 0.3], "Iris setosa"],
    [[4.5, 2.3, 1.3, 0.3], "Iris setosa"],
    [[4.4, 3.2, 1.3, 0.2], "Iris setosa"],
    [[5.0, 3.5, 1.6, 0.6], "Iris setosa"],
    [[5.1, 3.8, 1.9, 0.4], "Iris setosa"],
    [[4.8, 3.0, 1.4, 0.3], "Iris setosa"],
    [[5.1, 3.8, 1.6, 0.2], "Iris setosa"],
    [[4.6, 3.2, 1.4, 0.2], "Iris setosa"],
    [[5.3, 3.7, 1.5, 0.2], "Iris setosa"],
    [[5.0, 3.3, 1.4, 0.2], "Iris setosa"],
    [[6.0, 2.7, 5.1, 1.6], "Iris versicolor"],
    [[5.4, 3.0, 4.5, 1.5], "Iris versicolor"],
    [[6.0, 3.4, 4.5, 1.6], "Iris versicolor"],
    [[6.7, 3.1, 4.7, 1.5], "Iris versicolor"],
    [[6.3, 2.3, 4.4, 1.3], "Iris versicolor"],
    [[5.6, 3.0, 4.1, 1.3], "Iris versicolor"],
    [[5.5, 2.5, 4.0, 1.3], "Iris versicolor"],
    [[5.5, 2.6, 4.4, 1.2], "Iris versicolor"],
    [[6.1, 3.0, 4.6, 1.4], "Iris versicolor"],
    [[5.8, 2.6, 4.0, 1.2], "Iris versicolor"],
    [[5.0, 2.3, 3.3, 1.0], "Iris versicolor"],
    [[5.6, 2.7, 4.2, 1.3], "Iris versicolor"],
    [[5.7, 3.0, 4.2, 1.2], "Iris versicolor"],
    [[5.7, 2.9, 4.2, 1.3], "Iris versicolor"],
    [[6.2, 2.9, 4.3, 1.3], "Iris versicolor"],
    [[5.1, 2.5, 3.0, 1.1], "Iris versicolor"],
    [[5.7, 2.8, 4.1, 1.3], "Iris versicolor"],
    [[6.3, 2.8, 5.1, 1.5], "Iris virginica"],
    [[6.1, 2.6, 5.6, 1.4], "Iris virginica"],
    [[7.7, 3.0, 6.1, 2.3], "Iris virginica"],
    [[6.3, 3.4, 5.6, 2.4], "Iris virginica"],
    [[6.4, 3.1, 5.5, 1.8], "Iris virginica"],
    [[6.0, 3.0, 4.8, 1.8], "Iris virginica"],
    [[6.9, 3.1, 5.4, 2.1], "Iris virginica"],
    [[6.7, 3.1, 5.6, 2.4], "Iris virginica"],
    [[6.9, 3.1, 5.1, 2.3], "Iris virginica"],
    [[5.8, 2.7, 5.1, 1.9], "Iris virginica"],
    [[6.8, 3.2, 5.9, 2.3], "Iris virginica"],
    [[6.7, 3.3, 5.7, 2.5], "Iris virginica"],
    [[6.7, 3.0, 5.2, 2.3], "Iris virginica"],
    [[6.3, 2.5, 5.0, 1.9], "Iris virginica"],
    [[6.5, 3.0, 5.2, 2.0], "Iris virginica"],
    [[6.2, 3.4, 5.4, 2.3], "Iris virginica"],
    [[5.9, 3.0, 5.1, 1.8], "Iris virginica"]
  ];
  const { update, weights } = averagedPerceptron();
  for (let epoch = 0; epoch < 1000; epoch += 1) {
    // We should shuffle the training dataset, but we want a deterministic test
    trainingDataset.forEach(([features, label]) => update(features, label));
  }

  // Init a new perceptron from the weights obtained during training (optional)
  const { predict } = averagedPerceptron(weights());
  const errors = testDataset.reduce(
    (errors, [features, label]) => errors + (predict(features) !== label),
    0
  );
  equal(errors < 5, true, "predicts with less than 10% error rate");
  end();
});

test("averaged-perceptron with an invalid first argument", ({
  throws,
  end
}) => {
  throws(() => averagedPerceptron(true), TypeError);
  throws(() => averagedPerceptron(false), TypeError);
  throws(() => averagedPerceptron(null), TypeError);
  throws(() => averagedPerceptron(0), TypeError);
  throws(() => averagedPerceptron(""), TypeError);
  throws(() => averagedPerceptron(() => {}), TypeError);
  end();
});

test("averaged-perceptron with an invalid second argument", ({
  throws,
  end
}) => {
  throws(() => averagedPerceptron({}, true), RangeError);
  throws(() => averagedPerceptron({}, false), RangeError);
  throws(() => averagedPerceptron({}, null), RangeError);
  throws(() => averagedPerceptron({}, ""), RangeError);
  throws(() => averagedPerceptron({}, () => {}), RangeError);
  throws(() => averagedPerceptron({}, Infinity), RangeError);
  throws(() => averagedPerceptron({}, Number.POSITIVE_INFINITY), RangeError);
  throws(() => averagedPerceptron({}, Number.MAX_VALUE), RangeError);
  throws(() => averagedPerceptron({}, NaN), RangeError);
  throws(() => averagedPerceptron({}, 1.5), RangeError);
  throws(() => averagedPerceptron({}, -1), RangeError);
  end();
});

test("averaged-perceptron with correct arguments", ({ deepEqual, end }) => {
  deepEqual(
    averagedPerceptron({ 0: { 0: 1, 1: 2 }, 1: { 0: 3, 1: 4 } }).weights(),
    { 0: { 0: 1, 1: 2 }, 1: { 0: 3, 1: 4 } },
    "accepts objects"
  );
  deepEqual(
    averagedPerceptron({ 0: { 0: 1, 1: 2 }, 1: { 0: 3, 1: 4 } }, 1).weights(),
    { 0: { 0: 1, 1: 2 }, 1: { 0: 3, 1: 4 } },
    "accepts objects as weights and epochs argument"
  );
  deepEqual(
    averagedPerceptron([
      [1, 2],
      [3, 4]
    ]).weights(),
    { 0: { 0: 1, 1: 2 }, 1: { 0: 3, 1: 4 } },
    "accepts arrays"
  );
  deepEqual(
    averagedPerceptron([
      [1, 2],
      [3, 4]
    ]).weights(),
    { 0: { 0: 1, 1: 2 }, 1: { 0: 3, 1: 4 } },
    "accepts arrays and epochs argument"
  );
  end();
});

test("averaged-perceptron predict() with no arguments and no weigths", ({
  deepEqual,
  end
}) => {
  deepEqual(averagedPerceptron().predict(), "", 'returns "" given nothing');
  deepEqual(
    averagedPerceptron().predict({}),
    "",
    'returns "" given an empty object'
  );
  end();
});

test("averaged-perceptron predict() with no arguments and with weigths", ({
  deepEqual,
  end
}) => {
  deepEqual(
    averagedPerceptron({
      x: { a: 0.4, b: 0.6 },
      y: { a: 0.8, b: -0.4 }
    }).predict(),
    "",
    'returns "" given nothing and non-empty weights'
  );
  deepEqual(
    averagedPerceptron({
      x: { a: 0.4, b: 0.6 },
      y: { a: 0.8, b: -0.4 }
    }).predict({}),
    "",
    'returns "" given an empty object and non-empty weights'
  );
  end();
});

test("averaged-perceptron predict() with no weights", ({ deepEqual, end }) => {
  deepEqual(
    averagedPerceptron().predict({ x: 1 }),
    "",
    'returns "" given one feature and empty weights'
  );
  deepEqual(
    averagedPerceptron().predict({ x: 1, y: 1 }),
    "",
    'returns "" given many features and empty weights'
  );
  end();
});

test("averaged-perceptron predict() with one feature", ({ deepEqual, end }) => {
  deepEqual(
    averagedPerceptron({
      x: { a: 0.4, b: 0.6 },
      y: { a: 0.8, b: -0.4 }
    }).predict({ x: 1 }),
    "b",
    "returns label with the highest score given a feature with positive value"
  );
  deepEqual(
    averagedPerceptron({
      x: { a: 0.4, b: 0.6 },
      y: { a: 0.8, b: -0.4 }
    }).predict({ x: -1 }),
    "a",
    "returns label with the highest score given a feature with negative value"
  );
  end();
});

test("averaged-perceptron predict() with many features", ({
  deepEqual,
  end
}) => {
  deepEqual(
    averagedPerceptron({
      x: { a: 0.4, b: 0.6 },
      y: { a: 0.8, b: -0.4 }
    }).predict({ x: 1, y: 0 }),
    "b",
    "returns label with the highest score given a feature with a value of 0"
  );
  deepEqual(
    averagedPerceptron({
      x: { a: 0.4, b: 0.6 },
      y: { a: 0.8, b: -0.4 }
    }).predict({ x: 1, y: 1 }),
    "a",
    "returns label with the highest score given features with equal values"
  );
  deepEqual(
    averagedPerceptron({
      x: { a: 0.4, b: 0.6 },
      y: { a: 0.8, b: -0.4 }
    }).predict({ x: 1, y: 0.2 }),
    "a",
    "returns label with the highest score given a feature with decimal value"
  );
  end();
});

test("averaged-perceptron scores() with no arguments and no weigths", ({
  deepEqual,
  end
}) => {
  deepEqual(
    averagedPerceptron().scores(),
    {},
    "returns empty scores given nothing"
  );
  deepEqual(
    averagedPerceptron().scores({}),
    {},
    "returns empty scores given an empty object"
  );
  end();
});

test("averaged-perceptron scores() with no arguments and with weigths", ({
  deepEqual,
  end
}) => {
  deepEqual(
    averagedPerceptron({
      x: { a: 0.4, b: 0.6 },
      y: { a: 0.8, b: -0.4 }
    }).scores(),
    {},
    "returns empty scores given nothing and non-empty weights"
  );
  deepEqual(
    averagedPerceptron({
      x: { a: 0.4, b: 0.6 },
      y: { a: 0.8, b: -0.4 }
    }).scores({}),
    {},
    "returns empty scores given an empty object and non-empty weights"
  );
  end();
});

test("averaged-perceptron scores() with no weights", ({ deepEqual, end }) => {
  deepEqual(
    averagedPerceptron().scores({ x: 1 }),
    {},
    "returns empty scores given one feature and empty weights"
  );
  deepEqual(
    averagedPerceptron().scores({ x: 1, y: 1 }),
    {},
    "returns empty scores given many features and empty weights"
  );
  end();
});

test("averaged-perceptron scores() with one feature", ({ deepEqual, end }) => {
  deepEqual(
    averagedPerceptron({
      x: { a: 0.4, b: 0.6 },
      y: { a: 0.8, b: -0.4 }
    }).scores({ x: 1 }),
    { a: 0.4, b: 0.6 },
    "returns positive scores given a feature with positive value"
  );
  deepEqual(
    averagedPerceptron({
      x: { a: 0.4, b: 0.6 },
      y: { a: 0.8, b: -0.4 }
    }).scores({ x: -1 }),
    { a: -0.4, b: -0.6 },
    "returns negative scores given a feature with negative value"
  );
  end();
});

test("averaged-perceptron scores() with many features", ({
  deepEqual,
  end
}) => {
  deepEqual(
    averagedPerceptron({
      x: { a: 0.4, b: 0.6 },
      y: { a: 0.8, b: -0.4 }
    }).scores({ x: 1, y: 0 }),
    { a: 0.4, b: 0.6 },
    "returns scores given a feature with a value of 0"
  );
  deepEqual(
    averagedPerceptron({
      x: { a: 0.4, b: 0.6 },
      y: { a: 0.8, b: -0.4 }
    }).scores({ x: 1, y: 1 }),
    { a: 0.4 + 0.8, b: 0.6 - 0.4 },
    "returns scores given features with equal values"
  );
  deepEqual(
    averagedPerceptron({
      x: { a: 0.4, b: 0.6 },
      y: { a: 0.8, b: -0.4 }
    }).scores({ x: 1, y: 0.2 }),
    { a: 0.4 + 0.8 * 0.2, b: 0.6 - 0.4 * 0.2 },
    "returns scores given a feature with decimal value"
  );
  end();
});

test("averaged-perceptron update() with no arguments and no weights", ({
  deepEqual,
  end
}) => {
  deepEqual(
    averagedPerceptron()
      .update()
      .weights(),
    {},
    "returns empty weights given nothing"
  );
  deepEqual(
    averagedPerceptron()
      .update({})
      .weights(),
    {},
    "returns empty weights given an empty object"
  );
  end();
});

test("averaged-perceptron update() with no arguments and with weights", ({
  deepEqual,
  end
}) => {
  deepEqual(
    averagedPerceptron({ x: { a: 0.4, b: 0.6 }, y: { a: 0.8, b: -0.4 } })
      .update()
      .weights(),
    { x: { a: 0.4, b: 0.6 }, y: { a: 0.8, b: -0.4 } },
    "returns the same weights given nothing"
  );
  deepEqual(
    averagedPerceptron({ x: { a: 0.4, b: 0.6 }, y: { a: 0.8, b: -0.4 } })
      .update({})
      .weights(),
    { x: { a: 0.4, b: 0.6 }, y: { a: 0.8, b: -0.4 } },
    "returns the same weights given an empty object"
  );
  end();
});

test("averaged-perceptron update() with weights without classes", ({
  deepEqual,
  end
}) => {
  deepEqual(
    averagedPerceptron({ x: {}, y: {} })
      .update({ x: 1, y: 1 }, "a")
      .weights(),
    { x: { a: 1 }, y: { a: 1 } },
    "returns all classes given a wrong prediction"
  );
  deepEqual(
    averagedPerceptron({ x: {}, y: {} })
      .update({ x: 1, y: 1 }, "")
      .weights(),
    { x: {}, y: {} },
    "returns empty features given a correct prediction"
  );
  end();
});

test("averaged-perceptron update() with weights with some features", ({
  deepEqual,
  end
}) => {
  deepEqual(
    averagedPerceptron({ x: { a: 0.4, b: 0.6 } })
      .update({ x: 1, y: 1 }, "a")
      .weights(),
    { x: { a: 0.4 + 1, b: 0.6 - 1 }, y: { a: 1, b: -1 } },
    "returns all features given a wrong prediction"
  );
  deepEqual(
    averagedPerceptron({ x: { a: 0.4, b: 0.6 } })
      .update({ x: 1, y: 1 }, "b")
      .weights(),
    { x: { a: 0.4, b: 0.6 } },
    "returns some features given a correct prediction"
  );
  end();
});

test("averaged-perceptron update() with weights with all features", ({
  deepEqual,
  end
}) => {
  deepEqual(
    averagedPerceptron({
      x: { a: 0.4, b: 0.6 },
      y: { a: 0.8, b: -0.4 }
    })
      .update({ x: 1, y: 2 }, "b")
      .weights(),
    {
      x: { a: 0.4 - 1, b: 0.6 + 1 },
      y: { a: 0.8 - 2, b: -0.4 + 2 }
    },
    "returns all features given a wrong prediction and all features"
  );
  deepEqual(
    averagedPerceptron({
      x: { a: 0.4, b: 0.6 },
      y: { a: 0.8, b: -0.4 }
    })
      .update({ x: 1, y: 1 }, "a")
      .weights(),
    {
      x: { a: 0.4, b: 0.6 },
      y: { a: 0.8, b: -0.4 }
    },
    "returns all features given a correct prediction and all features"
  );
  end();
});

test("averaged-perceptron weights() with no weights", ({ deepEqual, end }) => {
  deepEqual(
    averagedPerceptron().weights(),
    {},
    "returns same weights given nothing"
  );
  deepEqual(
    averagedPerceptron({}).weights(),
    {},
    "returns same weights given an empty object"
  );
  end();
});

test("averaged-perceptron weights() with initial weights", ({
  deepEqual,
  end
}) => {
  deepEqual(
    averagedPerceptron({}).weights(),
    {},
    "returns empty weights given empty weights"
  );
  deepEqual(
    averagedPerceptron({
      x: { a: 0.4, b: 0.6 },
      y: { a: 0.8, b: -0.4 }
    }).weights(),
    {
      x: { a: 0.4, b: 0.6 },
      y: { a: 0.8, b: -0.4 }
    },
    "returns same weights given no updates"
  );
  end();
});

test("averaged-perceptron weights() with correct predictions", ({
  deepEqual,
  end
}) => {
  deepEqual(
    averagedPerceptron({
      x: { a: 0.4, b: 0.6 },
      y: { a: 0.8, b: -0.4 }
    })
      .update({ x: 1, y: 1 }, "a")
      .weights(),
    {
      x: { a: 0.4, b: 0.6 },
      y: { a: 0.8, b: -0.4 }
    },
    "returns same weights given a correct prediction"
  );
  deepEqual(
    averagedPerceptron({
      x: { a: 0.4, b: 0.6 },
      y: { a: 0.8, b: -0.4 }
    })
      .update({ x: 1, y: 1 }, "a")
      .update({ x: 1, y: 1 }, "a")
      .weights(),
    {
      x: { a: 0.4, b: 0.6 },
      y: { a: 0.8, b: -0.4 }
    },
    "returns same weights given many correct predictions"
  );
  end();
});

test("averaged-perceptron weights() with wrong predictions", ({
  deepEqual,
  end
}) => {
  deepEqual(
    averagedPerceptron({
      x: { a: 0.4, b: 0.6 },
      y: { a: 0.8, b: -0.4 }
    })
      .update({ x: 1, y: 2 }, "b")
      .weights(),
    {
      x: { a: 0.4 - 1, b: 0.6 + 1 },
      y: { a: 0.8 - 2, b: -0.4 + 2 }
    },
    "returns averages ignoring the initial weights given one wrong prediction"
  );
  deepEqual(
    averagedPerceptron({
      x: { a: 0.4, b: 0.6 },
      y: { a: 0.8, b: -0.4 }
    })
      .update({ x: 1, y: 2 }, "b")
      .update({ x: 3, y: 4 }, "a")
      .weights(),
    {
      x: {
        a: (0.4 - 1 + (0.4 - 1 + 3)) / 2,
        b: (0.6 + 1 + (0.6 + 1 - 3)) / 2
      },
      y: {
        a: (0.8 - 2 + (0.8 - 2 + 4)) / 2,
        b: (-0.4 + 2 + (-0.4 + 2 - 4)) / 2
      }
    },
    "returns averages ignoring the initial weights given many wrong predictions"
  );
  end();
});

test("averaged-perceptron weights() with already trained weights", ({
  deepEqual,
  end
}) => {
  deepEqual(
    averagedPerceptron(
      {
        x: { a: 0.4, b: 0.6 },
        y: { a: 0.8, b: -0.4 }
      },
      3
    )
      .update({ x: 1, y: 2 }, "b")
      .weights(),
    {
      x: { a: (3 * 0.4 + (0.4 - 1)) / 4, b: (3 * 0.6 + (0.6 + 1)) / 4 },
      y: { a: (3 * 0.8 + (0.8 - 2)) / 4, b: (3 * -0.4 + (-0.4 + 2)) / 4 }
    },
    "returns averages given one wrong prediction and already trained weights"
  );
  deepEqual(
    averagedPerceptron(
      {
        x: { a: 0.4, b: 0.6 },
        y: { a: 0.8, b: -0.4 }
      },
      3
    )
      .update({ x: 1, y: 2 }, "b")
      .update({ x: 3, y: 4 }, "a")
      .weights(),
    {
      x: {
        a: (3 * 0.4 + (0.4 - 1 + (0.4 - 1 + 3))) / 5,
        b: (3 * 0.6 + (0.6 + 1 + (0.6 + 1 - 3))) / 5
      },
      y: {
        a: (3 * 0.8 + (0.8 - 2 + (0.8 - 2 + 4))) / 5,
        b: (3 * -0.4 + (-0.4 + 2 + (-0.4 + 2 - 4))) / 5
      }
    },
    "returns averages given many wrong predictions and already trained weights"
  );
  end();
});
