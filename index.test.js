import averagedPerceptron from "./index.js";

test("averaged-perceptron with the Iris dataset", () => {
  // Note that the features can be either an object or an array
  const trainingDataset = [
    { features: [5.1, 3.5, 1.4, 0.2], label: "Iris setosa" },
    { features: [4.9, 3.0, 1.4, 0.2], label: "Iris setosa" },
    { features: [4.7, 3.2, 1.3, 0.2], label: "Iris setosa" },
    { features: [4.6, 3.1, 1.5, 0.2], label: "Iris setosa" },
    { features: [5.0, 3.6, 1.4, 0.2], label: "Iris setosa" },
    { features: [5.4, 3.9, 1.7, 0.4], label: "Iris setosa" },
    { features: [4.6, 3.4, 1.4, 0.3], label: "Iris setosa" },
    { features: [5.0, 3.4, 1.5, 0.2], label: "Iris setosa" },
    { features: [4.4, 2.9, 1.4, 0.2], label: "Iris setosa" },
    { features: [4.9, 3.1, 1.5, 0.1], label: "Iris setosa" },
    { features: [5.4, 3.7, 1.5, 0.2], label: "Iris setosa" },
    { features: [4.8, 3.4, 1.6, 0.2], label: "Iris setosa" },
    { features: [4.8, 3.0, 1.4, 0.1], label: "Iris setosa" },
    { features: [4.3, 3.0, 1.1, 0.1], label: "Iris setosa" },
    { features: [5.8, 4.0, 1.2, 0.2], label: "Iris setosa" },
    { features: [5.7, 4.4, 1.5, 0.4], label: "Iris setosa" },
    { features: [5.4, 3.9, 1.3, 0.4], label: "Iris setosa" },
    { features: [5.1, 3.5, 1.4, 0.3], label: "Iris setosa" },
    { features: [5.7, 3.8, 1.7, 0.3], label: "Iris setosa" },
    { features: [5.1, 3.8, 1.5, 0.3], label: "Iris setosa" },
    { features: [5.4, 3.4, 1.7, 0.2], label: "Iris setosa" },
    { features: [5.1, 3.7, 1.5, 0.4], label: "Iris setosa" },
    { features: [4.6, 3.6, 1.0, 0.2], label: "Iris setosa" },
    { features: [5.1, 3.3, 1.7, 0.5], label: "Iris setosa" },
    { features: [4.8, 3.4, 1.9, 0.2], label: "Iris setosa" },
    { features: [5.0, 3.0, 1.6, 0.2], label: "Iris setosa" },
    { features: [5.0, 3.4, 1.6, 0.4], label: "Iris setosa" },
    { features: [5.2, 3.5, 1.5, 0.2], label: "Iris setosa" },
    { features: [5.2, 3.4, 1.4, 0.2], label: "Iris setosa" },
    { features: [4.7, 3.2, 1.6, 0.2], label: "Iris setosa" },
    { features: [4.8, 3.1, 1.6, 0.2], label: "Iris setosa" },
    { features: [5.4, 3.4, 1.5, 0.4], label: "Iris setosa" },
    { features: [5.2, 4.1, 1.5, 0.1], label: "Iris setosa" },
    { features: [7.0, 3.2, 4.7, 1.4], label: "Iris versicolor" },
    { features: [6.4, 3.2, 4.5, 1.5], label: "Iris versicolor" },
    { features: [6.9, 3.1, 4.9, 1.5], label: "Iris versicolor" },
    { features: [5.5, 2.3, 4.0, 1.3], label: "Iris versicolor" },
    { features: [6.5, 2.8, 4.6, 1.5], label: "Iris versicolor" },
    { features: [5.7, 2.8, 4.5, 1.3], label: "Iris versicolor" },
    { features: [6.3, 3.3, 4.7, 1.6], label: "Iris versicolor" },
    { features: [4.9, 2.4, 3.3, 1.0], label: "Iris versicolor" },
    { features: [6.6, 2.9, 4.6, 1.3], label: "Iris versicolor" },
    { features: [5.2, 2.7, 3.9, 1.4], label: "Iris versicolor" },
    { features: [5.0, 2.0, 3.5, 1.0], label: "Iris versicolor" },
    { features: [5.9, 3.0, 4.2, 1.5], label: "Iris versicolor" },
    { features: [6.0, 2.2, 4.0, 1.0], label: "Iris versicolor" },
    { features: [6.1, 2.9, 4.7, 1.4], label: "Iris versicolor" },
    { features: [5.6, 2.9, 3.6, 1.3], label: "Iris versicolor" },
    { features: [6.7, 3.1, 4.4, 1.4], label: "Iris versicolor" },
    { features: [5.6, 3.0, 4.5, 1.5], label: "Iris versicolor" },
    { features: [5.8, 2.7, 4.1, 1.0], label: "Iris versicolor" },
    { features: [6.2, 2.2, 4.5, 1.5], label: "Iris versicolor" },
    { features: [5.6, 2.5, 3.9, 1.1], label: "Iris versicolor" },
    { features: [5.9, 3.2, 4.8, 1.8], label: "Iris versicolor" },
    { features: [6.1, 2.8, 4.0, 1.3], label: "Iris versicolor" },
    { features: [6.3, 2.5, 4.9, 1.5], label: "Iris versicolor" },
    { features: [6.1, 2.8, 4.7, 1.2], label: "Iris versicolor" },
    { features: [6.4, 2.9, 4.3, 1.3], label: "Iris versicolor" },
    { features: [6.6, 3.0, 4.4, 1.4], label: "Iris versicolor" },
    { features: [6.8, 2.8, 4.8, 1.4], label: "Iris versicolor" },
    { features: [6.7, 3.0, 5.0, 1.7], label: "Iris versicolor" },
    { features: [6.0, 2.9, 4.5, 1.5], label: "Iris versicolor" },
    { features: [5.7, 2.6, 3.5, 1.0], label: "Iris versicolor" },
    { features: [5.5, 2.4, 3.8, 1.1], label: "Iris versicolor" },
    { features: [5.5, 2.4, 3.7, 1.0], label: "Iris versicolor" },
    { features: [5.8, 2.7, 3.9, 1.2], label: "Iris versicolor" },
    { features: [6.3, 3.3, 6.0, 2.5], label: "Iris virginica" },
    { features: [5.8, 2.7, 5.1, 1.9], label: "Iris virginica" },
    { features: [7.1, 3.0, 5.9, 2.1], label: "Iris virginica" },
    { features: [6.3, 2.9, 5.6, 1.8], label: "Iris virginica" },
    { features: [6.5, 3.0, 5.8, 2.2], label: "Iris virginica" },
    { features: [7.6, 3.0, 6.6, 2.1], label: "Iris virginica" },
    { features: [4.9, 2.5, 4.5, 1.7], label: "Iris virginica" },
    { features: [7.3, 2.9, 6.3, 1.8], label: "Iris virginica" },
    { features: [6.7, 2.5, 5.8, 1.8], label: "Iris virginica" },
    { features: [7.2, 3.6, 6.1, 2.5], label: "Iris virginica" },
    { features: [6.5, 3.2, 5.1, 2.0], label: "Iris virginica" },
    { features: [6.4, 2.7, 5.3, 1.9], label: "Iris virginica" },
    { features: [6.8, 3.0, 5.5, 2.1], label: "Iris virginica" },
    { features: [5.7, 2.5, 5.0, 2.0], label: "Iris virginica" },
    { features: [5.8, 2.8, 5.1, 2.4], label: "Iris virginica" },
    { features: [6.4, 3.2, 5.3, 2.3], label: "Iris virginica" },
    { features: [6.5, 3.0, 5.5, 1.8], label: "Iris virginica" },
    { features: [7.7, 3.8, 6.7, 2.2], label: "Iris virginica" },
    { features: [7.7, 2.6, 6.9, 2.3], label: "Iris virginica" },
    { features: [6.0, 2.2, 5.0, 1.5], label: "Iris virginica" },
    { features: [6.9, 3.2, 5.7, 2.3], label: "Iris virginica" },
    { features: [5.6, 2.8, 4.9, 2.0], label: "Iris virginica" },
    { features: [7.7, 2.8, 6.7, 2.0], label: "Iris virginica" },
    { features: [6.3, 2.7, 4.9, 1.8], label: "Iris virginica" },
    { features: [6.7, 3.3, 5.7, 2.1], label: "Iris virginica" },
    { features: [7.2, 3.2, 6.0, 1.8], label: "Iris virginica" },
    { features: [6.2, 2.8, 4.8, 1.8], label: "Iris virginica" },
    { features: [6.1, 3.0, 4.9, 1.8], label: "Iris virginica" },
    { features: [6.4, 2.8, 5.6, 2.1], label: "Iris virginica" },
    { features: [7.2, 3.0, 5.8, 1.6], label: "Iris virginica" },
    { features: [7.4, 2.8, 6.1, 1.9], label: "Iris virginica" },
    { features: [7.9, 3.8, 6.4, 2.0], label: "Iris virginica" },
    { features: [6.4, 2.8, 5.6, 2.2], label: "Iris virginica" },
  ];
  const testDataset = [
    { features: [5.5, 4.2, 1.4, 0.2], label: "Iris setosa" },
    { features: [4.9, 3.1, 1.5, 0.1], label: "Iris setosa" },
    { features: [5.0, 3.2, 1.2, 0.2], label: "Iris setosa" },
    { features: [5.5, 3.5, 1.3, 0.2], label: "Iris setosa" },
    { features: [4.9, 3.1, 1.5, 0.1], label: "Iris setosa" },
    { features: [4.4, 3.0, 1.3, 0.2], label: "Iris setosa" },
    { features: [5.1, 3.4, 1.5, 0.2], label: "Iris setosa" },
    { features: [5.0, 3.5, 1.3, 0.3], label: "Iris setosa" },
    { features: [4.5, 2.3, 1.3, 0.3], label: "Iris setosa" },
    { features: [4.4, 3.2, 1.3, 0.2], label: "Iris setosa" },
    { features: [5.0, 3.5, 1.6, 0.6], label: "Iris setosa" },
    { features: [5.1, 3.8, 1.9, 0.4], label: "Iris setosa" },
    { features: [4.8, 3.0, 1.4, 0.3], label: "Iris setosa" },
    { features: [5.1, 3.8, 1.6, 0.2], label: "Iris setosa" },
    { features: [4.6, 3.2, 1.4, 0.2], label: "Iris setosa" },
    { features: [5.3, 3.7, 1.5, 0.2], label: "Iris setosa" },
    { features: [5.0, 3.3, 1.4, 0.2], label: "Iris setosa" },
    { features: [6.0, 2.7, 5.1, 1.6], label: "Iris versicolor" },
    { features: [5.4, 3.0, 4.5, 1.5], label: "Iris versicolor" },
    { features: [6.0, 3.4, 4.5, 1.6], label: "Iris versicolor" },
    { features: [6.7, 3.1, 4.7, 1.5], label: "Iris versicolor" },
    { features: [6.3, 2.3, 4.4, 1.3], label: "Iris versicolor" },
    { features: [5.6, 3.0, 4.1, 1.3], label: "Iris versicolor" },
    { features: [5.5, 2.5, 4.0, 1.3], label: "Iris versicolor" },
    { features: [5.5, 2.6, 4.4, 1.2], label: "Iris versicolor" },
    { features: [6.1, 3.0, 4.6, 1.4], label: "Iris versicolor" },
    { features: [5.8, 2.6, 4.0, 1.2], label: "Iris versicolor" },
    { features: [5.0, 2.3, 3.3, 1.0], label: "Iris versicolor" },
    { features: [5.6, 2.7, 4.2, 1.3], label: "Iris versicolor" },
    { features: [5.7, 3.0, 4.2, 1.2], label: "Iris versicolor" },
    { features: [5.7, 2.9, 4.2, 1.3], label: "Iris versicolor" },
    { features: [6.2, 2.9, 4.3, 1.3], label: "Iris versicolor" },
    { features: [5.1, 2.5, 3.0, 1.1], label: "Iris versicolor" },
    { features: [5.7, 2.8, 4.1, 1.3], label: "Iris versicolor" },
    { features: [6.3, 2.8, 5.1, 1.5], label: "Iris virginica" },
    { features: [6.1, 2.6, 5.6, 1.4], label: "Iris virginica" },
    { features: [7.7, 3.0, 6.1, 2.3], label: "Iris virginica" },
    { features: [6.3, 3.4, 5.6, 2.4], label: "Iris virginica" },
    { features: [6.4, 3.1, 5.5, 1.8], label: "Iris virginica" },
    { features: [6.0, 3.0, 4.8, 1.8], label: "Iris virginica" },
    { features: [6.9, 3.1, 5.4, 2.1], label: "Iris virginica" },
    { features: [6.7, 3.1, 5.6, 2.4], label: "Iris virginica" },
    { features: [6.9, 3.1, 5.1, 2.3], label: "Iris virginica" },
    { features: [5.8, 2.7, 5.1, 1.9], label: "Iris virginica" },
    { features: [6.8, 3.2, 5.9, 2.3], label: "Iris virginica" },
    { features: [6.7, 3.3, 5.7, 2.5], label: "Iris virginica" },
    { features: [6.7, 3.0, 5.2, 2.3], label: "Iris virginica" },
    { features: [6.3, 2.5, 5.0, 1.9], label: "Iris virginica" },
    { features: [6.5, 3.0, 5.2, 2.0], label: "Iris virginica" },
    { features: [6.2, 3.4, 5.4, 2.3], label: "Iris virginica" },
    { features: [5.9, 3.0, 5.1, 1.8], label: "Iris virginica" },
  ];

  const { update, weights } = averagedPerceptron();
  for (let epoch = 0; epoch < 1000; epoch += 1) {
    // We should shuffle the training dataset, but we want a deterministic test
    trainingDataset.forEach(({ features, label }) => update(features, label));
  }

  // Init a new perceptron from the weights obtained during training (optional)
  const { predict } = averagedPerceptron(weights());
  const errors = testDataset.reduce(
    (errors, { features, label }) =>
      errors + (predict(features) !== label ? 1 : 0),
    0,
  );

  // The dataset is not fully linearly separable, so some errors are expected
  expect(errors < 5).toBe(true);
});

test("averaged-perceptron factory function with invalid options", () => {
  expect(() => averagedPerceptron({}, Infinity)).toThrow(RangeError);
  expect(() => averagedPerceptron({}, NaN)).toThrow(RangeError);
  expect(() => averagedPerceptron({}, 1.5)).toThrow(RangeError);
  expect(() => averagedPerceptron({}, -1)).toThrow(RangeError);
});

test("averaged-perceptron factory function with weights object", () => {
  expect(
    averagedPerceptron({ 0: { 0: 1, 1: 2 }, 1: { 0: 3, 1: 4 } }).weights(),
  ).toEqual({
    0: { 0: 1, 1: 2 },
    1: { 0: 3, 1: 4 },
  });
});

test("averaged-perceptron factory function with weights object and options", () => {
  expect(
    averagedPerceptron({ 0: { 0: 1, 1: 2 }, 1: { 0: 3, 1: 4 } }, 1).weights(),
  ).toEqual({
    0: { 0: 1, 1: 2 },
    1: { 0: 3, 1: 4 },
  });
});

test("averaged-perceptron factory function with weights array", () => {
  expect(
    averagedPerceptron([
      { 0: 1, 1: 2 },
      { 0: 3, 1: 4 },
    ]).weights(),
  ).toEqual({
    0: { 0: 1, 1: 2 },
    1: { 0: 3, 1: 4 },
  });
});

test("averaged-perceptron factory function with weights array and options", () => {
  expect(
    averagedPerceptron(
      [
        { 0: 1, 1: 2 },
        { 0: 3, 1: 4 },
      ],
      1,
    ).weights(),
  ).toEqual({
    0: { 0: 1, 1: 2 },
    1: { 0: 3, 1: 4 },
  });
});

test("averaged-perceptron predict() with features", () => {
  const model = averagedPerceptron({
    x: { a: 0.4, b: 0.6 },
    y: { a: 0.8, b: -0.4 },
  });
  expect(model.predict({ x: 1 })).toEqual("b");
  expect(model.predict({ x: -1 })).toEqual("a");
  expect(model.predict({ x: 1, y: 0 })).toEqual("b");
  expect(model.predict({ x: 1, y: 1 })).toEqual("a");
});

test("averaged-perceptron update() with a correct label", () => {
  expect(
    averagedPerceptron({ x: { a: 0.4, b: 0.6 }, y: { a: 0.8, b: -0.4 } })
      .update({ x: 1, y: 1 }, "a")
      .weights(),
  ).toEqual({ x: { a: 0.4, b: 0.6 }, y: { a: 0.8, b: -0.4 } });
});

test("averaged-perceptron update() with a wrong label", () => {
  expect(
    averagedPerceptron({ x: { a: 0.4, b: 0.6 }, y: { a: 0.8, b: -0.4 } })
      .update({ x: 1, y: 2 }, "b")
      .weights(),
  ).toEqual({ x: { a: 0.4 - 1, b: 0.6 + 1 }, y: { a: 0.8 - 2, b: -0.4 + 2 } });
});

test("averaged-perceptron update() with unknown features", () => {
  expect(
    averagedPerceptron({ x: { a: 0.4, b: 0.6 } })
      .update({ x: 1, y: 1 }, "a")
      .weights(),
  ).toEqual({ x: { a: 0.4 + 1, b: 0.6 - 1 }, y: { a: 1, b: -1 } });
});

test("averaged-perceptron update() with unknown labels", () => {
  expect(
    averagedPerceptron({ x: {}, y: {} }).update({ x: 1, y: 1 }, "a").weights(),
  ).toEqual({ x: { a: 1 }, y: { a: 1 } });
});

test("averaged-perceptron weights() with uninitialized weights", () => {
  expect(averagedPerceptron().weights()).toEqual({});
});

test("averaged-perceptron weights() with unmodified weights", () => {
  const model = averagedPerceptron({
    x: { a: 0.4, b: 0.6 },
    y: { a: 0.8, b: -0.4 },
  });
  expect(model.weights()).toEqual({
    x: { a: 0.4, b: 0.6 },
    y: { a: 0.8, b: -0.4 },
  });
  expect(model.update({ x: 1, y: 1 }, "a").weights()).toEqual({
    x: { a: 0.4, b: 0.6 },
    y: { a: 0.8, b: -0.4 },
  });
  expect(model.update({ x: 1, y: 1 }, "a").weights()).toEqual({
    x: { a: 0.4, b: 0.6 },
    y: { a: 0.8, b: -0.4 },
  });
});

test("averaged-perceptron weights() with updated weights", () => {
  const model = averagedPerceptron({
    x: { a: 0.4, b: 0.6 },
    y: { a: 0.8, b: -0.4 },
  });
  expect(model.update({ x: 1, y: 2 }, "b").weights()).toEqual({
    x: { a: 0.4 - 1, b: 0.6 + 1 },
    y: { a: 0.8 - 2, b: -0.4 + 2 },
  });
  expect(model.update({ x: 3, y: 4 }, "a").weights()).toEqual({
    x: { a: (0.4 - 1 + (0.4 - 1 + 3)) / 2, b: (0.6 + 1 + (0.6 + 1 - 3)) / 2 },
    y: { a: (0.8 - 2 + (0.8 - 2 + 4)) / 2, b: (-0.4 + 2 + (-0.4 + 2 - 4)) / 2 },
  });
});

test("averaged-perceptron weights() with updated weights and options", () => {
  const model = averagedPerceptron(
    { x: { a: 0.4, b: 0.6 }, y: { a: 0.8, b: -0.4 } },
    3,
  );
  expect(model.update({ x: 1, y: 2 }, "b").weights()).toEqual({
    x: { a: (3 * 0.4 + (0.4 - 1)) / 4, b: (3 * 0.6 + (0.6 + 1)) / 4 },
    y: { a: (3 * 0.8 + (0.8 - 2)) / 4, b: (3 * -0.4 + (-0.4 + 2)) / 4 },
  });
  expect(model.update({ x: 3, y: 4 }, "a").weights()).toEqual({
    x: {
      a: (3 * 0.4 + (0.4 - 1 + (0.4 - 1 + 3))) / 5,
      b: (3 * 0.6 + (0.6 + 1 + (0.6 + 1 - 3))) / 5,
    },
    y: {
      a: (3 * 0.8 + (0.8 - 2 + (0.8 - 2 + 4))) / 5,
      b: (3 * -0.4 + (-0.4 + 2 + (-0.4 + 2 - 4))) / 5,
    },
  });
});
