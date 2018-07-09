import test from "./node_modules/tape/index.js";
import createPerceptron from "./index.js";
import shuffle from "./node_modules/array-shuffle/index.js";

test("averaged-perceptron", ({ equal, end }) => {
  const { predict, update } = createPerceptron();
  const trainingData = [
    [{ height: 4, width: 2 }, "slim"],
    [{ height: 2, width: 4 }, "fat"],
    [{ height: 1, width: 4 }, "fat"],
    [{ height: 2, width: 2.1 }, "fat"],
    [{ height: 2.1, width: 2 }, "slim"],
    [{ height: 2, width: 1 }, "slim"],
    [{ height: 1, width: 2 }, "fat"],
    [{ height: 1, width: 1.1 }, "fat"],
    [{ height: 1.1, width: 1 }, "slim"],
    [{ height: 4, width: 1 }, "slim"]
  ];
  const maxIterations = 1000;
  let iteration = 0;
  while (iteration < maxIterations) {
    const shuffledData = shuffle(trainingData);
    shuffledData.forEach(([features, actual]) => update(features, actual));
    iteration += 1;
  }

  equal(
    predict({ height: 8, width: 2 }),
    "slim",
    'returns "slim" given a height much higher than the width'
  );
  equal(
    predict({ height: 2.1, width: 2 }),
    "slim",
    'returns "slim" given a height slightly higher than the width'
  );
  equal(
    predict({ height: 2, width: 2.1 }),
    "fat",
    'returns "fat" given a height slightly lower than the width'
  );
  equal(
    predict({ height: 2, width: 8 }),
    "fat",
    'returns "fat" given a height much lower than the width'
  );
  end();
});

test("averaged-perceptron factory with an invalid first argument", ({
  throws,
  end
}) => {
  throws(() => createPerceptron(true), TypeError);
  throws(() => createPerceptron(false), TypeError);
  throws(() => createPerceptron(null), TypeError);
  throws(() => createPerceptron(0), TypeError);
  throws(() => createPerceptron(""), TypeError);
  throws(() => createPerceptron(() => {}), TypeError);
  end();
});

test("averaged-perceptron factory with an invalid second argument", ({
  throws,
  end
}) => {
  throws(() => createPerceptron({}, true), TypeError);
  throws(() => createPerceptron({}, false), TypeError);
  throws(() => createPerceptron({}, null), TypeError);
  throws(() => createPerceptron({}, ""), TypeError);
  throws(() => createPerceptron({}, () => {}), TypeError);
  throws(() => createPerceptron({}, Infinity), TypeError);
  throws(() => createPerceptron({}, -Infinity), TypeError);
  throws(() => createPerceptron({}, Number.POSITIVE_INFINITY), TypeError);
  throws(() => createPerceptron({}, Number.NEGATIVE_INFINITY), TypeError);
  throws(() => createPerceptron({}, Number.MAX_VALUE), TypeError);
  throws(() => createPerceptron({}, Number.MIN_VALUE), TypeError);
  throws(() => createPerceptron({}, NaN), TypeError);
  throws(() => createPerceptron({}, 1.5), TypeError);
  throws(() => createPerceptron({}, -1), TypeError);
  end();
});

test("averaged-perceptron factory with correct arguments", ({
  deepEqual,
  end
}) => {
  deepEqual(
    createPerceptron({ 0: { 0: 1, 1: 2 }, 1: { 0: 3, 1: 4 } }).weights(),
    { 0: { 0: 1, 1: 2 }, 1: { 0: 3, 1: 4 } },
    "accepts objects"
  );
  deepEqual(
    createPerceptron({ 0: { 0: 1, 1: 2 }, 1: { 0: 3, 1: 4 } }, 1).weights(),
    { 0: { 0: 1, 1: 2 }, 1: { 0: 3, 1: 4 } },
    "accepts objects as weights and iterations argument"
  );
  deepEqual(
    createPerceptron([[1, 2], [3, 4]]).weights(),
    { 0: { 0: 1, 1: 2 }, 1: { 0: 3, 1: 4 } },
    "accepts arrays"
  );
  deepEqual(
    createPerceptron([[1, 2], [3, 4]]).weights(),
    { 0: { 0: 1, 1: 2 }, 1: { 0: 3, 1: 4 } },
    "accepts arrays and iterations argument"
  );
  end();
});

test("averaged-perceptron predict() with no arguments and no weigths", ({
  deepEqual,
  end
}) => {
  deepEqual(createPerceptron().predict(), null, "returns null given nothing");
  deepEqual(
    createPerceptron().predict({}),
    null,
    "returns null given an empty object"
  );
  end();
});

test("averaged-perceptron predict() with no arguments and with weigths", ({
  deepEqual,
  end
}) => {
  deepEqual(
    createPerceptron({
      x: { a: 0.4, b: 0.6 },
      y: { a: 0.8, b: -0.4 }
    }).predict(),
    null,
    "returns null given nothing and non-empty weights"
  );
  deepEqual(
    createPerceptron({
      x: { a: 0.4, b: 0.6 },
      y: { a: 0.8, b: -0.4 }
    }).predict({}),
    null,
    "returns null given an empty object and non-empty weights"
  );
  end();
});

test("averaged-perceptron predict() with no weights", ({ deepEqual, end }) => {
  deepEqual(
    createPerceptron().predict({ x: 1 }),
    null,
    "returns null given one feature and empty weights"
  );
  deepEqual(
    createPerceptron().predict({ x: 1, y: 1 }),
    null,
    "returns null given many features and empty weights"
  );
  end();
});

test("averaged-perceptron predict() with one feature", ({ deepEqual, end }) => {
  deepEqual(
    createPerceptron({
      x: { a: 0.4, b: 0.6 },
      y: { a: 0.8, b: -0.4 }
    }).predict({ x: 1 }),
    "b",
    "returns label with the highest score given a feature with positive value"
  );
  deepEqual(
    createPerceptron({
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
    createPerceptron({
      x: { a: 0.4, b: 0.6 },
      y: { a: 0.8, b: -0.4 }
    }).predict({ x: 1, y: 0 }),
    "b",
    "returns label with the highest score given a feature with a value of 0"
  );
  deepEqual(
    createPerceptron({
      x: { a: 0.4, b: 0.6 },
      y: { a: 0.8, b: -0.4 }
    }).predict({ x: 1, y: 1 }),
    "a",
    "returns label with the highest score given features with equal values"
  );
  deepEqual(
    createPerceptron({
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
    createPerceptron().scores(),
    {},
    "returns empty scores given nothing"
  );
  deepEqual(
    createPerceptron().scores({}),
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
    createPerceptron({
      x: { a: 0.4, b: 0.6 },
      y: { a: 0.8, b: -0.4 }
    }).scores(),
    {},
    "returns empty scores given nothing and non-empty weights"
  );
  deepEqual(
    createPerceptron({
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
    createPerceptron().scores({ x: 1 }),
    {},
    "returns empty scores given one feature and empty weights"
  );
  deepEqual(
    createPerceptron().scores({ x: 1, y: 1 }),
    {},
    "returns empty scores given many features and empty weights"
  );
  end();
});

test("averaged-perceptron scores() with one feature", ({ deepEqual, end }) => {
  deepEqual(
    createPerceptron({
      x: { a: 0.4, b: 0.6 },
      y: { a: 0.8, b: -0.4 }
    }).scores({ x: 1 }),
    { a: 0.4, b: 0.6 },
    "returns positive scores given a feature with positive value"
  );
  deepEqual(
    createPerceptron({
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
    createPerceptron({
      x: { a: 0.4, b: 0.6 },
      y: { a: 0.8, b: -0.4 }
    }).scores({ x: 1, y: 0 }),
    { a: 0.4, b: 0.6 },
    "returns scores given a feature with a value of 0"
  );
  deepEqual(
    createPerceptron({
      x: { a: 0.4, b: 0.6 },
      y: { a: 0.8, b: -0.4 }
    }).scores({ x: 1, y: 1 }),
    { a: 0.4 + 0.8, b: 0.6 - 0.4 },
    "returns scores given features with equal values"
  );
  deepEqual(
    createPerceptron({
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
    createPerceptron()
      .update()
      .weights(),
    {},
    "returns empty weights given nothing"
  );
  deepEqual(
    createPerceptron()
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
    createPerceptron({ x: { a: 0.4, b: 0.6 }, y: { a: 0.8, b: -0.4 } })
      .update()
      .weights(),
    { x: { a: 0.4, b: 0.6 }, y: { a: 0.8, b: -0.4 } },
    "returns the same weights given nothing"
  );
  deepEqual(
    createPerceptron({ x: { a: 0.4, b: 0.6 }, y: { a: 0.8, b: -0.4 } })
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
    createPerceptron({ x: {}, y: {} })
      .update({ x: 1, y: 1 }, "a")
      .weights(),
    { x: { a: 1 }, y: { a: 1 } },
    "returns all classes given a wrong prediction"
  );
  deepEqual(
    createPerceptron({ x: {}, y: {} })
      .update({ x: 1, y: 1 }, null)
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
    createPerceptron({ x: { a: 0.4, b: 0.6 } })
      .update({ x: 1, y: 1 }, "a")
      .weights(),
    { x: { a: 0.4 + 1, b: 0.6 - 1 }, y: { a: 1, b: -1 } },
    "returns all features given a wrong prediction"
  );
  deepEqual(
    createPerceptron({ x: { a: 0.4, b: 0.6 } })
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
    createPerceptron({
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
    createPerceptron({
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
    createPerceptron().weights(),
    {},
    "returns same weights given nothing"
  );
  deepEqual(
    createPerceptron({}).weights(),
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
    createPerceptron({}).weights(),
    {},
    "returns empty weights given empty weights"
  );
  deepEqual(
    createPerceptron({
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
    createPerceptron({
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
    createPerceptron({
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
    createPerceptron({
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
    createPerceptron({
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
    createPerceptron(
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
    createPerceptron(
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
