import test from "./node_modules/tape/index.js";
import averagedPerceptron from "./index.js";
import shuffle from "./node_modules/array-shuffle/index.js";

test("averaged-perceptron with update() and then predict()", ({
  equal,
  end
}) => {
  const { predict, update } = averagedPerceptron();
  const examples = [
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
    const shuffled = shuffle(examples);
    shuffled.forEach(([features, actual]) => update(features, actual));
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
  throws(() => averagedPerceptron({}, -Infinity), RangeError);
  throws(() => averagedPerceptron({}, Number.POSITIVE_INFINITY), RangeError);
  throws(() => averagedPerceptron({}, Number.NEGATIVE_INFINITY), RangeError);
  throws(() => averagedPerceptron({}, Number.MAX_VALUE), RangeError);
  throws(() => averagedPerceptron({}, Number.MIN_VALUE), RangeError);
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
    "accepts objects as weights and iterations argument"
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
    "accepts arrays and iterations argument"
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
