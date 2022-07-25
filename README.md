# averaged-perceptron

[![build status](https://github.com/rtomrud/averaged-perceptron/workflows/build/badge.svg)](https://github.com/rtomrud/averaged-perceptron/actions?query=branch%3Amaster+workflow%3Aci)
[![npm version](https://badgen.net/npm/v/averaged-perceptron)](https://www.npmjs.com/package/averaged-perceptron)
[![bundle size](https://badgen.net/bundlephobia/minzip/averaged-perceptron)](https://bundlephobia.com/result?p=averaged-perceptron)

A linear classifier with the averaged [perceptron](https://en.wikipedia.org/wiki/Perceptron) algorithm

## Installing

```bash
npm install averaged-perceptron
```

## Using

A simple (and unrealistic) example:

```js
import averagedPerceptron from "averaged-perceptron";

const { predict, update } = averagedPerceptron();
const trainingDataset = [
  [{ height: 4, width: 2 }, "slim"],
  [{ height: 2, width: 4 }, "fat"],
  [{ height: 1, width: 4 }, "fat"],
  [{ height: 2, width: 2.1 }, "fat"],
  [{ height: 2.1, width: 2 }, "slim"],
  [{ height: 2, width: 1 }, "slim"],
  [{ height: 1, width: 2 }, "fat"],
  [{ height: 1, width: 1.1 }, "fat"],
  [{ height: 1.1, width: 1 }, "slim"],
  [{ height: 4, width: 1 }, "slim"],
];
const epochs = 1000;
for (let epoch = 0; epoch < epochs; epoch += 1) {
  const shuffledDataset = shuffle(trainingDataset); // Any Fisher–Yates shuffle
  shuffledDataset.forEach(([features, label]) => update(features, label));
}

predict({ height: 8, width: 2 }); // => "slim"
predict({ height: 2.1, width: 2 }); // => "slim"
predict({ height: 2, width: 2.1 }); // => "fat"
predict({ height: 2, width: 8 }); // => "fat"
```

A slightly more realistic example using the [Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) can be found in [the tests](./index.test.js).

## API

### `averagedPerceptron([weights [, iterations]])`

Returns a perceptron object. It may be initialized with `weights`, an object of objects with the weight of each feature-label pair. When initialized with `weights`, the number of iterations used to obtain them are `iterations`, or `0` by default.

```js
import averagedPerceptron from "averaged-perceptron";

// Create a new perceptron
const { predict, update, weights } = averagedPerceptron();
```

If you want to train the model in multiple sessions, you may resume training by specifying the `iterations`, which is the number of times `update()` was called to obtain the weights. That way new `update()` calls are properly averaged against the pretrained `weights`.

```js
import averagedPerceptron from "averaged-perceptron";

// Create a perceptron from pretrained weights to do further training
const weightsJSON = '{"x":{"a":0.4,"b":0.6},"y":{"a":0.8,"b":-0.4}}';
const weights = JSON.parse(weightsJSON);
const iterations = 1000; // weights obtained with 1000 update() calls
const { predict, update, weights } = averagedPerceptron(weights, iterations);
// Keep training by calling update()
```

### `predict(features)`

Returns the label predicted from the values in `features`, or `""` if none exists.

```js
import averagedPerceptron from "averaged-perceptron";

const { predict } = averagedPerceptron({
  x: { a: 0.4, b: 0.6 },
  y: { a: 0.8, b: -0.4 },
});
predict({ x: 1, y: 1 }); // => "a"
```

### `update(features, label [, guess])`

Returns the perceptron, updating its weights with the respective values in `features` if `label` does not equal `guess`. If `guess` is not given, it defaults to the output of `predict(features)`.

```js
import averagedPerceptron from "averaged-perceptron";

const { update } = averagedPerceptron();
update({ x: 1, y: 1 }, "a");
```

_Note that `update()` may be given feature-label pairs whose weights have not been preinitialized, so the model may be used for [online learning](https://en.wikipedia.org/wiki/Online_machine_learning) when the features or labels are unknown a priori._

### `weights()`

Returns an object of objects with the weight of each feature-label pair.

```js
import averagedPerceptron from "averaged-perceptron";

const { weights } = averagedPerceptron({
  x: { a: 0.4, b: 0.6 },
  y: { a: 0.8, b: -0.4 },
});
weights(); // => { x: { a: 0.4, b: 0.6 }, y: { a: 0.8, b: -0.4 } }
```

_Note that the weights are stored as an object of objects, because this perceptron is optimized for sparse features._

## License

[MIT](./LICENSE)
