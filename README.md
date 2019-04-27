# averaged-perceptron

[![npm version](https://img.shields.io/npm/v/averaged-perceptron.svg?style=flat-square)](https://www.npmjs.com/package/averaged-perceptron)
[![Build Status](https://travis-ci.com/rtomrud/averaged-perceptron.svg?branch=master)](https://travis-ci.com/rtomrud/averaged-perceptron)
[![Coverage Status](https://coveralls.io/repos/github/rtomrud/averaged-perceptron/badge.svg?branch=master)](https://coveralls.io/github/rtomrud/averaged-perceptron?branch=master)
[![Code size](https://badgen.net/bundlephobia/minzip/averaged-perceptron)](https://bundlephobia.com/result?p=averaged-perceptron)

A linear classifier with the averaged [perceptron](https://en.wikipedia.org/wiki/Perceptron) algorithm

- Optimized for very sparse features; weights are stored as an object (dictionary) instead of as an array (vector)
- It can be [efficiently initialized](#averagedperceptronweights-iterations) from given weights, e.g., previously trained weights returned by [`weights()`](#weights)
- Get the label with the best score with [`predict()`](#predictfeatures-scores), or all the scores with [`scores()`](#scoresfeatures)
- Efficient [`update()`](#updatefeatures-label-guess) that adds new features and labels as you go; no need to initialize all features beforehand

## Installing

```bash
npm install averaged-perceptron
```

## Using

```js
import averagedPerceptron from "averaged-perceptron";

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
  const shuffled = shuffle(examples); // A Fisher–Yates shuffle
  shuffled.forEach(([features, actual]) => update(features, actual));
  iteration += 1;
}

predict({ height: 8, width: 2 }); // => "slim"
predict({ height: 2.1, width: 2 }); // => "slim"
predict({ height: 2, width: 2.1 }); // => "fat"
predict({ height: 2, width: 8 }); // => "fat"
```

## API

### `averagedPerceptron(weights, iterations)`

Returns a perceptron object. It can be initialized from the given `weights`. When given `weights`, the number of iterations used to obtain them are the given `iterations`, or `0` by default.

```js
import averagedPerceptron from "averaged-perceptron";

// Create a new perceptron
const perceptron = averagedPerceptron();
```

If you want to train the model in multiple sessions, optionally persisting the weights between them, you can resume training by giving the number of iterations, that is, the number of times `update()` was called to obtain the given weights.

```js
import averagedPerceptron from "averaged-perceptron";

// Create a perceptron from already existing weights
const weightsJSON = '{"x":{"a":0.4,"b":0.6},"y":{"a":0.8,"b":-0.4}}';
const weights = JSON.parse(weightsJSON);
const iterations = 1000; // Weights obtained with 1000 update() calls
const perceptron = averagedPerceptron(weights, 1000);
```

### `predict(features, scores)`

Returns the label predicted from the given `features`, or `""` if none exists. It can be given the `scores` so that it does not have to recompute them.

```js
import averagedPerceptron from "averaged-perceptron";

averagedPerceptron({
  x: { a: 0.4, b: 0.6 },
  y: { a: 0.8, b: -0.4 }
}).predict({ x: 1, y: 1 });
// => "a"
```

### `scores(features)`

Returns an object with the scores of each label in the given `features`.

```js
import averagedPerceptron from "averaged-perceptron";

averagedPerceptron({
  x: { a: 0.4, b: 0.6 },
  y: { a: 0.8, b: -0.4 }
}).scores({ a: 1, b: 1 });
// => { a: 1.2, b: 0.2 }
```

### `update(features, label, guess)`

Returns the perceptron, updating the weights with the respective value of each of the given `features` if the given `label` is not predicted. It can be given the `guess` so that it does not have to recompute it.

```js
import averagedPerceptron from "averaged-perceptron";

averagedPerceptron().update({ x: 1, y: 1 }, "a");
```

### `weights()`

Returns an object of features where each feature is an object of labels with the weight of each feature-label pair.

```js
import averagedPerceptron from "averaged-perceptron";

averagedPerceptron({
  x: { a: 0.4, b: 0.6 },
  y: { a: 0.8, b: -0.4 }
}).weights();
// => { x: { a: 0.4, b: 0.6 }, y: { a: 0.8, b: -0.4 } }
```

## License

[MIT](./LICENSE)
