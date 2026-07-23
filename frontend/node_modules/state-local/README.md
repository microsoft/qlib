# State &middot; [![monthly downloads](https://img.shields.io/npm/dm/state-local)](https://www.npmjs.com/package/state-local) [![gitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/suren-atoyan/state-local/blob/master/LICENSE) [![Rate on Openbase](https://badges.openbase.io/js/rating/state-local.svg)](https://openbase.io/js/state-local?utm_source=embedded&utm_medium=badge&utm_campaign=rate-badge) [![build size](https://img.shields.io/bundlephobia/minzip/state-local)](https://bundlephobia.com/result?p=state-local) [![npm version](https://img.shields.io/npm/v/state-local.svg?style=flat)](https://www.npmjs.com/package/state-local)  [![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/suren-atoyan/state-local/pulls)

:zap: Tiny, simple, and robust technique for defining and acting with local states (for all js environments - node, browser, etc.)

## Synopsis

A local state for modules, functions, and other ECs

## Motivation

We all love functional programming and the concepts of it. It gives us many clean patterns, which we use in our code regardless of exactly which paradigm is in the base of our codebase. But sometimes, for some reason, we can't keep our code "clean" and have to interact with items that are outside of the current lexical environment

For example:

:x:
```javascript
let x = 0;
let y = 1;

// ...
function someFn() {
  // ...
  x++;
}

// ...
function anotherFn() {
 // ...
 y = 6;
 console.log(x);
}

// ...
function yetAnotherFn() {
  // ...
  y = x + 4;
  x = null; // 🚶
}
```

The example above lacks control over the mutations and consumption, which can lead to unpredictable and unwanted results. It is just an example of real-life usage and there are many similar cases that belong to the same class of the problem

**The purpose of this library is to give an opportunity to work with local states in a clear, predictable, trackable, and strict way**

:white_check_mark:

```javascript
import state from 'state-local';

const [getState, setState] = state.create({ x: 0, y: 1 });

// ...
function someFn() {
  // ...
  setState(state => ({ x: state.x + 1 }));
}

// ...
function anotherFn() {
 // ...
 setState({ y: 6 });
 const state = getState();
 console.log(state);
}

// ...
function yetAnotherFn() {
  // ...
  setState(state => ({ y: state.x + 4, x: null }));
}
```

[codesandbox](https://codesandbox.io/s/motivation-1-xv5el?file=/src/index.js)

We also can track the changes in items:

```javascript
import state from 'state-local';

const [getState, setState] = state.create({ x: 0, y: 1 }, {
  x: latestX => console.log('(⌐▀ ̯ʖ▀) Houston we have a problem; "x" has been changed. "x" now is:', latestX),
  y: latestY => console.log('(⌐▀ ̯ʖ▀) Houston we have a problem; "y" has been changed. "y" now is:', latestY),
});

// ...
```

[codesandbox](https://codesandbox.io/s/motivation-2-ivf7d)

We can use the subset of the state in some execution contexts:

```javascript
import state from 'state-local';

const [getState, setState] = state.create({ x: 5, y: 7 });

// ...
function someFn() {
  const state = getState(({ x }) => ({ x }));

  console.log(state.x); // 5
  console.log(state.y); // ❌ undefined - there is no y
}
```

[codesandbox](https://codesandbox.io/s/motivation-3-femne)

And much more...

## Documentation

#### Contents

* [Installation](#installation)
* Usage
  * [create](#create)
  * [initial state](#initial-state)
  * [handler](#handler)
  * [getState](#getstate)
  * [selector](#selector)
  * [setState](#setstate)

#### Installation

You can install this library as an npm package or download it from the CDN and use it in node or browser:

```bash
npm install state-local
```
or
```bash
yarn add state-local
```

or

```html
<script src="https://unpkg.com/state-local/dist/state-local.js"></script>

<script>
// now it is available in `window` (window.state)
const [getState, setState] = state.create({ x: 11, y: 13 });
// ...
</script>
```

#### create

The default export has a method called `create`, which is supposed to be a function to create a state:

```javascript
import state from 'state-local';

// state.create

// ...
```

[codesandbox](https://codesandbox.io/s/docs-create-t1cxe)

`create` is a function with two parameters:

1) [`initial state`](#initial-state) (**required**)
2) [`handler`](#handler) (**optional**)

#### initial state

`initial state` is a base structure and a value for the state. It should be a non-empty object

```javascript
import state from 'state-local';

/*
const [getState, setState] = state.create(); // ❌ error - initial state is required
const [getState, setState] = state.create(5); // ❌ error - initial state should be an object
const [getState, setState] = state.create({}); // ❌ error - initial state shouldn\'t be an empty object
*/

const [getState, setState] = state.create({ isLoading: false, payload: null }); // ✅
// ...
```

[codesandbox](https://codesandbox.io/s/docs-initial-state-22i3s)

#### handler

`handler` is a second parameter for `create` function and it is optional. It is going to be a handler for state updates. Hence it can be either a function or an object.

- If the handler is a function than it should be called immediately after every state update (with the latest state)
- If the handler is an object than the keys of that object should be a subset of the state and the values should be called immediately after every update of the corresponding field in the state (with the latest value of the field)

see example below:

if `handler` is a function
```javascript
import state from 'state-local';

const [getState, setState] = state.create({ x: 2, y: 3, z: 5 }, handleStateUpdate /* will be called immediately after every state update */);

function handleStateUpdate(latestState) {
  console.log('hey state has been updated; the new state is:', latestState); // { x: 7, y: 11, z: 13 }
}

setState({ x: 7, y: 11, z: 13 });
// ...
```

[codesandbox](https://codesandbox.io/s/handler-function-uevxj)

if `handler` is an object
```javascript
import state from 'state-local';

const [getState, setState] = state.create({ x: 2, y: 3, z: 5 }, {
  x: handleXUpdate, // will be called immediately after every "x" update
  y: handleYUpdate, // will be called immediately after every "y" update
  // and we don't want to listen "z" updates 😔
});

function handleXUpdate(latestX) {
  console.log('(⌐▀ ̯ʖ▀) Houston we have a problem; "x" has been changed. "x" now is:', latestX); // ... "x" now is 7
}

function handleYUpdate(latestY) {
  console.log('(⌐▀ ̯ʖ▀) Houston we have a problem; "y" has been changed. "y" now is:', latestY); // ... "y" now is 11
}

setState({ x: 7, y: 11, z: 13 });
// ...
```

[codesandbox](https://codesandbox.io/s/handler-object-8k0pt)

#### getState

`getState` is the first element of the pair returned by `create` function. It will return the current state or the subset of the current state depending on how it was called. It has an optional parameter `selector`

```javascript
import state from "state-local";

const [getState, setState] = state.create({ p1: 509, p2: 521 });

const state = getState();
console.log(state.p1); // 509
console.log(state.p2); // 521

// or

const { p1, p2 } = getState();
console.log(p1); // 509
console.log(p2); // 521
```

[codesandbox](https://codesandbox.io/s/getstate-zn3hj)

#### selector

`selector` is a function that is supposed to be passed (optional) as an argument to `getState`. It receives the current state and returns a subset of the state

```javascript
import state from 'state-local';

const [getState, setState] = state.create({ p1: 389, p2: 397, p3: 401 });

function someFn() {
  const state = getState(({ p1, p2 }) => ({ p1, p2 }));
  console.log(state.p1); // 389
  console.log(state.p2); // 397
  console.log(state.p3); // ❌ undefined - there is no p3
}
```

[codesandbox](https://codesandbox.io/s/selector-vjmdu)

#### setState

`setState` is the second element of the pair returned by `create` function. It is going to receive an object as a change for the state. The change object will be shallow merged with the current state and the result will be the next state

**NOTE: the change object can't contain a field that is not specified in the "initial" state**

```javascript
import state from 'state-local';

const [getState, setState] = state.create({ x:0, y: 0 });

setState({ z: 'some value' }); // ❌ error - it seams you want to change a field in the state which is not specified in the "initial" state

setState({ x: 11 }); // ✅ ok
setState({ y: 1 }); // ✅ ok
setState({ x: -11, y: 11 }); // ✅ ok
```

[codesandbox](https://codesandbox.io/s/setstate-1-u4fq0)

`setState` also can receive a function which will be called with the current state and it is supposed to return the change object

```javascript
import state from 'state-local';

const [getState, setState] = state.create({ x:0, y: 0 });

setState(state => ({ x: state.x + 2 })); // ✅ ok
setState(state => ({ x: state.x - 11, y: state.y + 11 })); // ✅ ok

setState(state => ({ z: 'some value' })); // ❌ error - it seams you want to change a field in the state which is not specified in the "initial" state
```

[codesandbox](https://codesandbox.io/s/smoosh-wildflower-nv9dg)

## License

[MIT](./LICENSE)
