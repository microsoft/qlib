# use-isomorphic-layout-effect

A React helper hook for scheduling a layout effect with a fallback to a regular effect for environments where layout effects should not be used (such as server-side rendering).

## Installation

```sh
$ npm i use-isomorphic-layout-effect
```

## Usage 

You only need to switch `useLayoutEffect` with `useIsomorphicLayoutEffect`

```diff
+ import useIsomorphicLayoutEffect from 'use-isomorphic-layout-effect';
- import { useLayoutEffect } from 'react';


const YourComponent = () => {
+  useIsomorphicLayoutEffect(() => {
-  useLayoutEffect(() => {
    // your implementation
  }, []);
};
```
