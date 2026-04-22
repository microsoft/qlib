[![NPM](https://img.shields.io/npm/v/react-select.svg)](https://www.npmjs.com/package/react-select)
[![CircleCI](https://circleci.com/gh/JedWatson/react-select/tree/master.svg?style=shield)](https://circleci.com/gh/JedWatson/react-select/tree/master)
[![Coverage Status](https://coveralls.io/repos/JedWatson/react-select/badge.svg?branch=master&service=github)](https://coveralls.io/github/JedWatson/react-select?branch=master)
[![Supported by Thinkmill](https://thinkmill.github.io/badge/heart.svg)](http://thinkmill.com.au/?utm_source=github&utm_medium=badge&utm_campaign=react-select)

# React-Select

The Select control for [React](https://reactjs.com). Initially built for use in [KeystoneJS](http://www.keystonejs.com).

See [react-select.com](https://www.react-select.com) for live demos and comprehensive docs.

React Select is funded by [Thinkmill](https://www.thinkmill.com.au) and [Atlassian](https://atlaskit.atlassian.com). It represents a whole new approach to developing powerful React.js components that _just work_ out of the box, while being extremely customisable.

For the story behind this component, watch Jed's talk at React Conf 2019 - [building React Select](https://youtu.be/yS0jUnmBujE)

Features include:

- Flexible approach to data, with customisable functions
- Extensible styling API with [emotion](https://emotion.sh)
- Component Injection API for complete control over the UI behaviour
- Controllable state props and modular architecture
- Long-requested features like option groups, portal support, animation, and more

## Using an older version?

- [v3, v4, and v5 upgrade guide](https://react-select.com/upgrade)
- [v2 upgrade guide](https://react-select.com/upgrade-to-v2)
- React Select v1 documentation and examples are available at [v1.react-select.com](https://v1.react-select.com)

# Installation and usage

The easiest way to use react-select is to install it from npm and build it into your app with Webpack.

```
yarn add react-select
```

Then use it in your app:

#### With React Component

```js
import React from 'react';
import Select from 'react-select';

const options = [
  { value: 'chocolate', label: 'Chocolate' },
  { value: 'strawberry', label: 'Strawberry' },
  { value: 'vanilla', label: 'Vanilla' },
];

class App extends React.Component {
  state = {
    selectedOption: null,
  };
  handleChange = (selectedOption) => {
    this.setState({ selectedOption }, () =>
      console.log(`Option selected:`, this.state.selectedOption)
    );
  };
  render() {
    const { selectedOption } = this.state;

    return (
      <Select
        value={selectedOption}
        onChange={this.handleChange}
        options={options}
      />
    );
  }
}
```

#### With React Hooks

```js
import React, { useState } from 'react';
import Select from 'react-select';

const options = [
  { value: 'chocolate', label: 'Chocolate' },
  { value: 'strawberry', label: 'Strawberry' },
  { value: 'vanilla', label: 'Vanilla' },
];

export default function App() {
  const [selectedOption, setSelectedOption] = useState(null);

  return (
    <div className="App">
      <Select
        defaultValue={selectedOption}
        onChange={setSelectedOption}
        options={options}
      />
    </div>
  );
}
```

## Props

Common props you may want to specify include:

- `autoFocus` - focus the control when it mounts
- `className` - apply a className to the control
- `classNamePrefix` - apply classNames to inner elements with the given prefix
- `isDisabled` - disable the control
- `isMulti` - allow the user to select multiple values
- `isSearchable` - allow the user to search for matching options
- `name` - generate an HTML input with this name, containing the current value
- `onChange` - subscribe to change events
- `options` - specify the options the user can select from
- `placeholder` - change the text displayed when no option is selected
- `noOptionsMessage` - ({ inputValue: string }) => string | null - Text to display when there are no options
- `value` - control the current value

See the [props documentation](https://www.react-select.com/props) for complete documentation on the props react-select supports.

## Controllable Props

You can control the following props by providing values for them. If you don't, react-select will manage them for you.

- `value` / `onChange` - specify the current value of the control
- `menuIsOpen` / `onMenuOpen` / `onMenuClose` - control whether the menu is open
- `inputValue` / `onInputChange` - control the value of the search input (changing this will update the available options)

If you don't provide these props, you can set the initial value of the state they control:

- `defaultValue` - set the initial value of the control
- `defaultMenuIsOpen` - set the initial open value of the menu
- `defaultInputValue` - set the initial value of the search input

## Methods

React-select exposes two public methods:

- `focus()` - focus the control programmatically
- `blur()` - blur the control programmatically

## Customisation

Check the docs for more information on:

- [Customising the styles](https://www.react-select.com/styles)
- [Using custom components](https://www.react-select.com/components)
- [Using the built-in animated components](https://www.react-select.com/home#animated-components)
- [Creating an async select](https://www.react-select.com/async)
- [Allowing users to create new options](https://www.react-select.com/creatable)
- [Advanced use-cases](https://www.react-select.com/advanced)
- [TypeScript guide](https://www.react-select.com/typescript)

## TypeScript

The v5 release represents a rewrite from JavaScript to TypeScript. The types for v4 and earlier releases are available at [@types](https://www.npmjs.com/package/@types/react-select). See the [TypeScript guide](https://www.react-select.com/typescript) for how to use the types starting with v5.

# Thanks

Thank you to everyone who has contributed to this project. It's been a wild ride.

If you like React Select, you should [follow me on twitter](https://twitter.com/jedwatson)!

Shout out to [Joss Mackison](https://github.com/jossmac), [Charles Lee](https://github.com/gwyneplaine), [Ben Conolly](https://github.com/Noviny), [Tom Walker](https://github.com/bladey), [Nathan Bierema](https://github.com/Methuselah96), [Eric Bonow](https://github.com/ebonow), [Emma Hamilton](https://github.com/emmatown), [Dave Brotherstone](https://github.com/bruderstein), [Brian Vaughn](https://github.com/bvaughn), and the [Atlassian Design System](https://atlassian.design) team who along with many other contributors have made this possible ❤️

## License

MIT Licensed. Copyright (c) Jed Watson 2022.
