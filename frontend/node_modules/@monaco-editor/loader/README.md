# @monaco-editor/loader &middot; [![monthly downloads](https://img.shields.io/npm/dm/@monaco-editor/loader)](https://www.npmjs.com/package/@monaco-editor/loader) [![gitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/suren-atoyan/monaco-loader/blob/master/LICENSE) [![npm version](https://img.shields.io/npm/v/@monaco-editor/loader.svg?style=flat)](https://www.npmjs.com/package/@monaco-editor/loader) [![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/suren-atoyan/monaco-loader/pulls)

The utility to easy setup `monaco-editor` into your browser

## Synopsis

Configure and download monaco sources via its loader script, without needing to use webpack's (or any other module bundler's) configuration files

## Motivation

It's been a while we are working with `monaco editor`. It's a great library and provides a powerful editor out of the box. Anyway, there were couple of problems related to the setup process. The main problem is the need to do some additional `webpack` configuration; that's not bad, but some quite useful tools, like `CRA`, aren't happy with that fact. The library [`@monaco-editor/react`](https://github.com/suren-atoyan/monaco-react) was being created to solve that problem - `monaco editor wrapper for easy/one-line integration with React applications without needing to use webpack's (or any other module bundler's) configuration files`. In that library, there was a utility that cares about the initialization process of monaco and overcomes the additional use of webpack configuration. That utility grows over time and now it's a separate library. Now, you can easily setup monaco into your browser, create your own editors, wrappers for React/Vue/Angular of whatever you want.

## How it works

Monaco editor provides a script called `loader`, which itself provides tooling to download monaco sources. The library, under the hood, handles the configuration and loading part and gives us an easy-to-use API to interact with it

## Documentation

#### Contents

* [Installation](#installation)
* [Introduction](#introduction)
* [Usage](#usage)
  * [.config](#config)
  * [.init](#init)
* [Notes](#notes)
  * [For `electron` users](#for-electron-users)
  * [For `Next.js` users](#for-nextjs-users)

### Installation

```bash
npm install @monaco-editor/loader
```

or

```bash
yarn add @monaco-editor/loader
```

NOTE: For TypeScript type definitions, this package uses the [monaco-editor](https://www.npmjs.com/package/monaco-editor) package as a peer dependency. So, if you need types and don't already have the [monaco-editor](https://www.npmjs.com/package/monaco-editor) package installed, you will need to do so.

### Introduction

The library exports types and the utility called `loader`, the last one has two methods

* [.config](#config)
* [.init](#init)

### Usage

```javascript
import loader from '@monaco-editor/loader';

loader.init().then(monaco => {
  monaco.editor.create(/* editor container, e.g. document.body */, {
    value: '// some comment',
    language: 'javascript',
  });
});
```

[codesandbox](https://codesandbox.io/s/simple-usage-os49p)

#### .config

By using the `.config` method we can configure the monaco loader. By default all sources come from CDN, you can change that behavior and load them from wherever you want

```javascript
import loader from '@monaco-editor/loader';

// you can change the source of the monaco files
loader.config({ paths: { vs: '...' } });

// you can configure the locales
loader.config({ 'vs/nls': { availableLanguages: { '*': 'de' } } });

// or
loader.config({
  paths: {
    vs: '...',
  },
  'vs/nls' : {
    availableLanguages: {
      '*': 'de',
    },
  },
});

loader.init().then(monaco => { /* ... */ });
```

[codesandbox](https://codesandbox.io/s/config-o6zn6)

#### Configure the loader to load the monaco as an npm package

```javascript
import loader from '@monaco-editor/loader';
import * as monaco from 'monaco-editor';

loader.config({ monaco });

loader.init().then(monacoInstance => { /* ... */ });
```

[codesandbox](https://codesandbox.io/s/npm-gswrvh)

#### .init

The `.init` method handles the initialization process. It returns the monaco instance, wrapped with cancelable promise

```javascript
import loader from '@monaco-editor/loader';

loader.init().then(monaco => {
  console.log('Here is the monaco instance', monaco);
});
```

[codesandbox](https://codesandbox.io/s/init-q2ipt)

```javascript
import loader from '@monaco-editor/loader';

const cancelable = loader.init();

cancelable.then(monaco => {
  console.log('You will not see it, as it is canceled');
});

cancelable.cancel();
```

[codesandbox](https://codesandbox.io/s/init-cancelable-9o42y)

#### Notes

##### For `electron` users

In general it works fine with electron, but there are several cases that developers usually face to and sometimes it can be confusing. Here they are:

1) **Download process fails** or if you use @monaco-editor/react **You see loading screen stuck**
Usually, it's because your environment doesn't allow you to load external sources. By default, it loads monaco sources from CDN. You can see the [default configuration](https://github.com/suren-atoyan/monaco-loader/blob/master/src/config/index.js#L3). But sure you can change that behavior; the library is fully configurable. Read about it [here](https://github.com/suren-atoyan/monaco-loader#config). So, if you want to download it from your local files, you can do it like this:

```javascript
import loader from '@monaco-editor/loader';

loader.config({ paths: { vs: '../path-to-monaco' } });
```

or, if you want to use it as an npm package, you can do it like this:

```javascript
import loader from '@monaco-editor/loader';
import * as monaco from 'monaco-editor';

loader.config({ monaco });

loader.init().then(monacoInstance => { /* ... */ });
```

2) **Based on your electron environment it can be required to have an absolute URL**
The utility function taken from [here](https://github.com/microsoft/monaco-editor-samples/blob/master/electron-amd-nodeIntegration/electron-index.html) can help you to achieve that. Let's imagine you have `monaco-editor` package installed and you want to load monaco from the `node_modules` rather than from CDN: in that case, you can write something like this:

```javascript
function ensureFirstBackSlash(str) {
    return str.length > 0 && str.charAt(0) !== '/'
        ? '/' + str
        : str;
}

function uriFromPath(_path) {
    const pathName = path.resolve(_path).replace(/\\/g, '/');
    return encodeURI('file://' + ensureFirstBackSlash(pathName));
}

loader.config({
  paths: {
    vs: uriFromPath(
      path.join(__dirname, '../node_modules/monaco-editor/min/vs')
    )
  }
});
```

or, just use it as an npm package.

There were several issues about this topic that can be helpful too - [1](https://github.com/suren-atoyan/monaco-react/issues/48) [2](https://github.com/suren-atoyan/monaco-react/issues/12) [3](https://github.com/suren-atoyan/monaco-react/issues/58) [4](https://github.com/suren-atoyan/monaco-react/issues/87)

And if you use `electron` with `monaco` and have faced an issue different than the above-discribed ones, please let us know to make this section more helpful.

##### For `Next.js` users

The part of the source that should be pre-parsed is optimized for server-side rendering, so, in usual cases, it will work fine, but if you want to have access, for example, to [`monacoInstance`](#config) you should be aware that it wants to access the `document` object, and it requires browser environment. Basically you just need to avoid running that part out of browser environment, there are several ways to do that. One of them is described [here](https://nextjs.org/docs/advanced-features/dynamic-import#with-no-ssr).

And if you use `monaco` with `Next.js` and have faced an issue different than the above-described one, please let us know to make this section more helpful.

## License

[MIT](./LICENSE)
