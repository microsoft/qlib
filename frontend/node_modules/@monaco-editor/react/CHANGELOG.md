### Versions

## 4.7.0

- package: update @monaco-editor/loader to the latest (v1.5.0) version (this uses monaco-editor v0.52.2)
- package: inherit all changes from v4.7.0-rc.0

## 4.7.0-rc.0

- package: add support for react/react-dom v19 as a peer dependency
- playground: update playground's React version to 19

## 4.6.0

###### _Oct 6, 2023_

- Editor/DiffEditor: use `'use client'` on top of `Editor.tsx` and `DiffEditor.tsx`
- loader: update `@monaco-editor/loader` version (1.4.0)
- playground: use createRoot for bootstrapping

## 4.5.2

###### _Aug 23, 2023_

- DiffEditor: apply updated on `originalModelPath` and `modifiedModelPath` before `original` and `modified` props

## 4.5.1

###### _May 5, 2023_

- DiffEditor: track `originalModelPath` and `modifiedModelPath` changes and get or create a new model accordingly
- types: fix typo in comment
- package: replace `prepublish` with `prepublishOnly`

## 4.5.0

###### _Apr 7, 2023_

- Editor: implement `preventTriggerChangeEvent` flag

from `4.5.0-beta.0`

- DiffEditor: add preventCreation flag to diff editor
- project: rewrite with TypeScript
- project: implement prettier
- loader: update `@monaco-editor/loader` version (1.3.2)

## 4.5.0-beta.0

###### _Apr 2, 2023_

- DiffEditor: add preventCreation flag to diff editor
- project: rewrite with TypeScript
- project: implement prettier
- loader: update `@monaco-editor/loader` version (1.3.2)

## 4.4.6

###### _Sep 24, 2022_

- fix onChange: unconditionally call onChange inside onDidChangeModelContent
- add preventCreation flag
- update lock files

## 4.4.5

###### _May 11, 2022_

- loader: update `@monaco-editor/loader` version (1.3.2)

## 4.4.4

###### _Apr 23, 2022_

- package: fix npm prepublish step

## 4.4.3

###### _Apr 23, 2022_

- loader: update `@monaco-editor/loader` version (1.3.1)

## 4.4.2

###### _Apr 12, 2022_

- package: support react/react-dom v18 as a peer dependency

## 4.4.1

###### _Mar 29, 2022_

- types: add missing type `monaco` in `loader.config`

## 4.4.0

###### _Mar 28, 2022_

- loader: update `@monaco-editor/loader` version (1.3.0); using `monaco` from `node_modules` is supported
- playground: update playground packages

## 4.3.1

###### _Oct 3, 2021_

- types: update types according to the new `loader` version and the new `wrapperProps` property

## 4.3.0

###### _Oct 3, 2021_

- Editor/DiffEditor: add `wrapperProps` property
- DiffEditor: allow `DiffEditor` to use existing models
- package.json: update `@monaco-editor/loader` version to `v1.2.0` (monaco version 0.28.1)

## 4.2.2

###### _Aug 9, 2021_

- Editor: `onValidate` integrate `onDidChangeMarkers` (released in `v0.22.0`)
- package.json: after `onDidChangeMarkers` integration `state-local` became redundant; remove it

## 4.2.1

###### _Jun 21, 2021_

- loader: update `@monaco-editor/loader` package version to the latest one (v1.1.1)
- monaco-editor: set `monaco-editor` peerDependency version to `>= 0.25.0 < 1`
- tests: update snapshots

## 4.2.0

###### _Jun 13, 2021_

- loader: update `@monaco-editor/loader` package version to the latest one (v1.1.0)
- demo: update demo examples
- tests: update snapshots

## 4.1.3

###### _Apr 21, 2021_

- types: add `keepCurrentOriginalModel` and `keepCurrentModifiedModel` to type definition

## 4.1.2

###### _Apr 19, 2021_

- DiffEditor: add `keepCurrentOriginalModel` and `keepCurrentModifiedModel` properties; indicator whether to dispose the current original/modified model when the DiffEditor is unmounted or not
- package.json: update monaco-editor peerDependency to the lates one (0.23.0)

## 4.1.1

###### _Apr 2, 2021_

- DiffEditor: update `DiffEditor`'s `modified` value by `executeEdits`
- README: add an example for getting the values of `DiffEditor`

## 4.1.0

###### _Mar 15, 2021_

- loader: update @monaco-editor/loader dependency to version 1.0.1
- types: fix Theme type; vs-dark instead of dark

## 4.0.11

###### _Feb 27, 2021_

- Editor: add an additional check in case if `line` is undefined

## 4.0.10

###### _Feb 16, 2021_

- Editor: use `revealLine` for line update instead of `setScrollPosition`

## 4.0.9

###### _Jan 29, 2021_

- Editor: save and restore current model view state, if `keepCurrentModel` is true

## 4.0.8

###### _Jan 29, 2021_

- Editor: add `keepCurrentModel` property to the `Editor` component; indicator whether to dispose the current model when the Editor is unmounted or not

## 4.0.7

###### _Jan 21, 2021_

- Editor: fire `onValidate` unconditionally, always, with the current model markers

## 4.0.6

###### _Jan 19, 2021_

- DiffEditor: check if `originalModelPath` and `modifiedModelPath` exist in `setModels` function
- DiffEditor: remove `originalModelPath` and `modifiedModelPath` from `defaultProps`

## 4.0.5

###### _Jan 19, 2021_

- utils: check if `path` exists in `createModel` utility function
- Editor: remove `defaultPath` from `defaultProps`

## 4.0.4

###### _Jan 18, 2021_

- package.json: update husky precommit hook to remove lib folder

## 4.0.3

###### _Jan 18, 2021_

- Editor: enable multi-model support
- types: add `path`, `defaultLanguage` and `saveViewState` for multi-model support

## 4.0.2

###### _Jan 18, 2021_

- types: declare and export `useMonaco` type

## 4.0.1

###### _Jan 18, 2021_

- Editor: dispose the current model if the Editor component is unmounted

## 4.0.0

###### _Jan 16, 2021_

- package.json: update dependency (`@monaco-editor/loader`) version to - `v1.0.0`
- hooks: create `useMonaco` hook
- lib: export (named) `useMonaco` from the entry file
- monaco: rename the main utility: monaco -> loader
- Editor/Diff: rename `editorDidMount` to `onMount`
- Editor/Diff: expose monaco instance from `onMount` as a second argument (first is the editor instance)
- Editor/Diff: add `beforeMount` prop: function with a single argument -> monaco instance
- Editor: add `defaultModelPath` prop, use it as a default model path
- Editor: add `defaultValue` prop and use it during default model creation
- Editor: add subscription (`onChange` prop) to listen default model content change
- Editor: remove `_isControlledMode` prop
- Diff: add `originalModelPath` and `modifiedModelPath` props, use them as model paths for original/modified models
- ControlledEditor: remove; the `Editor` component, now, handles both controlled and uncontrolled modes
- package.json: move `prop-types` to dependencies
- types: fix types according to changed
- Editor: add `onValidate` prop: an event emitted when the length of the model markers of the current model isn't 0

## 3.8.3

###### _Jan 8, 2021_

- README: fix DiffEditor `options` prop type name
- types: rename index.d.ts to types.d.ts

## 3.8.2

###### _Jan 7, 2021_

- package.json: add `@monaco-editor/loader` as a dependency
- Editor/Diff Editor components: use `@monaco-editor/loader` instead of `monaco` utility
- utilities: remove utilities that were being replaced by the `@monaco-editor/loader`
- utilities: collect remaining utilities all in the entry file / add some new ones for the next version
- config: remove config as it's already replaced by the `@monaco-editor/loader`
- hooks: create `usePrevious` hook
- cs: coding style fixes
- build: use `Rollup` as a build system; now, we have bundles for `cjs/es/umd`

## 3.7.5

###### _Jan 3, 2021_

- utilities (monaco): fix `state-local` import

## 3.7.4

###### _Dec 16, 2020_

- Editor/Diff Editor components: fix `componentDidMount` call order
- src: (minor) some corrections according to coding style

## 3.7.3

###### _Dec 15, 2020_

- Editor component: set `forceMoveMarkers` `true` in `executeEdits`

## 3.7.2

###### _Dec 5, 2020_

- package: add react/react-dom 17 version as a peer dependency

## 3.7.1

###### _Nov 29, 2020_

- editor: fix - remove unnecessary `value set` before language update

## 3.7.0

###### _Nov 11, 2020_

- monaco: update monaco version to 0.21.2

## 3.6.3

###### _Sep 22, 2020_

- types: add missing props; `className` and `wrapperClassName`

## 3.6.2

###### _Aug 19, 2020_

- eslint: update eslint rules: add 'eslint:recommended' and 'no-unused-vars' -> 'error'
- src: refactor according to new eslint rules
- package.json: update github username, add author email

## 3.6.1

###### _Aug 18, 2020_

- ControlledEditor: store current value in ref instead of making it a dependency of `handleEditorModelChange`

## 3.6.0

###### _Aug 18, 2020_

- ControlledEditor: fix onChange handler issue; dispose prev listener and attach a new one for every new onChange
- ControlledEditor: do not trigger onChange in programmatic changes

## 3.5.7

###### _Aug 9, 2020_

- utilities (monaco): remove intermediate function for injecting scripts

## 3.5.6

###### _Aug 6, 2020_

- dependencies: add `state-local` as a dependency (replace with `local-state` util)

## 3.5.5

###### _Aug 3, 2020_

- dependencies: move `@babel/runtime` from peer dependencies to dependencies

## 3.5.4

###### _Aug 3, 2020_

- dependencies: add `@babel/runtime` as a peer dependency

## 3.5.3

###### _Aug 3, 2020_

- babel: update babel version (v.7.11.0) / activate helpers (decrease bundle size)
- hooks: move out hooks from utils to root
- utilities: remove utils/store to utils/local-state

## 3.5.2

###### _Aug 2, 2020_

- utilities: redesign `store` utility

## 3.5.1

###### _July 30, 2020_

- utilities (monaco): correct config obj name

## 3.5.0

###### _July 30, 2020_

- utilities (monaco): redesign utility `monaco`; get rid of class, make it more fp
- utilities: create `compose` utility
- utilities: create `store` utility; for internal usage (in other utilities)

## 3.4.2

###### _July 15, 2020_

- controlled editor: fix undo/redo issue

## 3.4.1

###### _July 3, 2020_

- editor: improve initialization error handling

## 3.4.0

###### _June 28, 2020_

- editor: fix 'readOnly' option check
- editor: add className and wrapperClassName props
- diffEditor: add className and wrapperClassName props

## 3.3.2

###### _June 20, 2020_

- utils: (monaco) add a possibility to pass src of config script

## 3.3.1

###### _May 30, 2020_

- editor: add overrideServices prop

## 3.2.1

###### _Apr 13, 2020_

- package: update default package version to 0.20.0

## 3.2.1

###### _Mar 31, 2020_

- types: fix monaco.config types

## 3.2.0

###### _Mar 31, 2020_

- fix: check the existence of target[key] in deepMerge
- config: deprecate indirect way of configuration and add deprecation message
- config: create a new structure of the configuration; the passed object will be directly passed to require.config
- readme: redesign the config section according to the new structure

## 3.1.2

###### _Mar 16, 2020_

- diff editor: remove line prop as it's not used (and can't be used)

## 3.1.1

###### _Feb 25, 2020_

- package: update devDependencies
- demo: update all dependencies

## 3.1.0

###### _Feb 6, 2020_

- monaco: update monaco version to 0.19.0
- utils: create new util - makeCancelable (for promises)
- editor/diffEditor: cancel promise before unmount
- demo: make "dark" default theme, update package version

## 3.0.1

###### _Dec 26, 2019_

- readme: update installation section

## 3.0.0

###### _Dec 24, 2019_

- monaco: update monaco version to 0.19.0

## 2.6.1

###### _Dec 23, 2019_

- versions: fix version

## 2.5.1

###### _Dec 23, 2019_

- types: fix type of "loading"

## 2.5.0

###### _Dec 19, 2019_

- types: fix type of theme; user should be able to pass any kind of theme (string)

## 2.4.0

###### _Dec 11, 2019_

- types: add config into namespace monaco
- types: change type of "loading" from React.ElementType to React.ReactNode

## 2.3.5

###### _Dec 10, 2019_

- optimize babel build with runtime transform

## 2.3.4

###### _Dec 10, 2019_

- add xxx.spec.js.snap files to npmignore

## 2.3.2 & 3

###### _Dec 10, 2019_

- fix typo in npmignore

## 2.3.1

###### _Dec 10, 2019_

- add unnecessary files to npmignore

## 2.3.0

###### _Nov 9, 2019_

- prevent onchange in case of undo/redo (controlled editor)
- create separate component for MonacoContainer

## 2.2.0

###### _Nov 9, 2019_

- force additional tokenization in controlled mode to avoid blinking

## 2.1.1

###### _Oct 25, 2019_

- fix "options" types

## 2.1.0

###### _Oct 25, 2019_

- add monaco-editor as peer dependency for proper type definitions
- write more proper types

## 2.0.0

###### _Oct 9, 2019_

- set the default version of monaco to 0.18.1
- set last value by .setValue method before changing the language

## 1.2.3

###### _Oct 7, 2019_

- (TYPES) add "void" to the "ControlledEditorOnChange" return types

## 1.2.2

###### _Oct 3, 2019_

- update dev dependencies
- check editor existence in "removeEditor" function
- replace "jest-dom" with "@testing-library/jest-dom"

## 1.2.1

###### _Aug 20, 2019_

- Set editor value directly in case of read only

## 1.2.0

###### _Aug 16, 2019_

- Add method to modify default config

## 1.1.0

###### _July 26, 2019_

- Apply edit by using `executeEdits` method
- Correct ControlledEditor usage examples in Docs

## 1.0.8

###### _July 24, 2019_

- Export utility 'monaco' to be able to access to the monaco instance

## 1.0.7

###### _July 21, 2019_

- Add controlled version of editor component

## 1.0.5

###### _July 19, 2019_

- Add a possibility to interact with Editor before it is mounted

## 1.0.4

###### _July 13, 2019_

- FIX: add "types" fild to package.json

## 1.0.3

###### _July 13, 2019_

- Add basic support for TypeScript

## 1.0.2

###### _June 26, 2019_

- Update package description

## 1.0.1

###### _June 26, 2019_

- Move from 'unpkg.com' to 'cdn.jsdelivr.net' (NOTE: in the future, it will be configurable)

## 1.0.0

###### _June 25, 2019_

:tada: First stable version :tada:

- Add monaco version to CDN urls to avoid 302 redirects

## 0.0.3

###### _June 22, 2019_

- Remove redundant peer dependency

## 0.0.2

###### _June 22, 2019_

- Make text-align of the wrapper of editors independent from outside

## 0.0.1

###### _June 21, 2019_

First version of the library
