## 1.7.0
###### *Nov 21, 2025*

- loader: merged #61 - add backward compatibility for 0.53 and 0.54 versions
- monaco-editor: update to the latest version (0.55.1)

## 1.6.1
###### *Oct 14, 2025*

- eslint: use mjs for eslint config file
- package: remove type field

## 1.6.0
###### *Oct 12, 2025*

- monaco-editor: update to the latest version (0.54.0)
- package: update all dependencies to the latest version
- playground: update all dependencies to the latest version

## 1.5.0
###### *Feb 13, 2025*

- monaco-editor: update to the latest version (0.52.2)
- package: remove monaco-editor from peerDependencies

## 1.4.0
###### *Oct 1, 2023*

- monaco-editor: update to the latest version (0.43.0)

## 1.3.3
###### *Apr 2, 2023*

- monaco-editor: update to the latest version (0.36.1)

## 1.3.2
###### *May 11, 2022*

- utility: resolve monaco instance in case of provided monaco instance and global availability

## 1.3.1
###### *Apr 23, 2022*

- utility: implement isInitialized flag

## 1.3.0
###### *Mar 20, 2022*

- types: add optional monaco type into config params
- utility: implement optional monaco param for config
- test: fix a test case according to the new changes
- playground: create a playground for testing the library
- monaco-editor: update to the latest version (0.33.0)

## 1.2.0
###### *Oct 3, 2021*

- monaco-editor: update to the latest version (0.28.1)
- types: fix CancelablePromise type

## 1.1.1
###### *Jun 21, 2021*

- monaco-editor: update to the latest version (0.25.2)

## 1.1.0
###### *Jun 12, 2021*

- monaco-editor: update to the latest version (0.25.0)

## 1.0.1
###### *Mar 18, 2021*

- monaco-editor: update to the latest version (0.23.0)

## 1.0.0
###### *Jan 15, 2021*

🎉 First stable release

- utility: rename the main utility: monaco -> loader
- helpers: create (+ named export) `__getMonacoInstance` internal helper

## 0.1.3
###### *Jan 8, 2021*

- build: in `cjs` and `es` bundles `state-local` is marked as externam lib
- build: in `cjs` and `es` modules structure is preserved - `output.preserveModules = true`

## 0.1.2
###### *Jan 7, 2021*

- package: add jsdelivr source path

## 0.1.1
###### *Jan 7, 2021*

- lib: rename scripts name (from 'core' to 'loader')

## 0.1.0
###### *Jan 6, 2021*

🎉 First release
