## v4 changes

See also the [CHANGELOG](https://github.com/suren-atoyan/monaco-react/blob/master/CHANGELOG.md)

`v4` brings lots of new features and some breaking changes. First, let's see what's broken comparing with the `v3`

1) `editorDidMount` - **removed**

In the new version there is no `editorDidMount` property. Now it can be replaced by `onMount`. If you happen to remember, the signature of `editorDidMount` was `function(getEditorValue: () => string, editor: monaco.editor.IStandaloneCodeEditor) => void`

The `onMount` prop is a little bit different, the signature is `function(editor: monaco.editor.IStandaloneCodeEditor, monaco: Monaco) => void`. As you can see it also exposes the `monaco` instance as a second parameter. There is no `getEditorValue`. To get the current model value you can use `editor` instance (`editor.getValue()`) or `onChange` prop. [Read more](https://github.com/suren-atoyan/monaco-react#get-value)

2) `beforeMount` - **new** (`onMount` - **new**)

In addition to `onMount` now there is also `beforeMount` that exposes the `monaco` instance as a first parameter. Here you can do something before the editor is mounted. [Read more](https://github.com/suren-atoyan/monaco-react#monaco-instance)

3) `ControlledEditor` - **removed**

`ControlledEditor` component is removed. Now there is one `Editor` component that behaves in `controlled` mode or `uncontrolled` mode based on the usage of `value` property. [Read more](https://github.com/suren-atoyan/monaco-react#uncontrolled-controlled-modes)

4) `multi-model editor` - **new**

multi-model editor is supported now. Check [this](https://codesandbox.io/s/multi-model-editor-kugi6?file=/src/App.js) for simple demo. [Read more](https://github.com/suren-atoyan/monaco-react#multi-model-editor)

5) `defaultPath ` - **new**

Default path of the current model. Will be passed as the third argument to `.createModel` method `monaco.editor.createModel(..., ..., monaco.Uri.parse(defaultPath))`

For `DiffEditor` there are `originalModelPath` and `modifiedModelPath`

6) `path` - **new**

Path of the current model. Will be passed as the third argument to `.createModel` method `monaco.editor.createModel(..., ..., monaco.Uri.parse(defaultPath))`

7) `defaultLanguage` - **new**

Default language of the current model

8) `useMonaco` - **new**

`useMonaco` is a `React` hook that returns the instance of the `monaco`. [Read more](https://github.com/suren-atoyan/monaco-react#usemonaco)

9) `onValidate` - **new**

`onValidate` is an additional property. An event is emitted when the length of the model markers of the current model isn't 0. [Read more](https://github.com/suren-atoyan/monaco-react#onvalidate)

10) `defaultProp` - **new**

The initial value of the default (auto created) model

11) `monaco` utility - **renamed**

now, it's called `loader`. [Read more](https://github.com/suren-atoyan/monaco-react#loader-config)

12) `onChange` - **signature change**

old - `function(ev: any, value: string | undefined) => string | undefined`
<br />
new - `function(value: string | undefined, ev: monaco.editor.IModelContentChangedEvent) => void`
