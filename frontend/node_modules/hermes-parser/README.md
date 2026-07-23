# hermes-parser
A JavaScript parser built from the Hermes engine's parser compiled to WebAssembly. Can parse ES6, Flow, and JSX syntax.

## API
The Hermes parser exposes a single `parse(code, [options])` function, where `code` is the source code to parse as a string, and `options` is an optional object that may contain the following properties:
- **babel**: `boolean`, defaults to `false`. If `true`, output an AST conforming to Babel's AST format. If `false`, output an AST conforming to the ESTree AST format.
- **allowReturnOutsideFunction**: `boolean`, defaults to `false`. If `true`, do not error on return statements found outside functions.
- **flow**: `"all"` or `"detect"`, defaults to `"detect"`. If `"detect"`, only parse syntax as Flow syntax where it is ambiguous whether it is a Flow feature or regular JavaScript when the `@flow` pragma is present in the file. Otherwise if `"all"`, always parse ambiguous syntax as Flow syntax regardless of the presence of an `@flow` pragma. For example `foo<T>(x)` in a file without an `@flow` pragma will be parsed as two comparisons if set to `"detect"`, otherwise if set to `"all"` or the `@flow` pragma is included it will be parsed as a call expression with a type argument.
- **sourceFilename**: `string`, defaults to `null`. The filename corresponding to the code that is to be parsed. If non-null, the filename will be added to all source locations in the output AST.
- **sourceType**: `"module"`, `"script"`, or `"unambiguous"` (default). If `"unambiguous"`, source type will be automatically detected and set to `"module"` if any ES6 imports or exports are present in the code, otherwise source type will be set to `"script"`.
- **tokens**: `boolean`, defaults to `false`. If `true`, add all tokens to a `tokens` property on the root node.
