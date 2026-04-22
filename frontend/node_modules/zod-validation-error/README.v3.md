# zod-validation-error

Wrap zod validation errors in user-friendly readable messages.

[![Build Status](https://github.com/causaly/zod-validation-error/actions/workflows/ci.yml/badge.svg)](https://github.com/causaly/zod-validation-error/actions/workflows/ci.yml) [![npm version](https://img.shields.io/npm/v/zod-validation-error.svg?color=0c0)](https://www.npmjs.com/package/zod-validation-error)

#### Features

- User-friendly readable messages, configurable via options;
- Maintain original issues under `error.details`;
- Supports both `zod` v3 and v4.

**_Note:_** This is the v3 version of `zod-validation-error`. If you are looking for zod v4 support, please click [here](/README.md).

## Installation

```bash
npm install zod-validation-error
```

#### Requirements

- Node.js v.18+
- TypeScript v.4.5+

## Quick start

```typescript
import { z as zod } from 'zod/v3';
import { fromError } from 'zod-validation-error/v3';

// create zod schema
const zodSchema = zod.object({
  id: zod.number().int().positive(),
  email: zod.string().email(),
});

// parse some invalid value
try {
  zodSchema.parse({
    id: 1,
    email: 'foobar', // note: invalid email
  });
} catch (err) {
  const validationError = fromError(err);
  // the error is now readable by the user
  // you may print it to console
  console.log(validationError.toString());
  // or return it as an actual error
  return validationError;
}
```

## Motivation

Zod errors are difficult to consume for the end-user. This library wraps Zod validation errors in user-friendly readable messages that can be exposed to the outer world, while maintaining the original errors in an array for _dev_ use.

### Example

#### Input (from Zod)

```json
[
  {
    "code": "too_small",
    "inclusive": false,
    "message": "Number must be greater than 0",
    "minimum": 0,
    "path": ["id"],
    "type": "number"
  },
  {
    "code": "invalid_string",
    "message": "Invalid email",
    "path": ["email"],
    "validation": "email"
  }
]
```

#### Output

```
Validation error: Number must be greater than 0 at "id"; Invalid email at "email"
```

## API

- [ValidationError(message[, options])](#validationerror)
- [createMessageBuilder(props)](#createMessageBuilder)
- [errorMap](#errormap)
- [isValidationError(error)](#isvalidationerror)
- [isValidationErrorLike(error)](#isvalidationerrorlike)
- [isZodErrorLike(error)](#iszoderrorlike)
- [fromError(error[, options])](#fromerror)
- [fromZodIssue(zodIssue[, options])](#fromzodissue)
- [fromZodError(zodError[, options])](#fromzoderror)
- [toValidationError([options]) => (error) => ValidationError](#tovalidationerror)

### ValidationError

Main `ValidationError` class, extending native JavaScript `Error`.

#### Arguments

- `message` - _string_; error message (required)
- `options` - _ErrorOptions_; error options as per [JavaScript definition](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Error/Error#options) (optional)
  - `options.cause` - _any_; can be used to hold the original zod error (optional)

#### Example 1: construct new ValidationError with `message`

```typescript
const { ValidationError } = require('zod-validation-error');

const error = new ValidationError('foobar');
console.log(error instanceof Error); // prints true
```

#### Example 2: construct new ValidationError with `message` and `options.cause`

```typescript
import { z as zod } from 'zod/v3';
const { ValidationError } = require('zod-validation-error');

const error = new ValidationError('foobar', {
  cause: new zod.ZodError([
    {
      code: 'invalid_string',
      message: 'Invalid email',
      path: ['email'],
      validation: 'email',
    },
  ]),
});

console.log(error.details); // prints issues from zod error
```

### createMessageBuilder

Creates zod-validation-error's default `MessageBuilder`, which is used to produce user-friendly error messages.

Meant to be passed as an option to [fromError](#fromerror), [fromZodIssue](#fromzodissue), [fromZodError](#fromzoderror) or [toValidationError](#tovalidationerror).

You may read more on the concept of the `MessageBuilder` further [below](#MessageBuilder).

#### Arguments

- `props` - _Object_; formatting options (optional)
  - `maxIssuesInMessage` - _number_; max issues to include in user-friendly message (optional, defaults to 99)
  - `issueSeparator` - _string_; used to concatenate issues in user-friendly message (optional, defaults to ";")
  - `unionSeparator` - _string_; used to concatenate union-issues in user-friendly message (optional, defaults to ", or")
  - `prefix` - _string_ or _null_; prefix to use in user-friendly message (optional, defaults to "Validation error"). Pass `null` to disable prefix completely.
  - `prefixSeparator` - _string_; used to concatenate prefix with rest of the user-friendly message (optional, defaults to ": "). Not used when `prefix` is `null`.
  - `includePath` - _boolean_; used to provide control on whether to include the erroneous property name suffix or not (optional, defaults to `true`).

#### Example

```typescript
import { createMessageBuilder } from 'zod-validation-error/v3';

const messageBuilder = createMessageBuilder({
  includePath: false,
  maxIssuesInMessage: 3,
});
```

### errorMap

A custom error map to use with zod's `setErrorMap` method and get user-friendly messages automatically.

#### Example

```typescript
import { z as zod } from 'zod/v3';
import { errorMap } from 'zod-validation-error/v3';

zod.setErrorMap(errorMap);
```

### isValidationError

A [type guard](https://www.typescriptlang.org/docs/handbook/2/narrowing.html#using-type-predicates) utility function, based on `instanceof` comparison.

#### Arguments

- `error` - error instance (required)

#### Example

```typescript
import { z as zod } from 'zod/v3';
import { ValidationError, isValidationError } from 'zod-validation-error/v3';

const err = new ValidationError('foobar');
isValidationError(err); // returns true

const invalidErr = new Error('foobar');
isValidationError(err); // returns false
```

### isValidationErrorLike

A [type guard](https://www.typescriptlang.org/docs/handbook/2/narrowing.html#using-type-predicates) utility function, based on _heuristics_ comparison.

_Why do we need heuristics since we can use a simple `instanceof` comparison?_ Because of multi-version inconsistencies. For instance, it's possible that a dependency is using an older `zod-validation-error` version internally. In such case, the `instanceof` comparison will yield invalid results because module deduplication does not apply at npm/yarn level and the prototype is different.

tl;dr if you are uncertain then it is preferable to use `isValidationErrorLike` instead of `isValidationError`.

#### Arguments

- `error` - error instance (required)

#### Example

```typescript
import {
  ValidationError,
  isValidationErrorLike,
} from 'zod-validation-error/v3';

const err = new ValidationError('foobar');
isValidationErrorLike(err); // returns true

const invalidErr = new Error('foobar');
isValidationErrorLike(err); // returns false
```

### isZodErrorLike

A [type guard](https://www.typescriptlang.org/docs/handbook/2/narrowing.html#using-type-predicates) utility function, based on _heuristics_ comparison.

_Why do we need heuristics since we can use a simple `instanceof` comparison?_ Because of multi-version inconsistencies. For instance, it's possible that a dependency is using an older `zod` version internally. In such case, the `instanceof` comparison will yield invalid results because module deduplication does not apply at npm/yarn level and the prototype is different.

#### Arguments

- `error` - error instance (required)

#### Example

```typescript
import { z as zod } from 'zod/v3';
import { ValidationError, isZodErrorLike } from 'zod-validation-error/v3';

const zodValidationErr = new ValidationError('foobar');
isZodErrorLike(zodValidationErr); // returns false

const genericErr = new Error('foobar');
isZodErrorLike(genericErr); // returns false

const zodErr = new zod.ZodError([
  {
    code: zod.ZodIssueCode.custom,
    path: [],
    message: 'foobar',
    fatal: true,
  },
]);
isZodErrorLike(zodErr); // returns true
```

### fromError

Converts an error to `ValidationError`.

_What is the difference between `fromError` and `fromZodError`?_ The `fromError` function is a less strict version of `fromZodError`. It can accept an unknown error and attempt to convert it to a `ValidationError`.

#### Arguments

- `error` - _unknown_; an error (required)
- `options` - _Object_; formatting options (optional)
  - `messageBuilder` - _MessageBuilder_; a function that accepts an array of `zod.ZodIssue` objects and returns a user-friendly error message in the form of a `string` (optional).

#### Notes

Alternatively, you may pass the following `options` instead of a `messageBuilder`.

- `options` - _Object_; formatting options (optional)
  - `maxIssuesInMessage` - _number_; max issues to include in user-friendly message (optional, defaults to 99)
  - `issueSeparator` - _string_; used to concatenate issues in user-friendly message (optional, defaults to ";")
  - `unionSeparator` - _string_; used to concatenate union-issues in user-friendly message (optional, defaults to ", or")
  - `prefix` - _string_ or _null_; prefix to use in user-friendly message (optional, defaults to "Validation error"). Pass `null` to disable prefix completely.
  - `prefixSeparator` - _string_; used to concatenate prefix with rest of the user-friendly message (optional, defaults to ": "). Not used when `prefix` is `null`.
  - `includePath` - _boolean_; used to provide control on whether to include the erroneous property name suffix or not (optional, defaults to `true`).

They will be passed as arguments to the [createMessageBuilder](#createMessageBuilder) function. The only reason they exist is to provide backwards-compatibility with older versions of `zod-validation-error`. They should however be considered deprecated and may be removed in the future.

### fromZodIssue

Converts a single zod issue to `ValidationError`.

#### Arguments

- `zodIssue` - _zod.ZodIssue_; a ZodIssue instance (required)
- `options` - _Object_; formatting options (optional)
  - `messageBuilder` - _MessageBuilder_; a function that accepts an array of `zod.ZodIssue` objects and returns a user-friendly error message in the form of a `string` (optional).

#### Notes

Alternatively, you may pass the following `options` instead of a `messageBuilder`.

- `options` - _Object_; formatting options (optional)
  - `issueSeparator` - _string_; used to concatenate issues in user-friendly message (optional, defaults to ";")
  - `unionSeparator` - _string_; used to concatenate union-issues in user-friendly message (optional, defaults to ", or")
  - `prefix` - _string_ or _null_; prefix to use in user-friendly message (optional, defaults to "Validation error"). Pass `null` to disable prefix completely.
  - `prefixSeparator` - _string_; used to concatenate prefix with rest of the user-friendly message (optional, defaults to ": "). Not used when `prefix` is `null`.
  - `includePath` - _boolean_; used to provide control on whether to include the erroneous property name suffix or not (optional, defaults to `true`).

They will be passed as arguments to the [createMessageBuilder](#createMessageBuilder) function. The only reason they exist is to provide backwards-compatibility with older versions of `zod-validation-error`. They should however be considered deprecated and may be removed in the future.

### fromZodError

Converts zod error to `ValidationError`.

_Why is the difference between `ZodError` and `ZodIssue`?_ A `ZodError` is a collection of 1 or more `ZodIssue` instances. It's what you get when you call `zodSchema.parse()`.

#### Arguments

- `zodError` - _zod.ZodError_; a ZodError instance (required)
- `options` - _Object_; formatting options (optional)
  - `messageBuilder` - _MessageBuilder_; a function that accepts an array of `zod.ZodIssue` objects and returns a user-friendly error message in the form of a `string` (optional).

#### Notes

Alternatively, you may pass the following `options` instead of a `messageBuilder`.

- `options` - _Object_; formatting options (optional)
  - `maxIssuesInMessage` - _number_; max issues to include in user-friendly message (optional, defaults to 99)
  - `issueSeparator` - _string_; used to concatenate issues in user-friendly message (optional, defaults to ";")
  - `unionSeparator` - _string_; used to concatenate union-issues in user-friendly message (optional, defaults to ", or")
  - `prefix` - _string_ or _null_; prefix to use in user-friendly message (optional, defaults to "Validation error"). Pass `null` to disable prefix completely.
  - `prefixSeparator` - _string_; used to concatenate prefix with rest of the user-friendly message (optional, defaults to ": "). Not used when `prefix` is `null`.
  - `includePath` - _boolean_; used to provide control on whether to include the erroneous property name suffix or not (optional, defaults to `true`).

They will be passed as arguments to the [createMessageBuilder](#createMessageBuilder) function. The only reason they exist is to provide backwards-compatibility with older versions of `zod-validation-error`. They should however be considered deprecated and may be removed in the future.

### toValidationError

A curried version of `fromZodError` meant to be used for FP (Functional Programming). Note it first takes the options object if needed and returns a function that converts the `zodError` to a `ValidationError` object

```js
toValidationError(options) => (zodError) => ValidationError
```

#### Example using fp-ts

```typescript
import * as Either from 'fp-ts/Either';
import { z as zod } from 'zod/v3';
import { toValidationError, ValidationError } from 'zod-validation-error/v3';

// create zod schema
const zodSchema = zod
  .object({
    id: zod.number().int().positive(),
    email: zod.string().email(),
  })
  .brand<'User'>();

export type User = zod.infer<typeof zodSchema>;

export function parse(
  value: zod.input<typeof zodSchema>
): Either.Either<ValidationError, User> {
  return Either.tryCatch(() => schema.parse(value), toValidationError());
}
```

## MessageBuilder

`zod-validation-error` can be configured with a custom `MessageBuilder` function in order to produce case-specific error messages.

#### Example

For instance, one may want to print `invalid_string` errors to the console in red color.

```typescript
import { z as zod } from 'zod/v3';
import { type MessageBuilder, fromError } from 'zod-validation-error/v3';
import chalk from 'chalk';

// create custom MessageBuilder
const myMessageBuilder: MessageBuilder = (issues) => {
  return (
    issues
      // format error message
      .map((issue) => {
        if (issue.code === zod.ZodIssueCode.invalid_string) {
          return chalk.red(issue.message);
        }

        return issue.message;
      })
      // join as string with new-line character
      .join('\n')
  );
};

// create zod schema
const zodSchema = zod.object({
  id: zod.number().int().positive(),
  email: zod.string().email(),
});

// parse some invalid value
try {
  zodSchema.parse({
    id: 1,
    email: 'foobar', // note: invalid email value
  });
} catch (err) {
  const validationError = fromError(err, {
    messageBuilder: myMessageBuilder,
  });
  // the error is now displayed with red letters
  console.log(validationError.toString());
}
```

## FAQ

### How to distinguish between errors

Use the `isValidationErrorLike` type guard.

#### Example

Scenario: Distinguish between `ValidationError` VS generic `Error` in order to respond with 400 VS 500 HTTP status code respectively.

```typescript
import * as Either from 'fp-ts/Either';
import { z as zod } from 'zod/v3';
import { isValidationErrorLike } from 'zod-validation-error/v3';

try {
  func(); // throws Error - or - ValidationError
} catch (err) {
  if (isValidationErrorLike(err)) {
    return 400; // Bad Data (this is a client error)
  }

  return 500; // Server Error
}
```

### How to use `ValidationError` outside `zod`

It's possible to implement custom validation logic outside `zod` and throw a `ValidationError`.

#### Example 1: passing custom message

```typescript
import { ValidationError } from 'zod-validation-error/v3';
import { Buffer } from 'node:buffer';

function parseBuffer(buf: unknown): Buffer {
  if (!Buffer.isBuffer(buf)) {
    throw new ValidationError('Invalid argument; expected buffer');
  }

  return buf;
}
```

#### Example 2: passing custom message and original error as cause

```typescript
import { ValidationError } from 'zod-validation-error/v3';

try {
  // do something that throws an error
} catch (err) {
  throw new ValidationError('Something went deeply wrong', { cause: err });
}
```

### How to use `ValidationError` with custom "error map"

Zod supports customizing error messages by providing a custom "error map". You may combine this with `zod-validation-error` to produce user-friendly messages.

#### Example 1: produce user-friendly error messages using the `errorMap` property

If all you need is to produce user-friendly error messages you may use the `errorMap` property.

```typescript
import { z as zod } from 'zod/v3';
import { errorMap } from 'zod-validation-error/v3';

zod.setErrorMap(errorMap);
```

#### Example 2: extra customization using `fromZodIssue`

If you need to customize some error code, you may use the `fromZodIssue` function.

```typescript
import { z as zod } from 'zod/v3';
import { fromZodIssue } from 'zod-validation-error/v3';

const customErrorMap: zod.ZodErrorMap = (issue, ctx) => {
  switch (issue.code) {
    case ZodIssueCode.invalid_type: {
      return {
        message:
          'Custom error message of your preference for invalid_type errors',
      };
    }
    default: {
      const validationError = fromZodIssue({
        ...issue,
        // fallback to the default error message
        // when issue does not have a message
        message: issue.message ?? ctx.defaultError,
      });

      return {
        message: validationError.message,
      };
    }
  }
};

zod.setErrorMap(customErrorMap);
```

### How to use `zod-validation-error` with `react-hook-form`?

```typescript
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { errorMap } from 'zod-validation-error/v3';

useForm({
  resolver: zodResolver(schema, { errorMap }),
});
```

### Does `zod-validation-error` support CommonJS

Yes, `zod-validation-error` supports CommonJS out-of-the-box. All you need to do is import it using `require`.

#### Example

```typescript
const { ValidationError } = require('zod-validation-error');
```

## Contribute

Source code contributions are most welcome. Please open a PR, ensure the linter is satisfied and all tests pass.

#### We are hiring

Causaly is building the world's largest biomedical knowledge platform, using technologies such as TypeScript, React and Node.js. Find out more about our openings at https://jobs.ashbyhq.com/causaly.

## License

MIT
