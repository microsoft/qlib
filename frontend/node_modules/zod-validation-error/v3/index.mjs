// lib/v3/isZodErrorLike.ts
function isZodErrorLike(err) {
  return err instanceof Error && err.name === "ZodError" && "issues" in err && Array.isArray(err.issues);
}

// lib/v3/ValidationError.ts
var ValidationError = class extends Error {
  name;
  details;
  constructor(message, options) {
    super(message, options);
    this.name = "ZodValidationError";
    this.details = getIssuesFromErrorOptions(options);
  }
  toString() {
    return this.message;
  }
};
function getIssuesFromErrorOptions(options) {
  if (options) {
    const cause = options.cause;
    if (isZodErrorLike(cause)) {
      return cause.issues;
    }
  }
  return [];
}

// lib/v3/isValidationError.ts
function isValidationError(err) {
  return err instanceof ValidationError;
}

// lib/v3/isValidationErrorLike.ts
function isValidationErrorLike(err) {
  return err instanceof Error && err.name === "ZodValidationError";
}

// lib/v3/fromZodIssue.ts
import * as zod2 from "zod/v3";

// lib/v3/MessageBuilder.ts
import * as zod from "zod/v3";

// lib/utils/NonEmptyArray.ts
function isNonEmptyArray(value) {
  return value.length !== 0;
}

// lib/utils/stringify.ts
function stringifySymbol(symbol) {
  return symbol.description ?? "";
}

// lib/utils/joinPath.ts
var identifierRegex = /[$_\p{ID_Start}][$\u200c\u200d\p{ID_Continue}]*/u;
function joinPath(path) {
  if (path.length === 1) {
    let propertyKey = path[0];
    if (typeof propertyKey === "symbol") {
      propertyKey = stringifySymbol(propertyKey);
    }
    return propertyKey.toString() || '""';
  }
  return path.reduce((acc, propertyKey) => {
    if (typeof propertyKey === "number") {
      return acc + "[" + propertyKey.toString() + "]";
    }
    if (typeof propertyKey === "symbol") {
      propertyKey = stringifySymbol(propertyKey);
    }
    if (propertyKey.includes('"')) {
      return acc + '["' + escapeQuotes(propertyKey) + '"]';
    }
    if (!identifierRegex.test(propertyKey)) {
      return acc + '["' + propertyKey + '"]';
    }
    const separator = acc.length === 0 ? "" : ".";
    return acc + separator + propertyKey;
  }, "");
}
function escapeQuotes(str) {
  return str.replace(/"/g, '\\"');
}

// lib/v3/config.ts
var ISSUE_SEPARATOR = "; ";
var MAX_ISSUES_IN_MESSAGE = 99;
var PREFIX = "Validation error";
var PREFIX_SEPARATOR = ": ";
var UNION_SEPARATOR = ", or ";

// lib/v3/MessageBuilder.ts
function createMessageBuilder(props = {}) {
  const {
    issueSeparator = ISSUE_SEPARATOR,
    unionSeparator = UNION_SEPARATOR,
    prefixSeparator = PREFIX_SEPARATOR,
    prefix = PREFIX,
    includePath = true,
    maxIssuesInMessage = MAX_ISSUES_IN_MESSAGE
  } = props;
  return (issues) => {
    const message = issues.slice(0, maxIssuesInMessage).map(
      (issue) => getMessageFromZodIssue({
        issue,
        issueSeparator,
        unionSeparator,
        includePath
      })
    ).join(issueSeparator);
    return prefixMessage(message, prefix, prefixSeparator);
  };
}
function getMessageFromZodIssue(props) {
  const { issue, issueSeparator, unionSeparator, includePath } = props;
  if (issue.code === zod.ZodIssueCode.invalid_union) {
    return issue.unionErrors.reduce((acc, zodError) => {
      const newIssues = zodError.issues.map(
        (issue2) => getMessageFromZodIssue({
          issue: issue2,
          issueSeparator,
          unionSeparator,
          includePath
        })
      ).join(issueSeparator);
      if (!acc.includes(newIssues)) {
        acc.push(newIssues);
      }
      return acc;
    }, []).join(unionSeparator);
  }
  if (issue.code === zod.ZodIssueCode.invalid_arguments) {
    return [
      issue.message,
      ...issue.argumentsError.issues.map(
        (issue2) => getMessageFromZodIssue({
          issue: issue2,
          issueSeparator,
          unionSeparator,
          includePath
        })
      )
    ].join(issueSeparator);
  }
  if (issue.code === zod.ZodIssueCode.invalid_return_type) {
    return [
      issue.message,
      ...issue.returnTypeError.issues.map(
        (issue2) => getMessageFromZodIssue({
          issue: issue2,
          issueSeparator,
          unionSeparator,
          includePath
        })
      )
    ].join(issueSeparator);
  }
  if (includePath && isNonEmptyArray(issue.path)) {
    if (issue.path.length === 1) {
      const identifier = issue.path[0];
      if (typeof identifier === "number") {
        return `${issue.message} at index ${identifier}`;
      }
    }
    return `${issue.message} at "${joinPath(issue.path)}"`;
  }
  return issue.message;
}
function prefixMessage(message, prefix, prefixSeparator) {
  if (prefix !== null) {
    if (message.length > 0) {
      return [prefix, message].join(prefixSeparator);
    }
    return prefix;
  }
  if (message.length > 0) {
    return message;
  }
  return PREFIX;
}

// lib/v3/fromZodIssue.ts
function fromZodIssue(issue, options = {}) {
  const messageBuilder = createMessageBuilderFromOptions(options);
  const message = messageBuilder([issue]);
  return new ValidationError(message, { cause: new zod2.ZodError([issue]) });
}
function createMessageBuilderFromOptions(options) {
  if ("messageBuilder" in options) {
    return options.messageBuilder;
  }
  return createMessageBuilder(options);
}

// lib/v3/errorMap.ts
var errorMap = (issue, ctx) => {
  const error = fromZodIssue({
    ...issue,
    // fallback to the default error message
    // when issue does not have a message
    message: issue.message ?? ctx.defaultError
  });
  return {
    message: error.message
  };
};

// lib/v3/fromZodError.ts
function fromZodError(zodError, options = {}) {
  if (!isZodErrorLike(zodError)) {
    throw new TypeError(
      `Invalid zodError param; expected instance of ZodError. Did you mean to use the "${fromError.name}" method instead?`
    );
  }
  return fromZodErrorWithoutRuntimeCheck(zodError, options);
}
function fromZodErrorWithoutRuntimeCheck(zodError, options = {}) {
  const zodIssues = zodError.errors;
  let message;
  if (isNonEmptyArray(zodIssues)) {
    const messageBuilder = createMessageBuilderFromOptions2(options);
    message = messageBuilder(zodIssues);
  } else {
    message = zodError.message;
  }
  return new ValidationError(message, { cause: zodError });
}
function createMessageBuilderFromOptions2(options) {
  if ("messageBuilder" in options) {
    return options.messageBuilder;
  }
  return createMessageBuilder(options);
}

// lib/v3/toValidationError.ts
var toValidationError = (options = {}) => (err) => {
  if (isZodErrorLike(err)) {
    return fromZodErrorWithoutRuntimeCheck(err, options);
  }
  if (err instanceof Error) {
    return new ValidationError(err.message, { cause: err });
  }
  return new ValidationError("Unknown error");
};

// lib/v3/fromError.ts
function fromError(err, options = {}) {
  return toValidationError(options)(err);
}
export {
  ValidationError,
  createMessageBuilder,
  errorMap,
  fromError,
  fromZodError,
  fromZodIssue,
  isValidationError,
  isValidationErrorLike,
  isZodErrorLike,
  toValidationError
};
//# sourceMappingURL=index.mjs.map