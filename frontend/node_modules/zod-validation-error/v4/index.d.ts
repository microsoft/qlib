import * as zod from 'zod/v4/core';

declare const ZOD_VALIDATION_ERROR_NAME = "ZodValidationError";
interface ErrorOptions {
    cause?: unknown;
}
declare class ValidationError extends Error {
    name: typeof ZOD_VALIDATION_ERROR_NAME;
    details: Array<zod.$ZodIssue>;
    constructor(message?: string, options?: ErrorOptions);
    toString(): string;
}

declare function isValidationError(err: unknown): err is ValidationError;

declare function isValidationErrorLike(err: unknown): err is ValidationError;

declare function isZodErrorLike(err: unknown): err is zod.$ZodError;

type ErrorMapOptions = {
    dateLocalization: boolean | Intl.LocalesArgument;
    numberLocalization: boolean | Intl.LocalesArgument;
    displayInvalidFormatDetails: boolean;
    allowedValuesSeparator: string;
    allowedValuesLastSeparator: string | undefined;
    wrapAllowedValuesInQuote: boolean;
    maxAllowedValuesToDisplay: number;
    unrecognizedKeysSeparator: string;
    unrecognizedKeysLastSeparator: string | undefined;
    wrapUnrecognizedKeysInQuote: boolean;
    maxUnrecognizedKeysToDisplay: number;
};

declare function createErrorMap(partialOptions?: Partial<ErrorMapOptions>): zod.$ZodErrorMap<zod.$ZodIssue>;

type NonEmptyArray<T> = [T, ...T[]];

type ZodIssue = zod.$ZodIssue;
type MessageBuilder = (issues: NonEmptyArray<ZodIssue>) => string;
type MessageBuilderOptions = {
    prefix: string | null | undefined;
    prefixSeparator: string;
    maxIssuesInMessage: number;
    issueSeparator: string;
    unionSeparator: string;
    includePath: boolean;
    forceTitleCase: boolean;
};
declare function createMessageBuilder(partialOptions?: Partial<MessageBuilderOptions>): MessageBuilder;

type ZodError = zod.$ZodError;
type FromZodErrorOptions = {
    messageBuilder: MessageBuilder;
} | Partial<MessageBuilderOptions>;
declare function fromZodError(zodError: ZodError, options?: FromZodErrorOptions): ValidationError;

declare function fromError(err: unknown, options?: FromZodErrorOptions): ValidationError;

type FromZodIssueOptions = {
    messageBuilder: MessageBuilder;
} | Partial<Omit<MessageBuilderOptions, 'maxIssuesInMessage'>>;
declare function fromZodIssue(issue: ZodIssue, options?: FromZodIssueOptions): ValidationError;

declare const toValidationError: (options?: FromZodErrorOptions) => (err: unknown) => ValidationError;

export { type ErrorMapOptions, type ErrorOptions, type FromZodErrorOptions, type FromZodIssueOptions, type MessageBuilder, type MessageBuilderOptions, type NonEmptyArray, ValidationError, type ZodError, type ZodIssue, createErrorMap, createMessageBuilder, fromError, fromZodError, fromZodIssue, isValidationError, isValidationErrorLike, isZodErrorLike, toValidationError };
