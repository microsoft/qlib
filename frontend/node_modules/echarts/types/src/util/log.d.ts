export declare function log(str: string, onlyOnce?: boolean): void;
export declare function warn(str: string, onlyOnce?: boolean): void;
export declare function error(str: string, onlyOnce?: boolean): void;
export declare function deprecateLog(str: string): void;
export declare function deprecateReplaceLog(oldOpt: string, newOpt: string, scope?: string): void;
/**
 * If in __DEV__ environment, get console printable message for users hint.
 * Parameters are separated by ' '.
 * @usage
 * makePrintable('This is an error on', someVar, someObj);
 *
 * @param hintInfo anything about the current execution context to hint users.
 * @throws Error
 */
export declare function makePrintable(...hintInfo: unknown[]): string;
/**
 * @throws Error
 */
export declare function throwError(msg?: string): void;
