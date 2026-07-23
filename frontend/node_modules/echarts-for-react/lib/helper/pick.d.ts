/**
 * 保留 object 中的部分内容
 * @param obj
 * @param keys
 */
export declare function pick<T extends object>(obj: T, keys: readonly (keyof T)[]): Partial<T>;
