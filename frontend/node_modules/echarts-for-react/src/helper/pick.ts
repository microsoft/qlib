/**
 * 保留 object 中的部分内容
 * @param obj
 * @param keys
 */
export function pick<T extends object>(obj: T, keys: readonly (keyof T)[]): Partial<T> {
  const r = {} as Partial<T>;
  keys.forEach((key) => {
    r[key] = obj[key];
  });
  return r;
}
