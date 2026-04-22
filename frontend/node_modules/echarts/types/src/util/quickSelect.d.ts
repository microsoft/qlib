/**
 * Quick select n-th element in an array.
 *
 * Note: it will change the elements placement in array.
 */
declare type CompareFunc<T> = (a: T, b: T) => number;
/**
 * @example
 *     let arr = [5, 2, 1, 4, 3]
 *     quickSelect(arr, 3);
 *     quickSelect(arr, 0, 3, 1, function (a, b) {return a - b});
 *
 * @return {number}
 */
declare function quickSelect<T>(arr: T[], nth: number, compareFunc: CompareFunc<T>): number;
declare function quickSelect<T>(arr: T[], nth: number, left: number, right: number, compareFunc: CompareFunc<T>): number;
export default quickSelect;
