/* global: defineProperty */
import { Dictionary, ArrayLike, KeyOfDistributive } from './types';
import { GradientObject } from '../graphic/Gradient';
import { ImagePatternObject } from '../graphic/Pattern';
import { platformApi } from './platform';

// 用于处理merge时无法遍历Date等对象的问题
const BUILTIN_OBJECT: Record<string, boolean> = reduce([
    'Function',
    'RegExp',
    'Date',
    'Error',
    'CanvasGradient',
    'CanvasPattern',
    // For node-canvas
    'Image',
    'Canvas'
], (obj, val) => {
    obj['[object ' + val + ']'] = true;
    return obj;
}, {} as Record<string, boolean>);

const TYPED_ARRAY: Record<string, boolean> = reduce([
    'Int8',
    'Uint8',
    'Uint8Clamped',
    'Int16',
    'Uint16',
    'Int32',
    'Uint32',
    'Float32',
    'Float64'
], (obj, val) => {
    obj['[object ' + val + 'Array]'] = true;
    return obj;
}, {} as Record<string, boolean>);

const objToString = Object.prototype.toString;

const arrayProto = Array.prototype;
const nativeForEach = arrayProto.forEach;
const nativeFilter = arrayProto.filter;
const nativeSlice = arrayProto.slice;
const nativeMap = arrayProto.map;
// In case some env may redefine the global variable `Function`.
const ctorFunction = function () {}.constructor;
const protoFunction = ctorFunction ? ctorFunction.prototype : null;
const protoKey = '__proto__';


let idStart = 0x0907;

/**
 * Generate unique id
 */
export function guid(): number {
    return idStart++;
}

export function logError(...args: any[]) {
    if (typeof console !== 'undefined') {
        console.error.apply(console, args);
    }
}
/**
 * Those data types can be cloned:
 *     Plain object, Array, TypedArray, number, string, null, undefined.
 * Those data types will be assigned using the original data:
 *     BUILTIN_OBJECT
 * Instance of user defined class will be cloned to a plain object, without
 * properties in prototype.
 * Other data types is not supported (not sure what will happen).
 *
 * Caution: do not support clone Date, for performance consideration.
 * (There might be a large number of date in `series.data`).
 * So date should not be modified in and out of echarts.
 */
export function clone<T extends any>(source: T): T {
    if (source == null || typeof source !== 'object') {
        return source;
    }

    let result = source as any;
    const typeStr = <string>objToString.call(source);

    if (typeStr === '[object Array]') {
        if (!isPrimitive(source)) {
            result = [] as any;
            for (let i = 0, len = (source as any[]).length; i < len; i++) {
                result[i] = clone((source as any[])[i]);
            }
        }
    }
    else if (TYPED_ARRAY[typeStr]) {
        if (!isPrimitive(source)) {
            /* eslint-disable-next-line */
            const Ctor = source.constructor as typeof Float32Array;
            if (Ctor.from) {
                result = Ctor.from(source as Float32Array);
            }
            else {
                result = new Ctor((source as Float32Array).length);
                for (let i = 0, len = (source as Float32Array).length; i < len; i++) {
                    result[i] = (source as Float32Array)[i];
                }
            }
        }
    }
    else if (!BUILTIN_OBJECT[typeStr] && !isPrimitive(source) && !isDom(source)) {
        result = {} as any;
        for (let key in source) {
            // Check if key is __proto__ to avoid prototype pollution
            if (source.hasOwnProperty(key) && key !== protoKey) {
                result[key] = clone(source[key]);
            }
        }
    }

    return result;
}

export function merge<
    T extends Dictionary<any>,
    S extends Dictionary<any>
>(target: T, source: S, overwrite?: boolean): T & S;
export function merge<
    T extends any,
    S extends any
>(target: T, source: S, overwrite?: boolean): T | S;
export function merge(target: any, source: any, overwrite?: boolean): any {
    // We should escapse that source is string
    // and enter for ... in ...
    if (!isObject(source) || !isObject(target)) {
        return overwrite ? clone(source) : target;
    }

    for (let key in source) {
        // Check if key is __proto__ to avoid prototype pollution
        if (source.hasOwnProperty(key) && key !== protoKey) {
            const targetProp = target[key];
            const sourceProp = source[key];

            if (isObject(sourceProp)
                && isObject(targetProp)
                && !isArray(sourceProp)
                && !isArray(targetProp)
                && !isDom(sourceProp)
                && !isDom(targetProp)
                && !isBuiltInObject(sourceProp)
                && !isBuiltInObject(targetProp)
                && !isPrimitive(sourceProp)
                && !isPrimitive(targetProp)
            ) {
                // 如果需要递归覆盖，就递归调用merge
                merge(targetProp, sourceProp, overwrite);
            }
            else if (overwrite || !(key in target)) {
                // 否则只处理overwrite为true，或者在目标对象中没有此属性的情况
                // NOTE，在 target[key] 不存在的时候也是直接覆盖
                target[key] = clone(source[key]);
            }
        }
    }

    return target;
}

/**
 * @param targetAndSources The first item is target, and the rests are source.
 * @param overwrite
 * @return Merged result
 */
export function mergeAll(targetAndSources: any[], overwrite?: boolean): any {
    let result = targetAndSources[0];
    for (let i = 1, len = targetAndSources.length; i < len; i++) {
        result = merge(result, targetAndSources[i], overwrite);
    }
    return result;
}

export function extend<
    T extends Dictionary<any>,
    S extends Dictionary<any>
>(target: T, source: S): T & S {
    // @ts-ignore
    if (Object.assign) {
        // @ts-ignore
        Object.assign(target, source);
    }
    else {
        for (let key in source) {
            // Check if key is __proto__ to avoid prototype pollution
            if (source.hasOwnProperty(key) && key !== protoKey) {
                (target as S & T)[key] = (source as T & S)[key];
            }
        }
    }
    return target as T & S;
}

export function defaults<
    T extends Dictionary<any>,
    S extends Dictionary<any>
>(target: T, source: S, overlay?: boolean): T & S {
    const keysArr = keys(source);
    for (let i = 0, len = keysArr.length; i < len; i++) {
        let key = keysArr[i];
        if ((overlay ? source[key] != null : (target as T & S)[key] == null)) {
            (target as S & T)[key] = (source as T & S)[key];
        }
    }
    return target as T & S;
}

// Expose createCanvas in util for compatibility
export const createCanvas = platformApi.createCanvas;

/**
 * 查询数组中元素的index
 */
export function indexOf<T>(array: T[] | readonly T[] | ArrayLike<T>, value: T): number {
    if (array) {
        if ((array as T[]).indexOf) {
            return (array as T[]).indexOf(value);
        }
        for (let i = 0, len = array.length; i < len; i++) {
            if (array[i] === value) {
                return i;
            }
        }
    }
    return -1;
}

/**
 * 构造类继承关系
 *
 * @param clazz 源类
 * @param baseClazz 基类
 */
export function inherits(clazz: Function, baseClazz: Function) {
    const clazzPrototype = clazz.prototype;
    function F() {}
    F.prototype = baseClazz.prototype;
    clazz.prototype = new (F as any)();

    for (let prop in clazzPrototype) {
        if (clazzPrototype.hasOwnProperty(prop)) {
            clazz.prototype[prop] = clazzPrototype[prop];
        }
    }
    clazz.prototype.constructor = clazz;
    (clazz as any).superClass = baseClazz;
}

export function mixin<T, S>(target: T | Function, source: S | Function, override?: boolean) {
    target = 'prototype' in target ? target.prototype : target;
    source = 'prototype' in source ? source.prototype : source;
    // If build target is ES6 class. prototype methods is not enumerable. Use getOwnPropertyNames instead
    // TODO: Determine if source is ES6 class?
    if (Object.getOwnPropertyNames) {
        const keyList = Object.getOwnPropertyNames(source);
        for (let i = 0; i < keyList.length; i++) {
            const key = keyList[i];
            if (key !== 'constructor') {
                if ((override ? (source as any)[key] != null : (target as any)[key] == null)) {
                    (target as any)[key] = (source as any)[key];
                }
            }
        }
    }
    else {
        defaults(target, source, override);
    }
}

/**
 * Consider typed array.
 * @param data
 */
export function isArrayLike(data: any): data is ArrayLike<any> {
    if (!data) {
        return false;
    }
    if (typeof data === 'string') {
        return false;
    }
    return typeof data.length === 'number';
}

/**
 * 数组或对象遍历
 */
export function each<I extends Dictionary<any> | any[] | readonly any[] | ArrayLike<any>, Context>(
    arr: I,
    cb: (
        this: Context,
        // Use unknown to avoid to infer to "any", which may disable typo check.
        value: I extends (infer T)[] | readonly (infer T)[] | ArrayLike<infer T> ? T
            // Use Dictionary<infer T> may cause infer fail when I is an interface.
            // So here use a Record to infer type.
            : I extends Dictionary<any> ? I extends Record<infer K, infer T> ? T : unknown : unknown,
        index?: I extends any[] | readonly any[] | ArrayLike<any> ? number : keyof I & string,  // keyof Dictionary will return number | string
        arr?: I
    ) => void,
    context?: Context
) {
    if (!(arr && cb)) {
        return;
    }
    if ((arr as any).forEach && (arr as any).forEach === nativeForEach) {
        (arr as any).forEach(cb, context);
    }
    else if (arr.length === +arr.length) {
        for (let i = 0, len = arr.length; i < len; i++) {
            // FIXME: should the elided item be travelled? like `[33,,55]`.
            cb.call(context, (arr as any[])[i], i as any, arr);
        }
    }
    else {
        for (let key in arr) {
            if (arr.hasOwnProperty(key)) {
                cb.call(context, (arr as Dictionary<any>)[key], key as any, arr);
            }
        }
    }
}

/**
 * Array mapping.
 * @typeparam T Type in Array
 * @typeparam R Type Returned
 * @return Must be an array.
 */
export function map<T, R, Context>(
    arr: readonly T[],
    cb: (this: Context, val: T, index?: number, arr?: readonly T[]) => R,
    context?: Context
): R[] {
    // Take the same behavior with lodash when !arr and !cb,
    // which might be some common sense.
    if (!arr) {
        return [];
    }
    if (!cb) {
        return slice(arr) as unknown[] as R[];
    }
    if (arr.map && arr.map === nativeMap) {
        return arr.map(cb, context);
    }
    else {
        const result = [];
        for (let i = 0, len = arr.length; i < len; i++) {
            // FIXME: should the elided item be travelled, like `[33,,55]`.
            result.push(cb.call(context, arr[i], i, arr));
        }
        return result;
    }
}

export function reduce<T, S, Context>(
    arr: readonly T[],
    cb: (this: Context, previousValue: S, currentValue: T, currentIndex?: number, arr?: readonly T[]) => S,
    memo?: S,
    context?: Context
): S {
    if (!(arr && cb)) {
        return;
    }
    for (let i = 0, len = arr.length; i < len; i++) {
        memo = cb.call(context, memo, arr[i], i, arr);
    }
    return memo;
}

/**
 * Array filtering.
 * @return Must be an array.
 */
export function filter<T, Context>(
    arr: readonly T[],
    cb: (this: Context, value: T, index: number, arr: readonly T[]) => boolean,
    context?: Context
): T[] {
    // Take the same behavior with lodash when !arr and !cb,
    // which might be some common sense.
    if (!arr) {
        return [];
    }
    if (!cb) {
        return slice(arr);
    }
    if (arr.filter && arr.filter === nativeFilter) {
        return arr.filter(cb, context);
    }
    else {
        const result = [];
        for (let i = 0, len = arr.length; i < len; i++) {
            // FIXME: should the elided items be travelled? like `[33,,55]`.
            if (cb.call(context, arr[i], i, arr)) {
                result.push(arr[i]);
            }
        }
        return result;
    }
}

/**
 * 数组项查找
 */
export function find<T, Context>(
    arr: readonly T[],
    cb: (this: Context, value: T, index?: number, arr?: readonly T[]) => boolean,
    context?: Context
): T {
    if (!(arr && cb)) {
        return;
    }
    for (let i = 0, len = arr.length; i < len; i++) {
        if (cb.call(context, arr[i], i, arr)) {
            return arr[i];
        }
    }
}

/**
 * Get all object keys
 *
 * Will return an empty array if obj is null/undefined
 */
export function keys<T extends object>(obj: T): (KeyOfDistributive<T> & string)[] {
    if (!obj) {
        return [];
    }
    // Return type should be `keyof T` but exclude `number`, becuase
    // `Object.keys` only return string rather than `number | string`.
    type TKeys = KeyOfDistributive<T> & string;
    if (Object.keys) {
        return Object.keys(obj) as TKeys[];
    }
    let keyList: TKeys[] = [];
    for (let key in obj) {
        if (obj.hasOwnProperty(key)) {
            keyList.push(key as any);
        }
    }
    return keyList;
}

// Remove this type in returned function. Or it will conflicts wicth callback with given context. Like Eventful.
// According to lib.es5.d.ts
/* eslint-disable max-len*/
export type Bind1<F, Ctx> = F extends (this: Ctx, ...args: infer A) => infer R ? (...args: A) => R : unknown;
export type Bind2<F, Ctx, T1> = F extends (this: Ctx, a: T1, ...args: infer A) => infer R ? (...args: A) => R : unknown;
export type Bind3<F, Ctx, T1, T2> = F extends (this: Ctx, a: T1, b: T2, ...args: infer A) => infer R ? (...args: A) => R : unknown;
export type Bind4<F, Ctx, T1, T2, T3> = F extends (this: Ctx, a: T1, b: T2, c: T3, ...args: infer A) => infer R ? (...args: A) => R : unknown;
export type Bind5<F, Ctx, T1, T2, T3, T4> = F extends (this: Ctx, a: T1, b: T2, c: T3, d: T4, ...args: infer A) => infer R ? (...args: A) => R : unknown;
type BindFunc<Ctx> = (this: Ctx, ...arg: any[]) => any

interface FunctionBind {
    <F extends BindFunc<Ctx>, Ctx>(func: F, ctx: Ctx): Bind1<F, Ctx>
    <F extends BindFunc<Ctx>, Ctx, T1 extends Parameters<F>[0]>(func: F, ctx: Ctx, a: T1): Bind2<F, Ctx, T1>
    <F extends BindFunc<Ctx>, Ctx, T1 extends Parameters<F>[0], T2 extends Parameters<F>[1]>(func: F, ctx: Ctx, a: T1, b: T2): Bind3<F, Ctx, T1, T2>
    <F extends BindFunc<Ctx>, Ctx, T1 extends Parameters<F>[0], T2 extends Parameters<F>[1], T3 extends Parameters<F>[2]>(func: F, ctx: Ctx, a: T1, b: T2, c: T3): Bind4<F, Ctx, T1, T2, T3>
    <F extends BindFunc<Ctx>, Ctx, T1 extends Parameters<F>[0], T2 extends Parameters<F>[1], T3 extends Parameters<F>[2], T4 extends Parameters<F>[3]>(func: F, ctx: Ctx, a: T1, b: T2, c: T3, d: T4): Bind5<F, Ctx, T1, T2, T3, T4>
}
function bindPolyfill<Ctx, Fn extends(...args: any) => any>(
    func: Fn, context: Ctx, ...args: any[]
): (...args: Parameters<Fn>) => ReturnType<Fn> {
    return function (this: Ctx) {
        return func.apply(context, args.concat(nativeSlice.call(arguments)));
    };
}
export const bind: FunctionBind = (protoFunction && isFunction(protoFunction.bind))
    ? protoFunction.call.bind(protoFunction.bind)
    : bindPolyfill;

export type Curry1<F, T1> = F extends (a: T1, ...args: infer A) => infer R ? (...args: A) => R : unknown;
export type Curry2<F, T1, T2> = F extends (a: T1, b: T2, ...args: infer A) => infer R ? (...args: A) => R : unknown;
export type Curry3<F, T1, T2, T3> = F extends (a: T1, b: T2, c: T3, ...args: infer A) => infer R ? (...args: A) => R : unknown;
export type Curry4<F, T1, T2, T3, T4> = F extends (a: T1, b: T2, c: T3, d: T4, ...args: infer A) => infer R ? (...args: A) => R : unknown;
type CurryFunc = (...arg: any[]) => any

function curry<F extends CurryFunc, T1 extends Parameters<F>[0]>(func: F, a: T1): Curry1<F, T1>
function curry<F extends CurryFunc, T1 extends Parameters<F>[0], T2 extends Parameters<F>[1]>(func: F, a: T1, b: T2): Curry2<F, T1, T2>
function curry<F extends CurryFunc, T1 extends Parameters<F>[0], T2 extends Parameters<F>[1], T3 extends Parameters<F>[2]>(func: F, a: T1, b: T2, c: T3): Curry3<F, T1, T2, T3>
function curry<F extends CurryFunc, T1 extends Parameters<F>[0], T2 extends Parameters<F>[1], T3 extends Parameters<F>[2], T4 extends Parameters<F>[3]>(func: F, a: T1, b: T2, c: T3, d: T4): Curry4<F, T1, T2, T3, T4>
function curry(func: Function, ...args: any[]): Function {
    return function (this: any) {
        return func.apply(this, args.concat(nativeSlice.call(arguments)));
    };
}
export {curry};
/* eslint-enable max-len*/

export function isArray(value: any): value is any[] {
    if (Array.isArray) {
        return Array.isArray(value);
    }
    return objToString.call(value) === '[object Array]';
}

export function isFunction(value: any): value is Function {
    return typeof value === 'function';
}

export function isString(value: any): value is string {
    // Faster than `objToString.call` several times in chromium and webkit.
    // And `new String()` is rarely used.
    return typeof value === 'string';
}

export function isStringSafe(value: any): value is string {
    return objToString.call(value) === '[object String]';
}

export function isNumber(value: any): value is number {
    // Faster than `objToString.call` several times in chromium and webkit.
    // And `new Number()` is rarely used.
    return typeof value === 'number';
}

// Usage: `isObject(xxx)` or `isObject(SomeType)(xxx)`
// Generic T can be used to avoid "ts type gruards" casting the `value` from its original
// type `Object` implicitly so that loose its original type info in the subsequent code.
export function isObject<T = unknown>(value: T): value is (object & T) {
    // Avoid a V8 JIT bug in Chrome 19-20.
    // See https://code.google.com/p/v8/issues/detail?id=2291 for more details.
    const type = typeof value;
    return type === 'function' || (!!value && type === 'object');
}

export function isBuiltInObject(value: any): boolean {
    return !!BUILTIN_OBJECT[objToString.call(value)];
}

export function isTypedArray(value: any): boolean {
    return !!TYPED_ARRAY[objToString.call(value)];
}

export function isDom(value: any): value is HTMLElement {
    return typeof value === 'object'
        && typeof value.nodeType === 'number'
        && typeof value.ownerDocument === 'object';
}

export function isGradientObject(value: any): value is GradientObject {
    return (value as GradientObject).colorStops != null;
}

export function isImagePatternObject(value: any): value is ImagePatternObject {
    return (value as ImagePatternObject).image != null;
}

export function isRegExp(value: unknown): value is RegExp {
    return objToString.call(value) === '[object RegExp]';
}

/**
 * Whether is exactly NaN. Notice isNaN('a') returns true.
 */
export function eqNaN(value: any): boolean {
    /* eslint-disable-next-line no-self-compare */
    return value !== value;
}

/**
 * If value1 is not null, then return value1, otherwise judget rest of values.
 * Low performance.
 * @return Final value
 */
export function retrieve<T>(...args: T[]): T {
    for (let i = 0, len = args.length; i < len; i++) {
        if (args[i] != null) {
            return args[i];
        }
    }
}

export function retrieve2<T, R>(value0: T, value1: R): T | R {
    return value0 != null
        ? value0
        : value1;
}

export function retrieve3<T, R, W>(value0: T, value1: R, value2: W): T | R | W {
    return value0 != null
        ? value0
        : value1 != null
        ? value1
        : value2;
}

type SliceParams = Parameters<typeof nativeSlice>;
export function slice<T>(arr: ArrayLike<T>, ...args: SliceParams): T[] {
    return nativeSlice.apply(arr, args as any[]);
}

/**
 * Normalize css liked array configuration
 * e.g.
 *  3 => [3, 3, 3, 3]
 *  [4, 2] => [4, 2, 4, 2]
 *  [4, 3, 2] => [4, 3, 2, 3]
 */
export function normalizeCssArray(val: number | number[]) {
    if (typeof (val) === 'number') {
        return [val, val, val, val];
    }
    const len = val.length;
    if (len === 2) {
        // vertical | horizontal
        return [val[0], val[1], val[0], val[1]];
    }
    else if (len === 3) {
        // top | horizontal | bottom
        return [val[0], val[1], val[2], val[1]];
    }
    return val;
}

export function assert(condition: any, message?: string) {
    if (!condition) {
        throw new Error(message);
    }
}

/**
 * @param str string to be trimmed
 * @return trimmed string
 */
export function trim(str: string): string {
    if (str == null) {
        return null;
    }
    else if (typeof str.trim === 'function') {
        return str.trim();
    }
    else {
        return str.replace(/^[\s\uFEFF\xA0]+|[\s\uFEFF\xA0]+$/g, '');
    }
}

const primitiveKey = '__ec_primitive__';
/**
 * Set an object as primitive to be ignored traversing children in clone or merge
 */
export function setAsPrimitive(obj: any) {
    obj[primitiveKey] = true;
}

export function isPrimitive(obj: any): boolean {
    return obj[primitiveKey];
}

interface MapInterface<T, KEY extends string | number = string | number> {
    delete(key: KEY): boolean;
    has(key: KEY): boolean;
    get(key: KEY): T | undefined;
    set(key: KEY, value: T): this;
    keys(): KEY[];
    forEach(callback: (value: T, key: KEY) => void): void;
}

class MapPolyfill<T, KEY extends string | number = string | number> implements MapInterface<T, KEY> {
    private data: Record<KEY, T> = {} as Record<KEY, T>;

    delete(key: KEY): boolean {
        const existed = this.has(key);
        if (existed) {
            delete this.data[key];
        }
        return existed;
    }
    has(key: KEY): boolean {
        return this.data.hasOwnProperty(key);
    }
    get(key: KEY): T | undefined {
        return this.data[key];
    }
    set(key: KEY, value: T): this {
        this.data[key] = value;
        return this;
    }
    keys(): KEY[] {
        return keys(this.data);
    }
    forEach(callback: (value: T, key: KEY) => void): void {
        // This is a potential performance bottleneck, see details in
        // https://github.com/ecomfe/zrender/issues/965, however it is now
        // less likely to occur as we default to native maps when possible.
        const data = this.data;
        for (const key in data) {
            if (data.hasOwnProperty(key)) {
                callback(data[key], key);
            }
        }
    }
}

// We want to use native Map if it is available, but we do not want to polyfill the global scope
// in case users ship their own polyfills or patch the native map object in any way.
const isNativeMapSupported = typeof Map === 'function';
function maybeNativeMap<T, KEY extends string | number = string | number>(): MapInterface<T, KEY> {
    // Map may be a native class if we are running in an ES6 compatible environment.
    // eslint-disable-next-line
    return (isNativeMapSupported ? new Map<KEY, T>() : new MapPolyfill<T, KEY>()) as MapInterface<T, KEY>;
}

/**
 * @constructor
 * @param {Object} obj
 */
export class HashMap<T, KEY extends string | number = string | number> {
    data: MapInterface<T, KEY>

    constructor(obj?: HashMap<T, KEY> | { [key in KEY]?: T } | KEY[]) {
        const isArr = isArray(obj);
        // Key should not be set on this, otherwise
        // methods get/set/... may be overridden.
        this.data = maybeNativeMap<T, KEY>();
        const thisMap = this;

        (obj instanceof HashMap)
            ? obj.each(visit)
            : (obj && each(obj, visit));

        function visit(value: any, key: any) {
            isArr ? thisMap.set(value, key) : thisMap.set(key, value);
        }
    }

    // `hasKey` instead of `has` for potential misleading.
    hasKey(key: KEY): boolean {
        return this.data.has(key);
    }
    get(key: KEY): T {
        return this.data.get(key);
    }
    set(key: KEY, value: T): T {
        // Comparing with invocation chaining, `return value` is more commonly
        // used in this case: `const someVal = map.set('a', genVal());`
        this.data.set(key, value);
        return value;
    }
    // Although util.each can be performed on this hashMap directly, user
    // should not use the exposed keys, who are prefixed.
    each<Context>(
        cb: (this: Context, value?: T, key?: KEY) => void,
        context?: Context
    ) {
        this.data.forEach((value, key) => {
            cb.call(context, value, key);
        });
    }
    keys(): KEY[] {
        const keys = this.data.keys();
        return isNativeMapSupported
            // Native map returns an iterator so we need to convert it to an array
            ? Array.from(keys)
            : keys;
    }
    // Do not use this method if performance sensitive.
    removeKey(key: KEY): void {
        this.data.delete(key);
    }
}

export function createHashMap<T, KEY extends string | number = string | number>(
    obj?: HashMap<T, KEY> | { [key in KEY]?: T } | KEY[]
) {
    return new HashMap<T, KEY>(obj);
}

export function concatArray<T, R>(a: ArrayLike<T>, b: ArrayLike<R>): ArrayLike<T | R> {
    const newArray = new (a as any).constructor(a.length + b.length);
    for (let i = 0; i < a.length; i++) {
        newArray[i] = a[i];
    }
    const offset = a.length;
    for (let i = 0; i < b.length; i++) {
        newArray[i + offset] = b[i];
    }
    return newArray;
}

export function createObject<T>(proto?: object, properties?: T): T {
    // Performance of Object.create
    // https://jsperf.com/style-strategy-proto-or-others
    let obj: T;
    if (Object.create) {
        obj = Object.create(proto);
    }
    else {
        const StyleCtor = function () {};
        StyleCtor.prototype = proto;
        obj = new (StyleCtor as any)();
    }
    if (properties) {
        extend(obj, properties);
    }

    return obj;
}


export function disableUserSelect(dom: HTMLElement) {
    const domStyle = dom.style;
    domStyle.webkitUserSelect = 'none';
    domStyle.userSelect = 'none';
    // @ts-ignore
    domStyle.webkitTapHighlightColor = 'rgba(0,0,0,0)';
    (domStyle as any)['-webkit-touch-callout'] = 'none';
}

export function hasOwn(own: object, prop: string): boolean {
    return own.hasOwnProperty(prop);
}

export function noop() {}

export const RADIAN_TO_DEGREE = 180 / Math.PI;