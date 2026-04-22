export default class WeakMap<K extends object, V> {
    protected _id: string;
    constructor();
    get(key: K): V;
    set(key: K, value: V): WeakMap<K, V>;
    delete(key: K): boolean;
    has(key: K): boolean;
    protected _guard(key: K): K;
}
