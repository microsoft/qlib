let wmUniqueIndex = Math.round(Math.random() * 9);

const supportDefineProperty = typeof Object.defineProperty === 'function';

export default class WeakMap<K extends object, V> {

    protected _id: string;

    constructor() {
        this._id = '__ec_inner_' + wmUniqueIndex++;
    }

    get(key: K): V {
        return (this._guard(key) as any)[this._id];
    }

    set(key: K, value: V): WeakMap<K, V> {
        const target = this._guard(key) as any;
        if (supportDefineProperty) {
            Object.defineProperty(target, this._id, {
                value: value,
                enumerable: false,
                configurable: true
            });
        }
        else {
            target[this._id] = value;
        }
        return this;
    }

    delete(key: K): boolean {
        if (this.has(key)) {
            delete (this._guard(key) as any)[this._id];
            return true;
        }
        return false;
    }

    has(key: K): boolean {
        return !!(this._guard(key) as any)[this._id];
    }

    protected _guard(key: K): K {
        if (key !== Object(key)) {
            throw TypeError('Value of WeakMap is not a non-null object.');
        }
        return key;
    }
}
