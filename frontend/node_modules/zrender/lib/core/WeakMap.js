var wmUniqueIndex = Math.round(Math.random() * 9);
var supportDefineProperty = typeof Object.defineProperty === 'function';
var WeakMap = (function () {
    function WeakMap() {
        this._id = '__ec_inner_' + wmUniqueIndex++;
    }
    WeakMap.prototype.get = function (key) {
        return this._guard(key)[this._id];
    };
    WeakMap.prototype.set = function (key, value) {
        var target = this._guard(key);
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
    };
    WeakMap.prototype["delete"] = function (key) {
        if (this.has(key)) {
            delete this._guard(key)[this._id];
            return true;
        }
        return false;
    };
    WeakMap.prototype.has = function (key) {
        return !!this._guard(key)[this._id];
    };
    WeakMap.prototype._guard = function (key) {
        if (key !== Object(key)) {
            throw TypeError('Value of WeakMap is not a non-null object.');
        }
        return key;
    };
    return WeakMap;
}());
export default WeakMap;
