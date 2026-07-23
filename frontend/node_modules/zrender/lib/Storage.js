import * as util from './core/util.js';
import timsort from './core/timsort.js';
import { REDRAW_BIT } from './graphic/constants.js';
var invalidZErrorLogged = false;
function logInvalidZError() {
    if (invalidZErrorLogged) {
        return;
    }
    invalidZErrorLogged = true;
    console.warn('z / z2 / zlevel of displayable is invalid, which may cause unexpected errors');
}
function shapeCompareFunc(a, b) {
    if (a.zlevel === b.zlevel) {
        if (a.z === b.z) {
            return a.z2 - b.z2;
        }
        return a.z - b.z;
    }
    return a.zlevel - b.zlevel;
}
var Storage = (function () {
    function Storage() {
        this._roots = [];
        this._displayList = [];
        this._displayListLen = 0;
        this.displayableSortFunc = shapeCompareFunc;
    }
    Storage.prototype.traverse = function (cb, context) {
        for (var i = 0; i < this._roots.length; i++) {
            this._roots[i].traverse(cb, context);
        }
    };
    Storage.prototype.getDisplayList = function (update, includeIgnore) {
        includeIgnore = includeIgnore || false;
        var displayList = this._displayList;
        if (update || !displayList.length) {
            this.updateDisplayList(includeIgnore);
        }
        return displayList;
    };
    Storage.prototype.updateDisplayList = function (includeIgnore) {
        this._displayListLen = 0;
        var roots = this._roots;
        var displayList = this._displayList;
        for (var i = 0, len = roots.length; i < len; i++) {
            this._updateAndAddDisplayable(roots[i], null, includeIgnore);
        }
        displayList.length = this._displayListLen;
        timsort(displayList, shapeCompareFunc);
    };
    Storage.prototype._updateAndAddDisplayable = function (el, clipPaths, includeIgnore) {
        if (el.ignore && !includeIgnore) {
            return;
        }
        el.beforeUpdate();
        el.update();
        el.afterUpdate();
        var userSetClipPath = el.getClipPath();
        if (el.ignoreClip) {
            clipPaths = null;
        }
        else if (userSetClipPath) {
            if (clipPaths) {
                clipPaths = clipPaths.slice();
            }
            else {
                clipPaths = [];
            }
            var currentClipPath = userSetClipPath;
            var parentClipPath = el;
            while (currentClipPath) {
                currentClipPath.parent = parentClipPath;
                currentClipPath.updateTransform();
                clipPaths.push(currentClipPath);
                parentClipPath = currentClipPath;
                currentClipPath = currentClipPath.getClipPath();
            }
        }
        if (el.childrenRef) {
            var children = el.childrenRef();
            for (var i = 0; i < children.length; i++) {
                var child = children[i];
                if (el.__dirty) {
                    child.__dirty |= REDRAW_BIT;
                }
                this._updateAndAddDisplayable(child, clipPaths, includeIgnore);
            }
            el.__dirty = 0;
        }
        else {
            var disp = el;
            if (clipPaths && clipPaths.length) {
                disp.__clipPaths = clipPaths;
            }
            else if (disp.__clipPaths && disp.__clipPaths.length > 0) {
                disp.__clipPaths = [];
            }
            if (isNaN(disp.z)) {
                logInvalidZError();
                disp.z = 0;
            }
            if (isNaN(disp.z2)) {
                logInvalidZError();
                disp.z2 = 0;
            }
            if (isNaN(disp.zlevel)) {
                logInvalidZError();
                disp.zlevel = 0;
            }
            this._displayList[this._displayListLen++] = disp;
        }
        var decalEl = el.getDecalElement && el.getDecalElement();
        if (decalEl) {
            this._updateAndAddDisplayable(decalEl, clipPaths, includeIgnore);
        }
        var textGuide = el.getTextGuideLine();
        if (textGuide) {
            this._updateAndAddDisplayable(textGuide, clipPaths, includeIgnore);
        }
        var textEl = el.getTextContent();
        if (textEl) {
            this._updateAndAddDisplayable(textEl, clipPaths, includeIgnore);
        }
    };
    Storage.prototype.addRoot = function (el) {
        if (el.__zr && el.__zr.storage === this) {
            return;
        }
        this._roots.push(el);
    };
    Storage.prototype.delRoot = function (el) {
        if (el instanceof Array) {
            for (var i = 0, l = el.length; i < l; i++) {
                this.delRoot(el[i]);
            }
            return;
        }
        var idx = util.indexOf(this._roots, el);
        if (idx >= 0) {
            this._roots.splice(idx, 1);
        }
    };
    Storage.prototype.delAllRoots = function () {
        this._roots = [];
        this._displayList = [];
        this._displayListLen = 0;
        return;
    };
    Storage.prototype.getRoots = function () {
        return this._roots;
    };
    Storage.prototype.dispose = function () {
        this._displayList = null;
        this._roots = null;
    };
    return Storage;
}());
export default Storage;
