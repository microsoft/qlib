import { __extends } from "tslib";
import Displayble from './Displayable.js';
import BoundingRect from '../core/BoundingRect.js';
var m = [];
var IncrementalDisplayable = (function (_super) {
    __extends(IncrementalDisplayable, _super);
    function IncrementalDisplayable() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.notClear = true;
        _this.incremental = true;
        _this._displayables = [];
        _this._temporaryDisplayables = [];
        _this._cursor = 0;
        return _this;
    }
    IncrementalDisplayable.prototype.traverse = function (cb, context) {
        cb.call(context, this);
    };
    IncrementalDisplayable.prototype.useStyle = function () {
        this.style = {};
    };
    IncrementalDisplayable.prototype.getCursor = function () {
        return this._cursor;
    };
    IncrementalDisplayable.prototype.innerAfterBrush = function () {
        this._cursor = this._displayables.length;
    };
    IncrementalDisplayable.prototype.clearDisplaybles = function () {
        this._displayables = [];
        this._temporaryDisplayables = [];
        this._cursor = 0;
        this.markRedraw();
        this.notClear = false;
    };
    IncrementalDisplayable.prototype.clearTemporalDisplayables = function () {
        this._temporaryDisplayables = [];
    };
    IncrementalDisplayable.prototype.addDisplayable = function (displayable, notPersistent) {
        if (notPersistent) {
            this._temporaryDisplayables.push(displayable);
        }
        else {
            this._displayables.push(displayable);
        }
        this.markRedraw();
    };
    IncrementalDisplayable.prototype.addDisplayables = function (displayables, notPersistent) {
        notPersistent = notPersistent || false;
        for (var i = 0; i < displayables.length; i++) {
            this.addDisplayable(displayables[i], notPersistent);
        }
    };
    IncrementalDisplayable.prototype.getDisplayables = function () {
        return this._displayables;
    };
    IncrementalDisplayable.prototype.getTemporalDisplayables = function () {
        return this._temporaryDisplayables;
    };
    IncrementalDisplayable.prototype.eachPendingDisplayable = function (cb) {
        for (var i = this._cursor; i < this._displayables.length; i++) {
            cb && cb(this._displayables[i]);
        }
        for (var i = 0; i < this._temporaryDisplayables.length; i++) {
            cb && cb(this._temporaryDisplayables[i]);
        }
    };
    IncrementalDisplayable.prototype.update = function () {
        this.updateTransform();
        for (var i = this._cursor; i < this._displayables.length; i++) {
            var displayable = this._displayables[i];
            displayable.parent = this;
            displayable.update();
            displayable.parent = null;
        }
        for (var i = 0; i < this._temporaryDisplayables.length; i++) {
            var displayable = this._temporaryDisplayables[i];
            displayable.parent = this;
            displayable.update();
            displayable.parent = null;
        }
    };
    IncrementalDisplayable.prototype.getBoundingRect = function () {
        if (!this._rect) {
            var rect = new BoundingRect(Infinity, Infinity, -Infinity, -Infinity);
            for (var i = 0; i < this._displayables.length; i++) {
                var displayable = this._displayables[i];
                var childRect = displayable.getBoundingRect().clone();
                if (displayable.needLocalTransform()) {
                    childRect.applyTransform(displayable.getLocalTransform(m));
                }
                rect.union(childRect);
            }
            this._rect = rect;
        }
        return this._rect;
    };
    IncrementalDisplayable.prototype.contain = function (x, y) {
        var localPos = this.transformCoordToLocal(x, y);
        var rect = this.getBoundingRect();
        if (rect.contain(localPos[0], localPos[1])) {
            for (var i = 0; i < this._displayables.length; i++) {
                var displayable = this._displayables[i];
                if (displayable.contain(x, y)) {
                    return true;
                }
            }
        }
        return false;
    };
    return IncrementalDisplayable;
}(Displayble));
export default IncrementalDisplayable;
