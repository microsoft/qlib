import { __extends } from "tslib";
import * as zrUtil from '../core/util.js';
import Element from '../Element.js';
import BoundingRect from '../core/BoundingRect.js';
var Group = (function (_super) {
    __extends(Group, _super);
    function Group(opts) {
        var _this = _super.call(this) || this;
        _this.isGroup = true;
        _this._children = [];
        _this.attr(opts);
        return _this;
    }
    Group.prototype.childrenRef = function () {
        return this._children;
    };
    Group.prototype.children = function () {
        return this._children.slice();
    };
    Group.prototype.childAt = function (idx) {
        return this._children[idx];
    };
    Group.prototype.childOfName = function (name) {
        var children = this._children;
        for (var i = 0; i < children.length; i++) {
            if (children[i].name === name) {
                return children[i];
            }
        }
    };
    Group.prototype.childCount = function () {
        return this._children.length;
    };
    Group.prototype.add = function (child) {
        if (child) {
            if (child !== this && child.parent !== this) {
                this._children.push(child);
                this._doAdd(child);
            }
            if (process.env.NODE_ENV !== 'production') {
                if (child.__hostTarget) {
                    throw 'This elemenet has been used as an attachment';
                }
            }
        }
        return this;
    };
    Group.prototype.addBefore = function (child, nextSibling) {
        if (child && child !== this && child.parent !== this
            && nextSibling && nextSibling.parent === this) {
            var children = this._children;
            var idx = children.indexOf(nextSibling);
            if (idx >= 0) {
                children.splice(idx, 0, child);
                this._doAdd(child);
            }
        }
        return this;
    };
    Group.prototype.replace = function (oldChild, newChild) {
        var idx = zrUtil.indexOf(this._children, oldChild);
        if (idx >= 0) {
            this.replaceAt(newChild, idx);
        }
        return this;
    };
    Group.prototype.replaceAt = function (child, index) {
        var children = this._children;
        var old = children[index];
        if (child && child !== this && child.parent !== this && child !== old) {
            children[index] = child;
            old.parent = null;
            var zr = this.__zr;
            if (zr) {
                old.removeSelfFromZr(zr);
            }
            this._doAdd(child);
        }
        return this;
    };
    Group.prototype._doAdd = function (child) {
        if (child.parent) {
            child.parent.remove(child);
        }
        child.parent = this;
        var zr = this.__zr;
        if (zr && zr !== child.__zr) {
            child.addSelfToZr(zr);
        }
        zr && zr.refresh();
    };
    Group.prototype.remove = function (child) {
        var zr = this.__zr;
        var children = this._children;
        var idx = zrUtil.indexOf(children, child);
        if (idx < 0) {
            return this;
        }
        children.splice(idx, 1);
        child.parent = null;
        if (zr) {
            child.removeSelfFromZr(zr);
        }
        zr && zr.refresh();
        return this;
    };
    Group.prototype.removeAll = function () {
        var children = this._children;
        var zr = this.__zr;
        for (var i = 0; i < children.length; i++) {
            var child = children[i];
            if (zr) {
                child.removeSelfFromZr(zr);
            }
            child.parent = null;
        }
        children.length = 0;
        return this;
    };
    Group.prototype.eachChild = function (cb, context) {
        var children = this._children;
        for (var i = 0; i < children.length; i++) {
            var child = children[i];
            cb.call(context, child, i);
        }
        return this;
    };
    Group.prototype.traverse = function (cb, context) {
        for (var i = 0; i < this._children.length; i++) {
            var child = this._children[i];
            var stopped = cb.call(context, child);
            if (child.isGroup && !stopped) {
                child.traverse(cb, context);
            }
        }
        return this;
    };
    Group.prototype.addSelfToZr = function (zr) {
        _super.prototype.addSelfToZr.call(this, zr);
        for (var i = 0; i < this._children.length; i++) {
            var child = this._children[i];
            child.addSelfToZr(zr);
        }
    };
    Group.prototype.removeSelfFromZr = function (zr) {
        _super.prototype.removeSelfFromZr.call(this, zr);
        for (var i = 0; i < this._children.length; i++) {
            var child = this._children[i];
            child.removeSelfFromZr(zr);
        }
    };
    Group.prototype.getBoundingRect = function (includeChildren) {
        var tmpRect = new BoundingRect(0, 0, 0, 0);
        var children = includeChildren || this._children;
        var tmpMat = [];
        var rect = null;
        for (var i = 0; i < children.length; i++) {
            var child = children[i];
            if (child.ignore || child.invisible) {
                continue;
            }
            var childRect = child.getBoundingRect();
            var transform = child.getLocalTransform(tmpMat);
            if (transform) {
                BoundingRect.applyTransform(tmpRect, childRect, transform);
                rect = rect || tmpRect.clone();
                rect.union(tmpRect);
            }
            else {
                rect = rect || childRect.clone();
                rect.union(childRect);
            }
        }
        return rect || tmpRect;
    };
    return Group;
}(Element));
Group.prototype.type = 'group';
export default Group;
