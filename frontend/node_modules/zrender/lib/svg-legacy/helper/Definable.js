import { createElement } from '../../svg/core.js';
import * as zrUtil from '../../core/util.js';
var MARK_UNUSED = '0';
var MARK_USED = '1';
var Definable = (function () {
    function Definable(zrId, svgRoot, tagNames, markLabel, domName) {
        this.nextId = 0;
        this._domName = '_dom';
        this._zrId = zrId;
        this._svgRoot = svgRoot;
        this._tagNames = typeof tagNames === 'string' ? [tagNames] : tagNames;
        this._markLabel = markLabel;
        if (domName) {
            this._domName = domName;
        }
    }
    Definable.prototype.getDefs = function (isForceCreating) {
        var svgRoot = this._svgRoot;
        var defs = this._svgRoot.getElementsByTagName('defs');
        if (defs.length === 0) {
            if (isForceCreating) {
                var defs_1 = svgRoot.insertBefore(createElement('defs'), svgRoot.firstChild);
                if (!defs_1.contains) {
                    defs_1.contains = function (el) {
                        var children = defs_1.children;
                        if (!children) {
                            return false;
                        }
                        for (var i = children.length - 1; i >= 0; --i) {
                            if (children[i] === el) {
                                return true;
                            }
                        }
                        return false;
                    };
                }
                return defs_1;
            }
            else {
                return null;
            }
        }
        else {
            return defs[0];
        }
    };
    Definable.prototype.doUpdate = function (target, onUpdate) {
        if (!target) {
            return;
        }
        var defs = this.getDefs(false);
        if (target[this._domName] && defs.contains(target[this._domName])) {
            if (typeof onUpdate === 'function') {
                onUpdate(target);
            }
        }
        else {
            var dom = this.add(target);
            if (dom) {
                target[this._domName] = dom;
            }
        }
    };
    Definable.prototype.add = function (target) {
        return null;
    };
    Definable.prototype.addDom = function (dom) {
        var defs = this.getDefs(true);
        if (dom.parentNode !== defs) {
            defs.appendChild(dom);
        }
    };
    Definable.prototype.removeDom = function (target) {
        var defs = this.getDefs(false);
        if (defs && target[this._domName]) {
            defs.removeChild(target[this._domName]);
            target[this._domName] = null;
        }
    };
    Definable.prototype.getDoms = function () {
        var defs = this.getDefs(false);
        if (!defs) {
            return [];
        }
        var doms = [];
        zrUtil.each(this._tagNames, function (tagName) {
            var tags = defs.getElementsByTagName(tagName);
            for (var i = 0; i < tags.length; i++) {
                doms.push(tags[i]);
            }
        });
        return doms;
    };
    Definable.prototype.markAllUnused = function () {
        var doms = this.getDoms();
        var that = this;
        zrUtil.each(doms, function (dom) {
            dom[that._markLabel] = MARK_UNUSED;
        });
    };
    Definable.prototype.markDomUsed = function (dom) {
        dom && (dom[this._markLabel] = MARK_USED);
    };
    ;
    Definable.prototype.markDomUnused = function (dom) {
        dom && (dom[this._markLabel] = MARK_UNUSED);
    };
    ;
    Definable.prototype.isDomUnused = function (dom) {
        return dom && dom[this._markLabel] !== MARK_USED;
    };
    Definable.prototype.removeUnused = function () {
        var _this = this;
        var defs = this.getDefs(false);
        if (!defs) {
            return;
        }
        var doms = this.getDoms();
        zrUtil.each(doms, function (dom) {
            if (_this.isDomUnused(dom)) {
                defs.removeChild(dom);
            }
        });
    };
    Definable.prototype.getSvgElement = function (displayable) {
        return displayable.__svgEl;
    };
    return Definable;
}());
export default Definable;
