import { __extends } from "tslib";
import Definable from './Definable.js';
import * as zrUtil from '../../core/util.js';
import { path } from '../graphic.js';
import { isClipPathChanged } from '../../canvas/helper.js';
import { getClipPathsKey, getIdURL } from '../../svg/helper.js';
import { createElement } from '../../svg/core.js';
export function hasClipPath(displayable) {
    var clipPaths = displayable.__clipPaths;
    return clipPaths && clipPaths.length > 0;
}
var ClippathManager = (function (_super) {
    __extends(ClippathManager, _super);
    function ClippathManager(zrId, svgRoot) {
        var _this = _super.call(this, zrId, svgRoot, 'clipPath', '__clippath_in_use__') || this;
        _this._refGroups = {};
        _this._keyDuplicateCount = {};
        return _this;
    }
    ClippathManager.prototype.markAllUnused = function () {
        _super.prototype.markAllUnused.call(this);
        var refGroups = this._refGroups;
        for (var key in refGroups) {
            if (refGroups.hasOwnProperty(key)) {
                this.markDomUnused(refGroups[key]);
            }
        }
        this._keyDuplicateCount = {};
    };
    ClippathManager.prototype._getClipPathGroup = function (displayable, prevDisplayable) {
        if (!hasClipPath(displayable)) {
            return;
        }
        var clipPaths = displayable.__clipPaths;
        var keyDuplicateCount = this._keyDuplicateCount;
        var clipPathKey = getClipPathsKey(clipPaths);
        if (isClipPathChanged(clipPaths, prevDisplayable && prevDisplayable.__clipPaths)) {
            keyDuplicateCount[clipPathKey] = keyDuplicateCount[clipPathKey] || 0;
            keyDuplicateCount[clipPathKey] && (clipPathKey += '-' + keyDuplicateCount[clipPathKey]);
            keyDuplicateCount[clipPathKey]++;
        }
        return this._refGroups[clipPathKey]
            || (this._refGroups[clipPathKey] = createElement('g'));
    };
    ClippathManager.prototype.update = function (displayable, prevDisplayable) {
        var clipGroup = this._getClipPathGroup(displayable, prevDisplayable);
        if (clipGroup) {
            this.markDomUsed(clipGroup);
            this.updateDom(clipGroup, displayable.__clipPaths);
        }
        return clipGroup;
    };
    ;
    ClippathManager.prototype.updateDom = function (parentEl, clipPaths) {
        if (clipPaths && clipPaths.length > 0) {
            var defs = this.getDefs(true);
            var clipPath = clipPaths[0];
            var clipPathEl = void 0;
            var id = void 0;
            if (clipPath._dom) {
                id = clipPath._dom.getAttribute('id');
                clipPathEl = clipPath._dom;
                if (!defs.contains(clipPathEl)) {
                    defs.appendChild(clipPathEl);
                }
            }
            else {
                id = 'zr' + this._zrId + '-clip-' + this.nextId;
                ++this.nextId;
                clipPathEl = createElement('clipPath');
                clipPathEl.setAttribute('id', id);
                defs.appendChild(clipPathEl);
                clipPath._dom = clipPathEl;
            }
            path.brush(clipPath);
            var pathEl = this.getSvgElement(clipPath);
            clipPathEl.innerHTML = '';
            clipPathEl.appendChild(pathEl);
            parentEl.setAttribute('clip-path', getIdURL(id));
            if (clipPaths.length > 1) {
                this.updateDom(clipPathEl, clipPaths.slice(1));
            }
        }
        else {
            if (parentEl) {
                parentEl.setAttribute('clip-path', 'none');
            }
        }
    };
    ;
    ClippathManager.prototype.markUsed = function (displayable) {
        var _this = this;
        if (displayable.__clipPaths) {
            zrUtil.each(displayable.__clipPaths, function (clipPath) {
                if (clipPath._dom) {
                    _super.prototype.markDomUsed.call(_this, clipPath._dom);
                }
            });
        }
    };
    ;
    ClippathManager.prototype.removeUnused = function () {
        _super.prototype.removeUnused.call(this);
        var newRefGroupsMap = {};
        var refGroups = this._refGroups;
        for (var key in refGroups) {
            if (refGroups.hasOwnProperty(key)) {
                var group = refGroups[key];
                if (!this.isDomUnused(group)) {
                    newRefGroupsMap[key] = group;
                }
                else if (group.parentNode) {
                    group.parentNode.removeChild(group);
                }
            }
        }
        this._refGroups = newRefGroupsMap;
    };
    return ClippathManager;
}(Definable));
export default ClippathManager;
