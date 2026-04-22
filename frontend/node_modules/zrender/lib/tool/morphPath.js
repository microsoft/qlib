import { cubicSubdivide } from '../core/curve.js';
import Path from '../graphic/Path.js';
import { defaults, map } from '../core/util.js';
import { lerp } from '../core/vector.js';
import { clonePath } from './path.js';
import Transformable from '../core/Transformable.js';
import { split } from './dividePath.js';
import { pathToBezierCurves } from './convertPath.js';
function alignSubpath(subpath1, subpath2) {
    var len1 = subpath1.length;
    var len2 = subpath2.length;
    if (len1 === len2) {
        return [subpath1, subpath2];
    }
    var tmpSegX = [];
    var tmpSegY = [];
    var shorterPath = len1 < len2 ? subpath1 : subpath2;
    var shorterLen = Math.min(len1, len2);
    var diff = Math.abs(len2 - len1) / 6;
    var shorterBezierCount = (shorterLen - 2) / 6;
    var eachCurveSubDivCount = Math.ceil(diff / shorterBezierCount) + 1;
    var newSubpath = [shorterPath[0], shorterPath[1]];
    var remained = diff;
    for (var i = 2; i < shorterLen;) {
        var x0 = shorterPath[i - 2];
        var y0 = shorterPath[i - 1];
        var x1 = shorterPath[i++];
        var y1 = shorterPath[i++];
        var x2 = shorterPath[i++];
        var y2 = shorterPath[i++];
        var x3 = shorterPath[i++];
        var y3 = shorterPath[i++];
        if (remained <= 0) {
            newSubpath.push(x1, y1, x2, y2, x3, y3);
            continue;
        }
        var actualSubDivCount = Math.min(remained, eachCurveSubDivCount - 1) + 1;
        for (var k = 1; k <= actualSubDivCount; k++) {
            var p = k / actualSubDivCount;
            cubicSubdivide(x0, x1, x2, x3, p, tmpSegX);
            cubicSubdivide(y0, y1, y2, y3, p, tmpSegY);
            x0 = tmpSegX[3];
            y0 = tmpSegY[3];
            newSubpath.push(tmpSegX[1], tmpSegY[1], tmpSegX[2], tmpSegY[2], x0, y0);
            x1 = tmpSegX[5];
            y1 = tmpSegY[5];
            x2 = tmpSegX[6];
            y2 = tmpSegY[6];
        }
        remained -= actualSubDivCount - 1;
    }
    return shorterPath === subpath1 ? [newSubpath, subpath2] : [subpath1, newSubpath];
}
function createSubpath(lastSubpathSubpath, otherSubpath) {
    var len = lastSubpathSubpath.length;
    var lastX = lastSubpathSubpath[len - 2];
    var lastY = lastSubpathSubpath[len - 1];
    var newSubpath = [];
    for (var i = 0; i < otherSubpath.length;) {
        newSubpath[i++] = lastX;
        newSubpath[i++] = lastY;
    }
    return newSubpath;
}
export function alignBezierCurves(array1, array2) {
    var _a;
    var lastSubpath1;
    var lastSubpath2;
    var newArray1 = [];
    var newArray2 = [];
    for (var i = 0; i < Math.max(array1.length, array2.length); i++) {
        var subpath1 = array1[i];
        var subpath2 = array2[i];
        var newSubpath1 = void 0;
        var newSubpath2 = void 0;
        if (!subpath1) {
            newSubpath1 = createSubpath(lastSubpath1 || subpath2, subpath2);
            newSubpath2 = subpath2;
        }
        else if (!subpath2) {
            newSubpath2 = createSubpath(lastSubpath2 || subpath1, subpath1);
            newSubpath1 = subpath1;
        }
        else {
            _a = alignSubpath(subpath1, subpath2), newSubpath1 = _a[0], newSubpath2 = _a[1];
            lastSubpath1 = newSubpath1;
            lastSubpath2 = newSubpath2;
        }
        newArray1.push(newSubpath1);
        newArray2.push(newSubpath2);
    }
    return [newArray1, newArray2];
}
export function centroid(array) {
    var signedArea = 0;
    var cx = 0;
    var cy = 0;
    var len = array.length;
    for (var i = 0, j = len - 2; i < len; j = i, i += 2) {
        var x0 = array[j];
        var y0 = array[j + 1];
        var x1 = array[i];
        var y1 = array[i + 1];
        var a = x0 * y1 - x1 * y0;
        signedArea += a;
        cx += (x0 + x1) * a;
        cy += (y0 + y1) * a;
    }
    if (signedArea === 0) {
        return [array[0] || 0, array[1] || 0];
    }
    return [cx / signedArea / 3, cy / signedArea / 3, signedArea];
}
function findBestRingOffset(fromSubBeziers, toSubBeziers, fromCp, toCp) {
    var bezierCount = (fromSubBeziers.length - 2) / 6;
    var bestScore = Infinity;
    var bestOffset = 0;
    var len = fromSubBeziers.length;
    var len2 = len - 2;
    for (var offset = 0; offset < bezierCount; offset++) {
        var cursorOffset = offset * 6;
        var score = 0;
        for (var k = 0; k < len; k += 2) {
            var idx = k === 0 ? cursorOffset : ((cursorOffset + k - 2) % len2 + 2);
            var x0 = fromSubBeziers[idx] - fromCp[0];
            var y0 = fromSubBeziers[idx + 1] - fromCp[1];
            var x1 = toSubBeziers[k] - toCp[0];
            var y1 = toSubBeziers[k + 1] - toCp[1];
            var dx = x1 - x0;
            var dy = y1 - y0;
            score += dx * dx + dy * dy;
        }
        if (score < bestScore) {
            bestScore = score;
            bestOffset = offset;
        }
    }
    return bestOffset;
}
function reverse(array) {
    var newArr = [];
    var len = array.length;
    for (var i = 0; i < len; i += 2) {
        newArr[i] = array[len - i - 2];
        newArr[i + 1] = array[len - i - 1];
    }
    return newArr;
}
function findBestMorphingRotation(fromArr, toArr, searchAngleIteration, searchAngleRange) {
    var result = [];
    var fromNeedsReverse;
    for (var i = 0; i < fromArr.length; i++) {
        var fromSubpathBezier = fromArr[i];
        var toSubpathBezier = toArr[i];
        var fromCp = centroid(fromSubpathBezier);
        var toCp = centroid(toSubpathBezier);
        if (fromNeedsReverse == null) {
            fromNeedsReverse = fromCp[2] < 0 !== toCp[2] < 0;
        }
        var newFromSubpathBezier = [];
        var newToSubpathBezier = [];
        var bestAngle = 0;
        var bestScore = Infinity;
        var tmpArr = [];
        var len = fromSubpathBezier.length;
        if (fromNeedsReverse) {
            fromSubpathBezier = reverse(fromSubpathBezier);
        }
        var offset = findBestRingOffset(fromSubpathBezier, toSubpathBezier, fromCp, toCp) * 6;
        var len2 = len - 2;
        for (var k = 0; k < len2; k += 2) {
            var idx = (offset + k) % len2 + 2;
            newFromSubpathBezier[k + 2] = fromSubpathBezier[idx] - fromCp[0];
            newFromSubpathBezier[k + 3] = fromSubpathBezier[idx + 1] - fromCp[1];
        }
        newFromSubpathBezier[0] = fromSubpathBezier[offset] - fromCp[0];
        newFromSubpathBezier[1] = fromSubpathBezier[offset + 1] - fromCp[1];
        if (searchAngleIteration > 0) {
            var step = searchAngleRange / searchAngleIteration;
            for (var angle = -searchAngleRange / 2; angle <= searchAngleRange / 2; angle += step) {
                var sa = Math.sin(angle);
                var ca = Math.cos(angle);
                var score = 0;
                for (var k = 0; k < fromSubpathBezier.length; k += 2) {
                    var x0 = newFromSubpathBezier[k];
                    var y0 = newFromSubpathBezier[k + 1];
                    var x1 = toSubpathBezier[k] - toCp[0];
                    var y1 = toSubpathBezier[k + 1] - toCp[1];
                    var newX1 = x1 * ca - y1 * sa;
                    var newY1 = x1 * sa + y1 * ca;
                    tmpArr[k] = newX1;
                    tmpArr[k + 1] = newY1;
                    var dx = newX1 - x0;
                    var dy = newY1 - y0;
                    score += dx * dx + dy * dy;
                }
                if (score < bestScore) {
                    bestScore = score;
                    bestAngle = angle;
                    for (var m = 0; m < tmpArr.length; m++) {
                        newToSubpathBezier[m] = tmpArr[m];
                    }
                }
            }
        }
        else {
            for (var i_1 = 0; i_1 < len; i_1 += 2) {
                newToSubpathBezier[i_1] = toSubpathBezier[i_1] - toCp[0];
                newToSubpathBezier[i_1 + 1] = toSubpathBezier[i_1 + 1] - toCp[1];
            }
        }
        result.push({
            from: newFromSubpathBezier,
            to: newToSubpathBezier,
            fromCp: fromCp,
            toCp: toCp,
            rotation: -bestAngle
        });
    }
    return result;
}
export function isCombineMorphing(path) {
    return path.__isCombineMorphing;
}
export function isMorphing(el) {
    return el.__morphT >= 0;
}
var SAVED_METHOD_PREFIX = '__mOriginal_';
function saveAndModifyMethod(obj, methodName, modifiers) {
    var savedMethodName = SAVED_METHOD_PREFIX + methodName;
    var originalMethod = obj[savedMethodName] || obj[methodName];
    if (!obj[savedMethodName]) {
        obj[savedMethodName] = obj[methodName];
    }
    var replace = modifiers.replace;
    var after = modifiers.after;
    var before = modifiers.before;
    obj[methodName] = function () {
        var args = arguments;
        var res;
        before && before.apply(this, args);
        if (replace) {
            res = replace.apply(this, args);
        }
        else {
            res = originalMethod.apply(this, args);
        }
        after && after.apply(this, args);
        return res;
    };
}
function restoreMethod(obj, methodName) {
    var savedMethodName = SAVED_METHOD_PREFIX + methodName;
    if (obj[savedMethodName]) {
        obj[methodName] = obj[savedMethodName];
        obj[savedMethodName] = null;
    }
}
function applyTransformOnBeziers(bezierCurves, mm) {
    for (var i = 0; i < bezierCurves.length; i++) {
        var subBeziers = bezierCurves[i];
        for (var k = 0; k < subBeziers.length;) {
            var x = subBeziers[k];
            var y = subBeziers[k + 1];
            subBeziers[k++] = mm[0] * x + mm[2] * y + mm[4];
            subBeziers[k++] = mm[1] * x + mm[3] * y + mm[5];
        }
    }
}
function prepareMorphPath(fromPath, toPath) {
    var fromPathProxy = fromPath.getUpdatedPathProxy();
    var toPathProxy = toPath.getUpdatedPathProxy();
    var _a = alignBezierCurves(pathToBezierCurves(fromPathProxy), pathToBezierCurves(toPathProxy)), fromBezierCurves = _a[0], toBezierCurves = _a[1];
    var fromPathTransform = fromPath.getComputedTransform();
    var toPathTransform = toPath.getComputedTransform();
    function updateIdentityTransform() {
        this.transform = null;
    }
    fromPathTransform && applyTransformOnBeziers(fromBezierCurves, fromPathTransform);
    toPathTransform && applyTransformOnBeziers(toBezierCurves, toPathTransform);
    saveAndModifyMethod(toPath, 'updateTransform', { replace: updateIdentityTransform });
    toPath.transform = null;
    var morphingData = findBestMorphingRotation(fromBezierCurves, toBezierCurves, 10, Math.PI);
    var tmpArr = [];
    saveAndModifyMethod(toPath, 'buildPath', { replace: function (path) {
            var t = toPath.__morphT;
            var onet = 1 - t;
            var newCp = [];
            for (var i = 0; i < morphingData.length; i++) {
                var item = morphingData[i];
                var from = item.from;
                var to = item.to;
                var angle = item.rotation * t;
                var fromCp = item.fromCp;
                var toCp = item.toCp;
                var sa = Math.sin(angle);
                var ca = Math.cos(angle);
                lerp(newCp, fromCp, toCp, t);
                for (var m = 0; m < from.length; m += 2) {
                    var x0_1 = from[m];
                    var y0_1 = from[m + 1];
                    var x1 = to[m];
                    var y1 = to[m + 1];
                    var x = x0_1 * onet + x1 * t;
                    var y = y0_1 * onet + y1 * t;
                    tmpArr[m] = (x * ca - y * sa) + newCp[0];
                    tmpArr[m + 1] = (x * sa + y * ca) + newCp[1];
                }
                var x0 = tmpArr[0];
                var y0 = tmpArr[1];
                path.moveTo(x0, y0);
                for (var m = 2; m < from.length;) {
                    var x1 = tmpArr[m++];
                    var y1 = tmpArr[m++];
                    var x2 = tmpArr[m++];
                    var y2 = tmpArr[m++];
                    var x3 = tmpArr[m++];
                    var y3 = tmpArr[m++];
                    if (x0 === x1 && y0 === y1 && x2 === x3 && y2 === y3) {
                        path.lineTo(x3, y3);
                    }
                    else {
                        path.bezierCurveTo(x1, y1, x2, y2, x3, y3);
                    }
                    x0 = x3;
                    y0 = y3;
                }
            }
        } });
}
export function morphPath(fromPath, toPath, animationOpts) {
    if (!fromPath || !toPath) {
        return toPath;
    }
    var oldDone = animationOpts.done;
    var oldDuring = animationOpts.during;
    prepareMorphPath(fromPath, toPath);
    toPath.__morphT = 0;
    function restoreToPath() {
        restoreMethod(toPath, 'buildPath');
        restoreMethod(toPath, 'updateTransform');
        toPath.__morphT = -1;
        toPath.createPathProxy();
        toPath.dirtyShape();
    }
    toPath.animateTo({
        __morphT: 1
    }, defaults({
        during: function (p) {
            toPath.dirtyShape();
            oldDuring && oldDuring(p);
        },
        done: function () {
            restoreToPath();
            oldDone && oldDone();
        }
    }, animationOpts));
    return toPath;
}
function hilbert(x, y, minX, minY, maxX, maxY) {
    var bits = 16;
    x = (maxX === minX) ? 0 : Math.round(32767 * (x - minX) / (maxX - minX));
    y = (maxY === minY) ? 0 : Math.round(32767 * (y - minY) / (maxY - minY));
    var d = 0;
    var tmp;
    for (var s = (1 << bits) / 2; s > 0; s /= 2) {
        var rx = 0;
        var ry = 0;
        if ((x & s) > 0) {
            rx = 1;
        }
        if ((y & s) > 0) {
            ry = 1;
        }
        d += s * s * ((3 * rx) ^ ry);
        if (ry === 0) {
            if (rx === 1) {
                x = s - 1 - x;
                y = s - 1 - y;
            }
            tmp = x;
            x = y;
            y = tmp;
        }
    }
    return d;
}
function sortPaths(pathList) {
    var xMin = Infinity;
    var yMin = Infinity;
    var xMax = -Infinity;
    var yMax = -Infinity;
    var cps = map(pathList, function (path) {
        var rect = path.getBoundingRect();
        var m = path.getComputedTransform();
        var x = rect.x + rect.width / 2 + (m ? m[4] : 0);
        var y = rect.y + rect.height / 2 + (m ? m[5] : 0);
        xMin = Math.min(x, xMin);
        yMin = Math.min(y, yMin);
        xMax = Math.max(x, xMax);
        yMax = Math.max(y, yMax);
        return [x, y];
    });
    var items = map(cps, function (cp, idx) {
        return {
            cp: cp,
            z: hilbert(cp[0], cp[1], xMin, yMin, xMax, yMax),
            path: pathList[idx]
        };
    });
    return items.sort(function (a, b) { return a.z - b.z; }).map(function (item) { return item.path; });
}
;
function defaultDividePath(param) {
    return split(param.path, param.count);
}
function createEmptyReturn() {
    return {
        fromIndividuals: [],
        toIndividuals: [],
        count: 0
    };
}
export function combineMorph(fromList, toPath, animationOpts) {
    var fromPathList = [];
    function addFromPath(fromList) {
        for (var i = 0; i < fromList.length; i++) {
            var from = fromList[i];
            if (isCombineMorphing(from)) {
                addFromPath(from.childrenRef());
            }
            else if (from instanceof Path) {
                fromPathList.push(from);
            }
        }
    }
    addFromPath(fromList);
    var separateCount = fromPathList.length;
    if (!separateCount) {
        return createEmptyReturn();
    }
    var dividePath = animationOpts.dividePath || defaultDividePath;
    var toSubPathList = dividePath({
        path: toPath, count: separateCount
    });
    if (toSubPathList.length !== separateCount) {
        console.error('Invalid morphing: unmatched splitted path');
        return createEmptyReturn();
    }
    fromPathList = sortPaths(fromPathList);
    toSubPathList = sortPaths(toSubPathList);
    var oldDone = animationOpts.done;
    var oldDuring = animationOpts.during;
    var individualDelay = animationOpts.individualDelay;
    var identityTransform = new Transformable();
    for (var i = 0; i < separateCount; i++) {
        var from = fromPathList[i];
        var to = toSubPathList[i];
        to.parent = toPath;
        to.copyTransform(identityTransform);
        if (!individualDelay) {
            prepareMorphPath(from, to);
        }
    }
    toPath.__isCombineMorphing = true;
    toPath.childrenRef = function () {
        return toSubPathList;
    };
    function addToSubPathListToZr(zr) {
        for (var i = 0; i < toSubPathList.length; i++) {
            toSubPathList[i].addSelfToZr(zr);
        }
    }
    saveAndModifyMethod(toPath, 'addSelfToZr', {
        after: function (zr) {
            addToSubPathListToZr(zr);
        }
    });
    saveAndModifyMethod(toPath, 'removeSelfFromZr', {
        after: function (zr) {
            for (var i = 0; i < toSubPathList.length; i++) {
                toSubPathList[i].removeSelfFromZr(zr);
            }
        }
    });
    function restoreToPath() {
        toPath.__isCombineMorphing = false;
        toPath.__morphT = -1;
        toPath.childrenRef = null;
        restoreMethod(toPath, 'addSelfToZr');
        restoreMethod(toPath, 'removeSelfFromZr');
    }
    var toLen = toSubPathList.length;
    if (individualDelay) {
        var animating_1 = toLen;
        var eachDone = function () {
            animating_1--;
            if (animating_1 === 0) {
                restoreToPath();
                oldDone && oldDone();
            }
        };
        for (var i = 0; i < toLen; i++) {
            var indivdualAnimationOpts = individualDelay ? defaults({
                delay: (animationOpts.delay || 0) + individualDelay(i, toLen, fromPathList[i], toSubPathList[i]),
                done: eachDone
            }, animationOpts) : animationOpts;
            morphPath(fromPathList[i], toSubPathList[i], indivdualAnimationOpts);
        }
    }
    else {
        toPath.__morphT = 0;
        toPath.animateTo({
            __morphT: 1
        }, defaults({
            during: function (p) {
                for (var i = 0; i < toLen; i++) {
                    var child = toSubPathList[i];
                    child.__morphT = toPath.__morphT;
                    child.dirtyShape();
                }
                oldDuring && oldDuring(p);
            },
            done: function () {
                restoreToPath();
                for (var i = 0; i < fromList.length; i++) {
                    restoreMethod(fromList[i], 'updateTransform');
                }
                oldDone && oldDone();
            }
        }, animationOpts));
    }
    if (toPath.__zr) {
        addToSubPathListToZr(toPath.__zr);
    }
    return {
        fromIndividuals: fromPathList,
        toIndividuals: toSubPathList,
        count: toLen
    };
}
export function separateMorph(fromPath, toPathList, animationOpts) {
    var toLen = toPathList.length;
    var fromPathList = [];
    var dividePath = animationOpts.dividePath || defaultDividePath;
    function addFromPath(fromList) {
        for (var i = 0; i < fromList.length; i++) {
            var from = fromList[i];
            if (isCombineMorphing(from)) {
                addFromPath(from.childrenRef());
            }
            else if (from instanceof Path) {
                fromPathList.push(from);
            }
        }
    }
    if (isCombineMorphing(fromPath)) {
        addFromPath(fromPath.childrenRef());
        var fromLen = fromPathList.length;
        if (fromLen < toLen) {
            var k = 0;
            for (var i = fromLen; i < toLen; i++) {
                fromPathList.push(clonePath(fromPathList[k++ % fromLen]));
            }
        }
        fromPathList.length = toLen;
    }
    else {
        fromPathList = dividePath({ path: fromPath, count: toLen });
        var fromPathTransform = fromPath.getComputedTransform();
        for (var i = 0; i < fromPathList.length; i++) {
            fromPathList[i].setLocalTransform(fromPathTransform);
        }
        if (fromPathList.length !== toLen) {
            console.error('Invalid morphing: unmatched splitted path');
            return createEmptyReturn();
        }
    }
    fromPathList = sortPaths(fromPathList);
    toPathList = sortPaths(toPathList);
    var individualDelay = animationOpts.individualDelay;
    for (var i = 0; i < toLen; i++) {
        var indivdualAnimationOpts = individualDelay ? defaults({
            delay: (animationOpts.delay || 0) + individualDelay(i, toLen, fromPathList[i], toPathList[i])
        }, animationOpts) : animationOpts;
        morphPath(fromPathList[i], toPathList[i], indivdualAnimationOpts);
    }
    return {
        fromIndividuals: fromPathList,
        toIndividuals: toPathList,
        count: toPathList.length
    };
}
export { split as defaultDividePath };
