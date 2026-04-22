import { copyTransform } from '../core/Transformable.js';
import { createBrushScope } from './core.js';
import SVGPathRebuilder from './SVGPathRebuilder.js';
import PathProxy from '../core/PathProxy.js';
import { getPathPrecision, getSRTTransformString } from './helper.js';
import { each, extend, filter, isNumber, isString, keys } from '../core/util.js';
import CompoundPath from '../graphic/CompoundPath.js';
import { createCubicEasingFunc } from '../animation/cubicEasing.js';
import { getClassId } from './cssClassId.js';
export var EASING_MAP = {
    cubicIn: '0.32,0,0.67,0',
    cubicOut: '0.33,1,0.68,1',
    cubicInOut: '0.65,0,0.35,1',
    quadraticIn: '0.11,0,0.5,0',
    quadraticOut: '0.5,1,0.89,1',
    quadraticInOut: '0.45,0,0.55,1',
    quarticIn: '0.5,0,0.75,0',
    quarticOut: '0.25,1,0.5,1',
    quarticInOut: '0.76,0,0.24,1',
    quinticIn: '0.64,0,0.78,0',
    quinticOut: '0.22,1,0.36,1',
    quinticInOut: '0.83,0,0.17,1',
    sinusoidalIn: '0.12,0,0.39,0',
    sinusoidalOut: '0.61,1,0.88,1',
    sinusoidalInOut: '0.37,0,0.63,1',
    exponentialIn: '0.7,0,0.84,0',
    exponentialOut: '0.16,1,0.3,1',
    exponentialInOut: '0.87,0,0.13,1',
    circularIn: '0.55,0,1,0.45',
    circularOut: '0,0.55,0.45,1',
    circularInOut: '0.85,0,0.15,1'
};
var transformOriginKey = 'transform-origin';
function buildPathString(el, kfShape, path) {
    var shape = extend({}, el.shape);
    extend(shape, kfShape);
    el.buildPath(path, shape);
    var svgPathBuilder = new SVGPathRebuilder();
    svgPathBuilder.reset(getPathPrecision(el));
    path.rebuildPath(svgPathBuilder, 1);
    svgPathBuilder.generateStr();
    return svgPathBuilder.getStr();
}
function setTransformOrigin(target, transform) {
    var originX = transform.originX, originY = transform.originY;
    if (originX || originY) {
        target[transformOriginKey] = originX + "px " + originY + "px";
    }
}
export var ANIMATE_STYLE_MAP = {
    fill: 'fill',
    opacity: 'opacity',
    lineWidth: 'stroke-width',
    lineDashOffset: 'stroke-dashoffset'
};
function addAnimation(cssAnim, scope) {
    var animationName = scope.zrId + '-ani-' + scope.cssAnimIdx++;
    scope.cssAnims[animationName] = cssAnim;
    return animationName;
}
function createCompoundPathCSSAnimation(el, attrs, scope) {
    var paths = el.shape.paths;
    var composedAnim = {};
    var cssAnimationCfg;
    var cssAnimationName;
    each(paths, function (path) {
        var subScope = createBrushScope(scope.zrId);
        subScope.animation = true;
        createCSSAnimation(path, {}, subScope, true);
        var cssAnims = subScope.cssAnims;
        var cssNodes = subScope.cssNodes;
        var animNames = keys(cssAnims);
        var len = animNames.length;
        if (!len) {
            return;
        }
        cssAnimationName = animNames[len - 1];
        var lastAnim = cssAnims[cssAnimationName];
        for (var percent in lastAnim) {
            var kf = lastAnim[percent];
            composedAnim[percent] = composedAnim[percent] || { d: '' };
            composedAnim[percent].d += kf.d || '';
        }
        for (var className in cssNodes) {
            var val = cssNodes[className].animation;
            if (val.indexOf(cssAnimationName) >= 0) {
                cssAnimationCfg = val;
            }
        }
    });
    if (!cssAnimationCfg) {
        return;
    }
    attrs.d = false;
    var animationName = addAnimation(composedAnim, scope);
    return cssAnimationCfg.replace(cssAnimationName, animationName);
}
function getEasingFunc(easing) {
    return isString(easing)
        ? EASING_MAP[easing]
            ? "cubic-bezier(" + EASING_MAP[easing] + ")"
            : createCubicEasingFunc(easing) ? easing : ''
        : '';
}
export function createCSSAnimation(el, attrs, scope, onlyShape) {
    var animators = el.animators;
    var len = animators.length;
    var cssAnimations = [];
    if (el instanceof CompoundPath) {
        var animationCfg = createCompoundPathCSSAnimation(el, attrs, scope);
        if (animationCfg) {
            cssAnimations.push(animationCfg);
        }
        else if (!len) {
            return;
        }
    }
    else if (!len) {
        return;
    }
    var groupAnimators = {};
    for (var i = 0; i < len; i++) {
        var animator = animators[i];
        var cfgArr = [animator.getMaxTime() / 1000 + 's'];
        var easing = getEasingFunc(animator.getClip().easing);
        var delay = animator.getDelay();
        if (easing) {
            cfgArr.push(easing);
        }
        else {
            cfgArr.push('linear');
        }
        if (delay) {
            cfgArr.push(delay / 1000 + 's');
        }
        if (animator.getLoop()) {
            cfgArr.push('infinite');
        }
        var cfg = cfgArr.join(' ');
        groupAnimators[cfg] = groupAnimators[cfg] || [cfg, []];
        groupAnimators[cfg][1].push(animator);
    }
    function createSingleCSSAnimation(groupAnimator) {
        var animators = groupAnimator[1];
        var len = animators.length;
        var transformKfs = {};
        var shapeKfs = {};
        var finalKfs = {};
        var animationTimingFunctionAttrName = 'animation-timing-function';
        function saveAnimatorTrackToCssKfs(animator, cssKfs, toCssAttrName) {
            var tracks = animator.getTracks();
            var maxTime = animator.getMaxTime();
            for (var k = 0; k < tracks.length; k++) {
                var track = tracks[k];
                if (track.needsAnimate()) {
                    var kfs = track.keyframes;
                    var attrName = track.propName;
                    toCssAttrName && (attrName = toCssAttrName(attrName));
                    if (attrName) {
                        for (var i = 0; i < kfs.length; i++) {
                            var kf = kfs[i];
                            var percent = Math.round(kf.time / maxTime * 100) + '%';
                            var kfEasing = getEasingFunc(kf.easing);
                            var rawValue = kf.rawValue;
                            if (isString(rawValue) || isNumber(rawValue)) {
                                cssKfs[percent] = cssKfs[percent] || {};
                                cssKfs[percent][attrName] = kf.rawValue;
                                if (kfEasing) {
                                    cssKfs[percent][animationTimingFunctionAttrName] = kfEasing;
                                }
                            }
                        }
                    }
                }
            }
        }
        for (var i = 0; i < len; i++) {
            var animator = animators[i];
            var targetProp = animator.targetName;
            if (!targetProp) {
                !onlyShape && saveAnimatorTrackToCssKfs(animator, transformKfs);
            }
            else if (targetProp === 'shape') {
                saveAnimatorTrackToCssKfs(animator, shapeKfs);
            }
        }
        for (var percent in transformKfs) {
            var transform = {};
            copyTransform(transform, el);
            extend(transform, transformKfs[percent]);
            var str = getSRTTransformString(transform);
            var timingFunction = transformKfs[percent][animationTimingFunctionAttrName];
            finalKfs[percent] = str ? {
                transform: str
            } : {};
            setTransformOrigin(finalKfs[percent], transform);
            if (timingFunction) {
                finalKfs[percent][animationTimingFunctionAttrName] = timingFunction;
            }
        }
        ;
        var path;
        var canAnimateShape = true;
        for (var percent in shapeKfs) {
            finalKfs[percent] = finalKfs[percent] || {};
            var isFirst = !path;
            var timingFunction = shapeKfs[percent][animationTimingFunctionAttrName];
            if (isFirst) {
                path = new PathProxy();
            }
            var len_1 = path.len();
            path.reset();
            finalKfs[percent].d = buildPathString(el, shapeKfs[percent], path);
            var newLen = path.len();
            if (!isFirst && len_1 !== newLen) {
                canAnimateShape = false;
                break;
            }
            if (timingFunction) {
                finalKfs[percent][animationTimingFunctionAttrName] = timingFunction;
            }
        }
        ;
        if (!canAnimateShape) {
            for (var percent in finalKfs) {
                delete finalKfs[percent].d;
            }
        }
        if (!onlyShape) {
            for (var i = 0; i < len; i++) {
                var animator = animators[i];
                var targetProp = animator.targetName;
                if (targetProp === 'style') {
                    saveAnimatorTrackToCssKfs(animator, finalKfs, function (propName) { return ANIMATE_STYLE_MAP[propName]; });
                }
            }
        }
        var percents = keys(finalKfs);
        var allTransformOriginSame = true;
        var transformOrigin;
        for (var i = 1; i < percents.length; i++) {
            var p0 = percents[i - 1];
            var p1 = percents[i];
            if (finalKfs[p0][transformOriginKey] !== finalKfs[p1][transformOriginKey]) {
                allTransformOriginSame = false;
                break;
            }
            transformOrigin = finalKfs[p0][transformOriginKey];
        }
        if (allTransformOriginSame && transformOrigin) {
            for (var percent in finalKfs) {
                if (finalKfs[percent][transformOriginKey]) {
                    delete finalKfs[percent][transformOriginKey];
                }
            }
            attrs[transformOriginKey] = transformOrigin;
        }
        if (filter(percents, function (percent) { return keys(finalKfs[percent]).length > 0; }).length) {
            var animationName = addAnimation(finalKfs, scope);
            return animationName + " " + groupAnimator[0] + " both";
        }
    }
    for (var key in groupAnimators) {
        var animationCfg = createSingleCSSAnimation(groupAnimators[key]);
        if (animationCfg) {
            cssAnimations.push(animationCfg);
        }
    }
    if (cssAnimations.length) {
        var className = scope.zrId + '-cls-' + getClassId();
        scope.cssNodes['.' + className] = {
            animation: cssAnimations.join(',')
        };
        attrs["class"] = className;
    }
}
