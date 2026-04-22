import Transformable, { copyTransform } from '../core/Transformable';
import Displayable from '../graphic/Displayable';
import { SVGVNodeAttrs, BrushScope, createBrushScope} from './core';
import Path from '../graphic/Path';
import SVGPathRebuilder from './SVGPathRebuilder';
import PathProxy from '../core/PathProxy';
import { getPathPrecision, getSRTTransformString } from './helper';
import { each, extend, filter, isNumber, isString, keys } from '../core/util';
import Animator from '../animation/Animator';
import CompoundPath from '../graphic/CompoundPath';
import { AnimationEasing } from '../animation/easing';
import { createCubicEasingFunc } from '../animation/cubicEasing';
import { getClassId } from './cssClassId';

export const EASING_MAP: Record<string, string> = {
    // From https://easings.net/
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
    // TODO elastic, bounce
};

const transformOriginKey = 'transform-origin';

function buildPathString(el: Path, kfShape: Path['shape'], path: PathProxy) {
    const shape = extend({}, el.shape);
    extend(shape, kfShape);

    el.buildPath(path, shape);
    const svgPathBuilder = new SVGPathRebuilder();
    svgPathBuilder.reset(getPathPrecision(el));
    path.rebuildPath(svgPathBuilder, 1);
    svgPathBuilder.generateStr();
    // will add path("") when generated to css string in the final step.
    return svgPathBuilder.getStr();
}

function setTransformOrigin(target: Record<string, string>, transform: Transformable) {
    const {originX, originY} = transform;
    if (originX || originY) {
        target[transformOriginKey] = `${originX}px ${originY}px`;
    }
}

export const ANIMATE_STYLE_MAP: Record<string, string> = {
    fill: 'fill',
    opacity: 'opacity',
    lineWidth: 'stroke-width',
    lineDashOffset: 'stroke-dashoffset'
    // TODO shadow is not supported.
};

type CssKF = Record<string, any>;

function addAnimation(cssAnim: Record<string, CssKF>, scope: BrushScope) {
    const animationName = scope.zrId + '-ani-' + scope.cssAnimIdx++;
    scope.cssAnims[animationName] = cssAnim;
    return animationName;
}

function createCompoundPathCSSAnimation(
    el: CompoundPath,
    attrs: SVGVNodeAttrs,
    scope: BrushScope
) {
    const paths = el.shape.paths;
    const composedAnim: Record<string, CssKF> = {};
    let cssAnimationCfg: string;
    let cssAnimationName: string;
    each(paths, path => {
        const subScope = createBrushScope(scope.zrId);
        subScope.animation = true;
        createCSSAnimation(path, {}, subScope, true);
        const cssAnims = subScope.cssAnims;
        const cssNodes = subScope.cssNodes;
        const animNames = keys(cssAnims);
        const len = animNames.length;
        if (!len) {
            return;
        }
        cssAnimationName = animNames[len - 1];
        // Only use last animation because they are conflicted.
        const lastAnim = cssAnims[cssAnimationName];
        // eslint-disable-next-line
        for (let percent in lastAnim) {
            const kf = lastAnim[percent];
            composedAnim[percent] = composedAnim[percent] || { d: '' };
            composedAnim[percent].d += kf.d || '';
        }
        // eslint-disable-next-line
        for (let className in cssNodes) {
            const val = cssNodes[className].animation;
            if (val.indexOf(cssAnimationName) >= 0) {
                // Only pick the animation configuration of last subpath.
                cssAnimationCfg = val;
            }
        }
    });

    if (!cssAnimationCfg) {
        return;
    }

    // Remove the attrs in the element because it will be set by animation.
    // Reduce the size.
    attrs.d = false;
    const animationName = addAnimation(composedAnim, scope);
    return cssAnimationCfg.replace(cssAnimationName, animationName);
}

function getEasingFunc(easing: AnimationEasing) {
    return isString(easing)
        ? EASING_MAP[easing]
            ? `cubic-bezier(${EASING_MAP[easing]})`
            : createCubicEasingFunc(easing) ? easing : ''
        : '';
}

export function createCSSAnimation(
    el: Displayable,
    attrs: SVGVNodeAttrs,
    scope: BrushScope,
    onlyShape?: boolean
) {
    const animators = el.animators;
    const len = animators.length;

    const cssAnimations: string[] = [];

    if (el instanceof CompoundPath) {
        const animationCfg = createCompoundPathCSSAnimation(el, attrs, scope);
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
    // Group animators by it's configuration
    const groupAnimators: Record<string, [string, Animator<any>[]]> = {};
    for (let i = 0; i < len; i++) {
        const animator = animators[i];
        const cfgArr: (string | number)[] = [animator.getMaxTime() / 1000 + 's'];
        const easing = getEasingFunc(animator.getClip().easing);
        const delay = animator.getDelay();

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
        const cfg = cfgArr.join(' ');

        // TODO fill mode
        groupAnimators[cfg] = groupAnimators[cfg] || [cfg, [] as Animator<any>[]];
        groupAnimators[cfg][1].push(animator);
    }

    function createSingleCSSAnimation(groupAnimator: [string, Animator<any>[]]) {
        const animators = groupAnimator[1];
        const len = animators.length;
        const transformKfs: Record<string, CssKF> = {};
        const shapeKfs: Record<string, CssKF> = {};

        const finalKfs: Record<string, CssKF> = {};

        const animationTimingFunctionAttrName = 'animation-timing-function';

        function saveAnimatorTrackToCssKfs(
            animator: Animator<any>,
            cssKfs: Record<string, CssKF>,
            toCssAttrName?: (propName: string) => string
        ) {
            const tracks = animator.getTracks();
            const maxTime = animator.getMaxTime();
            for (let k = 0; k < tracks.length; k++) {
                const track = tracks[k];
                if (track.needsAnimate()) {
                    const kfs = track.keyframes;
                    let attrName = track.propName;
                    toCssAttrName && (attrName = toCssAttrName(attrName));
                    if (attrName) {
                        for (let i = 0; i < kfs.length; i++) {
                            const kf = kfs[i];
                            const percent = Math.round(kf.time / maxTime * 100) + '%';
                            const kfEasing = getEasingFunc(kf.easing);
                            const rawValue = kf.rawValue;

                            // TODO gradient
                            if (isString(rawValue) || isNumber(rawValue)) {
                                cssKfs[percent] = cssKfs[percent] || {};
                                cssKfs[percent][attrName] = kf.rawValue;

                                if (kfEasing) {
                                    // TODO. If different property have different easings.
                                    cssKfs[percent][animationTimingFunctionAttrName] = kfEasing;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Find all transform animations.
        // TODO origin, parent
        for (let i = 0; i < len; i++) {
            const animator = animators[i];
            const targetProp = animator.targetName;
            if (!targetProp) {
                !onlyShape && saveAnimatorTrackToCssKfs(animator, transformKfs);
            }
            else if (targetProp === 'shape') {
                saveAnimatorTrackToCssKfs(animator, shapeKfs);
            }
        }

        // eslint-disable-next-line
        for (let percent in transformKfs) {
            const transform = {} as Transformable;
            copyTransform(transform, el);
            extend(transform, transformKfs[percent]);
            const str = getSRTTransformString(transform);
            const timingFunction = transformKfs[percent][animationTimingFunctionAttrName];
            finalKfs[percent] = str ? {
                transform: str
            } : {};
            // TODO set transform origin in element?
            setTransformOrigin(finalKfs[percent], transform);

            // Save timing function
            if (timingFunction) {
                finalKfs[percent][animationTimingFunctionAttrName] = timingFunction;
            }
        };


        let path: PathProxy;
        let canAnimateShape = true;
        // eslint-disable-next-line
        for (let percent in shapeKfs) {
            finalKfs[percent] = finalKfs[percent] || {};

            const isFirst = !path;
            const timingFunction = shapeKfs[percent][animationTimingFunctionAttrName];

            if (isFirst) {
                path = new PathProxy();
            }
            let len = path.len();
            path.reset();
            finalKfs[percent].d = buildPathString(el as Path, shapeKfs[percent], path);
            let newLen = path.len();
            // Path data don't match.
            if (!isFirst && len !== newLen) {
                canAnimateShape = false;
                break;
            }

            // Save timing function
            if (timingFunction) {
                finalKfs[percent][animationTimingFunctionAttrName] = timingFunction;
            }
        };
        if (!canAnimateShape) {
            // eslint-disable-next-line
            for (let percent in finalKfs) {
                delete finalKfs[percent].d;
            }
        }

        if (!onlyShape) {
            for (let i = 0; i < len; i++) {
                const animator = animators[i];
                const targetProp = animator.targetName;
                if (targetProp === 'style') {
                    saveAnimatorTrackToCssKfs(
                        animator, finalKfs, (propName) => ANIMATE_STYLE_MAP[propName]
                    );
                }
            }
        }

        const percents = keys(finalKfs);

        // Set transform origin in attribute to reduce the size.
        let allTransformOriginSame = true;
        let transformOrigin;
        for (let i = 1; i < percents.length; i++) {
            const p0 = percents[i - 1];
            const p1 = percents[i];
            if (finalKfs[p0][transformOriginKey] !== finalKfs[p1][transformOriginKey]) {
                allTransformOriginSame = false;
                break;
            }
            transformOrigin = finalKfs[p0][transformOriginKey];
        }
        if (allTransformOriginSame && transformOrigin) {
            for (const percent in finalKfs) {
                if (finalKfs[percent][transformOriginKey]) {
                    delete finalKfs[percent][transformOriginKey];
                }
            }
            attrs[transformOriginKey] = transformOrigin;
        }

        if (filter(
            percents, (percent) => keys(finalKfs[percent]).length > 0
        ).length) {
            const animationName = addAnimation(finalKfs, scope);
            // eslint-disable-next-line
            // for (const attrName in finalKfs[percents[0]]) {
            //     // Remove the attrs in the element because it will be set by animation.
            //     // Reduce the size.
            //     attrs[attrName] = false;
            // }
            // animationName {duration easing delay loop} fillMode
            return `${animationName} ${groupAnimator[0]} both`;
        }
    }

    // eslint-disable-next-line
    for (let key in groupAnimators) {
        const animationCfg = createSingleCSSAnimation(groupAnimators[key]);
        if (animationCfg) {
            cssAnimations.push(animationCfg);
        }
    }

    if (cssAnimations.length) {
        const className = scope.zrId + '-cls-' + getClassId();
        scope.cssNodes['.' + className] = {
            animation: cssAnimations.join(',')
        };
        // TODO exists class?
        attrs.class = className;
    }
}