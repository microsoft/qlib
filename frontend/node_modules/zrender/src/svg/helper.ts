// Shared methods of svg and svg-ssr

import { MatrixArray } from '../core/matrix';
import Transformable, { TransformProp } from '../core/Transformable';
import { RADIAN_TO_DEGREE, retrieve2, logError, isFunction } from '../core/util';
import Displayable from '../graphic/Displayable';
import { GradientObject } from '../graphic/Gradient';
import { LinearGradientObject } from '../graphic/LinearGradient';
import Path from '../graphic/Path';
import { ImagePatternObject, PatternObject, SVGPatternObject } from '../graphic/Pattern';
import { RadialGradientObject } from '../graphic/RadialGradient';
import { parse } from '../tool/color';
import env from '../core/env';

const mathRound = Math.round;

export function normalizeColor(color: string): { color: string; opacity: number; } {
    let opacity;
    if (!color || color === 'transparent') {
        color = 'none';
    }
    else if (typeof color === 'string' && color.indexOf('rgba') > -1) {
        const arr = parse(color);
        if (arr) {
            // TODO use hex?
            color = 'rgb(' + arr[0] + ',' + arr[1] + ',' + arr[2] + ')';
            opacity = arr[3];
        }
    }
    return {
        color,
        opacity: opacity == null ? 1 : opacity
    };
}
const EPSILON = 1e-4;
export function isAroundZero(transform: number) {
    return transform < EPSILON && transform > -EPSILON;
}

export function round3(transform: number) {
    return mathRound(transform * 1e3) / 1e3;
}
export function round4(transform: number) {
    return mathRound(transform * 1e4) / 1e4;
}
export function round1(transform: number) {
    return mathRound(transform * 10) / 10;
}

export function getMatrixStr(m: MatrixArray) {
    return 'matrix('
        // Avoid large string of matrix
        // PENDING If have precision issue when scaled
        + round3(m[0]) + ','
        + round3(m[1]) + ','
        + round3(m[2]) + ','
        + round3(m[3]) + ','
        + round4(m[4]) + ','
        + round4(m[5])
        + ')';
}

export const TEXT_ALIGN_TO_ANCHOR = {
    left: 'start',
    right: 'end',
    center: 'middle',
    middle: 'middle'
};

export function adjustTextY(y: number, lineHeight: number, textBaseline: CanvasTextBaseline): number {
    // TODO Other baselines.
    if (textBaseline === 'top') {
        y += lineHeight / 2;
    }
    else if (textBaseline === 'bottom') {
        y -= lineHeight / 2;
    }
    return y;
}


export function hasShadow(style: Displayable['style']) {
    // TODO: textBoxShadowBlur is not supported yet
    return style
        && (style.shadowBlur || style.shadowOffsetX || style.shadowOffsetY);
}

export function getShadowKey(displayable: Displayable) {
    const style = displayable.style;
    const globalScale = displayable.getGlobalScale();
    return [
        style.shadowColor,
        (style.shadowBlur || 0).toFixed(2), // Reduce the precision
        (style.shadowOffsetX || 0).toFixed(2),
        (style.shadowOffsetY || 0).toFixed(2),
        globalScale[0],
        globalScale[1]
    ].join(',');
}

export function getClipPathsKey(clipPaths: Path[]) {
    let key: number[] = [];
    if (clipPaths) {
        for (let i = 0; i < clipPaths.length; i++) {
            const clipPath = clipPaths[i];
            key.push(clipPath.id);
        }
    }
    return key.join(',');
}

export function isImagePattern(val: any): val is ImagePatternObject {
    return val && (!!(val as ImagePatternObject).image);
}
export function isSVGPattern(val: any): val is SVGPatternObject {
    return val && (!!(val as SVGPatternObject).svgElement);
}
export function isPattern(val: any): val is PatternObject {
    return isImagePattern(val) || isSVGPattern(val);
}

export function isLinearGradient(val: GradientObject): val is LinearGradientObject {
    return val.type === 'linear';
}

export function isRadialGradient(val: GradientObject): val is RadialGradientObject {
    return val.type === 'radial';
}

export function isGradient(val: any): val is GradientObject {
    return val && (
        (val as GradientObject).type === 'linear'
        || (val as GradientObject).type === 'radial'
    );
}

export function getIdURL(id: string) {
    return `url(#${id})`;
}

export function getPathPrecision(el: Path) {
    const scale = el.getGlobalScale();
    const size = Math.max(scale[0], scale[1]);
    return Math.max(Math.ceil(Math.log(size) / Math.log(10)), 1);
}

export function getSRTTransformString(
    transform: Partial<Pick<Transformable, TransformProp>>
) {
    const x = transform.x || 0;
    const y = transform.y || 0;
    const rotation = (transform.rotation || 0) * RADIAN_TO_DEGREE;
    const scaleX = retrieve2(transform.scaleX, 1);
    const scaleY = retrieve2(transform.scaleY, 1);
    const skewX = transform.skewX || 0;
    const skewY = transform.skewY || 0;
    const res = [];
    if (x || y) {
        // TODO not using px unit?
        res.push(`translate(${x}px,${y}px)`);
    }
    if (rotation) {
        res.push(`rotate(${rotation})`);
    }
    if (scaleX !== 1 || scaleY !== 1) {
        res.push(`scale(${scaleX},${scaleY})`);
    }
    if (skewX || skewY) {
        res.push(`skew(${mathRound(skewX * RADIAN_TO_DEGREE)}deg, ${mathRound(skewY * RADIAN_TO_DEGREE)}deg)`);
    }

    return res.join(' ');
}

export const encodeBase64 = (function () {
    if (env.hasGlobalWindow && isFunction(window.btoa)) {
        return function (str: string) {
            return window.btoa(unescape(encodeURIComponent(str)));
        };
    }
    if (typeof Buffer !== 'undefined') {
        return function (str: string) {
            return Buffer.from(str).toString('base64');
        };
    }
    return function (str: string): string {
        if (process.env.NODE_ENV !== 'production') {
            logError('Base64 isn\'t natively supported in the current environment.');
        }
        return null;
    };
})();