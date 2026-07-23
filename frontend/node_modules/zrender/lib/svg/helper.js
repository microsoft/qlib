import { RADIAN_TO_DEGREE, retrieve2, logError, isFunction } from '../core/util.js';
import { parse } from '../tool/color.js';
import env from '../core/env.js';
var mathRound = Math.round;
export function normalizeColor(color) {
    var opacity;
    if (!color || color === 'transparent') {
        color = 'none';
    }
    else if (typeof color === 'string' && color.indexOf('rgba') > -1) {
        var arr = parse(color);
        if (arr) {
            color = 'rgb(' + arr[0] + ',' + arr[1] + ',' + arr[2] + ')';
            opacity = arr[3];
        }
    }
    return {
        color: color,
        opacity: opacity == null ? 1 : opacity
    };
}
var EPSILON = 1e-4;
export function isAroundZero(transform) {
    return transform < EPSILON && transform > -EPSILON;
}
export function round3(transform) {
    return mathRound(transform * 1e3) / 1e3;
}
export function round4(transform) {
    return mathRound(transform * 1e4) / 1e4;
}
export function round1(transform) {
    return mathRound(transform * 10) / 10;
}
export function getMatrixStr(m) {
    return 'matrix('
        + round3(m[0]) + ','
        + round3(m[1]) + ','
        + round3(m[2]) + ','
        + round3(m[3]) + ','
        + round4(m[4]) + ','
        + round4(m[5])
        + ')';
}
export var TEXT_ALIGN_TO_ANCHOR = {
    left: 'start',
    right: 'end',
    center: 'middle',
    middle: 'middle'
};
export function adjustTextY(y, lineHeight, textBaseline) {
    if (textBaseline === 'top') {
        y += lineHeight / 2;
    }
    else if (textBaseline === 'bottom') {
        y -= lineHeight / 2;
    }
    return y;
}
export function hasShadow(style) {
    return style
        && (style.shadowBlur || style.shadowOffsetX || style.shadowOffsetY);
}
export function getShadowKey(displayable) {
    var style = displayable.style;
    var globalScale = displayable.getGlobalScale();
    return [
        style.shadowColor,
        (style.shadowBlur || 0).toFixed(2),
        (style.shadowOffsetX || 0).toFixed(2),
        (style.shadowOffsetY || 0).toFixed(2),
        globalScale[0],
        globalScale[1]
    ].join(',');
}
export function getClipPathsKey(clipPaths) {
    var key = [];
    if (clipPaths) {
        for (var i = 0; i < clipPaths.length; i++) {
            var clipPath = clipPaths[i];
            key.push(clipPath.id);
        }
    }
    return key.join(',');
}
export function isImagePattern(val) {
    return val && (!!val.image);
}
export function isSVGPattern(val) {
    return val && (!!val.svgElement);
}
export function isPattern(val) {
    return isImagePattern(val) || isSVGPattern(val);
}
export function isLinearGradient(val) {
    return val.type === 'linear';
}
export function isRadialGradient(val) {
    return val.type === 'radial';
}
export function isGradient(val) {
    return val && (val.type === 'linear'
        || val.type === 'radial');
}
export function getIdURL(id) {
    return "url(#" + id + ")";
}
export function getPathPrecision(el) {
    var scale = el.getGlobalScale();
    var size = Math.max(scale[0], scale[1]);
    return Math.max(Math.ceil(Math.log(size) / Math.log(10)), 1);
}
export function getSRTTransformString(transform) {
    var x = transform.x || 0;
    var y = transform.y || 0;
    var rotation = (transform.rotation || 0) * RADIAN_TO_DEGREE;
    var scaleX = retrieve2(transform.scaleX, 1);
    var scaleY = retrieve2(transform.scaleY, 1);
    var skewX = transform.skewX || 0;
    var skewY = transform.skewY || 0;
    var res = [];
    if (x || y) {
        res.push("translate(" + x + "px," + y + "px)");
    }
    if (rotation) {
        res.push("rotate(" + rotation + ")");
    }
    if (scaleX !== 1 || scaleY !== 1) {
        res.push("scale(" + scaleX + "," + scaleY + ")");
    }
    if (skewX || skewY) {
        res.push("skew(" + mathRound(skewX * RADIAN_TO_DEGREE) + "deg, " + mathRound(skewY * RADIAN_TO_DEGREE) + "deg)");
    }
    return res.join(' ');
}
export var encodeBase64 = (function () {
    if (env.hasGlobalWindow && isFunction(window.btoa)) {
        return function (str) {
            return window.btoa(unescape(encodeURIComponent(str)));
        };
    }
    if (typeof Buffer !== 'undefined') {
        return function (str) {
            return Buffer.from(str).toString('base64');
        };
    }
    return function (str) {
        if (process.env.NODE_ENV !== 'production') {
            logError('Base64 isn\'t natively supported in the current environment.');
        }
        return null;
    };
})();
