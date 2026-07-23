import { DEFAULT_COMMON_STYLE } from '../graphic/Displayable.js';
import PathProxy from '../core/PathProxy.js';
import { createOrUpdateImage, isImageReady } from '../graphic/helper/image.js';
import { getCanvasGradient, isClipPathChanged } from './helper.js';
import Path from '../graphic/Path.js';
import ZRImage from '../graphic/Image.js';
import TSpan from '../graphic/TSpan.js';
import { RADIAN_TO_DEGREE } from '../core/util.js';
import { getLineDash } from './dashStyle.js';
import { REDRAW_BIT, SHAPE_CHANGED_BIT } from '../graphic/constants.js';
import { DEFAULT_FONT } from '../core/platform.js';
var pathProxyForDraw = new PathProxy(true);
function styleHasStroke(style) {
    var stroke = style.stroke;
    return !(stroke == null || stroke === 'none' || !(style.lineWidth > 0));
}
function isValidStrokeFillStyle(strokeOrFill) {
    return typeof strokeOrFill === 'string' && strokeOrFill !== 'none';
}
function styleHasFill(style) {
    var fill = style.fill;
    return fill != null && fill !== 'none';
}
function doFillPath(ctx, style) {
    if (style.fillOpacity != null && style.fillOpacity !== 1) {
        var originalGlobalAlpha = ctx.globalAlpha;
        ctx.globalAlpha = style.fillOpacity * style.opacity;
        ctx.fill();
        ctx.globalAlpha = originalGlobalAlpha;
    }
    else {
        ctx.fill();
    }
}
function doStrokePath(ctx, style) {
    if (style.strokeOpacity != null && style.strokeOpacity !== 1) {
        var originalGlobalAlpha = ctx.globalAlpha;
        ctx.globalAlpha = style.strokeOpacity * style.opacity;
        ctx.stroke();
        ctx.globalAlpha = originalGlobalAlpha;
    }
    else {
        ctx.stroke();
    }
}
export function createCanvasPattern(ctx, pattern, el) {
    var image = createOrUpdateImage(pattern.image, pattern.__image, el);
    if (isImageReady(image)) {
        var canvasPattern = ctx.createPattern(image, pattern.repeat || 'repeat');
        if (typeof DOMMatrix === 'function'
            && canvasPattern
            && canvasPattern.setTransform) {
            var matrix = new DOMMatrix();
            matrix.translateSelf((pattern.x || 0), (pattern.y || 0));
            matrix.rotateSelf(0, 0, (pattern.rotation || 0) * RADIAN_TO_DEGREE);
            matrix.scaleSelf((pattern.scaleX || 1), (pattern.scaleY || 1));
            canvasPattern.setTransform(matrix);
        }
        return canvasPattern;
    }
}
function brushPath(ctx, el, style, inBatch) {
    var _a;
    var hasStroke = styleHasStroke(style);
    var hasFill = styleHasFill(style);
    var strokePercent = style.strokePercent;
    var strokePart = strokePercent < 1;
    var firstDraw = !el.path;
    if ((!el.silent || strokePart) && firstDraw) {
        el.createPathProxy();
    }
    var path = el.path || pathProxyForDraw;
    var dirtyFlag = el.__dirty;
    if (!inBatch) {
        var fill = style.fill;
        var stroke = style.stroke;
        var hasFillGradient = hasFill && !!fill.colorStops;
        var hasStrokeGradient = hasStroke && !!stroke.colorStops;
        var hasFillPattern = hasFill && !!fill.image;
        var hasStrokePattern = hasStroke && !!stroke.image;
        var fillGradient = void 0;
        var strokeGradient = void 0;
        var fillPattern = void 0;
        var strokePattern = void 0;
        var rect = void 0;
        if (hasFillGradient || hasStrokeGradient) {
            rect = el.getBoundingRect();
        }
        if (hasFillGradient) {
            fillGradient = dirtyFlag
                ? getCanvasGradient(ctx, fill, rect)
                : el.__canvasFillGradient;
            el.__canvasFillGradient = fillGradient;
        }
        if (hasStrokeGradient) {
            strokeGradient = dirtyFlag
                ? getCanvasGradient(ctx, stroke, rect)
                : el.__canvasStrokeGradient;
            el.__canvasStrokeGradient = strokeGradient;
        }
        if (hasFillPattern) {
            fillPattern = (dirtyFlag || !el.__canvasFillPattern)
                ? createCanvasPattern(ctx, fill, el)
                : el.__canvasFillPattern;
            el.__canvasFillPattern = fillPattern;
        }
        if (hasStrokePattern) {
            strokePattern = (dirtyFlag || !el.__canvasStrokePattern)
                ? createCanvasPattern(ctx, stroke, el)
                : el.__canvasStrokePattern;
            el.__canvasStrokePattern = fillPattern;
        }
        if (hasFillGradient) {
            ctx.fillStyle = fillGradient;
        }
        else if (hasFillPattern) {
            if (fillPattern) {
                ctx.fillStyle = fillPattern;
            }
            else {
                hasFill = false;
            }
        }
        if (hasStrokeGradient) {
            ctx.strokeStyle = strokeGradient;
        }
        else if (hasStrokePattern) {
            if (strokePattern) {
                ctx.strokeStyle = strokePattern;
            }
            else {
                hasStroke = false;
            }
        }
    }
    var scale = el.getGlobalScale();
    path.setScale(scale[0], scale[1], el.segmentIgnoreThreshold);
    var lineDash;
    var lineDashOffset;
    if (ctx.setLineDash && style.lineDash) {
        _a = getLineDash(el), lineDash = _a[0], lineDashOffset = _a[1];
    }
    var needsRebuild = true;
    if (firstDraw || (dirtyFlag & SHAPE_CHANGED_BIT)) {
        path.setDPR(ctx.dpr);
        if (strokePart) {
            path.setContext(null);
        }
        else {
            path.setContext(ctx);
            needsRebuild = false;
        }
        path.reset();
        el.buildPath(path, el.shape, inBatch);
        path.toStatic();
        el.pathUpdated();
    }
    if (needsRebuild) {
        path.rebuildPath(ctx, strokePart ? strokePercent : 1);
    }
    if (lineDash) {
        ctx.setLineDash(lineDash);
        ctx.lineDashOffset = lineDashOffset;
    }
    if (!inBatch) {
        if (style.strokeFirst) {
            if (hasStroke) {
                doStrokePath(ctx, style);
            }
            if (hasFill) {
                doFillPath(ctx, style);
            }
        }
        else {
            if (hasFill) {
                doFillPath(ctx, style);
            }
            if (hasStroke) {
                doStrokePath(ctx, style);
            }
        }
    }
    if (lineDash) {
        ctx.setLineDash([]);
    }
}
function brushImage(ctx, el, style) {
    var image = el.__image = createOrUpdateImage(style.image, el.__image, el, el.onload);
    if (!image || !isImageReady(image)) {
        return;
    }
    var x = style.x || 0;
    var y = style.y || 0;
    var width = el.getWidth();
    var height = el.getHeight();
    var aspect = image.width / image.height;
    if (width == null && height != null) {
        width = height * aspect;
    }
    else if (height == null && width != null) {
        height = width / aspect;
    }
    else if (width == null && height == null) {
        width = image.width;
        height = image.height;
    }
    if (style.sWidth && style.sHeight) {
        var sx = style.sx || 0;
        var sy = style.sy || 0;
        ctx.drawImage(image, sx, sy, style.sWidth, style.sHeight, x, y, width, height);
    }
    else if (style.sx && style.sy) {
        var sx = style.sx;
        var sy = style.sy;
        var sWidth = width - sx;
        var sHeight = height - sy;
        ctx.drawImage(image, sx, sy, sWidth, sHeight, x, y, width, height);
    }
    else {
        ctx.drawImage(image, x, y, width, height);
    }
}
function brushText(ctx, el, style) {
    var _a;
    var text = style.text;
    text != null && (text += '');
    if (text) {
        ctx.font = style.font || DEFAULT_FONT;
        ctx.textAlign = style.textAlign;
        ctx.textBaseline = style.textBaseline;
        var lineDash = void 0;
        var lineDashOffset = void 0;
        if (ctx.setLineDash && style.lineDash) {
            _a = getLineDash(el), lineDash = _a[0], lineDashOffset = _a[1];
        }
        if (lineDash) {
            ctx.setLineDash(lineDash);
            ctx.lineDashOffset = lineDashOffset;
        }
        if (style.strokeFirst) {
            if (styleHasStroke(style)) {
                ctx.strokeText(text, style.x, style.y);
            }
            if (styleHasFill(style)) {
                ctx.fillText(text, style.x, style.y);
            }
        }
        else {
            if (styleHasFill(style)) {
                ctx.fillText(text, style.x, style.y);
            }
            if (styleHasStroke(style)) {
                ctx.strokeText(text, style.x, style.y);
            }
        }
        if (lineDash) {
            ctx.setLineDash([]);
        }
    }
}
var SHADOW_NUMBER_PROPS = ['shadowBlur', 'shadowOffsetX', 'shadowOffsetY'];
var STROKE_PROPS = [
    ['lineCap', 'butt'], ['lineJoin', 'miter'], ['miterLimit', 10]
];
function bindCommonProps(ctx, style, prevStyle, forceSetAll, scope) {
    var styleChanged = false;
    if (!forceSetAll) {
        prevStyle = prevStyle || {};
        if (style === prevStyle) {
            return false;
        }
    }
    if (forceSetAll || style.opacity !== prevStyle.opacity) {
        flushPathDrawn(ctx, scope);
        styleChanged = true;
        var opacity = Math.max(Math.min(style.opacity, 1), 0);
        ctx.globalAlpha = isNaN(opacity) ? DEFAULT_COMMON_STYLE.opacity : opacity;
    }
    if (forceSetAll || style.blend !== prevStyle.blend) {
        if (!styleChanged) {
            flushPathDrawn(ctx, scope);
            styleChanged = true;
        }
        ctx.globalCompositeOperation = style.blend || DEFAULT_COMMON_STYLE.blend;
    }
    for (var i = 0; i < SHADOW_NUMBER_PROPS.length; i++) {
        var propName = SHADOW_NUMBER_PROPS[i];
        if (forceSetAll || style[propName] !== prevStyle[propName]) {
            if (!styleChanged) {
                flushPathDrawn(ctx, scope);
                styleChanged = true;
            }
            ctx[propName] = ctx.dpr * (style[propName] || 0);
        }
    }
    if (forceSetAll || style.shadowColor !== prevStyle.shadowColor) {
        if (!styleChanged) {
            flushPathDrawn(ctx, scope);
            styleChanged = true;
        }
        ctx.shadowColor = style.shadowColor || DEFAULT_COMMON_STYLE.shadowColor;
    }
    return styleChanged;
}
function bindPathAndTextCommonStyle(ctx, el, prevEl, forceSetAll, scope) {
    var style = getStyle(el, scope.inHover);
    var prevStyle = forceSetAll
        ? null
        : (prevEl && getStyle(prevEl, scope.inHover) || {});
    if (style === prevStyle) {
        return false;
    }
    var styleChanged = bindCommonProps(ctx, style, prevStyle, forceSetAll, scope);
    if (forceSetAll || style.fill !== prevStyle.fill) {
        if (!styleChanged) {
            flushPathDrawn(ctx, scope);
            styleChanged = true;
        }
        isValidStrokeFillStyle(style.fill) && (ctx.fillStyle = style.fill);
    }
    if (forceSetAll || style.stroke !== prevStyle.stroke) {
        if (!styleChanged) {
            flushPathDrawn(ctx, scope);
            styleChanged = true;
        }
        isValidStrokeFillStyle(style.stroke) && (ctx.strokeStyle = style.stroke);
    }
    if (forceSetAll || style.opacity !== prevStyle.opacity) {
        if (!styleChanged) {
            flushPathDrawn(ctx, scope);
            styleChanged = true;
        }
        ctx.globalAlpha = style.opacity == null ? 1 : style.opacity;
    }
    if (el.hasStroke()) {
        var lineWidth = style.lineWidth;
        var newLineWidth = lineWidth / ((style.strokeNoScale && el.getLineScale) ? el.getLineScale() : 1);
        if (ctx.lineWidth !== newLineWidth) {
            if (!styleChanged) {
                flushPathDrawn(ctx, scope);
                styleChanged = true;
            }
            ctx.lineWidth = newLineWidth;
        }
    }
    for (var i = 0; i < STROKE_PROPS.length; i++) {
        var prop = STROKE_PROPS[i];
        var propName = prop[0];
        if (forceSetAll || style[propName] !== prevStyle[propName]) {
            if (!styleChanged) {
                flushPathDrawn(ctx, scope);
                styleChanged = true;
            }
            ctx[propName] = style[propName] || prop[1];
        }
    }
    return styleChanged;
}
function bindImageStyle(ctx, el, prevEl, forceSetAll, scope) {
    return bindCommonProps(ctx, getStyle(el, scope.inHover), prevEl && getStyle(prevEl, scope.inHover), forceSetAll, scope);
}
function setContextTransform(ctx, el) {
    var m = el.transform;
    var dpr = ctx.dpr || 1;
    if (m) {
        ctx.setTransform(dpr * m[0], dpr * m[1], dpr * m[2], dpr * m[3], dpr * m[4], dpr * m[5]);
    }
    else {
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
}
function updateClipStatus(clipPaths, ctx, scope) {
    var allClipped = false;
    for (var i = 0; i < clipPaths.length; i++) {
        var clipPath = clipPaths[i];
        allClipped = allClipped || clipPath.isZeroArea();
        setContextTransform(ctx, clipPath);
        ctx.beginPath();
        clipPath.buildPath(ctx, clipPath.shape);
        ctx.clip();
    }
    scope.allClipped = allClipped;
}
function isTransformChanged(m0, m1) {
    if (m0 && m1) {
        return m0[0] !== m1[0]
            || m0[1] !== m1[1]
            || m0[2] !== m1[2]
            || m0[3] !== m1[3]
            || m0[4] !== m1[4]
            || m0[5] !== m1[5];
    }
    else if (!m0 && !m1) {
        return false;
    }
    return true;
}
var DRAW_TYPE_PATH = 1;
var DRAW_TYPE_IMAGE = 2;
var DRAW_TYPE_TEXT = 3;
var DRAW_TYPE_INCREMENTAL = 4;
function canPathBatch(style) {
    var hasFill = styleHasFill(style);
    var hasStroke = styleHasStroke(style);
    return !(style.lineDash
        || !(+hasFill ^ +hasStroke)
        || (hasFill && typeof style.fill !== 'string')
        || (hasStroke && typeof style.stroke !== 'string')
        || style.strokePercent < 1
        || style.strokeOpacity < 1
        || style.fillOpacity < 1);
}
function flushPathDrawn(ctx, scope) {
    scope.batchFill && ctx.fill();
    scope.batchStroke && ctx.stroke();
    scope.batchFill = '';
    scope.batchStroke = '';
}
function getStyle(el, inHover) {
    return inHover ? (el.__hoverStyle || el.style) : el.style;
}
export function brushSingle(ctx, el) {
    brush(ctx, el, { inHover: false, viewWidth: 0, viewHeight: 0 }, true);
}
export function brush(ctx, el, scope, isLast) {
    var m = el.transform;
    if (!el.shouldBePainted(scope.viewWidth, scope.viewHeight, false, false)) {
        el.__dirty &= ~REDRAW_BIT;
        el.__isRendered = false;
        return;
    }
    var clipPaths = el.__clipPaths;
    var prevElClipPaths = scope.prevElClipPaths;
    var forceSetTransform = false;
    var forceSetStyle = false;
    if (!prevElClipPaths || isClipPathChanged(clipPaths, prevElClipPaths)) {
        if (prevElClipPaths && prevElClipPaths.length) {
            flushPathDrawn(ctx, scope);
            ctx.restore();
            forceSetStyle = forceSetTransform = true;
            scope.prevElClipPaths = null;
            scope.allClipped = false;
            scope.prevEl = null;
        }
        if (clipPaths && clipPaths.length) {
            flushPathDrawn(ctx, scope);
            ctx.save();
            updateClipStatus(clipPaths, ctx, scope);
            forceSetTransform = true;
        }
        scope.prevElClipPaths = clipPaths;
    }
    if (scope.allClipped) {
        el.__isRendered = false;
        return;
    }
    el.beforeBrush && el.beforeBrush();
    el.innerBeforeBrush();
    var prevEl = scope.prevEl;
    if (!prevEl) {
        forceSetStyle = forceSetTransform = true;
    }
    var canBatchPath = el instanceof Path
        && el.autoBatch
        && canPathBatch(el.style);
    if (forceSetTransform || isTransformChanged(m, prevEl.transform)) {
        flushPathDrawn(ctx, scope);
        setContextTransform(ctx, el);
    }
    else if (!canBatchPath) {
        flushPathDrawn(ctx, scope);
    }
    var style = getStyle(el, scope.inHover);
    if (el instanceof Path) {
        if (scope.lastDrawType !== DRAW_TYPE_PATH) {
            forceSetStyle = true;
            scope.lastDrawType = DRAW_TYPE_PATH;
        }
        bindPathAndTextCommonStyle(ctx, el, prevEl, forceSetStyle, scope);
        if (!canBatchPath || (!scope.batchFill && !scope.batchStroke)) {
            ctx.beginPath();
        }
        brushPath(ctx, el, style, canBatchPath);
        if (canBatchPath) {
            scope.batchFill = style.fill || '';
            scope.batchStroke = style.stroke || '';
        }
    }
    else {
        if (el instanceof TSpan) {
            if (scope.lastDrawType !== DRAW_TYPE_TEXT) {
                forceSetStyle = true;
                scope.lastDrawType = DRAW_TYPE_TEXT;
            }
            bindPathAndTextCommonStyle(ctx, el, prevEl, forceSetStyle, scope);
            brushText(ctx, el, style);
        }
        else if (el instanceof ZRImage) {
            if (scope.lastDrawType !== DRAW_TYPE_IMAGE) {
                forceSetStyle = true;
                scope.lastDrawType = DRAW_TYPE_IMAGE;
            }
            bindImageStyle(ctx, el, prevEl, forceSetStyle, scope);
            brushImage(ctx, el, style);
        }
        else if (el.getTemporalDisplayables) {
            if (scope.lastDrawType !== DRAW_TYPE_INCREMENTAL) {
                forceSetStyle = true;
                scope.lastDrawType = DRAW_TYPE_INCREMENTAL;
            }
            brushIncremental(ctx, el, scope);
        }
    }
    if (canBatchPath && isLast) {
        flushPathDrawn(ctx, scope);
    }
    el.innerAfterBrush();
    el.afterBrush && el.afterBrush();
    scope.prevEl = el;
    el.__dirty = 0;
    el.__isRendered = true;
}
function brushIncremental(ctx, el, scope) {
    var displayables = el.getDisplayables();
    var temporalDisplayables = el.getTemporalDisplayables();
    ctx.save();
    var innerScope = {
        prevElClipPaths: null,
        prevEl: null,
        allClipped: false,
        viewWidth: scope.viewWidth,
        viewHeight: scope.viewHeight,
        inHover: scope.inHover
    };
    var i;
    var len;
    for (i = el.getCursor(), len = displayables.length; i < len; i++) {
        var displayable = displayables[i];
        displayable.beforeBrush && displayable.beforeBrush();
        displayable.innerBeforeBrush();
        brush(ctx, displayable, innerScope, i === len - 1);
        displayable.innerAfterBrush();
        displayable.afterBrush && displayable.afterBrush();
        innerScope.prevEl = displayable;
    }
    for (var i_1 = 0, len_1 = temporalDisplayables.length; i_1 < len_1; i_1++) {
        var displayable = temporalDisplayables[i_1];
        displayable.beforeBrush && displayable.beforeBrush();
        displayable.innerBeforeBrush();
        brush(ctx, displayable, innerScope, i_1 === len_1 - 1);
        displayable.innerAfterBrush();
        displayable.afterBrush && displayable.afterBrush();
        innerScope.prevEl = displayable;
    }
    el.clearTemporalDisplayables();
    el.notClear = true;
    ctx.restore();
}
