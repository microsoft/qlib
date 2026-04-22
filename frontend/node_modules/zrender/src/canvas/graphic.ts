import Displayable, { DEFAULT_COMMON_STYLE } from '../graphic/Displayable';
import PathProxy from '../core/PathProxy';
import { GradientObject } from '../graphic/Gradient';
import { ImagePatternObject, InnerImagePatternObject } from '../graphic/Pattern';
import { LinearGradientObject } from '../graphic/LinearGradient';
import { RadialGradientObject } from '../graphic/RadialGradient';
import { ZRCanvasRenderingContext } from '../core/types';
import { createOrUpdateImage, isImageReady } from '../graphic/helper/image';
import { getCanvasGradient, isClipPathChanged } from './helper';
import Path, { PathStyleProps } from '../graphic/Path';
import ZRImage, { ImageStyleProps } from '../graphic/Image';
import TSpan, {TSpanStyleProps} from '../graphic/TSpan';
import { MatrixArray } from '../core/matrix';
import { RADIAN_TO_DEGREE } from '../core/util';
import { getLineDash } from './dashStyle';
import { REDRAW_BIT, SHAPE_CHANGED_BIT } from '../graphic/constants';
import type IncrementalDisplayable from '../graphic/IncrementalDisplayable';
import { DEFAULT_FONT } from '../core/platform';

const pathProxyForDraw = new PathProxy(true);

// Not use el#hasStroke because style may be different.
function styleHasStroke(style: PathStyleProps) {
    const stroke = style.stroke;
    return !(stroke == null || stroke === 'none' || !(style.lineWidth > 0));
}

// ignore lineWidth and must be string
// Expected color but found '[' when color is gradient
function isValidStrokeFillStyle(
    strokeOrFill: PathStyleProps['stroke'] | PathStyleProps['fill']
): strokeOrFill is string {
    return typeof strokeOrFill === 'string' && strokeOrFill !== 'none';
}

function styleHasFill(style: PathStyleProps) {
    const fill = style.fill;
    return fill != null && fill !== 'none';
}
function doFillPath(ctx: CanvasRenderingContext2D, style: PathStyleProps) {
    if (style.fillOpacity != null && style.fillOpacity !== 1) {
        const originalGlobalAlpha = ctx.globalAlpha;
        ctx.globalAlpha = style.fillOpacity * style.opacity;
        ctx.fill();
        // Set back globalAlpha
        ctx.globalAlpha = originalGlobalAlpha;
    }
    else {
        ctx.fill();
    }
}

function doStrokePath(ctx: CanvasRenderingContext2D, style: PathStyleProps) {
    if (style.strokeOpacity != null && style.strokeOpacity !== 1) {
        const originalGlobalAlpha = ctx.globalAlpha;
        ctx.globalAlpha = style.strokeOpacity * style.opacity;
        ctx.stroke();
        // Set back globalAlpha
        ctx.globalAlpha = originalGlobalAlpha;
    }
    else {
        ctx.stroke();
    }
}

export function createCanvasPattern(
    this: void,
    ctx: CanvasRenderingContext2D,
    pattern: ImagePatternObject,
    el: {dirty: () => void}
): CanvasPattern {
    const image = createOrUpdateImage(pattern.image, (pattern as InnerImagePatternObject).__image, el);
    if (isImageReady(image)) {
        const canvasPattern = ctx.createPattern(image, pattern.repeat || 'repeat');
        if (
            typeof DOMMatrix === 'function'
            && canvasPattern                // image may be not ready
            && canvasPattern.setTransform   // setTransform may not be supported in some old devices.
        ) {
            const matrix = new DOMMatrix();
            matrix.translateSelf((pattern.x || 0), (pattern.y || 0));
            matrix.rotateSelf(0, 0, (pattern.rotation || 0) * RADIAN_TO_DEGREE);
            matrix.scaleSelf((pattern.scaleX || 1), (pattern.scaleY || 1));
            canvasPattern.setTransform(matrix);
        }
        return canvasPattern;
    }
}

// Draw Path Elements
function brushPath(ctx: CanvasRenderingContext2D, el: Path, style: PathStyleProps, inBatch: boolean) {
    let hasStroke = styleHasStroke(style);
    let hasFill = styleHasFill(style);

    const strokePercent = style.strokePercent;
    const strokePart = strokePercent < 1;

    // TODO Reduce path memory cost.
    const firstDraw = !el.path;
    // Create path for each element when:
    // 1. Element has interactions.
    // 2. Element draw part of the line.
    if ((!el.silent || strokePart) && firstDraw) {
        el.createPathProxy();
    }

    const path = el.path || pathProxyForDraw;
    const dirtyFlag = el.__dirty;

    if (!inBatch) {
        const fill = style.fill;
        const stroke = style.stroke;

        const hasFillGradient = hasFill && !!(fill as GradientObject).colorStops;
        const hasStrokeGradient = hasStroke && !!(stroke as GradientObject).colorStops;
        const hasFillPattern = hasFill && !!(fill as ImagePatternObject).image;
        const hasStrokePattern = hasStroke && !!(stroke as ImagePatternObject).image;

        let fillGradient;
        let strokeGradient;
        let fillPattern;
        let strokePattern;
        let rect;
        if (hasFillGradient || hasStrokeGradient) {
            rect = el.getBoundingRect();
        }

        // Update gradient because bounding rect may changed
        if (hasFillGradient) {
            fillGradient = dirtyFlag
                ? getCanvasGradient(ctx, fill as (LinearGradientObject | RadialGradientObject), rect)
                : el.__canvasFillGradient;
            // No need to clear cache when fill is not gradient.
            // It will always been updated when fill changed back to gradient.
            el.__canvasFillGradient = fillGradient;
        }
        if (hasStrokeGradient) {
            strokeGradient = dirtyFlag
                ? getCanvasGradient(ctx, stroke as (LinearGradientObject | RadialGradientObject), rect)
                : el.__canvasStrokeGradient;
            el.__canvasStrokeGradient = strokeGradient;
        }
        if (hasFillPattern) {
            // Pattern might be null if image not ready (even created from dataURI)
            fillPattern = (dirtyFlag || !el.__canvasFillPattern)
                ? createCanvasPattern(ctx, fill as ImagePatternObject, el)
                : el.__canvasFillPattern;
            el.__canvasFillPattern = fillPattern;
        }
        if (hasStrokePattern) {
            // Pattern might be null if image not ready (even created from dataURI)
            strokePattern = (dirtyFlag || !el.__canvasStrokePattern)
                ? createCanvasPattern(ctx, stroke as ImagePatternObject, el)
                : el.__canvasStrokePattern;
            el.__canvasStrokePattern = fillPattern;
        }
        // Use the gradient or pattern
        if (hasFillGradient) {
            // PENDING If may have affect the state
            ctx.fillStyle = fillGradient;
        }
        else if (hasFillPattern) {
            if (fillPattern) {  // createCanvasPattern may return false if image is not ready.
                ctx.fillStyle = fillPattern;
            }
            else {
                // Don't fill if image is not ready
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
                // Don't stroke if image is not ready
                hasStroke = false;
            }
        }
    }

    // Update path sx, sy
    const scale = el.getGlobalScale();
    path.setScale(scale[0], scale[1], el.segmentIgnoreThreshold);

    let lineDash;
    let lineDashOffset;
    if (ctx.setLineDash && style.lineDash) {
        [lineDash, lineDashOffset] = getLineDash(el);
    }

    let needsRebuild = true;

    if (firstDraw || (dirtyFlag & SHAPE_CHANGED_BIT)) {
        path.setDPR((ctx as any).dpr);
        if (strokePart) {
            // Use rebuildPath for percent stroke, so no context.
            path.setContext(null);
        }
        else {
            path.setContext(ctx);
            needsRebuild = false;
        }
        path.reset();

        el.buildPath(path, el.shape, inBatch);
        path.toStatic();

        // Clear path dirty flag
        el.pathUpdated();
    }

    // Not support separate fill and stroke. For the compatibility of SVG
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
        // PENDING
        // Remove lineDash
        ctx.setLineDash([]);
    }
}

// Draw Image Elements
function brushImage(ctx: CanvasRenderingContext2D, el: ZRImage, style: ImageStyleProps) {
    const image = el.__image = createOrUpdateImage(
        style.image,
        el.__image,
        el,
        el.onload
    );

    if (!image || !isImageReady(image)) {
        return;
    }

    const x = style.x || 0;
    const y = style.y || 0;
    let width = el.getWidth();
    let height = el.getHeight();
    const aspect = image.width / image.height;
    if (width == null && height != null) {
        // Keep image/height ratio
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
        const sx = style.sx || 0;
        const sy = style.sy || 0;
        ctx.drawImage(
            image,
            sx, sy, style.sWidth, style.sHeight,
            x, y, width, height
        );
    }
    else if (style.sx && style.sy) {
        const sx = style.sx;
        const sy = style.sy;
        const sWidth = width - sx;
        const sHeight = height - sy;
        ctx.drawImage(
            image,
            sx, sy, sWidth, sHeight,
            x, y, width, height
        );
    }
    else {
        ctx.drawImage(image, x, y, width, height);
    }
}

// Draw Text Elements
function brushText(ctx: CanvasRenderingContext2D, el: TSpan, style: TSpanStyleProps) {

    let text = style.text;
    // Convert to string
    text != null && (text += '');

    if (text) {
        ctx.font = style.font || DEFAULT_FONT;
        ctx.textAlign = style.textAlign;
        ctx.textBaseline = style.textBaseline;

        let lineDash;
        let lineDashOffset;
        if (ctx.setLineDash && style.lineDash) {
            [lineDash, lineDashOffset] = getLineDash(el);
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
            // Remove lineDash
            ctx.setLineDash([]);
        }
    }

}

const SHADOW_NUMBER_PROPS = ['shadowBlur', 'shadowOffsetX', 'shadowOffsetY'] as const;
const STROKE_PROPS = [
    ['lineCap', 'butt'], ['lineJoin', 'miter'], ['miterLimit', 10]
] as const;

type AllStyleOption = PathStyleProps | TSpanStyleProps | ImageStyleProps;
// type ShadowPropNames = typeof SHADOW_PROPS[number][0];
// type StrokePropNames = typeof STROKE_PROPS[number][0];
// type DrawPropNames = typeof DRAW_PROPS[number][0];

function bindCommonProps(
    ctx: CanvasRenderingContext2D,
    style: AllStyleOption,
    prevStyle: AllStyleOption,
    forceSetAll: boolean,
    scope: BrushScope
): boolean {
    let styleChanged = false;

    if (!forceSetAll) {
        prevStyle = prevStyle || {};

        // Shared same style.
        if (style === prevStyle) {
            return false;
        }
    }
    if (forceSetAll || style.opacity !== prevStyle.opacity) {
        flushPathDrawn(ctx, scope);
        styleChanged = true;
        // Ensure opacity is between 0 ~ 1. Invalid opacity will lead to a failure set and use the leaked opacity from the previous.
        const opacity = Math.max(Math.min(style.opacity, 1), 0);
        ctx.globalAlpha = isNaN(opacity) ? DEFAULT_COMMON_STYLE.opacity : opacity;
    }

    if (forceSetAll || style.blend !== prevStyle.blend) {
        if (!styleChanged) {
            flushPathDrawn(ctx, scope);
            styleChanged = true;
        }
        ctx.globalCompositeOperation = style.blend || DEFAULT_COMMON_STYLE.blend;
    }
    for (let i = 0; i < SHADOW_NUMBER_PROPS.length; i++) {
        const propName = SHADOW_NUMBER_PROPS[i];
        if (forceSetAll || style[propName] !== prevStyle[propName]) {
            if (!styleChanged) {
                flushPathDrawn(ctx, scope);
                styleChanged = true;
            }
            // FIXME Invalid property value will cause style leak from previous element.
            ctx[propName] = (ctx as ZRCanvasRenderingContext).dpr * (style[propName] || 0);
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

function bindPathAndTextCommonStyle(
    ctx: CanvasRenderingContext2D,
    el: TSpan | Path,
    prevEl: TSpan | Path,
    forceSetAll: boolean,
    scope: BrushScope
) {
    const style = getStyle(el, scope.inHover);
    const prevStyle = forceSetAll
        ? null
        : (prevEl && getStyle(prevEl, scope.inHover) || {});
    // Shared same style. prevStyle will be null if forceSetAll.
    if (style === prevStyle) {
        return false;
    }

    let styleChanged = bindCommonProps(ctx, style, prevStyle, forceSetAll, scope);

    if (forceSetAll || style.fill !== prevStyle.fill) {
        if (!styleChanged) {
            // Flush before set
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
        const lineWidth = style.lineWidth;
        const newLineWidth = lineWidth / (
            (style.strokeNoScale && el.getLineScale) ? el.getLineScale() : 1
        );
        if (ctx.lineWidth !== newLineWidth) {
            if (!styleChanged) {
                flushPathDrawn(ctx, scope);
                styleChanged = true;
            }
            ctx.lineWidth = newLineWidth;
        }
    }

    for (let i = 0; i < STROKE_PROPS.length; i++) {
        const prop = STROKE_PROPS[i];
        const propName = prop[0];
        if (forceSetAll || style[propName] !== prevStyle[propName]) {
            if (!styleChanged) {
                flushPathDrawn(ctx, scope);
                styleChanged = true;
            }
            // FIXME Invalid property value will cause style leak from previous element.
            (ctx as any)[propName] = style[propName] || prop[1];
        }
    }

    return styleChanged;
}

function bindImageStyle(
    ctx: CanvasRenderingContext2D,
    el: ZRImage,
    prevEl: ZRImage,
    // forceSetAll must be true if prevEl is null
    forceSetAll: boolean,
    scope: BrushScope
) {
    return bindCommonProps(
        ctx,
        getStyle(el, scope.inHover),
        prevEl && getStyle(prevEl, scope.inHover),
        forceSetAll,
        scope
    );
}

function setContextTransform(ctx: CanvasRenderingContext2D, el: Displayable) {
    const m = el.transform;
    const dpr = (ctx as ZRCanvasRenderingContext).dpr || 1;
    if (m) {
        ctx.setTransform(dpr * m[0], dpr * m[1], dpr * m[2], dpr * m[3], dpr * m[4], dpr * m[5]);
    }
    else {
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
}

function updateClipStatus(clipPaths: Path[], ctx: CanvasRenderingContext2D, scope: BrushScope) {
    let allClipped = false;
    for (let i = 0; i < clipPaths.length; i++) {
        const clipPath = clipPaths[i];
        // Ignore draw following elements if clipPath has zero area.
        allClipped = allClipped || clipPath.isZeroArea();

        setContextTransform(ctx, clipPath);
        ctx.beginPath();
        clipPath.buildPath(ctx, clipPath.shape);
        ctx.clip();
    }
    scope.allClipped = allClipped;
}

function isTransformChanged(m0: MatrixArray, m1: MatrixArray): boolean {
    if (m0 && m1) {
        return m0[0] !== m1[0]
            || m0[1] !== m1[1]
            || m0[2] !== m1[2]
            || m0[3] !== m1[3]
            || m0[4] !== m1[4]
            || m0[5] !== m1[5];
    }
    else if (!m0 && !m1) {  // All identity matrix.
        return false;
    }

    return true;
}

const DRAW_TYPE_PATH = 1;
const DRAW_TYPE_IMAGE = 2;
const DRAW_TYPE_TEXT = 3;
const DRAW_TYPE_INCREMENTAL = 4;

export type BrushScope = {
    inHover: boolean

    // width / height of viewport
    viewWidth: number
    viewHeight: number

    // Status for clipping
    prevElClipPaths?: Path[]
    prevEl?: Displayable
    allClipped?: boolean    // If the whole element can be clipped

    // Status for batching
    batchFill?: string
    batchStroke?: string

    lastDrawType?: number
}

// If path can be batched
function canPathBatch(style: PathStyleProps) {

    const hasFill = styleHasFill(style);
    const hasStroke = styleHasStroke(style);

    return !(
        // Line dash is dynamically set in brush function.
        style.lineDash
        // Can't batch if element is both set fill and stroke. Or both not set
        || !(+hasFill ^ +hasStroke)
        // Can't batch if element is drawn with gradient or pattern.
        || (hasFill && typeof style.fill !== 'string')
        || (hasStroke && typeof style.stroke !== 'string')
        // Can't batch if element only stroke part of line.
        || style.strokePercent < 1
        // Has stroke or fill opacity
        || style.strokeOpacity < 1
        || style.fillOpacity < 1
    );
}

function flushPathDrawn(ctx: CanvasRenderingContext2D, scope: BrushScope) {
    // Force flush all after drawn last element
    scope.batchFill && ctx.fill();
    scope.batchStroke && ctx.stroke();
    scope.batchFill = '';
    scope.batchStroke = '';
}

function getStyle(el: Displayable, inHover?: boolean) {
    return inHover ? (el.__hoverStyle || el.style) : el.style;
}

export function brushSingle(ctx: CanvasRenderingContext2D, el: Displayable) {
    brush(ctx, el, { inHover: false, viewWidth: 0, viewHeight: 0 }, true);
}

// Brush different type of elements.
export function brush(
    ctx: CanvasRenderingContext2D,
    el: Displayable,
    scope: BrushScope,
    isLast: boolean
) {
    const m = el.transform;

    if (!el.shouldBePainted(scope.viewWidth, scope.viewHeight, false, false)) {
        // Needs to mark el rendered.
        // Or this element will always been rendered in progressive rendering.
        // But other dirty bit should not be cleared, otherwise it cause the shape
        // can not be updated in this case.
        el.__dirty &= ~REDRAW_BIT;
        el.__isRendered = false;
        return;
    }

    // HANDLE CLIPPING
    const clipPaths = el.__clipPaths;
    const prevElClipPaths = scope.prevElClipPaths;

    let forceSetTransform = false;
    let forceSetStyle = false;
    // Optimize when clipping on group with several elements
    if (!prevElClipPaths || isClipPathChanged(clipPaths, prevElClipPaths)) {
        // If has previous clipping state, restore from it
        if (prevElClipPaths && prevElClipPaths.length) {
            // Flush restore
            flushPathDrawn(ctx, scope);

            ctx.restore();
            // Must set all style and transform because context changed by restore
            forceSetStyle = forceSetTransform = true;

            scope.prevElClipPaths = null;
            scope.allClipped = false;
            // Reset prevEl since context has been restored
            scope.prevEl = null;
        }
        // New clipping state
        if (clipPaths && clipPaths.length) {
            // Flush before clip
            flushPathDrawn(ctx, scope);

            ctx.save();
            updateClipStatus(clipPaths, ctx, scope);
            // Must set transform because it's changed when clip.
            forceSetTransform = true;
        }
        scope.prevElClipPaths = clipPaths;
    }

    // Not rendering elements if it's clipped by a zero area path.
    // Or it may cause bug on some version of IE11 (like 11.0.9600.178**),
    // where exception "unexpected call to method or property access"
    // might be thrown when calling ctx.fill or ctx.stroke after a path
    // whose area size is zero is drawn and ctx.clip() is called and
    // shadowBlur is set. See #4572, #3112, #5777.
    // (e.g.,
    //  ctx.moveTo(10, 10);
    //  ctx.lineTo(20, 10);
    //  ctx.closePath();
    //  ctx.clip();
    //  ctx.shadowBlur = 10;
    //  ...
    //  ctx.fill();
    // )
    if (scope.allClipped) {
        el.__isRendered = false;
        return;
    }

    // START BRUSH
    el.beforeBrush && el.beforeBrush();
    el.innerBeforeBrush();

    const prevEl = scope.prevEl;
    // TODO el type changed.
    if (!prevEl) {
        forceSetStyle = forceSetTransform = true;
    }

    let canBatchPath = el instanceof Path   // Only path supports batch
        && el.autoBatch
        && canPathBatch(el.style);

    if (forceSetTransform || isTransformChanged(m, prevEl.transform)) {
        // Flush
        flushPathDrawn(ctx, scope);
        setContextTransform(ctx, el);
    }
    else if (!canBatchPath) {
        // Flush
        flushPathDrawn(ctx, scope);
    }

    const style = getStyle(el, scope.inHover);
    if (el instanceof Path) {
        // PENDING do we need to rebind all style if displayable type changed?
        if (scope.lastDrawType !== DRAW_TYPE_PATH) {
            forceSetStyle = true;
            scope.lastDrawType = DRAW_TYPE_PATH;
        }

        bindPathAndTextCommonStyle(ctx, el as Path, prevEl as Path, forceSetStyle, scope);
        // Begin path at start
        if (!canBatchPath || (!scope.batchFill && !scope.batchStroke)) {
            ctx.beginPath();
        }
        brushPath(ctx, el as Path, style, canBatchPath);

        if (canBatchPath) {
            scope.batchFill = style.fill as string || '';
            scope.batchStroke = style.stroke as string || '';
        }
    }
    else {
        if (el instanceof TSpan) {
            if (scope.lastDrawType !== DRAW_TYPE_TEXT) {
                forceSetStyle = true;
                scope.lastDrawType = DRAW_TYPE_TEXT;
            }

            bindPathAndTextCommonStyle(ctx, el as TSpan, prevEl as TSpan, forceSetStyle, scope);
            brushText(ctx, el as TSpan, style);
        }
        else if (el instanceof ZRImage) {
            if (scope.lastDrawType !== DRAW_TYPE_IMAGE) {
                forceSetStyle = true;
                scope.lastDrawType = DRAW_TYPE_IMAGE;
            }

            bindImageStyle(ctx, el as ZRImage, prevEl as ZRImage, forceSetStyle, scope);
            brushImage(ctx, el as ZRImage, style);
        }
        // Assume it's a IncrementalDisplayable
        else if ((el as IncrementalDisplayable).getTemporalDisplayables) {
            if (scope.lastDrawType !== DRAW_TYPE_INCREMENTAL) {
                forceSetStyle = true;
                scope.lastDrawType = DRAW_TYPE_INCREMENTAL;
            }

            brushIncremental(ctx, el as IncrementalDisplayable, scope);
        }

    }

    if (canBatchPath && isLast) {
        flushPathDrawn(ctx, scope);
    }

    el.innerAfterBrush();
    el.afterBrush && el.afterBrush();

    scope.prevEl = el;

    // Mark as painted.
    el.__dirty = 0;
    el.__isRendered = true;
}

function brushIncremental(
    ctx: CanvasRenderingContext2D,
    el: IncrementalDisplayable,
    scope: BrushScope
) {
    let displayables = el.getDisplayables();
    let temporalDisplayables = el.getTemporalDisplayables();
    // Provide an inner scope.
    // Save current context and restore after brushed.
    ctx.save();
    let innerScope: BrushScope = {
        prevElClipPaths: null,
        prevEl: null,
        allClipped: false,
        viewWidth: scope.viewWidth,
        viewHeight: scope.viewHeight,
        inHover: scope.inHover
    };
    let i;
    let len;
    // Render persistant displayables.
    for (i = el.getCursor(), len = displayables.length; i < len; i++) {
        const displayable = displayables[i];
        displayable.beforeBrush && displayable.beforeBrush();
        displayable.innerBeforeBrush();
        brush(ctx, displayable, innerScope, i === len - 1);
        displayable.innerAfterBrush();
        displayable.afterBrush && displayable.afterBrush();
        innerScope.prevEl = displayable;
    }
    // Render temporary displayables.
    for (let i = 0, len = temporalDisplayables.length; i < len; i++) {
        const displayable = temporalDisplayables[i];
        displayable.beforeBrush && displayable.beforeBrush();
        displayable.innerBeforeBrush();
        brush(ctx, displayable, innerScope, i === len - 1);
        displayable.innerAfterBrush();
        displayable.afterBrush && displayable.afterBrush();
        innerScope.prevEl = displayable;
    }
    el.clearTemporalDisplayables();
    el.notClear = true;

    ctx.restore();
}