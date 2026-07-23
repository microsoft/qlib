// TODO
// 1. shadow
// 2. Image: sx, sy, sw, sh

import {
    adjustTextY,
    getIdURL,
    getMatrixStr,
    getPathPrecision,
    getShadowKey,
    getSRTTransformString,
    hasShadow,
    isAroundZero,
    isGradient,
    isImagePattern,
    isLinearGradient,
    isPattern,
    isRadialGradient,
    normalizeColor,
    round4,
    TEXT_ALIGN_TO_ANCHOR
} from './helper';
import Path, { PathStyleProps } from '../graphic/Path';
import ZRImage, { ImageStyleProps } from '../graphic/Image';
import { getLineHeight } from '../contain/text';
import TSpan, { TSpanStyleProps } from '../graphic/TSpan';
import SVGPathRebuilder from './SVGPathRebuilder';
import mapStyleToAttrs from './mapStyleToAttrs';
import { SVGVNodeAttrs, createVNode, SVGVNode, vNodeToString, BrushScope, META_DATA_PREFIX } from './core';
import { MatrixArray } from '../core/matrix';
import Displayable from '../graphic/Displayable';
import { assert, clone, isFunction, isString, logError, map, retrieve2 } from '../core/util';
import Polyline from '../graphic/shape/Polyline';
import Polygon from '../graphic/shape/Polygon';
import { GradientObject } from '../graphic/Gradient';
import { ImagePatternObject, SVGPatternObject } from '../graphic/Pattern';
import { createOrUpdateImage } from '../graphic/helper/image';
import { ImageLike } from '../core/types';
import { createCSSAnimation } from './cssAnimation';
import { hasSeparateFont, parseFontSize } from '../graphic/Text';
import { DEFAULT_FONT, DEFAULT_FONT_FAMILY } from '../core/platform';
import { createCSSEmphasis } from './cssEmphasis';
import { getElementSSRData } from '../zrender';

const round = Math.round;

function isImageLike(val: any): val is HTMLImageElement {
    return val && isString(val.src);
}
function isCanvasLike(val: any): val is HTMLCanvasElement {
    return val && isFunction(val.toDataURL);
}


type AllStyleOption = PathStyleProps | TSpanStyleProps | ImageStyleProps;

function setStyleAttrs(attrs: SVGVNodeAttrs, style: AllStyleOption, el: Path | TSpan | ZRImage, scope: BrushScope) {
    mapStyleToAttrs((key, val) => {
        const isFillStroke = key === 'fill' || key === 'stroke';
        if (isFillStroke && isGradient(val)) {
            setGradient(style, attrs, key, scope);
        }
        else if (isFillStroke && isPattern(val)) {
            setPattern(el, attrs, key, scope);
        }
        else {
            attrs[key] = val;
        }
        if (isFillStroke && scope.ssr && val === 'none') {
            // When is none, it cannot be interacted when ssr
            // Setting `pointer-events` as `visible` to make it responding
            // See also https://www.w3.org/TR/SVG/interact.html#PointerEventsProperty
            attrs['pointer-events'] = 'visible';
        }
    }, style, el, false);

    setShadow(el, attrs, scope);
}

function setMetaData(attrs: SVGVNodeAttrs, el: Path | TSpan | ZRImage) {
    const metaData = getElementSSRData(el);
    if (metaData) {
        metaData.each((val, key) => {
            val != null && (attrs[(META_DATA_PREFIX + key).toLowerCase()] = val + '');
        });
        if (el.isSilent()) {
            attrs[META_DATA_PREFIX + 'silent'] = 'true';
        }
    }
}

function noRotateScale(m: MatrixArray) {
    return isAroundZero(m[0] - 1)
        && isAroundZero(m[1])
        && isAroundZero(m[2])
        && isAroundZero(m[3] - 1);
}

function noTranslate(m: MatrixArray) {
    return isAroundZero(m[4]) && isAroundZero(m[5]);
}

function setTransform(attrs: SVGVNodeAttrs, m: MatrixArray, compress?: boolean) {
    if (m && !(noTranslate(m) && noRotateScale(m))) {
        const mul = compress ? 10 : 1e4;
        // Use translate possible to reduce the size a bit.
        attrs.transform = noRotateScale(m)
            ? `translate(${round(m[4] * mul) / mul} ${round(m[5] * mul) / mul})` : getMatrixStr(m);
    }
}

type ShapeMapDesc = (string | [string, string])[];
type ConvertShapeToAttr = (shape: any, attrs: SVGVNodeAttrs, mul?: number) => void;
type ShapeValidator = (shape: any) => boolean;

function convertPolyShape(shape: Polygon['shape'], attrs: SVGVNodeAttrs, mul: number) {
    const points = shape.points;
    const strArr = [];
    for (let i = 0; i < points.length; i++) {
        strArr.push(round(points[i][0] * mul) / mul);
        strArr.push(round(points[i][1] * mul) / mul);
    }
    attrs.points = strArr.join(' ');
}

function validatePolyShape(shape: Polyline['shape']) {
    return !shape.smooth;
}

function createAttrsConvert(desc: ShapeMapDesc): ConvertShapeToAttr {
    const normalizedDesc: [string, string][] = map(desc, (item) =>
        (typeof item === 'string' ? [item, item] : item)
    );

    return function (shape, attrs, mul) {
        for (let i = 0; i < normalizedDesc.length; i++) {
            const item = normalizedDesc[i];
            const val = shape[item[0]];
            if (val != null) {
                attrs[item[1]] = round(val * mul) / mul;
            }
        }
    };
}

const builtinShapesDef: Record<string, [ConvertShapeToAttr, ShapeValidator?]> = {
    circle: [createAttrsConvert(['cx', 'cy', 'r'])],
    polyline: [convertPolyShape, validatePolyShape],
    polygon: [convertPolyShape, validatePolyShape]
    // Ignore line because it will be larger.
};

interface PathWithSVGBuildPath extends Path {
    __svgPathVersion: number
    __svgPathBuilder: SVGPathRebuilder
    __svgPathStrokePercent: number
}

function hasShapeAnimation(el: Displayable) {
    const animators = el.animators;
    for (let i = 0; i < animators.length; i++) {
        if (animators[i].targetName === 'shape') {
            return true;
        }
    }
    return false;
}

export function brushSVGPath(el: Path, scope: BrushScope) {
    const style = el.style;
    const shape = el.shape;
    const builtinShpDef = builtinShapesDef[el.type];
    const attrs: SVGVNodeAttrs = {};
    const needsAnimate = scope.animation;
    let svgElType = 'path';
    const strokePercent = el.style.strokePercent;
    const precision = (scope.compress && getPathPrecision(el)) || 4;
    // Using SVG builtin shapes if possible
    if (builtinShpDef
        // Force to use path if it will update later.
        // To avoid some animation(like morph) fail
        && !scope.willUpdate
        && !(builtinShpDef[1] && !builtinShpDef[1](shape))
        // use `path` to simplify the animate element creation logic.
        && !(needsAnimate && hasShapeAnimation(el))
        && !(strokePercent < 1)
    ) {
        svgElType = el.type;
        const mul = Math.pow(10, precision);
        builtinShpDef[0](shape, attrs, mul);
    }
    else {
        const needBuildPath = !el.path || el.shapeChanged();
        if (!el.path) {
            el.createPathProxy();
        }
        const path = el.path;

        if (needBuildPath) {
            path.beginPath();
            el.buildPath(path, el.shape);
            el.pathUpdated();
        }
        const pathVersion = path.getVersion();
        const elExt = el as PathWithSVGBuildPath;

        let svgPathBuilder = elExt.__svgPathBuilder;
        if (elExt.__svgPathVersion !== pathVersion
            || !svgPathBuilder
            || strokePercent !== elExt.__svgPathStrokePercent
        ) {
            if (!svgPathBuilder) {
                svgPathBuilder = elExt.__svgPathBuilder = new SVGPathRebuilder();
            }
            svgPathBuilder.reset(precision);
            path.rebuildPath(svgPathBuilder, strokePercent);
            svgPathBuilder.generateStr();
            elExt.__svgPathVersion = pathVersion;
            elExt.__svgPathStrokePercent = strokePercent;
        }

        attrs.d = svgPathBuilder.getStr();
    }

    setTransform(attrs, el.transform);
    setStyleAttrs(attrs, style, el, scope);
    setMetaData(attrs, el);

    scope.animation && createCSSAnimation(el, attrs, scope);
    scope.emphasis && createCSSEmphasis(el, attrs, scope);

    return createVNode(svgElType, el.id + '', attrs);
}

export function brushSVGImage(el: ZRImage, scope: BrushScope) {
    const style = el.style;
    let image = style.image;

    if (image && !isString(image)) {
        if (isImageLike(image)) {
            image = image.src;
        }
        // heatmap layer in geo may be a canvas
        else if (isCanvasLike(image)) {
            image = image.toDataURL();
        }
    }

    if (!image) {
        return;
    }

    const x = style.x || 0;
    const y = style.y || 0;

    const dw = style.width;
    const dh = style.height;

    const attrs: SVGVNodeAttrs = {
        href: image as string,
        width: dw,
        height: dh
    };
    if (x) {
        attrs.x = x;
    }
    if (y) {
        attrs.y = y;
    }

    setTransform(attrs, el.transform);
    setStyleAttrs(attrs, style, el, scope);
    setMetaData(attrs, el);

    scope.animation && createCSSAnimation(el, attrs, scope);

    return createVNode('image', el.id + '', attrs);
};

export function brushSVGTSpan(el: TSpan, scope: BrushScope) {
    const style = el.style;

    let text = style.text;
    // Convert to string
    text != null && (text += '');
    if (!text || isNaN(style.x) || isNaN(style.y)) {
        return;
    }

    // style.font has been normalized by `normalizeTextStyle`.
    const font = style.font || DEFAULT_FONT;

    // Consider different font display differently in vertical align, we always
    // set verticalAlign as 'middle', and use 'y' to locate text vertically.
    const x = style.x || 0;
    const y = adjustTextY(style.y || 0, getLineHeight(font), style.textBaseline);
    const textAlign = TEXT_ALIGN_TO_ANCHOR[style.textAlign as keyof typeof TEXT_ALIGN_TO_ANCHOR]
        || style.textAlign;

    const attrs: SVGVNodeAttrs = {
        'dominant-baseline': 'central',
        'text-anchor': textAlign
    };

    if (hasSeparateFont(style)) {
        // Set separate font attributes if possible. Or some platform like PowerPoint may not support it.
        let separatedFontStr = '';
        const fontStyle = style.fontStyle;
        const fontSize = parseFontSize(style.fontSize);
        if (!parseFloat(fontSize)) {    // is 0px
            return;
        }

        const fontFamily = style.fontFamily || DEFAULT_FONT_FAMILY;
        const fontWeight = style.fontWeight;
        separatedFontStr += `font-size:${fontSize};font-family:${fontFamily};`;

        // TODO reduce the attribute to set. But should it inherit from the container element?
        if (fontStyle && fontStyle !== 'normal') {
            separatedFontStr += `font-style:${fontStyle};`;
        }
        if (fontWeight && fontWeight !== 'normal') {
            separatedFontStr += `font-weight:${fontWeight};`;
        }
        attrs.style = separatedFontStr;
    }
    else {
        // Use set font manually
        attrs.style = `font: ${font}`;
    }


    if (text.match(/\s/)) {
        // only enabled when have space in text.
        attrs['xml:space'] = 'preserve';
    }
    if (x) {
        attrs.x = x;
    }
    if (y) {
        attrs.y = y;
    }
    setTransform(attrs, el.transform);
    setStyleAttrs(attrs, style, el, scope);
    setMetaData(attrs, el);

    scope.animation && createCSSAnimation(el, attrs, scope);

    return createVNode('text', el.id + '', attrs, undefined, text);
}

export function brush(el: Displayable, scope: BrushScope): SVGVNode {
    if (el instanceof Path) {
        return brushSVGPath(el, scope);
    }
    else if (el instanceof ZRImage) {
        return brushSVGImage(el, scope);
    }
    else if (el instanceof TSpan) {
        return brushSVGTSpan(el, scope);
    }
}

function setShadow(
    el: Displayable,
    attrs: SVGVNodeAttrs,
    scope: BrushScope
) {
    const style = el.style;
    if (hasShadow(style)) {
        const shadowKey = getShadowKey(el);
        const shadowCache = scope.shadowCache;
        let shadowId = shadowCache[shadowKey];
        if (!shadowId) {
            const globalScale = el.getGlobalScale();
            const scaleX = globalScale[0];
            const scaleY = globalScale[1];
            if (!scaleX || !scaleY) {
                return;
            }

            const offsetX = style.shadowOffsetX || 0;
            const offsetY = style.shadowOffsetY || 0;
            const blur = style.shadowBlur;
            const {opacity, color} = normalizeColor(style.shadowColor);
            const stdDx = blur / 2 / scaleX;
            const stdDy = blur / 2 / scaleY;
            const stdDeviation = stdDx + ' ' + stdDy;
            // Use a simple prefix to reduce the size
            shadowId = scope.zrId + '-s' + scope.shadowIdx++;
            scope.defs[shadowId] = createVNode(
                'filter', shadowId,
                {
                    'id': shadowId,
                    'x': '-100%',
                    'y': '-100%',
                    'width': '300%',
                    'height': '300%'
                },
                [
                    createVNode('feDropShadow', '', {
                        'dx': offsetX / scaleX,
                        'dy': offsetY / scaleY,
                        'stdDeviation': stdDeviation,
                        'flood-color': color,
                        'flood-opacity': opacity
                    })
                ]
            );
            shadowCache[shadowKey] = shadowId;
        }
        attrs.filter = getIdURL(shadowId);
    }
}

export function setGradient(
    style: PathStyleProps,
    attrs: SVGVNodeAttrs,
    target: 'fill' | 'stroke',
    scope: BrushScope
) {
    const val = style[target] as GradientObject;
    let gradientTag;
    let gradientAttrs: SVGVNodeAttrs = {
        'gradientUnits': val.global
            ? 'userSpaceOnUse' // x1, x2, y1, y2 in range of 0 to canvas width or height
            : 'objectBoundingBox' // x1, x2, y1, y2 in range of 0 to 1]
    };
    if (isLinearGradient(val)) {
        gradientTag = 'linearGradient';
        gradientAttrs.x1 = val.x;
        gradientAttrs.y1 = val.y;
        gradientAttrs.x2 = val.x2;
        gradientAttrs.y2 = val.y2;
    }
    else if (isRadialGradient(val)) {
        gradientTag = 'radialGradient';
        gradientAttrs.cx = retrieve2(val.x, 0.5);
        gradientAttrs.cy = retrieve2(val.y, 0.5);
        gradientAttrs.r = retrieve2(val.r, 0.5);
    }
    else {
        if (process.env.NODE_ENV !== 'production') {
            logError('Illegal gradient type.');
        }
        return;
    }

    const colors = val.colorStops;

    const colorStops = [];
    for (let i = 0, len = colors.length; i < len; ++i) {
        const offset = round4(colors[i].offset) * 100 + '%';

        const stopColor = colors[i].color;
        // Fix Safari bug that stop-color not recognizing alpha #9014
        const {color, opacity} = normalizeColor(stopColor);

        const stopsAttrs: SVGVNodeAttrs = {
            'offset': offset
        };
        // stop-color cannot be color, since:
        // The opacity value used for the gradient calculation is the
        // *product* of the value of stop-opacity and the opacity of the
        // value of stop-color.
        // See https://www.w3.org/TR/SVG2/pservers.html#StopOpacityProperty

        stopsAttrs['stop-color'] = color;
        if (opacity < 1) {
            stopsAttrs['stop-opacity'] = opacity;
        }
        colorStops.push(
            createVNode('stop', i + '', stopsAttrs)
        );
    }

    // Use the whole html as cache key.
    const gradientVNode = createVNode(gradientTag, '', gradientAttrs, colorStops);
    const gradientKey = vNodeToString(gradientVNode);
    const gradientCache = scope.gradientCache;
    let gradientId = gradientCache[gradientKey];
    if (!gradientId) {
        gradientId = scope.zrId + '-g' + scope.gradientIdx++;
        gradientCache[gradientKey] = gradientId;

        gradientAttrs.id = gradientId;
        scope.defs[gradientId] = createVNode(
            gradientTag, gradientId, gradientAttrs, colorStops
        );
    }

    attrs[target] = getIdURL(gradientId);
}

export function setPattern(
    el: Displayable,
    attrs: SVGVNodeAttrs,
    target: 'fill' | 'stroke',
    scope: BrushScope
) {
    const val = el.style[target] as ImagePatternObject | SVGPatternObject;
    const boundingRect = el.getBoundingRect();
    const patternAttrs: SVGVNodeAttrs = {};
    const repeat = (val as ImagePatternObject).repeat;
    const noRepeat = repeat === 'no-repeat';
    const repeatX = repeat === 'repeat-x';
    const repeatY = repeat === 'repeat-y';
    let child: SVGVNode;
    if (isImagePattern(val)) {
        let imageWidth = val.imageWidth;
        let imageHeight = val.imageHeight;
        let imageSrc;
        const patternImage = val.image;
        if (isString(patternImage)) {
            imageSrc = patternImage;
        }
        else if (isImageLike(patternImage)) {
            imageSrc = patternImage.src;
        }
        else if (isCanvasLike(patternImage)) {
            imageSrc = patternImage.toDataURL();
        }

        if (typeof Image === 'undefined') {
            const errMsg = 'Image width/height must been given explictly in svg-ssr renderer.';
            assert(imageWidth, errMsg);
            assert(imageHeight, errMsg);
        }
        else if (imageWidth == null || imageHeight == null) {
            // TODO
            const setSizeToVNode = (vNode: SVGVNode, img: ImageLike) => {
                if (vNode) {
                    const svgEl = vNode.elm as SVGElement;
                    let width = imageWidth || img.width;
                    let height = imageHeight || img.height;
                    if (vNode.tag === 'pattern') {
                        if (repeatX) {
                            height = 1;
                            width /= boundingRect.width;
                        }
                        else if (repeatY) {
                            width = 1;
                            height /= boundingRect.height;
                        }
                    }
                    vNode.attrs.width = width;
                    vNode.attrs.height = height;
                    if (svgEl) {
                        svgEl.setAttribute('width', width as any);
                        svgEl.setAttribute('height', height as any);
                    }
                }
            };
            const createdImage = createOrUpdateImage(
                imageSrc, null, el, (img) => {
                    noRepeat || setSizeToVNode(patternVNode, img);
                    setSizeToVNode(child, img);
                }
            );
            if (createdImage && createdImage.width && createdImage.height) {
                // Loaded before
                imageWidth = imageWidth || createdImage.width;
                imageHeight = imageHeight || createdImage.height;
            }
        }

        child = createVNode(
            'image',
            'img',
            {
                href: imageSrc,
                width: imageWidth,
                height: imageHeight
            }
        );
        patternAttrs.width = imageWidth;
        patternAttrs.height = imageHeight;
    }
    else if (val.svgElement) {  // Only string supported in SSR.
        // TODO it's not so good to use textContent as innerHTML
        child = clone(val.svgElement);
        patternAttrs.width = val.svgWidth;
        patternAttrs.height = val.svgHeight;
    }
    if (!child) {
        return;
    }

    let patternWidth;
    let patternHeight;
    if (noRepeat) {
        patternWidth = patternHeight = 1;
    }
    else if (repeatX) {
        patternHeight = 1;
        patternWidth = (patternAttrs.width as number) / boundingRect.width;
    }
    else if (repeatY) {
        patternWidth = 1;
        patternHeight = (patternAttrs.height as number) / boundingRect.height;
    }
    else {
        patternAttrs.patternUnits = 'userSpaceOnUse';
    }

    if (patternWidth != null && !isNaN(patternWidth)) {
        patternAttrs.width = patternWidth;
    }
    if (patternHeight != null && !isNaN(patternHeight)) {
        patternAttrs.height = patternHeight;
    }

    const patternTransform = getSRTTransformString(val);
    patternTransform && (patternAttrs.patternTransform = patternTransform);

    // Use the whole html as cache key.
    let patternVNode = createVNode(
        'pattern',
        '',
        patternAttrs,
        [child]
    );
    const patternKey = vNodeToString(patternVNode);
    const patternCache = scope.patternCache;
    let patternId = patternCache[patternKey];
    if (!patternId) {
        patternId = scope.zrId + '-p' + scope.patternIdx++;
        patternCache[patternKey] = patternId;
        patternAttrs.id = patternId;
        patternVNode = scope.defs[patternId] = createVNode(
            'pattern',
            patternId,
            patternAttrs,
            [child]
        );
    }

    attrs[target] = getIdURL(patternId);
}

export function setClipPath(
    clipPath: Path,
    attrs: SVGVNodeAttrs,
    scope: BrushScope
) {
    const {clipPathCache, defs} = scope;
    let clipPathId = clipPathCache[clipPath.id];
    if (!clipPathId) {
        clipPathId = scope.zrId + '-c' + scope.clipPathIdx++;
        const clipPathAttrs: SVGVNodeAttrs = {
            id: clipPathId
        };

        clipPathCache[clipPath.id] = clipPathId;
        defs[clipPathId] = createVNode(
            'clipPath', clipPathId, clipPathAttrs,
            [brushSVGPath(clipPath, scope)]
        );
    }
    attrs['clip-path'] = getIdURL(clipPathId);
}
