import { adjustTextY, getIdURL, getMatrixStr, getPathPrecision, getShadowKey, getSRTTransformString, hasShadow, isAroundZero, isGradient, isImagePattern, isLinearGradient, isPattern, isRadialGradient, normalizeColor, round4, TEXT_ALIGN_TO_ANCHOR } from './helper.js';
import Path from '../graphic/Path.js';
import ZRImage from '../graphic/Image.js';
import { getLineHeight } from '../contain/text.js';
import TSpan from '../graphic/TSpan.js';
import SVGPathRebuilder from './SVGPathRebuilder.js';
import mapStyleToAttrs from './mapStyleToAttrs.js';
import { createVNode, vNodeToString, META_DATA_PREFIX } from './core.js';
import { assert, clone, isFunction, isString, logError, map, retrieve2 } from '../core/util.js';
import { createOrUpdateImage } from '../graphic/helper/image.js';
import { createCSSAnimation } from './cssAnimation.js';
import { hasSeparateFont, parseFontSize } from '../graphic/Text.js';
import { DEFAULT_FONT, DEFAULT_FONT_FAMILY } from '../core/platform.js';
import { createCSSEmphasis } from './cssEmphasis.js';
import { getElementSSRData } from '../zrender.js';
var round = Math.round;
function isImageLike(val) {
    return val && isString(val.src);
}
function isCanvasLike(val) {
    return val && isFunction(val.toDataURL);
}
function setStyleAttrs(attrs, style, el, scope) {
    mapStyleToAttrs(function (key, val) {
        var isFillStroke = key === 'fill' || key === 'stroke';
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
            attrs['pointer-events'] = 'visible';
        }
    }, style, el, false);
    setShadow(el, attrs, scope);
}
function setMetaData(attrs, el) {
    var metaData = getElementSSRData(el);
    if (metaData) {
        metaData.each(function (val, key) {
            val != null && (attrs[(META_DATA_PREFIX + key).toLowerCase()] = val + '');
        });
        if (el.isSilent()) {
            attrs[META_DATA_PREFIX + 'silent'] = 'true';
        }
    }
}
function noRotateScale(m) {
    return isAroundZero(m[0] - 1)
        && isAroundZero(m[1])
        && isAroundZero(m[2])
        && isAroundZero(m[3] - 1);
}
function noTranslate(m) {
    return isAroundZero(m[4]) && isAroundZero(m[5]);
}
function setTransform(attrs, m, compress) {
    if (m && !(noTranslate(m) && noRotateScale(m))) {
        var mul = compress ? 10 : 1e4;
        attrs.transform = noRotateScale(m)
            ? "translate(" + round(m[4] * mul) / mul + " " + round(m[5] * mul) / mul + ")" : getMatrixStr(m);
    }
}
function convertPolyShape(shape, attrs, mul) {
    var points = shape.points;
    var strArr = [];
    for (var i = 0; i < points.length; i++) {
        strArr.push(round(points[i][0] * mul) / mul);
        strArr.push(round(points[i][1] * mul) / mul);
    }
    attrs.points = strArr.join(' ');
}
function validatePolyShape(shape) {
    return !shape.smooth;
}
function createAttrsConvert(desc) {
    var normalizedDesc = map(desc, function (item) {
        return (typeof item === 'string' ? [item, item] : item);
    });
    return function (shape, attrs, mul) {
        for (var i = 0; i < normalizedDesc.length; i++) {
            var item = normalizedDesc[i];
            var val = shape[item[0]];
            if (val != null) {
                attrs[item[1]] = round(val * mul) / mul;
            }
        }
    };
}
var builtinShapesDef = {
    circle: [createAttrsConvert(['cx', 'cy', 'r'])],
    polyline: [convertPolyShape, validatePolyShape],
    polygon: [convertPolyShape, validatePolyShape]
};
function hasShapeAnimation(el) {
    var animators = el.animators;
    for (var i = 0; i < animators.length; i++) {
        if (animators[i].targetName === 'shape') {
            return true;
        }
    }
    return false;
}
export function brushSVGPath(el, scope) {
    var style = el.style;
    var shape = el.shape;
    var builtinShpDef = builtinShapesDef[el.type];
    var attrs = {};
    var needsAnimate = scope.animation;
    var svgElType = 'path';
    var strokePercent = el.style.strokePercent;
    var precision = (scope.compress && getPathPrecision(el)) || 4;
    if (builtinShpDef
        && !scope.willUpdate
        && !(builtinShpDef[1] && !builtinShpDef[1](shape))
        && !(needsAnimate && hasShapeAnimation(el))
        && !(strokePercent < 1)) {
        svgElType = el.type;
        var mul = Math.pow(10, precision);
        builtinShpDef[0](shape, attrs, mul);
    }
    else {
        var needBuildPath = !el.path || el.shapeChanged();
        if (!el.path) {
            el.createPathProxy();
        }
        var path = el.path;
        if (needBuildPath) {
            path.beginPath();
            el.buildPath(path, el.shape);
            el.pathUpdated();
        }
        var pathVersion = path.getVersion();
        var elExt = el;
        var svgPathBuilder = elExt.__svgPathBuilder;
        if (elExt.__svgPathVersion !== pathVersion
            || !svgPathBuilder
            || strokePercent !== elExt.__svgPathStrokePercent) {
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
export function brushSVGImage(el, scope) {
    var style = el.style;
    var image = style.image;
    if (image && !isString(image)) {
        if (isImageLike(image)) {
            image = image.src;
        }
        else if (isCanvasLike(image)) {
            image = image.toDataURL();
        }
    }
    if (!image) {
        return;
    }
    var x = style.x || 0;
    var y = style.y || 0;
    var dw = style.width;
    var dh = style.height;
    var attrs = {
        href: image,
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
}
;
export function brushSVGTSpan(el, scope) {
    var style = el.style;
    var text = style.text;
    text != null && (text += '');
    if (!text || isNaN(style.x) || isNaN(style.y)) {
        return;
    }
    var font = style.font || DEFAULT_FONT;
    var x = style.x || 0;
    var y = adjustTextY(style.y || 0, getLineHeight(font), style.textBaseline);
    var textAlign = TEXT_ALIGN_TO_ANCHOR[style.textAlign]
        || style.textAlign;
    var attrs = {
        'dominant-baseline': 'central',
        'text-anchor': textAlign
    };
    if (hasSeparateFont(style)) {
        var separatedFontStr = '';
        var fontStyle = style.fontStyle;
        var fontSize = parseFontSize(style.fontSize);
        if (!parseFloat(fontSize)) {
            return;
        }
        var fontFamily = style.fontFamily || DEFAULT_FONT_FAMILY;
        var fontWeight = style.fontWeight;
        separatedFontStr += "font-size:" + fontSize + ";font-family:" + fontFamily + ";";
        if (fontStyle && fontStyle !== 'normal') {
            separatedFontStr += "font-style:" + fontStyle + ";";
        }
        if (fontWeight && fontWeight !== 'normal') {
            separatedFontStr += "font-weight:" + fontWeight + ";";
        }
        attrs.style = separatedFontStr;
    }
    else {
        attrs.style = "font: " + font;
    }
    if (text.match(/\s/)) {
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
export function brush(el, scope) {
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
function setShadow(el, attrs, scope) {
    var style = el.style;
    if (hasShadow(style)) {
        var shadowKey = getShadowKey(el);
        var shadowCache = scope.shadowCache;
        var shadowId = shadowCache[shadowKey];
        if (!shadowId) {
            var globalScale = el.getGlobalScale();
            var scaleX = globalScale[0];
            var scaleY = globalScale[1];
            if (!scaleX || !scaleY) {
                return;
            }
            var offsetX = style.shadowOffsetX || 0;
            var offsetY = style.shadowOffsetY || 0;
            var blur_1 = style.shadowBlur;
            var _a = normalizeColor(style.shadowColor), opacity = _a.opacity, color = _a.color;
            var stdDx = blur_1 / 2 / scaleX;
            var stdDy = blur_1 / 2 / scaleY;
            var stdDeviation = stdDx + ' ' + stdDy;
            shadowId = scope.zrId + '-s' + scope.shadowIdx++;
            scope.defs[shadowId] = createVNode('filter', shadowId, {
                'id': shadowId,
                'x': '-100%',
                'y': '-100%',
                'width': '300%',
                'height': '300%'
            }, [
                createVNode('feDropShadow', '', {
                    'dx': offsetX / scaleX,
                    'dy': offsetY / scaleY,
                    'stdDeviation': stdDeviation,
                    'flood-color': color,
                    'flood-opacity': opacity
                })
            ]);
            shadowCache[shadowKey] = shadowId;
        }
        attrs.filter = getIdURL(shadowId);
    }
}
export function setGradient(style, attrs, target, scope) {
    var val = style[target];
    var gradientTag;
    var gradientAttrs = {
        'gradientUnits': val.global
            ? 'userSpaceOnUse'
            : 'objectBoundingBox'
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
    var colors = val.colorStops;
    var colorStops = [];
    for (var i = 0, len = colors.length; i < len; ++i) {
        var offset = round4(colors[i].offset) * 100 + '%';
        var stopColor = colors[i].color;
        var _a = normalizeColor(stopColor), color = _a.color, opacity = _a.opacity;
        var stopsAttrs = {
            'offset': offset
        };
        stopsAttrs['stop-color'] = color;
        if (opacity < 1) {
            stopsAttrs['stop-opacity'] = opacity;
        }
        colorStops.push(createVNode('stop', i + '', stopsAttrs));
    }
    var gradientVNode = createVNode(gradientTag, '', gradientAttrs, colorStops);
    var gradientKey = vNodeToString(gradientVNode);
    var gradientCache = scope.gradientCache;
    var gradientId = gradientCache[gradientKey];
    if (!gradientId) {
        gradientId = scope.zrId + '-g' + scope.gradientIdx++;
        gradientCache[gradientKey] = gradientId;
        gradientAttrs.id = gradientId;
        scope.defs[gradientId] = createVNode(gradientTag, gradientId, gradientAttrs, colorStops);
    }
    attrs[target] = getIdURL(gradientId);
}
export function setPattern(el, attrs, target, scope) {
    var val = el.style[target];
    var boundingRect = el.getBoundingRect();
    var patternAttrs = {};
    var repeat = val.repeat;
    var noRepeat = repeat === 'no-repeat';
    var repeatX = repeat === 'repeat-x';
    var repeatY = repeat === 'repeat-y';
    var child;
    if (isImagePattern(val)) {
        var imageWidth_1 = val.imageWidth;
        var imageHeight_1 = val.imageHeight;
        var imageSrc = void 0;
        var patternImage = val.image;
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
            var errMsg = 'Image width/height must been given explictly in svg-ssr renderer.';
            assert(imageWidth_1, errMsg);
            assert(imageHeight_1, errMsg);
        }
        else if (imageWidth_1 == null || imageHeight_1 == null) {
            var setSizeToVNode_1 = function (vNode, img) {
                if (vNode) {
                    var svgEl = vNode.elm;
                    var width = imageWidth_1 || img.width;
                    var height = imageHeight_1 || img.height;
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
                        svgEl.setAttribute('width', width);
                        svgEl.setAttribute('height', height);
                    }
                }
            };
            var createdImage = createOrUpdateImage(imageSrc, null, el, function (img) {
                noRepeat || setSizeToVNode_1(patternVNode, img);
                setSizeToVNode_1(child, img);
            });
            if (createdImage && createdImage.width && createdImage.height) {
                imageWidth_1 = imageWidth_1 || createdImage.width;
                imageHeight_1 = imageHeight_1 || createdImage.height;
            }
        }
        child = createVNode('image', 'img', {
            href: imageSrc,
            width: imageWidth_1,
            height: imageHeight_1
        });
        patternAttrs.width = imageWidth_1;
        patternAttrs.height = imageHeight_1;
    }
    else if (val.svgElement) {
        child = clone(val.svgElement);
        patternAttrs.width = val.svgWidth;
        patternAttrs.height = val.svgHeight;
    }
    if (!child) {
        return;
    }
    var patternWidth;
    var patternHeight;
    if (noRepeat) {
        patternWidth = patternHeight = 1;
    }
    else if (repeatX) {
        patternHeight = 1;
        patternWidth = patternAttrs.width / boundingRect.width;
    }
    else if (repeatY) {
        patternWidth = 1;
        patternHeight = patternAttrs.height / boundingRect.height;
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
    var patternTransform = getSRTTransformString(val);
    patternTransform && (patternAttrs.patternTransform = patternTransform);
    var patternVNode = createVNode('pattern', '', patternAttrs, [child]);
    var patternKey = vNodeToString(patternVNode);
    var patternCache = scope.patternCache;
    var patternId = patternCache[patternKey];
    if (!patternId) {
        patternId = scope.zrId + '-p' + scope.patternIdx++;
        patternCache[patternKey] = patternId;
        patternAttrs.id = patternId;
        patternVNode = scope.defs[patternId] = createVNode('pattern', patternId, patternAttrs, [child]);
    }
    attrs[target] = getIdURL(patternId);
}
export function setClipPath(clipPath, attrs, scope) {
    var clipPathCache = scope.clipPathCache, defs = scope.defs;
    var clipPathId = clipPathCache[clipPath.id];
    if (!clipPathId) {
        clipPathId = scope.zrId + '-c' + scope.clipPathIdx++;
        var clipPathAttrs = {
            id: clipPathId
        };
        clipPathCache[clipPath.id] = clipPathId;
        defs[clipPathId] = createVNode('clipPath', clipPathId, clipPathAttrs, [brushSVGPath(clipPath, scope)]);
    }
    attrs['clip-path'] = getIdURL(clipPathId);
}
