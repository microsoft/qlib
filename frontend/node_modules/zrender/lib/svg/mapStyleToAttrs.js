import { DEFAULT_PATH_STYLE } from '../graphic/Path.js';
import ZRImage from '../graphic/Image.js';
import { getLineDash } from '../canvas/dashStyle.js';
import { map } from '../core/util.js';
import { normalizeColor } from './helper.js';
var NONE = 'none';
var mathRound = Math.round;
function pathHasFill(style) {
    var fill = style.fill;
    return fill != null && fill !== NONE;
}
function pathHasStroke(style) {
    var stroke = style.stroke;
    return stroke != null && stroke !== NONE;
}
var strokeProps = ['lineCap', 'miterLimit', 'lineJoin'];
var svgStrokeProps = map(strokeProps, function (prop) { return "stroke-" + prop.toLowerCase(); });
export default function mapStyleToAttrs(updateAttr, style, el, forceUpdate) {
    var opacity = style.opacity == null ? 1 : style.opacity;
    if (el instanceof ZRImage) {
        updateAttr('opacity', opacity);
        return;
    }
    if (pathHasFill(style)) {
        var fill = normalizeColor(style.fill);
        updateAttr('fill', fill.color);
        var fillOpacity = style.fillOpacity != null
            ? style.fillOpacity * fill.opacity * opacity
            : fill.opacity * opacity;
        if (forceUpdate || fillOpacity < 1) {
            updateAttr('fill-opacity', fillOpacity);
        }
    }
    else {
        updateAttr('fill', NONE);
    }
    if (pathHasStroke(style)) {
        var stroke = normalizeColor(style.stroke);
        updateAttr('stroke', stroke.color);
        var strokeScale = style.strokeNoScale
            ? el.getLineScale()
            : 1;
        var strokeWidth = (strokeScale ? (style.lineWidth || 0) / strokeScale : 0);
        var strokeOpacity = style.strokeOpacity != null
            ? style.strokeOpacity * stroke.opacity * opacity
            : stroke.opacity * opacity;
        var strokeFirst = style.strokeFirst;
        if (forceUpdate || strokeWidth !== 1) {
            updateAttr('stroke-width', strokeWidth);
        }
        if (forceUpdate || strokeFirst) {
            updateAttr('paint-order', strokeFirst ? 'stroke' : 'fill');
        }
        if (forceUpdate || strokeOpacity < 1) {
            updateAttr('stroke-opacity', strokeOpacity);
        }
        if (style.lineDash) {
            var _a = getLineDash(el), lineDash = _a[0], lineDashOffset = _a[1];
            if (lineDash) {
                lineDashOffset = mathRound(lineDashOffset || 0);
                updateAttr('stroke-dasharray', lineDash.join(','));
                if (lineDashOffset || forceUpdate) {
                    updateAttr('stroke-dashoffset', lineDashOffset);
                }
            }
        }
        else if (forceUpdate) {
            updateAttr('stroke-dasharray', NONE);
        }
        for (var i = 0; i < strokeProps.length; i++) {
            var propName = strokeProps[i];
            if (forceUpdate || style[propName] !== DEFAULT_PATH_STYLE[propName]) {
                var val = style[propName] || DEFAULT_PATH_STYLE[propName];
                val && updateAttr(svgStrokeProps[i], val);
            }
        }
    }
    else if (forceUpdate) {
        updateAttr('stroke', NONE);
    }
}
