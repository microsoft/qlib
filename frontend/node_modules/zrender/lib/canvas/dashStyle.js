import { isArray, isNumber, map } from '../core/util.js';
export function normalizeLineDash(lineType, lineWidth) {
    if (!lineType || lineType === 'solid' || !(lineWidth > 0)) {
        return null;
    }
    return lineType === 'dashed'
        ? [4 * lineWidth, 2 * lineWidth]
        : lineType === 'dotted'
            ? [lineWidth]
            : isNumber(lineType)
                ? [lineType] : isArray(lineType) ? lineType : null;
}
export function getLineDash(el) {
    var style = el.style;
    var lineDash = style.lineDash && style.lineWidth > 0 && normalizeLineDash(style.lineDash, style.lineWidth);
    var lineDashOffset = style.lineDashOffset;
    if (lineDash) {
        var lineScale_1 = (style.strokeNoScale && el.getLineScale) ? el.getLineScale() : 1;
        if (lineScale_1 && lineScale_1 !== 1) {
            lineDash = map(lineDash, function (rawVal) {
                return rawVal / lineScale_1;
            });
            lineDashOffset /= lineScale_1;
        }
    }
    return [lineDash, lineDashOffset];
}
