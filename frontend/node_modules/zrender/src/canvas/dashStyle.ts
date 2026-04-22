import { isArray, isNumber, map } from '../core/util';
import Path from '../graphic/Path';
import TSpan from '../graphic/TSpan';

export function normalizeLineDash(lineType: any, lineWidth?: number): number[] | false {
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
export function getLineDash(el: Path | TSpan): [number[] | false, number] {
    const style = el.style;

    let lineDash = style.lineDash && style.lineWidth > 0 && normalizeLineDash(style.lineDash, style.lineWidth);
    let lineDashOffset = style.lineDashOffset;

    if (lineDash) {
        const lineScale = (style.strokeNoScale && el.getLineScale) ? el.getLineScale() : 1;
        if (lineScale && lineScale !== 1) {
            lineDash = map(lineDash, function (rawVal) {
                return rawVal / lineScale;
            });
            lineDashOffset /= lineScale;
        }
    }
    return [lineDash, lineDashOffset];
}