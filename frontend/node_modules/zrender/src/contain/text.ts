import BoundingRect, { RectLike } from '../core/BoundingRect';
import { Dictionary, TextAlign, TextVerticalAlign, BuiltinTextPosition } from '../core/types';
import LRU from '../core/LRU';
import { DEFAULT_FONT, platformApi } from '../core/platform';

let textWidthCache: Dictionary<LRU<number>> = {};

export function getWidth(text: string, font: string): number {
    font = font || DEFAULT_FONT;
    let cacheOfFont = textWidthCache[font];
    if (!cacheOfFont) {
        cacheOfFont = textWidthCache[font] = new LRU(500);
    }
    let width = cacheOfFont.get(text);
    if (width == null) {
        width = platformApi.measureText(text, font).width;
        cacheOfFont.put(text, width);
    }

    return width;
}

/**
 *
 * Get bounding rect for inner usage(TSpan)
 * Which not include text newline.
 */
export function innerGetBoundingRect(
    text: string,
    font: string,
    textAlign?: TextAlign,
    textBaseline?: TextVerticalAlign
): BoundingRect {
    const width = getWidth(text, font);
    const height = getLineHeight(font);

    const x = adjustTextX(0, width, textAlign);
    const y = adjustTextY(0, height, textBaseline);

    const rect = new BoundingRect(x, y, width, height);

    return rect;
}

/**
 *
 * Get bounding rect for outer usage. Compatitable with old implementation
 * Which includes text newline.
 */
export function getBoundingRect(
    text: string,
    font: string,
    textAlign?: TextAlign,
    textBaseline?: TextVerticalAlign
) {
    const textLines = ((text || '') + '').split('\n');
    const len = textLines.length;
    if (len === 1) {
        return innerGetBoundingRect(textLines[0], font, textAlign, textBaseline);
    }
    else {
        const uniondRect = new BoundingRect(0, 0, 0, 0);
        for (let i = 0; i < textLines.length; i++) {
            const rect = innerGetBoundingRect(textLines[i], font, textAlign, textBaseline);
            i === 0 ? uniondRect.copy(rect) : uniondRect.union(rect);
        }
        return uniondRect;
    }
}

export function adjustTextX(x: number, width: number, textAlign: TextAlign): number {
    // TODO Right to left language
    if (textAlign === 'right') {
        x -= width;
    }
    else if (textAlign === 'center') {
        x -= width / 2;
    }
    return x;
}

export function adjustTextY(y: number, height: number, verticalAlign: TextVerticalAlign): number {
    if (verticalAlign === 'middle') {
        y -= height / 2;
    }
    else if (verticalAlign === 'bottom') {
        y -= height;
    }
    return y;
}


export function getLineHeight(font?: string): number {
    // FIXME A rough approach.
    return getWidth('国', font);
}

export function measureText(text: string, font?: string): {
    width: number
} {
    return platformApi.measureText(text, font);
}


export function parsePercent(value: number | string, maxValue: number): number {
    if (typeof value === 'string') {
        if (value.lastIndexOf('%') >= 0) {
            return parseFloat(value) / 100 * maxValue;
        }
        return parseFloat(value);
    }
    return value;
}

export interface TextPositionCalculationResult {
    x: number
    y: number
    align: TextAlign
    verticalAlign: TextVerticalAlign
}
/**
 * Follow same interface to `Displayable.prototype.calculateTextPosition`.
 * @public
 * @param out Prepared out object. If not input, auto created in the method.
 * @param style where `textPosition` and `textDistance` are visited.
 * @param rect {x, y, width, height} Rect of the host elment, according to which the text positioned.
 * @return The input `out`. Set: {x, y, textAlign, textVerticalAlign}
 */
export function calculateTextPosition(
    out: TextPositionCalculationResult,
    opts: {
        position?: BuiltinTextPosition | (number | string)[]
        distance?: number   // Default 5
        global?: boolean
    },
    rect: RectLike
): TextPositionCalculationResult {
    const textPosition = opts.position || 'inside';
    const distance = opts.distance != null ? opts.distance : 5;

    const height = rect.height;
    const width = rect.width;
    const halfHeight = height / 2;

    let x = rect.x;
    let y = rect.y;

    let textAlign: TextAlign = 'left';
    let textVerticalAlign: TextVerticalAlign = 'top';

    if (textPosition instanceof Array) {
        x += parsePercent(textPosition[0], rect.width);
        y += parsePercent(textPosition[1], rect.height);
        // Not use textAlign / textVerticalAlign
        textAlign = null;
        textVerticalAlign = null;
    }
    else {
        switch (textPosition) {
            case 'left':
                x -= distance;
                y += halfHeight;
                textAlign = 'right';
                textVerticalAlign = 'middle';
                break;
            case 'right':
                x += distance + width;
                y += halfHeight;
                textVerticalAlign = 'middle';
                break;
            case 'top':
                x += width / 2;
                y -= distance;
                textAlign = 'center';
                textVerticalAlign = 'bottom';
                break;
            case 'bottom':
                x += width / 2;
                y += height + distance;
                textAlign = 'center';
                break;
            case 'inside':
                x += width / 2;
                y += halfHeight;
                textAlign = 'center';
                textVerticalAlign = 'middle';
                break;
            case 'insideLeft':
                x += distance;
                y += halfHeight;
                textVerticalAlign = 'middle';
                break;
            case 'insideRight':
                x += width - distance;
                y += halfHeight;
                textAlign = 'right';
                textVerticalAlign = 'middle';
                break;
            case 'insideTop':
                x += width / 2;
                y += distance;
                textAlign = 'center';
                break;
            case 'insideBottom':
                x += width / 2;
                y += height - distance;
                textAlign = 'center';
                textVerticalAlign = 'bottom';
                break;
            case 'insideTopLeft':
                x += distance;
                y += distance;
                break;
            case 'insideTopRight':
                x += width - distance;
                y += distance;
                textAlign = 'right';
                break;
            case 'insideBottomLeft':
                x += distance;
                y += height - distance;
                textVerticalAlign = 'bottom';
                break;
            case 'insideBottomRight':
                x += width - distance;
                y += height - distance;
                textAlign = 'right';
                textVerticalAlign = 'bottom';
                break;
        }
    }

    out = out || {} as TextPositionCalculationResult;
    out.x = x;
    out.y = y;
    out.align = textAlign;
    out.verticalAlign = textVerticalAlign;

    return out;
}
