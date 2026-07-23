import { PathStyleProps } from '../Path';

/**
 * Sub-pixel optimize for canvas rendering, prevent from blur
 * when rendering a thin vertical/horizontal line.
 */

const round = Math.round;

type LineShape = {
    x1: number
    y1: number
    x2: number
    y2: number
}

type RectShape = {
    x: number
    y: number
    width: number
    height: number
    r?: number | number[]
}
/**
 * Sub pixel optimize line for canvas
 *
 * @param outputShape The modification will be performed on `outputShape`.
 *                 `outputShape` and `inputShape` can be the same object.
 *                 `outputShape` object can be used repeatly, because all of
 *                 the `x1`, `x2`, `y1`, `y2` will be assigned in this method.
 */
export function subPixelOptimizeLine(
    outputShape: Partial<LineShape>,
    inputShape: LineShape,
    style: Pick<PathStyleProps, 'lineWidth'>   // DO not optimize when lineWidth is 0
): LineShape {
    if (!inputShape) {
        return;
    }

    const x1 = inputShape.x1;
    const x2 = inputShape.x2;
    const y1 = inputShape.y1;
    const y2 = inputShape.y2;

    outputShape.x1 = x1;
    outputShape.x2 = x2;
    outputShape.y1 = y1;
    outputShape.y2 = y2;

    const lineWidth = style && style.lineWidth;
    if (!lineWidth) {
        return outputShape as LineShape;
    }

    if (round(x1 * 2) === round(x2 * 2)) {
        outputShape.x1 = outputShape.x2 = subPixelOptimize(x1, lineWidth, true);
    }
    if (round(y1 * 2) === round(y2 * 2)) {
        outputShape.y1 = outputShape.y2 = subPixelOptimize(y1, lineWidth, true);
    }

    return outputShape as LineShape;
}

/**
 * Sub pixel optimize rect for canvas
 *
 * @param outputShape The modification will be performed on `outputShape`.
 *                 `outputShape` and `inputShape` can be the same object.
 *                 `outputShape` object can be used repeatly, because all of
 *                 the `x`, `y`, `width`, `height` will be assigned in this method.
 */
export function subPixelOptimizeRect(
    outputShape: Partial<RectShape>,
    inputShape: RectShape,
    style: Pick<PathStyleProps, 'lineWidth'>   // DO not optimize when lineWidth is 0
): RectShape {
    if (!inputShape) {
        return;
    }

    const originX = inputShape.x;
    const originY = inputShape.y;
    const originWidth = inputShape.width;
    const originHeight = inputShape.height;

    outputShape.x = originX;
    outputShape.y = originY;
    outputShape.width = originWidth;
    outputShape.height = originHeight;

    const lineWidth = style && style.lineWidth;
    if (!lineWidth) {
        return outputShape as RectShape;
    }

    outputShape.x = subPixelOptimize(originX, lineWidth, true);
    outputShape.y = subPixelOptimize(originY, lineWidth, true);
    outputShape.width = Math.max(
        subPixelOptimize(originX + originWidth, lineWidth, false) - outputShape.x,
        originWidth === 0 ? 0 : 1
    );
    outputShape.height = Math.max(
        subPixelOptimize(originY + originHeight, lineWidth, false) - outputShape.y,
        originHeight === 0 ? 0 : 1
    );

    return outputShape as RectShape;
}

/**
 * Sub pixel optimize for canvas
 *
 * @param position Coordinate, such as x, y
 * @param lineWidth If `null`/`undefined`/`0`, do not optimize.
 * @param positiveOrNegative Default false (negative).
 * @return Optimized position.
 */
export function subPixelOptimize(
    position: number,
    lineWidth?: number,
    positiveOrNegative?: boolean
) {
    if (!lineWidth) {
        return position;
    }
    // Assure that (position + lineWidth / 2) is near integer edge,
    // otherwise line will be fuzzy in canvas.
    const doubledPosition = round(position * 2);
    return (doubledPosition + round(lineWidth)) % 2 === 0
        ? doubledPosition / 2
        : (doubledPosition + (positiveOrNegative ? 1 : -1)) / 2;
}
