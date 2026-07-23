import { LinearGradientObject } from '../graphic/LinearGradient';
import { RadialGradientObject } from '../graphic/RadialGradient';
import { GradientObject } from '../graphic/Gradient';
import { RectLike } from '../core/BoundingRect';
import Path from '../graphic/Path';

function isSafeNum(num: number) {
    // NaN、Infinity、undefined、'xx'
    return isFinite(num);
}

export function createLinearGradient(
    this: void,
    ctx: CanvasRenderingContext2D,
    obj: LinearGradientObject,
    rect: RectLike
) {
    let x = obj.x == null ? 0 : obj.x;
    let x2 = obj.x2 == null ? 1 : obj.x2;
    let y = obj.y == null ? 0 : obj.y;
    let y2 = obj.y2 == null ? 0 : obj.y2;

    if (!obj.global) {
        x = x * rect.width + rect.x;
        x2 = x2 * rect.width + rect.x;
        y = y * rect.height + rect.y;
        y2 = y2 * rect.height + rect.y;
    }

    // Fix NaN when rect is Infinity
    x = isSafeNum(x) ? x : 0;
    x2 = isSafeNum(x2) ? x2 : 1;
    y = isSafeNum(y) ? y : 0;
    y2 = isSafeNum(y2) ? y2 : 0;

    const canvasGradient = ctx.createLinearGradient(x, y, x2, y2);

    return canvasGradient;
}

export function createRadialGradient(
    this: void,
    ctx: CanvasRenderingContext2D,
    obj: RadialGradientObject,
    rect: RectLike
) {
    const width = rect.width;
    const height = rect.height;
    const min = Math.min(width, height);

    let x = obj.x == null ? 0.5 : obj.x;
    let y = obj.y == null ? 0.5 : obj.y;
    let r = obj.r == null ? 0.5 : obj.r;

    if (!obj.global) {
        x = x * width + rect.x;
        y = y * height + rect.y;
        r = r * min;
    }

    x = isSafeNum(x) ? x : 0.5;
    y = isSafeNum(y) ? y : 0.5;
    r = r >= 0 && isSafeNum(r) ? r : 0.5;

    const canvasGradient = ctx.createRadialGradient(x, y, 0, x, y, r);

    return canvasGradient;
}

export function getCanvasGradient(this: void, ctx: CanvasRenderingContext2D, obj: GradientObject, rect: RectLike) {
    // TODO Cache?
    const canvasGradient = obj.type === 'radial'
        ? createRadialGradient(ctx, obj as RadialGradientObject, rect)
        : createLinearGradient(ctx, obj as LinearGradientObject, rect);

    const colorStops = obj.colorStops;
    for (let i = 0; i < colorStops.length; i++) {
        canvasGradient.addColorStop(
            colorStops[i].offset, colorStops[i].color
        );
    }
    return canvasGradient;
}

export function isClipPathChanged(clipPaths: Path[], prevClipPaths: Path[]): boolean {
    // displayable.__clipPaths can only be `null`/`undefined` or an non-empty array.
    if (clipPaths === prevClipPaths || (!clipPaths && !prevClipPaths)) {
        return false;
    }
    if (!clipPaths || !prevClipPaths || (clipPaths.length !== prevClipPaths.length)) {
        return true;
    }
    for (let i = 0; i < clipPaths.length; i++) {
        if (clipPaths[i] !== prevClipPaths[i]) {
            return true;
        }
    }
    return false;
}

function parseInt10(val: string) {
    return parseInt(val, 10);
}
export function getSize(
    root: HTMLElement,
    whIdx: number,
    opts: { width?: number | string, height?: number | string}
) {

    const wh = ['width', 'height'][whIdx] as 'width' | 'height';
    const cwh = ['clientWidth', 'clientHeight'][whIdx] as 'clientWidth' | 'clientHeight';
    const plt = ['paddingLeft', 'paddingTop'][whIdx] as 'paddingLeft' | 'paddingTop';
    const prb = ['paddingRight', 'paddingBottom'][whIdx] as 'paddingRight' | 'paddingBottom';

    if (opts[wh] != null && opts[wh] !== 'auto') {
        return parseFloat(opts[wh] as string);
    }

    // IE8 does not support getComputedStyle, but it use VML.
    const stl = document.defaultView.getComputedStyle(root);

    return (
        (root[cwh] || parseInt10(stl[wh]) || parseInt10(root.style[wh]))
        - (parseInt10(stl[plt]) || 0)
        - (parseInt10(stl[prb]) || 0)
    ) | 0;
}