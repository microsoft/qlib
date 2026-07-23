function isSafeNum(num) {
    return isFinite(num);
}
export function createLinearGradient(ctx, obj, rect) {
    var x = obj.x == null ? 0 : obj.x;
    var x2 = obj.x2 == null ? 1 : obj.x2;
    var y = obj.y == null ? 0 : obj.y;
    var y2 = obj.y2 == null ? 0 : obj.y2;
    if (!obj.global) {
        x = x * rect.width + rect.x;
        x2 = x2 * rect.width + rect.x;
        y = y * rect.height + rect.y;
        y2 = y2 * rect.height + rect.y;
    }
    x = isSafeNum(x) ? x : 0;
    x2 = isSafeNum(x2) ? x2 : 1;
    y = isSafeNum(y) ? y : 0;
    y2 = isSafeNum(y2) ? y2 : 0;
    var canvasGradient = ctx.createLinearGradient(x, y, x2, y2);
    return canvasGradient;
}
export function createRadialGradient(ctx, obj, rect) {
    var width = rect.width;
    var height = rect.height;
    var min = Math.min(width, height);
    var x = obj.x == null ? 0.5 : obj.x;
    var y = obj.y == null ? 0.5 : obj.y;
    var r = obj.r == null ? 0.5 : obj.r;
    if (!obj.global) {
        x = x * width + rect.x;
        y = y * height + rect.y;
        r = r * min;
    }
    x = isSafeNum(x) ? x : 0.5;
    y = isSafeNum(y) ? y : 0.5;
    r = r >= 0 && isSafeNum(r) ? r : 0.5;
    var canvasGradient = ctx.createRadialGradient(x, y, 0, x, y, r);
    return canvasGradient;
}
export function getCanvasGradient(ctx, obj, rect) {
    var canvasGradient = obj.type === 'radial'
        ? createRadialGradient(ctx, obj, rect)
        : createLinearGradient(ctx, obj, rect);
    var colorStops = obj.colorStops;
    for (var i = 0; i < colorStops.length; i++) {
        canvasGradient.addColorStop(colorStops[i].offset, colorStops[i].color);
    }
    return canvasGradient;
}
export function isClipPathChanged(clipPaths, prevClipPaths) {
    if (clipPaths === prevClipPaths || (!clipPaths && !prevClipPaths)) {
        return false;
    }
    if (!clipPaths || !prevClipPaths || (clipPaths.length !== prevClipPaths.length)) {
        return true;
    }
    for (var i = 0; i < clipPaths.length; i++) {
        if (clipPaths[i] !== prevClipPaths[i]) {
            return true;
        }
    }
    return false;
}
function parseInt10(val) {
    return parseInt(val, 10);
}
export function getSize(root, whIdx, opts) {
    var wh = ['width', 'height'][whIdx];
    var cwh = ['clientWidth', 'clientHeight'][whIdx];
    var plt = ['paddingLeft', 'paddingTop'][whIdx];
    var prb = ['paddingRight', 'paddingBottom'][whIdx];
    if (opts[wh] != null && opts[wh] !== 'auto') {
        return parseFloat(opts[wh]);
    }
    var stl = document.defaultView.getComputedStyle(root);
    return ((root[cwh] || parseInt10(stl[wh]) || parseInt10(root.style[wh]))
        - (parseInt10(stl[plt]) || 0)
        - (parseInt10(stl[prb]) || 0)) | 0;
}
