import Eventful from './Eventful.js';
import env from './env.js';
import { isCanvasEl, transformCoordWithViewport } from './dom.js';
var MOUSE_EVENT_REG = /^(?:mouse|pointer|contextmenu|drag|drop)|click/;
var _calcOut = [];
var firefoxNotSupportOffsetXY = env.browser.firefox
    && +env.browser.version.split('.')[0] < 39;
export function clientToLocal(el, e, out, calculate) {
    out = out || {};
    if (calculate) {
        calculateZrXY(el, e, out);
    }
    else if (firefoxNotSupportOffsetXY
        && e.layerX != null
        && e.layerX !== e.offsetX) {
        out.zrX = e.layerX;
        out.zrY = e.layerY;
    }
    else if (e.offsetX != null) {
        out.zrX = e.offsetX;
        out.zrY = e.offsetY;
    }
    else {
        calculateZrXY(el, e, out);
    }
    return out;
}
function calculateZrXY(el, e, out) {
    if (env.domSupported && el.getBoundingClientRect) {
        var ex = e.clientX;
        var ey = e.clientY;
        if (isCanvasEl(el)) {
            var box = el.getBoundingClientRect();
            out.zrX = ex - box.left;
            out.zrY = ey - box.top;
            return;
        }
        else {
            if (transformCoordWithViewport(_calcOut, el, ex, ey)) {
                out.zrX = _calcOut[0];
                out.zrY = _calcOut[1];
                return;
            }
        }
    }
    out.zrX = out.zrY = 0;
}
export function getNativeEvent(e) {
    return e
        || window.event;
}
export function normalizeEvent(el, e, calculate) {
    e = getNativeEvent(e);
    if (e.zrX != null) {
        return e;
    }
    var eventType = e.type;
    var isTouch = eventType && eventType.indexOf('touch') >= 0;
    if (!isTouch) {
        clientToLocal(el, e, e, calculate);
        var wheelDelta = getWheelDeltaMayPolyfill(e);
        e.zrDelta = wheelDelta ? wheelDelta / 120 : -(e.detail || 0) / 3;
    }
    else {
        var touch = eventType !== 'touchend'
            ? e.targetTouches[0]
            : e.changedTouches[0];
        touch && clientToLocal(el, touch, e, calculate);
    }
    var button = e.button;
    if (e.which == null && button !== undefined && MOUSE_EVENT_REG.test(e.type)) {
        e.which = (button & 1 ? 1 : (button & 2 ? 3 : (button & 4 ? 2 : 0)));
    }
    return e;
}
function getWheelDeltaMayPolyfill(e) {
    var rawWheelDelta = e.wheelDelta;
    if (rawWheelDelta) {
        return rawWheelDelta;
    }
    var deltaX = e.deltaX;
    var deltaY = e.deltaY;
    if (deltaX == null || deltaY == null) {
        return rawWheelDelta;
    }
    var delta = deltaY !== 0 ? Math.abs(deltaY) : Math.abs(deltaX);
    var sign = deltaY > 0 ? -1
        : deltaY < 0 ? 1
            : deltaX > 0 ? -1
                : 1;
    return 3 * delta * sign;
}
export function addEventListener(el, name, handler, opt) {
    el.addEventListener(name, handler, opt);
}
export function removeEventListener(el, name, handler, opt) {
    el.removeEventListener(name, handler, opt);
}
export var stop = function (e) {
    e.preventDefault();
    e.stopPropagation();
    e.cancelBubble = true;
};
export function isMiddleOrRightButtonOnMouseUpDown(e) {
    return e.which === 2 || e.which === 3;
}
export { Eventful as Dispatcher };
