/**
 * Utilities for mouse or touch events.
 */

import Eventful from './Eventful';
import env from './env';
import { ZRRawEvent } from './types';
import {isCanvasEl, transformCoordWithViewport} from './dom';

const MOUSE_EVENT_REG = /^(?:mouse|pointer|contextmenu|drag|drop)|click/;
const _calcOut: number[] = [];
const firefoxNotSupportOffsetXY = env.browser.firefox
    // use offsetX/offsetY for Firefox >= 39
    // PENDING: consider Firefox for Android and Firefox OS? >= 43
    && +(env.browser.version as string).split('.')[0] < 39;

type FirefoxMouseEvent = {
    layerX: number
    layerY: number
}


/**
 * Get the `zrX` and `zrY`, which are relative to the top-left of
 * the input `el`.
 * CSS transform (2D & 3D) is supported.
 *
 * The strategy to fetch the coords:
 * + If `calculate` is not set as `true`, users of this method should
 * ensure that `el` is the same or the same size & location as `e.target`.
 * Otherwise the result coords are probably not expected. Because we
 * firstly try to get coords from e.offsetX/e.offsetY.
 * + If `calculate` is set as `true`, the input `el` can be any element
 * and we force to calculate the coords based on `el`.
 * + The input `el` should be positionable (not position:static).
 *
 * The force `calculate` can be used in case like:
 * When mousemove event triggered on ec tooltip, `e.target` is not `el`(zr painter.dom).
 *
 * @param  el DOM element.
 * @param  e Mouse event or touch event.
 * @param  out Get `out.zrX` and `out.zrY` as the result.
 * @param  calculate Whether to force calculate
 *        the coordinates but not use ones provided by browser.
 */
export function clientToLocal(
    el: HTMLElement,
    e: ZRRawEvent | FirefoxMouseEvent | Touch,
    out: {zrX?: number, zrY?: number},
    calculate?: boolean
) {
    out = out || {};

    // According to the W3C Working Draft, offsetX and offsetY should be relative
    // to the padding edge of the target element. The only browser using this convention
    // is IE. Webkit uses the border edge, Opera uses the content edge, and FireFox does
    // not support the properties.
    // (see http://www.jacklmoore.com/notes/mouse-position/)
    // In zr painter.dom, padding edge equals to border edge.
    if (calculate) {
        calculateZrXY(el, e as ZRRawEvent, out);
    }
    // Caution: In FireFox, layerX/layerY Mouse position relative to the closest positioned
    // ancestor element, so we should make sure el is positioned (e.g., not position:static).
    // BTW1, Webkit don't return the same results as FF in non-simple cases (like add
    // zoom-factor, overflow / opacity layers, transforms ...)
    // BTW2, (ev.offsetY || ev.pageY - $(ev.target).offset().top) is not correct in preserve-3d.
    // <https://bugs.jquery.com/ticket/8523#comment:14>
    // BTW3, In ff, offsetX/offsetY is always 0.
    else if (firefoxNotSupportOffsetXY
        && (e as FirefoxMouseEvent).layerX != null
        && (e as FirefoxMouseEvent).layerX !== (e as MouseEvent).offsetX
    ) {
        out.zrX = (e as FirefoxMouseEvent).layerX;
        out.zrY = (e as FirefoxMouseEvent).layerY;
    }
    // For IE6+, chrome, safari, opera, firefox >= 39
    else if ((e as MouseEvent).offsetX != null) {
        out.zrX = (e as MouseEvent).offsetX;
        out.zrY = (e as MouseEvent).offsetY;
    }
    // For some other device, e.g., IOS safari.
    else {
        calculateZrXY(el, e as ZRRawEvent, out);
    }

    return out;
}

function calculateZrXY(
    el: HTMLElement,
    e: ZRRawEvent,
    out: {zrX?: number, zrY?: number}
) {
    // BlackBerry 5, iOS 3 (original iPhone) don't have getBoundingRect.
    if (env.domSupported && el.getBoundingClientRect) {
        const ex = (e as MouseEvent).clientX;
        const ey = (e as MouseEvent).clientY;

        if (isCanvasEl(el)) {
            // Original approach, which do not support CSS transform.
            // marker can not be locationed in a canvas container
            // (getBoundingClientRect is always 0). We do not support
            // that input a pre-created canvas to zr while using css
            // transform in iOS.
            const box = el.getBoundingClientRect();
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

/**
 * Find native event compat for legency IE.
 * Should be called at the begining of a native event listener.
 *
 * @param e Mouse event or touch event or pointer event.
 *        For lagency IE, we use `window.event` is used.
 * @return The native event.
 */
export function getNativeEvent(e: ZRRawEvent): ZRRawEvent {
    return e
        || (window.event as any);   // For IE
}

/**
 * Normalize the coordinates of the input event.
 *
 * Get the `e.zrX` and `e.zrY`, which are relative to the top-left of
 * the input `el`.
 * Get `e.zrDelta` if using mouse wheel.
 * Get `e.which`, see the comment inside this function.
 *
 * Do not calculate repeatly if `zrX` and `zrY` already exist.
 *
 * Notice: see comments in `clientToLocal`. check the relationship
 * between the result coords and the parameters `el` and `calculate`.
 *
 * @param el DOM element.
 * @param e See `getNativeEvent`.
 * @param calculate Whether to force calculate
 *        the coordinates but not use ones provided by browser.
 * @return The normalized native UIEvent.
 */
export function normalizeEvent(
    el: HTMLElement,
    e: ZRRawEvent,
    calculate?: boolean
) {

    e = getNativeEvent(e);

    if (e.zrX != null) {
        return e;
    }

    const eventType = e.type;
    const isTouch = eventType && eventType.indexOf('touch') >= 0;

    if (!isTouch) {
        clientToLocal(el, e, e, calculate);
        const wheelDelta = getWheelDeltaMayPolyfill(e);
        // FIXME: IE8- has "wheelDeta" in event "mousewheel" but hat different value (120 times)
        // with Chrome and Safari. It's not correct for zrender event but we left it as it was.
        e.zrDelta = wheelDelta ? wheelDelta / 120 : -(e.detail || 0) / 3;
    }
    else {
        const touch = eventType !== 'touchend'
            ? (<TouchEvent>e).targetTouches[0]
            : (<TouchEvent>e).changedTouches[0];
        touch && clientToLocal(el, touch, e, calculate);
    }

    // Add which for click: 1 === left; 2 === middle; 3 === right; otherwise: 0;
    // See jQuery: https://github.com/jquery/jquery/blob/master/src/event.js
    // If e.which has been defined, it may be readonly,
    // see: https://developer.mozilla.org/en-US/docs/Web/API/MouseEvent/which
    const button = (<MouseEvent>e).button;
    if (e.which == null && button !== undefined && MOUSE_EVENT_REG.test(e.type)) {
        (e as any).which = (button & 1 ? 1 : (button & 2 ? 3 : (button & 4 ? 2 : 0)));
    }
    // [Caution]: `e.which` from browser is not always reliable. For example,
    // when press left button and `mousemove (pointermove)` in Edge, the `e.which`
    // is 65536 and the `e.button` is -1. But the `mouseup (pointerup)` and
    // `mousedown (pointerdown)` is the same as Chrome does.

    return e;
}

// TODO: also provide prop "deltaX" "deltaY" in zrender "mousewheel" event.
function getWheelDeltaMayPolyfill(e: ZRRawEvent): number {
    // Although event "wheel" do not has the prop "wheelDelta" in spec,
    // agent like Chrome and Safari still provide "wheelDelta" like
    // event "mousewheel" did (perhaps for backward compat).
    // Since zrender has been using "wheelDeta" in zrender event "mousewheel".
    // we currently do not break it.
    // But event "wheel" in firefox do not has "wheelDelta", so we calculate
    // "wheelDeta" from "deltaX", "deltaY" (which is the props in spec).
    const rawWheelDelta = (e as any).wheelDelta;
    // Theroetically `e.wheelDelta` won't be 0 unless some day it has been deprecated
    // by agent like Chrome or Safari. So we also calculate it if rawWheelDelta is 0.
    if (rawWheelDelta) {
        return rawWheelDelta;
    }

    const deltaX = (e as any).deltaX;
    const deltaY = (e as any).deltaY;
    if (deltaX == null || deltaY == null) {
        return rawWheelDelta;
    }

    // Test in Chrome and Safari (MacOS):
    // The sign is corrent.
    // The abs value is 99% corrent (inconsist case only like 62~63, 125~126 ...)
    const delta = deltaY !== 0 ? Math.abs(deltaY) : Math.abs(deltaX);
    const sign = deltaY > 0 ? -1
        : deltaY < 0 ? 1
        : deltaX > 0 ? -1
        : 1;
    return 3 * delta * sign;
}


type AddEventListenerParams = Parameters<typeof HTMLElement.prototype.addEventListener>
type RemoveEventListenerParams = Parameters<typeof HTMLElement.prototype.removeEventListener>
/**
 * @param  el
 * @param  name
 * @param  handler
 * @param  opt If boolean, means `opt.capture`
 * @param  opt.capture
 * @param  opt.passive
 */
export function addEventListener(
    el: HTMLElement | HTMLDocument,
    name: AddEventListenerParams[0],
    handler: AddEventListenerParams[1],
    opt?: AddEventListenerParams[2]
) {
    // Reproduct the console warning:
    // [Violation] Added non-passive event listener to a scroll-blocking <some> event.
    // Consider marking event handler as 'passive' to make the page more responsive.
    // Just set console log level: verbose in chrome dev tool.
    // then the warning log will be printed when addEventListener called.
    // See https://github.com/WICG/EventListenerOptions/blob/gh-pages/explainer.md
    // We have not yet found a neat way to using passive. Because in zrender the dom event
    // listener delegate all of the upper events of element. Some of those events need
    // to prevent default. For example, the feature `preventDefaultMouseMove` of echarts.
    // Before passive can be adopted, these issues should be considered:
    // (1) Whether and how a zrender user specifies an event listener passive. And by default,
    // passive or not.
    // (2) How to tread that some zrender event listener is passive, and some is not. If
    // we use other way but not preventDefault of mousewheel and touchmove, browser
    // compatibility should be handled.

    // const opts = (env.passiveSupported && name === 'mousewheel')
    //     ? {passive: true}
    //     // By default, the third param of el.addEventListener is `capture: false`.
    //     : void 0;
    // el.addEventListener(name, handler /* , opts */);
    el.addEventListener(name, handler, opt);
}

/**
 * Parameter are the same as `addEventListener`.
 *
 * Notice that if a listener is registered twice, one with capture and one without,
 * remove each one separately. Removal of a capturing listener does not affect a
 * non-capturing version of the same listener, and vice versa.
 */
export function removeEventListener(
    el: HTMLElement | HTMLDocument,
    name: RemoveEventListenerParams[0],
    handler: RemoveEventListenerParams[1],
    opt: RemoveEventListenerParams[2]
) {
    el.removeEventListener(name, handler, opt);
}

/**
 * preventDefault and stopPropagation.
 * Notice: do not use this method in zrender. It can only be
 * used by upper applications if necessary.
 *
 * @param {Event} e A mouse or touch event.
 */
export const stop = function (e: MouseEvent | TouchEvent | PointerEvent) {
    e.preventDefault();
    e.stopPropagation();
    e.cancelBubble = true;
};

/**
 * This method only works for mouseup and mousedown. The functionality is restricted
 * for fault tolerance, See the `e.which` compatibility above.
 *
 * params can be MouseEvent or ElementEvent
 */
export function isMiddleOrRightButtonOnMouseUpDown(e: { which: number }) {
    return e.which === 2 || e.which === 3;
}

// For backward compatibility
export {Eventful as Dispatcher};
