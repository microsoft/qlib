import * as util from './core/util';
import * as vec2 from './core/vector';
import Draggable from './mixin/Draggable';
import Eventful from './core/Eventful';
import * as eventTool from './core/event';
import {GestureMgr} from './core/GestureMgr';
import Displayable from './graphic/Displayable';
import {PainterBase} from './PainterBase';
import HandlerDomProxy, { HandlerProxyInterface } from './dom/HandlerProxy';
import { ZRRawEvent, ZRPinchEvent, ElementEventName, ElementEventNameWithOn, ZRRawTouchEvent } from './core/types';
import Storage from './Storage';
import Element, {ElementEvent} from './Element';
import CanvasPainter from './canvas/Painter';
import BoundingRect from './core/BoundingRect';

/**
 * [The interface between `Handler` and `HandlerProxy`]:
 *
 * The default `HandlerProxy` only support the common standard web environment
 * (e.g., standalone browser, headless browser, embed browser in mobild APP, ...).
 * But `HandlerProxy` can be replaced to support more non-standard environment
 * (e.g., mini app), or to support more feature that the default `HandlerProxy`
 * not provided (like echarts-gl did).
 * So the interface between `Handler` and `HandlerProxy` should be stable. Do not
 * make break changes util inevitable. The interface include the public methods
 * of `Handler` and the events listed in `handlerNames` below, by which `HandlerProxy`
 * drives `Handler`.
 */

/**
 * [DRAG_OUTSIDE]:
 *
 * That is, triggering `mousemove` and `mouseup` event when the pointer is out of the
 * zrender area when dragging. That is important for the improvement of the user experience
 * when dragging something near the boundary without being terminated unexpectedly.
 *
 * We originally consider to introduce new events like `pagemovemove` and `pagemouseup`
 * to resolve this issue. But some drawbacks of it is described in
 * https://github.com/ecomfe/zrender/pull/536#issuecomment-560286899
 *
 * Instead, we referenced the specifications:
 * https://www.w3.org/TR/touch-events/#the-touchmove-event
 * https://www.w3.org/TR/2014/WD-DOM-Level-3-Events-20140925/#event-type-mousemove
 * where the the mousemove/touchmove can be continue to fire if the user began a drag
 * operation and the pointer has left the boundary. (for the mouse event, browsers
 * only do it on `document` and when the pointer has left the boundary of the browser.)
 *
 * So the default `HandlerProxy` supports this feature similarly: if it is in the dragging
 * state (see `pointerCapture` in `HandlerProxy`), the `mousemove` and `mouseup` continue
 * to fire until release the pointer. That is implemented by listen to those event on
 * `document`.
 * If we implement some other `HandlerProxy` only for touch device, that would be easier.
 * The touch event support this feature by default.
 * The term "pointer capture" is from the spec:
 * https://www.w3.org/TR/pointerevents2/#idl-def-element-setpointercapture-pointerid
 *
 * Note:
 * There might be some cases that the mouse event can not be received on `document`.
 * For example,
 * (A) When `useCapture` is not supported and some user defined event listeners on the ancestor
 * of zr dom throw Error.
 * (B) When `useCapture` is not supported and some user defined event listeners on the ancestor of
 * zr dom call `stopPropagation`.
 * In these cases, the `mousemove` event might be keep triggering event when the mouse is released.
 * We try to reduce the side-effect in those cases, that is, use `isOutsideBoundary` to prevent
 * it from do anything (especially, `findHover`).
 * (`useCapture` mean, `addEvnetListener(listener, {capture: true})`, althought it may not be
 * suppported in some environments.)
 *
 * Note:
 * If `HandlerProxy` listens to `document` with `useCapture`, `HandlerProxy` needs to
 * prevent user-registered-handler from calling `stopPropagation` and `preventDefault`
 * when the `event.target` is not a zrender dom element. Otherwise the user-registered-handler
 * may be able to prevent other elements (that not relevant to zrender) in the web page from receiving
 * dom events.
 */

const SILENT = 'silent';

function makeEventPacket(eveType: ElementEventName, targetInfo: {
    target?: Element
    topTarget?: Element
}, event: ZRRawEvent): ElementEvent {
    return {
        type: eveType,
        event: event,
        // target can only be an element that is not silent.
        target: targetInfo.target,
        // topTarget can be a silent element.
        topTarget: targetInfo.topTarget,
        cancelBubble: false,
        offsetX: event.zrX,
        offsetY: event.zrY,
        gestureEvent: (event as ZRPinchEvent).gestureEvent,
        pinchX: (event as ZRPinchEvent).pinchX,
        pinchY: (event as ZRPinchEvent).pinchY,
        pinchScale: (event as ZRPinchEvent).pinchScale,
        wheelDelta: event.zrDelta,
        zrByTouch: event.zrByTouch,
        which: event.which,
        stop: stopEvent
    };
}

function stopEvent(this: ElementEvent) {
    eventTool.stop(this.event);
}

class EmptyProxy extends Eventful {
    handler: Handler = null
    dispose() {}
    setCursor() {}
}

class HoveredResult {
    x: number
    y: number
    target: Displayable
    topTarget: Displayable
    constructor(x?: number, y?: number) {
        this.x = x;
        this.y = y;
    }
}

const handlerNames = [
    'click', 'dblclick', 'mousewheel', 'mouseout',
    'mouseup', 'mousedown', 'mousemove', 'contextmenu'
];

type HandlerName = 'click' |'dblclick' |'mousewheel' |'mouseout' |
    'mouseup' |'mousedown' |'mousemove' |'contextmenu';

const tmpRect = new BoundingRect(0, 0, 0, 0);

// TODO draggable
class Handler extends Eventful {

    storage: Storage
    painter: PainterBase
    painterRoot: HTMLElement

    proxy: HandlerProxyInterface

    private _hovered = new HoveredResult(0, 0)

    private _gestureMgr: GestureMgr

    private _draggingMgr: Draggable

    private _pointerSize: number

    _downEl: Element
    _upEl: Element
    _downPoint: [number, number]

    constructor(
        storage: Storage,
        painter: PainterBase,
        proxy: HandlerProxyInterface,
        painterRoot: HTMLElement,
        pointerSize: number
    ) {
        super();

        this.storage = storage;

        this.painter = painter;

        this.painterRoot = painterRoot;

        this._pointerSize = pointerSize;

        proxy = proxy || new EmptyProxy();

        /**
         * Proxy of event. can be Dom, WebGLSurface, etc.
         */
        this.proxy = null;

        this.setHandlerProxy(proxy);

        this._draggingMgr = new Draggable(this);
    }

    setHandlerProxy(proxy: HandlerProxyInterface) {
        if (this.proxy) {
            this.proxy.dispose();
        }

        if (proxy) {
            util.each(handlerNames, function (name) {
                proxy.on && proxy.on(name, this[name as HandlerName], this);
            }, this);
            // Attach handler
            proxy.handler = this;
        }
        this.proxy = proxy;
    }

    mousemove(event: ZRRawEvent) {
        const x = event.zrX;
        const y = event.zrY;

        const isOutside = isOutsideBoundary(this, x, y);

        let lastHovered = this._hovered;
        let lastHoveredTarget = lastHovered.target;

        // If lastHoveredTarget is removed from zr (detected by '__zr') by some API call
        // (like 'setOption' or 'dispatchAction') in event handlers, we should find
        // lastHovered again here. Otherwise 'mouseout' can not be triggered normally.
        // See #6198.
        if (lastHoveredTarget && !lastHoveredTarget.__zr) {
            lastHovered = this.findHover(lastHovered.x, lastHovered.y);
            lastHoveredTarget = lastHovered.target;
        }

        const hovered = this._hovered = isOutside ? new HoveredResult(x, y) : this.findHover(x, y);
        const hoveredTarget = hovered.target;

        const proxy = this.proxy;
        proxy.setCursor && proxy.setCursor(hoveredTarget ? hoveredTarget.cursor : 'default');

        // Mouse out on previous hovered element
        if (lastHoveredTarget && hoveredTarget !== lastHoveredTarget) {
            this.dispatchToElement(lastHovered, 'mouseout', event);
        }

        // Mouse moving on one element
        this.dispatchToElement(hovered, 'mousemove', event);

        // Mouse over on a new element
        if (hoveredTarget && hoveredTarget !== lastHoveredTarget) {
            this.dispatchToElement(hovered, 'mouseover', event);
        }
    }

    mouseout(event: ZRRawEvent) {
        const eventControl = event.zrEventControl;

        if (eventControl !== 'only_globalout') {
            this.dispatchToElement(this._hovered, 'mouseout', event);
        }

        if (eventControl !== 'no_globalout') {
            // FIXME: if the pointer moving from the extra doms to realy "outside",
            // the `globalout` should have been triggered. But currently not.
            this.trigger('globalout', {type: 'globalout', event: event});
        }
    }

    /**
     * Resize
     */
    resize() {
        this._hovered = new HoveredResult(0, 0);
    }

    /**
     * Dispatch event
     */
    dispatch(eventName: HandlerName, eventArgs?: any) {
        const handler = this[eventName];
        handler && handler.call(this, eventArgs);
    }

    /**
     * Dispose
     */
    dispose() {

        this.proxy.dispose();

        this.storage = null;
        this.proxy = null;
        this.painter = null;
    }

    /**
     * 设置默认的cursor style
     * @param cursorStyle 例如 crosshair，默认为 'default'
     */
    setCursorStyle(cursorStyle: string) {
        const proxy = this.proxy;
        proxy.setCursor && proxy.setCursor(cursorStyle);
    }

    /**
     * 事件分发代理
     *
     * @private
     * @param {Object} targetInfo {target, topTarget} 目标图形元素
     * @param {string} eventName 事件名称
     * @param {Object} event 事件对象
     */
    dispatchToElement(targetInfo: {
        target?: Element
        topTarget?: Element
    }, eventName: ElementEventName, event: ZRRawEvent) {

        targetInfo = targetInfo || {};

        let el = targetInfo.target as Element;
        if (el && el.silent) {
            return;
        }
        const eventKey = ('on' + eventName) as ElementEventNameWithOn;
        const eventPacket = makeEventPacket(eventName, targetInfo, event);

        while (el) {
            el[eventKey]
                && (eventPacket.cancelBubble = !!el[eventKey].call(el, eventPacket));

            el.trigger(eventName, eventPacket);

            // Bubble to the host if on the textContent.
            // PENDING
            el = el.__hostTarget ? el.__hostTarget : el.parent;

            if (eventPacket.cancelBubble) {
                break;
            }
        }

        if (!eventPacket.cancelBubble) {
            // 冒泡到顶级 zrender 对象
            this.trigger(eventName, eventPacket);
            // 分发事件到用户自定义层
            // 用户有可能在全局 click 事件中 dispose，所以需要判断下 painter 是否存在
            if (this.painter && (this.painter as CanvasPainter).eachOtherLayer) {
                (this.painter as CanvasPainter).eachOtherLayer(function (layer) {
                    if (typeof (layer[eventKey]) === 'function') {
                        layer[eventKey].call(layer, eventPacket);
                    }
                    if (layer.trigger) {
                        layer.trigger(eventName, eventPacket);
                    }
                });
            }
        }
    }

    findHover(x: number, y: number, exclude?: Displayable): HoveredResult {
        const list = this.storage.getDisplayList();
        const out = new HoveredResult(x, y);
        setHoverTarget(list, out, x, y, exclude);

        if (this._pointerSize && !out.target) {
            /**
             * If no element at pointer position, check intersection with
             * elements with pointer enlarged by target size.
             */
            const candidates: Displayable[] = [];
            const pointerSize = this._pointerSize;
            const targetSizeHalf = pointerSize / 2;
            const pointerRect = new BoundingRect(x - targetSizeHalf, y - targetSizeHalf, pointerSize, pointerSize);

            for (let i = list.length - 1; i >= 0; i--) {
                const el = list[i];
                if (el !== exclude
                    && !el.ignore
                    && !el.ignoreCoarsePointer
                    // If an element ignores, its textContent should also ignore.
                    // TSpan's parent is not a Group but a ZRText.
                    // See Text.js _getOrCreateChild
                    && (!el.parent || !(el.parent as any).ignoreCoarsePointer)
                ) {
                    tmpRect.copy(el.getBoundingRect());
                    if (el.transform) {
                        tmpRect.applyTransform(el.transform);
                    }
                    if (tmpRect.intersect(pointerRect)) {
                        candidates.push(el);
                    }
                }
            }

            /**
             * If there are elements whose bounding boxes are near the pointer,
             * use the most top one that intersects with the enlarged pointer.
             */
            if (candidates.length) {
                const rStep = 4;
                const thetaStep = Math.PI / 12;
                const PI2 = Math.PI * 2;
                for (let r = 0; r < targetSizeHalf; r += rStep) {
                    for (let theta = 0; theta < PI2; theta += thetaStep) {
                        const x1 = x + r * Math.cos(theta);
                        const y1 = y + r * Math.sin(theta);
                        setHoverTarget(candidates, out, x1, y1, exclude);
                        if (out.target) {
                            return out;
                        }
                    }
                }
            }
        }

        return out;
    }

    processGesture(event: ZRRawEvent, stage?: 'start' | 'end' | 'change') {
        if (!this._gestureMgr) {
            this._gestureMgr = new GestureMgr();
        }
        const gestureMgr = this._gestureMgr;

        stage === 'start' && gestureMgr.clear();

        const gestureInfo = gestureMgr.recognize(
            event as ZRRawTouchEvent,
            this.findHover(event.zrX, event.zrY, null).target,
            (this.proxy as HandlerDomProxy).dom
        );

        stage === 'end' && gestureMgr.clear();

        // Do not do any preventDefault here. Upper application do that if necessary.
        if (gestureInfo) {
            const type = gestureInfo.type;
            (event as ZRPinchEvent).gestureEvent = type;

            let res = new HoveredResult();
            res.target = gestureInfo.target;
            this.dispatchToElement(res, type as ElementEventName, gestureInfo.event as ZRRawEvent);
        }
    }

    click: (event: ZRRawEvent) => void
    mousedown: (event: ZRRawEvent) => void
    mouseup: (event: ZRRawEvent) => void
    mousewheel: (event: ZRRawEvent) => void
    dblclick: (event: ZRRawEvent) => void
    contextmenu: (event: ZRRawEvent) => void
}

// Common handlers
util.each(['click', 'mousedown', 'mouseup', 'mousewheel', 'dblclick', 'contextmenu'], function (name: HandlerName) {
    Handler.prototype[name] = function (event) {
        const x = event.zrX;
        const y = event.zrY;
        const isOutside = isOutsideBoundary(this, x, y);

        let hovered;
        let hoveredTarget;

        if (name !== 'mouseup' || !isOutside) {
            // Find hover again to avoid click event is dispatched manually. Or click is triggered without mouseover
            hovered = this.findHover(x, y);
            hoveredTarget = hovered.target;
        }

        if (name === 'mousedown') {
            this._downEl = hoveredTarget;
            this._downPoint = [event.zrX, event.zrY];
            // In case click triggered before mouseup
            this._upEl = hoveredTarget;
        }
        else if (name === 'mouseup') {
            this._upEl = hoveredTarget;
        }
        else if (name === 'click') {
            if (this._downEl !== this._upEl
                // Original click event is triggered on the whole canvas element,
                // including the case that `mousedown` - `mousemove` - `mouseup`,
                // which should be filtered, otherwise it will bring trouble to
                // pan and zoom.
                || !this._downPoint
                // Arbitrary value
                || vec2.dist(this._downPoint, [event.zrX, event.zrY]) > 4
            ) {
                return;
            }
            this._downPoint = null;
        }

        this.dispatchToElement(hovered, name, event);
    };
});

function isHover(displayable: Displayable, x: number, y: number) {
    if (displayable[displayable.rectHover ? 'rectContain' : 'contain'](x, y)) {
        let el: Element = displayable;
        let isSilent;
        let ignoreClip = false;
        while (el) {
            // Ignore clip on any ancestors.
            if (el.ignoreClip) {
                ignoreClip = true;
            }
            if (!ignoreClip) {
                let clipPath = el.getClipPath();
                // If clipped by ancestor.
                // FIXME: If clipPath has neither stroke nor fill,
                // el.clipPath.contain(x, y) will always return false.
                if (clipPath && !clipPath.contain(x, y)) {
                    return false;
                }
            }
            if (el.silent) {
                isSilent = true;
            }
            // Consider when el is textContent, also need to be silent
            // if any of its host el and its ancestors is silent.
            const hostEl = el.__hostTarget;
            el = hostEl ? hostEl : el.parent;
        }
        return isSilent ? SILENT : true;
    }

    return false;
}

function setHoverTarget(
    list: Displayable[],
    out: HoveredResult,
    x: number,
    y: number,
    exclude: Displayable
) {
    for (let i = list.length - 1; i >= 0; i--) {
        const el = list[i];
        let hoverCheckResult;
        if (el !== exclude
            // getDisplayList may include ignored item in VML mode
            && !el.ignore
            && (hoverCheckResult = isHover(el, x, y))
        ) {
            !out.topTarget && (out.topTarget = el);
            if (hoverCheckResult !== SILENT) {
                out.target = el;
                break;
            }
        }
    }
}

/**
 * See [DRAG_OUTSIDE].
 */
function isOutsideBoundary(handlerInstance: Handler, x: number, y: number) {
    const painter = handlerInstance.painter;
    return x < 0 || x > painter.getWidth() || y < 0 || y > painter.getHeight();
}

export default Handler;
