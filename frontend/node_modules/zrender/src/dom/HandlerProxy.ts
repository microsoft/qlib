
/* global document */

import {
    addEventListener,
    removeEventListener,
    normalizeEvent,
    getNativeEvent
} from '../core/event';
import * as zrUtil from '../core/util';
import Eventful from '../core/Eventful';
import env from '../core/env';
import { Dictionary, ZRRawEvent, ZRRawMouseEvent } from '../core/types';
import { VectorArray } from '../core/vector';
import Handler from '../Handler';

type DomHandlersMap = Dictionary<(this: HandlerDomProxy, event: ZRRawEvent) => void>

type DomExtended = Node & {
    domBelongToZr: boolean
}

const TOUCH_CLICK_DELAY = 300;

const globalEventSupported = env.domSupported;


const localNativeListenerNames = (function () {
    const mouseHandlerNames = [
        'click', 'dblclick', 'mousewheel', 'wheel', 'mouseout',
        'mouseup', 'mousedown', 'mousemove', 'contextmenu'
    ];
    const touchHandlerNames = [
        'touchstart', 'touchend', 'touchmove'
    ];
    const pointerEventNameMap = {
        pointerdown: 1, pointerup: 1, pointermove: 1, pointerout: 1
    };
    const pointerHandlerNames = zrUtil.map(mouseHandlerNames, function (name) {
        const nm = name.replace('mouse', 'pointer');
        return pointerEventNameMap.hasOwnProperty(nm) ? nm : name;
    });

    return {
        mouse: mouseHandlerNames,
        touch: touchHandlerNames,
        pointer: pointerHandlerNames
    };
})();

const globalNativeListenerNames = {
    mouse: ['mousemove', 'mouseup'],
    pointer: ['pointermove', 'pointerup']
};

let wheelEventSupported = false;


// Although firfox has 'DOMMouseScroll' event and do not has 'mousewheel' event,
// the 'DOMMouseScroll' event do not performe the same behavior on touch pad device
// (like on Mac) ('DOMMouseScroll' will be triggered only if a big wheel delta).
// So we should not use it.
// function eventNameFix(name: string) {
//     return (name === 'mousewheel' && env.browser.firefox) ? 'DOMMouseScroll' : name;
// }

function isPointerFromTouch(event: ZRRawEvent) {
    const pointerType = (event as any).pointerType;
    return pointerType === 'pen' || pointerType === 'touch';
}

// function useMSGuesture(handlerProxy, event) {
//     return isPointerFromTouch(event) && !!handlerProxy._msGesture;
// }

// function onMSGestureChange(proxy, event) {
//     if (event.translationX || event.translationY) {
//         // mousemove is carried by MSGesture to reduce the sensitivity.
//         proxy.handler.dispatchToElement(event.target, 'mousemove', event);
//     }
//     if (event.scale !== 1) {
//         event.pinchX = event.offsetX;
//         event.pinchY = event.offsetY;
//         event.pinchScale = event.scale;
//         proxy.handler.dispatchToElement(event.target, 'pinch', event);
//     }
// }

/**
 * Prevent mouse event from being dispatched after Touch Events action
 * @see <https://github.com/deltakosh/handjs/blob/master/src/hand.base.js>
 * 1. Mobile browsers dispatch mouse events 300ms after touchend.
 * 2. Chrome for Android dispatch mousedown for long-touch about 650ms
 * Result: Blocking Mouse Events for 700ms.
 *
 * @param {DOMHandlerScope} scope
 */
function setTouchTimer(scope: DOMHandlerScope) {
    scope.touching = true;
    if (scope.touchTimer != null) {
        clearTimeout(scope.touchTimer);
        scope.touchTimer = null;
    }
    scope.touchTimer = setTimeout(function () {
        scope.touching = false;
        scope.touchTimer = null;
    }, 700);
}

// Mark touch, which is useful in distinguish touch and
// mouse event in upper applicatoin.
function markTouch(event: ZRRawEvent) {
    event && (event.zrByTouch = true);
}


// function markTriggeredFromLocal(event) {
//     event && (event.__zrIsFromLocal = true);
// }

// function isTriggeredFromLocal(instance, event) {
//     return !!(event && event.__zrIsFromLocal);
// }

function normalizeGlobalEvent(instance: HandlerDomProxy, event: ZRRawEvent) {
    // offsetX, offsetY still need to be calculated. They are necessary in the event
    // handlers of the upper applications. Set `true` to force calculate them.
    return normalizeEvent(
        instance.dom,
        // TODO ANY TYPE
        new FakeGlobalEvent(instance, event) as any as ZRRawEvent,
        true
    );
}

/**
 * Detect whether the given el is in `painterRoot`.
 */
function isLocalEl(instance: HandlerDomProxy, el: Node) {
    let elTmp = el;
    let isLocal = false;
    while (elTmp && elTmp.nodeType !== 9
        && !(
            isLocal = (elTmp as DomExtended).domBelongToZr
                || (elTmp !== el && elTmp === instance.painterRoot)
        )
    ) {
        elTmp = elTmp.parentNode;
    }
    return isLocal;
}

/**
 * Make a fake event but not change the original event,
 * because the global event probably be used by other
 * listeners not belonging to zrender.
 * @class
 */
class FakeGlobalEvent {
    type: string
    target: HTMLElement
    currentTarget: HTMLElement

    pointerType: string
    clientX: number
    clientY: number

    constructor(instance: HandlerDomProxy, event: ZRRawEvent) {
        this.type = event.type;
        this.target = this.currentTarget = instance.dom;
        this.pointerType = (event as any).pointerType;
        // Necessray for the force calculation of zrX, zrY
        this.clientX = (event as ZRRawMouseEvent).clientX;
        this.clientY = (event as ZRRawMouseEvent).clientY;
        // Because we do not mount global listeners to touch events,
        // we do not copy `targetTouches` and `changedTouches` here.
    }

    // we make the default methods on the event do nothing,
    // otherwise it is dangerous. See more details in
    // [DRAG_OUTSIDE] in `Handler.js`.
    stopPropagation = zrUtil.noop
    stopImmediatePropagation = zrUtil.noop
    preventDefault = zrUtil.noop
}


/**
 * Local DOM Handlers
 * @this {HandlerProxy}
 */
const localDOMHandlers: DomHandlersMap = {

    mousedown(event: ZRRawEvent) {
        event = normalizeEvent(this.dom, event);

        this.__mayPointerCapture = [event.zrX, event.zrY];

        this.trigger('mousedown', event);
    },

    mousemove(event: ZRRawEvent) {
        event = normalizeEvent(this.dom, event);

        const downPoint = this.__mayPointerCapture;
        if (downPoint && (event.zrX !== downPoint[0] || event.zrY !== downPoint[1])) {
            this.__togglePointerCapture(true);
        }

        this.trigger('mousemove', event);
    },

    mouseup(event: ZRRawEvent) {
        event = normalizeEvent(this.dom, event);

        this.__togglePointerCapture(false);

        this.trigger('mouseup', event);
    },

    mouseout(event: ZRRawEvent) {
        event = normalizeEvent(this.dom, event);

        // There might be some doms created by upper layer application
        // at the same level of painter.getViewportRoot() (e.g., tooltip
        // dom created by echarts), where 'globalout' event should not
        // be triggered when mouse enters these doms. (But 'mouseout'
        // should be triggered at the original hovered element as usual).
        const element = (event as any).toElement || (event as ZRRawMouseEvent).relatedTarget;

        // For SVG rendering, there are SVG elements inside `this.dom`.
        // (especially in decal case). Should not to handle those "mouseout"..
        if (!isLocalEl(this, element)) {
            // Similarly to the browser did on `document` and touch event,
            // `globalout` will be delayed to final pointer cature release.
            if (this.__pointerCapturing) {
                event.zrEventControl = 'no_globalout';
            }

            this.trigger('mouseout', event);
        }
    },

    wheel(event: ZRRawEvent) {
        // Morden agent has supported event `wheel` instead of `mousewheel`.
        // About the polyfill of the props "delta", see "arc/core/event.ts".

        // Firefox only support `wheel` rather than `mousewheel`. Although firfox has been supporting
        // event `DOMMouseScroll`, it do not act the same behavior as `wheel` on touch pad device
        // like on Mac, where `DOMMouseScroll` will be triggered only if a big wheel delta occurs,
        // and it results in no chance to "preventDefault". So we should not use `DOMMouseScroll`.

        wheelEventSupported = true;
        event = normalizeEvent(this.dom, event);
        // Follow the definition of the previous version, the zrender event name is still 'mousewheel'.
        this.trigger('mousewheel', event);
    },

    mousewheel(event: ZRRawEvent) {
        // IE8- and some other lagacy agent do not support event `wheel`, so we still listen
        // to the legacy event `mouseevent`.
        // Typically if event `wheel` is supported and the handler has been mounted on a
        // DOM element, the legacy `mousewheel` event will not be triggered (Chrome and Safari).
        // But we still do this guard to avoid to duplicated handle.
        if (wheelEventSupported) {
            return;
        }
        event = normalizeEvent(this.dom, event);
        this.trigger('mousewheel', event);
    },

    touchstart(event: ZRRawEvent) {
        // Default mouse behaviour should not be disabled here.
        // For example, page may needs to be slided.
        event = normalizeEvent(this.dom, event);

        markTouch(event);

        this.__lastTouchMoment = new Date();

        this.handler.processGesture(event, 'start');

        // For consistent event listener for both touch device and mouse device,
        // we simulate "mouseover-->mousedown" in touch device. So we trigger
        // `mousemove` here (to trigger `mouseover` inside), and then trigger
        // `mousedown`.
        localDOMHandlers.mousemove.call(this, event);
        localDOMHandlers.mousedown.call(this, event);
    },

    touchmove(event: ZRRawEvent) {
        event = normalizeEvent(this.dom, event);

        markTouch(event);

        this.handler.processGesture(event, 'change');

        // Mouse move should always be triggered no matter whether
        // there is gestrue event, because mouse move and pinch may
        // be used at the same time.
        localDOMHandlers.mousemove.call(this, event);
    },

    touchend(event: ZRRawEvent) {
        event = normalizeEvent(this.dom, event);

        markTouch(event);

        this.handler.processGesture(event, 'end');

        localDOMHandlers.mouseup.call(this, event);

        // Do not trigger `mouseout` here, in spite of `mousemove`(`mouseover`) is
        // triggered in `touchstart`. This seems to be illogical, but by this mechanism,
        // we can conveniently implement "hover style" in both PC and touch device just
        // by listening to `mouseover` to add "hover style" and listening to `mouseout`
        // to remove "hover style" on an element, without any additional code for
        // compatibility. (`mouseout` will not be triggered in `touchend`, so "hover
        // style" will remain for user view)

        // click event should always be triggered no matter whether
        // there is gestrue event. System click can not be prevented.
        if (+new Date() - (+this.__lastTouchMoment) < TOUCH_CLICK_DELAY) {
            localDOMHandlers.click.call(this, event);
        }
    },

    pointerdown(event: ZRRawEvent) {
        localDOMHandlers.mousedown.call(this, event);

        // if (useMSGuesture(this, event)) {
        //     this._msGesture.addPointer(event.pointerId);
        // }
    },

    pointermove(event: ZRRawEvent) {
        // FIXME
        // pointermove is so sensitive that it always triggered when
        // tap(click) on touch screen, which affect some judgement in
        // upper application. So, we don't support mousemove on MS touch
        // device yet.
        if (!isPointerFromTouch(event)) {
            localDOMHandlers.mousemove.call(this, event);
        }
    },

    pointerup(event: ZRRawEvent) {
        localDOMHandlers.mouseup.call(this, event);
    },

    pointerout(event: ZRRawEvent) {
        // pointerout will be triggered when tap on touch screen
        // (IE11+/Edge on MS Surface) after click event triggered,
        // which is inconsistent with the mousout behavior we defined
        // in touchend. So we unify them.
        // (check localDOMHandlers.touchend for detailed explanation)
        if (!isPointerFromTouch(event)) {
            localDOMHandlers.mouseout.call(this, event);
        }
    }

};

/**
 * Othere DOM UI Event handlers for zr dom.
 * @this {HandlerProxy}
 */
zrUtil.each(['click', 'dblclick', 'contextmenu'], function (name) {
    localDOMHandlers[name] = function (event) {
        event = normalizeEvent(this.dom, event);
        this.trigger(name, event);
    };
});


/**
 * DOM UI Event handlers for global page.
 *
 * [Caution]:
 * those handlers should both support in capture phase and bubble phase!
 */
const globalDOMHandlers: DomHandlersMap = {

    pointermove: function (event: ZRRawEvent) {
        // FIXME
        // pointermove is so sensitive that it always triggered when
        // tap(click) on touch screen, which affect some judgement in
        // upper application. So, we don't support mousemove on MS touch
        // device yet.
        if (!isPointerFromTouch(event)) {
            globalDOMHandlers.mousemove.call(this, event);
        }
    },

    pointerup: function (event: ZRRawEvent) {
        globalDOMHandlers.mouseup.call(this, event);
    },

    mousemove: function (event: ZRRawEvent) {
        this.trigger('mousemove', event);
    },

    mouseup: function (event: ZRRawEvent) {
        const pointerCaptureReleasing = this.__pointerCapturing;

        this.__togglePointerCapture(false);

        this.trigger('mouseup', event);

        if (pointerCaptureReleasing) {
            event.zrEventControl = 'only_globalout';
            this.trigger('mouseout', event);
        }
    }

};


function mountLocalDOMEventListeners(instance: HandlerDomProxy, scope: DOMHandlerScope) {
    const domHandlers = scope.domHandlers;

    if (env.pointerEventsSupported) { // Only IE11+/Edge
        // 1. On devices that both enable touch and mouse (e.g., MS Surface and lenovo X240),
        // IE11+/Edge do not trigger touch event, but trigger pointer event and mouse event
        // at the same time.
        // 2. On MS Surface, it probablely only trigger mousedown but no mouseup when tap on
        // screen, which do not occurs in pointer event.
        // So we use pointer event to both detect touch gesture and mouse behavior.
        zrUtil.each(localNativeListenerNames.pointer, function (nativeEventName) {
            mountSingleDOMEventListener(scope, nativeEventName, function (event) {
                // markTriggeredFromLocal(event);
                domHandlers[nativeEventName].call(instance, event);
            });
        });

        // FIXME
        // Note: MS Gesture require CSS touch-action set. But touch-action is not reliable,
        // which does not prevent defuault behavior occasionally (which may cause view port
        // zoomed in but use can not zoom it back). And event.preventDefault() does not work.
        // So we have to not to use MSGesture and not to support touchmove and pinch on MS
        // touch screen. And we only support click behavior on MS touch screen now.

        // MS Gesture Event is only supported on IE11+/Edge and on Windows 8+.
        // We don't support touch on IE on win7.
        // See <https://msdn.microsoft.com/en-us/library/dn433243(v=vs.85).aspx>
        // if (typeof MSGesture === 'function') {
        //     (this._msGesture = new MSGesture()).target = dom; // jshint ignore:line
        //     dom.addEventListener('MSGestureChange', onMSGestureChange);
        // }
    }
    else {
        if (env.touchEventsSupported) {
            zrUtil.each(localNativeListenerNames.touch, function (nativeEventName) {
                mountSingleDOMEventListener(scope, nativeEventName, function (event) {
                    // markTriggeredFromLocal(event);
                    domHandlers[nativeEventName].call(instance, event);
                    setTouchTimer(scope);
                });
            });
            // Handler of 'mouseout' event is needed in touch mode, which will be mounted below.
            // addEventListener(root, 'mouseout', this._mouseoutHandler);
        }

        // 1. Considering some devices that both enable touch and mouse event (like on MS Surface
        // and lenovo X240, @see #2350), we make mouse event be always listened, otherwise
        // mouse event can not be handle in those devices.
        // 2. On MS Surface, Chrome will trigger both touch event and mouse event. How to prevent
        // mouseevent after touch event triggered, see `setTouchTimer`.
        zrUtil.each(localNativeListenerNames.mouse, function (nativeEventName) {
            mountSingleDOMEventListener(scope, nativeEventName, function (event: ZRRawEvent) {
                event = getNativeEvent(event);
                if (!scope.touching) {
                    // markTriggeredFromLocal(event);
                    domHandlers[nativeEventName].call(instance, event);
                }
            });
        });
    }
}

function mountGlobalDOMEventListeners(instance: HandlerDomProxy, scope: DOMHandlerScope) {
    // Only IE11+/Edge. See the comment in `mountLocalDOMEventListeners`.
    if (env.pointerEventsSupported) {
        zrUtil.each(globalNativeListenerNames.pointer, mount);
    }
    // Touch event has implemented "drag outside" so we do not mount global listener for touch event.
    // (see https://www.w3.org/TR/touch-events/#the-touchmove-event) (see also `DRAG_OUTSIDE`).
    // We do not consider "both-support-touch-and-mouse device" for this feature (see the comment of
    // `mountLocalDOMEventListeners`) to avoid bugs util some requirements come.
    else if (!env.touchEventsSupported) {
        zrUtil.each(globalNativeListenerNames.mouse, mount);
    }

    function mount(nativeEventName: string) {
        function nativeEventListener(event: ZRRawEvent) {
            event = getNativeEvent(event);
            // See the reason in [DRAG_OUTSIDE] in `Handler.js`
            // This checking supports both `useCapture` or not.
            // PENDING: if there is performance issue in some devices,
            // we probably can not use `useCapture` and change a easier
            // to judes whether local (mark).
            if (!isLocalEl(instance, event.target as Node)) {
                event = normalizeGlobalEvent(instance, event);
                scope.domHandlers[nativeEventName].call(instance, event);
            }
        }
        mountSingleDOMEventListener(
            scope, nativeEventName, nativeEventListener,
            {capture: true} // See [DRAG_OUTSIDE] in `Handler.js`
        );
    }
}

function mountSingleDOMEventListener(
    scope: DOMHandlerScope,
    nativeEventName: string,
    listener: EventListener,
    opt?: boolean | AddEventListenerOptions
) {
    scope.mounted[nativeEventName] = listener;
    scope.listenerOpts[nativeEventName] = opt;
    addEventListener(scope.domTarget, nativeEventName, listener, opt);
}

function unmountDOMEventListeners(scope: DOMHandlerScope) {
    const mounted = scope.mounted;
    for (let nativeEventName in mounted) {
        if (mounted.hasOwnProperty(nativeEventName)) {
            removeEventListener(
                scope.domTarget, nativeEventName, mounted[nativeEventName],
                scope.listenerOpts[nativeEventName]
            );
        }
    }
    scope.mounted = {};
}


class DOMHandlerScope {
    domTarget: HTMLElement | HTMLDocument
    domHandlers: DomHandlersMap

    // Key: eventName, value: mounted handler functions.
    // Used for unmount.
    mounted: Dictionary<EventListener> = {};

    listenerOpts: Dictionary<boolean | AddEventListenerOptions> = {};

    touchTimer: ReturnType<typeof setTimeout>;
    touching = false;

    constructor(
        domTarget: HTMLElement | HTMLDocument,
        domHandlers: DomHandlersMap
    ) {
        this.domTarget = domTarget;
        this.domHandlers = domHandlers;

    }
}


export default class HandlerDomProxy extends Eventful {

    dom: HTMLElement
    painterRoot: HTMLElement

    handler: Handler

    private _localHandlerScope: DOMHandlerScope
    private _globalHandlerScope: DOMHandlerScope

    __lastTouchMoment: Date

    // See [DRAG_OUTSIDE] in `Handler.ts`.
    __pointerCapturing = false
    // [x, y]
    __mayPointerCapture: VectorArray


    constructor(dom: HTMLElement, painterRoot: HTMLElement) {
        super();

        this.dom = dom;
        this.painterRoot = painterRoot;

        this._localHandlerScope = new DOMHandlerScope(dom, localDOMHandlers);

        if (globalEventSupported) {
            this._globalHandlerScope = new DOMHandlerScope(document, globalDOMHandlers);
        }

        mountLocalDOMEventListeners(this, this._localHandlerScope);
    }

    dispose() {
        unmountDOMEventListeners(this._localHandlerScope);
        if (globalEventSupported) {
            unmountDOMEventListeners(this._globalHandlerScope);
        }
    }

    setCursor(cursorStyle: string) {
        this.dom.style && (this.dom.style.cursor = cursorStyle || 'default');
    }

    /**
     * See [DRAG_OUTSIDE] in `Handler.js`.
     * @implement
     * @param isPointerCapturing Should never be `null`/`undefined`.
     *        `true`: start to capture pointer if it is not capturing.
     *        `false`: end the capture if it is capturing.
     */
    __togglePointerCapture(isPointerCapturing?: boolean) {
        this.__mayPointerCapture = null;

        if (globalEventSupported
            && ((+this.__pointerCapturing) ^ (+isPointerCapturing))
        ) {
            this.__pointerCapturing = isPointerCapturing;

            const globalHandlerScope = this._globalHandlerScope;
            isPointerCapturing
                ? mountGlobalDOMEventListeners(this, globalHandlerScope)
                : unmountDOMEventListeners(globalHandlerScope);
        }
    }
}

export interface HandlerProxyInterface extends Eventful {
    handler: Handler
    dispose: () => void
    setCursor: (cursorStyle?: string) => void
}