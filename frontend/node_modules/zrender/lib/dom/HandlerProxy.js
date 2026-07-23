import { __extends } from "tslib";
import { addEventListener, removeEventListener, normalizeEvent, getNativeEvent } from '../core/event.js';
import * as zrUtil from '../core/util.js';
import Eventful from '../core/Eventful.js';
import env from '../core/env.js';
var TOUCH_CLICK_DELAY = 300;
var globalEventSupported = env.domSupported;
var localNativeListenerNames = (function () {
    var mouseHandlerNames = [
        'click', 'dblclick', 'mousewheel', 'wheel', 'mouseout',
        'mouseup', 'mousedown', 'mousemove', 'contextmenu'
    ];
    var touchHandlerNames = [
        'touchstart', 'touchend', 'touchmove'
    ];
    var pointerEventNameMap = {
        pointerdown: 1, pointerup: 1, pointermove: 1, pointerout: 1
    };
    var pointerHandlerNames = zrUtil.map(mouseHandlerNames, function (name) {
        var nm = name.replace('mouse', 'pointer');
        return pointerEventNameMap.hasOwnProperty(nm) ? nm : name;
    });
    return {
        mouse: mouseHandlerNames,
        touch: touchHandlerNames,
        pointer: pointerHandlerNames
    };
})();
var globalNativeListenerNames = {
    mouse: ['mousemove', 'mouseup'],
    pointer: ['pointermove', 'pointerup']
};
var wheelEventSupported = false;
function isPointerFromTouch(event) {
    var pointerType = event.pointerType;
    return pointerType === 'pen' || pointerType === 'touch';
}
function setTouchTimer(scope) {
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
function markTouch(event) {
    event && (event.zrByTouch = true);
}
function normalizeGlobalEvent(instance, event) {
    return normalizeEvent(instance.dom, new FakeGlobalEvent(instance, event), true);
}
function isLocalEl(instance, el) {
    var elTmp = el;
    var isLocal = false;
    while (elTmp && elTmp.nodeType !== 9
        && !(isLocal = elTmp.domBelongToZr
            || (elTmp !== el && elTmp === instance.painterRoot))) {
        elTmp = elTmp.parentNode;
    }
    return isLocal;
}
var FakeGlobalEvent = (function () {
    function FakeGlobalEvent(instance, event) {
        this.stopPropagation = zrUtil.noop;
        this.stopImmediatePropagation = zrUtil.noop;
        this.preventDefault = zrUtil.noop;
        this.type = event.type;
        this.target = this.currentTarget = instance.dom;
        this.pointerType = event.pointerType;
        this.clientX = event.clientX;
        this.clientY = event.clientY;
    }
    return FakeGlobalEvent;
}());
var localDOMHandlers = {
    mousedown: function (event) {
        event = normalizeEvent(this.dom, event);
        this.__mayPointerCapture = [event.zrX, event.zrY];
        this.trigger('mousedown', event);
    },
    mousemove: function (event) {
        event = normalizeEvent(this.dom, event);
        var downPoint = this.__mayPointerCapture;
        if (downPoint && (event.zrX !== downPoint[0] || event.zrY !== downPoint[1])) {
            this.__togglePointerCapture(true);
        }
        this.trigger('mousemove', event);
    },
    mouseup: function (event) {
        event = normalizeEvent(this.dom, event);
        this.__togglePointerCapture(false);
        this.trigger('mouseup', event);
    },
    mouseout: function (event) {
        event = normalizeEvent(this.dom, event);
        var element = event.toElement || event.relatedTarget;
        if (!isLocalEl(this, element)) {
            if (this.__pointerCapturing) {
                event.zrEventControl = 'no_globalout';
            }
            this.trigger('mouseout', event);
        }
    },
    wheel: function (event) {
        wheelEventSupported = true;
        event = normalizeEvent(this.dom, event);
        this.trigger('mousewheel', event);
    },
    mousewheel: function (event) {
        if (wheelEventSupported) {
            return;
        }
        event = normalizeEvent(this.dom, event);
        this.trigger('mousewheel', event);
    },
    touchstart: function (event) {
        event = normalizeEvent(this.dom, event);
        markTouch(event);
        this.__lastTouchMoment = new Date();
        this.handler.processGesture(event, 'start');
        localDOMHandlers.mousemove.call(this, event);
        localDOMHandlers.mousedown.call(this, event);
    },
    touchmove: function (event) {
        event = normalizeEvent(this.dom, event);
        markTouch(event);
        this.handler.processGesture(event, 'change');
        localDOMHandlers.mousemove.call(this, event);
    },
    touchend: function (event) {
        event = normalizeEvent(this.dom, event);
        markTouch(event);
        this.handler.processGesture(event, 'end');
        localDOMHandlers.mouseup.call(this, event);
        if (+new Date() - (+this.__lastTouchMoment) < TOUCH_CLICK_DELAY) {
            localDOMHandlers.click.call(this, event);
        }
    },
    pointerdown: function (event) {
        localDOMHandlers.mousedown.call(this, event);
    },
    pointermove: function (event) {
        if (!isPointerFromTouch(event)) {
            localDOMHandlers.mousemove.call(this, event);
        }
    },
    pointerup: function (event) {
        localDOMHandlers.mouseup.call(this, event);
    },
    pointerout: function (event) {
        if (!isPointerFromTouch(event)) {
            localDOMHandlers.mouseout.call(this, event);
        }
    }
};
zrUtil.each(['click', 'dblclick', 'contextmenu'], function (name) {
    localDOMHandlers[name] = function (event) {
        event = normalizeEvent(this.dom, event);
        this.trigger(name, event);
    };
});
var globalDOMHandlers = {
    pointermove: function (event) {
        if (!isPointerFromTouch(event)) {
            globalDOMHandlers.mousemove.call(this, event);
        }
    },
    pointerup: function (event) {
        globalDOMHandlers.mouseup.call(this, event);
    },
    mousemove: function (event) {
        this.trigger('mousemove', event);
    },
    mouseup: function (event) {
        var pointerCaptureReleasing = this.__pointerCapturing;
        this.__togglePointerCapture(false);
        this.trigger('mouseup', event);
        if (pointerCaptureReleasing) {
            event.zrEventControl = 'only_globalout';
            this.trigger('mouseout', event);
        }
    }
};
function mountLocalDOMEventListeners(instance, scope) {
    var domHandlers = scope.domHandlers;
    if (env.pointerEventsSupported) {
        zrUtil.each(localNativeListenerNames.pointer, function (nativeEventName) {
            mountSingleDOMEventListener(scope, nativeEventName, function (event) {
                domHandlers[nativeEventName].call(instance, event);
            });
        });
    }
    else {
        if (env.touchEventsSupported) {
            zrUtil.each(localNativeListenerNames.touch, function (nativeEventName) {
                mountSingleDOMEventListener(scope, nativeEventName, function (event) {
                    domHandlers[nativeEventName].call(instance, event);
                    setTouchTimer(scope);
                });
            });
        }
        zrUtil.each(localNativeListenerNames.mouse, function (nativeEventName) {
            mountSingleDOMEventListener(scope, nativeEventName, function (event) {
                event = getNativeEvent(event);
                if (!scope.touching) {
                    domHandlers[nativeEventName].call(instance, event);
                }
            });
        });
    }
}
function mountGlobalDOMEventListeners(instance, scope) {
    if (env.pointerEventsSupported) {
        zrUtil.each(globalNativeListenerNames.pointer, mount);
    }
    else if (!env.touchEventsSupported) {
        zrUtil.each(globalNativeListenerNames.mouse, mount);
    }
    function mount(nativeEventName) {
        function nativeEventListener(event) {
            event = getNativeEvent(event);
            if (!isLocalEl(instance, event.target)) {
                event = normalizeGlobalEvent(instance, event);
                scope.domHandlers[nativeEventName].call(instance, event);
            }
        }
        mountSingleDOMEventListener(scope, nativeEventName, nativeEventListener, { capture: true });
    }
}
function mountSingleDOMEventListener(scope, nativeEventName, listener, opt) {
    scope.mounted[nativeEventName] = listener;
    scope.listenerOpts[nativeEventName] = opt;
    addEventListener(scope.domTarget, nativeEventName, listener, opt);
}
function unmountDOMEventListeners(scope) {
    var mounted = scope.mounted;
    for (var nativeEventName in mounted) {
        if (mounted.hasOwnProperty(nativeEventName)) {
            removeEventListener(scope.domTarget, nativeEventName, mounted[nativeEventName], scope.listenerOpts[nativeEventName]);
        }
    }
    scope.mounted = {};
}
var DOMHandlerScope = (function () {
    function DOMHandlerScope(domTarget, domHandlers) {
        this.mounted = {};
        this.listenerOpts = {};
        this.touching = false;
        this.domTarget = domTarget;
        this.domHandlers = domHandlers;
    }
    return DOMHandlerScope;
}());
var HandlerDomProxy = (function (_super) {
    __extends(HandlerDomProxy, _super);
    function HandlerDomProxy(dom, painterRoot) {
        var _this = _super.call(this) || this;
        _this.__pointerCapturing = false;
        _this.dom = dom;
        _this.painterRoot = painterRoot;
        _this._localHandlerScope = new DOMHandlerScope(dom, localDOMHandlers);
        if (globalEventSupported) {
            _this._globalHandlerScope = new DOMHandlerScope(document, globalDOMHandlers);
        }
        mountLocalDOMEventListeners(_this, _this._localHandlerScope);
        return _this;
    }
    HandlerDomProxy.prototype.dispose = function () {
        unmountDOMEventListeners(this._localHandlerScope);
        if (globalEventSupported) {
            unmountDOMEventListeners(this._globalHandlerScope);
        }
    };
    HandlerDomProxy.prototype.setCursor = function (cursorStyle) {
        this.dom.style && (this.dom.style.cursor = cursorStyle || 'default');
    };
    HandlerDomProxy.prototype.__togglePointerCapture = function (isPointerCapturing) {
        this.__mayPointerCapture = null;
        if (globalEventSupported
            && ((+this.__pointerCapturing) ^ (+isPointerCapturing))) {
            this.__pointerCapturing = isPointerCapturing;
            var globalHandlerScope = this._globalHandlerScope;
            isPointerCapturing
                ? mountGlobalDOMEventListeners(this, globalHandlerScope)
                : unmountDOMEventListeners(globalHandlerScope);
        }
    };
    return HandlerDomProxy;
}(Eventful));
export default HandlerDomProxy;
