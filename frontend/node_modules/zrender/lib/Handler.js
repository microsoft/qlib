import { __extends } from "tslib";
import * as util from './core/util.js';
import * as vec2 from './core/vector.js';
import Draggable from './mixin/Draggable.js';
import Eventful from './core/Eventful.js';
import * as eventTool from './core/event.js';
import { GestureMgr } from './core/GestureMgr.js';
import BoundingRect from './core/BoundingRect.js';
var SILENT = 'silent';
function makeEventPacket(eveType, targetInfo, event) {
    return {
        type: eveType,
        event: event,
        target: targetInfo.target,
        topTarget: targetInfo.topTarget,
        cancelBubble: false,
        offsetX: event.zrX,
        offsetY: event.zrY,
        gestureEvent: event.gestureEvent,
        pinchX: event.pinchX,
        pinchY: event.pinchY,
        pinchScale: event.pinchScale,
        wheelDelta: event.zrDelta,
        zrByTouch: event.zrByTouch,
        which: event.which,
        stop: stopEvent
    };
}
function stopEvent() {
    eventTool.stop(this.event);
}
var EmptyProxy = (function (_super) {
    __extends(EmptyProxy, _super);
    function EmptyProxy() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.handler = null;
        return _this;
    }
    EmptyProxy.prototype.dispose = function () { };
    EmptyProxy.prototype.setCursor = function () { };
    return EmptyProxy;
}(Eventful));
var HoveredResult = (function () {
    function HoveredResult(x, y) {
        this.x = x;
        this.y = y;
    }
    return HoveredResult;
}());
var handlerNames = [
    'click', 'dblclick', 'mousewheel', 'mouseout',
    'mouseup', 'mousedown', 'mousemove', 'contextmenu'
];
var tmpRect = new BoundingRect(0, 0, 0, 0);
var Handler = (function (_super) {
    __extends(Handler, _super);
    function Handler(storage, painter, proxy, painterRoot, pointerSize) {
        var _this = _super.call(this) || this;
        _this._hovered = new HoveredResult(0, 0);
        _this.storage = storage;
        _this.painter = painter;
        _this.painterRoot = painterRoot;
        _this._pointerSize = pointerSize;
        proxy = proxy || new EmptyProxy();
        _this.proxy = null;
        _this.setHandlerProxy(proxy);
        _this._draggingMgr = new Draggable(_this);
        return _this;
    }
    Handler.prototype.setHandlerProxy = function (proxy) {
        if (this.proxy) {
            this.proxy.dispose();
        }
        if (proxy) {
            util.each(handlerNames, function (name) {
                proxy.on && proxy.on(name, this[name], this);
            }, this);
            proxy.handler = this;
        }
        this.proxy = proxy;
    };
    Handler.prototype.mousemove = function (event) {
        var x = event.zrX;
        var y = event.zrY;
        var isOutside = isOutsideBoundary(this, x, y);
        var lastHovered = this._hovered;
        var lastHoveredTarget = lastHovered.target;
        if (lastHoveredTarget && !lastHoveredTarget.__zr) {
            lastHovered = this.findHover(lastHovered.x, lastHovered.y);
            lastHoveredTarget = lastHovered.target;
        }
        var hovered = this._hovered = isOutside ? new HoveredResult(x, y) : this.findHover(x, y);
        var hoveredTarget = hovered.target;
        var proxy = this.proxy;
        proxy.setCursor && proxy.setCursor(hoveredTarget ? hoveredTarget.cursor : 'default');
        if (lastHoveredTarget && hoveredTarget !== lastHoveredTarget) {
            this.dispatchToElement(lastHovered, 'mouseout', event);
        }
        this.dispatchToElement(hovered, 'mousemove', event);
        if (hoveredTarget && hoveredTarget !== lastHoveredTarget) {
            this.dispatchToElement(hovered, 'mouseover', event);
        }
    };
    Handler.prototype.mouseout = function (event) {
        var eventControl = event.zrEventControl;
        if (eventControl !== 'only_globalout') {
            this.dispatchToElement(this._hovered, 'mouseout', event);
        }
        if (eventControl !== 'no_globalout') {
            this.trigger('globalout', { type: 'globalout', event: event });
        }
    };
    Handler.prototype.resize = function () {
        this._hovered = new HoveredResult(0, 0);
    };
    Handler.prototype.dispatch = function (eventName, eventArgs) {
        var handler = this[eventName];
        handler && handler.call(this, eventArgs);
    };
    Handler.prototype.dispose = function () {
        this.proxy.dispose();
        this.storage = null;
        this.proxy = null;
        this.painter = null;
    };
    Handler.prototype.setCursorStyle = function (cursorStyle) {
        var proxy = this.proxy;
        proxy.setCursor && proxy.setCursor(cursorStyle);
    };
    Handler.prototype.dispatchToElement = function (targetInfo, eventName, event) {
        targetInfo = targetInfo || {};
        var el = targetInfo.target;
        if (el && el.silent) {
            return;
        }
        var eventKey = ('on' + eventName);
        var eventPacket = makeEventPacket(eventName, targetInfo, event);
        while (el) {
            el[eventKey]
                && (eventPacket.cancelBubble = !!el[eventKey].call(el, eventPacket));
            el.trigger(eventName, eventPacket);
            el = el.__hostTarget ? el.__hostTarget : el.parent;
            if (eventPacket.cancelBubble) {
                break;
            }
        }
        if (!eventPacket.cancelBubble) {
            this.trigger(eventName, eventPacket);
            if (this.painter && this.painter.eachOtherLayer) {
                this.painter.eachOtherLayer(function (layer) {
                    if (typeof (layer[eventKey]) === 'function') {
                        layer[eventKey].call(layer, eventPacket);
                    }
                    if (layer.trigger) {
                        layer.trigger(eventName, eventPacket);
                    }
                });
            }
        }
    };
    Handler.prototype.findHover = function (x, y, exclude) {
        var list = this.storage.getDisplayList();
        var out = new HoveredResult(x, y);
        setHoverTarget(list, out, x, y, exclude);
        if (this._pointerSize && !out.target) {
            var candidates = [];
            var pointerSize = this._pointerSize;
            var targetSizeHalf = pointerSize / 2;
            var pointerRect = new BoundingRect(x - targetSizeHalf, y - targetSizeHalf, pointerSize, pointerSize);
            for (var i = list.length - 1; i >= 0; i--) {
                var el = list[i];
                if (el !== exclude
                    && !el.ignore
                    && !el.ignoreCoarsePointer
                    && (!el.parent || !el.parent.ignoreCoarsePointer)) {
                    tmpRect.copy(el.getBoundingRect());
                    if (el.transform) {
                        tmpRect.applyTransform(el.transform);
                    }
                    if (tmpRect.intersect(pointerRect)) {
                        candidates.push(el);
                    }
                }
            }
            if (candidates.length) {
                var rStep = 4;
                var thetaStep = Math.PI / 12;
                var PI2 = Math.PI * 2;
                for (var r = 0; r < targetSizeHalf; r += rStep) {
                    for (var theta = 0; theta < PI2; theta += thetaStep) {
                        var x1 = x + r * Math.cos(theta);
                        var y1 = y + r * Math.sin(theta);
                        setHoverTarget(candidates, out, x1, y1, exclude);
                        if (out.target) {
                            return out;
                        }
                    }
                }
            }
        }
        return out;
    };
    Handler.prototype.processGesture = function (event, stage) {
        if (!this._gestureMgr) {
            this._gestureMgr = new GestureMgr();
        }
        var gestureMgr = this._gestureMgr;
        stage === 'start' && gestureMgr.clear();
        var gestureInfo = gestureMgr.recognize(event, this.findHover(event.zrX, event.zrY, null).target, this.proxy.dom);
        stage === 'end' && gestureMgr.clear();
        if (gestureInfo) {
            var type = gestureInfo.type;
            event.gestureEvent = type;
            var res = new HoveredResult();
            res.target = gestureInfo.target;
            this.dispatchToElement(res, type, gestureInfo.event);
        }
    };
    return Handler;
}(Eventful));
util.each(['click', 'mousedown', 'mouseup', 'mousewheel', 'dblclick', 'contextmenu'], function (name) {
    Handler.prototype[name] = function (event) {
        var x = event.zrX;
        var y = event.zrY;
        var isOutside = isOutsideBoundary(this, x, y);
        var hovered;
        var hoveredTarget;
        if (name !== 'mouseup' || !isOutside) {
            hovered = this.findHover(x, y);
            hoveredTarget = hovered.target;
        }
        if (name === 'mousedown') {
            this._downEl = hoveredTarget;
            this._downPoint = [event.zrX, event.zrY];
            this._upEl = hoveredTarget;
        }
        else if (name === 'mouseup') {
            this._upEl = hoveredTarget;
        }
        else if (name === 'click') {
            if (this._downEl !== this._upEl
                || !this._downPoint
                || vec2.dist(this._downPoint, [event.zrX, event.zrY]) > 4) {
                return;
            }
            this._downPoint = null;
        }
        this.dispatchToElement(hovered, name, event);
    };
});
function isHover(displayable, x, y) {
    if (displayable[displayable.rectHover ? 'rectContain' : 'contain'](x, y)) {
        var el = displayable;
        var isSilent = void 0;
        var ignoreClip = false;
        while (el) {
            if (el.ignoreClip) {
                ignoreClip = true;
            }
            if (!ignoreClip) {
                var clipPath = el.getClipPath();
                if (clipPath && !clipPath.contain(x, y)) {
                    return false;
                }
            }
            if (el.silent) {
                isSilent = true;
            }
            var hostEl = el.__hostTarget;
            el = hostEl ? hostEl : el.parent;
        }
        return isSilent ? SILENT : true;
    }
    return false;
}
function setHoverTarget(list, out, x, y, exclude) {
    for (var i = list.length - 1; i >= 0; i--) {
        var el = list[i];
        var hoverCheckResult = void 0;
        if (el !== exclude
            && !el.ignore
            && (hoverCheckResult = isHover(el, x, y))) {
            !out.topTarget && (out.topTarget = el);
            if (hoverCheckResult !== SILENT) {
                out.target = el;
                break;
            }
        }
    }
}
function isOutsideBoundary(handlerInstance, x, y) {
    var painter = handlerInstance.painter;
    return x < 0 || x > painter.getWidth() || y < 0 || y > painter.getHeight();
}
export default Handler;
