/*!
* ZRender, a high performance 2d drawing library.
*
* Copyright (c) 2013, Baidu Inc.
* All rights reserved.
*
* LICENSE
* https://github.com/ecomfe/zrender/blob/master/LICENSE.txt
*/
import env from './core/env.js';
import * as zrUtil from './core/util.js';
import Handler from './Handler.js';
import Storage from './Storage.js';
import Animation, { getTime } from './animation/Animation.js';
import HandlerProxy from './dom/HandlerProxy.js';
import { lum } from './tool/color.js';
import { DARK_MODE_THRESHOLD } from './config.js';
import Group from './graphic/Group.js';
var painterCtors = {};
var instances = {};
function delInstance(id) {
    delete instances[id];
}
function isDarkMode(backgroundColor) {
    if (!backgroundColor) {
        return false;
    }
    if (typeof backgroundColor === 'string') {
        return lum(backgroundColor, 1) < DARK_MODE_THRESHOLD;
    }
    else if (backgroundColor.colorStops) {
        var colorStops = backgroundColor.colorStops;
        var totalLum = 0;
        var len = colorStops.length;
        for (var i = 0; i < len; i++) {
            totalLum += lum(colorStops[i].color, 1);
        }
        totalLum /= len;
        return totalLum < DARK_MODE_THRESHOLD;
    }
    return false;
}
var ZRender = (function () {
    function ZRender(id, dom, opts) {
        var _this = this;
        this._sleepAfterStill = 10;
        this._stillFrameAccum = 0;
        this._needsRefresh = true;
        this._needsRefreshHover = true;
        this._darkMode = false;
        opts = opts || {};
        this.dom = dom;
        this.id = id;
        var storage = new Storage();
        var rendererType = opts.renderer || 'canvas';
        if (!painterCtors[rendererType]) {
            rendererType = zrUtil.keys(painterCtors)[0];
        }
        if (process.env.NODE_ENV !== 'production') {
            if (!painterCtors[rendererType]) {
                throw new Error("Renderer '" + rendererType + "' is not imported. Please import it first.");
            }
        }
        opts.useDirtyRect = opts.useDirtyRect == null
            ? false
            : opts.useDirtyRect;
        var painter = new painterCtors[rendererType](dom, storage, opts, id);
        var ssrMode = opts.ssr || painter.ssrOnly;
        this.storage = storage;
        this.painter = painter;
        var handlerProxy = (!env.node && !env.worker && !ssrMode)
            ? new HandlerProxy(painter.getViewportRoot(), painter.root)
            : null;
        var useCoarsePointer = opts.useCoarsePointer;
        var usePointerSize = (useCoarsePointer == null || useCoarsePointer === 'auto')
            ? env.touchEventsSupported
            : !!useCoarsePointer;
        var defaultPointerSize = 44;
        var pointerSize;
        if (usePointerSize) {
            pointerSize = zrUtil.retrieve2(opts.pointerSize, defaultPointerSize);
        }
        this.handler = new Handler(storage, painter, handlerProxy, painter.root, pointerSize);
        this.animation = new Animation({
            stage: {
                update: ssrMode ? null : function () { return _this._flush(true); }
            }
        });
        if (!ssrMode) {
            this.animation.start();
        }
    }
    ZRender.prototype.add = function (el) {
        if (this._disposed || !el) {
            return;
        }
        this.storage.addRoot(el);
        el.addSelfToZr(this);
        this.refresh();
    };
    ZRender.prototype.remove = function (el) {
        if (this._disposed || !el) {
            return;
        }
        this.storage.delRoot(el);
        el.removeSelfFromZr(this);
        this.refresh();
    };
    ZRender.prototype.configLayer = function (zLevel, config) {
        if (this._disposed) {
            return;
        }
        if (this.painter.configLayer) {
            this.painter.configLayer(zLevel, config);
        }
        this.refresh();
    };
    ZRender.prototype.setBackgroundColor = function (backgroundColor) {
        if (this._disposed) {
            return;
        }
        if (this.painter.setBackgroundColor) {
            this.painter.setBackgroundColor(backgroundColor);
        }
        this.refresh();
        this._backgroundColor = backgroundColor;
        this._darkMode = isDarkMode(backgroundColor);
    };
    ZRender.prototype.getBackgroundColor = function () {
        return this._backgroundColor;
    };
    ZRender.prototype.setDarkMode = function (darkMode) {
        this._darkMode = darkMode;
    };
    ZRender.prototype.isDarkMode = function () {
        return this._darkMode;
    };
    ZRender.prototype.refreshImmediately = function (fromInside) {
        if (this._disposed) {
            return;
        }
        if (!fromInside) {
            this.animation.update(true);
        }
        this._needsRefresh = false;
        this.painter.refresh();
        this._needsRefresh = false;
    };
    ZRender.prototype.refresh = function () {
        if (this._disposed) {
            return;
        }
        this._needsRefresh = true;
        this.animation.start();
    };
    ZRender.prototype.flush = function () {
        if (this._disposed) {
            return;
        }
        this._flush(false);
    };
    ZRender.prototype._flush = function (fromInside) {
        var triggerRendered;
        var start = getTime();
        if (this._needsRefresh) {
            triggerRendered = true;
            this.refreshImmediately(fromInside);
        }
        if (this._needsRefreshHover) {
            triggerRendered = true;
            this.refreshHoverImmediately();
        }
        var end = getTime();
        if (triggerRendered) {
            this._stillFrameAccum = 0;
            this.trigger('rendered', {
                elapsedTime: end - start
            });
        }
        else if (this._sleepAfterStill > 0) {
            this._stillFrameAccum++;
            if (this._stillFrameAccum > this._sleepAfterStill) {
                this.animation.stop();
            }
        }
    };
    ZRender.prototype.setSleepAfterStill = function (stillFramesCount) {
        this._sleepAfterStill = stillFramesCount;
    };
    ZRender.prototype.wakeUp = function () {
        if (this._disposed) {
            return;
        }
        this.animation.start();
        this._stillFrameAccum = 0;
    };
    ZRender.prototype.refreshHover = function () {
        this._needsRefreshHover = true;
    };
    ZRender.prototype.refreshHoverImmediately = function () {
        if (this._disposed) {
            return;
        }
        this._needsRefreshHover = false;
        if (this.painter.refreshHover && this.painter.getType() === 'canvas') {
            this.painter.refreshHover();
        }
    };
    ZRender.prototype.resize = function (opts) {
        if (this._disposed) {
            return;
        }
        opts = opts || {};
        this.painter.resize(opts.width, opts.height);
        this.handler.resize();
    };
    ZRender.prototype.clearAnimation = function () {
        if (this._disposed) {
            return;
        }
        this.animation.clear();
    };
    ZRender.prototype.getWidth = function () {
        if (this._disposed) {
            return;
        }
        return this.painter.getWidth();
    };
    ZRender.prototype.getHeight = function () {
        if (this._disposed) {
            return;
        }
        return this.painter.getHeight();
    };
    ZRender.prototype.setCursorStyle = function (cursorStyle) {
        if (this._disposed) {
            return;
        }
        this.handler.setCursorStyle(cursorStyle);
    };
    ZRender.prototype.findHover = function (x, y) {
        if (this._disposed) {
            return;
        }
        return this.handler.findHover(x, y);
    };
    ZRender.prototype.on = function (eventName, eventHandler, context) {
        if (!this._disposed) {
            this.handler.on(eventName, eventHandler, context);
        }
        return this;
    };
    ZRender.prototype.off = function (eventName, eventHandler) {
        if (this._disposed) {
            return;
        }
        this.handler.off(eventName, eventHandler);
    };
    ZRender.prototype.trigger = function (eventName, event) {
        if (this._disposed) {
            return;
        }
        this.handler.trigger(eventName, event);
    };
    ZRender.prototype.clear = function () {
        if (this._disposed) {
            return;
        }
        var roots = this.storage.getRoots();
        for (var i = 0; i < roots.length; i++) {
            if (roots[i] instanceof Group) {
                roots[i].removeSelfFromZr(this);
            }
        }
        this.storage.delAllRoots();
        this.painter.clear();
    };
    ZRender.prototype.dispose = function () {
        if (this._disposed) {
            return;
        }
        this.animation.stop();
        this.clear();
        this.storage.dispose();
        this.painter.dispose();
        this.handler.dispose();
        this.animation =
            this.storage =
                this.painter =
                    this.handler = null;
        this._disposed = true;
        delInstance(this.id);
    };
    return ZRender;
}());
export function init(dom, opts) {
    var zr = new ZRender(zrUtil.guid(), dom, opts);
    instances[zr.id] = zr;
    return zr;
}
export function dispose(zr) {
    zr.dispose();
}
export function disposeAll() {
    for (var key in instances) {
        if (instances.hasOwnProperty(key)) {
            instances[key].dispose();
        }
    }
    instances = {};
}
export function getInstance(id) {
    return instances[id];
}
export function registerPainter(name, Ctor) {
    painterCtors[name] = Ctor;
}
var ssrDataGetter;
export function getElementSSRData(el) {
    if (typeof ssrDataGetter === 'function') {
        return ssrDataGetter(el);
    }
}
export function registerSSRDataGetter(getter) {
    ssrDataGetter = getter;
}
export var version = '5.6.1';
;
