var Eventful = (function () {
    function Eventful(eventProcessors) {
        if (eventProcessors) {
            this._$eventProcessor = eventProcessors;
        }
    }
    Eventful.prototype.on = function (event, query, handler, context) {
        if (!this._$handlers) {
            this._$handlers = {};
        }
        var _h = this._$handlers;
        if (typeof query === 'function') {
            context = handler;
            handler = query;
            query = null;
        }
        if (!handler || !event) {
            return this;
        }
        var eventProcessor = this._$eventProcessor;
        if (query != null && eventProcessor && eventProcessor.normalizeQuery) {
            query = eventProcessor.normalizeQuery(query);
        }
        if (!_h[event]) {
            _h[event] = [];
        }
        for (var i = 0; i < _h[event].length; i++) {
            if (_h[event][i].h === handler) {
                return this;
            }
        }
        var wrap = {
            h: handler,
            query: query,
            ctx: (context || this),
            callAtLast: handler.zrEventfulCallAtLast
        };
        var lastIndex = _h[event].length - 1;
        var lastWrap = _h[event][lastIndex];
        (lastWrap && lastWrap.callAtLast)
            ? _h[event].splice(lastIndex, 0, wrap)
            : _h[event].push(wrap);
        return this;
    };
    Eventful.prototype.isSilent = function (eventName) {
        var _h = this._$handlers;
        return !_h || !_h[eventName] || !_h[eventName].length;
    };
    Eventful.prototype.off = function (eventType, handler) {
        var _h = this._$handlers;
        if (!_h) {
            return this;
        }
        if (!eventType) {
            this._$handlers = {};
            return this;
        }
        if (handler) {
            if (_h[eventType]) {
                var newList = [];
                for (var i = 0, l = _h[eventType].length; i < l; i++) {
                    if (_h[eventType][i].h !== handler) {
                        newList.push(_h[eventType][i]);
                    }
                }
                _h[eventType] = newList;
            }
            if (_h[eventType] && _h[eventType].length === 0) {
                delete _h[eventType];
            }
        }
        else {
            delete _h[eventType];
        }
        return this;
    };
    Eventful.prototype.trigger = function (eventType) {
        var args = [];
        for (var _i = 1; _i < arguments.length; _i++) {
            args[_i - 1] = arguments[_i];
        }
        if (!this._$handlers) {
            return this;
        }
        var _h = this._$handlers[eventType];
        var eventProcessor = this._$eventProcessor;
        if (_h) {
            var argLen = args.length;
            var len = _h.length;
            for (var i = 0; i < len; i++) {
                var hItem = _h[i];
                if (eventProcessor
                    && eventProcessor.filter
                    && hItem.query != null
                    && !eventProcessor.filter(eventType, hItem.query)) {
                    continue;
                }
                switch (argLen) {
                    case 0:
                        hItem.h.call(hItem.ctx);
                        break;
                    case 1:
                        hItem.h.call(hItem.ctx, args[0]);
                        break;
                    case 2:
                        hItem.h.call(hItem.ctx, args[0], args[1]);
                        break;
                    default:
                        hItem.h.apply(hItem.ctx, args);
                        break;
                }
            }
        }
        eventProcessor && eventProcessor.afterTrigger
            && eventProcessor.afterTrigger(eventType);
        return this;
    };
    Eventful.prototype.triggerWithContext = function (type) {
        var args = [];
        for (var _i = 1; _i < arguments.length; _i++) {
            args[_i - 1] = arguments[_i];
        }
        if (!this._$handlers) {
            return this;
        }
        var _h = this._$handlers[type];
        var eventProcessor = this._$eventProcessor;
        if (_h) {
            var argLen = args.length;
            var ctx = args[argLen - 1];
            var len = _h.length;
            for (var i = 0; i < len; i++) {
                var hItem = _h[i];
                if (eventProcessor
                    && eventProcessor.filter
                    && hItem.query != null
                    && !eventProcessor.filter(type, hItem.query)) {
                    continue;
                }
                switch (argLen) {
                    case 0:
                        hItem.h.call(ctx);
                        break;
                    case 1:
                        hItem.h.call(ctx, args[0]);
                        break;
                    case 2:
                        hItem.h.call(ctx, args[0], args[1]);
                        break;
                    default:
                        hItem.h.apply(ctx, args.slice(1, argLen - 1));
                        break;
                }
            }
        }
        eventProcessor && eventProcessor.afterTrigger
            && eventProcessor.afterTrigger(type);
        return this;
    };
    return Eventful;
}());
export default Eventful;
