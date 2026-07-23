import { extend } from '../core/util.js';
var DebugRect = (function () {
    function DebugRect(style) {
        var dom = this.dom = document.createElement('div');
        dom.className = 'ec-debug-dirty-rect';
        style = extend({}, style);
        extend(style, {
            backgroundColor: 'rgba(0, 0, 255, 0.2)',
            border: '1px solid #00f'
        });
        dom.style.cssText = "\nposition: absolute;\nopacity: 0;\ntransition: opacity 0.5s linear;\npointer-events: none;\n";
        for (var key in style) {
            if (style.hasOwnProperty(key)) {
                dom.style[key] = style[key];
            }
        }
    }
    DebugRect.prototype.update = function (rect) {
        var domStyle = this.dom.style;
        domStyle.width = rect.width + 'px';
        domStyle.height = rect.height + 'px';
        domStyle.left = rect.x + 'px';
        domStyle.top = rect.y + 'px';
    };
    DebugRect.prototype.hide = function () {
        this.dom.style.opacity = '0';
    };
    DebugRect.prototype.show = function (autoHideDelay) {
        var _this = this;
        clearTimeout(this._hideTimeout);
        this.dom.style.opacity = '1';
        this._hideTimeout = setTimeout(function () {
            _this.hide();
        }, autoHideDelay || 1000);
    };
    return DebugRect;
}());
export default function showDebugDirtyRect(zr, opts) {
    opts = opts || {};
    var painter = zr.painter;
    if (!painter.getLayers) {
        throw new Error('Debug dirty rect can only been used on canvas renderer.');
    }
    if (painter.isSingleCanvas()) {
        throw new Error('Debug dirty rect can only been used on zrender inited with container.');
    }
    var debugViewRoot = document.createElement('div');
    debugViewRoot.style.cssText = "\nposition:absolute;\nleft:0;\ntop:0;\nright:0;\nbottom:0;\npointer-events:none;\n";
    debugViewRoot.className = 'ec-debug-dirty-rect-container';
    var debugRects = [];
    var dom = zr.dom;
    dom.appendChild(debugViewRoot);
    var computedStyle = getComputedStyle(dom);
    if (computedStyle.position === 'static') {
        dom.style.position = 'relative';
    }
    zr.on('rendered', function () {
        if (painter.getLayers) {
            var idx_1 = 0;
            painter.eachBuiltinLayer(function (layer) {
                if (!layer.debugGetPaintRects) {
                    return;
                }
                var paintRects = layer.debugGetPaintRects();
                for (var i = 0; i < paintRects.length; i++) {
                    if (!paintRects[i].width || !paintRects[i].height) {
                        continue;
                    }
                    if (!debugRects[idx_1]) {
                        debugRects[idx_1] = new DebugRect(opts.style);
                        debugViewRoot.appendChild(debugRects[idx_1].dom);
                    }
                    debugRects[idx_1].show(opts.autoHideDelay);
                    debugRects[idx_1].update(paintRects[i]);
                    idx_1++;
                }
            });
            for (var i = idx_1; i < debugRects.length; i++) {
                debugRects[i].hide();
            }
        }
    });
}
