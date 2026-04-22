import { __extends } from "tslib";
import Displayable from './Displayable.js';
import { getBoundingRect } from '../contain/text.js';
import { DEFAULT_PATH_STYLE } from './Path.js';
import { createObject, defaults } from '../core/util.js';
import { DEFAULT_FONT } from '../core/platform.js';
export var DEFAULT_TSPAN_STYLE = defaults({
    strokeFirst: true,
    font: DEFAULT_FONT,
    x: 0,
    y: 0,
    textAlign: 'left',
    textBaseline: 'top',
    miterLimit: 2
}, DEFAULT_PATH_STYLE);
var TSpan = (function (_super) {
    __extends(TSpan, _super);
    function TSpan() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    TSpan.prototype.hasStroke = function () {
        var style = this.style;
        var stroke = style.stroke;
        return stroke != null && stroke !== 'none' && style.lineWidth > 0;
    };
    TSpan.prototype.hasFill = function () {
        var style = this.style;
        var fill = style.fill;
        return fill != null && fill !== 'none';
    };
    TSpan.prototype.createStyle = function (obj) {
        return createObject(DEFAULT_TSPAN_STYLE, obj);
    };
    TSpan.prototype.setBoundingRect = function (rect) {
        this._rect = rect;
    };
    TSpan.prototype.getBoundingRect = function () {
        var style = this.style;
        if (!this._rect) {
            var text = style.text;
            text != null ? (text += '') : (text = '');
            var rect = getBoundingRect(text, style.font, style.textAlign, style.textBaseline);
            rect.x += style.x || 0;
            rect.y += style.y || 0;
            if (this.hasStroke()) {
                var w = style.lineWidth;
                rect.x -= w / 2;
                rect.y -= w / 2;
                rect.width += w;
                rect.height += w;
            }
            this._rect = rect;
        }
        return this._rect;
    };
    TSpan.initDefaultProps = (function () {
        var tspanProto = TSpan.prototype;
        tspanProto.dirtyRectTolerance = 10;
    })();
    return TSpan;
}(Displayable));
TSpan.prototype.type = 'tspan';
export default TSpan;
