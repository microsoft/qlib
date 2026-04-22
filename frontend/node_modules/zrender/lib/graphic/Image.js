import { __extends } from "tslib";
import Displayable, { DEFAULT_COMMON_STYLE, DEFAULT_COMMON_ANIMATION_PROPS } from './Displayable.js';
import BoundingRect from '../core/BoundingRect.js';
import { defaults, createObject } from '../core/util.js';
export var DEFAULT_IMAGE_STYLE = defaults({
    x: 0,
    y: 0
}, DEFAULT_COMMON_STYLE);
export var DEFAULT_IMAGE_ANIMATION_PROPS = {
    style: defaults({
        x: true,
        y: true,
        width: true,
        height: true,
        sx: true,
        sy: true,
        sWidth: true,
        sHeight: true
    }, DEFAULT_COMMON_ANIMATION_PROPS.style)
};
function isImageLike(source) {
    return !!(source
        && typeof source !== 'string'
        && source.width && source.height);
}
var ZRImage = (function (_super) {
    __extends(ZRImage, _super);
    function ZRImage() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    ZRImage.prototype.createStyle = function (obj) {
        return createObject(DEFAULT_IMAGE_STYLE, obj);
    };
    ZRImage.prototype._getSize = function (dim) {
        var style = this.style;
        var size = style[dim];
        if (size != null) {
            return size;
        }
        var imageSource = isImageLike(style.image)
            ? style.image : this.__image;
        if (!imageSource) {
            return 0;
        }
        var otherDim = dim === 'width' ? 'height' : 'width';
        var otherDimSize = style[otherDim];
        if (otherDimSize == null) {
            return imageSource[dim];
        }
        else {
            return imageSource[dim] / imageSource[otherDim] * otherDimSize;
        }
    };
    ZRImage.prototype.getWidth = function () {
        return this._getSize('width');
    };
    ZRImage.prototype.getHeight = function () {
        return this._getSize('height');
    };
    ZRImage.prototype.getAnimationStyleProps = function () {
        return DEFAULT_IMAGE_ANIMATION_PROPS;
    };
    ZRImage.prototype.getBoundingRect = function () {
        var style = this.style;
        if (!this._rect) {
            this._rect = new BoundingRect(style.x || 0, style.y || 0, this.getWidth(), this.getHeight());
        }
        return this._rect;
    };
    return ZRImage;
}(Displayable));
ZRImage.prototype.type = 'image';
export default ZRImage;
