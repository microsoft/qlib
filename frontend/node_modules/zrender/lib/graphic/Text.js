import { __extends } from "tslib";
import { parseRichText, parsePlainText } from './helper/parseText.js';
import TSpan from './TSpan.js';
import { retrieve2, each, normalizeCssArray, trim, retrieve3, extend, keys, defaults } from '../core/util.js';
import { adjustTextX, adjustTextY } from '../contain/text.js';
import ZRImage from './Image.js';
import Rect from './shape/Rect.js';
import BoundingRect from '../core/BoundingRect.js';
import Displayable, { DEFAULT_COMMON_ANIMATION_PROPS } from './Displayable.js';
import { DEFAULT_FONT, DEFAULT_FONT_SIZE } from '../core/platform.js';
var DEFAULT_RICH_TEXT_COLOR = {
    fill: '#000'
};
var DEFAULT_STROKE_LINE_WIDTH = 2;
export var DEFAULT_TEXT_ANIMATION_PROPS = {
    style: defaults({
        fill: true,
        stroke: true,
        fillOpacity: true,
        strokeOpacity: true,
        lineWidth: true,
        fontSize: true,
        lineHeight: true,
        width: true,
        height: true,
        textShadowColor: true,
        textShadowBlur: true,
        textShadowOffsetX: true,
        textShadowOffsetY: true,
        backgroundColor: true,
        padding: true,
        borderColor: true,
        borderWidth: true,
        borderRadius: true
    }, DEFAULT_COMMON_ANIMATION_PROPS.style)
};
var ZRText = (function (_super) {
    __extends(ZRText, _super);
    function ZRText(opts) {
        var _this = _super.call(this) || this;
        _this.type = 'text';
        _this._children = [];
        _this._defaultStyle = DEFAULT_RICH_TEXT_COLOR;
        _this.attr(opts);
        return _this;
    }
    ZRText.prototype.childrenRef = function () {
        return this._children;
    };
    ZRText.prototype.update = function () {
        _super.prototype.update.call(this);
        if (this.styleChanged()) {
            this._updateSubTexts();
        }
        for (var i = 0; i < this._children.length; i++) {
            var child = this._children[i];
            child.zlevel = this.zlevel;
            child.z = this.z;
            child.z2 = this.z2;
            child.culling = this.culling;
            child.cursor = this.cursor;
            child.invisible = this.invisible;
        }
    };
    ZRText.prototype.updateTransform = function () {
        var innerTransformable = this.innerTransformable;
        if (innerTransformable) {
            innerTransformable.updateTransform();
            if (innerTransformable.transform) {
                this.transform = innerTransformable.transform;
            }
        }
        else {
            _super.prototype.updateTransform.call(this);
        }
    };
    ZRText.prototype.getLocalTransform = function (m) {
        var innerTransformable = this.innerTransformable;
        return innerTransformable
            ? innerTransformable.getLocalTransform(m)
            : _super.prototype.getLocalTransform.call(this, m);
    };
    ZRText.prototype.getComputedTransform = function () {
        if (this.__hostTarget) {
            this.__hostTarget.getComputedTransform();
            this.__hostTarget.updateInnerText(true);
        }
        return _super.prototype.getComputedTransform.call(this);
    };
    ZRText.prototype._updateSubTexts = function () {
        this._childCursor = 0;
        normalizeTextStyle(this.style);
        this.style.rich
            ? this._updateRichTexts()
            : this._updatePlainTexts();
        this._children.length = this._childCursor;
        this.styleUpdated();
    };
    ZRText.prototype.addSelfToZr = function (zr) {
        _super.prototype.addSelfToZr.call(this, zr);
        for (var i = 0; i < this._children.length; i++) {
            this._children[i].__zr = zr;
        }
    };
    ZRText.prototype.removeSelfFromZr = function (zr) {
        _super.prototype.removeSelfFromZr.call(this, zr);
        for (var i = 0; i < this._children.length; i++) {
            this._children[i].__zr = null;
        }
    };
    ZRText.prototype.getBoundingRect = function () {
        if (this.styleChanged()) {
            this._updateSubTexts();
        }
        if (!this._rect) {
            var tmpRect = new BoundingRect(0, 0, 0, 0);
            var children = this._children;
            var tmpMat = [];
            var rect = null;
            for (var i = 0; i < children.length; i++) {
                var child = children[i];
                var childRect = child.getBoundingRect();
                var transform = child.getLocalTransform(tmpMat);
                if (transform) {
                    tmpRect.copy(childRect);
                    tmpRect.applyTransform(transform);
                    rect = rect || tmpRect.clone();
                    rect.union(tmpRect);
                }
                else {
                    rect = rect || childRect.clone();
                    rect.union(childRect);
                }
            }
            this._rect = rect || tmpRect;
        }
        return this._rect;
    };
    ZRText.prototype.setDefaultTextStyle = function (defaultTextStyle) {
        this._defaultStyle = defaultTextStyle || DEFAULT_RICH_TEXT_COLOR;
    };
    ZRText.prototype.setTextContent = function (textContent) {
        if (process.env.NODE_ENV !== 'production') {
            throw new Error('Can\'t attach text on another text');
        }
    };
    ZRText.prototype._mergeStyle = function (targetStyle, sourceStyle) {
        if (!sourceStyle) {
            return targetStyle;
        }
        var sourceRich = sourceStyle.rich;
        var targetRich = targetStyle.rich || (sourceRich && {});
        extend(targetStyle, sourceStyle);
        if (sourceRich && targetRich) {
            this._mergeRich(targetRich, sourceRich);
            targetStyle.rich = targetRich;
        }
        else if (targetRich) {
            targetStyle.rich = targetRich;
        }
        return targetStyle;
    };
    ZRText.prototype._mergeRich = function (targetRich, sourceRich) {
        var richNames = keys(sourceRich);
        for (var i = 0; i < richNames.length; i++) {
            var richName = richNames[i];
            targetRich[richName] = targetRich[richName] || {};
            extend(targetRich[richName], sourceRich[richName]);
        }
    };
    ZRText.prototype.getAnimationStyleProps = function () {
        return DEFAULT_TEXT_ANIMATION_PROPS;
    };
    ZRText.prototype._getOrCreateChild = function (Ctor) {
        var child = this._children[this._childCursor];
        if (!child || !(child instanceof Ctor)) {
            child = new Ctor();
        }
        this._children[this._childCursor++] = child;
        child.__zr = this.__zr;
        child.parent = this;
        return child;
    };
    ZRText.prototype._updatePlainTexts = function () {
        var style = this.style;
        var textFont = style.font || DEFAULT_FONT;
        var textPadding = style.padding;
        var text = getStyleText(style);
        var contentBlock = parsePlainText(text, style);
        var needDrawBg = needDrawBackground(style);
        var bgColorDrawn = !!(style.backgroundColor);
        var outerHeight = contentBlock.outerHeight;
        var outerWidth = contentBlock.outerWidth;
        var contentWidth = contentBlock.contentWidth;
        var textLines = contentBlock.lines;
        var lineHeight = contentBlock.lineHeight;
        var defaultStyle = this._defaultStyle;
        this.isTruncated = !!contentBlock.isTruncated;
        var baseX = style.x || 0;
        var baseY = style.y || 0;
        var textAlign = style.align || defaultStyle.align || 'left';
        var verticalAlign = style.verticalAlign || defaultStyle.verticalAlign || 'top';
        var textX = baseX;
        var textY = adjustTextY(baseY, contentBlock.contentHeight, verticalAlign);
        if (needDrawBg || textPadding) {
            var boxX = adjustTextX(baseX, outerWidth, textAlign);
            var boxY = adjustTextY(baseY, outerHeight, verticalAlign);
            needDrawBg && this._renderBackground(style, style, boxX, boxY, outerWidth, outerHeight);
        }
        textY += lineHeight / 2;
        if (textPadding) {
            textX = getTextXForPadding(baseX, textAlign, textPadding);
            if (verticalAlign === 'top') {
                textY += textPadding[0];
            }
            else if (verticalAlign === 'bottom') {
                textY -= textPadding[2];
            }
        }
        var defaultLineWidth = 0;
        var useDefaultFill = false;
        var textFill = getFill('fill' in style
            ? style.fill
            : (useDefaultFill = true, defaultStyle.fill));
        var textStroke = getStroke('stroke' in style
            ? style.stroke
            : (!bgColorDrawn
                && (!defaultStyle.autoStroke || useDefaultFill))
                ? (defaultLineWidth = DEFAULT_STROKE_LINE_WIDTH, defaultStyle.stroke)
                : null);
        var hasShadow = style.textShadowBlur > 0;
        var fixedBoundingRect = style.width != null
            && (style.overflow === 'truncate' || style.overflow === 'break' || style.overflow === 'breakAll');
        var calculatedLineHeight = contentBlock.calculatedLineHeight;
        for (var i = 0; i < textLines.length; i++) {
            var el = this._getOrCreateChild(TSpan);
            var subElStyle = el.createStyle();
            el.useStyle(subElStyle);
            subElStyle.text = textLines[i];
            subElStyle.x = textX;
            subElStyle.y = textY;
            if (textAlign) {
                subElStyle.textAlign = textAlign;
            }
            subElStyle.textBaseline = 'middle';
            subElStyle.opacity = style.opacity;
            subElStyle.strokeFirst = true;
            if (hasShadow) {
                subElStyle.shadowBlur = style.textShadowBlur || 0;
                subElStyle.shadowColor = style.textShadowColor || 'transparent';
                subElStyle.shadowOffsetX = style.textShadowOffsetX || 0;
                subElStyle.shadowOffsetY = style.textShadowOffsetY || 0;
            }
            subElStyle.stroke = textStroke;
            subElStyle.fill = textFill;
            if (textStroke) {
                subElStyle.lineWidth = style.lineWidth || defaultLineWidth;
                subElStyle.lineDash = style.lineDash;
                subElStyle.lineDashOffset = style.lineDashOffset || 0;
            }
            subElStyle.font = textFont;
            setSeparateFont(subElStyle, style);
            textY += lineHeight;
            if (fixedBoundingRect) {
                el.setBoundingRect(new BoundingRect(adjustTextX(subElStyle.x, contentWidth, subElStyle.textAlign), adjustTextY(subElStyle.y, calculatedLineHeight, subElStyle.textBaseline), contentWidth, calculatedLineHeight));
            }
        }
    };
    ZRText.prototype._updateRichTexts = function () {
        var style = this.style;
        var text = getStyleText(style);
        var contentBlock = parseRichText(text, style);
        var contentWidth = contentBlock.width;
        var outerWidth = contentBlock.outerWidth;
        var outerHeight = contentBlock.outerHeight;
        var textPadding = style.padding;
        var baseX = style.x || 0;
        var baseY = style.y || 0;
        var defaultStyle = this._defaultStyle;
        var textAlign = style.align || defaultStyle.align;
        var verticalAlign = style.verticalAlign || defaultStyle.verticalAlign;
        this.isTruncated = !!contentBlock.isTruncated;
        var boxX = adjustTextX(baseX, outerWidth, textAlign);
        var boxY = adjustTextY(baseY, outerHeight, verticalAlign);
        var xLeft = boxX;
        var lineTop = boxY;
        if (textPadding) {
            xLeft += textPadding[3];
            lineTop += textPadding[0];
        }
        var xRight = xLeft + contentWidth;
        if (needDrawBackground(style)) {
            this._renderBackground(style, style, boxX, boxY, outerWidth, outerHeight);
        }
        var bgColorDrawn = !!(style.backgroundColor);
        for (var i = 0; i < contentBlock.lines.length; i++) {
            var line = contentBlock.lines[i];
            var tokens = line.tokens;
            var tokenCount = tokens.length;
            var lineHeight = line.lineHeight;
            var remainedWidth = line.width;
            var leftIndex = 0;
            var lineXLeft = xLeft;
            var lineXRight = xRight;
            var rightIndex = tokenCount - 1;
            var token = void 0;
            while (leftIndex < tokenCount
                && (token = tokens[leftIndex], !token.align || token.align === 'left')) {
                this._placeToken(token, style, lineHeight, lineTop, lineXLeft, 'left', bgColorDrawn);
                remainedWidth -= token.width;
                lineXLeft += token.width;
                leftIndex++;
            }
            while (rightIndex >= 0
                && (token = tokens[rightIndex], token.align === 'right')) {
                this._placeToken(token, style, lineHeight, lineTop, lineXRight, 'right', bgColorDrawn);
                remainedWidth -= token.width;
                lineXRight -= token.width;
                rightIndex--;
            }
            lineXLeft += (contentWidth - (lineXLeft - xLeft) - (xRight - lineXRight) - remainedWidth) / 2;
            while (leftIndex <= rightIndex) {
                token = tokens[leftIndex];
                this._placeToken(token, style, lineHeight, lineTop, lineXLeft + token.width / 2, 'center', bgColorDrawn);
                lineXLeft += token.width;
                leftIndex++;
            }
            lineTop += lineHeight;
        }
    };
    ZRText.prototype._placeToken = function (token, style, lineHeight, lineTop, x, textAlign, parentBgColorDrawn) {
        var tokenStyle = style.rich[token.styleName] || {};
        tokenStyle.text = token.text;
        var verticalAlign = token.verticalAlign;
        var y = lineTop + lineHeight / 2;
        if (verticalAlign === 'top') {
            y = lineTop + token.height / 2;
        }
        else if (verticalAlign === 'bottom') {
            y = lineTop + lineHeight - token.height / 2;
        }
        var needDrawBg = !token.isLineHolder && needDrawBackground(tokenStyle);
        needDrawBg && this._renderBackground(tokenStyle, style, textAlign === 'right'
            ? x - token.width
            : textAlign === 'center'
                ? x - token.width / 2
                : x, y - token.height / 2, token.width, token.height);
        var bgColorDrawn = !!tokenStyle.backgroundColor;
        var textPadding = token.textPadding;
        if (textPadding) {
            x = getTextXForPadding(x, textAlign, textPadding);
            y -= token.height / 2 - textPadding[0] - token.innerHeight / 2;
        }
        var el = this._getOrCreateChild(TSpan);
        var subElStyle = el.createStyle();
        el.useStyle(subElStyle);
        var defaultStyle = this._defaultStyle;
        var useDefaultFill = false;
        var defaultLineWidth = 0;
        var textFill = getFill('fill' in tokenStyle ? tokenStyle.fill
            : 'fill' in style ? style.fill
                : (useDefaultFill = true, defaultStyle.fill));
        var textStroke = getStroke('stroke' in tokenStyle ? tokenStyle.stroke
            : 'stroke' in style ? style.stroke
                : (!bgColorDrawn
                    && !parentBgColorDrawn
                    && (!defaultStyle.autoStroke || useDefaultFill)) ? (defaultLineWidth = DEFAULT_STROKE_LINE_WIDTH, defaultStyle.stroke)
                    : null);
        var hasShadow = tokenStyle.textShadowBlur > 0
            || style.textShadowBlur > 0;
        subElStyle.text = token.text;
        subElStyle.x = x;
        subElStyle.y = y;
        if (hasShadow) {
            subElStyle.shadowBlur = tokenStyle.textShadowBlur || style.textShadowBlur || 0;
            subElStyle.shadowColor = tokenStyle.textShadowColor || style.textShadowColor || 'transparent';
            subElStyle.shadowOffsetX = tokenStyle.textShadowOffsetX || style.textShadowOffsetX || 0;
            subElStyle.shadowOffsetY = tokenStyle.textShadowOffsetY || style.textShadowOffsetY || 0;
        }
        subElStyle.textAlign = textAlign;
        subElStyle.textBaseline = 'middle';
        subElStyle.font = token.font || DEFAULT_FONT;
        subElStyle.opacity = retrieve3(tokenStyle.opacity, style.opacity, 1);
        setSeparateFont(subElStyle, tokenStyle);
        if (textStroke) {
            subElStyle.lineWidth = retrieve3(tokenStyle.lineWidth, style.lineWidth, defaultLineWidth);
            subElStyle.lineDash = retrieve2(tokenStyle.lineDash, style.lineDash);
            subElStyle.lineDashOffset = style.lineDashOffset || 0;
            subElStyle.stroke = textStroke;
        }
        if (textFill) {
            subElStyle.fill = textFill;
        }
        var textWidth = token.contentWidth;
        var textHeight = token.contentHeight;
        el.setBoundingRect(new BoundingRect(adjustTextX(subElStyle.x, textWidth, subElStyle.textAlign), adjustTextY(subElStyle.y, textHeight, subElStyle.textBaseline), textWidth, textHeight));
    };
    ZRText.prototype._renderBackground = function (style, topStyle, x, y, width, height) {
        var textBackgroundColor = style.backgroundColor;
        var textBorderWidth = style.borderWidth;
        var textBorderColor = style.borderColor;
        var isImageBg = textBackgroundColor && textBackgroundColor.image;
        var isPlainOrGradientBg = textBackgroundColor && !isImageBg;
        var textBorderRadius = style.borderRadius;
        var self = this;
        var rectEl;
        var imgEl;
        if (isPlainOrGradientBg || style.lineHeight || (textBorderWidth && textBorderColor)) {
            rectEl = this._getOrCreateChild(Rect);
            rectEl.useStyle(rectEl.createStyle());
            rectEl.style.fill = null;
            var rectShape = rectEl.shape;
            rectShape.x = x;
            rectShape.y = y;
            rectShape.width = width;
            rectShape.height = height;
            rectShape.r = textBorderRadius;
            rectEl.dirtyShape();
        }
        if (isPlainOrGradientBg) {
            var rectStyle = rectEl.style;
            rectStyle.fill = textBackgroundColor || null;
            rectStyle.fillOpacity = retrieve2(style.fillOpacity, 1);
        }
        else if (isImageBg) {
            imgEl = this._getOrCreateChild(ZRImage);
            imgEl.onload = function () {
                self.dirtyStyle();
            };
            var imgStyle = imgEl.style;
            imgStyle.image = textBackgroundColor.image;
            imgStyle.x = x;
            imgStyle.y = y;
            imgStyle.width = width;
            imgStyle.height = height;
        }
        if (textBorderWidth && textBorderColor) {
            var rectStyle = rectEl.style;
            rectStyle.lineWidth = textBorderWidth;
            rectStyle.stroke = textBorderColor;
            rectStyle.strokeOpacity = retrieve2(style.strokeOpacity, 1);
            rectStyle.lineDash = style.borderDash;
            rectStyle.lineDashOffset = style.borderDashOffset || 0;
            rectEl.strokeContainThreshold = 0;
            if (rectEl.hasFill() && rectEl.hasStroke()) {
                rectStyle.strokeFirst = true;
                rectStyle.lineWidth *= 2;
            }
        }
        var commonStyle = (rectEl || imgEl).style;
        commonStyle.shadowBlur = style.shadowBlur || 0;
        commonStyle.shadowColor = style.shadowColor || 'transparent';
        commonStyle.shadowOffsetX = style.shadowOffsetX || 0;
        commonStyle.shadowOffsetY = style.shadowOffsetY || 0;
        commonStyle.opacity = retrieve3(style.opacity, topStyle.opacity, 1);
    };
    ZRText.makeFont = function (style) {
        var font = '';
        if (hasSeparateFont(style)) {
            font = [
                style.fontStyle,
                style.fontWeight,
                parseFontSize(style.fontSize),
                style.fontFamily || 'sans-serif'
            ].join(' ');
        }
        return font && trim(font) || style.textFont || style.font;
    };
    return ZRText;
}(Displayable));
var VALID_TEXT_ALIGN = { left: true, right: 1, center: 1 };
var VALID_TEXT_VERTICAL_ALIGN = { top: 1, bottom: 1, middle: 1 };
var FONT_PARTS = ['fontStyle', 'fontWeight', 'fontSize', 'fontFamily'];
export function parseFontSize(fontSize) {
    if (typeof fontSize === 'string'
        && (fontSize.indexOf('px') !== -1
            || fontSize.indexOf('rem') !== -1
            || fontSize.indexOf('em') !== -1)) {
        return fontSize;
    }
    else if (!isNaN(+fontSize)) {
        return fontSize + 'px';
    }
    else {
        return DEFAULT_FONT_SIZE + 'px';
    }
}
function setSeparateFont(targetStyle, sourceStyle) {
    for (var i = 0; i < FONT_PARTS.length; i++) {
        var fontProp = FONT_PARTS[i];
        var val = sourceStyle[fontProp];
        if (val != null) {
            targetStyle[fontProp] = val;
        }
    }
}
export function hasSeparateFont(style) {
    return style.fontSize != null || style.fontFamily || style.fontWeight;
}
export function normalizeTextStyle(style) {
    normalizeStyle(style);
    each(style.rich, normalizeStyle);
    return style;
}
function normalizeStyle(style) {
    if (style) {
        style.font = ZRText.makeFont(style);
        var textAlign = style.align;
        textAlign === 'middle' && (textAlign = 'center');
        style.align = (textAlign == null || VALID_TEXT_ALIGN[textAlign]) ? textAlign : 'left';
        var verticalAlign = style.verticalAlign;
        verticalAlign === 'center' && (verticalAlign = 'middle');
        style.verticalAlign = (verticalAlign == null || VALID_TEXT_VERTICAL_ALIGN[verticalAlign]) ? verticalAlign : 'top';
        var textPadding = style.padding;
        if (textPadding) {
            style.padding = normalizeCssArray(style.padding);
        }
    }
}
function getStroke(stroke, lineWidth) {
    return (stroke == null || lineWidth <= 0 || stroke === 'transparent' || stroke === 'none')
        ? null
        : (stroke.image || stroke.colorStops)
            ? '#000'
            : stroke;
}
function getFill(fill) {
    return (fill == null || fill === 'none')
        ? null
        : (fill.image || fill.colorStops)
            ? '#000'
            : fill;
}
function getTextXForPadding(x, textAlign, textPadding) {
    return textAlign === 'right'
        ? (x - textPadding[1])
        : textAlign === 'center'
            ? (x + textPadding[3] / 2 - textPadding[1] / 2)
            : (x + textPadding[3]);
}
function getStyleText(style) {
    var text = style.text;
    text != null && (text += '');
    return text;
}
function needDrawBackground(style) {
    return !!(style.backgroundColor
        || style.lineHeight
        || (style.borderWidth && style.borderColor));
}
export default ZRText;
