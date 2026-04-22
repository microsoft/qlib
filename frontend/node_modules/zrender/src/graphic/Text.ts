/**
 * RichText is a container that manages complex text label.
 * It will parse text string and create sub displayble elements respectively.
 */
import { TextAlign, TextVerticalAlign, ImageLike, Dictionary, MapToType, FontWeight, FontStyle } from '../core/types';
import { parseRichText, parsePlainText } from './helper/parseText';
import TSpan, { TSpanStyleProps } from './TSpan';
import { retrieve2, each, normalizeCssArray, trim, retrieve3, extend, keys, defaults } from '../core/util';
import { adjustTextX, adjustTextY } from '../contain/text';
import ZRImage from './Image';
import Rect from './shape/Rect';
import BoundingRect from '../core/BoundingRect';
import { MatrixArray } from '../core/matrix';
import Displayable, {
    DisplayableStatePropNames,
    DisplayableProps,
    DEFAULT_COMMON_ANIMATION_PROPS
} from './Displayable';
import { ZRenderType } from '../zrender';
import Animator from '../animation/Animator';
import Transformable from '../core/Transformable';
import { ElementCommonState } from '../Element';
import { GroupLike } from './Group';
import { DEFAULT_FONT, DEFAULT_FONT_SIZE } from '../core/platform';

type TextContentBlock = ReturnType<typeof parseRichText>
type TextLine = TextContentBlock['lines'][0]
type TextToken = TextLine['tokens'][0]

// TODO Default value?
export interface TextStylePropsPart {
    // TODO Text is assigned inside zrender
    text?: string

    fill?: string
    stroke?: string
    strokeNoScale?: boolean

    opacity?: number
    fillOpacity?: number
    strokeOpacity?: number
    /**
     * textStroke may be set as some color as a default
     * value in upper applicaion, where the default value
     * of lineWidth should be 0 to make sure that
     * user can choose to do not use text stroke.
     */
    lineWidth?: number
    lineDash?: false | number[]
    lineDashOffset?: number
    borderDash?: false | number[]
    borderDashOffset?: number

    /**
     * If `fontSize` or `fontFamily` exists, `font` will be reset by
     * `fontSize`, `fontStyle`, `fontWeight`, `fontFamily`.
     * So do not visit it directly in upper application (like echarts),
     * but use `contain/text#makeFont` instead.
     */
    font?: string
    /**
     * The same as font. Use font please.
     * @deprecated
     */
    textFont?: string

    /**
     * It helps merging respectively, rather than parsing an entire font string.
     */
    fontStyle?: FontStyle
    /**
     * It helps merging respectively, rather than parsing an entire font string.
     */
    fontWeight?: FontWeight
    /**
     * It helps merging respectively, rather than parsing an entire font string.
     */
    fontFamily?: string
    /**
     * It helps merging respectively, rather than parsing an entire font string.
     * Should be 12 but not '12px'.
     */
    fontSize?: number | string

    align?: TextAlign
    verticalAlign?: TextVerticalAlign

    /**
     * Line height. Default to be text height of '国'
     */
    lineHeight?: number
    /**
     * Width of text block. Not include padding
     * Used for background, truncate, wrap
     */
    width?: number | string
    /**
     * Height of text block. Not include padding
     * Used for background, truncate
     */
    height?: number
    /**
     * Reserved for special functinality, like 'hr'.
     */
    tag?: string

    textShadowColor?: string
    textShadowBlur?: number
    textShadowOffsetX?: number
    textShadowOffsetY?: number

    // Shadow, background, border of text box.
    backgroundColor?: string | {
        image: ImageLike | string
    }

    /**
     * Can be `2` or `[2, 4]` or `[2, 3, 4, 5]`
     */
    padding?: number | number[]
    /**
     * Margin of label. Used when layouting the label.
     */
    margin?: number

    borderColor?: string
    borderWidth?: number
    borderRadius?: number | number[]

    /**
     * Shadow color for background box.
     */
    shadowColor?: string
    /**
     * Shadow blur for background box.
     */
    shadowBlur?: number
    /**
     * Shadow offset x for background box.
     */
    shadowOffsetX?: number
    /**
     * Shadow offset y for background box.
     */
    shadowOffsetY?: number
}
export interface TextStyleProps extends TextStylePropsPart {

    text?: string

    x?: number
    y?: number

    /**
     * Only support number in the top block.
     */
    width?: number
    /**
     * Text styles for rich text.
     */
    rich?: Dictionary<TextStylePropsPart>

    /**
     * Strategy when calculated text width exceeds textWidth.
     * break: break by word
     * break: will break inside the word
     * truncate: truncate the text and show ellipsis
     * Do nothing if not set
     */
    overflow?: 'break' | 'breakAll' | 'truncate' | 'none'

    /**
     * Strategy when text lines exceeds textHeight.
     * Do nothing if not set
     */
    lineOverflow?: 'truncate'

    /**
     * Epllipsis used if text is truncated
     */
    ellipsis?: string
    /**
     * Placeholder used if text is truncated to empty
     */
    placeholder?: string
    /**
     * Min characters for truncating
     */
    truncateMinChar?: number
}

export interface TextProps extends DisplayableProps {
    style?: TextStyleProps

    zlevel?: number
    z?: number
    z2?: number

    culling?: boolean
    cursor?: string
}

export type TextState = Pick<TextProps, DisplayableStatePropNames> & ElementCommonState

export type DefaultTextStyle = Pick<TextStyleProps, 'fill' | 'stroke' | 'align' | 'verticalAlign'> & {
    autoStroke?: boolean
};

const DEFAULT_RICH_TEXT_COLOR = {
    fill: '#000'
};
const DEFAULT_STROKE_LINE_WIDTH = 2;

// const DEFAULT_TEXT_STYLE: TextStyleProps = {
//     x: 0,
//     y: 0,
//     fill: '#000',
//     stroke: null,
//     opacity: 0,
//     fillOpacity:
// }

export const DEFAULT_TEXT_ANIMATION_PROPS: MapToType<TextProps, boolean> = {
    style: defaults<MapToType<TextStyleProps, boolean>, MapToType<TextStyleProps, boolean>>({
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
        padding: true,  // TODO needs normalize padding before animate
        borderColor: true,
        borderWidth: true,
        borderRadius: true  // TODO needs normalize radius before animate
    }, DEFAULT_COMMON_ANIMATION_PROPS.style)
 };


interface ZRText {
    animate(key?: '', loop?: boolean): Animator<this>
    animate(key: 'style', loop?: boolean): Animator<this['style']>

    getState(stateName: string): TextState
    ensureState(stateName: string): TextState

    states: Dictionary<TextState>
    stateProxy: (stateName: string) => TextState
}

class ZRText extends Displayable<TextProps> implements GroupLike {

    type = 'text'

    style: TextStyleProps

    /**
     * How to handling label overlap
     *
     * hidden:
     */
    overlap: 'hidden' | 'show' | 'blur'

    /**
     * Will use this to calculate transform matrix
     * instead of Element itseelf if it's give.
     * Not exposed to developers
     */
    innerTransformable: Transformable

    // Be `true` if and only if the result text is modified due to overflow, due to
    // settings on either `overflow` or `lineOverflow`. Based on this the caller can
    // take some action like showing the original text in a particular tip.
    // Only take effect after rendering. So do not visit it before it.
    isTruncated: boolean

    private _children: (ZRImage | Rect | TSpan)[] = []

    private _childCursor: 0

    private _defaultStyle: DefaultTextStyle = DEFAULT_RICH_TEXT_COLOR

    constructor(opts?: TextProps) {
        super();
        this.attr(opts);
    }

    childrenRef() {
        return this._children;
    }

    update() {

        super.update();

        // Update children
        if (this.styleChanged()) {
            this._updateSubTexts();
        }

        for (let i = 0; i < this._children.length; i++) {
            const child = this._children[i];
            // Set common properties.
            child.zlevel = this.zlevel;
            child.z = this.z;
            child.z2 = this.z2;
            child.culling = this.culling;
            child.cursor = this.cursor;
            child.invisible = this.invisible;
        }
    }

     updateTransform() {
        const innerTransformable = this.innerTransformable;
        if (innerTransformable) {
            innerTransformable.updateTransform();
            if (innerTransformable.transform) {
                this.transform = innerTransformable.transform;
            }
        }
        else {
            super.updateTransform();
        }
    }

    getLocalTransform(m?: MatrixArray): MatrixArray {
        const innerTransformable = this.innerTransformable;
        return innerTransformable
            ? innerTransformable.getLocalTransform(m)
            : super.getLocalTransform(m);
    }

    // TODO override setLocalTransform?
    getComputedTransform() {
        if (this.__hostTarget) {
            // Update host target transform
            this.__hostTarget.getComputedTransform();
            // Update text position.
            this.__hostTarget.updateInnerText(true);
        }

        return super.getComputedTransform();
    }

    private _updateSubTexts() {
        // Reset child visit cursor
        this._childCursor = 0;

        normalizeTextStyle(this.style);
        this.style.rich
            ? this._updateRichTexts()
            : this._updatePlainTexts();

        this._children.length = this._childCursor;

        this.styleUpdated();
    }

    addSelfToZr(zr: ZRenderType) {
        super.addSelfToZr(zr);
        for (let i = 0; i < this._children.length; i++) {
            // Also need mount __zr for case like hover detection.
            // The case: hover on a label (position: 'top') causes host el
            // scaled and label Y position lifts a bit so that out of the
            // pointer, then mouse move should be able to trigger "mouseout".
            this._children[i].__zr = zr;
        }
    }

    removeSelfFromZr(zr: ZRenderType) {
        super.removeSelfFromZr(zr);
        for (let i = 0; i < this._children.length; i++) {
            this._children[i].__zr = null;
        }
    }

    getBoundingRect(): BoundingRect {
        if (this.styleChanged()) {
            this._updateSubTexts();
        }
        if (!this._rect) {
            // TODO: Optimize when using width and overflow: wrap/truncate
            const tmpRect = new BoundingRect(0, 0, 0, 0);
            const children = this._children;
            const tmpMat: MatrixArray = [];
            let rect = null;

            for (let i = 0; i < children.length; i++) {
                const child = children[i];
                const childRect = child.getBoundingRect();
                const transform = child.getLocalTransform(tmpMat);

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
    }

    // Can be set in Element. To calculate text fill automatically when textContent is inside element
    setDefaultTextStyle(defaultTextStyle: DefaultTextStyle) {
        // Use builtin if defaultTextStyle is not given.
        this._defaultStyle = defaultTextStyle || DEFAULT_RICH_TEXT_COLOR;
    }

    setTextContent(textContent: never) {
        if (process.env.NODE_ENV !== 'production') {
            throw new Error('Can\'t attach text on another text');
        }
    }

    // getDefaultStyleValue<T extends keyof TextStyleProps>(key: T): TextStyleProps[T] {
    //     // Default value is on the prototype.
    //     return this.style.prototype[key];
    // }

    protected _mergeStyle(targetStyle: TextStyleProps, sourceStyle: TextStyleProps) {
        if (!sourceStyle) {
            return targetStyle;
        }

        // DO deep merge on rich configurations.
        const sourceRich = sourceStyle.rich;
        const targetRich = targetStyle.rich || (sourceRich && {});  // Create a new one if source have rich but target don't

        extend(targetStyle, sourceStyle);

        if (sourceRich && targetRich) {
            // merge rich and assign rich again.
            this._mergeRich(targetRich, sourceRich);
            targetStyle.rich = targetRich;
        }
        else if (targetRich) {
            // If source rich not exists. DON'T override the target rich
            targetStyle.rich = targetRich;
        }

        return targetStyle;
    }

    private _mergeRich(targetRich: TextStyleProps['rich'], sourceRich: TextStyleProps['rich']) {
        const richNames = keys(sourceRich);
        // Merge by rich names.
        for (let i = 0; i < richNames.length; i++) {
            const richName = richNames[i];
            targetRich[richName] = targetRich[richName] || {};
            extend(targetRich[richName], sourceRich[richName]);
        }
    }

    getAnimationStyleProps() {
        return DEFAULT_TEXT_ANIMATION_PROPS;
    }


    private _getOrCreateChild(Ctor: {new(): TSpan}): TSpan
    private _getOrCreateChild(Ctor: {new(): ZRImage}): ZRImage
    private _getOrCreateChild(Ctor: {new(): Rect}): Rect
    private _getOrCreateChild(Ctor: {new(): TSpan | Rect | ZRImage}): TSpan | Rect | ZRImage {
        let child = this._children[this._childCursor];
        if (!child || !(child instanceof Ctor)) {
            child = new Ctor();
        }
        this._children[this._childCursor++] = child;
        child.__zr = this.__zr;
        // TODO to users parent can only be group.
        child.parent = this as any;
        return child;
    }

    private _updatePlainTexts() {
        const style = this.style;
        const textFont = style.font || DEFAULT_FONT;
        const textPadding = style.padding as number[];

        const text = getStyleText(style);
        const contentBlock = parsePlainText(text, style);
        const needDrawBg = needDrawBackground(style);
        const bgColorDrawn = !!(style.backgroundColor);

        const outerHeight = contentBlock.outerHeight;
        const outerWidth = contentBlock.outerWidth;
        const contentWidth = contentBlock.contentWidth;

        const textLines = contentBlock.lines;
        const lineHeight = contentBlock.lineHeight;

        const defaultStyle = this._defaultStyle;

        this.isTruncated = !!contentBlock.isTruncated;

        const baseX = style.x || 0;
        const baseY = style.y || 0;
        const textAlign = style.align || defaultStyle.align || 'left';
        const verticalAlign = style.verticalAlign || defaultStyle.verticalAlign || 'top';

        let textX = baseX;
        let textY = adjustTextY(baseY, contentBlock.contentHeight, verticalAlign);

        if (needDrawBg || textPadding) {
            // Consider performance, do not call getTextWidth util necessary.
            const boxX = adjustTextX(baseX, outerWidth, textAlign);
            const boxY = adjustTextY(baseY, outerHeight, verticalAlign);
            needDrawBg && this._renderBackground(style, style, boxX, boxY, outerWidth, outerHeight);
        }

        // `textBaseline` is set as 'middle'.
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

        let defaultLineWidth = 0;
        let useDefaultFill = false;
        const textFill = getFill(
            'fill' in style
                ? style.fill
                : (useDefaultFill = true, defaultStyle.fill)
        );
        const textStroke = getStroke(
            'stroke' in style
                ? style.stroke
                : (!bgColorDrawn
                    // If we use "auto lineWidth" widely, it probably bring about some bad case.
                    // So the current strategy is:
                    // If `style.fill` is specified (i.e., `useDefaultFill` is `false`)
                    // (A) And if `textConfig.insideStroke/outsideStroke` is not specified as a color
                    //   (i.e., `defaultStyle.autoStroke` is `true`), we do not actually display
                    //   the auto stroke because we can not make sure wether the stoke is approperiate to
                    //   the given `fill`.
                    // (B) But if `textConfig.insideStroke/outsideStroke` is specified as a color,
                    // we give the auto lineWidth to display the given stoke color.
                    && (!defaultStyle.autoStroke || useDefaultFill)
                )
                ? (defaultLineWidth = DEFAULT_STROKE_LINE_WIDTH, defaultStyle.stroke)
                : null
        );

        const hasShadow = style.textShadowBlur > 0;

        const fixedBoundingRect = style.width != null
            && (style.overflow === 'truncate' || style.overflow === 'break' || style.overflow === 'breakAll');
        const calculatedLineHeight = contentBlock.calculatedLineHeight;

        for (let i = 0; i < textLines.length; i++) {
            const el = this._getOrCreateChild(TSpan);
            // Always create new style.
            const subElStyle: TSpanStyleProps = el.createStyle();
            el.useStyle(subElStyle);
            subElStyle.text = textLines[i];
            subElStyle.x = textX;
            subElStyle.y = textY;
            // Always set textAlign and textBase line, because it is difficute to calculate
            // textAlign from prevEl, and we dont sure whether textAlign will be reset if
            // font set happened.
            if (textAlign) {
                subElStyle.textAlign = textAlign;
            }
            // Force baseline to be "middle". Otherwise, if using "top", the
            // text will offset downward a little bit in font "Microsoft YaHei".
            subElStyle.textBaseline = 'middle';
            subElStyle.opacity = style.opacity;
            // Fill after stroke so the outline will not cover the main part.
            subElStyle.strokeFirst = true;

            if (hasShadow) {
                subElStyle.shadowBlur = style.textShadowBlur || 0;
                subElStyle.shadowColor = style.textShadowColor || 'transparent';
                subElStyle.shadowOffsetX = style.textShadowOffsetX || 0;
                subElStyle.shadowOffsetY = style.textShadowOffsetY || 0;
            }

            // Always override default fill and stroke value.
            subElStyle.stroke = textStroke as string;
            subElStyle.fill = textFill as string;

            if (textStroke) {
                subElStyle.lineWidth = style.lineWidth || defaultLineWidth;
                subElStyle.lineDash = style.lineDash;
                subElStyle.lineDashOffset = style.lineDashOffset || 0;
            }

            subElStyle.font = textFont;
            setSeparateFont(subElStyle, style);

            textY += lineHeight;

            if (fixedBoundingRect) {
                el.setBoundingRect(new BoundingRect(
                    adjustTextX(subElStyle.x, contentWidth, subElStyle.textAlign as TextAlign),
                    adjustTextY(subElStyle.y, calculatedLineHeight, subElStyle.textBaseline as TextVerticalAlign),
                    /**
                     * Text boundary should be the real text width.
                     * Otherwise, there will be extra space in the
                     * bounding rect calculated.
                     */
                    contentWidth,
                    calculatedLineHeight
                ));
            }
        }
    }


    private _updateRichTexts() {
        const style = this.style;

        // TODO Only parse when text changed?
        const text = getStyleText(style);
        const contentBlock = parseRichText(text, style);

        const contentWidth = contentBlock.width;
        const outerWidth = contentBlock.outerWidth;
        const outerHeight = contentBlock.outerHeight;
        const textPadding = style.padding as number[];

        const baseX = style.x || 0;
        const baseY = style.y || 0;
        const defaultStyle = this._defaultStyle;
        const textAlign = style.align || defaultStyle.align;
        const verticalAlign = style.verticalAlign || defaultStyle.verticalAlign;

        this.isTruncated = !!contentBlock.isTruncated;

        const boxX = adjustTextX(baseX, outerWidth, textAlign);
        const boxY = adjustTextY(baseY, outerHeight, verticalAlign);
        let xLeft = boxX;
        let lineTop = boxY;

        if (textPadding) {
            xLeft += textPadding[3];
            lineTop += textPadding[0];
        }

        let xRight = xLeft + contentWidth;

        if (needDrawBackground(style)) {
            this._renderBackground(style, style, boxX, boxY, outerWidth, outerHeight);
        }
        const bgColorDrawn = !!(style.backgroundColor);

        for (let i = 0; i < contentBlock.lines.length; i++) {
            const line = contentBlock.lines[i];
            const tokens = line.tokens;
            const tokenCount = tokens.length;
            const lineHeight = line.lineHeight;

            let remainedWidth = line.width;
            let leftIndex = 0;
            let lineXLeft = xLeft;
            let lineXRight = xRight;
            let rightIndex = tokenCount - 1;
            let token;

            while (
                leftIndex < tokenCount
                && (token = tokens[leftIndex], !token.align || token.align === 'left')
            ) {
                this._placeToken(token, style, lineHeight, lineTop, lineXLeft, 'left', bgColorDrawn);
                remainedWidth -= token.width;
                lineXLeft += token.width;
                leftIndex++;
            }

            while (
                rightIndex >= 0
                && (token = tokens[rightIndex], token.align === 'right')
            ) {
                this._placeToken(token, style, lineHeight, lineTop, lineXRight, 'right', bgColorDrawn);
                remainedWidth -= token.width;
                lineXRight -= token.width;
                rightIndex--;
            }

            // The other tokens are placed as textAlign 'center' if there is enough space.
            lineXLeft += (contentWidth - (lineXLeft - xLeft) - (xRight - lineXRight) - remainedWidth) / 2;
            while (leftIndex <= rightIndex) {
                token = tokens[leftIndex];
                // Consider width specified by user, use 'center' rather than 'left'.
                this._placeToken(
                    token, style, lineHeight, lineTop,
                    lineXLeft + token.width / 2, 'center', bgColorDrawn
                );
                lineXLeft += token.width;
                leftIndex++;
            }

            lineTop += lineHeight;
        }
    }

    private _placeToken(
        token: TextToken,
        style: TextStyleProps,
        lineHeight: number,
        lineTop: number,
        x: number,
        textAlign: string,
        parentBgColorDrawn: boolean
    ) {
        const tokenStyle = style.rich[token.styleName] || {};
        tokenStyle.text = token.text;

        // 'ctx.textBaseline' is always set as 'middle', for sake of
        // the bias of "Microsoft YaHei".
        const verticalAlign = token.verticalAlign;
        let y = lineTop + lineHeight / 2;
        if (verticalAlign === 'top') {
            y = lineTop + token.height / 2;
        }
        else if (verticalAlign === 'bottom') {
            y = lineTop + lineHeight - token.height / 2;
        }

        const needDrawBg = !token.isLineHolder && needDrawBackground(tokenStyle);
        needDrawBg && this._renderBackground(
            tokenStyle,
            style,
            textAlign === 'right'
                ? x - token.width
                : textAlign === 'center'
                ? x - token.width / 2
                : x,
            y - token.height / 2,
            token.width,
            token.height
        );
        const bgColorDrawn = !!tokenStyle.backgroundColor;

        const textPadding = token.textPadding;
        if (textPadding) {
            x = getTextXForPadding(x, textAlign, textPadding);
            y -= token.height / 2 - textPadding[0] - token.innerHeight / 2;
        }

        const el = this._getOrCreateChild(TSpan);
        const subElStyle: TSpanStyleProps = el.createStyle();
        // Always create new style.
        el.useStyle(subElStyle);

        const defaultStyle = this._defaultStyle;
        let useDefaultFill = false;
        let defaultLineWidth = 0;
        const textFill = getFill(
            'fill' in tokenStyle ? tokenStyle.fill
                : 'fill' in style ? style.fill
                : (useDefaultFill = true, defaultStyle.fill)
        );
        const textStroke = getStroke(
            'stroke' in tokenStyle ? tokenStyle.stroke
                : 'stroke' in style ? style.stroke
                : (
                    !bgColorDrawn
                    && !parentBgColorDrawn
                    // See the strategy explained above.
                    && (!defaultStyle.autoStroke || useDefaultFill)
                ) ? (defaultLineWidth = DEFAULT_STROKE_LINE_WIDTH, defaultStyle.stroke)
                : null
        );

        const hasShadow = tokenStyle.textShadowBlur > 0
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

        subElStyle.textAlign = textAlign as CanvasTextAlign;
        // Force baseline to be "middle". Otherwise, if using "top", the
        // text will offset downward a little bit in font "Microsoft YaHei".
        subElStyle.textBaseline = 'middle';
        subElStyle.font = token.font || DEFAULT_FONT;
        subElStyle.opacity = retrieve3(tokenStyle.opacity, style.opacity, 1);


        // TODO inherit each item from top style in token style?
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

        const textWidth = token.contentWidth;
        const textHeight = token.contentHeight;
        // NOTE: Should not call dirtyStyle after setBoundingRect. Or it will be cleared.
        el.setBoundingRect(new BoundingRect(
            adjustTextX(subElStyle.x, textWidth, subElStyle.textAlign as TextAlign),
            adjustTextY(subElStyle.y, textHeight, subElStyle.textBaseline as TextVerticalAlign),
            textWidth,
            textHeight
        ));
    }

    private _renderBackground(
        style: TextStylePropsPart,
        topStyle: TextStylePropsPart,
        x: number,
        y: number,
        width: number,
        height: number
    ) {
        const textBackgroundColor = style.backgroundColor;
        const textBorderWidth = style.borderWidth;
        const textBorderColor = style.borderColor;
        const isImageBg = textBackgroundColor && (textBackgroundColor as {image: ImageLike}).image;
        const isPlainOrGradientBg = textBackgroundColor && !isImageBg;
        const textBorderRadius = style.borderRadius;
        const self = this;

        let rectEl: Rect;
        let imgEl: ZRImage;
        if (isPlainOrGradientBg || style.lineHeight || (textBorderWidth && textBorderColor)) {
            // Background is color
            rectEl = this._getOrCreateChild(Rect);
            rectEl.useStyle(rectEl.createStyle());    // Create an empty style.
            rectEl.style.fill = null;
            const rectShape = rectEl.shape;
            rectShape.x = x;
            rectShape.y = y;
            rectShape.width = width;
            rectShape.height = height;
            rectShape.r = textBorderRadius;
            rectEl.dirtyShape();
        }

        if (isPlainOrGradientBg) {
            const rectStyle = rectEl.style;
            rectStyle.fill = textBackgroundColor as string || null;
            rectStyle.fillOpacity = retrieve2(style.fillOpacity, 1);
        }
        else if (isImageBg) {
            imgEl = this._getOrCreateChild(ZRImage);
            imgEl.onload = function () {
                // Refresh and relayout after image loaded.
                self.dirtyStyle();
            };
            const imgStyle = imgEl.style;
            imgStyle.image = (textBackgroundColor as {image: ImageLike}).image;
            imgStyle.x = x;
            imgStyle.y = y;
            imgStyle.width = width;
            imgStyle.height = height;
        }

        if (textBorderWidth && textBorderColor) {
            const rectStyle = rectEl.style;
            rectStyle.lineWidth = textBorderWidth;
            rectStyle.stroke = textBorderColor;
            rectStyle.strokeOpacity = retrieve2(style.strokeOpacity, 1);
            rectStyle.lineDash = style.borderDash;
            rectStyle.lineDashOffset = style.borderDashOffset || 0;
            rectEl.strokeContainThreshold = 0;

            // Making shadow looks better.
            if (rectEl.hasFill() && rectEl.hasStroke()) {
                rectStyle.strokeFirst = true;
                rectStyle.lineWidth *= 2;
            }
        }

        const commonStyle = (rectEl || imgEl).style;
        commonStyle.shadowBlur = style.shadowBlur || 0;
        commonStyle.shadowColor = style.shadowColor || 'transparent';
        commonStyle.shadowOffsetX = style.shadowOffsetX || 0;
        commonStyle.shadowOffsetY = style.shadowOffsetY || 0;
        commonStyle.opacity = retrieve3(style.opacity, topStyle.opacity, 1);
    }

    static makeFont(style: TextStylePropsPart): string {
        // FIXME in node-canvas fontWeight is before fontStyle
        // Use `fontSize` `fontFamily` to check whether font properties are defined.
        let font = '';
        if (hasSeparateFont(style)) {
            font = [
                style.fontStyle,
                style.fontWeight,
                parseFontSize(style.fontSize),
                // If font properties are defined, `fontFamily` should not be ignored.
                style.fontFamily || 'sans-serif'
            ].join(' ');
        }
        return font && trim(font) || style.textFont || style.font;
    }
}


const VALID_TEXT_ALIGN = {left: true, right: 1, center: 1};
const VALID_TEXT_VERTICAL_ALIGN = {top: 1, bottom: 1, middle: 1};

const FONT_PARTS = ['fontStyle', 'fontWeight', 'fontSize', 'fontFamily'] as const;

export function parseFontSize(fontSize: number | string) {
    if (
        typeof fontSize === 'string'
        && (
            fontSize.indexOf('px') !== -1
            || fontSize.indexOf('rem') !== -1
            || fontSize.indexOf('em') !== -1
        )
    ) {
        return fontSize;
    }
    else if (!isNaN(+fontSize)) {
        return fontSize + 'px';
    }
    else {
        return DEFAULT_FONT_SIZE + 'px';
    }
}

function setSeparateFont(
    targetStyle: TSpanStyleProps,
    sourceStyle: TextStylePropsPart
) {
    for (let i = 0; i < FONT_PARTS.length; i++) {
        const fontProp = FONT_PARTS[i];
        const val = sourceStyle[fontProp];
        if (val != null) {
            (targetStyle as any)[fontProp] = val;
        }
    }
}

export function hasSeparateFont(style: Pick<TextStylePropsPart, 'fontSize' | 'fontFamily' | 'fontWeight'>) {
    return style.fontSize != null || style.fontFamily || style.fontWeight;
}

export function normalizeTextStyle(style: TextStyleProps): TextStyleProps {
    normalizeStyle(style);
    // TODO inherit each item from top style in token style?
    each(style.rich, normalizeStyle);
    return style;
}

function normalizeStyle(style: TextStylePropsPart) {
    if (style) {
        style.font = ZRText.makeFont(style);
        let textAlign = style.align;
        // 'middle' is invalid, convert it to 'center'
        (textAlign as string) === 'middle' && (textAlign = 'center');
        style.align = (
            textAlign == null || VALID_TEXT_ALIGN[textAlign]
        ) ? textAlign : 'left';

        // Compatible with textBaseline.
        let verticalAlign = style.verticalAlign;
        (verticalAlign as string) === 'center' && (verticalAlign = 'middle');
        style.verticalAlign = (
            verticalAlign == null || VALID_TEXT_VERTICAL_ALIGN[verticalAlign]
        ) ? verticalAlign : 'top';

        // TODO Should not change the orignal value.
        const textPadding = style.padding;
        if (textPadding) {
            style.padding = normalizeCssArray(style.padding);
        }
    }
}

/**
 * @param stroke If specified, do not check style.textStroke.
 * @param lineWidth If specified, do not check style.textStroke.
 */
function getStroke(
    stroke?: TextStylePropsPart['stroke'],
    lineWidth?: number
) {
    return (stroke == null || lineWidth <= 0 || stroke === 'transparent' || stroke === 'none')
        ? null
        : ((stroke as any).image || (stroke as any).colorStops)
        ? '#000'
        : stroke;
}

function getFill(
    fill?: TextStylePropsPart['fill']
) {
    return (fill == null || fill === 'none')
        ? null
        // TODO pattern and gradient?
        : ((fill as any).image || (fill as any).colorStops)
        ? '#000'
        : fill;
}

function getTextXForPadding(x: number, textAlign: string, textPadding: number[]): number {
    return textAlign === 'right'
        ? (x - textPadding[1])
        : textAlign === 'center'
        ? (x + textPadding[3] / 2 - textPadding[1] / 2)
        : (x + textPadding[3]);
}

function getStyleText(style: TextStylePropsPart): string {
    // Compat: set number to text is supported.
    // set null/undefined to text is supported.
    let text = style.text;
    text != null && (text += '');
    return text;
}

/**
 * If needs draw background
 * @param style Style of element
 */
function needDrawBackground(style: TextStylePropsPart): boolean {
    return !!(
        style.backgroundColor
        || style.lineHeight
        || (style.borderWidth && style.borderColor)
    );
}

export default ZRText;