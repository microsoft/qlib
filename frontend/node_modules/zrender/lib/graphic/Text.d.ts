import { TextAlign, TextVerticalAlign, ImageLike, Dictionary, MapToType, FontWeight, FontStyle } from '../core/types';
import TSpan from './TSpan';
import ZRImage from './Image';
import Rect from './shape/Rect';
import BoundingRect from '../core/BoundingRect';
import { MatrixArray } from '../core/matrix';
import Displayable, { DisplayableStatePropNames, DisplayableProps } from './Displayable';
import { ZRenderType } from '../zrender';
import Animator from '../animation/Animator';
import Transformable from '../core/Transformable';
import { ElementCommonState } from '../Element';
import { GroupLike } from './Group';
export interface TextStylePropsPart {
    text?: string;
    fill?: string;
    stroke?: string;
    strokeNoScale?: boolean;
    opacity?: number;
    fillOpacity?: number;
    strokeOpacity?: number;
    lineWidth?: number;
    lineDash?: false | number[];
    lineDashOffset?: number;
    borderDash?: false | number[];
    borderDashOffset?: number;
    font?: string;
    textFont?: string;
    fontStyle?: FontStyle;
    fontWeight?: FontWeight;
    fontFamily?: string;
    fontSize?: number | string;
    align?: TextAlign;
    verticalAlign?: TextVerticalAlign;
    lineHeight?: number;
    width?: number | string;
    height?: number;
    tag?: string;
    textShadowColor?: string;
    textShadowBlur?: number;
    textShadowOffsetX?: number;
    textShadowOffsetY?: number;
    backgroundColor?: string | {
        image: ImageLike | string;
    };
    padding?: number | number[];
    margin?: number;
    borderColor?: string;
    borderWidth?: number;
    borderRadius?: number | number[];
    shadowColor?: string;
    shadowBlur?: number;
    shadowOffsetX?: number;
    shadowOffsetY?: number;
}
export interface TextStyleProps extends TextStylePropsPart {
    text?: string;
    x?: number;
    y?: number;
    width?: number;
    rich?: Dictionary<TextStylePropsPart>;
    overflow?: 'break' | 'breakAll' | 'truncate' | 'none';
    lineOverflow?: 'truncate';
    ellipsis?: string;
    placeholder?: string;
    truncateMinChar?: number;
}
export interface TextProps extends DisplayableProps {
    style?: TextStyleProps;
    zlevel?: number;
    z?: number;
    z2?: number;
    culling?: boolean;
    cursor?: string;
}
export declare type TextState = Pick<TextProps, DisplayableStatePropNames> & ElementCommonState;
export declare type DefaultTextStyle = Pick<TextStyleProps, 'fill' | 'stroke' | 'align' | 'verticalAlign'> & {
    autoStroke?: boolean;
};
export declare const DEFAULT_TEXT_ANIMATION_PROPS: MapToType<TextProps, boolean>;
interface ZRText {
    animate(key?: '', loop?: boolean): Animator<this>;
    animate(key: 'style', loop?: boolean): Animator<this['style']>;
    getState(stateName: string): TextState;
    ensureState(stateName: string): TextState;
    states: Dictionary<TextState>;
    stateProxy: (stateName: string) => TextState;
}
declare class ZRText extends Displayable<TextProps> implements GroupLike {
    type: string;
    style: TextStyleProps;
    overlap: 'hidden' | 'show' | 'blur';
    innerTransformable: Transformable;
    isTruncated: boolean;
    private _children;
    private _childCursor;
    private _defaultStyle;
    constructor(opts?: TextProps);
    childrenRef(): (ZRImage | Rect | TSpan)[];
    update(): void;
    updateTransform(): void;
    getLocalTransform(m?: MatrixArray): MatrixArray;
    getComputedTransform(): MatrixArray;
    private _updateSubTexts;
    addSelfToZr(zr: ZRenderType): void;
    removeSelfFromZr(zr: ZRenderType): void;
    getBoundingRect(): BoundingRect;
    setDefaultTextStyle(defaultTextStyle: DefaultTextStyle): void;
    setTextContent(textContent: never): void;
    protected _mergeStyle(targetStyle: TextStyleProps, sourceStyle: TextStyleProps): TextStyleProps;
    private _mergeRich;
    getAnimationStyleProps(): MapToType<TextProps, boolean>;
    private _getOrCreateChild;
    private _updatePlainTexts;
    private _updateRichTexts;
    private _placeToken;
    private _renderBackground;
    static makeFont(style: TextStylePropsPart): string;
}
export declare function parseFontSize(fontSize: number | string): string;
export declare function hasSeparateFont(style: Pick<TextStylePropsPart, 'fontSize' | 'fontFamily' | 'fontWeight'>): string | number | true;
export declare function normalizeTextStyle(style: TextStyleProps): TextStyleProps;
export default ZRText;
