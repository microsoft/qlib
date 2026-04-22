import Displayable, { DisplayableProps, CommonStyleProps, DisplayableStatePropNames } from './Displayable';
import BoundingRect from '../core/BoundingRect';
import { ImageLike, MapToType } from '../core/types';
import { ElementCommonState } from '../Element';
export interface ImageStyleProps extends CommonStyleProps {
    image?: string | ImageLike;
    x?: number;
    y?: number;
    width?: number;
    height?: number;
    sx?: number;
    sy?: number;
    sWidth?: number;
    sHeight?: number;
}
export declare const DEFAULT_IMAGE_STYLE: CommonStyleProps;
export declare const DEFAULT_IMAGE_ANIMATION_PROPS: MapToType<ImageProps, boolean>;
export interface ImageProps extends DisplayableProps {
    style?: ImageStyleProps;
    onload?: (image: ImageLike) => void;
}
export declare type ImageState = Pick<ImageProps, DisplayableStatePropNames> & ElementCommonState;
declare class ZRImage extends Displayable<ImageProps> {
    style: ImageStyleProps;
    __image: ImageLike;
    __imageSrc: string;
    onload: (image: ImageLike) => void;
    createStyle(obj?: ImageStyleProps): ImageStyleProps;
    private _getSize;
    getWidth(): number;
    getHeight(): number;
    getAnimationStyleProps(): MapToType<ImageProps, boolean>;
    getBoundingRect(): BoundingRect;
}
export default ZRImage;
