import Displayable, { DisplayableProps,
    CommonStyleProps,
    DEFAULT_COMMON_STYLE,
    DisplayableStatePropNames,
    DEFAULT_COMMON_ANIMATION_PROPS
} from './Displayable';
import BoundingRect from '../core/BoundingRect';
import { ImageLike, MapToType } from '../core/types';
import { defaults, createObject } from '../core/util';
import { ElementCommonState } from '../Element';

export interface ImageStyleProps extends CommonStyleProps {
    image?: string | ImageLike
    x?: number
    y?: number
    width?: number
    height?: number
    sx?: number
    sy?: number
    sWidth?: number
    sHeight?: number
}

export const DEFAULT_IMAGE_STYLE: CommonStyleProps = defaults({
    x: 0,
    y: 0
}, DEFAULT_COMMON_STYLE);

export const DEFAULT_IMAGE_ANIMATION_PROPS: MapToType<ImageProps, boolean> = {
    style: defaults<MapToType<ImageStyleProps, boolean>, MapToType<ImageStyleProps, boolean>>({
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

export interface ImageProps extends DisplayableProps {
    style?: ImageStyleProps

    onload?: (image: ImageLike) => void
}

export type ImageState = Pick<ImageProps, DisplayableStatePropNames> & ElementCommonState

function isImageLike(source: unknown): source is HTMLImageElement {
    return !!(source
        && typeof source !== 'string'
        // Image source is an image, canvas, video.
        && (source as HTMLImageElement).width && (source as HTMLImageElement).height);
}

class ZRImage extends Displayable<ImageProps> {

    style: ImageStyleProps

    // FOR CANVAS RENDERER
    __image: ImageLike
    // FOR SVG RENDERER
    __imageSrc: string

    onload: (image: ImageLike) => void

    /**
     * Create an image style object with default values in it's prototype.
     * @override
     */
    createStyle(obj?: ImageStyleProps) {
        return createObject(DEFAULT_IMAGE_STYLE, obj);
    }

    private _getSize(dim: 'width' | 'height') {
        const style = this.style;

        let size = style[dim];
        if (size != null) {
            return size;
        }

        const imageSource = isImageLike(style.image)
            ? style.image : this.__image;

        if (!imageSource) {
            return 0;
        }

        const otherDim = dim === 'width' ? 'height' : 'width';
        let otherDimSize = style[otherDim];
        if (otherDimSize == null) {
            return imageSource[dim];
        }
        else {
            return imageSource[dim] / imageSource[otherDim] * otherDimSize;
        }
    }

    getWidth(): number {
        return this._getSize('width');
    }

    getHeight(): number {
        return this._getSize('height');
    }

    getAnimationStyleProps() {
        return DEFAULT_IMAGE_ANIMATION_PROPS;
    }

    getBoundingRect(): BoundingRect {
        const style = this.style;
        if (!this._rect) {
            this._rect = new BoundingRect(
                style.x || 0, style.y || 0, this.getWidth(), this.getHeight()
            );
        }
        return this._rect;
    }
}

ZRImage.prototype.type = 'image';

export default ZRImage;