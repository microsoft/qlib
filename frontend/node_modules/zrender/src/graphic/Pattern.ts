import { ImageLike } from '../core/types';
import { SVGVNode } from '../svg/core';

type ImagePatternRepeat = 'repeat' | 'repeat-x' | 'repeat-y' | 'no-repeat'

export interface PatternObjectBase {
    id?: number
    // type is now unused, so make it optional
    type?: 'pattern'

    x?: number
    y?: number
    rotation?: number
    scaleX?: number
    scaleY?: number
}

export interface ImagePatternObject extends PatternObjectBase {
    image: ImageLike | string
    repeat?: ImagePatternRepeat

    /**
     * Width and height of image.
     * `imageWidth` and `imageHeight` are only used in svg-ssr renderer.
     * Because we can't get the size of image in svg-ssr renderer.
     * They need to be give explictly.
     */
    imageWidth?: number
    imageHeight?: number
}

export interface InnerImagePatternObject extends ImagePatternObject {
    // Cached image. Which is created in the canvas painter.
    __image?: ImageLike
}

export interface SVGPatternObject extends PatternObjectBase {
    /**
     * svg vnode can only be used in svg renderer currently.
     * svgWidth, svgHeight defines width and height used for pattern.
     */
    svgElement?: SVGVNode
    svgWidth?: number
    svgHeight?: number
}

export type PatternObject = ImagePatternObject | SVGPatternObject

class Pattern {

    type: 'pattern'

    image: ImageLike | string
    /**
     * svg element can only be used in svg renderer currently.
     *
     * Will be string if using SSR rendering.
     */
    svgElement: SVGElement | string

    repeat: ImagePatternRepeat

    x: number
    y: number
    rotation: number
    scaleX: number
    scaleY: number

    constructor(image: ImageLike | string, repeat: ImagePatternRepeat) {
        // Should do nothing more in this constructor. Because gradient can be
        // declard by `color: {image: ...}`, where this constructor will not be called.
        this.image = image;
        this.repeat = repeat;

        this.x = 0;
        this.y = 0;
        this.rotation = 0;
        this.scaleX = 1;
        this.scaleY = 1;
    }
}

export default Pattern;