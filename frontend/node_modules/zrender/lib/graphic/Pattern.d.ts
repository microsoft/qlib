import { ImageLike } from '../core/types';
import { SVGVNode } from '../svg/core';
declare type ImagePatternRepeat = 'repeat' | 'repeat-x' | 'repeat-y' | 'no-repeat';
export interface PatternObjectBase {
    id?: number;
    type?: 'pattern';
    x?: number;
    y?: number;
    rotation?: number;
    scaleX?: number;
    scaleY?: number;
}
export interface ImagePatternObject extends PatternObjectBase {
    image: ImageLike | string;
    repeat?: ImagePatternRepeat;
    imageWidth?: number;
    imageHeight?: number;
}
export interface InnerImagePatternObject extends ImagePatternObject {
    __image?: ImageLike;
}
export interface SVGPatternObject extends PatternObjectBase {
    svgElement?: SVGVNode;
    svgWidth?: number;
    svgHeight?: number;
}
export declare type PatternObject = ImagePatternObject | SVGPatternObject;
declare class Pattern {
    type: 'pattern';
    image: ImageLike | string;
    svgElement: SVGElement | string;
    repeat: ImagePatternRepeat;
    x: number;
    y: number;
    rotation: number;
    scaleX: number;
    scaleY: number;
    constructor(image: ImageLike | string, repeat: ImagePatternRepeat);
}
export default Pattern;
