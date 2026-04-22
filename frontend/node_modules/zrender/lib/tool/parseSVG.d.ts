import Group from '../graphic/Group';
import Element from '../Element';
import { RectLike } from '../core/BoundingRect';
import { parseXML } from './parseXML';
interface SVGParserOption {
    width?: number;
    height?: number;
    ignoreViewBox?: boolean;
    ignoreRootClip?: boolean;
}
export interface SVGParserResult {
    root: Group;
    width: number;
    height: number;
    viewBoxRect: RectLike;
    viewBoxTransform: {
        x: number;
        y: number;
        scale: number;
    };
    named: SVGParserResultNamedItem[];
}
export interface SVGParserResultNamedItem {
    name: string;
    namedFrom: SVGParserResultNamedItem;
    svgNodeTagLower: SVGNodeTagLower;
    el: Element;
}
export declare type SVGNodeTagLower = 'g' | 'rect' | 'circle' | 'line' | 'ellipse' | 'polygon' | 'polyline' | 'image' | 'text' | 'tspan' | 'path' | 'defs' | 'switch';
export declare function makeViewBoxTransform(viewBoxRect: RectLike, boundingRect: RectLike): {
    scale: number;
    x: number;
    y: number;
};
export declare function parseSVG(xml: string | Document | SVGElement, opt: SVGParserOption): SVGParserResult;
export { parseXML };
