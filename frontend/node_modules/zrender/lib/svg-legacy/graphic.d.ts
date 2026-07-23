import Path from '../graphic/Path';
import ZRImage from '../graphic/Image';
import TSpan from '../graphic/TSpan';
export interface SVGProxy<T> {
    brush(el: T): void;
}
declare const svgPath: SVGProxy<Path>;
export { svgPath as path };
declare const svgImage: SVGProxy<ZRImage>;
export { svgImage as image };
declare const svgText: SVGProxy<TSpan>;
export { svgText as text };
